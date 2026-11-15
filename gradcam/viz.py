"""Utilidades de visualizacao: upsampling, normalizacao, colormap e overlay."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

CMAP = matplotlib.colormaps["jet"]


def upsample(heatmap: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Redimensiona (H, W) ou (1, H, W) para `size` via interpolacao bilinear."""
    squeeze = heatmap.ndim == 2
    x = heatmap.float()
    if squeeze:
        x = x.unsqueeze(0)
    x = x.unsqueeze(0) if x.ndim == 2 else x
    out = F.interpolate(
        x[None], size=size, mode="bilinear", align_corners=False
    )[0]
    return out[0] if squeeze else out


def scale_for_display(
    heatmap: torch.Tensor,
    high: float | None = None,
    gamma: float = 1.0,
) -> torch.Tensor:
    """Escala um mapa para [0, 1] **para exibicao**, com corte e gamma.

    `high` e' o quantil usado como branco (ex.: 0.99). Sem ele, usa o maximo,
    que e' o comportamento de `normalize`. Motivacao medida no VGG-16: o pico
    do gradiente vanilla e' ~5e5, enquanto a mediana e' 5% disso; escalar pelo
    maximo deixa o mapa quase preto e a estrutura some. O quantil corta a cauda
    de outliers e o gamma < 1 levanta as regioes fracas.

    Nao serve para metrica: para isso, use o mapa cru (a escala nao preserva
    ordem nem distancia).
    """
    values = heatmap.float()
    peak = values.max() if high is None else torch.quantile(values.flatten(), high)
    if not torch.isfinite(peak) or peak <= 0:
        return torch.zeros_like(values)
    scaled = (values / peak).clamp(0, 1)
    return scaled if gamma == 1.0 else scaled.pow(gamma)


def normalize(heatmap: torch.Tensor) -> torch.Tensor:
    """Escala para [0, 1] pelo maximo. Retorna zeros se o mapa for vazio.

    Nao usar o minimo: o Grad-CAM e' nao-negativo apos o ReLU, logo o minimo
    seria sempre zero e a escala ficaria correta mesmo assim, mas mapas com
    apenas um pixel ativo ficariam apagados.

    Para figura, prefira `scale_for_display` (corte por quantil + gamma).
    """
    return scale_for_display(heatmap, high=None, gamma=1.0)


def to_rgb(
    heatmap: torch.Tensor,
    high: float | None = None,
    gamma: float = 1.0,
) -> np.ndarray:
    """Mapa 2D -> array RGB uint8 via colormap jet."""
    array = scale_for_display(heatmap, high, gamma).detach().cpu().numpy()
    return (CMAP(array)[..., :3] * 255).astype(np.uint8)


def overlay(
    image: np.ndarray,
    heatmap: torch.Tensor,
    alpha: float = 0.5,
    high: float | None = None,
    gamma: float = 1.0,
) -> np.ndarray:
    """Sobrepoe o mapa colorido a imagem RGB uint8 `(H, W, 3)`."""
    height, width = image.shape[:2]
    resized = upsample(heatmap, (height, width))
    color = to_rgb(resized, high, gamma).astype(np.float32)
    blended = (1 - alpha) * image.astype(np.float32) + alpha * color
    return blended.clip(0, 255).astype(np.uint8)


def tensor_to_numpy(x: torch.Tensor) -> np.ndarray:
    """(1, 3, H, W) em [0, 1] -> (H, W, 3) uint8."""
    array = x[0].detach().cpu().permute(1, 2, 0).numpy()
    return (array.clip(0, 1) * 255).astype(np.uint8)


def gray_rgb(
    heatmap: torch.Tensor,
    high: float | None = None,
    gamma: float = 1.0,
) -> np.ndarray:
    """Mapa 2D nao-negativo -> RGB uint8 em escala de cinza.

    Devolve RGB em vez de (H, W) de proposito: assim `figure_grid` nao precisa
    saber se cada celula e' mapa ou foto, e nao ha risco de `imshow` aplicar a
    colormap padrao (viridis) por engano.

    Cinza, e nao `jet`, porque o objetivo das Fases 1-2 e' comparar a *textura*
    do gradiente entre as tres regras de ReLU; a cor atrapalharia.
    """
    array = scale_for_display(heatmap, high, gamma).detach().cpu().numpy()
    return np.repeat((array * 255).astype(np.uint8)[..., None], 3, axis=2)


def figure_grid(
    rows: Sequence[Sequence[np.ndarray]],
    col_titles: Sequence[str] | None = None,
    row_titles: Sequence[str] | None = None,
    suptitle: str | None = None,
    figsize: tuple[float, float] | None = None,
    dpi: int = 130,
):
    """Grade de imagens RGB uint8, uma linha por caso, pronta para salvar.

    `rows` e' uma sequencia de linhas; cada linha e' uma sequencia de arrays
    (H, W, 3). Linhas de tamanhos diferentes sao permitidas: as celulas que
    faltam ficam em branco.
    """
    if not rows:
        raise ValueError("figure_grid precisa de pelo menos uma linha")

    n_rows = len(rows)
    n_cols = max(len(row) for row in rows)
    if figsize is None:
        height, width = rows[0][0].shape[:2]
        aspect = height / width
        figsize = (2.2 * n_cols, 2.2 * aspect * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, dpi=dpi,
                             squeeze=False)

    for i, row in enumerate(rows):
        for j in range(n_cols):
            ax = axes[i][j]
            ax.set_axis_off()
            if j < len(row):
                ax.imshow(row[j], interpolation="nearest")
                if i == 0 and col_titles and j < len(col_titles):
                    ax.set_title(col_titles[j], fontsize=10)
        if row_titles and i < len(row_titles):
            # `ax.text` continua sendo desenhado com o eixo desligado.
            axes[i][0].text(
                -0.04, 0.5, row_titles[i],
                transform=axes[i][0].transAxes,
                rotation=90, va="center", ha="right", fontsize=8,
            )

    if suptitle:
        fig.suptitle(suptitle, fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
    else:
        fig.tight_layout()
    return fig
