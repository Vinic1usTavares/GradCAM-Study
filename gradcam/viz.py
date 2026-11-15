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
    label_width: float = 1.0,
):
    """Grade de imagens RGB uint8, uma linha por caso, pronta para salvar.

    `rows` e' uma sequencia de linhas; cada linha e' uma sequencia de arrays
    (H, W, 3). Linhas de tamanhos diferentes sao permitidas: as celulas que
    faltam ficam em branco.

    `row_titles` viram uma coluna de rotulos a esquerda (largura relativa
    `label_width`), com `wrap` na largura da celula. Nao usamos texto
    rotacionado: o `tight_layout` mede a caixa do texto antes da rotacao e,
    com rotulo longo, encolhia os eixos para ~3% da largura.
    """
    if not rows:
        raise ValueError("figure_grid precisa de pelo menos uma linha")

    n_rows = len(rows)
    n_cols = max(len(row) for row in rows)
    has_labels = bool(row_titles)
    if row_titles and len(row_titles) != n_rows:
        raise ValueError(
            f"row_titles tem {len(row_titles)} itens para {n_rows} linhas"
        )

    height, width = rows[0][0].shape[:2]
    if figsize is None:
        cols = n_cols + (label_width if has_labels else 0.0)
        figsize = (2.1 * cols, 2.1 * (height / width) * n_rows)

    fig = plt.figure(figsize=figsize, dpi=dpi, layout="constrained")
    widths = ([label_width] if has_labels else []) + [1.0] * n_cols
    grid = fig.add_gridspec(n_rows, len(widths), width_ratios=widths)

    for i, row in enumerate(rows):
        offset = 1 if has_labels else 0
        if has_labels:
            label_ax = fig.add_subplot(grid[i, 0])
            label_ax.set_axis_off()
            label_ax.text(
                0.0, 0.5, row_titles[i],
                ha="left", va="center", fontsize=8, wrap=True,
                transform=label_ax.transAxes,
            )
        for j, cell in enumerate(row):
            ax = fig.add_subplot(grid[i, j + offset])
            ax.set_axis_off()
            ax.imshow(cell, interpolation="nearest")
            if i == 0 and col_titles and j < len(col_titles):
                ax.set_title(col_titles[j], fontsize=10)

    if suptitle:
        fig.suptitle(suptitle, fontsize=11)
    return fig
