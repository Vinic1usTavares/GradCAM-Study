"""Utilidades de visualizacao: upsampling, normalizacao, colormap e overlay."""

from __future__ import annotations

import matplotlib
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


def normalize(heatmap: torch.Tensor) -> torch.Tensor:
    """Escala para [0, 1] pelo maximo. Retorna zeros se o mapa for vazio.

    Nao usar o minimo: o Grad-CAM e' nao-negativo apos o ReLU, logo o minimo
    seria sempre zero e a escala ficaria correta mesmo assim, mas mapas com
    apenas um pixel ativo ficariam apagados.
    """
    peak = heatmap.max()
    if not torch.isfinite(peak) or peak <= 0:
        return torch.zeros_like(heatmap)
    return heatmap / peak


def to_rgb(heatmap: torch.Tensor) -> np.ndarray:
    """Mapa 2D em [0, 1] -> array RGB uint8 via colormap jet."""
    array = normalize(heatmap).detach().cpu().numpy()
    return (CMAP(array)[..., :3] * 255).astype(np.uint8)


def overlay(
    image: np.ndarray, heatmap: torch.Tensor, alpha: float = 0.5
) -> np.ndarray:
    """Sobrepoe o mapa colorido a imagem RGB uint8 `(H, W, 3)`."""
    height, width = image.shape[:2]
    resized = upsample(heatmap, (height, width))
    color = to_rgb(resized).astype(np.float32)
    blended = (1 - alpha) * image.astype(np.float32) + alpha * color
    return blended.clip(0, 255).astype(np.uint8)


def tensor_to_numpy(x: torch.Tensor) -> np.ndarray:
    """(1, 3, H, W) em [0, 1] -> (H, W, 3) uint8."""
    array = x[0].detach().cpu().permute(1, 2, 0).numpy()
    return (array.clip(0, 1) * 255).astype(np.uint8)
