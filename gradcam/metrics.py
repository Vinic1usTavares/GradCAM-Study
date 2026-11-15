"""Metricas de avaliacao dos mapas de atribuicao.

Toda avaliacao consome mapas pela mesma assinatura plugavel (plans/general_plan.md,
secao 2.4):

    MapFn = (modelo, entrada, classe) -> mapa HxW

Assim `03_class_discrimination.py`, o pointing game e a correlacao com oclusao
rodam a mesma bateria de metodos sem duplicar codigo por metodo.

Nota sobre escala: `spearman` e' invariante a transformacao monotonica, entao a
escala de exibicao (`viz.scale_for_display`, quantil + gamma) **nao** contamina
a metrica. Ainda assim, as metricas recebem o mapa cru: nao ha motivo para
normalizar antes de ranquear.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
from scipy import stats

from .relu_variants import MODES
from .saliency import saliency

MapFn = Callable[[nn.Module, torch.Tensor, int], torch.Tensor]


def gradcam_map_fn(layer: nn.Module | str | None = None) -> MapFn:
    """Adapta `core.gradcam` a assinatura `MapFn`.

    `layer` fixa a camada (nome ou modulo); `None` deixa o `resolve_layer`
    escolher a ultima conv. O adaptador existe para a bateria da Fase 8 rodar
    Grad-CAM e as saliencias pela mesma interface.
    """
    from .core import gradcam

    def _map_fn(model: nn.Module, x: torch.Tensor, class_idx: int) -> torch.Tensor:
        heatmap, _ = gradcam(model, x, class_idx, layer=layer)
        return heatmap

    return _map_fn


def saliency_map_fn(mode: str) -> MapFn:
    """Adapta `saliency(mode=...)` a assinatura `MapFn`.

    `saliency` devolve `(mapa, class_idx)`; a avaliacao so' quer o mapa. Fazer
    o adaptador aqui (em vez de mudar a assinatura de `saliency`) mantem a
    funcao de biblioteca util para uso interativo, onde a classe inferida
    importa.
    """
    if mode not in MODES:
        raise ValueError(f"modo invalido: {mode!r}; use um de {MODES}")

    def _map_fn(model: nn.Module, x: torch.Tensor, class_idx: int) -> torch.Tensor:
        heatmap, _ = saliency(model, x, class_idx, mode=mode)
        return heatmap

    return _map_fn


def saliency_map_fns() -> dict[str, MapFn]:
    """Os tres metodos de gradiente, na ordem de MODES."""
    return {mode: saliency_map_fn(mode) for mode in MODES}


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    """Correlacao de Spearman entre dois mapas, achatados.

    Empates recebem rank medio (comportamento do `scipy.stats.spearmanr`).
    Devolve `nan` quando a correlacao nao esta definida — mapa constante, por
    exemplo um Grad-CAM zerado pelo ReLU. Quem agrega deve descartar o `nan`
    e reportar quantos valores validos entraram.
    """
    x = np.asarray(a.detach().flatten().cpu(), dtype=np.float64)
    y = np.asarray(b.detach().flatten().cpu(), dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"mapas de formas diferentes: {a.shape} vs {b.shape}")
    if x.size < 3:
        raise ValueError(f"mapa pequeno demais para correlacionar: {x.size} pixels")
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(stats.spearmanr(x, y).statistic)


def class_discrimination(
    model: nn.Module,
    x: torch.Tensor,
    class_a: int,
    class_b: int,
    map_fns: Mapping[str, MapFn],
) -> dict[str, float]:
    """Spearman entre o mapa da classe A e o da classe B, por metodo.

    Mede o defeito que o Grad-CAM ataca: se o mapa de "gato" e o de "cachorro"
    na mesma imagem forem quase identicos, o metodo localiza o objeto mas nao
    diz *qual* classe esta vendo.
    """
    maps_a = {name: fn(model, x, class_a) for name, fn in map_fns.items()}
    maps_b = {name: fn(model, x, class_b) for name, fn in map_fns.items()}
    return {name: spearman(maps_a[name], maps_b[name]) for name in map_fns}


def summarize(values: Sequence[float]) -> dict[str, float | int]:
    """Media, mediana, desvio e contagem de valores validos (ignora `nan`)."""
    array = np.asarray(list(values), dtype=np.float64)
    valid = array[~np.isnan(array)]
    if valid.size == 0:
        return {"n_valid": 0, "mean": float("nan"), "median": float("nan"),
                "std": float("nan")}
    return {
        "n_valid": int(valid.size),
        "mean": float(valid.mean()),
        "median": float(np.median(valid)),
        "std": float(valid.std(ddof=1)) if valid.size > 1 else 0.0,
    }
