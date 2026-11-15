"""Testes da Fase 3: correlacao de Spearman e o contrato `MapFn`."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gradcam.metrics import (
    class_discrimination,
    saliency_map_fn,
    saliency_map_fns,
    spearman,
    summarize,
)


def test_spearman_perfect_positive():
    a = torch.arange(25, dtype=torch.float32).reshape(5, 5)
    assert spearman(a, a * 3 + 1) == pytest.approx(1.0)


def test_spearman_perfect_negative():
    a = torch.arange(25, dtype=torch.float32).reshape(5, 5)
    assert spearman(a, -a) == pytest.approx(-1.0)


def test_spearman_monotone_transform_is_invariant():
    a = torch.rand(16, 16)
    # Spearman ignora escala: e' por isso que a normalizacao de figura nao
    # contamina a metrica.
    assert spearman(a, a.pow(2)) == pytest.approx(1.0)


def test_spearman_constant_map_is_undefined():
    constant = torch.ones(7, 7)
    value = spearman(constant, torch.rand(7, 7))
    assert math.isnan(value)


def test_spearman_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        spearman(torch.rand(4, 4), torch.rand(4, 5))


def test_saliency_map_fn_matches_map_fn_signature():
    fn = saliency_map_fn("guided")
    model = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU(),
                          nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 5))
    heatmap = fn(model, torch.randn(1, 3, 8, 8), 2)
    assert heatmap.shape == (8, 8)
    assert heatmap.min() >= 0


def test_saliency_map_fns_covers_the_three_modes():
    assert list(saliency_map_fns()) == ["vanilla", "deconv", "guided"]
    with pytest.raises(ValueError):
        saliency_map_fn("cam")


def test_gradcam_map_fn_matches_core():
    """O adaptador tem que entregar exatamente o mapa de `core.gradcam`."""
    from gradcam.core import gradcam
    from gradcam.metrics import gradcam_map_fn

    model = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU(),
                          nn.Conv2d(4, 6, 3, padding=1), nn.ReLU(),
                          nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(6, 5))
    x = torch.randn(1, 3, 8, 8)

    expected, _ = gradcam(model, x, 3)
    heatmap = gradcam_map_fn()(model, x, 3)
    assert torch.allclose(heatmap, expected)
    assert heatmap.shape == (8, 8)
    assert heatmap.min() >= 0

    # `layer` e' repassado: explicitar a camada nao muda o padrao aqui (ela ja'
    # e' a escolhida por `resolve_layer`), mas tem que continuar valida.
    assert torch.allclose(gradcam_map_fn(model[3])(model, x, 3), expected)


class LinearMap(nn.Module):
    """modelo sem ReLU com pesos fixos positivos: gradiente sempre nao-constante."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(12, 3, bias=False)
        with torch.no_grad():
            self.fc.weight.copy_(
                torch.arange(1, 37, dtype=torch.float32).reshape(3, 12)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.flatten(1))


def test_class_discrimination_identical_class_gives_one():
    """A mesma classe nos dois lados tem que dar rho = 1 em qualquer metodo."""
    model = LinearMap()
    x = torch.full((1, 3, 2, 2), 2.0)  # positivo: guided e' identico a vanilla
    rhos = class_discrimination(model, x, 1, 1, saliency_map_fns())
    assert set(rhos) == {"vanilla", "deconv", "guided"}
    for value in rhos.values():
        assert value == pytest.approx(1.0)


def test_summarize_ignores_nan():
    result = summarize([1.0, float("nan"), 0.5, 0.75])
    assert result["n_valid"] == 3
    assert result["mean"] == pytest.approx(0.75)


def test_summarize_all_nan_is_empty():
    result = summarize([float("nan"), float("nan")])
    assert result["n_valid"] == 0
    assert math.isnan(result["mean"])
