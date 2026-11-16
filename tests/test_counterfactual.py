"""Testes da Fase 7: contrafactual = Grad-CAM com `alpha` negado em (1)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gradcam.core import counterfactual, gradcam


class TinyNet(nn.Module):
    """Rede minima com a mesma topologia de interesse: conv -> GAP -> fc."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 4, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(4, 6, 3, padding=1),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(6, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.pool(self.features(x)).flatten(1))


def make_model() -> tuple[TinyNet, torch.Tensor]:
    torch.manual_seed(0)
    model = TinyNet().eval()
    return model, torch.randn(1, 3, 16, 16)


def test_counterfactual_shape_and_range():
    model, x = make_model()
    heatmap, class_idx = counterfactual(model, x, class_idx=1)
    assert heatmap.shape == (16, 16)
    assert heatmap.min() >= 0
    assert class_idx == 1


def test_counterfactual_is_gradcam_with_negated_alpha():
    """Sem ReLU, o contrafactual e' exatamente o negativo do Grad-CAM."""
    model, x = make_model()
    cf, _ = counterfactual(model, x, class_idx=2, apply_relu=False)
    gc, _ = gradcam(model, x, class_idx=2, apply_relu=False)
    assert torch.allclose(cf, -gc, atol=1e-6)


def test_counterfactual_infers_argmax_like_gradcam():
    model, x = make_model()
    expected = int(model(x).argmax(dim=1).item())
    _, gc_idx = gradcam(model, x)
    _, cf_idx = counterfactual(model, x)
    assert gc_idx == cf_idx == expected


def test_counterfactual_removes_hook_between_calls():
    """Duas chamadas seguidas nao podem acumular ativacoes (hook vazando)."""
    model, x = make_model()
    first, _ = counterfactual(model, x, class_idx=0)
    second, _ = counterfactual(model, x, class_idx=0)
    assert torch.equal(first, second)
    assert len(model.features[2]._forward_hooks) == 0
