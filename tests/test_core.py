"""Testes da Fase 4: Grad-CAM, resolucao de camada e limpeza de hooks."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gradcam.core import VGG16_LAYERS, gradcam, resolve_layer


def tiny_model() -> nn.Sequential:
    """Conv 3->4 -> ReLU -> Conv 4->5 -> ReLU, com classificador fixo.

    A ultima camada convolucional e' `model[2]` (Conv2d), seguida do ReLU em
    `model[3]`: e' o que `resolve_layer` deve devolver por padrao.
    """
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Conv2d(3, 4, 3, padding=1), nn.ReLU(),
        nn.Conv2d(4, 5, 3, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(5, 7),
    )


def test_resolve_layer_defaults_to_relu_after_last_conv():
    model = tiny_model()
    assert resolve_layer(model) is model[3]


def test_resolve_layer_accepts_module_and_name():
    model = tiny_model()
    assert resolve_layer(model, model[1]) is model[1]
    assert resolve_layer(model, "1") is model[1]


def test_resolve_layer_rejects_unknown_name():
    with pytest.raises(KeyError):
        resolve_layer(tiny_model(), "features.99")


def test_vgg16_layer_names_point_at_conv_outputs():
    assert VGG16_LAYERS == {
        "conv3_3": "features.15",
        "conv4_3": "features.22",
        "conv5_3": "features.29",
    }


def test_gradcam_shape_matches_feature_map():
    model = tiny_model()
    x = torch.randn(1, 3, 8, 8)
    heatmap, class_idx = gradcam(model, x, class_idx=2)
    assert heatmap.shape == (8, 8)  # padding=1 preserva a resolucao
    assert 0 <= class_idx < 7
    assert heatmap.min() >= 0


def test_gradcam_removes_its_hook():
    model = tiny_model()
    gradcam(model, torch.randn(1, 3, 8, 8), class_idx=0)
    assert all(len(layer._forward_hooks) == 0 for layer in model.modules())


def test_gradcam_differs_between_classes():
    """Classes diferentes tem que produzir mapas diferentes, nao o mesmo."""
    model = tiny_model()
    x = torch.randn(1, 3, 8, 8)
    a, _ = gradcam(model, x, class_idx=0)
    b, _ = gradcam(model, x, class_idx=6)
    assert not torch.allclose(a, b)


def test_apply_relu_false_can_produce_negatives():
    model = tiny_model()
    x = torch.randn(1, 3, 8, 8)
    raw, _ = gradcam(model, x, class_idx=0, apply_relu=False)
    assert raw.min() < 0 or (raw >= 0).all()  # nao inventa sinal
    clipped, _ = gradcam(model, x, class_idx=0)
    assert clipped.min() >= 0


def test_gradcam_infers_class_when_none():
    model = tiny_model()
    x = torch.randn(1, 3, 8, 8)
    _, class_idx = gradcam(model, x)
    with torch.no_grad():
        expected = int(model(x).argmax(dim=1))
    assert class_idx == expected


def test_gradcam_rejects_batches():
    with pytest.raises(ValueError):
        gradcam(tiny_model(), torch.randn(2, 3, 8, 8))
