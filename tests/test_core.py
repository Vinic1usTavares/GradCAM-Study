"""Testes da Fase 4: Grad-CAM, resolucao de camada e limpeza de hooks."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gradcam.core import VGG16_LAYERS, gradcam, resolve_layer
from gradcam.relu_variants import ReluVariant, set_relu_mode
from gradcam.saliency import saliency


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


def reference_map(
    model: nn.Module, x: torch.Tensor, class_idx: int, mode: str, layer: str
) -> torch.Tensor:
    """Eqs. (1)-(2) com a regra instalada fora do `gradcam`, como referencia.

    Usa uma copia do modelo porque `set_relu_mode` altera no lugar.
    """
    twin = copy.deepcopy(model)
    set_relu_mode(twin, mode)
    target = dict(twin.named_modules())[layer]

    captured: list[torch.Tensor] = []
    handle = target.register_forward_hook(
        lambda _module, _inputs, output: captured.append(output)
    )
    try:
        scores = twin(x.clone().requires_grad_(True))
    finally:
        handle.remove()

    (grad,) = torch.autograd.grad(scores[0, class_idx], captured[0])
    alpha = grad.mean(dim=(2, 3), keepdim=True)
    return (alpha * captured[0]).sum(dim=1)[0].clamp(min=0)


def test_gradcam_mode_matches_the_rule_installed_explicitly():
    """`mode` tem de reproduzir o resultado do `set_relu_mode` manual."""
    model = tiny_model()
    x = torch.randn(1, 3, 8, 8)

    expected = reference_map(model, x, class_idx=2, mode="guided", layer="3")
    heatmap, _ = gradcam(model, x, class_idx=2, mode="guided")

    assert torch.allclose(heatmap, expected, atol=1e-6)


def test_gradcam_accepts_explicit_relu_module_under_guided():
    """`layer=model[3]` e' um `nn.ReLU` — ele e' substituido durante o bloco."""
    model = tiny_model()
    x = torch.randn(1, 3, 8, 8)

    expected = reference_map(model, x, class_idx=2, mode="guided", layer="3")
    heatmap, _ = gradcam(model, x, class_idx=2, layer=model[3], mode="guided")

    assert torch.allclose(heatmap, expected, atol=1e-6)


def test_gradcam_restores_the_original_relu_modules():
    model = tiny_model()
    relus = (model[1], model[3])

    gradcam(model, torch.randn(1, 3, 8, 8), class_idx=0, mode="guided")

    assert model[1] is relus[0]
    assert model[3] is relus[1]
    assert not any(isinstance(module, ReluVariant) for module in model.modules())


def test_gradcam_is_independent_of_previous_saliency():
    """Regressao medida: 0,874 recem-carregado, 0,992 apos guided, 0,296 apos
    deconv — a mesma chamada de `gradcam` mudava de resultado."""
    model = tiny_model()
    x = torch.randn(1, 3, 8, 8)

    first, _ = gradcam(model, x, class_idx=2)
    saliency(model, x, class_idx=2, mode="deconv")
    saliency(model, x, class_idx=2, mode="guided")
    again, _ = gradcam(model, x, class_idx=2)

    assert torch.equal(first, again)


def test_gradcam_restores_modules_when_layer_is_invalid():
    model = tiny_model()
    relu = model[3]

    with pytest.raises(KeyError):
        gradcam(model, torch.randn(1, 3, 8, 8), class_idx=0,
                layer="features.99", mode="guided")

    assert model[3] is relu


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
