"""Testes da Fase 6: Guided Grad-CAM como composicao de dois mapas.

O que estes testes fixam, e que nao e' obvio na leitura do codigo:

1. **A composicao e' literal**: `guided_gradcam` tem que ser igual a
   `upsample(gradcam(...)) * guided_backprop(...)` calculado por fora. Se
   alguem "melhorar" o metodo (normalizar antes, aplicar ReLU de novo), o teste
   pega.
2. **O termo Grad-CAM e' o `vanilla`**, nao o `guided`. Trocar a regra muda o
   mapa sem erro nenhum — e' a armadilha silenciosa que `relu_mode` resolveu na
   Fase 4.
3. **O upsample acontece**: com stride 2, o CAM sai 4x4 e o resultado tem que
   sair 8x8.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gradcam.core import gradcam
from gradcam.guided import compose_guided_gradcam, guided_gradcam
from gradcam.metrics import guided_gradcam_map_fn
from gradcam.relu_variants import ReluVariant
from gradcam.saliency import guided_backprop, saliency
from gradcam.viz import upsample


def tiny_model(stride: int = 1) -> nn.Sequential:
    """Conv -> ReLU -> Conv -> ReLU -> GAP -> fc, pesos fixos pela semente.

    `stride=2` deixa o mapa de ativacao menor que a entrada: e' o caso que
    prova que o upsample de fato acontece (o VGG-16 faz 14x14 -> 224x224).
    """
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Conv2d(3, 4, 3, stride=stride, padding=1), nn.ReLU(),
        nn.Conv2d(4, 5, 3, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(5, 7),
    )


def test_compose_upsamples_and_multiplies():
    cam = torch.arange(4.0).reshape(2, 2)
    detail = torch.ones(6, 6)

    out = compose_guided_gradcam(cam, detail)

    assert out.shape == (6, 6)
    assert torch.allclose(out, upsample(cam, (6, 6)))


def test_compose_rejects_bad_shapes():
    with pytest.raises(ValueError):
        compose_guided_gradcam(torch.ones(2, 2, 2), torch.ones(4, 4))
    with pytest.raises(ValueError):
        compose_guided_gradcam(torch.ones(2, 2), torch.ones(4, 4), size=(5, 5))


def test_matches_explicit_composition():
    model = tiny_model()
    x = torch.randn(1, 3, 8, 8)

    heatmap, class_idx = guided_gradcam(model, x, class_idx=2)

    cam, _ = gradcam(model, x, class_idx=2, mode="vanilla")
    detail, _ = guided_backprop(model, x, class_idx=2)
    expected = compose_guided_gradcam(cam, detail, (8, 8))

    assert class_idx == 2
    assert torch.allclose(heatmap, expected, atol=1e-6)


def test_cam_term_uses_the_vanilla_rule():
    """O Grad-CAM da composicao e' o de gradiente comum, nao o `guided`.

    A comparacao usa a **primeira** ReLU (`layer="1"`). Na ultima, as duas
    regras coincidem: o gradiente da ativacao e' `w_k / Z`, constante por canal,
    entao a mascara `grad_out > 0` do `guided` chega tarde demais para mudar o
    mapa. Na primeira, o caminho passa pela segunda conv, e as duas divergem.
    """
    model = tiny_model()
    x = torch.randn(1, 3, 8, 8)

    vanilla, _ = gradcam(model, x, class_idx=2, layer="1", mode="vanilla")
    guided, _ = gradcam(model, x, class_idx=2, layer="1", mode="guided")
    # Sem esta premissa o teste nao discriminaria nada.
    assert (vanilla - guided).abs().max() > 1e-5

    heatmap, _ = guided_gradcam(model, x, class_idx=2, layer="1")
    detail, _ = guided_backprop(model, x, class_idx=2)
    vanilla_composition = compose_guided_gradcam(vanilla, detail, (8, 8))
    guided_composition = compose_guided_gradcam(guided, detail, (8, 8))

    # Comparacao relativa (`atol=0`): os mapas deste modelo sao ~1e-4, e um
    # `atol` absoluto apagaria justamente a diferenca que o teste procura.
    assert torch.allclose(heatmap, vanilla_composition, rtol=1e-5, atol=0)
    assert not torch.allclose(heatmap, guided_composition, rtol=1e-5, atol=0)


def test_upsamples_when_the_cam_is_smaller_than_the_input():
    model = tiny_model(stride=2)
    x = torch.randn(1, 3, 8, 8)

    cam, _ = gradcam(model, x, class_idx=2)
    heatmap, _ = guided_gradcam(model, x, class_idx=2)

    assert cam.shape == (4, 4)
    assert heatmap.shape == (8, 8)


def test_output_is_non_negative_and_finite():
    model = tiny_model()

    heatmap, _ = guided_gradcam(model, torch.randn(1, 3, 8, 8), class_idx=1)

    assert torch.isfinite(heatmap).all()
    assert heatmap.min() >= 0


def test_leaves_no_relu_rule_behind():
    model = tiny_model()
    relus = (model[1], model[3])

    guided_gradcam(model, torch.randn(1, 3, 8, 8), class_idx=0)

    assert model[1] is relus[0]
    assert model[3] is relus[1]
    assert not any(isinstance(module, ReluVariant) for module in model.modules())


def test_independent_of_previous_saliency():
    """Regressao: a regra `guided` do termo de detalhe nao pode vazar."""
    model = tiny_model()
    x = torch.randn(1, 3, 8, 8)

    first, _ = guided_gradcam(model, x, class_idx=2)
    saliency(model, x, class_idx=2, mode="deconv")
    saliency(model, x, class_idx=2, mode="guided")
    again, _ = guided_gradcam(model, x, class_idx=2)

    assert torch.equal(first, again)


def test_infers_class_and_reuses_it_in_both_terms():
    model = tiny_model()
    x = torch.randn(1, 3, 8, 8)

    heatmap, class_idx = guided_gradcam(model, x)

    with torch.no_grad():
        assert class_idx == int(model(x).argmax(dim=1))

    cam, _ = gradcam(model, x, class_idx=class_idx)
    detail, _ = guided_backprop(model, x, class_idx=class_idx)
    assert torch.allclose(
        heatmap, compose_guided_gradcam(cam, detail, (8, 8)), atol=1e-6
    )


def test_map_fn_adapter_matches_the_library_call():
    model = tiny_model()
    x = torch.randn(1, 3, 8, 8)

    expected, _ = guided_gradcam(model, x, class_idx=2)

    assert torch.allclose(guided_gradcam_map_fn()(model, x, 2), expected)


def test_rejects_batches():
    with pytest.raises(ValueError):
        guided_gradcam(tiny_model(), torch.randn(2, 3, 8, 8))
