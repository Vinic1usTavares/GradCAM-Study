"""Testes das Fases 1-2: as tres regras de backward pelo ReLU e a saliencia."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gradcam.relu_variants import MODES, ReluVariant, relu_mode, set_relu_mode
from gradcam.saliency import saliency

# Entrada com sinais mistos: posicoes 0 e 2 negativas, 1 e 3 positivas.
X = torch.tensor([-1.0, 2.0, -3.0, 4.0])


class TinyModel(nn.Module):
    """modelo de 4 entradas / 2 classes com pesos fixos, sem conv.

    y0 = +sum(relu(x))  -> gradiente a montante todo positivo
    y1 = -sum(relu(x))  -> gradiente a montante todo negativo

    E' o menor modelo que separa as tres regras: `vanilla` olha o sinal de x,
    `deconv` olha o sinal do gradiente a montante e `guided` olha os dois. Com
    y0 (a montante > 0) vanilla e guided coincidem; com y1 (a montante < 0)
    deconv e guided zeram tudo e vanilla nao.
    """

    def __init__(self, mode: str = "vanilla") -> None:
        super().__init__()
        self.relu = ReluVariant(mode)
        self.fc = nn.Linear(4, 2, bias=False)
        with torch.no_grad():
            self.fc.weight.copy_(torch.tensor([[1.0] * 4, [-1.0] * 4]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.relu(x))


def _grad(model: nn.Module, x: torch.Tensor, class_idx: int, mode: str) -> torch.Tensor:
    """Saliencia pela API publica; devolve o mapa (4,) achatado."""
    heatmap, returned = saliency(model, x.unsqueeze(0), class_idx, mode=mode)
    assert returned == class_idx
    return heatmap


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("vanilla", [0.0, 1.0, 0.0, 1.0]),
        ("deconv", [1.0, 1.0, 1.0, 1.0]),
        ("guided", [0.0, 1.0, 0.0, 1.0]),
    ],
)
def test_relu_rule_with_positive_upstream(mode, expected):
    grad = _grad(TinyModel(), X, class_idx=0, mode=mode)
    assert torch.equal(grad, torch.tensor(expected))


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("vanilla", [0.0, 1.0, 0.0, 1.0]),
        ("deconv", [0.0, 0.0, 0.0, 0.0]),
        ("guided", [0.0, 0.0, 0.0, 0.0]),
    ],
)
def test_relu_rule_with_negative_upstream(mode, expected):
    grad = _grad(TinyModel(), X, class_idx=1, mode=mode)
    assert torch.equal(grad, torch.tensor(expected))


def test_vanilla_matches_autograd():
    """`vanilla` tem que ser o gradiente comum — inclusive a mascara x > 0.

    Teste de regressao: a primeira versao deixava o gradiente passar sempre
    pelo ReLU, o que nao e' o gradiente da rede (e produzia mapas densos com
    pico ~10^6 em vez de contornos).
    """
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Conv2d(3, 4, 3, padding=1), nn.ReLU(),
        nn.Conv2d(4, 2, 3, padding=1), nn.ReLU(),
        nn.Flatten(), nn.Linear(2 * 8 * 8, 5),
    )
    x = torch.randn(1, 3, 8, 8)

    xr = x.clone().requires_grad_(True)
    (grad,) = torch.autograd.grad(model(xr)[0, 3], xr)
    expected = grad[0].abs().amax(dim=0)

    heatmap, _ = saliency(model, x, 3, mode="vanilla")
    assert torch.allclose(heatmap, expected, atol=1e-6)


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        ReluVariant("guided-backprop")
    with pytest.raises(ValueError):
        set_relu_mode(TinyModel(), "nope")


def test_set_relu_mode_preserves_weights_and_output():
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
    x = torch.randn(3, 4)

    with torch.no_grad():
        before = model(x).clone()
    keys = list(model.state_dict())

    set_relu_mode(model, "guided")

    assert isinstance(model[1], ReluVariant) and model[1].mode == "guided"
    assert list(model.state_dict()) == keys
    with torch.no_grad():
        assert torch.allclose(model(x), before)


def test_relu_mode_restores_the_original_module():
    """A regra vale so' no bloco: o `nn.ReLU` original volta, mesmo objeto."""
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
    original = model[1]

    with relu_mode(model, "guided"):
        assert isinstance(model[1], ReluVariant)
        assert model[1].mode == "guided"

    assert model[1] is original
    assert not isinstance(model[1], ReluVariant)


def test_relu_mode_restores_on_exception():
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
    original = model[1]

    with pytest.raises(RuntimeError, match="falhou no meio"):
        with relu_mode(model, "deconv"):
            raise RuntimeError("falhou no meio")

    assert model[1] is original


def test_relu_mode_restores_the_previous_variant_mode():
    """Um `ReluVariant` que ja' existia mantem a identidade; so' o modo muda."""
    model = TinyModel("deconv")

    with relu_mode(model, "guided"):
        assert model.relu.mode == "guided"

    assert model.relu.mode == "deconv"


def test_relu_mode_rejects_unknown_mode():
    with pytest.raises(ValueError):
        with relu_mode(TinyModel(), "nope"):
            pass


def test_saliency_leaves_no_rule_behind():
    model = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU(),
                          nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 5))
    original = model[1]

    saliency(model, torch.randn(1, 3, 8, 8), mode="guided")

    assert model[1] is original
    assert isinstance(model[1], nn.ReLU)
    assert not isinstance(model[1], ReluVariant)


def test_saliency_is_independent_of_previous_calls():
    """Regressao: `guided` deixava a mascara instalada para a proxima chamada."""
    model = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU(),
                          nn.Conv2d(4, 2, 3, padding=1), nn.ReLU(),
                          nn.Flatten(), nn.Linear(2 * 8 * 8, 5))
    x = torch.randn(1, 3, 8, 8)

    vanilla, _ = saliency(model, x, 3, mode="vanilla")
    saliency(model, x, 3, mode="deconv")
    saliency(model, x, 3, mode="guided")
    again, _ = saliency(model, x, 3, mode="vanilla")

    assert torch.equal(vanilla, again)


def test_saliency_map_is_non_negative_and_input_sized():
    model = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU(),
                          nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 5))
    heatmap, class_idx = saliency(model, torch.randn(1, 3, 8, 8))

    assert heatmap.shape == (8, 8)
    assert heatmap.min() >= 0
    assert 0 <= class_idx < 5


def test_saliency_accepts_forced_class():
    model = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU(),
                          nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 5))
    _, class_idx = saliency(model, torch.randn(1, 3, 8, 8), class_idx=3)
    assert class_idx == 3
    assert MODES == ("vanilla", "deconv", "guided")
