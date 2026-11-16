"""Testes da Fase 10: cascade de randomizacao de parametros (Adebayo et al.)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gradcam.metrics import gradcam_map_fn
from gradcam.sanity import (
    cascade,
    parameter_modules,
    randomization_order,
    reinitialize_,
    sanity_curve,
)


class TinyNet(nn.Module):
    """Mesma topologia de interesse: conv -> GAP -> fc (o Grad-CAM funciona)."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 4, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(4, 6, 3, padding=1),
            nn.BatchNorm2d(6),
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


def snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: param.detach().clone() for name, param in model.named_parameters()}


def test_order_is_output_to_input_and_skips_stateless_modules():
    model, _ = make_model()
    order = randomization_order(model)
    names = [name for name, _ in order]

    # O classifier vem primeiro: e' o topo da rede.
    assert names[0] == "fc"
    assert names[-1] == "features.0"
    # ReLU/Pool nao tem parametro: nao entram no cascade.
    assert "features.1" not in names
    assert "pool" not in names
    assert len(names) == len(parameter_modules(model))


def test_parameter_modules_and_order_are_reverses():
    model, _ = make_model()
    forward = [name for name, _ in parameter_modules(model)]
    assert [name for name, _ in randomization_order(model)] == forward[::-1]


def test_reinitialize_changes_values_but_not_shapes():
    model, _ = make_model()
    before = snapshot(model)

    generator = torch.Generator().manual_seed(7)
    reinitialize_(model.features[0], generator)

    after = {name: param for name, param in model.named_parameters()}
    assert after["features.0.weight"].shape == before["features.0.weight"].shape
    assert not torch.equal(after["features.0.weight"], before["features.0.weight"])
    # Bias volta a zero, como na inicializacao do torchvision.
    assert torch.count_nonzero(after["features.0.bias"]) == 0
    # Os outros modulos nao foram tocados.
    assert torch.equal(after["fc.weight"], before["fc.weight"])


def test_reinitialize_is_deterministic_given_the_seed():
    model, _ = make_model()
    first = model.features[0]
    reinitialize_(first, torch.Generator().manual_seed(3))
    reference = first.weight.detach().clone()

    reinitialize_(first, torch.Generator().manual_seed(3))
    assert torch.equal(first.weight, reference)


def test_reinitialize_keeps_batchnorm_usable():
    """Zerar o `weight` 1-D do BatchNorm apagaria a camada; deve virar 1."""
    model, x = make_model()
    reinitialize_(model.features[3], torch.Generator().manual_seed(0))

    assert torch.allclose(model.features[3].weight, torch.ones(6))
    assert torch.allclose(model.features[3].bias, torch.zeros(6))
    out = model(x)
    assert torch.isfinite(out).all()


def test_cascade_is_cumulative():
    """Cada passo preserva as randomizacoes anteriores (nao sorteia modelo novo)."""
    model, _ = make_model()
    order = randomization_order(model)
    generator = torch.Generator().manual_seed(11)

    added_first = cascade(model, order, 2, generator)
    assert added_first == [name for name, _ in order[:2]]
    frozen = snapshot(model)

    # Re-randomizar o mesmo k com a mesma semente e' idempotente do ponto de
    # vista do chamador: `already=2` nao toca nos dois primeiros.
    added_next = cascade(model, order, 3, generator, already=2)
    assert added_next == [order[2][0]]
    after = snapshot(model)
    for name, _ in order[:2]:
        for param_name, value in after.items():
            if param_name.startswith(name + "."):
                assert torch.equal(value, frozen[param_name])


def test_sanity_curve_starts_at_one_and_is_destructive():
    model, x = make_model()
    before = snapshot(model)
    curve = sanity_curve(
        model, x, class_idx=1, map_fn=gradcam_map_fn(), cutoffs=(0, 2, 4), seed=0
    )

    assert [step["k"] for step in curve["steps"]] == [0, 2, 4]
    assert curve["steps"][0]["spearman"] == pytest.approx(1.0)
    assert curve["steps"][0]["abs_spearman"] == pytest.approx(1.0)
    # O modulo e' a leitura de Adebayo: um modelo randomizado pode inverter o
    # sinal do gradiente, e o que se pergunta e' se a estrutura sobreviveu.
    assert all(
        step["abs_spearman"] == pytest.approx(abs(step["spearman"]))
        for step in curve["steps"]
    )
    assert curve["steps"][0]["added"] == []
    assert curve["steps"][1]["added"] == curve["order"][:2]
    assert curve["n_parameters_modules"] == len(curve["order"])
    # O modelo sai randomizado (documentado): e' o preco de nao copiar 528 MB.
    assert not torch.equal(model.fc.weight, before["fc.weight"])


def test_sanity_curve_keeps_the_class_fixed():
    """A classe nao pode ser re-inferida a cada passo, senao mede outra coisa."""
    model, x = make_model()
    seen: list[int] = []

    def map_fn(model, x, class_idx):
        seen.append(class_idx)
        return gradcam_map_fn()(model, x, class_idx)

    curve = sanity_curve(model, x, class_idx=2, map_fn=map_fn, cutoffs=(0, 1), seed=0)
    assert set(seen) == {2}
    assert curve["class_idx"] == 2
    assert all(step["predicted_idx"] in (0, 1, 2) for step in curve["steps"])


def test_sanity_curve_rejects_cutoff_beyond_the_model():
    model, x = make_model()
    with pytest.raises(ValueError, match="maior que os"):
        sanity_curve(model, x, class_idx=0, map_fn=gradcam_map_fn(), cutoffs=(0, 99))
