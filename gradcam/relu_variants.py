"""As tres regras de backward pelo ReLU.

Os metodos de saliencia por gradiente diferem apenas na mascara aplicada ao
gradiente quando ele atravessa o ReLU, na volta:

    vanilla | passa se x > 0                  (gradiente comum / Simonyan 2013)
    deconv  | passa se grad_out > 0           (Zeiler & Fergus, 2014)
    guided  | passa se grad_out > 0 e x > 0   (Springenberg et al., 2014)

`vanilla` **nao** e' "deixa passar sempre": o gradiente comum de `relu` ja'
zera onde a entrada e' negativa. Deixar passar tudo seria o gradiente de uma
rede com os ReLUs linearizados na volta — outro objeto, com mapas densos e
magnitudes muito maiores. Essa era a definicao errada na primeira versao desta
implementacao; o teste `test_vanilla_matches_autograd` fixa a correta.

Ver plans/general_plan.md, secao 2.2.

Nota conceitual: `guided` e `deconv` NAO sao gradientes de nenhuma funcao (o
operador deixa de ser conservativo). Sao heuristicas de visualizacao; isso
importa para a discussao de sanity checks no seminario.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch
import torch.nn as nn

MODES = ("vanilla", "deconv", "guided")


class _ReluVariant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, mode: str) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.mode = mode
        return x.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (x,) = ctx.saved_tensors
        grad = grad_output
        if ctx.mode == "vanilla":
            grad = grad * (x > 0)
        elif ctx.mode == "deconv":
            grad = grad * (grad > 0)
        elif ctx.mode == "guided":
            grad = grad * (grad > 0) * (x > 0)
        return grad, None


class ReluVariant(nn.Module):
    """Substituto de `nn.ReLU` cuja regra de backward e' configuravel."""

    def __init__(self, mode: str = "vanilla") -> None:
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"modo invalido: {mode!r}; use um de {MODES}")
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _ReluVariant.apply(x, self.mode)

    def extra_repr(self) -> str:
        return f"mode={self.mode!r}"


def set_relu_mode(model: nn.Module, mode: str) -> nn.Module:
    """Troca, no lugar, todo `nn.ReLU`/`ReluVariant` por `ReluVariant(mode)`.

    ReLU nao tem parametros, entao a substituicao preserva os pesos
    pre-treinados. O modelo e' modificado e devolvido.

    Prefira `relu_mode` (context manager): esta funcao deixa a regra instalada
    no modelo, e o proximo metodo a rodar no mesmo modelo herda a mascara.
    """
    if mode not in MODES:
        raise ValueError(f"modo invalido: {mode!r}; use um de {MODES}")

    def _walk(module: nn.Module) -> None:
        for name, child in module.named_children():
            if isinstance(child, (nn.ReLU, ReluVariant)):
                setattr(module, name, ReluVariant(mode))
            else:
                _walk(child)

    _walk(model)
    return model


def _relu_children(model: nn.Module) -> Iterator[tuple[nn.Module, str, nn.Module]]:
    """Pares (pai, nome, filho) de todo ReLU-like, sem recursao em ReLU."""
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, (nn.ReLU, ReluVariant)):
                yield parent, name, child


@contextmanager
def relu_mode(model: nn.Module, mode: str) -> Iterator[nn.Module]:
    """Aplica a regra de backward `mode` so' durante o bloco.

    `set_relu_mode` troca os modulos e nao volta: como Grad-CAM e as saliencias
    fazem backward pela mesma rede, o resultado passava a depender da ordem das
    chamadas. Medido antes da correcao, na mesma imagem e classes:

        recem-carregado     rho = 0,874
        apos saliency guided rho = 0,992
        apos saliency deconv rho = 0,296

    Aqui os modulos de `ReluVariant` mantem a identidade (so' o `mode` muda) e
    os `nn.ReLU` originais sao repostos no `finally` — inclusive se o backward
    levantar. Assim a regra nunca sobrevive a chamada.
    """
    if mode not in MODES:
        raise ValueError(f"modo invalido: {mode!r}; use um de {MODES}")

    saved: list[tuple[nn.Module, str, nn.Module, str | None]] = []
    for parent, name, child in _relu_children(model):
        if isinstance(child, ReluVariant):
            saved.append((parent, name, child, child.mode))
            child.mode = mode
        else:
            saved.append((parent, name, child, None))
            setattr(parent, name, ReluVariant(mode))

    try:
        yield model
    finally:
        for parent, name, original, previous in reversed(saved):
            if previous is None:
                setattr(parent, name, original)
            else:
                original.mode = previous
