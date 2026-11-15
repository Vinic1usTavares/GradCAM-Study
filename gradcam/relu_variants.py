"""As tres regras de backward pelo ReLU.

Os metodos de saliencia por gradiente diferem apenas na mascara aplicada ao
gradiente quando ele atravessa o ReLU, na volta:

    vanilla | sempre passa
    deconv  | passa se grad_out > 0          (Zeiler & Fergus, 2014)
    guided  | passa se grad_out > 0 e x > 0  (Springenberg et al., 2014)

Ver plans/general_plan.md, secao 2.2.

Nota conceitual: `guided` e `deconv` NAO sao gradientes de nenhuma funcao (o
operador deixa de ser conservativo). Sao heuristicas de visualizacao; isso
importa para a discussao de sanity checks no seminario.
"""

from __future__ import annotations

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
        if ctx.mode == "deconv":
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
