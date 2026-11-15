"""Fases 1-2: saliencia por gradiente em relacao a entrada.

    saliency(x) = max_k | d y^c / d x_k |        (Simonyan et al., 2013)

Com `set_relu_mode(model, "guided")` o mesmo codigo produz Guided Backprop;
com `"deconv"`, Deconvolution. Nada mais muda.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .relu_variants import MODES, set_relu_mode


def input_gradient(
    model: nn.Module,
    x: torch.Tensor,
    class_idx: int | None = None,
) -> tuple[torch.Tensor, int]:
    """Gradiente do score da classe em relacao a entrada.

    Retorna `(grad, class_idx)` com `grad` de forma (3, H, W), com sinal.
    Usa o score ANTES do softmax, como no artigo (secao 3).

    `x` deve vir ja normalizado; a desnormalizacao e' so para exibir.
    """
    model.zero_grad(set_to_none=True)
    x = x.detach().clone().requires_grad_(True)

    scores = model(x)
    if class_idx is None:
        class_idx = int(scores.argmax(dim=1).item())

    (grad,) = torch.autograd.grad(scores[0, class_idx], x)
    return grad[0], class_idx


def saliency_map(grad: torch.Tensor) -> torch.Tensor:
    """(3, H, W) com sinal -> (H, W) nao-negativo, pelo maximo do modulo por pixel."""
    return grad.abs().amax(dim=0)


def saliency(
    model: nn.Module,
    x: torch.Tensor,
    class_idx: int | None = None,
    mode: str = "vanilla",
) -> tuple[torch.Tensor, int]:
    """Saliencia em uma chamada: configura a regra de ReLU e calcula o mapa."""
    if mode not in MODES:
        raise ValueError(f"modo invalido: {mode!r}; use um de {MODES}")
    set_relu_mode(model, mode)
    grad, class_idx = input_gradient(model, x, class_idx)
    return saliency_map(grad), class_idx


def guided_backprop(
    model: nn.Module,
    x: torch.Tensor,
    class_idx: int | None = None,
) -> tuple[torch.Tensor, int]:
    """Atalho para Guided Backpropagation."""
    return saliency(model, x, class_idx, mode="guided")
