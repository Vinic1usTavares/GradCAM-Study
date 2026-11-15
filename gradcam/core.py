"""Grad-CAM (Selvaraju et al., 2016), Eqs. 1-2 do artigo.

    alpha_k^c = (1/Z) sum_ij  d y^c / d A^k_ij        (1)
    L^c       = ReLU( sum_k alpha_k^c . A^k )         (2)

`y^c` e' o score **antes** do softmax (o artigo e' explicito, secao 3). A
semente do backward e' uma one-hot na classe `c`, o que zera o gradiente das
demais classes (Fig. 2 do artigo).

O ReLU em (2) nao e' opcional: sem ele o erro de localizacao sobe (Tabela 3).
Ele tambem e' o que permite o contrafactual da Fase 7: negar `alpha` em (1)
antes do ReLU desloca o mapa para a evidencia contraria.

Assinatura compativel com `metrics.MapFn`: `(modelo, entrada, classe) -> HxW`.
Aqui a classe pode ser `None` (usa o argmax), o que `MapFn` nao permite — a
avaliacao sempre passa a classe explicita.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .relu_variants import ReluVariant

# Camadas nomeadas do VGG-16 para a ablacao da Fase 9. Os indices sao do
# `features` do torchvision: ReLU logo apos a conv correspondente.
VGG16_LAYERS = {
    "conv3_3": "features.15",
    "conv4_3": "features.22",
    "conv5_3": "features.29",
}


def resolve_layer(model: nn.Module, layer: nn.Module | str | None = None) -> nn.Module:
    """Aceita um modulo, um nome (`"features.29"`) ou `None` (ultima conv).

    Sem camada explicita, usa o ReLU que vem logo depois do ultimo `Conv2d` —
    ou o proprio `Conv2d`, se ele for seguido de outra coisa (o que acontece
    no ResNet, cujo conv final e' seguido de BatchNorm). Usar a saida pos-ReLU
    da ultima conv e' o que a maioria das implementacoes faz e corresponde ao
    `conv5_3` do artigo.
    """
    if isinstance(layer, nn.Module):
        return layer

    if isinstance(layer, str):
        modules = dict(model.named_modules())
        if layer not in modules:
            raise KeyError(f"camada inexistente: {layer!r}")
        return modules[layer]

    modules = list(model.modules())
    for index in range(len(modules) - 1, -1, -1):
        if not isinstance(modules[index], nn.Conv2d):
            continue
        following = modules[index + 1] if index + 1 < len(modules) else None
        if isinstance(following, (nn.ReLU, ReluVariant)):
            return following
        return modules[index]

    raise ValueError(
        "modelo sem Conv2d: passe `layer` explicitamente (mapa de um modelo "
        "so' faz sentido em alguma camada convolucional)"
    )


def _forward_with_activations(
    model: nn.Module, x: torch.Tensor, layer: nn.Module
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward guardando as ativacoes de `layer`. Remove o hook sempre."""
    captured: list[torch.Tensor] = []
    handle = layer.register_forward_hook(
        lambda _module, _inputs, output: captured.append(output)
    )
    try:
        with torch.enable_grad():
            scores = model(x)
    finally:
        handle.remove()

    if not captured:
        raise RuntimeError(f"a camada {layer} nao foi executada no forward")
    return scores, captured[0]


def gradcam(
    model: nn.Module,
    x: torch.Tensor,
    class_idx: int | None = None,
    layer: nn.Module | str | None = None,
    apply_relu: bool = True,
) -> tuple[torch.Tensor, int]:
    """Mapa Grad-CAM de uma imagem e uma classe.

    Retorna `(mapa HxW, class_idx)`. O mapa sai **cru** (sem normalizar):
    as metricas trabalham com ele assim, e a escala de exibicao fica em
    `viz.scale_for_display`.

    `apply_relu=False` desliga o ReLU de (2) e existe para a ablacao
    "sem ReLU" da Tabela 3 do artigo — nao e' o metodo.
    """
    if x.ndim != 4 or x.shape[0] != 1:
        raise ValueError(f"esperava (1, C, H, W); recebi {tuple(x.shape)}")

    target = resolve_layer(model, layer)
    x = x.detach().clone().requires_grad_(True)

    scores, activations = _forward_with_activations(model, x, target)
    if class_idx is None:
        class_idx = int(scores.argmax(dim=1).item())

    (grad,) = torch.autograd.grad(scores[0, class_idx], activations)

    # (1): alpha_k = media espacial do gradiente da ativacao k.
    alpha = grad.mean(dim=(2, 3), keepdim=True)
    # (2): combinacao linear das ativacoes com os pesos alpha.
    weighted = (alpha * activations).sum(dim=1)[0]
    return (weighted.clamp(min=0) if apply_relu else weighted), class_idx
