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

from .relu_variants import ReluVariant, relu_mode

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


def _qualified_name(model: nn.Module, module: nn.Module) -> str | None:
    """Nome de `module` dentro de `model`, ou `None` se ele nao for do modelo.

    `nn.Module` nao sobrescreve `__eq__`, entao a comparacao e' de identidade.
    """
    for name, candidate in model.named_modules():
        if candidate is module:
            return name
    return None


def _name_for_layer(
    model: nn.Module, layer: nn.Module | str | None
) -> nn.Module | str | None:
    """Troca um modulo por seu nome **antes** da substituicao dos ReLUs.

    `relu_mode` troca os `nn.ReLU` por `ReluVariant`; um modulo recebido
    explicitamente (tipicamente `model[3]`, um ReLU) deixaria de estar no grafo
    e o forward hook nunca dispararia. Nome continua valido depois da troca, e
    `resolve_layer` ja' aceita nome, modulo ou `None`. Modulo que nao pertence
    ao modelo fica como esta (pode ser um wrapper externo).
    """
    if not isinstance(layer, nn.Module):
        return layer
    name = _qualified_name(model, layer)
    return layer if name in (None, "") else name


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


def resolve_cam_layers(
    model: nn.Module,
    layer: nn.Module | str | None = None,
    fc: nn.Linear | str | None = None,
) -> tuple[nn.Module, nn.Linear]:
    """Resolve a camada de ativacao e a camada linear para o CAM (Zhou et al., 2016).

    O CAM exige arquitetura com Global Average Pooling (GAP) seguido de uma
    camada linear (`conv -> GAP -> fc`):
    - `layer`: camada anterior ao GAP. Se `None`, usa `model.layer4` se existir
      (padrao ResNet), senao a ultima conv/bloco por `resolve_layer`.
    - `fc`: camada linear de classificacao. Se `None`, usa `model.fc` se
      existir, senao o ultimo `nn.Linear` do modelo.
    """
    if layer is not None:
        target_layer = resolve_layer(model, layer)
    elif hasattr(model, "layer4") and isinstance(model.layer4, nn.Module):
        target_layer = model.layer4
    else:
        target_layer = resolve_layer(model, None)

    if fc is None:
        if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
            fc_layer = model.fc
        else:
            linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
            if not linears:
                raise ValueError("modelo sem camada linear (nn.Linear) para o CAM")
            fc_layer = linears[-1]
    elif isinstance(fc, str):
        modules = dict(model.named_modules())
        if fc not in modules:
            raise KeyError(f"camada linear inexistente: {fc!r}")
        if not isinstance(modules[fc], nn.Linear):
            raise TypeError(f"camada {fc!r} nao e' nn.Linear")
        fc_layer = modules[fc]
    else:
        if not isinstance(fc, nn.Linear):
            raise TypeError(f"esperava nn.Linear; recebi {type(fc)}")
        fc_layer = fc

    return target_layer, fc_layer


def cam(
    model: nn.Module,
    x: torch.Tensor,
    class_idx: int | None = None,
    layer: nn.Module | str | None = None,
    fc: nn.Linear | str | None = None,
    apply_relu: bool = True,
) -> tuple[torch.Tensor, int]:
    """Class Activation Map (Zhou et al., 2016).

        L_CAM^c = ReLU( sum_k w_k^c . A^k )

    onde `w_k^c` sao os pesos da camada linear apos o Global Average Pooling.

    Em redes `conv -> GAP -> fc` (como ResNet-18), a derivada do score em relacao
    a ativacao `A_ij^k` e' exatamente `(1 / Z) * w_k^c`. O Grad-CAM (Eqs. 1-2)
    produz:
        alpha_k^c = (1 / Z) * w_k^c
    logo:
        Z * L_GradCAM^c = L_CAM^c

    comprovando analitica e numericamente que o Grad-CAM e' uma generalizacao do
    CAM para arquiteturas arbitrarias (secao 3.1 do artigo).
    """
    if x.ndim != 4 or x.shape[0] != 1:
        raise ValueError(f"esperava (1, C, H, W); recebi {tuple(x.shape)}")

    target_layer, fc_layer = resolve_cam_layers(model, layer, fc)
    scores, activations = _forward_with_activations(model, x, target_layer)
    if class_idx is None:
        class_idx = int(scores.argmax(dim=1).item())

    # w_k^c: pesos da classe c para cada canal k
    w_c = fc_layer.weight[class_idx]
    weighted = (w_c.view(1, -1, 1, 1) * activations).sum(dim=1)[0]
    return (weighted.clamp(min=0) if apply_relu else weighted), class_idx


def _class_weights(
    model: nn.Module,
    x: torch.Tensor,
    class_idx: int | None,
    layer: nn.Module | str | None,
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """`alpha_k` de (1), as ativacoes `A^k` e a classe usada.

    Fatorado para `gradcam` e `counterfactual` compartilharem a conta: os dois
    diferem apenas no sinal de `alpha` aplicado em (2).
    """
    x = x.detach().clone().requires_grad_(True)
    layer = _name_for_layer(model, layer)

    with relu_mode(model, mode):
        target = resolve_layer(model, layer)
        scores, activations = _forward_with_activations(model, x, target)
        if class_idx is None:
            class_idx = int(scores.argmax(dim=1).item())

        (grad,) = torch.autograd.grad(scores[0, class_idx], activations)
        # (1): alpha_k = media espacial do gradiente da ativacao k.
        alpha = grad.mean(dim=(2, 3), keepdim=True)

    return alpha, activations, class_idx


def gradcam(
    model: nn.Module,
    x: torch.Tensor,
    class_idx: int | None = None,
    layer: nn.Module | str | None = None,
    apply_relu: bool = True,
    mode: str = "vanilla",
) -> tuple[torch.Tensor, int]:
    """Mapa Grad-CAM de uma imagem e uma classe.

    Retorna `(mapa HxW, class_idx)`. O mapa sai **cru** (sem normalizar):
    as metricas trabalham com ele assim, e a escala de exibicao fica em
    `viz.scale_for_display`.

    `apply_relu=False` desliga o ReLU de (2) e existe para a ablacao
    "sem ReLU" da Tabela 3 do artigo — nao e' o metodo.

    `mode` escolhe a regra de backward pelo ReLU (`vanilla` | `deconv` |
    `guided`). A regra vale so' durante a chamada: sem isso, o resultado
    dependia de qual saliencia tinha rodado antes no mesmo modelo (rho 0,874
    recem-carregado, 0,992 apos `guided`, 0,296 apos `deconv` na imagem 15278).
    """
    if x.ndim != 4 or x.shape[0] != 1:
        raise ValueError(f"esperava (1, C, H, W); recebi {tuple(x.shape)}")

    alpha, activations, class_idx = _class_weights(
        model, x, class_idx, layer, mode
    )
    # (2): combinacao linear das ativacoes com os pesos alpha.
    weighted = (alpha * activations).sum(dim=1)[0]
    return (weighted.clamp(min=0) if apply_relu else weighted), class_idx


def counterfactual(
    model: nn.Module,
    x: torch.Tensor,
    class_idx: int | None = None,
    layer: nn.Module | str | None = None,
    apply_relu: bool = True,
    mode: str = "vanilla",
) -> tuple[torch.Tensor, int]:
    """Mapa da evidencia **contra** a classe `c` (negacao de `alpha` em (1)).

    O Grad-CAM responde "onde esta' o que sustenta `c`"; negar `alpha` antes do
    ReLU de (2) responde "onde esta' o que empurra a rede para longe de `c`".
    O ReLU continua sendo o mesmo de (2): sem ele a combinacao negativa
    mostraria tambem as regioes neutras, e nao a evidencia contraria.

    Retorna `(mapa HxW, class_idx)`, mapa cru como em `gradcam`.
    """
    if x.ndim != 4 or x.shape[0] != 1:
        raise ValueError(f"esperava (1, C, H, W); recebi {tuple(x.shape)}")

    alpha, activations, class_idx = _class_weights(
        model, x, class_idx, layer, mode
    )
    weighted = (-alpha * activations).sum(dim=1)[0]
    return (weighted.clamp(min=0) if apply_relu else weighted), class_idx
