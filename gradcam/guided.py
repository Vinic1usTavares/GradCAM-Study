"""Fase 6: Guided Grad-CAM (Selvaraju et al., 2016, secao 3).

    L^c_guided = Upsample(L^c_GradCAM) . GuidedBackprop(x)

O Grad-CAM responde *onde* (14x14 no VGG-16), o Guided Backprop responde *o que*
(224x224, borda e textura). O produto devolve as duas coisas: a mascara
grosseira seleciona a regiao que sustenta a classe `c`, e o gradiente guiado
desenha o detalhe dentro dela.

Duas decisoes que mudam o mapa e nao sao obvias:

1. **O termo Grad-CAM entra com a regra `vanilla`** (gradiente comum), nao com
   `guided`. Calcular o CAM dentro de `relu_mode(model, "guided")` produz outro
   mapa, e a diferenca e' silenciosa — o resultado continua plausivel. E' o
   mesmo tipo de armadilha que `relu_mode` resolveu na Fase 4.
2. **O termo de detalhe e' reduzido a `max |d y^c / d x|` por pixel**
   (`saliency.saliency_map`), a mesma reducao das Fases 1-3. Assim o mapa de
   saida e' nao-negativo e `HxW`, que e' o contrato `metrics.MapFn` da Fase 8.
   As figuras do artigo mostram magnitude (detalhe claro em fundo escuro), nao o
   sinal do gradiente.

Nao normalizamos o CAM antes do produto: qualquer fator positivo e' irrelevante
para a ordenacao dos pixels, e a escala de exibicao (`viz.scale_for_display`)
fica na figura, nunca no metodo.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .core import gradcam
from .saliency import guided_backprop
from .viz import upsample


def compose_guided_gradcam(
    cam: torch.Tensor,
    detail: torch.Tensor,
    size: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Upsample bilinear do CAM e produto elementar com o mapa de detalhe.

    Funcao pura: nao toca no modelo. Existe para quem ja' calculou as duas
    pecas (o script da fase precisa das duas para desenhar as colunas
    intermediarias) nao pagar dois forwards e dois backwards por mapa.

    `cam` e' `HxW` (ex.: 14x14 no VGG-16); `detail` e' `HxW` na resolucao da
    entrada; `size` existe para deixar explicito o alvo do upsample — por
    padrao, a forma de `detail`.
    """
    if cam.ndim != 2 or detail.ndim != 2:
        raise ValueError(
            f"esperava dois mapas HxW; recebi {tuple(cam.shape)} e {tuple(detail.shape)}"
        )

    target = tuple(detail.shape) if size is None else tuple(size)
    if tuple(detail.shape) != target:
        raise ValueError(
            f"detail tem forma {tuple(detail.shape)}, esperava {target}: o produto "
            f"elementar exige a mesma resolucao"
        )
    return upsample(cam, target) * detail


def guided_gradcam(
    model: nn.Module,
    x: torch.Tensor,
    class_idx: int | None = None,
    layer: nn.Module | str | None = None,
) -> tuple[torch.Tensor, int]:
    """Guided Grad-CAM de uma imagem e uma classe, em uma chamada.

    Retorna `(mapa HxW nao-negativo na resolucao da entrada, class_idx)`.
    Custo: um forward + um backward do Grad-CAM mais um do Guided Backprop
    (~2,9 s no VGG-16 em CPU).

    `class_idx=None` usa o `argmax` do score, e o mesmo indice alimenta os dois
    termos — o mapa e o rotulo nunca saem de classes diferentes.
    """
    heatmap, class_idx = gradcam(
        model, x, class_idx, layer=layer, mode="vanilla"
    )
    detail, _ = guided_backprop(model, x, class_idx)
    return compose_guided_gradcam(heatmap, detail, tuple(x.shape[-2:])), class_idx
