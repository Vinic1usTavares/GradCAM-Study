"""Fase 10 - sanity checks de Adebayo et al. (2018).

A pergunta: um mapa de atribuicao depende dos **pesos** da rede, ou e' so' um
detector de bordas que sairia igual com qualquer modelo? O teste operacional:

1. calcula o mapa no modelo treinado;
2. reamostra os pesos de uma camada de cada vez, **de cima para baixo** (do
   `classifier` para a primeira `conv`), de forma cumulativa;
3. recalcula o mapa com a mesma imagem e a **mesma classe** e mede o Spearman
   contra o mapa original.

Se o mapa for fiel ao modelo, a correlacao cai conforme as camadas sao
randomizadas. Se ficar alta ate' o fim, o mapa nao depende dos pesos - que e'
exatamente o resultado de Adebayo et al. para Guided Backprop e Guided Grad-CAM
(as bordas de entrada sobrevivem).

Duas notas de escopo:

- A "data randomization" do artigo exige **retreinar** o modelo com rotulos
  embaralhados; nao ha treino neste repositorio, entao so' a randomizacao de
  parametros e' reproduzida.
- `cascade` **destroi** o modelo recebido (randomiza no lugar, para nao
  duplicar 528 MB de VGG-16 por passo). O chamador calcula o mapa original antes.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from .metrics import MapFn, spearman

DEFAULT_CUTOFFS = (0, 1, 2, 4, 6, 9, 12, 16)


def parameter_modules(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """Modulos com parametros proprios, na ordem do fluxo (entrada -> saida).

    `named_modules` ja' percorre nessa ordem; quem randomiza de cima para baixo
    usa `randomization_order`, que inverte.
    """
    return [
        (name, module)
        for name, module in model.named_modules()
        if any(True for _ in module.parameters(recurse=False))
    ]


def randomization_order(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """Ordem do cascade: da saida para a entrada (`classifier` -> `features.0`)."""
    return list(reversed(parameter_modules(model)))


def reinitialize_(module: nn.Module, generator: torch.Generator | None = None) -> None:
    """Reamostra os pesos de um modulo, no lugar, com a semente do `generator`.

    Conv/Linear: Kaiming normal (pesos) e zero (bias), como na inicializacao
    padrao do torchvision. BatchNorm e' caso a' parte: zerar o `weight` (que e'
    1-D) apagaria a camada inteira, entao volta-se a escala 1 e desvio 0.

    O `generator` e' o que torna o cascade reproduzivel: a mesma semente da' a
    mesma sequencia de pesos, passo a passo.
    """
    if isinstance(module, nn.modules.batchnorm._BatchNorm):
        module.running_mean.zero_()
        module.running_var.fill_(1.0)
        if module.num_batches_tracked is not None:
            module.num_batches_tracked.zero_()
        for name, param in module.named_parameters(recurse=False):
            if name == "weight":
                nn.init.ones_(param)
            else:
                nn.init.zeros_(param)
        return

    for _, param in module.named_parameters(recurse=False):
        if not param.is_floating_point():
            continue
        if param.ndim >= 2:
            nn.init.kaiming_normal_(param, generator=generator)
        else:
            nn.init.zeros_(param)


def cascade(
    model: nn.Module,
    order: Sequence[tuple[str, nn.Module]],
    k: int,
    generator: torch.Generator,
    already: int = 0,
) -> list[str]:
    """Randomiza, cumulativamente, os modulos `order[already:k]`.

    Devolve os nomes randomizados agora. Manter `already` explicito e' o que
    garante o cascade: cada passo preserva as randomizacoes dos anteriores, em
    vez de sortear um modelo novo.
    """
    if k < already:
        raise ValueError(f"k={k} menor que o ja' randomizado ({already})")
    names = []
    for index in range(already, k):
        name, module = order[index]
        reinitialize_(module, generator)
        names.append(name)
    return names


def sanity_curve(
    model: nn.Module,
    x: torch.Tensor,
    class_idx: int,
    map_fn: MapFn,
    cutoffs: Sequence[int] = DEFAULT_CUTOFFS,
    seed: int = 0,
) -> dict:
    """Curva do cascade: Spearman(original, randomizado) por numero de camadas.

    `class_idx` e' fixo em todos os passos - e' o que separa "o mapa mudou
    porque o modelo mudou" de "o mapa mudou porque a predicao mudou". O
    `prob_class` por passo mostra o quanto o modelo ja' degradou.

    Cada passo reporta `spearman` (com sinal) e `abs_spearman`. O **modulo** e'
    a leitura de Adebayo et al.: com pesos randomizados o gradiente pode
    inverter, e a pergunta nao e' o sentido e sim se a estrutura do mapa
    sobreviveu.

    Destrutivo: o modelo sai randomizado ate' o maior `cutoff`. Rode em um
    modelo cujo mapa original nao seja mais necessario (o script calcula as
    figuras antes).
    """
    order = randomization_order(model)
    steps_wanted = sorted({int(c) for c in cutoffs})
    if steps_wanted[0] != 0:
        steps_wanted.insert(0, 0)
    if steps_wanted[-1] > len(order):
        raise ValueError(
            f"cutoff {steps_wanted[-1]} maior que os {len(order)} modulos "
            "com parametros do modelo"
        )

    with torch.no_grad():
        probs_original = torch.softmax(model(x)[0], dim=0)

    base = map_fn(model, x, class_idx).detach()
    generator = torch.Generator().manual_seed(seed)

    steps: list[dict] = []
    done = 0
    for k in steps_wanted:
        added = cascade(model, order, k, generator, already=done)
        done = k
        current = map_fn(model, x, class_idx).detach()
        with torch.no_grad():
            probs = torch.softmax(model(x)[0], dim=0)
        rho = spearman(base, current)
        steps.append(
            {
                "k": k,
                "added": added,
                # `spearman` tem sinal porque um modelo randomizado pode inverter
                # a direcao do gradiente. Adebayo et al. plotam o **modulo**: a
                # pergunta e' "a estrutura do mapa sobreviveu", nao o sentido.
                "spearman": rho,
                "abs_spearman": abs(rho),
                "predicted_idx": int(probs.argmax()),
                "prob_class": float(probs[class_idx]),
            }
        )

    return {
        "class_idx": int(class_idx),
        "prob_original": float(probs_original[class_idx]),
        "n_parameters_modules": len(order),
        "seed": seed,
        "order": [name for name, _ in order],
        "steps": steps,
    }
