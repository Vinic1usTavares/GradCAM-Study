"""Fase 8: avaliacao em lote — pointing game e correlacao com oclusao.

Duas perguntas, duas metricas:

1. **O mapa aponta para o objeto certo?** Pointing game (Zhang et al., 2018,
   usado na secao 4.3 do paper): o cue e' o rotulo de ground truth, nao a
   predicao; o hit e' o pixel de maximo do mapa cair dentro de uma instancia
   anotada daquela categoria. E' comparavel entre metodos de resolucoes
   diferentes porque todos passam pelo mesmo upsample antes do `argmax`.
2. **O mapa e' fiel ao modelo?** Correlacao com oclusao: oclui-se uma grade de
   patches, mede-se a queda de probabilidade da classe e compara-se a
   *ordenacao* dos patches com a ordenacao do mapa. E' a referencia de
   fidelidade local: um mapa que ordena os patches como a oclusao diz onde o
   modelo realmente olha.

O paper descreve o pointing game em detalhe, mas a oclusao em uma frase ("we
measure the difference in CNN scores when patches of the input image are
masked"), sem tamanho de patch nem grade. O protocolo operacionalizado aqui:

- mapa -> grade 224x224 por interpolacao bilinear; pico por `argmax`
  (empate resolve pelo primeiro pixel em ordem row-major — e' o que expoe o
  caso degenerado, mapa constante, que vira `degenerate=True`);
- oclusao em grade `grid` x `grid` (padrao 6x6 = 36 patches, a estimativa de
  custo de `plans/general_plan.md`, secao 0); o patch recebe **zero no espaco
  normalizado**, que e' a cor media do ImageNet — o valor "sem informacao" que
  `preprocess` define. Gray 128 seria outro valor, arbitrario;
- a correlacao compara 36 celulas: o mapa do metodo e' **reduzido a media por
  celula** (`pool_to_grid`), nao o contrario. Upsample da oclusao para 224
  inventaria estrutura de 224 a partir de 36 numeros e inflaria a correlacao
  por causa dos empates.

A queda de probabilidade pode ser negativa (ocluir um patch que *ajuda* a
classe). Isso entra na correlacao como esta: e' informacao, nao ruido.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn

from .data import Box, CocoVal
from .metrics import MapFn, spearman
from .viz import upsample


def scaled_boxes(
    meta: Mapping[str, float],
    boxes: Mapping[str, Sequence[Box]],
    size: int = 224,
) -> dict[str, list[Box]]:
    """Boxes do COCO na grade `size` x `size` do `preprocess`.

    `preprocess` redimensiona direto para 224x224, sem crop, entao a box escala
    por `size/width` e `size/height` e continua valida. Mapear para a grade do
    mapa (14x14, 7x7) seria pior: todas as metricas comparam pixels da entrada.
    """
    sx, sy = size / meta["width"], size / meta["height"]
    return {
        name: [Box(b.x * sx, b.y * sy, b.w * sx, b.h * sy) for b in cells]
        for name, cells in boxes.items()
    }


def cue_object(
    coco: CocoVal,
    labels: Sequence[str],
    image_id: int,
    min_side: float = 60,
) -> tuple[str, list[Box]] | None:
    """Categoria COCO verbatim de maior area na imagem, e todas as suas boxes.

    Cue do pointing game e' o rotulo de ground truth (secao 4.3 do paper), nao
    a predicao. Restringir ao conjunto verbatim evita o chute de mapear
    `person` ou `dog` para um synset: para as 18 categorias que *sao* classe
    ImageNet, o conjunto de cues e' unitario (ver `CocoVal.verbatim_categories`).

    Devolve `None` quando a imagem nao tem objeto verbatim grande o bastante.
    """
    allowed = set(coco.verbatim_categories(labels))
    best: tuple[str, Box] | None = None
    for name, cells in coco.boxes[image_id].items():
        if name not in allowed:
            continue
        box = max(cells, key=lambda candidate: candidate.area())
        if min(box.w, box.h) < min_side:
            continue
        if best is None or box.area() > best[1].area():
            best = (name, box)
    if best is None:
        return None
    return best[0], list(coco.boxes[image_id][best[0]])


def evaluation_pool(
    coco: CocoVal,
    labels: Sequence[str],
    per_category: int = 20,
    min_side: float = 60,
    seed: int = 0,
    n: int | None = None,
) -> list[int]:
    """Imagens do val2017 com um objeto verbatim grande, estratificadas.

    Uma imagem pode conter varias categorias verbatim; ela entra pelo **maior**
    objeto verbatim (o mesmo criterio que `cue_object` usa), senao a mesma
    imagem seria avaliada varias vezes com cues diferentes e as contagens
    deixariam de ser por imagem.

    `per_category` limita a categoria dominante: sem isso, `dining table` (323
    imagens do pool) responderia por um quarto da amostra e a media seria a
    media de mesa de jantar.
    """
    verbatim = set(coco.verbatim_categories(labels))
    by_category: dict[str, list[int]] = {}
    for image_id in coco.image_ids:
        best: tuple[str, Box] | None = None
        for name, cells in coco.boxes[image_id].items():
            if name not in verbatim:
                continue
            box = max(cells, key=lambda candidate: candidate.area())
            if min(box.w, box.h) < min_side:
                continue
            if best is None or box.area() > best[1].area():
                best = (name, box)
        if best is not None:
            by_category.setdefault(best[0], []).append(image_id)

    rng = random.Random(seed)
    selected: list[int] = []
    for name in sorted(by_category):
        cells = by_category[name]
        selected.extend(rng.sample(cells, min(per_category, len(cells))))
    selected.sort()
    if n is not None and len(selected) > n:
        selected = sorted(rng.sample(selected, n))
    return selected


def peak_point(heatmap: torch.Tensor, size: int = 224) -> tuple[int, int]:
    """Pixel de maximo do mapa na grade `size` x `size`, em `(x, y)`.

    O upsample vem antes do `argmax` para que um CAM 14x14 e uma saliencia
    224x224 sejam medidos no mesmo espaco. Fazer o contrario (mapear o pico do
    CAM para o centro da celula) mudaria o resultado em ~8 px e favoreceria
    mapas grosseiros, cujo pico e' uma regiao grande.

    Devolve `(x, y)` para casar com `Box.contains(px, py)`.
    """
    if heatmap.ndim != 2:
        raise ValueError(f"esperava mapa HxW; recebi {tuple(heatmap.shape)}")
    values = upsample(heatmap.detach().float(), (size, size))
    index = int(values.flatten().argmax())
    return index % size, index // size


def pointing_hit(
    heatmap: torch.Tensor,
    target_boxes: Sequence[Box],
    other_boxes: Sequence[Box] = (),
    size: int = 224,
) -> dict[str, bool]:
    """O pico do mapa caiu no objeto do cue? E, se nao, caiu em outro objeto?

    Separar `hit_other` de "caiu em qualquer lugar" e' o que o paper chama de
    precisao: apontar para outro objeto anotado e' um erro diferente de apontar
    para o fundo, e a diferenca diz se o mapa esta deslocado ou apenas ruidoso.
    """
    if not target_boxes:
        raise ValueError("pointing game sem box de ground truth nao mede nada")

    x, y = peak_point(heatmap, size)
    hit = any(box.contains(x, y) for box in target_boxes)
    return {
        "hit": hit,
        "hit_other": (not hit) and any(box.contains(x, y) for box in other_boxes),
        "degenerate": bool(heatmap.max() == heatmap.min()),
    }


def grid_cells(size: int, grid: int) -> list[tuple[int, int, int, int]]:
    """Bordas de uma grade `grid` x `grid` sobre uma imagem `size` x `size`.

    Uma unica funcao define as bordas, entao a oclusao e o pooling dos mapas
    caem exatamente nas mesmas celulas. `linspace(...).round()` distribui o
    resto (224/6 = 37,3) em celulas de 37 e 38 px, sem sobra nem sobreposicao.
    """
    if grid < 1:
        raise ValueError(f"grid invalido: {grid}")
    edges = torch.linspace(0, size, grid + 1).round().long().tolist()
    return [
        (edges[i], edges[j], edges[i + 1], edges[j + 1])
        for j in range(grid)
        for i in range(grid)
    ]


def pool_to_grid(heatmap: torch.Tensor, grid: int) -> torch.Tensor:
    """Media do mapa dentro de cada celula da grade `grid` x `grid`.

    A media (e nao o maximo) porque a pergunta e' "quanta atribuicao esta nesta
    regiao", a mesma que a oclusao responde por patch. Maximo daria 1 para
    qualquer celula com um pixel forte e empataria tudo.
    """
    if heatmap.ndim != 2:
        raise ValueError(f"esperava mapa HxW; recebi {tuple(heatmap.shape)}")
    values = heatmap.detach().float()
    height, width = values.shape
    if height != width:
        raise ValueError(f"grade nao quadrada: {height}x{width}")
    pooled = torch.empty(grid, grid, dtype=values.dtype)
    for index, (x0, y0, x1, y1) in enumerate(grid_cells(height, grid)):
        pooled[index // grid, index % grid] = values[y0:y1, x0:x1].mean()
    return pooled


def occlusion_map(
    model: nn.Module,
    x: torch.Tensor,
    class_idx: int,
    grid: int = 6,
    fill: float = 0.0,
) -> torch.Tensor:
    """Queda de probabilidade da classe ao ocluir cada patch da grade.

        drop_ij = p(y=c | x) - p(y=c | x com o patch ij apagado)

    Positivo onde o patch sustenta a classe; negativo onde ele a atrapalha.
    `fill=0` no espaco normalizado e' a cor media do ImageNet por canal (ver o
    modulo): o valor "sem informacao" que `preprocess` define. Gray 128 seria
    outro valor, arbitrario.

    Custo: `grid**2` forwards na mesma imagem (36 no padrao) — e' o item caro
    da fase, e a razao de a oclusao rodar em menos imagens que o pointing game.
    """
    if x.ndim != 4 or x.shape[0] != 1:
        raise ValueError(f"esperava (1, C, H, W); recebi {tuple(x.shape)}")
    height, width = x.shape[-2:]
    if height != width:
        raise ValueError(f"entrada nao quadrada: {height}x{width}")

    with torch.no_grad():
        base = float(model(x)[0].softmax(dim=0)[class_idx])
        drops = torch.empty(grid, grid, dtype=torch.float32)
        for index, (x0, y0, x1, y1) in enumerate(grid_cells(height, grid)):
            occluded = x.clone()
            occluded[..., y0:y1, x0:x1] = fill
            value = float(model(occluded)[0].softmax(dim=0)[class_idx])
            drops[index // grid, index % grid] = base - value
    return drops


def occlusion_correlation(
    model: nn.Module,
    x: torch.Tensor,
    class_idx: int,
    map_fns: Mapping[str, MapFn],
    grid: int = 6,
) -> dict[str, float]:
    """Spearman entre cada mapa (reduzido a grade) e a queda por oclusao.

    Quem agrega deve descartar `nan` (mapa constante) e reportar quantos
    valores validos entraram — `metrics.summarize` ja faz isso.
    """
    reference = occlusion_map(model, x, class_idx, grid=grid)
    return {
        name: spearman(reference, pool_to_grid(fn(model, x, class_idx), grid))
        for name, fn in map_fns.items()
    }


def wilson_interval(hits: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Intervalo de confianca de Wilson para uma proporcao.

    Wilson, e nao a aproximacao normal, porque perto de 0 ou 1 (metodo que
    acerta quase sempre) a normal devolve limites fora de [0, 1] e some com a
    incerteza que interessa: a diferenca de 2 pontos entre metodos.
    """
    if total <= 0:
        raise ValueError("intervalo precisa de total > 0")
    if not 0 <= hits <= total:
        raise ValueError(f"hits={hits} fora de [0, {total}]")
    p = hits / total
    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    margin = z * ((p * (1 - p) / total + z**2 / (4 * total**2)) ** 0.5) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)
