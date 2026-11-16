"""Testes da Fase 8: pointing game e correlacao com oclusao.

O que estes testes fixam, e que nao e' obvio na leitura do codigo:

1. **O pico e' medido na grade 224, depois do upsample.** Medir no mapa nativo
   (14x14) e depois mapear para pixel daria outro resultado e favoreceria
   mapas grosseiros.
2. **Mapa constante e' caso degenerado, nao acerto.** O `argmax` devolve o
   primeiro pixel; sem a marca `degenerate`, um Grad-CAM zerado pelo ReLU
   viraria "acertou o canto superior esquerdo" em algumas imagens.
3. **A oclusao e o pooling usam as mesmas celulas.** Se cada um arredondasse
   as bordas por conta propria, a correlacao compararia regioes deslocadas.
4. **A correlacao e' de ordenacao**: um mapa proporcional a queda de
   probabilidade da rho 1, um mapa constante da `nan`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gradcam.data import Box
from gradcam.eval import (
    cue_object,
    evaluation_pool,
    grid_cells,
    occlusion_correlation,
    occlusion_map,
    peak_point,
    pointing_hit,
    pool_to_grid,
    scaled_boxes,
    wilson_interval,
)


def square_map(size: int, hot: tuple[int, int], value: float = 1.0) -> torch.Tensor:
    """Mapa `size` x `size` zerado, com o pico em `(x, y)`."""
    heatmap = torch.zeros(size, size)
    x, y = hot
    heatmap[y, x] = value
    return heatmap


def test_peak_point_returns_xy_after_upsample():
    # 2x2 com pico no canto inferior esquerdo: em 224 ele tem que continuar
    # na metade esquerda de baixo, nao na celula (0, 0).
    heatmap = torch.tensor([[0.0, 0.0], [5.0, 0.0]])
    x, y = peak_point(heatmap, size=224)
    assert 0 <= x < 112
    assert 112 <= y < 224


def test_pointing_hit_inside_and_outside():
    box = Box(10, 10, 30, 30)
    assert pointing_hit(square_map(224, (20, 20)), [box])["hit"] is True
    assert pointing_hit(square_map(224, (200, 200)), [box])["hit"] is False


def test_pointing_separates_other_object_from_background():
    target = Box(10, 10, 30, 30)
    other = Box(120, 120, 40, 40)

    on_other = pointing_hit(square_map(224, (130, 130)), [target], [other])
    assert on_other == {"hit": False, "hit_other": True, "degenerate": False}

    on_background = pointing_hit(square_map(224, (200, 60)), [target], [other])
    assert on_background["hit"] is False
    assert on_background["hit_other"] is False


def test_constant_map_is_flagged_as_degenerate():
    result = pointing_hit(torch.zeros(14, 14), [Box(0, 0, 224, 224)])
    assert result["degenerate"] is True


def test_pointing_requires_ground_truth():
    with pytest.raises(ValueError):
        pointing_hit(torch.ones(14, 14), [])


def test_scaled_boxes_maps_to_the_preprocessed_grid():
    meta = {"width": 640, "height": 480}
    boxes = {"cup": [Box(320, 240, 320, 240)]}
    scaled = scaled_boxes(meta, boxes, size=224)["cup"][0]
    assert scaled.x == pytest.approx(112)
    assert scaled.y == pytest.approx(112)
    assert scaled.w == pytest.approx(112)
    assert scaled.h == pytest.approx(112)
    # o canto inferior direito continua dentro da grade
    assert scaled.x + scaled.w == pytest.approx(224)


def test_grid_cells_tile_the_image_without_gaps():
    cells = grid_cells(224, 6)
    assert len(cells) == 36
    assert cells[0] == (0, 0, 37, 37)
    assert cells[-1][2:] == (224, 224)
    # sem buraco nem sobreposicao na horizontal
    first_row = cells[:6]
    for left, right in zip(first_row, first_row[1:]):
        assert left[2] == right[0]
    # a soma das areas cobre a imagem inteira
    assert sum((x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in cells) == 224 * 224


def test_pool_to_grid_averages_each_cell():
    heatmap = torch.arange(16.0).reshape(4, 4)
    pooled = pool_to_grid(heatmap, grid=2)
    assert pooled.shape == (2, 2)
    assert pooled[0, 0].item() == pytest.approx(2.5)   # 0, 1, 4, 5
    assert pooled[1, 1].item() == pytest.approx(12.5)  # 10, 11, 14, 15


def test_pool_to_grid_rejects_non_square_maps():
    with pytest.raises(ValueError):
        pool_to_grid(torch.ones(4, 8), grid=2)


class MeanScore(nn.Module):
    """Score 0 cresce com o brilho total; score 1 cresce ao contrario."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        total = x.sum(dim=(1, 2, 3), keepdim=True)
        return torch.cat([total, -total], dim=1)


def bright_input(size: int = 8) -> torch.Tensor:
    """Zeros com um quadrante claro no canto superior direito."""
    x = torch.zeros(1, 3, size, size)
    x[..., : size // 2, size // 2 :] = 1.0
    return x


def test_occlusion_map_finds_the_informative_patch():
    x = bright_input(8)
    drops = occlusion_map(MeanScore(), x, class_idx=0, grid=2)

    # o quadrante claro e' a celula (linha 0, coluna 1)
    assert drops.shape == (2, 2)
    assert drops.argmax().item() == 1
    assert drops[0, 1] > 0.4
    for index in (0, 2, 3):
        assert drops.flatten()[index].item() == pytest.approx(0.0, abs=1e-6)


def test_occlusion_map_does_not_mutate_the_input():
    x = bright_input(8)
    before = x.clone()
    occlusion_map(MeanScore(), x, class_idx=0, grid=2)
    assert torch.equal(x, before)


def test_occlusion_correlation_matches_a_proportional_map():
    x = bright_input(8)

    def brightness(model, image, class_idx):
        return image.detach().sum(dim=1)[0]

    rho = occlusion_correlation(
        MeanScore(), x, class_idx=0, map_fns={"brilho": brightness}, grid=2
    )
    assert rho["brilho"] == pytest.approx(1.0)


def test_occlusion_correlation_is_nan_for_a_constant_map():
    def constante(model, image, class_idx):
        return torch.ones(8, 8)

    rho = occlusion_correlation(
        MeanScore(), bright_input(8), class_idx=0,
        map_fns={"constante": constante}, grid=2,
    )
    assert rho["constante"] != rho["constante"]  # nan


def test_wilson_interval_stays_inside_the_unit_interval():
    low, high = wilson_interval(0, 10)
    assert low == 0.0
    assert 0.0 < high < 0.5

    low, high = wilson_interval(10, 10)
    assert high == 1.0
    assert 0.5 < low < 1.0

    low, high = wilson_interval(5, 10)
    assert low == pytest.approx(0.2366, abs=1e-3)
    assert high == pytest.approx(0.7634, abs=1e-3)


class FakeCoco:
    """Indice minimo com a mesma superficie que `CocoVal` usa na amostragem."""

    def __init__(self, boxes, categories, labels_verbatim):
        self.boxes = boxes
        self.image_ids = sorted(boxes)
        self._categories = categories
        self._verbatim = labels_verbatim

    def verbatim_categories(self, labels):
        return self._verbatim


def fake_coco() -> FakeCoco:
    boxes = {
        1: {"cup": [Box(0, 0, 100, 100)]},
        2: {"cup": [Box(0, 0, 50, 50), Box(60, 60, 40, 40)], "dog": [Box(0, 0, 80, 80)]},
        3: {"dog": [Box(0, 0, 90, 90)]},          # nao e' categoria verbatim
        4: {"cup": [Box(0, 0, 10, 10)]},          # objeto pequeno demais
        5: {"vase": [Box(0, 0, 120, 120)], "cup": [Box(0, 0, 200, 200)]},
        6: {"vase": [Box(0, 0, 150, 150)]},
    }
    labels = ["cup", "vase", "dog"]
    return FakeCoco(boxes, {1: "cup", 2: "dog", 3: "dog", 4: "cup", 5: "cup"}, ["cup", "vase"])


def test_cue_object_picks_the_largest_verbatim_object():
    coco, labels = fake_coco(), ["cup", "vase", "dog"]
    assert cue_object(coco, labels, 1, min_side=60)[0] == "cup"

    # em 2, a maior box e' 'dog', que nao e' verbatim: sobra 'cup'
    name, cells = cue_object(coco, labels, 2, min_side=40)
    assert name == "cup"
    assert len(cells) == 2  # todas as instancias, nao so' a maior

    # em 5, 'cup' (200x200) ganha de 'vase' (120x120)
    assert cue_object(coco, labels, 5) == ("cup", [Box(0, 0, 200, 200)])

    assert cue_object(coco, labels, 3) is None
    assert cue_object(coco, labels, 4, min_side=60) is None


def test_evaluation_pool_caps_each_category_and_is_reproducible():
    coco, labels = fake_coco(), ["cup", "vase", "dog"]

    # per_category=1 pega uma imagem por categoria, nao a categoria dominante
    pool = evaluation_pool(coco, labels, per_category=1, min_side=60, seed=0)
    assert len(pool) == 2
    assert pool == evaluation_pool(coco, labels, per_category=1, min_side=60, seed=0)

    # n menor que o pool devolve subconjunto do pool
    pool = evaluation_pool(coco, labels, per_category=5, min_side=60, seed=0, n=1)
    assert len(pool) == 1
    assert pool[0] in (1, 5, 6)
