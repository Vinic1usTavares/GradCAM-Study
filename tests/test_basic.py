"""Testes das utilidades basicas (etapa 0)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gradcam.data import ANNOTATIONS, CocoVal
from gradcam.models import denormalize, preprocess
from gradcam.viz import figure_grid, gray_rgb, normalize, overlay, to_rgb, upsample


def test_preprocess_shape_and_range():
    x = preprocess(Image.new("RGB", (320, 240), color=(120, 80, 40)))
    assert x.shape == (1, 3, 224, 224)
    assert x.dtype == torch.float32


def test_denormalize_roundtrip():
    x = torch.rand(1, 3, 8, 8)
    y = denormalize((x - torch.tensor([0.5])) / torch.tensor([0.2]))
    # So verifica que o resultado volta para uma faixa plausivel.
    assert y.min() >= 0 and y.max() <= 1


def test_normalize_handles_empty_map():
    zeros = torch.zeros(7, 7)
    assert torch.equal(normalize(zeros), zeros)


def test_normalize_scales_peak_to_one():
    heatmap = torch.rand(7, 7) * 0.1
    assert torch.isclose(normalize(heatmap).max(), torch.tensor(1.0))


def test_upsample_preserves_layout():
    heatmap = torch.zeros(7, 7)
    heatmap[0, 0] = 1.0
    up = upsample(heatmap, (224, 224))
    assert up.shape == (224, 224)
    assert up[0, 0] > up[-1, -1]


def test_to_rgb_and_overlay_shapes():
    heatmap = torch.rand(14, 14)
    rgb = to_rgb(heatmap)
    assert rgb.shape == (14, 14, 3) and rgb.dtype.name == "uint8"
    image = torch.randint(0, 255, (28, 28, 3), dtype=torch.uint8).numpy()
    blended = overlay(image, heatmap)
    assert blended.shape == (28, 28, 3)


def test_gray_rgb_peaks_white_and_keeps_shape():
    heatmap = torch.zeros(9, 9)
    heatmap[4, 4] = 2.0
    rgb = gray_rgb(heatmap)
    assert rgb.shape == (9, 9, 3) and rgb.dtype.name == "uint8"
    assert rgb[4, 4].tolist() == [255, 255, 255]
    assert rgb[0, 0].tolist() == [0, 0, 0]


def test_figure_grid_one_axis_per_cell():
    import matplotlib.pyplot as plt

    cell = torch.zeros(4, 4, 3, dtype=torch.uint8).numpy()
    fig = figure_grid(
        [[cell, cell, cell], [cell, cell, cell]],
        col_titles=("a", "b", "c"),
        row_titles=("linha 1", "linha 2"),
        suptitle="teste",
    )
    # 6 celulas + 1 coluna de rotulos por linha.
    assert len(fig.axes) == 8
    assert fig.axes[1].get_title() == "a"
    assert fig.axes[0].texts[0].get_text() == "linha 1"
    assert fig.axes[4].get_title() == ""  # titulo so' na primeira linha
    plt.close(fig)


def test_figure_grid_rejects_wrong_number_of_titles():
    cell = torch.zeros(4, 4, 3, dtype=torch.uint8).numpy()
    with pytest.raises(ValueError):
        figure_grid([[cell]], row_titles=("a", "b"))


@pytest.mark.skipif(not ANNOTATIONS.exists(), reason="COCO nao baixado")
def test_coco_index():
    coco = CocoVal()
    assert len(coco) == 5000
    ids = coco.subset(3, seed=0)
    assert len(ids) == 3
    for image_id in ids:
        assert len(coco.object_names(image_id)) == 1
        assert coco.image_path(image_id).exists()
