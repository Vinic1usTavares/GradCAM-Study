"""Testes da Fase 5: CAM e equivalencia analitica com Grad-CAM no ResNet18."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gradcam.core import cam, gradcam, resolve_cam_layers


def test_resolve_cam_layers_finds_layer4_and_fc():
    resnet = torch.hub.load("pytorch/vision", "resnet18", weights=None)
    target, fc = resolve_cam_layers(resnet)
    assert target is resnet.layer4
    assert fc is resnet.fc


def test_cam_shape_and_range():
    resnet = torch.hub.load("pytorch/vision", "resnet18", weights=None)
    resnet.eval()
    x = torch.randn(1, 3, 224, 224)
    heatmap, class_idx = cam(resnet, x, class_idx=3)
    assert heatmap.shape == (7, 7)
    assert heatmap.min() >= 0
    assert class_idx == 3


def test_cam_vs_gradcam_equivalence():
    """Valida a equacao Z * Grad-CAM = CAM da secao 3.1 do paper."""
    resnet = torch.hub.load("pytorch/vision", "resnet18", weights=None)
    resnet.eval()
    x = torch.randn(1, 3, 224, 224)
    class_idx = 42

    cam_map, _ = cam(resnet, x, class_idx=class_idx)
    gc_map, _ = gradcam(resnet, x, class_idx=class_idx, layer="layer4")

    # Z = H * W das ativacoes (7 x 7 = 49 no layer4 do ResNet18)
    Z = cam_map.numel()
    diff = (Z * gc_map - cam_map).abs().max().item()

    # Diferenca numerica de ponto flutuante deve ser minima (<= 1e-5)
    assert diff < 1e-5
