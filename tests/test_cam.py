"""Testes da Fase 5: CAM e equivalencia analitica com Grad-CAM no ResNet18."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

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


def test_cam_accepts_explicit_layer_module():
    """`cam` aceita o modulo da camada; explicito = default no ResNet18."""
    resnet = torch.hub.load("pytorch/vision", "resnet18", weights=None)
    resnet.eval()
    x = torch.randn(1, 3, 224, 224)
    default_map, _ = cam(resnet, x, class_idx=3)
    explicit_map, _ = cam(resnet, x, class_idx=3, layer=resnet.layer4)
    assert explicit_map.shape == default_map.shape == (7, 7)
    assert torch.equal(explicit_map, default_map)


def test_cam_vs_gradcam_same_layer_from_resolve_cam_layers():
    """A bateria deve usar a camada do CAM nos dois metodos.

    `scripts/05_cam_vs_gradcam.py` resolve a camada uma vez e passa a mesma para
    `cam` e `gradcam`. Sem isso, o Grad-CAM cai na ultima `Conv2d` (pre-BatchNorm)
    e a igualdade de §3.1 some: medido na imagem 15278, rho 1,0000 na mesma
    camada contra 0,987 em camadas diferentes.
    """
    resnet = torch.hub.load("pytorch/vision", "resnet18", weights=None)
    resnet.eval()
    x = torch.randn(1, 3, 224, 224)
    class_idx = 42

    layer, _ = resolve_cam_layers(resnet)
    cam_map, _ = cam(resnet, x, class_idx=class_idx, layer=layer)
    gc_map, _ = gradcam(resnet, x, class_idx=class_idx, layer=layer)

    Z = cam_map.numel()
    diff = (Z * gc_map - cam_map).abs().max().item()
    assert diff < 1e-5
