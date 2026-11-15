"""Carregamento de modelos pré-treinados e utilidades de imagem.

Decisões relevantes (ver plans/general_plan.md, secao 2.1):
- `torchvision` usa `nn.ReLU(inplace=True)`, que quebra hooks de backward e
  invalida versoes de tensor. Desativamos o inplace uma unica vez aqui.
- Normalizacao ImageNet e aplicada antes do forward; a desnormalizacao existe
  apenas para exibicao.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
WEIGHTS_DIR = DATA_DIR / "weights"
COCO_DIR = DATA_DIR / "coco"

WEIGHT_URLS = {
    "vgg16": "https://download.pytorch.org/models/vgg16-397923af.pth",
    "alexnet": "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth",
    "resnet18": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
}


def disable_inplace_relu(model: nn.Module) -> nn.Module:
    """Troca `nn.ReLU(inplace=True)` por `inplace=False`, no lugar."""
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            module.inplace = False
    return model


def load_model(name: str, device: str | torch.device = "cpu") -> nn.Module:
    """Carrega um modelo pre-treinado no ImageNet, em modo eval.

    Os pesos vao para `data/weights/` em vez do cache padrao do torch, para o
    cache ficar dentro do repositorio (que ja e gitignored).
    """
    device = torch.device(device)
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    builder = {
        "vgg16": models.vgg16,
        "vgg16_bn": models.vgg16_bn,
        "alexnet": models.alexnet,
        "resnet18": models.resnet18,
        "resnet50": models.resnet50,
    }[name]

    if name in WEIGHT_URLS:
        local = WEIGHTS_DIR / WEIGHT_URLS[name].rsplit("/", 1)[-1]
        if not local.exists():
            raise FileNotFoundError(
                f"Pesos ausentes: {local}\n"
                f"Baixe com: curl.exe -L -o \"{local}\" {WEIGHT_URLS[name]}"
            )
        model = builder(weights=None)
        model.load_state_dict(torch.load(local, map_location="cpu", weights_only=True))
    else:
        model = builder(weights="DEFAULT")

    model = disable_inplace_relu(model)
    model.eval().to(device)
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def preprocess(image: Image.Image, size: int = 224) -> torch.Tensor:
    """Imagem PIL RGB -> tensor normalizado (1, 3, size, size).

    Redimensiona direto para `size` x `size`, sem preservar o aspecto. E' de
    proposito: a transformacao e' um afim alinhado aos eixos, entao as bounding
    boxes do COCO continuam validas sob a mesma escala (`sx = size/W`,
    `sy = size/H`) e nenhuma box e' cortada. Isso importa para o pointing game.

    O ganho de acuracia de `Resize(256) + CenterCrop(224)` medido neste
    ambiente foi nulo (ver README, etapa 0), e o crop descartaria parte das
    boxes.
    """
    pipeline = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    return pipeline(image.convert("RGB")).unsqueeze(0)


def denormalize(x: torch.Tensor) -> torch.Tensor:
    """Tensor normalizado -> [0, 1], para exibicao."""
    mean = torch.tensor(MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=x.device).view(1, 3, 1, 1)
    return (x * std + mean).clamp(0, 1)


def imagenet_labels() -> list[str]:
    """Rotulos do ImageNet-1k, na ordem dos indices do torchvision."""
    from torchvision.models import ResNet18_Weights

    return ResNet18_Weights.DEFAULT.meta["categories"]


@torch.no_grad()
def predict(
    model: nn.Module, x: torch.Tensor, topk: int = 5
) -> tuple[torch.Tensor, torch.Tensor]:
    """Retorna (indices, probabilidades) dos `topk` scores mais altos."""
    scores = model(x)
    probs = scores.softmax(dim=1)
    values, indices = probs.topk(topk, dim=1)
    return indices[0], values[0]
