"""Fase 7 - Contrafactual: a evidencia *contra* a classe.

O Grad-CAM responde "onde esta' o que sustenta `c`" (Eqs. 1-2). Negar
`alpha_k^c` em (1) antes do ReLU de (2) responde a pergunta inversa: "onde
esta' o que empurra a rede para longe de `c`".

    alpha_k^c =  (1/Z) sum_ij  d y^c / d A^k_ij      (Grad-CAM)
    alpha_k^c = -(1/Z) sum_ij  d y^c / d A^k_ij      (contrafactual)
    L^c       = ReLU( sum_k alpha_k^c . A^k )        (2, igual)

O criterio de aceite e' qualitativo e mensuravel: com dois objetos na mesma
foto (o caso `95786`, copo e vaso), o mapa a favor acende o objeto da classe e
o contrafactual desloca para o outro. O deslocamento do pico e' reportado em
celulas do mapa e em pixels da entrada.

Saida:
    results/figures/07_counterfactual_<modelo>[_<tag>].png
    results/metrics/counterfactual.json

Uso:
    .\\.venv\\Scripts\\python.exe scripts\\07_counterfactual.py
    .\\.venv\\Scripts\\python.exe scripts\\07_counterfactual.py --images 95786 --classes cup,vase
    .\\.venv\\Scripts\\python.exe scripts\\07_counterfactual.py --classes broccoli

Sem `--classes`, usa o top-1 do modelo como classe.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from PIL import Image

from gradcam.core import VGG16_LAYERS, counterfactual, gradcam, resolve_layer
from gradcam.data import CocoVal
from gradcam.metrics import spearman
from gradcam.models import (
    denormalize,
    imagenet_labels,
    load_model,
    predict,
    preprocess,
    resolve_class,
)
from gradcam.viz import figure_grid, overlay, tensor_to_numpy

OUT_DIR = Path(__file__).resolve().parent.parent / "results" / "figures"
METRICS_DIR = Path(__file__).resolve().parent.parent / "results" / "metrics"
COLUMNS = ("entrada", "Grad-CAM\n(a favor)", "contrafactual\n(contra)")

DISPLAY_HIGH = 0.99
DISPLAY_GAMMA = 0.5


def resolve_layer_arg(value: str | None) -> str | None:
    """Aceita `conv5_3` (nome do artigo) ou o caminho cru do torchvision."""
    if value is None:
        return None
    return VGG16_LAYERS.get(value, value)


def layer_name(model: torch.nn.Module, layer: str | None) -> str | None:
    """Nome qualificado da camada usada pelos dois mapas."""
    if layer is not None:
        return layer
    target = resolve_layer(model, None)
    for name, module in model.named_modules():
        if module is target:
            return name or None
    return None


def peak(heatmap: torch.Tensor) -> dict[str, int]:
    """Coordenada (y, x) do maximo do mapa, na grade do mapa."""
    index = int(heatmap.argmax().item())
    height, width = heatmap.shape
    return {"y": index // width, "x": index % width, "rows": height, "cols": width}


def parse_classes(value: str | None, labels: list[str]) -> list[int]:
    if value is None:
        return []
    tokens = [token.strip() for token in value.split(",") if token.strip()]
    if not tokens:
        raise ValueError("--classes vazio")
    return [resolve_class(token, labels) for token in tokens]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="vgg16")
    parser.add_argument("--images", default="95786", help="id, nome ou caminho")
    parser.add_argument(
        "--classes",
        default="cup,vase",
        help="uma ou duas classes do ImageNet, separadas por virgula",
    )
    parser.add_argument("--layer", default=None, help="conv3_3/conv4_3/conv5_3")
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    start = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"torch {torch.__version__} | dispositivo: {device}")

    layer = resolve_layer_arg(args.layer)
    model = load_model(args.model, device=device)
    labels = imagenet_labels()
    coco = CocoVal()

    try:
        image_id, path = coco.resolve(args.images)
    except ValueError:
        path = Path(args.images)
        if not path.exists():
            print(f"erro: imagem nao encontrada: {args.images}")
            return 1
        image_id = None

    try:
        classes = parse_classes(args.classes, labels)
    except ValueError as error:
        parser.error(str(error))

    image = Image.open(path).convert("RGB")
    x = preprocess(image)
    image_rgb = tensor_to_numpy(denormalize(x))

    if not classes:
        indices, _ = predict(model, x, topk=1)
        classes = [int(indices[0])]

    probs = torch.softmax(model(x)[0], dim=0)
    resolved_name = layer_name(model, layer)
    print(f"imagem: {image_id or path.name} | camada: {resolved_name or 'ultima conv'}")
    print(f"classes: {', '.join(labels[i] for i in classes)}")
    print()

    rows, titles, runs = [], [], []
    map_shape: tuple[int, ...] | None = None

    for class_idx in classes:
        for_map, _ = gradcam(model, x, class_idx, layer=layer)
        cf_map, _ = counterfactual(model, x, class_idx, layer=layer)
        map_shape = tuple(for_map.shape)

        for_peak, cf_peak = peak(for_map), peak(cf_map)
        dy = cf_peak["y"] - for_peak["y"]
        dx = cf_peak["x"] - for_peak["x"]
        shift_cells = (dy * dy + dx * dx) ** 0.5
        stride = 224 / for_map.shape[0]
        rho = spearman(for_map, cf_map)
        prob = float(probs[class_idx])

        title = (
            f"{labels[class_idx]} ({prob:.1%}) | "
            f"pico a favor ({for_peak['y']},{for_peak['x']}) -> "
            f"contra ({cf_peak['y']},{cf_peak['x']}) | "
            f"deslocamento {shift_cells:.1f} celulas"
        )
        titles.append(title)
        rows.append(
            [
                image_rgb,
                overlay(image_rgb, for_map, high=DISPLAY_HIGH, gamma=DISPLAY_GAMMA),
                overlay(image_rgb, cf_map, high=DISPLAY_HIGH, gamma=DISPLAY_GAMMA),
            ]
        )
        runs.append(
            {
                "class": labels[class_idx],
                "class_idx": class_idx,
                "prob": prob,
                "peak_gradcam": for_peak,
                "peak_counterfactual": cf_peak,
                "shift_cells": shift_cells,
                "shift_px": shift_cells * stride,
                "spearman_gradcam_counterfactual": rho,
            }
        )
        print(f"[{labels[class_idx]}] {title}  spearman={rho:+.3f}")

    print()
    print(f"resolucao do mapa: {map_shape}")
    if len(runs) == 2:
        print(
            "leitura: os dois mapas a favor apontam objetos diferentes (Fase 4); "
            "o contrafactual de cada classe deve cair no objeto vizinho."
        )

    args.out.mkdir(parents=True, exist_ok=True)
    tag = f"{image_id or path.stem}-" + "-".join(
        labels[i].replace(" ", "_") for i in classes
    )
    fig = figure_grid(
        rows,
        col_titles=COLUMNS,
        row_titles=titles,
        suptitle=f"{args.model} | Fase 7 | contrafactual (alpha negado em (1))",
    )
    fig_path = args.out / f"07_counterfactual_{args.model}_{tag}.png"
    fig.savefig(fig_path, bbox_inches="tight")
    print(f"salvo: {fig_path}")

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = METRICS_DIR / "counterfactual.json"
    metrics_path.write_text(
        json.dumps(
            {
                "script": "07_counterfactual.py",
                "model": args.model,
                "image": str(image_id or path.name),
                "layer": resolved_name or layer,
                "display_scale": {"high": DISPLAY_HIGH, "gamma": DISPLAY_GAMMA},
                "map_shape": list(map_shape) if map_shape else None,
                "runs": runs,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"metricas salvas: {metrics_path}")
    print(f"\ntempo total: {time.perf_counter() - start:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
