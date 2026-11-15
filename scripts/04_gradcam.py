"""Fase 4 - Grad-CAM: mapa de "gato" no gato e de "cachorro" no cachorro.

O criterio de aceite do plano e' qualitativo (mapa por classe, na mesma foto).
Aqui: 4 imagens x 2 classes (top-1 e top-2 do modelo), com o Grad-CAM em
overlay. Cada linha mostra a mesma imagem sob as duas classes; se o mapa
muda de posicao entre as colunas, o metodo discrimina.

Para o numero: imprime tambem o Spearman entre os dois mapas. E' a mesma
metrica da Fase 3. A comparacao honesta e' contra deconv/guided (0,999), que
operam na mesma resolucao de pixel: o mapa 14x14 do Grad-CAM muda a ordenacao
por suavizacao, entao ele fica em 0,760 — acima do 0,642 do vanilla, e isso
nao significa "menos discrimina". Ver plans/general_plan.md, secao 0.

Saida:
    results/figures/04_gradcam_<modelo>.png

Uso:
    .\\.venv\\Scripts\\python.exe scripts\\04_gradcam.py
    .\\.venv\\Scripts\\python.exe scripts\\04_gradcam.py --layer conv4_3

A ablacao "sem ReLU" de (2) fica na Fase 9, medida por localizacao: no mapa
exibido ela quase nao muda nada, porque a escala de exibicao ja' corta os
valores negativos. `core.gradcam(apply_relu=False)` existe para essa medicao.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from gradcam.core import VGG16_LAYERS, gradcam
from gradcam.data import CocoVal
from gradcam.metrics import spearman
from gradcam.models import (
    denormalize,
    imagenet_labels,
    load_model,
    predict,
    preprocess,
)
from gradcam.viz import figure_grid, overlay, tensor_to_numpy

OUT_DIR = Path(__file__).resolve().parent.parent / "results" / "figures"

COLUMNS = ("entrada", "grad-cam | classe A", "grad-cam | classe B")

# Escala de exibicao das Fases 1-3 (plans/general_plan.md, secao 2.5).
DISPLAY_HIGH = 0.99
DISPLAY_GAMMA = 0.5


def resolve_layer_arg(value: str | None) -> str | None:
    """Aceita `conv5_3` (nome do artigo) ou o caminho cru do torchvision."""
    if value is None:
        return None
    return VGG16_LAYERS.get(value, value)


def layer_suffix(raw: str | None, resolved: str | None) -> str:
    """Sufixo de arquivo por camada — sem ele, `--layer conv4_3` sobrescreve
    a figura padrao e a ablacao da Fase 9 perde a comparacao lado a lado."""
    if raw is None:
        return ""
    name = raw if raw in VGG16_LAYERS else (resolved or raw)
    return "_" + name.replace(".", "_")


def select_records(coco: CocoVal, model, labels: list[str], args) -> list[dict]:
    verbatim = coco.verbatim_categories(labels)
    ids = coco.stratified_subset(
        args.per_category, seed=args.seed, categories=verbatim
    )

    found: list[dict] = []
    for image_id in ids:
        x = preprocess(coco.load_image(image_id))
        indices, probs = predict(model, x, topk=2)
        found.append(
            {
                "image_id": image_id,
                "x": x,
                "truth": coco.object_names(image_id)[0],
                "indices": indices,
                "probs": probs,
            }
        )

    found.sort(key=lambda r: labels[int(r["indices"][0])] != r["truth"])
    return found[: args.n]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="vgg16")
    parser.add_argument("--n", type=int, default=4, help="numero de imagens")
    parser.add_argument(
        "--per-category", type=int, default=2, help="imagens por categoria"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--layer",
        default=None,
        help="camada: conv3_3/conv4_3/conv5_3 ou 'features.15' etc",
    )
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    start = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"torch {torch.__version__} | dispositivo: {device}")

    layer = resolve_layer_arg(args.layer)
    model = load_model(args.model, device=device)
    labels = imagenet_labels()
    coco = CocoVal()

    records = select_records(coco, model, labels, args)
    if not records:
        print("Nenhuma imagem candidata: verifique scripts/00_download.py e --n")
        return 1

    print(f"imagens: {len(records)} | camada: {layer or 'ultima conv'}")
    print()

    rows, titles, rhos = [], [], []
    map_shape: tuple[int, ...] | None = None

    for record in records:
        image_rgb = tensor_to_numpy(denormalize(record["x"]))
        classes = [
            (int(record["indices"][slot]), float(record["probs"][slot]))
            for slot in (0, 1)
        ]

        maps = []
        for class_idx, _ in classes:
            heatmap, _ = gradcam(model, record["x"], class_idx, layer=layer)
            map_shape = tuple(heatmap.shape)
            maps.append(heatmap)

        rho = spearman(maps[0], maps[1])
        rhos.append(rho)

        cells = [image_rgb] + [
            overlay(image_rgb, heatmap, high=DISPLAY_HIGH, gamma=DISPLAY_GAMMA)
            for heatmap in maps
        ]
        rows.append(cells)
        (a_idx, a_prob), (b_idx, b_prob) = classes
        titles.append(
            f"gt={record['truth']} | A={labels[a_idx]} ({a_prob:.0%}) | "
            f"B={labels[b_idx]} ({b_prob:.0%})"
        )
        print(f"[{record['image_id']}] {titles[-1]}  "
              f"spearman(A,B)={rho:+.3f}")

    valid = [rho for rho in rhos if rho == rho]
    mean_rho = sum(valid) / len(valid) if valid else float("nan")
    print()
    print(f"spearman entre classes: media {mean_rho:+.3f} (n={len(valid)}) "
          f"| Fase 3: guided +0.999, vanilla +0.642")
    print(f"resolucao do mapa: {map_shape}")

    args.out.mkdir(parents=True, exist_ok=True)
    fig = figure_grid(
        rows,
        col_titles=COLUMNS,
        row_titles=titles,
        suptitle=(
            f"{args.model} | Fase 4 | Grad-CAM ({layer or 'ultima conv'})"
        ),
    )
    path = args.out / f"04_gradcam_{args.model}{layer_suffix(args.layer, layer)}.png"
    fig.savefig(path, bbox_inches="tight")
    print(f"salvo: {path}")
    print(f"\ntempo total: {time.perf_counter() - start:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
