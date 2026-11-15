"""Etapa 0 - sanity check do ambiente.

Confere que o modelo pre-treinado carrega, roda no dispositivo disponivel e
acerta o top-1 em imagens do COCO val2017.

O critério de aceite usa apenas as categorias COCO que sao *exatamente* uma
classe ImageNet verbatim (`zebra`, `vase`, `traffic light`, `pizza`, `banana`,
`broccoli`, ...). Comparar rótulo COCO com rótulo ImageNet em geral nao faz
sentido: `person`, `car` e `dog` do COCO nao tem contraparte 1:1 no ImageNet.

Uso:
    .\\.venv\\Scripts\\python.exe scripts\\01_forward_sanity.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from gradcam.data import CocoVal
from gradcam.models import imagenet_labels, load_model, predict, preprocess

# Acerto minimo para considerar o ambiente saudavel.
MIN_TOP1 = 0.60


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="vgg16")
    parser.add_argument("--n", type=int, default=8, help="numero de imagens (legado)")
    parser.add_argument(
        "--per-category", type=int, default=2, help="imagens por categoria"
    )
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"torch {torch.__version__} | dispositivo: {device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name} | {props.total_memory / 2**20:.0f} MiB")

    model = load_model(args.model, device=device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"modelo: {args.model} | {n_params / 1e6:.1f}M parametros")

    labels = imagenet_labels()

    coco = CocoVal()
    verbatim = coco.verbatim_categories(labels)
    ids = coco.stratified_subset(args.per_category, seed=0, categories=verbatim)
    print(f"COCO val2017: {len(coco)} imagens | avaliando {len(ids)} "
          f"em {len(verbatim)} categorias")
    print()

    hits_top1 = 0
    hits_top5 = 0
    for image_id in ids:
        image = coco.load_image(image_id)
        x = preprocess(image).to(device)
        truth = coco.object_names(image_id)[0]

        start = time.perf_counter()
        indices, probs = predict(model, x, topk=args.topk)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000

        predicted = [labels[i] for i in indices.tolist()]
        hits_top1 += predicted[0] == truth
        hits_top5 += truth in predicted
        mark = "OK  " if predicted[0] == truth else "erro"
        print(f"{mark} [{image_id}] gt={truth:14s} top1={predicted[0]:20s}"
              f"{probs[0]:5.1%}  ({elapsed:.0f} ms)")

    rate_top1 = hits_top1 / len(ids)
    rate_top5 = hits_top5 / len(ids)
    print()
    print(f"top-1: {rate_top1:.0%}   top-5: {rate_top5:.0%}   "
          f"(minimo exigido para top-1: {MIN_TOP1:.0%})")

    if rate_top1 < MIN_TOP1:
        print("FALHOU: ambiente suspeito. Verifique preprocessing e normalizacao.")
        return 1
    print("ambiente OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
