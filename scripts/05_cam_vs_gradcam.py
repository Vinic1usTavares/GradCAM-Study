"""Fase 5: CAM vs Grad-CAM no ResNet18.

Demonstra a equivalencia analitica e numerica entre o CAM original (Zhou et al.,
2016) e o Grad-CAM (Selvaraju et al., 2016) em arquiteturas `conv -> GAP -> fc`:

    Z * L_GradCAM^c = L_CAM^c

O script carrega o ResNet18 com os pesos pre-treinados, seleciona uma imagem
do val2017 (ou customizada com `--images`), calcula CAM e Grad-CAM para a classe
predita (ou `--class`), reporta a correlacao de Spearman e a diferenca maxima,
gera a figura comparativa em results/figures/ e salva as metricas em
results/metrics/cam_vs_gradcam.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from gradcam.core import cam, gradcam, resolve_cam_layers
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
COLUMNS = ["Imagem", "CAM (Zhou 2016)", "Grad-CAM (Selvaraju 2016)"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="resnet18")
    parser.add_argument("--images", nargs="+", default=["15278"], help="id ou caminho da imagem")
    parser.add_argument("--class", default=None, help="classe opcional (nome ou indice)")
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"torch {torch.__version__} | dispositivo: {device}")

    model = load_model(args.model, device=device)
    labels = imagenet_labels()
    coco = CocoVal()

    token = args.images[0]
    try:
        image_id, path = coco.resolve(token)
    except ValueError:
        path = Path(token)
        if not path.exists():
            print(f"erro: imagem nao encontrada: {token}")
            return 1
        image_id, path = None, path

    from PIL import Image
    image_rgb = tensor_to_numpy(denormalize(preprocess(Image.open(path).convert("RGB"))))
    # Para o modelo, precisamos do tensor normalizado
    x = preprocess(Image.open(path).convert("RGB"))

    if getattr(args, "class") is not None:
        class_idx = resolve_class(getattr(args, "class"), labels)
    else:
        indices, _ = predict(model, x, topk=1)
        class_idx = int(indices[0])

    # Computa CAM e Grad-CAM na MESMA ativacao. Isto importa: `cam` usa a saida
    # do bloco (`layer4`), enquanto o `layer=None` do `gradcam` cai na ultima
    # `Conv2d` (`layer4[-1].conv2`, antes do BatchNorm e da soma residual).
    # Comparar tensores diferentes da' rho ~0,987 e diferenca ~6 numa escala de
    # 30 — plausivel o bastante para passar despercebido, e falso: no mesmo
    # tensor, Z * Grad-CAM = CAM vale ate' o ponto flutuante.
    layer, _ = resolve_cam_layers(model)
    cam_map, _ = cam(model, x, class_idx=class_idx, layer=layer)
    gc_map, _ = gradcam(model, x, class_idx=class_idx, layer=layer)

    # Equivalencia: Z * Grad-CAM = CAM
    Z = cam_map.numel() # ou H * W das ativacoes
    diff = (Z * gc_map - cam_map).abs().max().item()
    rho = spearman(cam_map, gc_map)

    print(f"modelo: {args.model} | imagem: {token} | classe: {labels[class_idx]}")
    print(f"Spearman(CAM, Grad-CAM): {rho:.6f}")
    print(f"Diferenca maxima abs (Z * Grad-CAM - CAM): {diff:.2e}")

    # Monta figura 1x3: [Imagem, CAM, Grad-CAM]
    cam_overlay = overlay(image_rgb, cam_map, high=0.99, gamma=0.5)
    gc_overlay = overlay(image_rgb, gc_map, high=0.99, gamma=0.5)

    row_title = f"gt={coco.object_names(image_id)[0] if image_id and coco.object_names(image_id) else '?'} | pred={labels[class_idx]} | rho={rho:.3f}"
    fig = figure_grid(
        [[image_rgb, cam_overlay, gc_overlay]],
        col_titles=COLUMNS,
        row_titles=[row_title],
        suptitle=f"{args.model} | Fase 5 | CAM vs Grad-CAM equivalencia",
    )

    args.out.mkdir(parents=True, exist_ok=True)
    fig_path = args.out / f"05_cam_vs_gradcam_{args.model}.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"salvo: {fig_path}")

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = METRICS_DIR / "cam_vs_gradcam.json"
    metrics_path.write_text(
        json.dumps(
            {
                "model": args.model,
                "image": token,
                "class_idx": class_idx,
                "class_name": labels[class_idx],
                "spearman": rho,
                "max_abs_diff": diff,
                "Z": Z,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"metricas salvas: {metrics_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
