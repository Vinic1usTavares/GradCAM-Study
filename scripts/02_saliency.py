"""Fases 1-2 - saliencia por gradiente: vanilla, deconv e guided backprop.

Para cada imagem, calcula o mapa de saliencia do gradiente em relacao a entrada
para **duas classes** e nas **tres regras de backward** pelo ReLU. As tres
regras so mudam a mascara aplicada ao gradiente na volta (ver
`gradcam/relu_variants.py`); o forward e' identico.

Criterio de aceite (plans/general_plan.md, secao 3):
    - Fase 1: mapas vanilla com aspecto de contorno ruidoso, 4 imagens x 2 classes
    - Fase 2: os tres mapas lado a lado, mesma imagem e mesma classe

Saida:
    results/figures/02_saliency_<modelo>_top1_<estilo>.png   (classe predita)
    results/figures/02_saliency_<modelo>_top2_<estilo>.png   (segunda classe)

As classes sao o top-1 e o top-2 do modelo. Nao usamos o rotulo do COCO como
cue: o mapeamento COCO -> synset do ImageNet e' assunto da Fase 8, e aqui o que
importa e' a textura do mapa, nao a classe estar "certa".

Uso:
    .\\.venv\\Scripts\\python.exe scripts\\02_saliency.py
    .\\.venv\\Scripts\\python.exe scripts\\02_saliency.py --n 6 --overlay
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from gradcam.data import CocoVal
from gradcam.models import (
    denormalize,
    imagenet_labels,
    load_model,
    predict,
    preprocess,
)
from gradcam.relu_variants import MODES, set_relu_mode
from gradcam.saliency import saliency
from gradcam.viz import figure_grid, gray_rgb, overlay, tensor_to_numpy

OUT_DIR = Path(__file__).resolve().parent.parent / "results" / "figures"

# Rotulos das colunas; a ordem casa com MODES.
COLUMNS = ("entrada", "vanilla", "deconv", "guided")

# Escala de exibicao (nao de metrica): corte em 99% + gamma 0.5. Sem isso o
# mapa vanilla, cujo pico e' ~5e5 e a mediana ~5% disso, sai quase preto.
DISPLAY_HIGH = 0.99
DISPLAY_GAMMA = 0.5


def select_records(coco: CocoVal, model, labels: list[str], args) -> list[dict]:
    """Imagens do subset verbatim, com o forward ja' feito e o top-2 guardado.

    As imagens em que o modelo acerta o top-1 vem primeiro: sao as mais
    interpretaveis para a figura do seminario. Nao descartamos as outras, so'
    as colocamos no fim, para `--n` grande nao quebrar.
    """
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

    found.sort(key=lambda record: labels[int(record["indices"][0])] != record["truth"])
    return found[: args.n]


def saliency_triplet(model, x: torch.Tensor, class_idx: int) -> list:
    """Os tres mapas da mesma entrada e da mesma classe, na ordem de MODES."""
    return [saliency(model, x, class_idx, mode=mode)[0] for mode in MODES]


def row_title(truth: str, label: str, prob: float) -> str:
    return f"gt={truth} | {label} ({prob:.0%})"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="vgg16")
    parser.add_argument("--n", type=int, default=4, help="numero de imagens")
    parser.add_argument(
        "--per-category", type=int, default=2, help="imagens por categoria"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--overlay",
        action="store_true",
        help="mostra os mapas como overlay jet em vez de cinza",
    )
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    start = time.perf_counter()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"torch {torch.__version__} | dispositivo: {device}")

    model = load_model(args.model, device=device)
    labels = imagenet_labels()
    coco = CocoVal()

    records = select_records(coco, model, labels, args)
    if not records:
        print("Nenhuma imagem candidata: verifique scripts/00_download.py e --n")
        return 1

    print(f"imagens: {len(records)} | classes por imagem: 2 | metodos: {len(MODES)}")
    print()

    # rows[slot] = (celulas, titulos), uma figura por slot de classe. As 8
    # linhas (4 imagens x 2 classes) numa figura so' deixariam cada celula
    # pequena demais para julgar a textura do gradiente.
    rows: dict[int, tuple[list, list[str]]] = {0: ([], []), 1: ([], [])}

    for record in records:
        image_rgb = tensor_to_numpy(denormalize(record["x"]))
        for slot in (0, 1):
            class_idx = int(record["indices"][slot])
            prob = float(record["probs"][slot])
            label = labels[class_idx]

            maps = saliency_triplet(model, record["x"], class_idx)
            cells = [image_rgb]
            for heatmap in maps:
                cells.append(
                    overlay(
                        image_rgb, heatmap, high=DISPLAY_HIGH, gamma=DISPLAY_GAMMA
                    )
                    if args.overlay
                    else gray_rgb(heatmap, high=DISPLAY_HIGH, gamma=DISPLAY_GAMMA)
                )

            peaks = " / ".join(f"{float(m.max()):.2e}" for m in maps)
            rows[slot][0].append(cells)
            rows[slot][1].append(row_title(record["truth"], label, prob))
            print(f"[{record['image_id']}] slot {slot + 1}: "
                  f"{row_title(record['truth'], label, prob)} | "
                  f"pico |grad|={peaks}")

    args.out.mkdir(parents=True, exist_ok=True)
    suffix = "overlay" if args.overlay else "gray"
    for slot, name in ((0, "top1"), (1, "top2")):
        cell_rows, titles = rows[slot]
        if not cell_rows:
            continue
        fig = figure_grid(
            cell_rows,
            col_titles=COLUMNS,
            row_titles=titles,
            suptitle=(
                f"{args.model} | Fases 1-2 | classe {name} | "
                f"{len(cell_rows)} imagens | {suffix}"
            ),
        )
        path = args.out / f"02_saliency_{args.model}_{name}_{suffix}.png"
        fig.savefig(path, bbox_inches="tight")
        print(f"salvo: {path}")

    print(f"\ntempo total: {time.perf_counter() - start:.1f} s")

    # Devolve o modelo ao estado neutro: `set_relu_mode` troca modulos no lugar
    # e um script seguinte nao deve herdar a mascara de guided/deconv.
    set_relu_mode(model, "vanilla")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
