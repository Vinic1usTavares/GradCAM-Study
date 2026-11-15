"""Fase 3 - teste de nao-discriminacao de classe.

Mede o defeito que o Grad-CAM ataca: os metodos de saliencia por gradiente
localizam o objeto, mas nao dizem *qual* classe estao vendo. Operacionalmente,
o mapa da classe A e o da classe B na mesma imagem sao quase identicos - e o
paper mostra isso na Fig. 2 (gato x cachorro).

Metrica: `spearman(map_A, map_B)` no mapa `max|d y^c / d x|` por pixel, por
metodo. Spearman e' invariante a escala monotonica, entao a normalizacao de
exibicao nao entra na conta.

Saida:
    results/metrics/class_discrimination.json    (fonte dos notebooks)
    results/figures/03_class_discrimination.png  (barras + exemplo qualitativo)

O JSON e' o contrato consumido por `notebooks/02_relus.ipynb`; ver
`plans/notebooks_plan.md` secao 4.4. Nao mudar o schema sem atualizar o notebook.

Uso:
    .\\.venv\\Scripts\\python.exe scripts\\03_class_discrimination.py
    .\\.venv\\Scripts\\python.exe scripts\\03_class_discrimination.py --n 40
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from gradcam.data import CocoVal  # noqa: E402
from gradcam.metrics import (  # noqa: E402
    class_discrimination,
    saliency_map_fn,
    saliency_map_fns,
    summarize,
)
from gradcam.models import (  # noqa: E402
    denormalize,
    imagenet_labels,
    load_model,
    predict,
    preprocess,
)
from gradcam.viz import (  # noqa: E402
    gray_rgb,
    tensor_to_numpy,
)

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "results" / "figures"
METRICS_DIR = ROOT / "results" / "metrics"

# Mesma escala de exibicao das Fases 1-2 (plans/general_plan.md, secao 2.5).
DISPLAY_HIGH = 0.99
DISPLAY_GAMMA = 0.5


def collect(coco: CocoVal, model, args) -> tuple[list[dict], dict[str, list[float]]]:
    """Roda a bateria de metodos sobre o subconjunto e devolve os rhos."""
    labels = imagenet_labels()
    verbatim = coco.verbatim_categories(labels)

    # Amostra estratificada por categoria: sem isso, uma amostra uniforme do
    # COCO val2017 acaba dominada por zebra (12 das 24 primeiras na semente 0).
    pool = coco.stratified_subset(
        args.per_category, seed=args.seed, categories=verbatim
    )
    rng = random.Random(args.seed)
    ids = pool if len(pool) <= args.n else sorted(rng.sample(pool, args.n))

    map_fns = saliency_map_fns()
    rows: list[dict] = []
    values: dict[str, list[float]] = {name: [] for name in map_fns}

    for image_id in ids:
        x = preprocess(coco.load_image(image_id))
        indices, probs = predict(model, x, topk=2)
        class_a, class_b = int(indices[0]), int(indices[1])

        rhos = class_discrimination(model, x, class_a, class_b, map_fns)
        for name, rho in rhos.items():
            values[name].append(rho)

        rows.append(
            {
                "image_id": image_id,
                "truth": coco.object_names(image_id)[0],
                "class_a": class_a,
                "class_b": class_b,
                "label_a": labels[class_a],
                "label_b": labels[class_b],
                "prob_a": float(probs[0]),
                "prob_b": float(probs[1]),
                "rho": rhos,
            }
        )
        rhos_text = "  ".join(
            f"{name}={rhos[name]:+.3f}" if rhos[name] == rhos[name] else f"{name}=n/a"
            for name in map_fns
        )
        print(
            f"[{image_id}] {coco.object_names(image_id)[0]:14s} "
            f"{labels[class_a]:16s} vs {labels[class_b]:16s}  {rhos_text}"
        )

    return rows, values


def plot(rows: list[dict], values: dict[str, list[float]], coco, model,
         path: Path) -> None:
    """Barras por metodo + um exemplo qualitativo das duas classes."""
    example = rows[0]
    x = preprocess(coco.load_image(example["image_id"]))
    image_rgb = tensor_to_numpy(denormalize(x))
    guided_fn = saliency_map_fn("guided")

    fig = plt.figure(figsize=(9.5, 7.4), dpi=130, layout="constrained")
    fig.get_layout_engine().set(h_pad=0.02)
    grid = fig.add_gridspec(2, 3, height_ratios=(1.5, 1.0))

    panels = [
        ("entrada", image_rgb),
        (
            f"guided | {example['label_a']}",
            gray_rgb(guided_fn(model, x, example["class_a"]),
                     high=DISPLAY_HIGH, gamma=DISPLAY_GAMMA),
        ),
        (
            f"guided | {example['label_b']}",
            gray_rgb(guided_fn(model, x, example["class_b"]),
                     high=DISPLAY_HIGH, gamma=DISPLAY_GAMMA),
        ),
    ]
    for col, (title, cell) in enumerate(panels):
        ax = fig.add_subplot(grid[0, col])
        ax.set_axis_off()
        ax.imshow(cell, interpolation="nearest")
        ax.set_title(title, fontsize=11)

    ax = fig.add_subplot(grid[1, :])
    names = list(values)
    summaries = {name: summarize(values[name]) for name in names}
    means = [summaries[name]["mean"] for name in names]
    stds = [summaries[name]["std"] for name in names]
    bars = ax.bar(names, means, yerr=stds, capsize=5, color="#4c72b0", width=0.5)
    for bar, mean, std in zip(bars, means, stds):
        ax.annotate(
            f"{mean:.3f}",
            (bar.get_x() + bar.get_width() / 2, mean + std),
            xytext=(0, 6), textcoords="offset points",
            ha="center", va="bottom", fontsize=10,
        )
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Spearman entre as classes", fontsize=11)
    ax.set_title(
        "mesma imagem, classes top-1 e top-2: mais alto = menos discriminativo",
        fontsize=11,
    )
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    n_images = len(rows)
    fig.suptitle(
        f"Fase 3 - nao-discriminacao de classe  ({n_images} imagens, "
        f"erro = desvio)",
        fontsize=12,
    )
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def jsonable(value):
    """`nan` -> `null`, para o JSON ser valido (json.dump escreveria NaN)."""
    if isinstance(value, float):
        return None if value != value else value
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="vgg16")
    parser.add_argument("--n", type=int, default=18, help="numero de imagens")
    parser.add_argument(
        "--per-category", type=int, default=1, help="imagens por categoria"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fig-dir", type=Path, default=FIG_DIR)
    parser.add_argument("--metrics-dir", type=Path, default=METRICS_DIR)
    args = parser.parse_args()

    start = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"torch {torch.__version__} | dispositivo: {device}")

    model = load_model(args.model, device=device)
    labels = imagenet_labels()
    coco = CocoVal()

    rows, values = collect(coco, model, args)
    if not rows:
        print("Nenhuma imagem candidata: verifique scripts/00_download.py e --n")
        return 1

    aggregate = {name: summarize(vals) for name, vals in values.items()}

    print()
    print(f"{'metodo':10s} {'media':>8s} {'mediana':>8s} {'desvio':>8s} {'n':>4s}")
    for name, stats in aggregate.items():
        print(
            f"{name:10s} {stats['mean']:+8.3f} {stats['median']:+8.3f} "
            f"{stats['std']:8.3f} {stats['n_valid']:4d}"
        )
    print(f"\nimagens: {len(rows)} | "
          f"pareamento: classe top-1 vs top-2 | semente: {args.seed}")

    args.fig_dir.mkdir(parents=True, exist_ok=True)
    args.metrics_dir.mkdir(parents=True, exist_ok=True)

    fig_path = args.fig_dir / "03_class_discrimination.png"
    plot(rows, values, coco, model, fig_path)
    print(f"salvo: {fig_path}")

    payload = {
        "script": "03_class_discrimination.py",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": args.model,
        "seed": args.seed,
        "n_images": len(rows),
        "pair_scheme": "top1_vs_top2",
        "metric": "spearman(map_a, map_b) sobre max|grad| por pixel",
        "display_scale": {"high": DISPLAY_HIGH, "gamma": DISPLAY_GAMMA,
                          "applies_to": "figuras apenas"},
        "aggregate": aggregate,
        "per_image": rows,
    }
    json_path = args.metrics_dir / "class_discrimination.json"
    json_path.write_text(
        json.dumps(jsonable(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"salvo: {json_path}")

    print(f"\ntempo total: {time.perf_counter() - start:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
