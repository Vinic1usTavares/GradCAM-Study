"""Fase 8a - pointing game: o pico do mapa cai no objeto do cue?

Protocolo do paper (secao 4.3, sobre Zhang et al., 2018):

1. o cue e' o **rotulo de ground truth**, nao a predicao do modelo;
2. extrai-se o ponto de maximo do mapa;
3. e' hit se o ponto cai dentro de alguma instancia anotada da categoria do cue.

Restricao do COCO (plans/general_plan.md, secao 0): so' as 18 categorias que sao
classe ImageNet verbatim entram. Para elas o conjunto de cues e' unitario, sem o
chute de mapear `dog` ou `person` para um synset. A imagem entra pelo **maior**
objeto verbatim, entao cada imagem e' avaliada uma vez, nao uma vez por
categoria.

O que este script mede e o que ele nao mede:

- mede *onde* o mapa aponta, a mesma pergunta para todos os metodos, com o pico
  calculado depois do upsample para 224 (um CAM 14x14 e uma saliencia 224x224
  sao comparados no mesmo espaco);
- nao mede fidelidade ao modelo — isso e' a correlacao com oclusao, em
  `09_occlusion_correlation.py`;
- a linha de referencia da figura (70,58%) e' de **outro protocolo** — mapas das
  top-5 classes com opcao de rejeitar (§4.2/apendice D) —, nao deste; ela indica
  ordem de grandeza, nao o acerto esperado;
- `counterfactual` fica de fora: ele explica a *ausencia* da classe, e apontar
  para a box dela nao significa acerto nem erro na mesma escala dos outros;
- `cam` fica de fora: exige `conv -> GAP -> fc` e a bateria roda no VGG-16.

Saida:
    results/metrics/pointing_game.json          (contrato de `08_numeros`)
    results/figures/08_pointing_game_<modelo>.png

Uso:
    .\\.venv\\Scripts\\python.exe scripts\\08_pointing_game.py
    .\\.venv\\Scripts\\python.exe scripts\\08_pointing_game.py --n 8      # smoke
    .\\.venv\\Scripts\\python.exe scripts\\08_pointing_game.py --n 40 --seed 1
"""

from __future__ import annotations

import argparse
import json
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
from gradcam.eval import (  # noqa: E402
    cue_object,
    evaluation_pool,
    peak_point,
    pointing_hit,
    scaled_boxes,
    wilson_interval,
)
from gradcam.metrics import (  # noqa: E402
    gradcam_map_fn,
    guided_gradcam_map_fn,
    saliency_map_fns,
)
from gradcam.models import (  # noqa: E402
    denormalize,
    imagenet_labels,
    load_model,
    preprocess,
    resolve_class,
)
from gradcam.viz import overlay, tensor_to_numpy  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "results" / "figures"
METRICS_DIR = ROOT / "results" / "metrics"

# Escala de exibicao das Fases 1-6; nunca entra em metrica.
DISPLAY_HIGH = 0.99
DISPLAY_GAMMA = 0.5

# Referencia do paper: 70,58% vs 60,30% (c-MWP) no VGG-16, mas o numero e' do
# **protocolo modificado** do §4.2/apendice D (mapas das top-5 classes com opcao
# de rejeitar), nao do pointing game com cue de ground truth descrito no §4.3.
# Serve como ordem de grandeza: a comparacao valida aqui e' a ordenacao entre os
# metodos, todos medidos no mesmo protocolo.
PAPER = {
    "pointing_game_vgg16": 0.7058,
    "pointing_game_c_mwp": 0.6030,
    "note": "§4.2/apendice D: protocolo top-5 com rejeicao, VGG-16; o §4.3 (cue de GT) nao tem numero reportado",
}


def battery() -> dict[str, object]:
    """Os cinco metodos comparaveis, na ordem em que aparecem nas figuras."""
    fns: dict[str, object] = dict(saliency_map_fns())
    fns["grad-cam"] = gradcam_map_fn()
    fns["guided grad-cam"] = guided_gradcam_map_fn()
    return fns


def collect(coco: CocoVal, model, labels: list[str], args) -> tuple[list[dict], dict[str, dict]]:
    """Roda a bateria no pool e devolve as linhas e o agregado por metodo."""
    ids = evaluation_pool(
        coco,
        labels,
        per_category=args.per_category,
        min_side=args.min_side,
        seed=args.seed,
        n=args.n,
    )
    fns = battery()
    rows: list[dict] = []

    for position, image_id in enumerate(ids, start=1):
        cue = cue_object(coco, labels, image_id, min_side=args.min_side)
        if cue is None:  # o pool ja filtra; guarda contra pool customizado
            continue
        category, _ = cue

        x = preprocess(coco.load_image(image_id))
        with torch.no_grad():
            probs = model(x)[0].softmax(dim=0)
        class_idx = resolve_class(category, labels)
        probability = float(probs[class_idx])
        rank = int((probs > probs[class_idx]).sum()) + 1

        boxes = scaled_boxes(coco.images[image_id], coco.boxes[image_id], args.peak_size)
        others = [
            box for name, cells in boxes.items() if name != category for box in cells
        ]

        hits: dict[str, dict[str, bool]] = {}
        for name, map_fn in fns.items():
            heatmap = map_fn(model, x, class_idx)
            hits[name] = pointing_hit(heatmap, boxes[category], others, size=args.peak_size)

        rows.append(
            {
                "image_id": image_id,
                "cue": category,
                "cue_idx": class_idx,
                "prob_cued": probability,
                "rank_cued": rank,
                "n_other_boxes": len(others),
                "hits": hits,
            }
        )
        marks = " ".join(
            f"{name[:6]}={'H' if hits[name]['hit'] else ('O' if hits[name]['hit_other'] else '.')}"
            for name in fns
        )
        print(
            f"[{position:3d}/{len(ids)}] {image_id:6d} {category:12s} "
            f"p={probability:6.1%} rank={rank:3d}  {marks}"
        )

    aggregate = aggregate_rows(rows, list(fns))
    return rows, aggregate


def aggregate_rows(rows: list[dict], methods: list[str]) -> dict[str, dict]:
    """Contagem, acuracia e IC de Wilson por metodo."""
    total = len(rows)
    aggregate: dict[str, dict] = {}
    for name in methods:
        hits = sum(1 for row in rows if row["hits"][name]["hit"])
        other = sum(1 for row in rows if row["hits"][name]["hit_other"])
        degenerate = sum(1 for row in rows if row["hits"][name]["degenerate"])
        low, high = wilson_interval(hits, total)
        aggregate[name] = {
            "hits": hits,
            "other_object": other,
            "outside": total - hits - other,
            "degenerate": degenerate,
            "n": total,
            "accuracy": hits / total,
            "ci_low": low,
            "ci_high": high,
        }
    return aggregate


def pick_example(rows: list[dict]) -> dict:
    """Primeira imagem em que o Guided Grad-CAM acerta e o vanilla erra.

    Exemplo escolhido por regra, nao a mao: e' a comparacao que a fase quer
    mostrar (a mascara do CAM salva o pico do gradiente cru). Exige tambem que
    o cue esteja no top-5 do modelo: mostrar o mapa de uma classe que o VGG
    praticamente nao prediz seria ilustrar o protocolo com um caso em que nem
    ha o que localizar. Se nenhuma imagem cumprir tudo, afrouxa nessa ordem e a
    legenda diz o veredito de cada metodo.
    """
    fallbacks = (
        lambda row: row["rank_cued"] <= 5,
        lambda row: True,
    )
    for condition in fallbacks:
        for row in rows:
            hits = row["hits"]
            if (hits["guided grad-cam"]["hit"] and not hits["vanilla"]["hit"]
                    and condition(row)):
                return row
    return rows[0]


def plot(rows: list[dict], aggregate: dict[str, dict], coco, model,
         args, path: Path) -> None:
    """Exemplo com o pico marcado + barras de acuracia com IC."""
    example = pick_example(rows)
    x = preprocess(coco.load_image(example["image_id"]))
    image_rgb = tensor_to_numpy(denormalize(x))
    class_idx = example["cue_idx"]

    fig = plt.figure(figsize=(13.0, 6.6), dpi=130, layout="constrained")
    grid = fig.add_gridspec(2, 6, height_ratios=(1.25, 1.0))

    ax = fig.add_subplot(grid[0, 0])
    ax.set_axis_off()
    ax.imshow(image_rgb, interpolation="nearest")
    ax.set_title(f"entrada | {example['cue']}", fontsize=10)

    fns = battery()
    maps = {name: map_fn(model, x, class_idx) for name, map_fn in fns.items()}
    for column, (name, heatmap) in enumerate(maps.items(), start=1):
        ax = fig.add_subplot(grid[0, column])
        ax.set_axis_off()
        ax.imshow(
            overlay(image_rgb, heatmap, high=DISPLAY_HIGH, gamma=DISPLAY_GAMMA),
            interpolation="nearest",
        )
        result = example["hits"][name]
        px, py = peak_point(heatmap, size=args.peak_size)
        colour = "lime" if result["hit"] else ("orange" if result["hit_other"] else "red")
        ax.plot([px], [py], marker="o", markersize=9, markerfacecolor="none",
                markeredgecolor=colour, markeredgewidth=2.0)
        verdict = "acertou" if result["hit"] else (
            "outro objeto" if result["hit_other"] else "fora"
        )
        ax.set_title(f"{name}\n{verdict}", fontsize=10, color=colour)

    ax = fig.add_subplot(grid[1, :])
    names = list(aggregate)
    values = [aggregate[name]["accuracy"] for name in names]
    errors_low = [aggregate[name]["accuracy"] - aggregate[name]["ci_low"] for name in names]
    errors_high = [aggregate[name]["ci_high"] - aggregate[name]["accuracy"] for name in names]
    bars = ax.bar(names, values, yerr=[errors_low, errors_high], capsize=5,
                  color="#4c72b0", width=0.55)
    for bar, name in zip(bars, names):
        ax.annotate(
            f"{aggregate[name]['accuracy']:.1%}",
            (bar.get_x() + bar.get_width() / 2, aggregate[name]["ci_high"]),
            xytext=(0, 6), textcoords="offset points",
            ha="center", va="bottom", fontsize=10,
        )
    ax.axhline(PAPER["pointing_game_vgg16"], color="#c44e52", linestyle="--",
               linewidth=1.2,
               label=f"paper: {PAPER['pointing_game_vgg16']:.1%} (protocolo top-5)")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("acuracia do pointing game", fontsize=11)
    ax.set_title(
        f"pico do mapa dentro da box do cue  (n={len(rows)}, IC 95% de Wilson)",
        fontsize=11,
    )
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, loc="lower right")
    plt.setp(ax.get_xticklabels(), rotation=12, ha="right")

    fig.suptitle(
        f"Fase 8 - pointing game | {args.model} | cue de ground truth, "
        f"box de {args.min_side:.0f} px minimo",
        fontsize=12,
    )
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def jsonable(value):
    """`nan` -> `null`, para o JSON ser valido."""
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
    parser.add_argument("--n", type=int, default=300, help="numero de imagens")
    parser.add_argument("--per-category", type=int, default=20,
                        help="teto de imagens por categoria COCO")
    parser.add_argument("--min-side", type=float, default=60.0,
                        help="lado minimo da box do cue, em px do COCO")
    parser.add_argument("--peak-size", type=int, default=224,
                        help="grade em que o pico e' medido (224 = pixel da entrada)")
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

    rows, aggregate = collect(coco, model, labels, args)
    if not rows:
        print("Nenhuma imagem candidata: verifique scripts/00_download.py e --n")
        return 1

    print()
    print(f"{'metodo':16s} {'acuracia':>9s} {'IC 95%':>17s} {'outro obj':>10s} {'fora':>6s} {'zerado':>7s}")
    for name, stats in aggregate.items():
        print(
            f"{name:16s} {stats['accuracy']:8.1%}  "
            f"[{stats['ci_low']:5.1%}, {stats['ci_high']:5.1%}] "
            f"{stats['other_object']:10d} {stats['outside']:6d} {stats['degenerate']:7d}"
        )

    ranks = sorted(row["rank_cued"] for row in rows)
    in_top5 = sum(1 for rank in ranks if rank <= 5) / len(ranks)
    probabilities = [row["prob_cued"] for row in rows]
    print(
        f"\nimagens: {len(rows)} | cue no top-5 do modelo: {in_top5:.1%} | "
        f"rank mediano: {ranks[len(ranks) // 2]} | "
        f"prob media do cue: {sum(probabilities) / len(probabilities):.1%}"
    )

    args.fig_dir.mkdir(parents=True, exist_ok=True)
    args.metrics_dir.mkdir(parents=True, exist_ok=True)

    fig_path = args.fig_dir / f"08_pointing_game_{args.model}.png"
    plot(rows, aggregate, coco, model, args, fig_path)
    print(f"salvo: {fig_path}")

    categories: dict[str, int] = {}
    for row in rows:
        categories[row["cue"]] = categories.get(row["cue"], 0) + 1

    payload = {
        "script": "08_pointing_game.py",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": args.model,
        "seed": args.seed,
        "n_images": len(rows),
        "protocol": {
            "cue": "rotulo de ground truth (COCO verbatim), nao a predicao",
            "hit": "pico do mapa, apos upsample, dentro de qualquer instancia anotada do cue",
            "other_object": "pico dentro da box de outra categoria anotada na imagem",
            "outside": "pico fora de todas as boxes anotadas",
            "peak_size": args.peak_size,
            "min_side_px": args.min_side,
            "per_category_cap": args.per_category,
            "one_cue_per_image": "a imagem entra pelo maior objeto verbatim",
            "categories": dict(sorted(categories.items())),
            "cue_diagnostics": {
                "in_top5_fraction": in_top5,
                "median_rank": ranks[len(ranks) // 2],
                "mean_prob": sum(probabilities) / len(probabilities),
            },
            "excluded": {
                "counterfactual": "explica a evidencia contra a classe; o cue de GT nao e' acerto/erro comparavel",
                "cam": "exige arquitetura conv->GAP->fc; a bateria roda no VGG-16",
            },
        },
        "display_scale": {"high": DISPLAY_HIGH, "gamma": DISPLAY_GAMMA,
                          "applies_to": "figuras apenas"},
        "paper": PAPER,
        "aggregate": aggregate,
        "per_image": rows,
    }
    json_path = args.metrics_dir / "pointing_game.json"
    json_path.write_text(
        json.dumps(jsonable(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"salvo: {json_path}")
    print(f"\ntempo total: {time.perf_counter() - start:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
