"""Fase 8b - correlacao com oclusao: o mapa ordena os patches como o modelo?

Referencia de fidelidade local. O paper descreve o procedimento em uma frase
("we measure the difference in CNN scores when patches of the input image are
masked"), sem tamanho de patch nem grade; o §5.3 usa exatamente a nocao de
"rank correlation with occlusion", com media em 2510 imagens do PASCAL VOC 2007
val. Os valores de la' (Grad-CAM 0,254; Guided Grad-CAM 0,261; Guided Backprop
0,168) nao sao reproduziveis em valor absoluto aqui: outra base, outro modelo
fino e outra granularidade de patch.

Protocolo operacionalizado aqui (ver `gradcam/eval.py`):

1. oclui-se uma grade `grid` x `grid` (padrao 6x6 = 36 patches) — o patch
   recebe zero no espaco normalizado, que e' a cor media do ImageNet por canal;
2. a referencia e' a queda de probabilidade da classe do cue em cada patch;
3. o mapa de cada metodo e' **reduzido a media por celula** e comparado a essa
   referencia por Spearman — 36 valores contra 36 valores.

Por que nao fazer o contrario (upsample da oclusao para 224): 36 numeros
interpolados em 224 px inventam estrutura e criam milhares de empates, o que
infla a correlacao. A pergunta e' sobre a ordenacao das 36 regioes.

A queda pode ser negativa (ocluir um patch que *ajuda* a classe): o sinal entra
na correlacao como esta; so' a figura usa escala divergente em torno de zero, com
azul para queda negativa, e o titulo diz isso.

Saida:
    results/metrics/occlusion_spearman.json       (contrato de `08_numeros`)
    results/figures/09_occlusion_<modelo>.png

Uso:
    .\\.venv\\Scripts\\python.exe scripts\\09_occlusion_correlation.py
    .\\.venv\\Scripts\\python.exe scripts\\09_occlusion_correlation.py --n 4   # smoke
    .\\.venv\\Scripts\\python.exe scripts\\09_occlusion_correlation.py --n 12
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
    occlusion_map,
    pool_to_grid,
)
from gradcam.metrics import (  # noqa: E402
    gradcam_map_fn,
    guided_gradcam_map_fn,
    saliency_map_fns,
    spearman,
    summarize,
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

# §5.3 do paper: correlacao de rank com oclusao, media em 2510 imagens do PASCAL
# VOC 2007 val. Outra base, outro modelo fino e outra granularidade de patch:
# so' a ordenacao dos metodos e' comparavel.
PAPER = {
    "grad-cam": 0.254,
    "guided grad-cam": 0.261,
    "guided (guided backprop)": 0.168,
    "c-mwp": 0.220,
    "cam": 0.208,
    "note": "§5.3: rank correlation com oclusao (VOC 2007 val, 2510 imagens, VGG-16)",
}


def battery() -> dict[str, object]:
    """Os cinco metodos comparaveis, na ordem em que aparecem nas figuras."""
    fns: dict[str, object] = dict(saliency_map_fns())
    fns["grad-cam"] = gradcam_map_fn()
    fns["guided grad-cam"] = guided_gradcam_map_fn()
    return fns


def collect(coco: CocoVal, model, labels: list[str], args) -> tuple[list[dict], dict[str, dict]]:
    """Roda a bateria no pool e devolve as linhas e o resumo por metodo."""
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
        class_idx = resolve_class(category, labels)

        x = preprocess(coco.load_image(image_id))
        drops = occlusion_map(model, x, class_idx, grid=args.grid)
        maps = {name: map_fn(model, x, class_idx) for name, map_fn in fns.items()}
        rho = {
            name: spearman(drops, pool_to_grid(heatmap, args.grid))
            for name, heatmap in maps.items()
        }

        rows.append(
            {
                "image_id": image_id,
                "cue": category,
                "cue_idx": class_idx,
                "spearman": rho,
                "occlusion": {
                    "max_drop": float(drops.max()),
                    "min_drop": float(drops.min()),
                    "positive_cells": int((drops > 0).sum()),
                },
            }
        )
        marks = " ".join(
            f"{name[:6]}={'nan' if value != value else f'{value:+.2f}'}"
            for name, value in rho.items()
        )
        print(
            f"[{position:3d}/{len(ids)}] {image_id:6d} {category:12s} "
            f"queda max={float(drops.max()):+.3f}  {marks}"
        )

    return rows, aggregate_rows(rows, list(fns))


def aggregate_rows(rows: list[dict], methods: list[str]) -> dict[str, dict]:
    """Media/mediana/desvio do Spearman por metodo, contando os invalidos.

    `nan` aparece quando o mapa e' constante (Grad-CAM zerado pelo ReLU, por
    exemplo). Descartar em silencio mudaria o denominador de um metodo para o
    outro; `n_valid` e `degenerate` deixam isso explicito.
    """
    aggregate: dict[str, dict] = {}
    for name in methods:
        values = [row["spearman"][name] for row in rows]
        stats = summarize(values)
        aggregate[name] = {
            **stats,
            "n": len(values),
            "degenerate": sum(1 for value in values if value != value),
        }
    return aggregate


def pick_example(rows: list[dict]) -> dict:
    """Imagem com sinal de oclusão forte em que o Grad-CAM bate o gradiente cru.

    Mostrar o metodo onde a metrica tem sinal: entre as imagens em que
    `rho(grad-cam) - rho(vanilla) >= 0.3`, a de maior queda de probabilidade.
    Escolher a primeira da lista pegaria um caso de queda ~0,005, que ilustra o
    protocolo mas nao mostra a estrutura que a correlacao mede. Sem nenhuma
    imagem nessa condicao, cai na de maior queda.
    """
    candidates = [
        row for row in rows
        if row["spearman"]["grad-cam"] - row["spearman"]["vanilla"] >= 0.3
    ]
    pool = candidates or rows
    return max(pool, key=lambda row: row["occlusion"]["max_drop"])


def draw_occlusion(ax, drops: torch.Tensor) -> None:
    """Grade de quedas com escala divergente e bordas visiveis.

    Escala divergente em torno de zero, e nao `clamp(min=0)`: a queda pode ser
    negativa (ocluir o patch *aumenta* a probabilidade da classe) e, cortada em
    zero, ela ficaria indistinguivel de um patch neutro. Azul = negativa, que
    entra na correlacao com o sinal.
    """
    values = drops.detach().float()
    limit = float(values.abs().max().clamp(min=1e-6))
    image = ax.imshow(
        values, cmap="coolwarm", vmin=-limit, vmax=limit, interpolation="nearest"
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for edge in range(drops.shape[0] + 1):
        ax.axhline(edge - 0.5, color="black", linewidth=0.6, alpha=0.35)
        ax.axvline(edge - 0.5, color="black", linewidth=0.6, alpha=0.35)
    return image


def plot(rows: list[dict], aggregate: dict[str, dict], coco, model,
         args, path: Path) -> None:
    """Exemplo (entrada, oclusao, mapas) + barras de Spearman medio."""
    example = pick_example(rows)
    x = preprocess(coco.load_image(example["image_id"]))
    class_idx = example["cue_idx"]
    image_rgb = tensor_to_numpy(denormalize(x))
    drops = occlusion_map(model, x, class_idx, grid=args.grid)

    fns = battery()
    maps = {name: map_fn(model, x, class_idx) for name, map_fn in fns.items()}

    columns = 2 + len(fns)
    fig = plt.figure(figsize=(2.0 * columns + 2.0, 7.2), dpi=130, layout="constrained")
    grid = fig.add_gridspec(2, columns, height_ratios=(1.0, 1.15))

    ax = fig.add_subplot(grid[0, 0])
    ax.set_axis_off()
    ax.imshow(image_rgb, interpolation="nearest")
    ax.set_title(f"entrada | {example['cue']}", fontsize=10)

    ax = fig.add_subplot(grid[0, 1])
    ax.set_axis_off()
    image = draw_occlusion(ax, drops)
    ax.set_title(f"oclusao {args.grid}x{args.grid}\nqueda de p (azul<0)", fontsize=9)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)

    for column, (name, heatmap) in enumerate(maps.items(), start=2):
        ax = fig.add_subplot(grid[0, column])
        ax.set_axis_off()
        ax.imshow(
            overlay(image_rgb, heatmap, high=DISPLAY_HIGH, gamma=DISPLAY_GAMMA),
            interpolation="nearest",
        )
        rho = example["spearman"][name]
        label = "mapa zerado" if rho != rho else f"rho={rho:+.2f}"
        ax.set_title(f"{name}\n{label}", fontsize=10, color="#333333")

    ax = fig.add_subplot(grid[1, :])
    names = list(aggregate)
    values = [aggregate[name]["mean"] for name in names]
    errors = [aggregate[name]["std"] / max(aggregate[name]["n_valid"], 1) ** 0.5
              for name in names]
    bars = ax.bar(names, values, yerr=errors, capsize=5, color="#4c72b0", width=0.55)
    for bar, name in zip(bars, names):
        stats = aggregate[name]
        error = errors[names.index(name)]
        # Rotulo sempre acima da linha do zero: embaixo colidiria com o rotulo
        # do eixo x girado, no caso da barra negativa pequena (deconv).
        ax.annotate(
            f"{stats['mean']:+.3f}\n(n={stats['n_valid']}/{stats['n']})",
            (bar.get_x() + bar.get_width() / 2, max(stats["mean"] + error, 0.0)),
            xytext=(0, 4), textcoords="offset points",
            ha="center", va="bottom", fontsize=9,
        )
    for label, value in (("Grad-CAM", PAPER["grad-cam"]),
                         ("Guided Grad-CAM", PAPER["guided grad-cam"])):
        # Duas linhas do paper, nao uma: e' a referencia que a figura compara.
        ax.axhline(value, color="#c44e52", linestyle="--", linewidth=1.0,
                   alpha=0.8, label=f"paper: {label} {value:.3f}")
    ax.axhline(0.0, color="#777777", linewidth=0.8)
    # Espaco para os rotulos das barras nao encostarem no titulo.
    low = min([value - error for value, error in zip(values, errors)] + [0.0])
    high = max([value + error for value, error in zip(values, errors)]
               + [PAPER["grad-cam"], PAPER["guided grad-cam"]])
    ax.set_ylim(low - 0.07, high + 0.12)
    ax.set_ylabel("Spearman medio vs oclusao", fontsize=11)
    ax.set_title(
        f"ordenacao das {args.grid * args.grid} celulas  "
        f"(n={len(rows)} imagens, barra de erro = erro padrao)",
        fontsize=11,
    )
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, loc="upper left")
    plt.setp(ax.get_xticklabels(), rotation=12, ha="right")

    fig.suptitle(
        f"Fase 8 - correlacao com oclusao | {args.model} | cue de ground truth, "
        f"boxes de {args.min_side:.0f} px",
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
    parser.add_argument("--n", type=int, default=100, help="numero de imagens")
    parser.add_argument("--per-category", type=int, default=20,
                        help="teto de imagens por categoria COCO")
    parser.add_argument("--min-side", type=float, default=60.0,
                        help="lado minimo da box do cue, em px do COCO")
    parser.add_argument("--grid", type=int, default=6,
                        help="grade da oclusao (grid x grid patches)")
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
    print(f"{'metodo':18s} {'rho medio':>10s} {'mediana':>8s} {'desvio':>7s} {'validos':>8s} {'zerados':>8s}")
    for name, stats in aggregate.items():
        print(
            f"{name:18s} {stats['mean']:+10.3f} {stats['median']:+8.3f} "
            f"{stats['std']:7.3f} {stats['n_valid']:5d}/{stats['n']:<3d} {stats['degenerate']:8d}"
        )

    # Diagnostico do cue, igual ao do pointing game: se o cue nao tem queda
    # positiva, a correlacao compara mapas que nao tem o que explicar.
    drops_all = [row["occlusion"]["max_drop"] for row in rows]
    print(
        f"\nimagens: {len(rows)} | queda maxima media: {sum(drops_all) / len(drops_all):+.3f} | "
        f"celulas positivas por imagem: "
        f"{sum(row['occlusion']['positive_cells'] for row in rows) / (len(rows) * args.grid ** 2):.1%}"
    )

    args.fig_dir.mkdir(parents=True, exist_ok=True)
    args.metrics_dir.mkdir(parents=True, exist_ok=True)

    fig_path = args.fig_dir / f"09_occlusion_{args.model}.png"
    plot(rows, aggregate, coco, model, args, fig_path)
    print(f"salvo: {fig_path}")

    categories: dict[str, int] = {}
    for row in rows:
        categories[row["cue"]] = categories.get(row["cue"], 0) + 1

    payload = {
        "script": "09_occlusion_correlation.py",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": args.model,
        "seed": args.seed,
        "n_images": len(rows),
        "protocol": {
            "cue": "rotulo de ground truth (COCO verbatim), nao a predicao",
            "grid": args.grid,
            "n_cells": args.grid**2,
            "patch_value": "0.0 no espaco normalizado (cor media do ImageNet)",
            "reference": "queda de softmax p(classe do cue) ao ocluir cada patch",
            "correlation": "Spearman entre 36 celulas; mapa do metodo reduzido a media por celula",
            "not_upsampled": "a oclusao NAO e' ampliada para 224: interpolar 36 valores criaria empates",
            "sign": "quedas negativas entram na correlacao; so' a figura usa clamp(min=0)",
            "min_side_px": args.min_side,
            "per_category_cap": args.per_category,
            "one_cue_per_image": "a imagem entra pelo maior objeto verbatim",
            "categories": dict(sorted(categories.items())),
            "excluded": {
                "counterfactual": "explica a evidencia contra a classe; a oclusao mede a favor dela",
                "cam": "exige arquitetura conv->GAP->fc; a bateria roda no VGG-16",
            },
        },
        "display_scale": {"high": DISPLAY_HIGH, "gamma": DISPLAY_GAMMA,
                          "applies_to": "figuras apenas"},
        "paper": PAPER,
        "aggregate": aggregate,
        "per_image": rows,
    }
    json_path = args.metrics_dir / "occlusion_spearman.json"
    json_path.write_text(
        json.dumps(jsonable(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"salvo: {json_path}")
    print(f"\ntempo total: {time.perf_counter() - start:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
