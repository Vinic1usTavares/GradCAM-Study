"""Fase 10 - sanity checks de Adebayo et al.: o mapa depende dos pesos?

Protocolo (cascade parameter randomization): calcula o mapa no modelo treinado,
reamostra os pesos de uma camada por vez **de cima para baixo** (`classifier` ->
`features.0`), de forma cumulativa, e mede o Spearman contra o mapa original com
a mesma imagem e a mesma classe.

O criterio do plano e' a **queda monotônica** da correlacao: se o mapa nao cai
conforme as camadas sao randomizadas, ele nao depende dos pesos. A comparacao
que interessa no seminario e' entre metodos - Guided Backprop e Guided Grad-CAM
sao os suspeitos classicos, porque a borda da entrada sobrevive ao modelo.

Escopo: a "data randomization" do artigo (retreinar com rotulos embaralhados)
nao e' reproduzida - nao ha treino neste repositorio.

Saida:
    results/figures/11_sanity_curves_<modelo>.png
    results/figures/11_sanity_maps_<modelo>.png
    results/metrics/sanity_cascade.json

O JSON e' o contrato consumido por `notebooks/09_sanity.ipynb`
(plans/notebooks_plan.md, secao 4.4). Nao mudar o schema sem atualizar o notebook.

Uso:
    .\\.venv\\Scripts\\python.exe scripts\\11_sanity_checks.py
    .\\.venv\\Scripts\\python.exe scripts\\11_sanity_checks.py --images 22892
    .\\.venv\\Scripts\\python.exe scripts\\11_sanity_checks.py --cutoffs 0,4,8,16
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
from gradcam.metrics import (  # noqa: E402
    MapFn,
    gradcam_map_fn,
    guided_gradcam_map_fn,
    saliency_map_fn,
)
from gradcam.models import (  # noqa: E402
    denormalize,
    imagenet_labels,
    load_model,
    predict,
    preprocess,
)
from gradcam.sanity import (  # noqa: E402
    DEFAULT_CUTOFFS,
    cascade,
    randomization_order,
    sanity_curve,
)
from gradcam.viz import figure_grid, overlay, tensor_to_numpy  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "results" / "figures"
METRICS_DIR = ROOT / "results" / "metrics"

# Mesma escala de exibicao das demais fases (plans/general_plan.md, secao 2.5).
DISPLAY_HIGH = 0.99
DISPLAY_GAMMA = 0.5

METHOD_LABELS = {
    "vanilla": "vanilla",
    "deconv": "deconv",
    "guided": "guided (GB)",
    "gradcam": "Grad-CAM",
    "guided_gradcam": "Guided Grad-CAM",
}


def build_map_fns(methods: list[str]) -> dict[str, MapFn]:
    fns: dict[str, MapFn] = {}
    for name in methods:
        if name in ("vanilla", "deconv", "guided"):
            fns[name] = saliency_map_fn(name)
        elif name == "gradcam":
            fns[name] = gradcam_map_fn()
        elif name == "guided_gradcam":
            fns[name] = guided_gradcam_map_fn()
        else:
            raise ValueError(f"metodo desconhecido: {name!r}")
    return fns


def first_below_half(steps: list[dict]) -> int | None:
    """Primeiro `k` em que |rho| cai abaixo de 0,5 (metade da estrutura perdida).

    E' um resumo robusto a monotonicidade estrita: quando o mapa zera cedo, o
    resto da curva e' ruido em torno de 0 e os passos seguintes nao dizem nada.
    O limiar 0,5 e' declarado, nao ajustado a' curva.
    """
    for step in steps:
        value = step["abs_spearman"]
        if value == value and value < 0.5:
            return step["k"]
    return None


def monotonic(steps: list[dict]) -> bool:
    """|rho| nunca sobe de um passo para o seguinte (com folga de 0,02).

    A folga existe porque o esperado e' uma queda ruidosa, nao uma curva
    analitica: sem ela, um +0,005 de ruido reprovaria um metodo que cai. O
    modulo e' necessario porque um modelo randomizado pode inverter o sinal do
    gradiente; a pergunta e' se a **estrutura** do mapa sobrevive.
    """
    values = [
        step["abs_spearman"]
        for step in steps
        if step["abs_spearman"] == step["abs_spearman"]
    ]
    return all(b <= a + 0.02 for a, b in zip(values, values[1:]))


def run_image(
    image_id: int,
    model_name: str,
    map_fns: dict[str, MapFn],
    cutoffs: list[int],
    seed: int,
    device: torch.device,
) -> dict:
    """Roda a curva de todos os metodos para uma imagem.

    Cada metodo recarrega o modelo: `sanity_curve` randomiza no lugar, entao
    reaproveitar o modelo entre metodos mediria o metodo seguinte num modelo ja'
    destruido.
    """
    coco = CocoVal()
    x = preprocess(coco.load_image(image_id))

    model = load_model(model_name, device=device)
    indices, probs = predict(model, x, topk=1)
    class_idx = int(indices[0])
    prob_original = float(probs[0])
    labels = imagenet_labels()
    del model

    curves: dict[str, dict] = {}
    for method, map_fn in map_fns.items():
        model = load_model(model_name, device=device)
        result = sanity_curve(
            model, x, class_idx, map_fn, cutoffs=cutoffs, seed=seed
        )
        curves[method] = result
        del model

        values = " -> ".join(
            f"{step['abs_spearman']:+.2f}" if step["abs_spearman"] == step["abs_spearman"] else "n/a"
            for step in result["steps"]
        )
        print(f"  [{method:16s}] {values}")

    return {
        "image": str(image_id),
        "class_idx": class_idx,
        "class": labels[class_idx],
        "prob_original": prob_original,
        "n_parameters_modules": len(
            next(iter(curves.values()))["order"]
        ),
        "order": next(iter(curves.values()))["order"],
        "curves": curves,
    }


def plot_curves(images: list[dict], cutoffs: list[int], model_name: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5), dpi=110, layout="constrained")
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for color, method in zip(colors, images[0]["curves"]):
        per_image = [
            [step["abs_spearman"] for step in image["curves"][method]["steps"]]
            for image in images
        ]
        for values in per_image:
            ax.plot(cutoffs, values, color=color, alpha=0.25, linewidth=1)
        mean = [
            sum(values[i] for values in per_image) / len(per_image)
            for i in range(len(cutoffs))
        ]
        ax.plot(
            cutoffs, mean, color=color, linewidth=2.4,
            marker="o", markersize=4, label=METHOD_LABELS.get(method, method),
        )

    ax.axhline(0.0, color="black", linewidth=0.8, linestyle=":")
    ax.set_xlabel("camadas randomizadas (de cima para baixo)", fontsize=11)
    ax.set_ylabel("|Spearman|(original, randomizado)", fontsize=11)
    ax.set_ylim(-0.1, 1.05)
    ax.set_xticks(cutoffs)
    ax.grid(alpha=0.3)
    n_modules = images[0]["n_parameters_modules"]
    ax.set_title(
        f"Fase 10 - cascade randomization ({model_name}, {n_modules} camadas "
        f"com parametros; linhas finas = imagens)",
        fontsize=11,
    )
    ax.legend(fontsize=9, ncols=2)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_maps(
    image: dict,
    model_name: str,
    map_fns: dict[str, MapFn],
    cutoffs: list[int],
    seed: int,
    device: torch.device,
    path: Path,
) -> None:
    """Original x totalmente randomizado, para os dois metodos de interesse."""
    coco = CocoVal()
    x = preprocess(coco.load_image(int(image["image"])))
    image_rgb = tensor_to_numpy(denormalize(x))
    display = {"gradcam": "Grad-CAM", "guided": "guided (GB)"}

    rows, titles = [], []
    for method, label in display.items():
        if method not in map_fns:
            continue
        # Um unico modelo: mapa original, randomiza ate' o ultimo cutoff (mesma
        # semente da curva) e refaz o mapa. O rho exibido vem da curva, nao e'
        # recalculado aqui.
        model = load_model(model_name, device=device)
        original = map_fns[method](model, x, image["class_idx"]).detach()
        order = randomization_order(model)
        cascade(model, order, cutoffs[-1], torch.Generator().manual_seed(seed))
        randomized = map_fns[method](model, x, image["class_idx"]).detach()
        del model

        rho = image["curves"][method]["steps"][-1]["abs_spearman"]
        rows.append(
            [
                image_rgb,
                overlay(image_rgb, original, high=DISPLAY_HIGH, gamma=DISPLAY_GAMMA),
                overlay(image_rgb, randomized, high=DISPLAY_HIGH, gamma=DISPLAY_GAMMA),
            ]
        )
        titles.append(f"{label} | {image['class']} | rho {rho:+.2f}")

    fig = figure_grid(
        rows,
        col_titles=("entrada", "modelo treinado", "pesos randomizados"),
        row_titles=titles,
        suptitle=(
            f"Fase 10 - o mapa sobrevive à randomização dos pesos? ({model_name})\n"
            "cada mapa normalizado para exibição; o rho mede a estrutura"
        ),
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


def parse_cutoffs(value: str | None) -> list[int]:
    if not value:
        return list(DEFAULT_CUTOFFS)
    cutoffs = sorted({int(token) for token in value.split(",") if token.strip()})
    if not cutoffs:
        raise ValueError("--cutoffs vazio")
    return cutoffs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="vgg16")
    parser.add_argument(
        "--images", nargs="+", default=["22892", "95786"], help="ids do COCO val2017"
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["vanilla", "deconv", "guided", "gradcam", "guided_gradcam"],
        help="metodos da bateria (MapFn)",
    )
    parser.add_argument(
        "--cutoffs",
        default=None,
        help=f"numeros de camadas randomizadas (default: {DEFAULT_CUTOFFS})",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fig-dir", type=Path, default=FIG_DIR)
    parser.add_argument("--metrics-dir", type=Path, default=METRICS_DIR)
    args = parser.parse_args()

    start = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"torch {torch.__version__} | dispositivo: {device}")

    try:
        cutoffs = parse_cutoffs(args.cutoffs)
        map_fns = build_map_fns(args.methods)
        coco = CocoVal()
        ids = [coco.resolve(token)[0] for token in args.images]
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))

    images: list[dict] = []
    for image_id in ids:
        print(f"\n[{image_id}]")
        images.append(
            run_image(image_id, args.model, map_fns, cutoffs, args.seed, device)
        )

    print()
    print(f"{'metodo':18s} {'k=0':>6s} {'final':>7s} {'queda':>7s} {'monotonica':>11s} {'1o k<0,5':>9s}")
    summary: dict[str, dict] = {}
    for method in map_fns:
        finals = [
            image["curves"][method]["steps"][-1]["abs_spearman"] for image in images
        ]
        starts = [
            image["curves"][method]["steps"][0]["abs_spearman"] for image in images
        ]
        mean_final = sum(finals) / len(finals)
        mean_start = sum(starts) / len(starts)
        flags = [monotonic(image["curves"][method]["steps"]) for image in images]
        halves = [first_below_half(image["curves"][method]["steps"]) for image in images]
        summary[method] = {
            "mean_abs_spearman_k0": mean_start,
            "mean_abs_spearman_final": mean_final,
            "mean_drop": mean_start - mean_final,
            "final_abs_spearman_per_image": finals,
            "k_first_below_0p5_per_image": halves,
            "monotonic_per_image": flags,
            "monotonic_all_images": all(flags),
        }
        half_text = ", ".join("nunca" if h is None else str(h) for h in halves)
        print(
            f"{METHOD_LABELS.get(method, method):18s} {mean_start:6.3f} "
            f"{mean_final:7.3f} {mean_start - mean_final:7.3f} "
            f"{'sim' if all(flags) else 'nao':>11s} {half_text:>8s}"
        )

    args.fig_dir.mkdir(parents=True, exist_ok=True)
    args.metrics_dir.mkdir(parents=True, exist_ok=True)

    curves_path = args.fig_dir / f"11_sanity_curves_{args.model}.png"
    plot_curves(images, cutoffs, args.model, curves_path)
    print(f"\nsalvo: {curves_path}")

    maps_path = args.fig_dir / f"11_sanity_maps_{args.model}.png"
    plot_maps(images[0], args.model, map_fns, cutoffs, args.seed, device, maps_path)
    print(f"salvo: {maps_path}")

    n_modules = images[0]["curves"][next(iter(images[0]["curves"]))][
        "n_parameters_modules"
    ]
    payload = {
        "script": "11_sanity_checks.py",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": args.model,
        "protocol": "cascade_parameter_randomization",
        "metric": "|spearman|(map_original, map_randomizado), mesma imagem e classe",
        "seed": args.seed,
        "cutoffs": cutoffs,
        "n_parameters_modules": n_modules,
        "order": images[0]["curves"][next(iter(images[0]["curves"]))]["order"],
        "display_scale": {"high": DISPLAY_HIGH, "gamma": DISPLAY_GAMMA,
                          "applies_to": "figuras apenas"},
        "scope": "data randomization (retreino) fora de escopo; so' parametros",
        "summary": summary,
        "images": images,
    }
    json_path = args.metrics_dir / "sanity_cascade.json"
    json_path.write_text(
        json.dumps(jsonable(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"salvo: {json_path}")
    print(f"\ntempo total: {time.perf_counter() - start:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
