"""Fase 6 - Guided Grad-CAM: detalhe fino sem perder a classe.

    L^c_guided = Upsample(L^c_GradCAM) . GuidedBackprop(x)

O criterio de aceite tem duas metades, e o script mede as duas:

1. **Detalhe fino.** O Grad-CAM tem 14x14 no VGG-16; o Guided Backprop, 224x224.
   O indice de detalhe (variacao total normalizada) do Guided Grad-CAM tem que
   ficar perto do Guided Backprop, nao do CAM.
2. **Discriminacao de classe.** No par de classes da mesma imagem, a fracao dos
   5% pixels mais fortes que cai dentro da box de ground truth do objeto certo
   continua alta depois do produto — e o que o `guided` sozinho nao faz (Fase 3:
   rho 0,999 entre as classes).

Casos curados (escolhidos por medicao; ver plans/general_plan.md, secao 0):

    22892  gato x cachorro   -> o caso da Fig. 2 do artigo, rho medido +0,062
    23781  brocoli x pepino  -> textura parecida, controle dificil
    47585  noivo x guarda-chuva -> classes bem distintas, top-1 em 91,2%
    54931  pessoa x cavalo   -> o top-1 e' `bubble` (o laco de corda), cross ~0

Duas imagens com a mesma estrutura foram medidas e descartadas: em `61333`
(gato x estante) e em `106389` (gato x estante/cadeira) **nenhuma** classe de
estante/cadeira localiza — o gato domina o gradiente e o Guided Grad-CAM herda
essa localizacao (top5% na box certa: 0,31 para `bookshop`, 0,69 na box do
gato). Fica registrado como limite do metodo, nao como figura.

`cat` e `dog` **nao** sao classes ImageNet verbatim: o cue e' um synset
explicito (`Egyptian cat`, `beagle`). Ver a restricao COCO x ImageNet na secao 0
do plano. Em 22892 o top-1 do modelo e' `koala` (49,8%) — o cue e' a classe que
*perguntamos*, nao a predicao, e a legenda da figura diz isso.

Saida:
    results/figures/06_guided_gradcam_<modelo>_<id>_<classeA>-<classeB>.png
    results/figures/06_guided_gradcam_<modelo>_<id>_zoom.png

Uso:
    .\\.venv\\Scripts\\python.exe scripts\\06_guided_gradcam.py
    .\\.venv\\Scripts\\python.exe scripts\\06_guided_gradcam.py --overlay
    .\\.venv\\Scripts\\python.exe scripts\\06_guided_gradcam.py --images 22892 --classes "Egyptian cat,beagle"
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from PIL import Image

from gradcam.core import VGG16_LAYERS, gradcam
from gradcam.data import Box, CocoVal
from gradcam.guided import compose_guided_gradcam
from gradcam.metrics import spearman
from gradcam.models import (
    denormalize,
    imagenet_labels,
    load_model,
    predict,
    preprocess,
    resolve_class,
)
from gradcam.saliency import guided_backprop
from gradcam.viz import figure_grid, gray_rgb, overlay, tensor_to_numpy, upsample

OUT_DIR = Path(__file__).resolve().parent.parent / "results" / "figures"

COLUMNS = ("entrada", "grad-cam (14x14)", "guided backprop", "guided grad-cam")

# Escala de exibicao (plans/general_plan.md, secao 2.5): nunca em metrica.
DISPLAY_HIGH = 0.99
DISPLAY_GAMMA = 0.5

# Fracao dos pixels mais fortes usada na medida de localizacao. Um pico unico
# e' fragil: o produto com |GB| pode deslocar o argmax para uma borda.
TOP_FRACTION = 0.05

CASES = (
    {
        "image": "22892",
        "classes": ("Egyptian cat", "beagle"),
        "objects": ("cat", "dog"),
        "note": "gato x cachorro (Fig. 2 do artigo)",
        "zoom": None,
    },
    {
        "image": "23781",
        "classes": ("broccoli", "cucumber"),
        "objects": ("broccoli", None),
        "note": "brocoli x pepino (textura parecida)",
        "zoom": "broccoli",
    },
    {
        "image": "47585",
        "classes": ("groom", "umbrella"),
        "objects": ("person", "umbrella"),
        "note": "noivo x guarda-chuva (91,2% e 3,8%)",
        "zoom": "umbrella",
    },
    {
        "image": "54931",
        "classes": ("bubble", "ox"),
        "objects": ("person", "horse"),
        "note": "pessoa x cavalo (top-1 'bubble': o laco de corda)",
        "zoom": None,
    },
)


def scaled_boxes(coco: CocoVal, image_id: int, size: int = 224) -> dict[str, list[Box]]:
    """Boxes do COCO na grade `size` x `size` do `preprocess`.

    `preprocess` redimensiona direto para 224x224, sem crop, entao a box escala
    por `size/width` e `size/height` e continua valida.
    """
    meta = coco.images[image_id]
    sx, sy = size / meta["width"], size / meta["height"]
    return {
        name: [Box(b.x * sx, b.y * sy, b.w * sx, b.h * sy) for b in boxes]
        for name, boxes in coco.boxes[image_id].items()
    }


def top_fraction_inside(
    heatmap: torch.Tensor,
    boxes: list[Box],
    grid: int = 224,
    fraction: float = TOP_FRACTION,
) -> float:
    """Fracao dos `fraction` pixels mais fortes do mapa que cai dentro das boxes.

    O mapa e' reamostrado para a grade da imagem antes da conta, para comparar
    o CAM 14x14 com o Guided Grad-CAM 224x224 em pe' de igualdade.
    """
    if not boxes:
        return float("nan")

    values = upsample(heatmap.detach().float(), (grid, grid))
    k = max(1, int(round(values.numel() * fraction)))
    selected = values >= torch.topk(values.flatten(), k).values.min()

    ys, xs = torch.meshgrid(
        torch.arange(grid), torch.arange(grid), indexing="ij"
    )
    inside = torch.zeros((grid, grid), dtype=torch.bool)
    for box in boxes:
        inside |= (
            (xs >= box.x) & (xs <= box.x + box.w)
            & (ys >= box.y) & (ys <= box.y + box.h)
        )
    return float((selected & inside).sum()) / float(selected.sum())


def detail_index(heatmap: torch.Tensor) -> float:
    """Variacao total normalizada: `(mean|dx| + mean|dy|) / mean|m|`.

    Sem unidade e invariante a escala, entao da' para comparar mapas com
    magnitudes muito diferentes (o pico do gradiente guided e' ~2, o do CAM
    ~0,1). Mapa suave (CAM 14x14 interpolado) fica baixo; textura de borda
    (Guided Backprop) fica alto.
    """
    values = heatmap.detach().abs().float()
    mean = values.mean()
    if not torch.isfinite(mean) or mean <= 0:
        return float("nan")
    dx = (values[:, 1:] - values[:, :-1]).abs().mean()
    dy = (values[1:, :] - values[:-1, :]).abs().mean()
    return float((dx + dy) / mean)


def resolve_cases(labels: list[str], args, model, coco: CocoVal) -> list[dict]:
    """Casos curados por padrao; `--images/--classes` monta um caso customizado."""
    if not args.images:
        return [
            {**case, "indices": [resolve_class(name, labels) for name in case["classes"]]}
            for case in CASES
        ]

    token = args.images[0]
    if len(args.images) > 1:
        raise ValueError("--images aceita uma imagem por vez nesta fase")
    image_id, path = coco.resolve(token)

    if args.classes:
        tokens = [part.strip() for part in args.classes.split(",") if part.strip()]
        if len(tokens) != 2:
            raise ValueError(
                f"--classes espera duas classes separadas por virgula; recebi {args.classes!r}"
            )
    else:
        image = coco.load_image(image_id) if image_id is not None else Image.open(path)
        x = preprocess(image.convert("RGB"))
        indices, _ = predict(model, x, topk=2)
        tokens = [labels[int(index)] for index in indices]

    return [
        {
            "image": token,
            "classes": tuple(tokens),
            "objects": (None, None),
            "note": "caso customizado",
            "zoom": None,
            "indices": [resolve_class(name, labels) for name in tokens],
        }
    ]


def map_cell(heatmap: torch.Tensor, as_overlay: bool, image_rgb, overlay_mode: bool):
    """Celula da figura para um mapa; overlay jet ou cinza (decisao da Fase 2)."""
    if overlay_mode and as_overlay:
        return overlay(image_rgb, heatmap, high=DISPLAY_HIGH, gamma=DISPLAY_GAMMA)
    return gray_rgb(heatmap, high=DISPLAY_HIGH, gamma=DISPLAY_GAMMA)


def run_case(case: dict, coco: CocoVal, model, labels: list[str], overlay_mode: bool,
             layer=None):
    """Calcula os mapas do caso, imprime as medidas e devolve os dados da figura."""
    image_id, path = coco.resolve(case["image"])
    image = Image.open(path).convert("RGB")
    x = preprocess(image)
    image_rgb = tensor_to_numpy(denormalize(x))

    boxes = scaled_boxes(coco, image_id) if image_id is not None else {}
    with torch.no_grad():
        probs = model(x)[0].softmax(dim=0)

    parts = []
    for class_idx in case["indices"]:
        cam, _ = gradcam(model, x, class_idx, layer=layer)
        detail, _ = guided_backprop(model, x, class_idx)
        composed = compose_guided_gradcam(cam, detail, (224, 224))
        parts.append({"class_idx": class_idx, "cam": cam, "detail": detail, "ggc": composed})

    rho_cam = spearman(parts[0]["cam"], parts[1]["cam"])
    rho_ggc = spearman(parts[0]["ggc"], parts[1]["ggc"])

    ident = image_id if image_id is not None else Path(case["image"]).stem
    print(f"[{ident}] {case['note']}")
    for slot, part in enumerate(parts):
        class_idx = part["class_idx"]
        own = case["objects"][slot]
        other = case["objects"][1 - slot]
        own_boxes = boxes.get(own, []) if own else []
        other_boxes = boxes.get(other, []) if other else []
        part["own_name"] = own
        part["own"] = top_fraction_inside(part["ggc"], own_boxes)
        part["other"] = top_fraction_inside(part["ggc"], other_boxes)
        part["cam_own"] = top_fraction_inside(part["cam"], own_boxes)
        part["detail_cam"] = detail_index(upsample(part["cam"], (224, 224)))
        part["detail_gb"] = detail_index(part["detail"])
        part["detail_ggc"] = detail_index(part["ggc"])
        part["prob"] = float(probs[class_idx])
        print(
            f"   {labels[class_idx]:16s} p={part['prob']:6.2%} | "
            f"GGC top5% em '{own or '?':9s}' {part['own']:.2f}"
            f" (outro objeto {part['other']:.2f}, CAM {part['cam_own']:.2f}) | "
            f"detalhe cam={part['detail_cam']:.2f} gb={part['detail_gb']:.2f} "
            f"ggc={part['detail_ggc']:.2f}"
        )
    print(f"   rho entre classes: cam {rho_cam:+.3f} | guided grad-cam {rho_ggc:+.3f}")

    rows = []
    titles = []
    for part in parts:
        rows.append(
            [
                image_rgb,
                map_cell(part["cam"], True, image_rgb, overlay_mode),
                map_cell(part["detail"], False, image_rgb, overlay_mode),
                map_cell(part["ggc"], False, image_rgb, overlay_mode),
            ]
        )
        titles.append(
            f"{labels[part['class_idx']]} ({part['prob']:.2%}) | "
            f"top5%={part['own']:.2f}"
        )

    return {
        "ident": ident,
        "image_rgb": image_rgb,
        "rows": rows,
        "titles": titles,
        "parts": parts,
        "rho_cam": rho_cam,
        "rho_ggc": rho_ggc,
        "boxes": boxes,
    }


def save_case_figure(result: dict, case: dict, model_name: str, args) -> Path:
    names = "-".join(
        labels_part.replace(" ", "_") for labels_part in case["classes"]
    )
    suffix = f"_{result['ident']}_{names}"
    fig = figure_grid(
        result["rows"],
        col_titles=COLUMNS,
        row_titles=result["titles"],
        suptitle=(
            f"{model_name} | Fase 6 | Guided Grad-CAM = Grad-CAM (14x14, upsample) "
            f"x Guided Backprop (224x224) | {case['note']}"
        ),
    )
    path = args.out / f"06_guided_gradcam_{model_name}{suffix}.png"
    fig.savefig(path, bbox_inches="tight")
    return path


def save_zoom(result: dict, case: dict, model_name: str, args) -> Path | None:
    """Recorte na box do objeto: e' onde a diferenca de resolucao aparece."""
    target = case["zoom"]
    if not target:
        return None
    boxes = result["boxes"].get(target, [])
    if not boxes:
        return None

    box = max(boxes, key=lambda candidate: candidate.area())
    # Janela quadrada em torno do objeto: com o recorte exato da box (que pode
    # ser 184x82) as quatro celulas viram tiras e a textura some na projecao.
    # Translada em vez de truncar, senao a janela perde o quadrado na borda.
    side = int(min(224, max(64, 1.28 * max(box.w, box.h))))
    center_x, center_y = (int(value) for value in box.center())
    x0 = min(max(0, center_x - side // 2), 224 - side)
    y0 = min(max(0, center_y - side // 2), 224 - side)
    x1, y1 = x0 + side, y0 + side

    def crop(cell):
        return cell[y0:y1, x0:x1]

    # O zoom mostra a classe que *pede* aquele objeto: recortar a box do
    # guarda-chuva e desenhar o mapa do `groom` seria outra figura.
    slot = case["objects"].index(target) if target in case["objects"] else 0
    part = result["parts"][slot]
    image_rgb = result["image_rgb"]
    cells = [
        crop(image_rgb),
        crop(gray_rgb(upsample(part["cam"], (224, 224)), high=DISPLAY_HIGH, gamma=DISPLAY_GAMMA)),
        crop(gray_rgb(part["detail"], high=DISPLAY_HIGH, gamma=DISPLAY_GAMMA)),
        crop(gray_rgb(part["ggc"], high=DISPLAY_HIGH, gamma=DISPLAY_GAMMA)),
    ]
    fig = figure_grid(
        [cells],
        col_titles=("entrada (zoom)", "grad-cam 14x14", "guided backprop", "guided grad-cam"),
        suptitle=(
            f"{model_name} | Fase 6 | zoom em '{target}': o CAM chega liso (14x14), "
            f"o Guided Grad-CAM traz a textura"
        ),
    )
    path = args.out / f"06_guided_gradcam_{model_name}_{result['ident']}_zoom.png"
    fig.savefig(path, bbox_inches="tight")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="vgg16")
    parser.add_argument("--layer", default=None, help="camada do Grad-CAM (conv3_3/conv4_3/conv5_3)")
    parser.add_argument("--images", nargs="+", default=None, metavar="TOKEN",
                        help="imagem customizada (id, nome do val2017 ou caminho)")
    parser.add_argument("--classes", default=None,
                        help="duas classes do ImageNet, ex.: 'Egyptian cat,beagle' (exige --images)")
    parser.add_argument("--overlay", action="store_true",
                        help="mostra o Grad-CAM como overlay jet em vez de cinza")
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    if args.classes and not args.images:
        parser.error("--classes so' faz sentido junto com --images")

    start = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"torch {torch.__version__} | dispositivo: {device}")

    model = load_model(args.model, device=device)
    labels = imagenet_labels()
    coco = CocoVal()

    try:
        cases = resolve_cases(labels, args, model, coco)
    except ValueError as error:
        print(f"erro: {error}")
        return 1

    # `--layer conv3_3` e' o nome do artigo; o torchvision usa `features.15`.
    layer = VGG16_LAYERS.get(args.layer, args.layer)

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"casos: {len(cases)} | camada: {layer or 'ultima conv'}")
    print()

    for case in cases:
        result = run_case(case, coco, model, labels, args.overlay, layer)
        path = save_case_figure(result, case, args.model, args)
        print(f"   salvo: {path.name}")
        zoom_path = save_zoom(result, case, args.model, args)
        if zoom_path is not None:
            print(f"   salvo: {zoom_path.name}")
        print()

    print("leitura: o detalhe (TV) do Guided Grad-CAM fica perto do Guided Backprop,")
    print("nao do CAM; e o top5% dentro da box certa continua alto. As duas metades")
    print("do criterio da Fase 6, medidas na mesma bateria.")
    print(f"\ntempo total: {time.perf_counter() - start:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
