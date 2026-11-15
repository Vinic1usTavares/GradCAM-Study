"""Acesso ao COCO val2017 (imagens + anotacoes de instancia).

Sem `pycocotools`: o `instances_val2017.json` e lido direto com `json`, o que
evita uma dependencia dificil de compilar no Windows.

Layout esperado (ver plans/general_plan.md, secao 0):

    data/coco/
        val2017/                       # 5000 imagens .jpg
        annotations/instances_val2017.json
"""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from .models import COCO_DIR

IMAGES_DIR = COCO_DIR / "val2017"
ANNOTATIONS = COCO_DIR / "annotations" / "instances_val2017.json"


@dataclass(frozen=True)
class Box:
    """Bounding box no formato COCO: origem no canto superior esquerdo."""

    x: float
    y: float
    w: float
    h: float

    def contains(self, px: float, py: float) -> bool:
        return self.x <= px <= self.x + self.w and self.y <= py <= self.y + self.h

    def center(self) -> tuple[float, float]:
        return self.x + self.w / 2, self.y + self.h / 2

    def area(self) -> float:
        return self.w * self.h


class CocoVal:
    """Indice em memoria das anotacoes de instancia do val2017."""

    def __init__(
        self,
        images_dir: Path = IMAGES_DIR,
        annotations: Path = ANNOTATIONS,
    ) -> None:
        if not annotations.exists():
            raise FileNotFoundError(
                f"Anotacoes ausentes: {annotations}\n"
                "Baixe e extraia annotations_trainval2017.zip (ver README)."
            )
        self.images_dir = images_dir

        raw = json.loads(annotations.read_text(encoding="utf-8"))
        self.categories = {c["id"]: c["name"] for c in raw["categories"]}
        self.images = {img["id"]: img for img in raw["images"]}
        self._file_name_index: dict[str, int] | None = None

        # image_id -> category_name -> [Box]
        self.boxes: dict[int, dict[str, list[Box]]] = {i: {} for i in self.images}
        for ann in raw["annotations"]:
            if ann.get("iscrowd", 0):
                continue
            name = self.categories[ann["category_id"]]
            box = Box(*ann["bbox"])
            self.boxes[ann["image_id"]].setdefault(name, []).append(box)

    def __len__(self) -> int:
        return len(self.images)

    @property
    def image_ids(self) -> list[int]:
        return sorted(self.images)

    def image_path(self, image_id: int) -> Path:
        return self.images_dir / self.images[image_id]["file_name"]

    @property
    def file_name_index(self) -> dict[str, int]:
        """`file_name` -> `image_id`, construido na primeira chamada.

        O indice direto (`images`) e' por id; para aceitar `--images
        000000015278.jpg` e' preciso o caminho inverso.
        """
        if self._file_name_index is None:
            self._file_name_index = {
                meta["file_name"]: image_id for image_id, meta in self.images.items()
            }
        return self._file_name_index

    def resolve(self, token: str) -> tuple[int | None, Path]:
        """Resolve um token de `--images` em `(image_id | None, caminho)`.

        Aceita, nesta ordem:

        1. caminho existente — `data/coco/val2017/000000015278.jpg` ou uma foto
           qualquer. Se o nome estiver no COCO, o id e' recuperado (mantem a
           anotacao); se nao, `image_id` e' `None` (imagem solta, sem `gt`).
        2. id numerico — `15278`.
        3. nome de arquivo — `000000015278.jpg`.

        Token que nao resolve levanta `ValueError`: um `--images` silenciosamente
        ignorado produziria figura errada sem ninguem ver.
        """
        path = Path(token)
        if path.exists():
            return self.file_name_index.get(path.name), path

        if token.isdigit():
            image_id = int(token)
            if image_id not in self.images:
                raise ValueError(
                    f"id {image_id} nao esta no val2017 ({len(self.images)} imagens)"
                )
            return image_id, self.image_path(image_id)

        image_id = self.file_name_index.get(path.name)
        if image_id is None:
            raise ValueError(
                f"imagem nao encontrada: {token!r}. Use um id (15278), um nome "
                f"do val2017 (000000015278.jpg) ou um caminho de arquivo."
            )
        return image_id, self.image_path(image_id)

    def load_image(self, image_id: int) -> Image.Image:
        return Image.open(self.image_path(image_id)).convert("RGB")

    def object_names(self, image_id: int) -> list[str]:
        return sorted(self.boxes[image_id])

    def verbatim_categories(self, imagenet_labels: Sequence[str]) -> list[str]:
        """Categorias COCO que sao *exatamente* uma classe ImageNet.

        Sao 18 das 80 (ver plans/general_plan.md, secao 0). Comparar rotulo
        COCO com rotulo ImageNet fora desse conjunto nao faz sentido: `person`,
        `car` e `dog` do COCO nao tem contraparte 1:1 no ImageNet. Todo script
        que filtra por classe deve passar por aqui, senao a lista se duplica.
        """
        label_set = set(imagenet_labels)
        return sorted({name for name in self.categories.values() if name in label_set})

    def largest_object(self, image_id: int) -> tuple[str, Box] | None:
        """Categoria com a maior area anotada na imagem."""
        best: tuple[str, Box] | None = None
        for name, boxes in self.boxes[image_id].items():
            for box in boxes:
                if best is None or box.area() > best[1].area():
                    best = (name, box)
        return best

    def single_object_ids(
        self,
        min_side: int = 80,
        categories: Sequence[str] | None = None,
    ) -> list[int]:
        """Imagens com exatamente uma categoria anotada e objeto nao minusculo.

        E o subset mais limpo para o pointing game: nao ha ambiguidade sobre
        qual objeto a box de ground truth representa.

        `categories` restringe as categorias COCO aceitas.
        """
        allowed = set(categories) if categories is not None else None
        selected = []
        for image_id in self.image_ids:
            names = self.object_names(image_id)
            if len(names) != 1:
                continue
            if allowed is not None and names[0] not in allowed:
                continue
            box = self.boxes[image_id][names[0]][0]
            if min(box.w, box.h) < min_side:
                continue
            selected.append(image_id)
        return selected

    def subset(
        self,
        n: int,
        seed: int = 0,
        min_side: int = 80,
        categories: Sequence[str] | None = None,
    ) -> list[int]:
        """Amostra reproduzivel de imagens com um unico objeto."""
        pool = self.single_object_ids(min_side=min_side, categories=categories)
        rng = random.Random(seed)
        if n >= len(pool):
            return pool
        return sorted(rng.sample(pool, n))

    def stratified_subset(
        self,
        per_category: int = 2,
        seed: int = 0,
        min_side: int = 80,
        categories: Sequence[str] | None = None,
    ) -> list[int]:
        """Ate `per_category` imagens por categoria, para nao deixar a amostra
        ser dominada pelas categorias mais frequentes (zebra, pessoa, ...).
        """
        rng = random.Random(seed)
        by_category: dict[str, list[int]] = {}
        for image_id in self.single_object_ids(min_side=min_side, categories=categories):
            by_category.setdefault(self.object_names(image_id)[0], []).append(image_id)

        selected: list[int] = []
        for name in sorted(by_category):
            pool = by_category[name]
            selected.extend(rng.sample(pool, min(per_category, len(pool))))
        return sorted(selected)
