"""Testes do seletor de imagens e de classes (`--images`, `--classes`)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gradcam.data import ANNOTATIONS, IMAGES_DIR, CocoVal
from gradcam.models import imagenet_labels, resolve_class

precisa_coco = pytest.mark.skipif(
    not ANNOTATIONS.exists() or not IMAGES_DIR.exists(),
    reason="COCO nao baixado",
)


@pytest.fixture(scope="module")
def coco() -> CocoVal:
    return CocoVal()


@precisa_coco
def test_resolve_accepts_id(coco):
    image_id, path = coco.resolve("15278")
    assert image_id == 15278
    assert path.exists()
    assert path.name == "000000015278.jpg"


@precisa_coco
def test_resolve_accepts_file_name(coco):
    image_id, path = coco.resolve("000000015278.jpg")
    assert image_id == 15278
    assert path == coco.image_path(15278)


@precisa_coco
def test_resolve_accepts_path_and_recovers_the_id(coco):
    image_id, path = coco.resolve(str(coco.image_path(15278)))
    assert image_id == 15278
    assert path.exists()


@precisa_coco
def test_resolve_accepts_outside_path_without_id(tmp_path):
    """Foto solta: resolve, mas sem anotacao — `gt` fica vazio."""
    from PIL import Image

    outside = tmp_path / "minha_foto.jpg"
    Image.new("RGB", (64, 64), color=(10, 20, 30)).save(outside)

    coco = CocoVal()
    image_id, path = coco.resolve(str(outside))
    assert image_id is None
    assert path == outside


@precisa_coco
def test_resolve_rejects_unknown_token(coco):
    with pytest.raises(ValueError, match="nao encontrada"):
        coco.resolve("nao_existe.jpg")


@precisa_coco
def test_resolve_rejects_unknown_id(coco):
    with pytest.raises(ValueError, match="nao esta no val2017"):
        coco.resolve("999999999")


@precisa_coco
def test_image_without_annotation_has_no_truth(coco):
    """48 imagens do val2017 nao tem anotacao nao-crowd: `truth` fica None."""
    sem_anotacao = next(i for i in coco.image_ids if not coco.object_names(i))
    image_id, path = coco.resolve(str(sem_anotacao))
    assert image_id == sem_anotacao
    assert path.exists()
    assert coco.object_names(image_id) == []


def test_resolve_class_exact_name():
    labels = imagenet_labels()
    assert labels[resolve_class("cup", labels)] == "cup"
    assert labels[resolve_class("vase", labels)] == "vase"


def test_resolve_class_is_case_insensitive():
    labels = imagenet_labels()
    assert resolve_class("CUP", labels) == resolve_class("cup", labels)


def test_resolve_class_accepts_index():
    labels = imagenet_labels()
    assert resolve_class(620, labels) == 620
    assert resolve_class("620", labels) == 620


def test_resolve_class_rejects_unknown():
    with pytest.raises(ValueError, match="desconhecida"):
        resolve_class("nao-existe-no-imagenet", imagenet_labels())


def test_resolve_class_lists_ambiguous_candidates():
    """`dog` casa com dezenas de racas: tem de recusar, nao pegar a primeira."""
    with pytest.raises(ValueError, match="ambigua"):
        resolve_class("dog", imagenet_labels())
