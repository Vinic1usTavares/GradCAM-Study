"""Baixa e verifica pesos e dados do projeto.

Os dados NAO vao para o repositorio (`data/` esta no `.gitignore`). Cada
colaborador roda este script uma vez.

Uso:
    .\\.venv\\Scripts\\python.exe scripts\\00_download.py            # baixa o que faltar
    .\\.venv\\Scripts\\python.exe scripts\\00_download.py --check    # so verifica, sem rede
    .\\.venv\\Scripts\\python.exe scripts\\00_download.py --clean    # apaga os .zip apos extrair
    .\\.venv\\Scripts\\python.exe scripts\\00_download.py --weights  # so pesos
    .\\.venv\\Scripts\\python.exe scripts\\00_download.py --coco     # so COCO

Codigos de saida: 0 = tudo pronto, 1 = falta algo (util para CI/pre-commit).

O modo `--check` nao faz nenhuma requisicao de rede: usa apenas os tamanhos
esperados conhecidos e a estrutura extraida em disco.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tqdm import tqdm

from gradcam.models import COCO_DIR, WEIGHT_URLS, WEIGHTS_DIR

COCO_FILES = {
    "val2017.zip": "http://images.cocodataset.org/zips/val2017.zip",
    "annotations_trainval2017.zip": (
        "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    ),
}

# Tamanhos em bytes, obtidos uma vez dos servidores oficiais. Permitem detectar
# download incompleto sem tocar na rede.
EXPECTED_SIZES = {
    "vgg16-397923af.pth": 553_433_881,
    "alexnet-owt-7be5be79.pth": 244_408_911,
    "resnet18-f37072fd.pth": 46_830_571,
    "val2017.zip": 815_585_330,
    "annotations_trainval2017.zip": 252_907_541,
}

COCO_VAL_IMAGES = 5000
MARKER_DIR = COCO_DIR / ".extracted"


@dataclass
class Check:
    name: str
    ok: bool
    detail: str


# --------------------------------------------------------------------------- #
# verificacao
# --------------------------------------------------------------------------- #


def check_weights() -> list[Check]:
    checks = []
    for url in WEIGHT_URLS.values():
        path = WEIGHTS_DIR / url.rsplit("/", 1)[-1]
        checks.append(_check_file(path, "peso"))
    return checks


def check_zips(require: bool = False) -> list[Check]:
    if require:
        names = list(COCO_FILES)
    else:
        names = [n for n in COCO_FILES if (COCO_DIR / n).exists()]
    return [_check_file(COCO_DIR / name, "zip") for name in names]


def check_extracted() -> list[Check]:
    checks: list[Check] = []

    images = COCO_DIR / "val2017"
    count = sum(1 for _ in images.glob("*.jpg")) if images.is_dir() else 0
    checks.append(
        Check(
            "val2017/ (imagens)",
            count == COCO_VAL_IMAGES,
            f"{count}/{COCO_VAL_IMAGES} .jpg",
        )
    )

    annotations = COCO_DIR / "annotations" / "instances_val2017.json"
    if not annotations.exists():
        checks.append(Check("instances_val2017.json", False, "ausente"))
        return checks

    try:
        raw = json.loads(annotations.read_text(encoding="utf-8"))
        n_images = len(raw.get("images", []))
        n_ann = len(raw.get("annotations", []))
        ok = n_images == COCO_VAL_IMAGES and n_ann > 0
        detail = f"{n_images} imagens, {n_ann} anotacoes"
    except (json.JSONDecodeError, OSError) as exc:
        checks.append(Check("instances_val2017.json", False, f"ilegivel: {exc}"))
        return checks

    checks.append(Check("instances_val2017.json", ok, detail))

    # Consistencia: toda imagem anotada precisa existir em disco. Pega extracao
    # parcial, que e' o modo de falha mais comum.
    if ok and images.is_dir():
        missing = [
            img["file_name"]
            for img in raw["images"]
            if not (images / img["file_name"]).exists()
        ]
        checks.append(
            Check(
                "imagens x anotacoes",
                not missing,
                "consistente" if not missing else f"{len(missing)} sem arquivo",
            )
        )
    return checks


def _check_file(path: Path, kind: str) -> Check:
    if not path.exists():
        return Check(path.name, False, "ausente")
    size = path.stat().st_size
    expected = EXPECTED_SIZES.get(path.name)
    if expected is None:
        return Check(path.name, True, f"{size / 2**20:.0f} MiB")
    ok = size == expected
    detail = f"{size / 2**20:.0f} MiB" if ok else (
        f"{size} bytes, esperado {expected} (incompleto?)"
    )
    return Check(path.name, ok, detail)


def verify(verbose: bool = True) -> bool:
    checks = [*check_weights(), *check_zips(), *check_extracted()]
    if verbose:
        width = max(len(c.name) for c in checks)
        for check in checks:
            mark = "ok  " if check.ok else "FALTA"
            print(f"  [{mark}] {check.name:{width}s}  {check.detail}")
    return all(c.ok for c in checks)


# --------------------------------------------------------------------------- #
# download
# --------------------------------------------------------------------------- #


def remote_size(url: str) -> int | None:
    request = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return int(response.headers["Content-Length"])
    except Exception:
        return None


def download(url: str, dest: Path) -> Path:
    expected = EXPECTED_SIZES.get(dest.name) or remote_size(url)
    if dest.exists() and expected and dest.stat().st_size == expected:
        print(f"ja existe: {dest.name} ({expected / 2**20:.0f} MiB)")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"baixando: {url}")

    with urllib.request.urlopen(url, timeout=60) as response:
        total = int(response.headers.get("Content-Length", 0)) or expected or 0
        with open(tmp, "wb") as handle, tqdm(
            total=total, unit="B", unit_scale=True, unit_divisor=1024, desc=dest.name
        ) as bar:
            while chunk := response.read(1 << 20):
                handle.write(chunk)
                bar.update(len(chunk))

    tmp.replace(dest)
    return dest


def extract(zip_path: Path, target_dir: Path) -> None:
    """Extrai uma vez por arquivo (marcador proprio, nao compartilhado)."""
    MARKER_DIR.mkdir(parents=True, exist_ok=True)
    marker = MARKER_DIR / f"{zip_path.name}.done"
    if marker.exists():
        print(f"ja extraido: {zip_path.name}")
        return
    print(f"extraindo: {zip_path.name}")
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(target_dir)
    marker.touch()


# --------------------------------------------------------------------------- #


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", action="store_true", help="so os pesos")
    parser.add_argument("--coco", action="store_true", help="so o COCO val2017")
    parser.add_argument(
        "--check", action="store_true", help="apenas verificar, sem rede"
    )
    parser.add_argument(
        "--clean", action="store_true", help="apagar os .zip apos extrair"
    )
    args = parser.parse_args()

    if args.check:
        print(f"verificando em {COCO_DIR.parent}")
        ok = verify()
        print()
        print("tudo pronto" if ok else "FALTAM arquivos - rode sem --check")
        return 0 if ok else 1

    everything = not (args.weights or args.coco)

    if everything or args.weights:
        for url in WEIGHT_URLS.values():
            download(url, WEIGHTS_DIR / url.rsplit("/", 1)[-1])

    if everything or args.coco:
        for name, url in COCO_FILES.items():
            path = download(url, COCO_DIR / name)
            extract(path, COCO_DIR)
            if args.clean:
                path.unlink(missing_ok=True)
                print(f"removido: {name}")

    print()
    print("verificacao final:")
    ok = verify()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
