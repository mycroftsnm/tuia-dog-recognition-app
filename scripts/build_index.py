"""Indexa la galería de entrenamiento de splits/train_augmented.csv (Etapa 1).

El manifiesto y las copias aumentadas se generan en 1-pre-processing.ipynb.

Requiere haber implementado SimilarityService.extract_embedding (o, si
EMBEDDING_MODEL != baseline, ClassifierService.extract_custom_embedding).
Si USE_PGVECTOR=true, la base de datos debe estar corriendo
(`docker compose up postgres -d`).

Uso:
    python scripts/build_index.py [--split train] [--limit 0]

Dentro de Docker (con `docker compose up -d` corriendo):
    docker compose exec backend python /scripts/build_index.py [--split train]
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if not (SRC / "lib").is_dir():
    # Dentro del contenedor del backend: src/ esta montado en /app (WORKDIR) y scripts/ en /scripts.
    ROOT = SRC = Path.cwd()
sys.path.insert(0, str(SRC))

# Usa la misma configuracion que el backend local (src/.env, con paths relativos a src/).
# Si src/.env no existe, se usan los defaults con paths relativos a la raiz del repo.
os.chdir(SRC if (SRC / ".env").is_file() else ROOT)

def load_gallery(
    manifest_path: Path, dataset_path: Path, limit: int
) -> dict[str, list[Path]]:
    """Agrupa las rutas seleccionadas por raza, en el orden del manifiesto."""
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"No existe {manifest_path}. Generá los manifiestos con 1-pre-processing.ipynb."
        )

    dataset_path = dataset_path.resolve()
    gallery: dict[str, list[Path]] = {}
    seen_paths: set[Path] = set()
    with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = csv.DictReader(handle)
        if not {"split", "breed", "path"}.issubset(rows.fieldnames or ()):
            raise ValueError(f"{manifest_path} debe contener las columnas split, breed y path.")

        for row in rows:
            if row["split"] != "train" or not row["breed"] or not row["path"]:
                raise ValueError(f"Fila {rows.line_num} inválida para train en {manifest_path}.")

            image_path = (dataset_path / row["path"]).resolve()
            if not image_path.is_relative_to(dataset_path):
                raise ValueError(f"Ruta fuera del dataset en {manifest_path}: {row['path']}")
            if image_path in seen_paths:
                raise ValueError(f"Ruta repetida en {manifest_path}: {row['path']}")
            seen_paths.add(image_path)

            breed_paths = gallery.setdefault(row["breed"], [])
            if limit and len(breed_paths) >= limit:
                continue
            if not image_path.is_file():
                raise FileNotFoundError(
                    f"Falta la imagen {image_path}. Revisá el dataset y las copias aumentadas."
                )
            breed_paths.append(image_path)

    if not gallery:
        raise ValueError(f"El manifiesto está vacío: {manifest_path}")
    return gallery


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split", default="train", choices=("train",),
        help="Partición a indexar desde train_augmented.csv.",
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Maximo de imagenes por raza (0 = todas)."
    )
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit debe ser mayor o igual a cero.")

    from lib.config import settings

    splits_dir = settings.splits_path
    if splits_dir == Path("splits") and "SPLITS_PATH" not in os.environ:
        # El manifiesto queda en la raíz del proyecto aunque src/.env cambie el directorio de trabajo.
        splits_dir = ROOT / "splits"
    try:
        gallery = load_gallery(splits_dir / "train_augmented.csv", settings.dataset_path, args.limit)
    except (OSError, ValueError, csv.Error) as error:
        parser.exit(1, f"{error}\n")

    from lib.bootstrap import build_classifier, build_similarity, build_store

    store = build_store(settings)
    similarity = build_similarity(settings, store)

    if settings.embedding_model == "baseline":
        extractor = similarity.extract_embedding
    else:
        classifier = build_classifier(settings)
        classifier.set_active_model(settings.embedding_model)
        extractor = classifier.extract_custom_embedding
    similarity.extract_embedding = extractor  # type: ignore[method-assign]

    total = 0
    for breed, image_paths in sorted(gallery.items()):
        for image_path in image_paths:
            similarity.index_image(
                str(image_path),
                breed,
                {"split": args.split, "model": settings.embedding_model},
            )
        total += len(image_paths)
        print(f"{breed}: {len(image_paths)} imágenes indexadas")

    print(f"Total: {total} embeddings almacenados (modelo: {settings.embedding_model})")


if __name__ == "__main__":
    main()
