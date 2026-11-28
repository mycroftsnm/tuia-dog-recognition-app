"""Partición reproducible por componentes de fotos confirmadas."""

import csv
import hashlib
import random
from collections import defaultdict
from pathlib import Path, PurePosixPath

import polars as pl

SPLITS = ("train", "valid", "test")
MANIFEST_COLUMNS = ("split", "breed", "path", "filename")
REVIEW_COLUMNS = ("path_a", "path_b", "decision", "reason", "review", "sha256_a", "sha256_b")
CONFLICT_SCHEMA = {
    "group": pl.String,
    "breeds": pl.String,
    "paths": pl.String,
    "excluded_originals": pl.Int64,
    "decision": pl.String,
    "reason": pl.String,
}


def load_reviews(path: Path, dataset_dir: Path) -> dict[tuple[str, str], dict[str, str]]:
    """Indexa decisiones por rutas originales y verifica el contenido revisado."""
    reviews = {}
    hashes = {}
    with path.open(encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle)
        if rows.fieldnames != list(REVIEW_COLUMNS):
            raise ValueError(f"Columnas inesperadas en el registro de auditoría: {path}")
        for row in rows:
            a, b = pair = (row["path_a"], row["path_b"])
            if a >= b or pair in reviews or row["decision"] not in {"confirmed", "rejected"}:
                raise ValueError(f"Decisión de auditoría inválida o repetida: {pair}")
            if not row["reason"].strip():
                raise ValueError(f"Falta el motivo de la decisión para {pair}")
            for original, expected in zip(pair, (row["sha256_a"], row["sha256_b"]), strict=True):
                relative = PurePosixPath(original)
                if len(relative.parts) != 3 or relative.parts[0] not in SPLITS or ".." in relative.parts:
                    raise ValueError(f"La ruta no identifica una foto original: {original}")
                if original not in hashes:
                    hashes[original] = hashlib.sha256((dataset_dir / original).read_bytes()).hexdigest()
                if hashes[original] != expected:
                    raise ValueError(f"Cambió el contenido de {original}. Revisá la decisión de {pair}.")
            reviews[pair] = row
    return reviews


def photo_groups(
    quality: pl.DataFrame, reviews: dict[tuple[str, str], dict[str, str]]
) -> dict[str, list[str]]:
    """Forma componentes por ruta, incluyendo los vínculos del filtro de calidad."""
    parents = {path: path for path in quality["path"]}

    def find(path: str) -> str:
        while parents[path] != path:
            parents[path] = parents[parents[path]]
            path = parents[path]
        return path

    def union(a: str, b: str) -> None:
        if a not in parents or b not in parents:
            raise ValueError(f"El inventario no contiene ambos originales de {(a, b)}")
        first, last = sorted((find(a), find(b)))
        parents[last] = first

    for (a, b), row in sorted(reviews.items()):
        if row["decision"] == "confirmed":
            union(a, b)
    for path, duplicate_of in quality.select("path", "duplicate_of").drop_nulls().iter_rows():
        union(path, duplicate_of)

    groups = defaultdict(list)
    for path in sorted(parents):
        groups[find(path)].append(path)
    return dict(groups)


def partition_originals(
    quality: pl.DataFrame, groups: dict[str, list[str]], seed: int
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Asigna componentes completas intentando cubrir las cuotas de cada raza."""
    inventory = {row["path"]: row for row in quality.iter_rows(named=True)}
    by_breed = defaultdict(list)
    conflicts = []
    for group, paths in groups.items():
        breeds = sorted({inventory[path]["breed"] for path in paths})
        accepted = [path for path in paths if inventory[path]["quality_rule"] == "Pasa reglas visuales"]
        if len(breeds) > 1:
            conflicts.append({
                "group": group,
                "breeds": " | ".join(breeds),
                "paths": " | ".join(paths),
                "excluded_originals": len(accepted),
                "decision": "exclude",
                "reason": (
                    "La evidencia confirma la foto común. Las etiquetas de carpeta discrepan "
                    "y no hay una fuente de raza verificable para resolverlas."
                ),
            })
        elif accepted:
            by_breed[breeds[0]].append(accepted)

    records = []
    for breed, components in sorted(by_breed.items()):
        rng = random.Random(f"{seed}:{breed}")
        rng.shuffle(components)
        components.sort(key=len, reverse=True)
        total = sum(map(len, components))
        held_out = round(total * 0.15)
        remaining = {"train": total - 2 * held_out, "valid": held_out, "test": held_out}
        for component in components:
            # La mayor cuota restante también minimiza el exceso si el grupo no cabe entero.
            split = max(SPLITS, key=lambda name: remaining[name])
            remaining[split] -= len(component)
            records.extend(
                {"split": split, "breed": breed, "path": path, "filename": inventory[path]["filename"]}
                for path in component
            )

    final = pl.from_dicts(records, schema=dict.fromkeys(MANIFEST_COLUMNS, pl.String))
    crossing = crossing_groups(groups, dict(final.select("path", "split").iter_rows()))
    if crossing:
        raise ValueError(f"Hay grupos confirmados en más de un split: {crossing}")
    return final.sort(["split", "breed", "path"]), pl.from_dicts(conflicts, schema=CONFLICT_SCHEMA).sort("group")


def crossing_groups(groups: dict[str, list[str]], split_by_path: dict[str, str]) -> dict[str, list[str]]:
    crossings = {}
    for group, paths in groups.items():
        splits = {split_by_path[path] for path in paths if path in split_by_path}
        if len(splits) > 1:
            crossings[group] = sorted(splits)
    return crossings


def validate_manifests(
    originals: pl.DataFrame,
    manifests: dict[str, pl.DataFrame],
    gallery: pl.DataFrame,
    groups: dict[str, list[str]],
) -> None:
    """Verifica la partición publicada y el linaje de las filas de entrenamiento."""
    combined = pl.concat([manifests[split].select(MANIFEST_COLUMNS) for split in SPLITS])
    order = ["split", "breed", "path"]
    if (
        any(set(manifests[split]["split"]) - {split} for split in SPLITS)
        or combined["path"].n_unique() != combined.height
        or not combined.sort(order).equals(originals.select(MANIFEST_COLUMNS).sort(order))
    ):
        raise ValueError("train, valid y test deben particionar las rutas originales de all.csv.")
    crossings = crossing_groups(groups, dict(combined.select("path", "split").iter_rows()))
    if crossings:
        raise ValueError(f"Hay grupos confirmados en más de un split: {crossings}")

    train = manifests["train"]
    original_rows = gallery.filter(pl.col("origin") == "original").select(MANIFEST_COLUMNS)
    copies = gallery.filter(pl.col("origin").ne_missing("original"))
    if (
        set(gallery["split"]) != {"train"}
        or gallery["path"].n_unique() != gallery.height
        or not original_rows.sort(order).equals(train.select(MANIFEST_COLUMNS).sort(order))
        or not copies.join(
            train.select(pl.col("path").alias("origin"), "breed"),
            on=["origin", "breed"],
            how="anti",
        ).is_empty()
        or not copies.filter(~pl.col("path").str.starts_with("augmented/train/")).is_empty()
    ):
        raise ValueError("train_augmented debe contener los originales de train y copias con origen y raza en train.")
