"""Búsqueda de posibles versiones de una foto entre las particiones."""

import hashlib
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from itertools import combinations
from pathlib import Path
from threading import local
from typing import NamedTuple

import cv2
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .dataset_partition import SPLITS


class AuditPhoto(NamedTuple):
    points: np.ndarray
    descriptors: np.ndarray | None
    views: np.ndarray
    sha256: str


audit_workers = local()
AUDIT_SIZE = 224


def audit_describe(path, dataset_dir):
    if not hasattr(audit_workers, "sift"):
        audit_workers.sift = cv2.SIFT_create(nfeatures=128)
    raw = (dataset_dir / path).read_bytes()
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"No se pudo leer {path}")
    gray = cv2.resize(image, (AUDIT_SIZE, AUDIT_SIZE), interpolation=cv2.INTER_AREA)
    points, descriptors = audit_workers.sift.detectAndCompute(gray, None)
    points = np.float32([point.pt for point in points])

    crop = round(AUDIT_SIZE * 0.8)
    margin = AUDIT_SIZE - crop
    offsets = (0, margin // 2, margin)
    crops = [gray[y:y + crop, x:x + crop] for y in offsets for x in offsets]
    views = np.float32([
        cv2.resize(view, (12, 12), interpolation=cv2.INTER_AREA).ravel()
        for view in [gray, *crops]
    ])
    views -= views.mean(axis=1, keepdims=True)
    views /= np.maximum(np.linalg.norm(views, axis=1, keepdims=True), 1)
    return AuditPhoto(points, descriptors, views, hashlib.sha256(raw).hexdigest())


def audit_global_candidates(database, queries):
    left, right = (np.stack([photo.views for photo in photos]) for photos in (database, queries))
    n_images, n_views, dimensions = left.shape
    all_left = left.reshape(-1, dimensions)
    with threadpool_limits(limits=4, user_api="blas"):
        for start in range(0, len(right), 64):
            batch = right[start:start + 64]
            cropped_left = (all_left @ batch[:, 0, :].T).reshape(n_images, n_views, -1)
            cropped_right = (left[:, 0, :] @ batch.reshape(-1, dimensions).T).reshape(
                n_images, -1, n_views
            )
            scores = np.maximum(cropped_left.max(axis=1), cropped_right.max(axis=2))
            for i, j in zip(*np.nonzero(scores >= 0.955)):
                yield (int(i), start + int(j)), float(scores[i, j])


def audit_sift_shortlist(database, queries):
    indexed = [photo.descriptors for photo in database if photo.descriptors is not None]
    if not indexed:
        return
    vectors = np.concatenate(indexed)
    owners = np.repeat(np.arange(len(database)), [len(photo.points) for photo in database])
    index = cv2.flann_Index(vectors, dict(algorithm=1, trees=4))
    for j, photo in enumerate(queries):
        if photo.descriptors is None:
            continue
        hits, distances = index.knnSearch(photo.descriptors, min(10, len(vectors)), params=dict(checks=40))
        votes = Counter()
        for neighbors in np.where(distances <= 240**2, owners[hits], -1):
            votes.update(set(neighbors) - {-1})
        for i, n_votes in votes.items():
            if n_votes >= 5:
                yield int(i), j


def audit_local_fit(left, right):
    if min(len(left.points), len(right.points)) < 2:
        return 0, 0., None
    matches = [
        first for first, second in cv2.BFMatcher().knnMatch(right.descriptors, left.descriptors, k=2)
        if first.distance < 0.78 * second.distance
    ]
    unique = {match.trainIdx: match for match in sorted(matches, key=lambda match: -match.distance)}
    if len(unique) < 4:
        return 0, 0., None
    src = np.float32([right.points[match.queryIdx] for match in unique.values()])
    dst = np.float32([left.points[match.trainIdx] for match in unique.values()])
    matrix, mask = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=3, maxIters=1000
    )
    if matrix is None:
        return 0, 0., None
    inliers = mask.ravel().astype(bool)
    area = min(cv2.contourArea(cv2.convexHull(points[inliers])) for points in (src, dst))
    return int(inliers.sum()), round(area / AUDIT_SIZE**2, 4), float(np.hypot(*matrix[:, 0]))


def audit_search(paths, dataset_dir: Path, seed: int):
    """Busca candidatos entre los originales de cada par de splits."""
    paths = {split: sorted(paths[split]) for split in SPLITS}
    all_paths = sorted(set().union(*paths.values()))
    previous_threads = cv2.getNumThreads()
    cv2.setNumThreads(1)
    cv2.setRNGSeed(seed)
    candidates = []
    checks = []
    try:
        with ThreadPoolExecutor(max_workers=6) as pool:
            photos = dict(zip(all_paths, pool.map(partial(audit_describe, dataset_dir=dataset_dir), all_paths)))
        for left_split, right_split in combinations(SPLITS, 2):
            left_paths, right_paths = paths[left_split], paths[right_split]
            left = [photos[path] for path in left_paths]
            right = [photos[path] for path in right_paths]
            if not left or not right:
                continue
            left_sha = defaultdict(list)
            for i, photo in enumerate(left):
                left_sha[photo.sha256].append(i)
            exact = {
                (i, j)
                for j, photo in enumerate(right)
                for i in left_sha.get(photo.sha256, ())
            }
            global_matches = dict(audit_global_candidates(left, right))
            local_shortlist = set(audit_sift_shortlist(left, right))
            visual = global_matches.keys() | local_shortlist
            cross = f"{left_split}–{right_split}"
            checks.append({
                "Cruce": cross,
                "Pares posibles": len(left) * len(right),
                "Rutas compartidas": len(set(left_paths) & set(right_paths)),
                "Pares SHA-256 idénticos": len(exact),
                "Preselección visual": len(visual),
            })
            for i, j in sorted(visual | exact):
                inliers, area, scale = audit_local_fit(left[i], right[j])
                if (inliers >= 6 and area >= 0.006) or (i, j) in global_matches or (i, j) in exact:
                    candidates.append({
                        "Cruce": cross,
                        "Ruta A": left_paths[i],
                        "Ruta B": right_paths[j],
                        "SHA-256 A": left[i].sha256,
                        "SHA-256 B": right[j].sha256,
                        "Puntos alineados": inliers,
                        "Área SIFT": area,
                        "Escala B → A": round(scale, 3) if scale is not None else None,
                        "Correlación global": round(global_matches.get((i, j), np.nan), 4),
                        "SHA-256 igual": (i, j) in exact,
                    })
    finally:
        cv2.setNumThreads(previous_threads)
    return (
        pd.DataFrame(candidates, columns=[
            "Cruce", "Ruta A", "Ruta B", "SHA-256 A", "SHA-256 B", "Puntos alineados",
            "Área SIFT", "Escala B → A", "Correlación global", "SHA-256 igual",
        ]),
        pd.DataFrame(checks),
    )


def classify_crossings(candidates, reviews):
    """Aplica decisiones previas y deja a la vista los pares que exigen revisión."""
    pending = []
    decisions = []
    for row in candidates.to_dict("records"):
        a, b = row["Ruta A"], row["Ruta B"]
        pair = tuple(sorted((a, b)))
        review = reviews.get(pair)
        if row["SHA-256 igual"] or (review and review["decision"] == "confirmed"):
            raise ValueError(f"La auditoría encontró una coincidencia confirmada entre splits: {pair}")
        if review is None:
            pending.append(row)
            decisions.append("Pendiente")
        else:
            decisions.append("Descartado")

    return candidates.assign(Decisión=decisions), pd.DataFrame(pending, columns=candidates.columns)


