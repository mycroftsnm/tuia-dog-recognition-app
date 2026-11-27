# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # TP2 — Etapa 2: entrenamiento, evaluación y comparación de modelos
#
# En esta notebook evaluamos la búsqueda de imágenes por similitud y comparamos dos
# clasificadores de 70 razas. El Modelo A es una ResNet18 preentrenada en ImageNet
# y ajustada al dataset. El Modelo B es una CNN diseñada por el equipo y entrenada
# desde cero.
# Para la búsqueda comparamos los embeddings de ambos modelos con los de una
# ResNet18 preentrenada que usamos como baseline.
#
# Para reproducir el trabajo, ejecutá la notebook desde la raíz del repositorio con
# el kernel de la `.venv`. Se requieren los manifiestos de `splits/` y las imágenes
# aumentadas de `data/dataset/augmented/`, generados por `1-pre-processing.py`.
# El procedimiento de auditoría y reparto está en [splits/README.md](splits/README.md).

# %% [markdown]
# ## Equipo
# - Alejandro Armas
# - Gabriel Soda

# %% [markdown]
# ## 1. Configuración del entorno
#
# En la notebook, la celda `%env` fija los directorios de modelos y salidas.
# `lib.config` lee la configuración y la salida indica las rutas usadas y el
# dispositivo de ejecución.

# %%
import json
import os
import sys
import time
from pathlib import Path

# Antes de importar torch/lib: caché de pesos y store JSON (la evaluación no usa Postgres).
os.environ.setdefault("USE_PGVECTOR", "false")
os.environ.setdefault("TORCH_HOME", str(Path.cwd() / "models" / "torch"))

import altair as alt
import cv2
import numpy as np
import polars as pl
import torch
from IPython.display import display

ROOT = Path.cwd()
if not (ROOT / "splits").is_dir():
    raise SystemExit("Ejecutá la notebook desde la raíz del repo: falta splits/.")
sys.path.insert(0, str(ROOT / "src"))

from lib.bootstrap import build_classifier, build_similarity
from lib.config import settings
from lib.evaluation.metrics import ndcg_at_k
from lib.schemas import Neighbor

alt.data_transformers.disable_max_rows()

SEED = 42
# 0 = usar todos los datos. Con un valor chico (ej. 5) se hace una prueba rápida de punta a punta.
LIMIT_PER_CLASS = int(os.getenv("LIMIT_PER_CLASS", 0))

COLORS = {
    "primary": "#4c78a8",
    "secondary": "#f58518",
    "negative": "#e45756",
    "positive": "#54a24b",
    "neutral": "#72b7b2",
    "gray": "#b3b3b3",
}
CATEGORICAL_COLORS = [
    COLORS["primary"], COLORS["secondary"], COLORS["negative"], COLORS["positive"],
    COLORS["neutral"], "#f2cf4a", "#9d7660", "#b279a2", "#ff9da6", "#4c78a8",
]

NOMBRES_ES = {
    "split": "partición",
    "breed": "raza",
    "path": "ruta",
    "filename": "archivo",
    "images": "imágenes",
    "metric": "métrica",
    "model": "modelo",
    "strategy": "estrategia",
    "epoch": "época",
    "duration_seconds": "duración (s)",
    "serie": "serie",
    "loss": "loss",
    "value": "valor",
    "accuracy": "exactitud",
    "precision": "precisión",
    "recall": "recall",
    "specificity": "especificidad",
    "f1": "F1",
    "umbral": "umbral",
    "cobertura": "cobertura",
    "rank": "puesto",
    "real": "real",
    "predicha": "predicha",
    "n": "imágenes",
    "proporción": "proporción",
    "support": "imágenes",
    "consulta": "consulta",
    "raza_real": "raza real",
    "raza_vecino": "raza del vecino",
    "score": "score",
    "train_loss": "loss train",
    "valid_loss": "loss valid",
    "train_accuracy": "exactitud train",
    "valid_accuracy": "exactitud valid",
    "valid_f1": "F1 valid",
    "ndcg@10": "NDCG@10",
    "precision@10": "precision@10",
    "precision@1": "precision@1",
}


def inline(df: pl.DataFrame) -> alt.InlineData:
    """Convierte un DataFrame de Polars a InlineData para Altair."""
    return alt.InlineData(values=df.to_dicts())


def mostrar(df: pl.DataFrame) -> None:
    """Muestra un DataFrame con los nombres de columnas en español."""
    display(df.rename({k: v for k, v in NOMBRES_ES.items() if k in df.columns}))


print("dataset:", settings.dataset_path)
print("splits: ", settings.splits_path)
print("modelos:", settings.model_path)
print("salidas:", settings.output_path)
print("dispositivo:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

# %% [markdown]
# ## 2. Dataset y splits
#
# Después del filtro de calidad, agrupamos las fotos duplicadas y casi duplicadas
# confirmadas por la auditoría. Cada grupo completo quedó en una sola partición
# para evitar que versiones de la misma foto llegaran a entrenamiento y evaluación.
# El reparto de originales apuntó a una proporción de 70/15/15 por raza para
# entrenamiento, validación y test.
#
# Aplicamos los aumentos a entrenamiento. Las 10.500 imágenes de `train_augmented`
# (150 por raza) sirven tanto para entrenar los clasificadores como para construir
# la galería de búsqueda. Validación y test tienen 1.371 imágenes originales cada
# uno. En validación seleccionamos los checkpoints y calibramos los umbrales de
# búsqueda. Test y el conjunto independiente se usan como consultas de evaluación
# y quedan fuera de la galería.

# %%
def load_split(name: str) -> pl.DataFrame:
    path = settings.splits_path / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Falta {path}. Ejecutá 1-pre-processing.py.")
    return pl.read_csv(path)


train_df = load_split("train_augmented")
valid_df = load_split("valid")
test_df = load_split("test")

if LIMIT_PER_CLASS:
    print(f"LIMIT_PER_CLASS={LIMIT_PER_CLASS}: submuestra para prueba rápida")
    train_df = train_df.group_by("breed", maintain_order=True).head(LIMIT_PER_CLASS)
    valid_df = valid_df.group_by("breed", maintain_order=True).head(LIMIT_PER_CLASS)
    test_df = test_df.group_by("breed", maintain_order=True).head(LIMIT_PER_CLASS)

splits_long = pl.concat(
    [
        train_df.select("breed").with_columns(pl.lit("train (galería)").alias("split")),
        valid_df.select("breed").with_columns(pl.lit("valid").alias("split")),
        test_df.select("breed").with_columns(pl.lit("test").alias("split")),
    ]
)
resumen_splits = (
    splits_long.group_by(["split", "breed"])
    .len()
    .group_by("split")
    .agg(
        pl.len().alias("razas"),
        pl.col("len").sum().alias("imágenes"),
        pl.col("len").min().alias("mín por raza"),
        pl.col("len").max().alias("máx por raza"),
    )
    .sort("split")
)
mostrar(resumen_splits)

# %% [markdown]
# ## 3. Etapa 1 — Evaluación de la búsqueda por similitud (baseline)
#
# Usamos la ResNet18 preentrenada de `SimilarityService`, con embeddings normalizados
# en L2. La galería se construye con `splits/train_augmented.csv`, el mismo manifiesto
# que usa la app para indexar. La tabla cruza las consultas de validación, test y el
# conjunto independiente con la galería. Los tres controles dieron cero coincidencias
# por ruta, por origen de las copias aumentadas y por SHA-256.
#
# Para reproducir la evaluación del baseline, ejecutá las secciones 1 a 3 con
# `LIMIT_PER_CLASS=0`.

# %%
from hashlib import sha256
from io import BytesIO
from textwrap import fill

from IPython.display import Image, Markdown
from matplotlib.figure import Figure

BASELINE_METRICS = ("cosine", "l2")
BASELINE_K = 10
gallery_df = train_df.sort("breed", maintain_order=True)
independent_df = load_split("independent_manifest").filter(pl.col("scene_type") == "un_perro")


def ruta_clave(value: str) -> str:
    if "://" in value:
        return value
    return os.path.normcase(str((settings.dataset_path / value).resolve()))


indexed_paths = set(map(ruta_clave, gallery_df["path"]))
gallery_origins = {
    ruta_clave(value) for value in gallery_df["origin"] if value and value != "original"
}
if len(indexed_paths) != gallery_df.height:
    raise ValueError("Hay rutas de galería repetidas en train_augmented.csv.")

gallery_identifiers = indexed_paths | gallery_origins
gallery_hashes = {sha256(Path(path).read_bytes()).digest() for path in gallery_identifiers}
split_checks = []
for split, queries in (("valid", valid_df), ("test", test_df), ("independiente", independent_df)):
    query_paths = set(map(ruta_clave, queries["path"]))
    query_origins = {
        ruta_clave(value)
        for column in ("origin", "source")
        if column in queries.columns
        for value in queries[column].drop_nulls()
        if value and value != "original"
    }
    shared_paths = query_paths & gallery_identifiers
    shared_origins = query_origins & gallery_identifiers
    identical_files = sum(
        sha256(Path(path).read_bytes()).digest() in gallery_hashes for path in query_paths
    )
    if len(query_paths) != queries.height or shared_paths or shared_origins or identical_files:
        raise ValueError(
            f"{split}: hay consultas repetidas o presentes en la galería. "
            f"Rutas compartidas: {sorted(shared_paths | shared_origins)}. "
            f"Archivos idénticos: {identical_files}."
        )
    split_checks.append(
        {
            "Partición": split,
            "Imágenes": queries.height,
            "Razas": queries["breed"].n_unique(),
            "Rutas en galería": len(shared_paths),
            "Orígenes en galería": len(shared_origins),
            "Archivos idénticos": identical_files,
        }
    )

print(f"LIMIT_PER_CLASS={LIMIT_PER_CLASS} · ResNet18 · {'CUDA' if torch.cuda.is_available() else 'CPU'}")
display(pl.DataFrame([{
    "Galería": "splits/train_augmented.csv",
    "Imágenes": gallery_df.height,
    "Razas": gallery_df["breed"].n_unique(),
}]))
display(pl.DataFrame(split_checks))

# %%
baseline = build_similarity(settings.model_copy(update={"embedding_model": "baseline"}), store=None)
gallery_paths = gallery_df["path"].to_list()
gallery_breeds = gallery_df["breed"].to_list()
gallery_counts = dict(gallery_df.group_by("breed").len().iter_rows())


def baseline_image(relative_path: str) -> np.ndarray:
    path = (settings.dataset_path / relative_path).resolve()
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"No se pudo leer {path}")
    return image


def embed_baseline(rows: pl.DataFrame, label: str, batch_size: int = 32) -> np.ndarray:
    paths = rows["path"].to_list()
    embeddings = []
    for start in range(0, len(paths), batch_size):
        images = [baseline_image(path) for path in paths[start : start + batch_size]]
        embeddings.extend(baseline.extract_embeddings(images))
    matrix = np.asarray(embeddings, dtype=np.float32)
    print(f"{label}: {len(matrix):,} embeddings de {matrix.shape[1]} dimensiones.", flush=True)
    return matrix


gallery_emb = embed_baseline(gallery_df, "Galería")
gallery_matrix = gallery_emb.astype(np.float64)
gallery_norm2 = np.einsum("ij,ij->i", gallery_matrix, gallery_matrix)


def retrieve_baseline(query_emb: np.ndarray) -> dict[str, list[list[Neighbor]]]:
    """Busca en la galería completa por lotes y puntúa el top con el servicio."""
    rankings = {metric: [] for metric in BASELINE_METRICS}
    for start in range(0, len(query_emb), 128):
        query_batch = query_emb[start : start + 128]
        query_matrix = query_batch.astype(np.float64)
        query_norm2 = np.einsum("ij,ij->i", query_matrix, query_matrix)[:, None]
        dot = query_matrix @ gallery_matrix.T
        cosine = dot / np.sqrt(query_norm2 * gallery_norm2)
        distance2 = query_norm2 + gallery_norm2 - 2 * dot

        for metric, order_by in (("cosine", -cosine), ("l2", distance2)):
            baseline.similarity_metric = metric
            columns = np.argsort(order_by, axis=1, kind="stable")[:, :BASELINE_K]
            for embedding, indices in zip(query_batch, columns, strict=True):
                neighbors = [
                    Neighbor(
                        path=gallery_paths[column],
                        breed=gallery_breeds[column],
                        score=baseline.similarity(embedding, gallery_emb[column]),
                    )
                    for column in indices
                ]
                neighbors.sort(key=lambda neighbor: neighbor.score, reverse=True)
                rankings[metric].append(neighbors)
    return rankings


def predict_baseline(rankings: list[list[Neighbor]], k: int, threshold: float) -> np.ndarray:
    baseline.similarity_threshold = threshold
    return np.array([
        baseline.predict_breed_from_neighbors(neighbors[:k])[0] for neighbors in rankings
    ])


def rank_metrics(rankings: list[list[Neighbor]], breeds: np.ndarray, k: int) -> np.ndarray:
    values = []
    for neighbors, breed in zip(rankings, breeds, strict=True):
        relevance = [float(neighbor.breed == breed) for neighbor in neighbors[:k]]
        ndcg = ndcg_at_k(relevance, k=k, n_relevant=gallery_counts.get(breed, 0))
        values.append((ndcg, sum(relevance) / k))
    return np.asarray(values)

# %% [markdown]
# ### Calibración en validación
#
# El servicio pondera el voto de cada vecino por su similitud. Probamos umbrales
# de 0 a 1 en pasos de 0,001 sobre el score del primer vecino. Si queda por debajo
# del umbral, la predicción es `unknown`. Elegimos el umbral que maximiza el F1 de
# aciertos aceptados, la media armónica entre la proporción de aciertos sobre las
# consultas aceptadas y la proporción de aciertos sobre todas las consultas.
# En caso de empate tomamos el umbral más bajo. Como validación contiene razas
# conocidas, esta calibración mide la abstención ante posibles errores del voto.
#
# Coseno usa el producto escalar normalizado y L2 usa `1 / (1 + distancia euclídea)`.
# En ambas escalas un score más alto indica un vecino más similar. La accuracy
# cuenta `unknown` como error. NDCG y precision miden el ranking para todas las consultas.
#
# En las 1.371 consultas de validación, pasar de K=5 a K=10 subió la accuracy
# del voto de 88,4026 % a 88,7673 % y el F1 de aciertos aceptados de 0,8843 a
# 0,8880 con ambas métricas. La precision bajó de 0,8397 a 0,8191 y el NDCG,
# de 0,8421 a 0,8269. Los dos valores de K dieron los mismos umbrales,
# **0,674 para coseno** y **0,554 para L2**. Cada combinación de métrica y K rechazó una consulta
# (0,0729 %) y la accuracy coincidió con la del voto sin rechazo.

# %%
valid_rankings = retrieve_baseline(embed_baseline(valid_df, "Validación"))
valid_breeds = valid_df["breed"].to_numpy()
threshold_grid = np.arange(1001) / 1000
calibration_rows = []
baseline_thresholds = {}

for metric, rankings in valid_rankings.items():
    best_scores = np.array([neighbors[0].score for neighbors in rankings])
    accepted = best_scores[None, :] >= threshold_grid[:, None]
    n_accepted = accepted.sum(axis=1)
    for k in (5, BASELINE_K):
        unfiltered = predict_baseline(rankings, k, -np.inf)
        hits = unfiltered == valid_breeds
        n_hits = (accepted & hits).sum(axis=1)
        f1 = 2 * n_hits / (len(valid_breeds) + n_accepted)
        chosen = int(f1.argmax())
        threshold = float(threshold_grid[chosen])
        predicted = predict_baseline(rankings, k, threshold)
        if k == BASELINE_K:
            baseline_thresholds[metric] = threshold

        ndcg, precision = rank_metrics(rankings, valid_breeds, k).mean(axis=0)
        covered = predicted != "unknown"
        calibration_rows.append({
            "Métrica": metric,
            "K": k,
            "Umbral": threshold,
            "NDCG@K": ndcg,
            "precision@K": precision,
            "Accuracy sin rechazo (%)": 100 * hits.mean(),
            "Accuracy (%)": 100 * (predicted == valid_breeds).mean(),
            "Aciertos entre aceptadas (%)": 100 * (predicted[covered] == valid_breeds[covered]).mean(),
            "Unknown (%)": 100 * (~covered).mean(),
            "F1": f1[chosen],
        })

with pl.Config(tbl_cols=-1, float_precision=4):
    display(pl.DataFrame(calibration_rows))

print(f"Umbrales fijados para K={BASELINE_K}: " + ", ".join(
    f"{metric}={threshold:.3f}" for metric, threshold in baseline_thresholds.items()
))

# %% [markdown]
# ### Evaluación final con K=10
#
# Aplicamos los umbrales elegidos en validación. En NDCG@10 la relevancia es binaria
# por raza. El ranking ideal se calcula con la cantidad de imágenes de la raza en
# toda la galería. Precision@10 es la fracción de vecinos de la raza real entre los
# diez recuperados.
#
# Coseno y L2 dieron el mismo orden de vecinos y las mismas predicciones en las
# 1.371 consultas de test y las 68 independientes. Ambos aceptaron todas las
# consultas. Del test al independiente, la accuracy del voto bajó de 89,3508 % a
# 58,8235 %, NDCG@10 de 0,8374 a 0,5156 y precision@10 de 0,8301 a 0,5059.

# %%
baseline_results = {}
baseline_rows = []
for split, queries in (("test", test_df), ("independiente", independent_df)):
    breeds = queries["breed"].to_numpy()
    rankings_by_metric = retrieve_baseline(embed_baseline(queries, split))
    for metric, rankings in rankings_by_metric.items():
        threshold = baseline_thresholds[metric]
        predicted = predict_baseline(rankings, BASELINE_K, threshold)
        ndcg, precision = rank_metrics(rankings, breeds, BASELINE_K).T
        baseline_results[split, metric] = (rankings, predicted, ndcg)
        baseline_rows.append({
            "Conjunto": split,
            "Métrica": metric,
            "Consultas": len(breeds),
            "Umbral": threshold,
            "NDCG@10": ndcg.mean(),
            "precision@10": precision.mean(),
            "Accuracy de raza (%)": 100 * (predicted == breeds).mean(),
            "Unknown (%)": 100 * (predicted == "unknown").mean(),
        })

with pl.Config(tbl_cols=-1, float_precision=4):
    display(pl.DataFrame(baseline_rows))

for split in ("test", "independiente"):
    cosine_ranks, cosine_pred, _ = baseline_results[split, "cosine"]
    l2_ranks, l2_pred, _ = baseline_results[split, "l2"]
    changed_ranks = sum(
        [neighbor.path for neighbor in cosine] != [neighbor.path for neighbor in l2]
        for cosine, l2 in zip(cosine_ranks, l2_ranks, strict=True)
    )
    print(f"{split}: {changed_ranks}/{len(cosine_ranks)} Top-10 con distinto orden, "
          f"{np.count_nonzero(cosine_pred != l2_pred)} predicciones distintas.")

# %%
def display_baseline_figure(figure: Figure, title: str) -> None:
    with BytesIO() as buffer:
        figure.savefig(buffer, format="png", dpi=110)
        display(Image(data=buffer.getvalue(), format="png", alt=title))


distribution = Figure(figsize=(10, 6), layout="constrained")
distribution.suptitle("Distribución de NDCG@10")
for axis, ((split, metric), (_, _, ndcg)) in zip(
    distribution.subplots(2, 2, sharex=True).flat, baseline_results.items(), strict=True
):
    axis.hist(ndcg, bins=np.linspace(0, 1, 11), color=COLORS["primary"], edgecolor="white")
    axis.set(title=f"{split} · {metric} (n={len(ndcg)})", xlim=(0, 1))
for axis in distribution.axes[::2]:
    axis.set_ylabel("Consultas")
for axis in distribution.axes[2:]:
    axis.set_xlabel("NDCG@10")
display_baseline_figure(distribution, "Distribución de NDCG@10 por conjunto y métrica")

# %% [markdown]
# ### Dos consultas y sus vecinos con coseno
#
# Los vecinos se leen de izquierda a derecha en cada fila. Las etiquetas verdes indican
# coincidencia con la raza real.

# %%
def show_baseline_example(split: str, queries: pl.DataFrame, index: int) -> None:
    rankings, predicted, ndcgs = baseline_results[split, "cosine"]
    query = queries.row(index, named=True)
    title = (
        f"{split.capitalize()} · {query['filename']} · cosine · NDCG@10 {ndcgs[index]:.3f}"
    )
    figure = Figure(figsize=(14, 5.4), layout="constrained")
    figure.suptitle(title, fontsize=16)
    grid = figure.add_gridspec(2, 6, width_ratios=[1.45, 1, 1, 1, 1, 1])
    axis = figure.add_subplot(grid[:, 0])
    axis.imshow(cv2.cvtColor(baseline_image(query["path"]), cv2.COLOR_BGR2RGB))
    axis.set_title(f"Consulta\nReal: {query['breed']}\nPredicción: {predicted[index]}", fontsize=12)
    axis.set_axis_off()
    for position, neighbor in enumerate(rankings[index]):
        axis = figure.add_subplot(grid[position // 5, 1 + position % 5])
        axis.imshow(cv2.cvtColor(baseline_image(neighbor.path), cv2.COLOR_BGR2RGB))
        label = fill(neighbor.breed, width=21)
        axis.set_title(
            f"#{position + 1} · {neighbor.score:.3f}\n{label}", fontsize=10,
            color="#176140" if neighbor.breed == query["breed"] else "#272727",
        )
        axis.set_axis_off()
    display_baseline_figure(figure, title)
    print(f"Consulta: {query['path']}")
    if split == "independiente":
        display(Markdown(
            f"Foto de la consulta de {query['author']} · "
            f"[{query['license']}]({query['license_url']}) · [Fuente]({query['source']})"
        ))


for success, datasets in (
    (True, (("test", test_df), ("independiente", independent_df))),
    (False, (("independiente", independent_df), ("test", test_df))),
):
    for split, queries in datasets:
        _, predicted, ndcg = baseline_results[split, "cosine"]
        candidates = np.flatnonzero((predicted == queries["breed"].to_numpy()) == success)
        if len(candidates):
            index = int(candidates[ndcg[candidates].argmax()] if success else candidates[0])
            show_baseline_example(split, queries, index)
            break

# %% [markdown]
# ### Qué muestran las consultas
#
# - **Afghan en test.** Los diez vecinos de `train/Afghan/023.jpg` son Afghan.
#   El voto acierta y el NDCG@10 es 1,000. La carpeta `train/` indica dónde estaba
#   el archivo en el dataset de origen. El manifiesto asigna esta foto a test.
# - **Poodle en el independiente.** En `Standard_black_Poodle.jpg`, el perro
#   negro está sentado sobre una cama con el hocico en sombra. El primer vecino es
#   Irish Spaniel (0,759), que también gana el voto. El top 10 contiene cero
#   Poodle y el NDCG@10 es 0,000.

# %% [markdown]
# ## 6. Modelo A (obligatorio): fine-tuning de ResNet
#
# Partimos de una ResNet18 preentrenada en ImageNet y ajustamos todas sus capas con
# `train_augmented`. La cabeza de clasificación combina dropout de `0.2` y
# `Linear(512, 70)`. Para extraer embeddings usamos los 512 componentes del pooling
# global, antes de la cabeza. La entrada es de 224×224 con normalización de ImageNet.
#
# Entrenamos con AdamW, `CosineAnnealingLR` y *label smoothing*. En la tabla de
# entrenamiento se muestran las métricas de cada época. Guardamos como checkpoint
# el modelo que obtuvo el mayor F1 macro de validación.

# %%
classifier = build_classifier(settings)
classifier.set_active_model("resnet18_finetuned")
classifier.train_classifier()

# %%
def load_history(model_name: str) -> pl.DataFrame | None:
    """Historial de `train_classifier`, o None si se reutilizó el checkpoint y no hay JSON."""
    path = settings.output_path / f"history_{model_name}.json"
    if not path.is_file():
        print(f"Sin historial para {model_name}: se reutilizó el checkpoint existente.")
        return None
    return pl.DataFrame(json.loads(path.read_text(encoding="utf-8"))["epochs"])


history = load_history("resnet18_finetuned")
if history is not None:
    mostrar(history)

# %%
metrics_a = classifier.evaluate_classifier()
evaluation_a = classifier.last_evaluation
(settings.output_path / "eval_resnet18_finetuned.json").write_text(
    json.dumps(evaluation_a, ensure_ascii=False, indent=2), encoding="utf-8"
)
mostrar(
    pl.DataFrame(
        [{"métrica": key, "valor": round(float(value), 4)} for key, value in metrics_a.items()]
    )
)

# %% [markdown]
# ### Curvas de entrenamiento y matriz de confusión (Modelo A)
#
# La ResNet completó 12 épocas. El F1 macro de validación subió de 77,75 % en la primera
# a un máximo de **88,62 % en la época 10**, que es la del checkpoint evaluado.
# Entre las épocas 10 y 12, la pérdida de entrenamiento siguió bajando de 0,826 a 0,815
# y el F1 macro de validación terminó en 88,36 %. En la última época, la accuracy fue
# 99,77 % en entrenamiento y 88,26 % en validación. La brecha y el descenso del F1
# después del máximo son indicios de sobreajuste al final del entrenamiento.
#
# En test, la confusión más frecuente fue predecir Shih-Tzu en 16 fotos de Lhasa.
# Lhasa fue además la raza con menor exactitud del Modelo A, con 9 aciertos entre
# 27 fotos.

# %%
def plot_training(history: pl.DataFrame, model: str) -> None:
    """Curvas de loss y accuracy. Train siempre azul, validación siempre naranja."""
    color = alt.Color(
        "split:N",
        title=None,
        scale=alt.Scale(
            domain=["train", "valid"],
            range=[COLORS["primary"], COLORS["secondary"]],
        ),
    )

    def long_frame(columns: list[str], value_name: str) -> pl.DataFrame:
        return (
            history.select(["epoch", *columns])
            .unpivot(index="epoch", variable_name="serie", value_name=value_name)
            .with_columns(
                pl.when(pl.col("serie").str.starts_with("train"))
                .then(pl.lit("train"))
                .otherwise(pl.lit("valid"))
                .alias("split")
            )
        )

    loss_chart = (
        alt.Chart(inline(long_frame(["train_loss", "valid_loss"], "loss")))
        .mark_line(point=True)
        .encode(
            x=alt.X("epoch:Q", title="Época"),
            y=alt.Y("loss:Q", title="Loss"),
            color=color,
            tooltip=["epoch:Q", "split:N", alt.Tooltip("loss:Q", format=".4f")],
        )
        .properties(width=520, height=300, title=f"{model} — loss: train vs validación")
    )

    accuracy_chart = (
        alt.Chart(inline(long_frame(["train_accuracy", "valid_accuracy"], "accuracy")))
        .mark_line(point=True)
        .encode(
            x=alt.X("epoch:Q", title="Época"),
            y=alt.Y("accuracy:Q", title="Accuracy", axis=alt.Axis(format="%")),
            color=color,
            tooltip=["epoch:Q", "split:N", alt.Tooltip("accuracy:Q", format=".3f")],
        )
        .properties(width=520, height=300, title=f"{model} — accuracy: train vs validación")
    )
    display(loss_chart | accuracy_chart)


if history is not None:
    plot_training(history, "Modelo A")

# %%
def plot_confusion(evaluation: dict, model: str, top_pairs: int = 15, worst: int = 10) -> None:
    """Matriz de confusión normalizada + pares más confundidos + peores razas."""
    labels = evaluation["class_names"]
    matrix = np.asarray(evaluation["confusion_matrix"], dtype=float)
    normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    long = pl.DataFrame(
        {
            "real": np.repeat(labels, len(labels)),
            "predicha": np.tile(labels, len(labels)),
            "n": matrix.reshape(-1).astype(int),
            "proporción": np.round(normalized.reshape(-1), 4),
        }
    ).filter(pl.col("n") > 0)
    chart = (
        alt.Chart(inline(long))
        .mark_rect()
        .encode(
            x=alt.X("predicha:N", sort=labels, title=None, axis=alt.Axis(labelAngle=-90, labelFontSize=6)),
            y=alt.Y("real:N", sort=labels, title=None, axis=alt.Axis(labelFontSize=6)),
            color=alt.Color(
                "proporción:Q",
                title="Proporción",
                scale=alt.Scale(range=["#ffffff", COLORS["primary"]]),
            ),
            tooltip=[
                alt.Tooltip("real:N", title="Real"),
                alt.Tooltip("predicha:N", title="Predicha"),
                alt.Tooltip("n:Q", title="Imágenes"),
                alt.Tooltip("proporción:Q", format=".1%"),
            ],
        )
        .properties(width=760, height=680, title=f"Matriz de confusión normalizada ({model})")
    )
    display(chart)

    off_diagonal = matrix.copy()
    np.fill_diagonal(off_diagonal, 0)
    pairs = [
        {"real": labels[row], "predicha": labels[column], "n": int(off_diagonal[row, column])}
        for row, column in zip(*np.nonzero(off_diagonal))
    ]
    if pairs:
        mostrar(pl.DataFrame(pairs).sort("n", descending=True).head(top_pairs))
    mostrar(pl.DataFrame(evaluation["per_class"]).sort("accuracy").head(worst))


plot_confusion(evaluation_a, "Modelo A")

# %% [markdown]
# ## 7. Modelo B (obligatorio): CNN propia
#
# Entrenamos la CNN desde cero y actualizamos todas sus capas. Recibe imágenes de
# 224×224 con normalización de ImageNet y clasifica las mismas 70 razas.
#
# | Componente | Configuración |
# |---|---|
# | Bloques convolucionales | 4 bloques con dos capas Conv-BatchNorm-ReLU y un MaxPool por bloque. Los canales pasan de 32 a 64, 128 y 256. |
# | Pooling global promedio | Resume cada uno de los 256 canales en un valor. |
# | Embedding | `Linear(256, 512)` seguida de ReLU. |
# | Cabeza de clasificación | Dropout de `0.2` antes de `Linear(512, 70)`. |
#
# La arquitectura está en `src/lib/models/modelo_b_cnn.py`.

# %%
classifier.set_active_model("cnn_custom")
classifier.train_classifier()

# %%
metrics_b = classifier.evaluate_classifier()
evaluation_b = classifier.last_evaluation
(settings.output_path / "eval_cnn_custom.json").write_text(
    json.dumps(evaluation_b, ensure_ascii=False, indent=2), encoding="utf-8"
)
mostrar(
    pl.DataFrame(
        [{"métrica": key, "valor": round(float(value), 4)} for key, value in metrics_b.items()]
    )
)

# %% [markdown]
# ### Curvas de entrenamiento y matriz de confusión (Modelo B)
#
# La CNN completó 12 épocas. Su F1 macro de validación subió de 4,18 % en la primera
# a un máximo de **34,83 % en la época 12**. En el mismo período la accuracy de
# entrenamiento subió de 6,25 % a 41,55 % y la de validación de 7,88 % a 37,20 %.
# La pérdida de validación repuntó en las épocas 6 y 7 y volvió a bajar hasta el
# final. Los avances en las dos particiones indican que seguía aprendiendo en la
# última época.
#
# En test, la confusión más frecuente fue predecir Boston Terrier en 10 fotos de
# Bulldog. La tabla por raza muestra cero aciertos en Borzoi, Corgi y Great Dane.

# %%
history_b = load_history("cnn_custom")
if history_b is not None:
    mostrar(history_b)
    plot_training(history_b, "Modelo B")

# %%
plot_confusion(evaluation_b, "Modelo B")

# %% [markdown]
# ## 8. Evaluación sobre el conjunto independiente
#
# Usamos las 68 fotos etiquetadas `un_perro` en el manifiesto independiente.
# Cada consulta es una foto completa con la raza del perro como referencia.
# Las métricas macro dan el mismo peso a cada una de las **20 razas presentes en
# las etiquetas reales**. La accuracy es la proporción de fotos clasificadas correctamente.
#
# - **ResNet18 (Modelo A).** La accuracy bajó de 90,30 % en test a 50,00 % en el
#   independiente (34/68).
# - **CNN (Modelo B).** La accuracy bajó de 37,71 % en test a 11,76 % en el
#   independiente (8/68).
#
# Las diferencias de fondo, iluminación y encuadre son una posible explicación.
# En la foto del Poodle negro de la sección 3, gran parte de la imagen es la cama y
# el hocico queda en sombra. Los dos clasificadores también se equivocaron en esa
# foto según sus predicciones guardadas. Las diferencias en las razas representadas
# pueden contribuir a la caída. Con estas 68 fotos, queda abierta la contribución
# de cada factor.

# %%
independent = load_split("independent_manifest")
independent_eval = (
    independent.filter(pl.col("scene_type") == "un_perro")
    .select(["breed", "path"])
    .with_columns(pl.lit("independiente").alias("split"))
    .select(["split", "breed", "path"])
)
independent_eval.write_csv(settings.splits_path / "independent_eval.csv")
mostrar(independent_eval.group_by("breed").len().sort("breed"))

# %%
independent_metrics = {}
for model_name in ("resnet18_finetuned", "cnn_custom"):
    classifier.set_active_model(model_name)
    independent_metrics[model_name] = classifier.evaluate_classifier("independent_eval")
    (settings.output_path / f"eval_independent_{model_name}.json").write_text(
        json.dumps(classifier.last_evaluation, ensure_ascii=False, indent=2), encoding="utf-8"
    )

with pl.Config(tbl_cols=-1, float_precision=4):
    display(
        pl.DataFrame(
            [{"Modelo": name, **metrics} for name, metrics in independent_metrics.items()]
        )
    )

# %% [markdown]
# ## 9. Embeddings de los modelos entrenados
#
# Comparamos el baseline y los modelos A y B como extractores con la misma galería,
# las mismas consultas y similitud coseno. Usamos K=10 y compartimos el umbral de
# 0,674 calibrado con los embeddings del baseline en validación. La tabla separa
# las métricas del ranking (NDCG@10 y precision@10) de las del voto (accuracy y F1 macro).

# %%
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score, silhouette_score


def embed_rows(extractor, rows: pl.DataFrame, label: str, batch_size: int = 32) -> np.ndarray:
    """Embeddings L2-normalizados de un manifiesto, en lotes."""
    paths = rows["path"].to_list()
    embeddings = []
    for start in range(0, len(paths), batch_size):
        images = [baseline_image(path) for path in paths[start : start + batch_size]]
        embeddings.extend(extractor(images))
        print(f"  {label}: {min(start + batch_size, len(paths))}/{len(paths)}", end="\r")
    print(f"  {label}: {len(paths)} embeddings")
    return np.asarray(embeddings, dtype=np.float32)


def retrieval_metrics(
    query_emb: np.ndarray,
    query_breeds: list[str],
    gallery_matrix: np.ndarray,
    *,
    k: int,
    threshold: float,
) -> tuple[float, float, float, list[str]]:
    """Calcula las métricas del ranking coseno y del voto con el umbral indicado."""
    voter = build_similarity(
        settings.model_copy(update={
            "similarity_metric": "cosine", "top_k": k, "similarity_threshold": threshold,
        }),
        store=None,
    )
    cosine = np.clip(query_emb @ gallery_matrix.T, -1.0, 1.0)
    order = np.argsort(-cosine, axis=1)[:, :k]
    ndcgs, precisions, predicted = [], [], []
    for index, breed in enumerate(query_breeds):
        neighbors = [
            Neighbor(
                path=gallery_paths[column],
                breed=gallery_breeds[column],
                score=float(cosine[index, column]),
            )
            for column in order[index]
        ]
        predicted.append(voter.predict_breed_from_neighbors(neighbors)[0])
        relevance = [float(gallery_breeds[column] == breed) for column in order[index]]
        ndcgs.append(ndcg_at_k(relevance, k=k, n_relevant=gallery_counts.get(breed, 0)))
        precisions.append(float(np.mean(relevance)))
    accuracy = float(np.mean(np.asarray(predicted) == np.asarray(query_breeds)))
    return float(np.mean(ndcgs)), float(np.mean(precisions)), accuracy, predicted


# %%
baseline_test = embed_baseline(test_df, "Test baseline")
baseline_independent = embed_baseline(independent_eval, "Independiente baseline")

classifier.set_active_model("resnet18_finetuned")
gallery_a = embed_rows(classifier.extract_custom_embeddings, gallery_df, "Galería A")
test_a = embed_rows(classifier.extract_custom_embeddings, test_df, "Test A")
independent_a = embed_rows(classifier.extract_custom_embeddings, independent_eval, "Independiente A")

classifier.set_active_model("cnn_custom")
gallery_b = embed_rows(classifier.extract_custom_embeddings, gallery_df, "Galería B")
test_b = embed_rows(classifier.extract_custom_embeddings, test_df, "Test B")
independent_b = embed_rows(classifier.extract_custom_embeddings, independent_eval, "Independiente B")

# %%
retrieval_threshold = baseline_thresholds["cosine"]
gallery_by_model = {"baseline": gallery_emb, "Modelo A": gallery_a, "Modelo B": gallery_b}
embeddings_by_model = {
    "baseline": (baseline_test, baseline_independent),
    "Modelo A": (test_a, independent_a),
    "Modelo B": (test_b, independent_b),
}
retrieval_rows = []
for name, (test_embedding, independent_embedding) in embeddings_by_model.items():
    for split, embedding, breeds in (
        ("test", test_embedding, test_df["breed"].to_list()),
        ("independiente", independent_embedding, independent_eval["breed"].to_list()),
    ):
        ndcg, precision, accuracy, predicted = retrieval_metrics(
            embedding, breeds, gallery_by_model[name], k=BASELINE_K, threshold=retrieval_threshold
        )
        # Macro sobre las razas presentes, igual que el clasificador.
        f1 = f1_score(
            breeds, predicted, labels=sorted(set(breeds)), average="macro", zero_division=0
        )
        retrieval_rows.append(
            {
                "Embeddings": name,
                "Conjunto": split,
                "NDCG@10": ndcg,
                "precision@10": precision,
                "Accuracy de raza (%)": 100 * accuracy,
                "F1 macro (%)": 100 * f1,
            }
        )
print(f"Métrica: cosine | K: {BASELINE_K} | Umbral: {retrieval_threshold:.3f}")
print(f"Consultas por extractor: {test_df.height} de test y {independent_eval.height} independientes.")
with pl.Config(tbl_cols=-1, float_precision=4):
    display(pl.DataFrame(retrieval_rows))

# %% [markdown]
# ### Elección del extractor
#
# En las 1.371 consultas de test, la ResNet ajustada obtuvo un NDCG@10 de **0,8933**,
# frente a **0,8374** del baseline. En las 68 consultas independientes, el baseline
# obtuvo **0,5156**, frente a **0,4977** de la ResNet ajustada. La CNN obtuvo el
# menor NDCG@10 en ambos conjuntos.
#
# Elegimos el **baseline** para `EMBEDDING_MODEL` porque priorizamos el ranking
# en las fotos independientes. La ventaja se observó en esas 68 consultas y queda
# por comprobar en otra muestra externa.
#
# La visualización t-SNE usa 171 imágenes de test de ocho razas seleccionadas por
# las confusiones del Modelo A. En los embeddings de ese subconjunto, la silueta
# media fue 0,1559 con el Modelo A, 0,0891 con el baseline y 0,0266 con el Modelo B.
# El Modelo A logró la mayor separación entre las razas de ese subconjunto según
# esta métrica.

# %%
labels = evaluation_a["class_names"]
off_diagonal = np.asarray(evaluation_a["confusion_matrix"]).copy()
np.fill_diagonal(off_diagonal, 0)
confused = (
    pl.DataFrame({"breed": labels, "n": off_diagonal.sum(axis=0) + off_diagonal.sum(axis=1)})
    .sort("n", descending=True)
    .head(8)
)
subset = confused["breed"].to_list()
mask = test_df["breed"].is_in(subset).to_numpy()
subset_breeds = test_df.filter(pl.Series(mask))["breed"].to_numpy()
print(f"t-SNE sobre {int(mask.sum())} imágenes de {len(subset)} razas confundidas: {', '.join(subset)}")

# %%
tsne_charts = []
silhouette_rows = []
for name, embedding in (("baseline", baseline_test), ("Modelo A", test_a), ("Modelo B", test_b)):
    subset_embedding = embedding[mask]
    silhouette_rows.append(
        {
            "Embeddings": name,
            "silhouette (test, razas confundidas)": silhouette_score(subset_embedding, subset_breeds),
        }
    )
    perplexity = min(30.0, max(5.0, (len(subset_embedding) - 1) / 3))
    points = TSNE(
        n_components=2, perplexity=perplexity, init="pca", random_state=SEED
    ).fit_transform(subset_embedding)
    frame = pl.DataFrame(
        {"x": points[:, 0], "y": points[:, 1], "breed": subset_breeds}
    )
    tsne_charts.append(
        alt.Chart(inline(frame))
        .mark_circle(size=45, opacity=0.75)
        .encode(
            x=alt.X("x:Q", axis=None),
            y=alt.Y("y:Q", axis=None),
            color=alt.Color(
                "breed:N",
                title=None,
                scale=alt.Scale(range=CATEGORICAL_COLORS[: len(subset)]),
            ),
            tooltip=["breed:N"],
        )
        .properties(width=300, height=280, title=name)
    )
display(alt.hconcat(*tsne_charts).properties(title="t-SNE de test (razas más confundidas)"))
with pl.Config(tbl_cols=-1, float_precision=4):
    display(pl.DataFrame(silhouette_rows))

# %% [markdown]
# ## 10. Análisis de errores
#
# La grilla muestra los errores del Modelo A con mayor confianza en test. El Lhasa
# tumbado en el césped fue clasificado como Shih-Tzu con un score de 0,97. El
# Labradoodle y el Cockapoo de la misma grilla se confundieron en ambos sentidos,
# con scores de 0,96 y 0,93. El pelaje rizado y claro que se ve en las dos fotos
# es un parecido que podría contribuir a esa confusión.

# %%
errors_a = (
    pl.DataFrame(
        {
            "path": evaluation_a["paths"],
            "real": evaluation_a["labels"],
            "predicha": evaluation_a["preds"],
            "score": evaluation_a["scores"],
        }
    )
    .filter(pl.col("real") != pl.col("predicha"))
    .sort("score", descending=True)
)
mostrar(errors_a.head(10))

# %%
figure = Figure(figsize=(12, 6), layout="constrained")
axes = figure.subplots(2, 3)
examples = errors_a.head(6).iter_rows(named=True)
for axis, row in zip(axes.flat, examples):
    axis.imshow(cv2.cvtColor(baseline_image(row["path"]), cv2.COLOR_BGR2RGB))
    axis.set_title(f"real: {row['real']}\npredicha: {row['predicha']} ({row['score']:.2f})", fontsize=9)
    axis.axis("off")
display_baseline_figure(figure, "Errores del Modelo A (mayor confianza)")

# %% [markdown]
# ## 11. Costo computacional
#
# El tiempo de CPU por imagen se calculó **en lotes de ocho**. Después de una pasada
# de calentamiento, se cronometraron tres pasadas sobre el mismo lote sintético y se
# dividió el tiempo por las 24 imágenes procesadas. La medición cubre la inferencia
# del modelo. Una consulta completa en la app incluye además el preprocesamiento y,
# para la búsqueda, la recuperación de vecinos.
#
# El Modelo A ocupó **44,9 MB** y tardó **11,89 ms por imagen**. La CNN ocupó
# **5,4 MB** y tardó **19,30 ms por imagen**. En esta prueba, la CNN requirió menos
# espacio en disco y más tiempo de CPU por imagen. El extractor baseline tardó
# **11,99 ms por imagen**. La tabla registra el tamaño de los checkpoints de los dos
# modelos entrenados.

# %%
def model_cost(model: torch.nn.Module, name: str, checkpoint: Path | None = None) -> dict:
    """Parámetros, tamaño, latencia en CPU y dimensión del embedding."""
    parameters = sum(parameter.numel() for parameter in model.parameters())
    model = model.cpu().eval()
    batch = torch.zeros(8, 3, settings.image_size, settings.image_size, device="cpu")
    with torch.no_grad():
        model(batch)
        start = time.perf_counter()
        for _ in range(3):
            model(batch)
    milliseconds = 1000 * (time.perf_counter() - start) / (3 * 8)
    return {
        "Modelo": name,
        "Parámetros (M)": parameters / 1e6,
        "Tamaño (MB)": round(checkpoint.stat().st_size / 1e6, 1) if checkpoint else None,
        "ms/imagen CPU": milliseconds,
        "Embedding": getattr(model, "embedding_dim", 512),
    }


from torchvision import models as tv_models

baseline_model = tv_models.resnet18(weights=None)
baseline_model.fc = torch.nn.Identity()
baseline_model.embedding_dim = 512

classifier.set_active_model("resnet18_finetuned")
cost_a = model_cost(
    classifier.load_model(), "Modelo A", settings.model_path / settings.resnet18_model_name
)
classifier.set_active_model("cnn_custom")
cost_b = model_cost(
    classifier.load_model(), "Modelo B", settings.model_path / settings.cnn_custom_model_name
)
cost_baseline = model_cost(baseline_model, "baseline")

with pl.Config(tbl_cols=-1, float_precision=4):
    display(pl.DataFrame([cost_baseline, cost_a, cost_b]))

# %% [markdown]
# ## 12. Comparación y elección final
#
# En las filas de los modelos A y B, la accuracy y el F1 macro provienen del
# clasificador supervisado de cada modelo. En la fila del baseline, provienen del
# voto ponderado de los diez vecinos de la sección 9. Las columnas de NDCG@10
# miden el orden de los vecinos recuperados con los embeddings de cada fila.

# %%
retrieval_table = {
    (row["Embeddings"], row["Conjunto"]): row for row in retrieval_rows
}
comparison = []
for name in ("Modelo A", "Modelo B"):
    metrics_test = metrics_a if name == "Modelo A" else metrics_b
    metrics_independent = (
        independent_metrics["resnet18_finetuned"]
        if name == "Modelo A"
        else independent_metrics["cnn_custom"]
    )
    cost = cost_a if name == "Modelo A" else cost_b
    comparison.append(
        {
            "Modelo": name,
            "NDCG@10 test": retrieval_table[(name, "test")]["NDCG@10"],
            "NDCG@10 indep": retrieval_table[(name, "independiente")]["NDCG@10"],
            "Accuracy test (%)": 100 * metrics_test["accuracy"],
            "F1 test (%)": 100 * metrics_test["f1"],
            "Accuracy indep (%)": 100 * metrics_independent["accuracy"],
            "F1 indep (%)": 100 * metrics_independent["f1"],
            "Parámetros (M)": cost["Parámetros (M)"],
            "ms/imagen CPU": cost["ms/imagen CPU"],
        }
    )
comparison.append(
    {
        "Modelo": "baseline (Etapa 1)",
        "NDCG@10 test": retrieval_table[("baseline", "test")]["NDCG@10"],
        "NDCG@10 indep": retrieval_table[("baseline", "independiente")]["NDCG@10"],
        "Accuracy test (%)": retrieval_table[("baseline", "test")]["Accuracy de raza (%)"],
        "F1 test (%)": retrieval_table[("baseline", "test")]["F1 macro (%)"],
        "Accuracy indep (%)": retrieval_table[("baseline", "independiente")]["Accuracy de raza (%)"],
        "F1 indep (%)": retrieval_table[("baseline", "independiente")]["F1 macro (%)"],
        "Parámetros (M)": cost_baseline["Parámetros (M)"],
        "ms/imagen CPU": cost_baseline["ms/imagen CPU"],
    }
)
with pl.Config(tbl_cols=-1, float_precision=4):
    display(pl.DataFrame(comparison))

# %%
best_classifier = max(("Modelo A", "Modelo B"), key=lambda name: next(row["F1 test (%)"] for row in comparison if row["Modelo"] == name))
best_embeddings = max(
    ("baseline", "Modelo A", "Modelo B"),
    key=lambda name: (
        retrieval_table[(name, "independiente")]["NDCG@10"],
        retrieval_table[(name, "test")]["NDCG@10"],
    ),
)
print(f"Modelo para clasificar: {best_classifier}")
print(f"Embeddings para la búsqueda (EMBEDDING_MODEL): {best_embeddings}")

# %% [markdown]
# - **Clasificación supervisada.** Elegimos la **ResNet18 ajustada (Modelo A)** por
#   su mayor F1 macro en test (90,42 % frente a 35,21 % de la CNN). También obtuvo mayor
#   F1 macro en el independiente (57,19 % frente a 16,00 %). La estimación
#   de rendimiento independiente corresponde a las 20 razas de esa muestra.
# - **Búsqueda por similitud.** Elegimos el **baseline preentrenado** por su mayor
#   NDCG@10 en las 68 consultas independientes. En test, la ResNet ajustada ordenó
#   mejor los vecinos. La preferencia para la app depende de cuánto se parezcan las
#   consultas reales a las de la muestra independiente.

# %% [markdown]
# ## 13. Checkpoints
#
# Para probar la app, descargá `resnet18_finetuned.pth` y `cnn_custom.pth`
# y colocalos directamente en `models/`. Consultá el estado del enlace de
# descarga en el README. [Configurá tus modelos](README.md#configurá-tus-modelos)
#
# Las salidas de la notebook registran los directorios usados para el
# entrenamiento. Los historiales y las evaluaciones de esa corrida están en
# `output/grupos-v1/`.

# %%
sorted(path.name for path in settings.model_path.glob("*.pth"))

# %% [markdown]
# ## Etapa 3 — Comparación de umbrales del detector YOLO
#
# Comparamos los umbrales de confianza 0,15, 0,25 y 0,50 de `models/yolov8n.pt`
# para la clase COCO `dog` (16). Para elegir el umbral usamos una selección fija
# de 24 fotos tomadas de validación antes del reparto por grupos. La tabla
# corresponde a esas 24 fotos. La comprobación con fotos independientes se
# presenta por separado.
#
# La muestra de 24 fotos incluye 22 imágenes con un perro doméstico y dos
# negativos, Dhole y Coyote. Revisamos las cajas en orden de confianza. Una caja
# cuenta como acierto si localiza una porción sustancial y reconocible de la cabeza
# o el cuerpo de un perro. Cada perro aporta como máximo un acierto. Otra caja del
# mismo perro cuenta como duplicada, una caja sin perro como falsa y un perro sin
# caja asignada como omisión. Esta prueba mide la localización de los perros.
#
# Para elegir el umbral priorizamos menos omisiones. En caso de empate, miramos el
# total de cajas falsas y duplicadas y luego la cercanía a 0,25. Para reproducir la
# revisión usá las mismas rutas bajo `data/dataset`. Las salidas se generaron en CPU
# con Python 3.12.11, Ultralytics 8.4.218 y PyTorch 2.11.0.

# %%
import csv
import sys
from io import BytesIO
from pathlib import Path

import cv2
import pandas as pd
from IPython.display import Image, Markdown, display
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

YOLO_RAIZ = Path.cwd()
YOLO_DATASET = YOLO_RAIZ / "data" / "dataset"
YOLO_PESOS = YOLO_RAIZ / "models" / "yolov8n.pt"
YOLO_UMBRALES = (0.15, 0.25, 0.50)
YOLO_CONTEOS = ["Aciertos", "Omisiones", "Cajas falsas", "Duplicadas"]

if not (YOLO_RAIZ / "src" / "lib").is_dir():
    raise RuntimeError("Abrí el kernel desde la raíz del repositorio.")
if not YOLO_PESOS.is_file():
    raise FileNotFoundError(YOLO_PESOS)

sys.path.insert(0, str(YOLO_RAIZ / "src"))
from lib.services.detection_service import DetectionService

# Se llama únicamente a detect_dogs. La instancia reutiliza los pesos entre umbrales.
yolo_detector = DetectionService(
    classifier=None, yolo_model=str(YOLO_PESOS), conf_threshold=0.25, dog_class_id=16
)

def yolo_leer_manifiesto(nombre):
    with (YOLO_RAIZ / "splits" / nombre).open(encoding="utf-8-sig", newline="") as archivo:
        return list(csv.DictReader(archivo))

def yolo_leer_imagen(ruta):
    archivo = YOLO_DATASET / ruta
    imagen = cv2.imread(str(archivo))
    if imagen is None:
        raise FileNotFoundError(archivo)
    return imagen

# %%
# Revisión de los originales y de las cajas de la corrida visual, por path del manifiesto.
# Cada entrada contiene (perros visibles, {caja de referencia (x1, y1, x2, y2): destino}).
# "perro" identifica al único perro de la foto. Los nombres en la escena identifican instancias.
# La caja del Bluetick se aceptó porque abarca su cabeza y pecho visibles.
# Los destinos fuera de la lista de perros visibles se revisaron como falsos positivos.
YOLO_REVISION = {
    "train/Clumber/055.jpg": (("perro",), {}),
    "train/Shiba Inu/48.jpg": (("perro",), {}),
    "train/Bloodhound/047.jpg": (("perro",), {(3, 20, 220, 209): "perro"}),
    "test/Bull Mastiff/04.jpg": (("perro",), {(0, 8, 213, 168): "perro"}),
    "train/Great Perenees/72.jpg": (("perro",), {}),
    "train/Bichon Frise/097.jpg": (("perro",), {}),
    "train/Pekinese/102.jpg": (("perro",), {(3, 21, 203, 197): "perro"}),
    "train/Basset/006.jpg": (("perro",), {(5, 8, 199, 224): "perro"}),
    "train/Beagle/100.jpg": (("perro",), {(6, 0, 215, 224): "perro"}),
    "test/Mex Hairless/08.jpg": (("perro",), {(27, 1, 219, 222): "perro"}),
    "train/Vizsla/07.jpg": (("perro",), {(16, 1, 217, 220): "perro"}),
    "train/Lhasa/120.jpg": (("perro",), {(14, 15, 214, 218): "perro"}),
    "train/Poodle/107.jpg": (("perro",), {(5, 6, 202, 224): "perro"}),
    "train/Bearded Collie/01.jpg": (("perro",), {}),
    "train/Bull Terrier/063.jpg": (("perro",), {}),
    "train/Cairn/109.jpg": (("perro",), {(0, 8, 218, 220): "perro"}),
    "train/Basenji/33.jpg": (("perro",), {(30, 2, 223, 224): "perro"}),
    "train/Dhole/056.jpg": ((), {}),
    "train/Chow/068.jpg": (("perro",), {(9, 35, 223, 207): "perro"}),
    "train/German Sheperd/101.jpg": (("perro",), {}),
    "train/Coyote/83.jpg": ((), {}),
    "train/Corgi/091.jpg": (("perro",), {(18, 17, 222, 223): "perro"}),
    "train/Bluetick/105.jpg": (("perro",), {(11, 6, 136, 176): "perro"}),
    "test/Border Collie/07.jpg": (("perro",), {(12, 7, 221, 222): "perro"}),
    "independent/Beagle/beagle_001.jpg": (("perro",), {(250, 373, 845, 1043): "perro"}),
    "independent/Beagle/beagle_002.jpg": (("perro",), {(109, 374, 814, 1280): "perro"}),
    "independent/Bearded Collie/bearded_collie_001.jpg": (("perro",), {(247, 30, 720, 718): "perro"}),
    "independent/Bearded Collie/bearded_collie_002.jpg": (("perro",), {(133, 104, 805, 1149): "perro"}),
    "independent/Border Collie/border_collie_001.jpg": (("perro",), {(46, 102, 262, 203): "perro"}),
    "independent/Border Collie/border_collie_002.jpg": (("perro",), {(26, 270, 380, 646): "perro"}),
    "independent/Chihuahua/chihuahua_001.jpg": (("perro",), {(109, 76, 559, 520): "perro"}),
    "independent/Chihuahua/chihuahua_002.jpg": (("perro",), {(144, 105, 801, 395): "perro"}),
    "independent/Chihuahua/chihuahua_003.jpg": (("perro",), {(143, 40, 1218, 1070): "perro"}),
    "independent/Cockapoo/cockapoo_001.jpg": (("perro",), {(1, 437, 736, 1331): "perro"}),
    "independent/Cockapoo/cockapoo_002.jpg": (("perro",), {(241, 132, 818, 1154): "perro"}),
    "independent/Collie/collie_001.jpg": (("perro",), {(224, 45, 706, 470): "perro"}),
    "independent/Collie/collie_002.jpg": (("perro",), {(1, 0, 738, 718): "perro"}),
    "independent/Golden Retriever/golden_retriever_001.jpg": (("perro",), {(1243, 558, 2576, 2255): "perro"}),
    "independent/Golden Retriever/golden_retriever_002.jpg": (("perro",), {(28, 4, 628, 478): "perro"}),
    "independent/Labradoodle/labradoodle_001.jpg": (("perro",), {(6, 285, 670, 873): "perro"}),
    "independent/Labradoodle/labradoodle_002.jpg": (("perro",), {(231, 303, 1246, 1879): "perro"}),
    "independent/Labrador/labrador_001.jpg": (("perro",), {(195, 739, 3024, 3741): "perro"}),
    "independent/Labrador/labrador_002.jpg": (("perro",), {}),
    "independent/Pekinese/pekinese_001.jpg": (("perro",), {(62, 153, 757, 674): "perro"}),
    "independent/Pekinese/pekinese_002.jpg": (("perro",), {(23, 187, 849, 816): "perro"}),
    "independent/Pomeranian/pomeranian_001.jpg": (("perro",), {(62, 18, 709, 617): "perro"}),
    "independent/Pomeranian/pomeranian_002.jpg": (("perro",), {(27, 27, 922, 828): "perro"}),
    "independent/Poodle/Poodle.jpg": (("perro",), {(187, 21, 718, 848): "perro"}),
    "independent/Poodle/Standard_black_Poodle.jpg": (("perro",), {(197, 15, 408, 425): "perro"}),
    "independent/escenas/escena_001.jpg": (
        ("Beagle", "Golden Retriever"),
        {
            (855, 1375, 2709, 3014): "Beagle",
            (1036, 453, 4032, 2755): "Golden Retriever",
            (513, 2, 1234, 1560): "persona del fondo",
        },
    ),
    "independent/sin_perro/gato_001.jpg": ((), {}),
}

# %%
def yolo_iou(caja, referencia):
    x1, y1, x2, y2 = caja
    rx1, ry1, rx2, ry2 = referencia
    interseccion = max(0, min(x2, rx2) - max(x1, rx1)) * max(0, min(y2, ry2) - max(y1, ry1))
    union = (x2 - x1) * (y2 - y1) + (rx2 - rx1) * (ry2 - ry1) - interseccion
    return interseccion / union

def yolo_evaluar(muestra):
    registros = []
    for ruta, tipo in muestra.items():
        visibles, referencias = YOLO_REVISION[ruta]
        imagen = yolo_leer_imagen(ruta)
        for umbral in YOLO_UMBRALES:
            yolo_detector.conf_threshold = umbral
            detecciones = sorted(yolo_detector.detect_dogs(imagen), key=lambda deteccion: -deteccion[1])
            asignados, cajas = set(), []
            for caja, confianza in detecciones:
                # IoU de 0.85 altera solo el emparejamiento con una caja ya revisada
                # ante una pequeña deriva en píxeles. El acierto se decide visualmente.
                destinos = {
                    destino for referencia, destino in referencias.items()
                    if yolo_iou(caja, referencia) >= 0.85
                }
                if len(destinos) != 1:
                    raise ValueError(
                        f"Revisá la caja {caja} de {ruta} con umbral {umbral:.2f} "
                        f"(confianza {confianza:.4f}). La anotación no es inequívoca."
                    )
                destino = destinos.pop()
                if destino not in visibles:
                    revision = "Falsa"
                elif destino in asignados:
                    revision = "Duplicada"
                else:
                    revision = "Acierto"
                    asignados.add(destino)
                cajas.append((caja, confianza, destino, revision))

            omitidos = tuple(perro for perro in visibles if perro not in asignados)
            registros.append({
                "Ruta": ruta, "Tipo de caso": tipo, "Umbral": umbral,
                "Visibles": visibles, "Omitidos": omitidos, "Cajas": cajas,
                "Aciertos": len(asignados), "Omisiones": len(omitidos),
                "Cajas falsas": sum(caja[3] == "Falsa" for caja in cajas),
                "Duplicadas": sum(caja[3] == "Duplicada" for caja in cajas),
            })
    return pd.DataFrame(registros)

def yolo_mostrar_revision(evaluacion):
    filas = []
    for ruta, datos in evaluacion.groupby("Ruta", sort=False):
        fila = {"Ruta": ruta, "Perros visibles": ", ".join(datos.iloc[0]["Visibles"]) or "0"}
        for resultado in datos.to_dict("records"):
            revision = [
                f"{estado}: {destino} ({confianza:.4f})"
                for _, confianza, destino, estado in resultado["Cajas"]
            ]
            revision.extend(f"Omisión: {perro}" for perro in resultado["Omitidos"])
            fila[f"{resultado['Umbral']:.2f}"] = " · ".join(revision) or "0 cajas"
        filas.append(fila)
    with pd.option_context("display.max_colwidth", None, "display.max_rows", None):
        display(pd.DataFrame(filas).style.hide(axis="index"))

# %%
yolo_muestra_historica = {
    ruta: "Validación anterior" for ruta in YOLO_REVISION if not ruta.startswith("independent/")
}
yolo_validacion = yolo_evaluar(yolo_muestra_historica)
yolo_mostrar_revision(yolo_validacion)

# %% [markdown]
# ### Resultados en la muestra de 24 fotos
#
# La tabla resume aciertos, omisiones, cajas falsas y duplicadas por umbral. En
# la figura, cada recuadro lleva la asignación de la revisión visual y el score
# del detector.

# %%
def yolo_mostrar_cajas(evaluacion, ruta, titulo, altura):
    original = cv2.cvtColor(yolo_leer_imagen(ruta), cv2.COLOR_BGR2RGB)
    figura = Figure(figsize=(15, altura), layout="constrained")
    figura.suptitle(titulo, fontsize=16)
    resultados = evaluacion[evaluacion["Ruta"] == ruta].to_dict("records")
    for eje, resultado in zip(figura.subplots(1, 3), resultados, strict=True):
        cajas = resultado["Cajas"]
        eje.imshow(original)
        cantidad = "1 caja" if len(cajas) == 1 else f"{len(cajas)} cajas"
        eje.set_title(f"Umbral {resultado['Umbral']:.2f} · {cantidad}", fontsize=13)
        eje.set_axis_off()
        aciertos = iter(("#ffe14a", "#f259d8"))
        for (x1, y1, x2, y2), confianza, destino, revision in cajas:
            color = {
                "Falsa": "#4de0ff",
                "Duplicada": "#ffa85d",
            }.get(revision)
            if revision == "Acierto":
                color = next(aciertos, "#ffe14a")
            eje.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, lw=2.4))
            eje.annotate(
                f"{destino} · {confianza:.4f}", (x1, y1), xytext=(2, -2),
                textcoords="offset points", va="top", fontsize=10, color="black",
                bbox={"facecolor": color, "edgecolor": "none", "pad": 2},
            )
    with BytesIO() as salida:
        figura.savefig(salida, format="png", dpi=105)
        display(Image(data=salida.getvalue(), format="png", alt=titulo))

yolo_totales_validacion = yolo_validacion.groupby("Umbral")[YOLO_CONTEOS].sum()
display(yolo_totales_validacion.style.format_index("{:.2f}", axis=0))
yolo_elegido = min(
    YOLO_UMBRALES,
    key=lambda umbral: (
        yolo_totales_validacion.loc[umbral, "Omisiones"],
        yolo_totales_validacion.loc[umbral, "Cajas falsas"] + yolo_totales_validacion.loc[umbral, "Duplicadas"],
        abs(umbral - 0.25),
    ),
)
yolo_mostrar_cajas(yolo_validacion, "train/Poodle/107.jpg", "Validación anterior · Poodle, 107.jpg", 4.8)
display(Markdown(f"**La validación histórica favorece `{yolo_elegido:.2f}`.**"))

# %% [markdown]
# Con 0,15 se omitieron 7 de los 22 perros, frente a 9 con 0,25 y 14 con 0,50.
# La tabla registra cero cajas falsas y duplicadas con los tres umbrales. En
# la figura de `107.jpg`, el Poodle blanco se detecta con 0,15 (score 0,2102).
# Con 0,25 y 0,50, la figura muestra cero cajas. Elegimos 0,15 por el menor
# número de omisiones en esta muestra de fotos individuales.

# %% [markdown]
# ### Comprobación independiente
#
# La tabla incluye las 27 fotos independientes con anotaciones en `YOLO_REVISION`.
# Cada ruta se cuenta una vez. Se revisaron 25 fotos de un perro, una escena de
# dos perros y una foto de un gato. Quedan 80 fotos del manifiesto pendientes de
# revisión visual.

# %%
yolo_tipos = {
    "un_perro": "Fotos individuales",
    "varios_perros": "Escena con dos perros",
    "gato": "Gato",
    "perro_contexto": "Perro en contexto",
    "objeto_sin_animal": "Objeto sin animal",
}
yolo_rutas_independientes = {}
for yolo_fila in yolo_leer_manifiesto("independent_manifest.csv"):
    yolo_ruta = yolo_fila["path"]
    yolo_tipo = yolo_tipos[yolo_fila["scene_type"]]
    if yolo_rutas_independientes.setdefault(yolo_ruta, yolo_tipo) != yolo_tipo:
        raise ValueError(f"Tipos de escena distintos para {yolo_ruta}.")

yolo_pendientes = sorted(set(yolo_rutas_independientes) - set(YOLO_REVISION))
print(f"Fotos independientes pendientes de revisión visual: {len(yolo_pendientes)}")
for yolo_ruta in yolo_pendientes:
    print(f"  {yolo_ruta}")
yolo_revisadas = {ruta: tipo for ruta, tipo in sorted(yolo_rutas_independientes.items()) if ruta in YOLO_REVISION}
yolo_independiente = yolo_evaluar(yolo_revisadas)
yolo_totales_independiente = (
    yolo_independiente.groupby(["Tipo de caso", "Umbral"], sort=False)[YOLO_CONTEOS].sum()
)
display(yolo_totales_independiente.style.format_index("{:.2f}", level="Umbral", axis=0))
yolo_mostrar_revision(yolo_independiente)

# %%
yolo_mostrar_cajas(
    yolo_independiente, "independent/escenas/escena_001.jpg", "Escena independiente · dos perros", 4.9
)
yolo_mostrar_cajas(
    yolo_independiente, "independent/sin_perro/gato_001.jpg", "Negativo independiente · gato", 4.1
)

# %% [markdown]
# En la escena con dos perros, el Beagle con arnés está delante del Golden
# Retriever. Con 0,25 aparecen las dos cajas (scores de 0,8686 y 0,2807). Con
# 0,50 queda la del Beagle. Con 0,15 aparecen las dos cajas y una caja falsa
# sobre la persona del fondo (score de 0,1504). La foto del gato muestra cero
# cajas con los tres umbrales.
#
# En las fotos independientes revisadas se detectaron los mismos 26 perros con
# 0,15 y 0,25. En esa muestra, 0,25 evitó la caja falsa de la escena. La elección
# de 0,15 se apoya en las 24 fotos usadas para comparar umbrales. La evidencia
# sobre perros superpuestos se limita a la escena que se muestra aquí.
