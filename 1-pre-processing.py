# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # TP2 - Parte 1: dataset y preprocesamiento

# %% [markdown]
# ## Equipo
# - Alumno 1 : Alejandro Armas
# - Alumno 2 : Gabriel Soda

# %%
# Configuración global: imports, paths, semilla y paleta de colores fija del proyecto.
import warnings
from pathlib import Path

import random
import hashlib
import altair as alt
import cv2
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from IPython.display import display
from PIL import Image, UnidentifiedImageError

alt.data_transformers.disable_max_rows()

ROOT = Path.cwd()
DATASET_DIR = ROOT / "data" / "dataset"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SEED = 42

COLORS = {
    "primary": "#4c78a8",
    "secondary": "#f58518",
    "negative": "#e45756",
    "positive": "#54a24b",
    "neutral": "#72b7b2",
    "gray": "#b3b3b3",
    "warning": "#f2cf4a",
}
CATEGORICAL_COLORS = [
    COLORS["primary"],
    COLORS["secondary"],
    COLORS["negative"],
    COLORS["positive"],
    COLORS["neutral"],
    COLORS["warning"],
    "#9d7660",
    "#b279a2",
    "#54a24b",
    "#ff9da6",
    "#72b7b2",
    "#f2cf4a",
    "#4c78a8",
    "#e45756",
    "#f58518",
    "#9d7660",
]


def inline(df: pl.DataFrame) -> alt.InlineData:
    """Convierte un DataFrame de Polars a InlineData para Altair."""
    return alt.InlineData(values=df.to_dicts())


# Mapeo para mostrar columnas en español; los nombres internos del código siguen en inglés.
NOMBRES_ES = {
    "split": "partición",
    "breed": "raza",
    "path": "ruta",
    "filename": "archivo",
    "suffix": "extensión",
    "images": "imágenes",
    "mean": "media",
    "median": "mediana",
    "min": "mín",
    "max": "máx",
    "std": "desvío",
    "cv": "coef. variación",
    "format": "formato",
    "mode": "modo",
    "width": "ancho",
    "height": "alto",
    "aspect_ratio": "relación aspecto",
    "pixels": "píxeles",
    "brightness": "brillo",
    "contrast": "contraste",
    "laplacian_var": "var. Laplaciano",
    "readable": "legible",
    "error": "error",
    "quality_rule": "regla",
}


def mostrar(df: pl.DataFrame) -> None:
    """Muestra un DataFrame con los nombres de columnas en español."""
    renombrar = {k: v for k, v in NOMBRES_ES.items() if k in df.columns}
    display(df.rename(renombrar))


# %%
# Verificamos si el dataset ya está descargado y damos la instrucción exacta si falta.
splits_available = [
    split for split in ("train", "valid", "test") if (DATASET_DIR / split).is_dir()
]
HAS_DATASET = len(splits_available) > 0

if HAS_DATASET:
    print(f"Dataset encontrado en {DATASET_DIR} (splits: {splits_available})")
else:
    print(f"No se encontró el dataset en {DATASET_DIR}.")
    print("Descargalo con: python scripts/download_dataset.py")
    print("O descomprimí manualmente train/, valid/ y test/ dentro de data/dataset/.")

# %% [markdown]
# ## 3.1 Descripción del dataset
#
# Usamos el **70 Dog Breeds Image Dataset** ([Kaggle](https://www.kaggle.com/datasets/gpiosenka/70-dog-breedsimage-data-set)),
# que ya viene particionado en `train/`, `valid/` y `test/` con una carpeta por raza.
# La licencia publicada en Kaggle es **CC0: Public Domain**; igualmente conviene revisar
# la página actual del dataset antes de cualquier uso comercial, porque los derechos de
# las imágenes individuales pueden tener otra procedencia.
#
# **Encabezado esperado.** Las imágenes del dataset suelen ser recortes centrados en un
# único perro, con fondo variable (césped, interior, personas, otros perros) y resolución
# chica a media. Eso contrasta con lo que va a recibir el clasificador en la **Etapa 3**:
# recortes de YOLO, con otro encuadre, escala y, muchas veces, un perro que ocupa menos
# la imagen. Ese cambio de dominio se mide con el conjunto independiente (3.4).
#
# **Limitaciones que asumimos desde el arranque:** razas visualmente muy parecidas entre
# sí (husky/malamute, etc.), sesgo de imágenes de origen web, ausencia de perros mestizos,
# etiquetas que pueden contener errores y poca variedad de contextos (mucho retrato y poco
# perro lejano o en movimiento).

# %%
# Inventario: recorremos las carpetas sin cargar píxeles, solo path + split + raza.
# Cada fila es una imagen candidata del dataset original (antes de cualquier filtro).
def scan_dataset(dataset_dir: Path) -> pl.DataFrame:
    """Recorre train/valid/test y devuelve una fila por imagen encontrada."""
    records = []
    for split in ("train", "valid", "test"):
        split_dir = dataset_dir / split
        if not split_dir.is_dir():
            continue
        for image_path in sorted(split_dir.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_SUFFIXES:
                records.append(
                    {
                        "split": split,
                        "breed": image_path.parent.name,
                        "path": image_path.relative_to(dataset_dir).as_posix(),
                        "filename": image_path.name,
                        "suffix": image_path.suffix.lower(),
                    }
                )

    return pl.from_dicts(
        records,
        schema={
            "split": pl.String,
            "breed": pl.String,
            "path": pl.String,
            "filename": pl.String,
            "suffix": pl.String,
        },
    )


inventory = scan_dataset(DATASET_DIR).with_columns(
    pl.col("breed").str.replace_all(r"\s+", " ").str.strip_chars()
)
print(f"Imágenes encontradas: {inventory.height}")
mostrar(inventory.head(5))

# %%
# Conteo de razas e imágenes por split, para ver de entrada si la partición es la esperada.
breed_split_counts = (
    inventory.group_by(["split", "breed"]).len().rename({"len": "images"})
)

split_summary = breed_split_counts.group_by("split").agg(
    pl.col("images").sum().alias("cant. imágenes"),
    pl.col("breed").n_unique().alias("razas"),
)

mostrar(split_summary)

# %% [markdown]
# ## 3.2 Análisis exploratorio (EDA)
#
# El objetivo del EDA es entender **desbalance de clases**, **resoluciones/formato** y
# **variabilidad fotométrica** antes de filtrar. Mantenemos la regla del proyecto: Polars
# para agregar y Altair para graficar, con la paleta fija definida arriba.

# %%
# Antes de decidir qué graficar, verificamos cuántas imágenes por raza hay en cada split.
# Si valid y test resultan uniformes, el desbalance solo puede estudiarse en train.
split_breed_stats = breed_split_counts.group_by("split").agg(
    pl.col("images").min().alias("mín por raza"),
    pl.col("images").max().alias("máx por raza"),
    pl.col("images").n_unique().alias("valores distintos"),
)
mostrar(split_breed_stats)

# %% [markdown]
# `valid` y `test` tienen exactamente **10 imágenes por raza** (mín = máx = 10 y un solo
# valor distinto). Con una partición tan uniforme, el histograma de desbalance solo aporta
# información en `train`, así que lo graficamos únicamente para ese split.

# %%
# Estadísticas de desbalance de train: media, mediana, mín, máx, desvío y coeficiente de variación.
train_breed_counts = breed_split_counts.filter(pl.col("split") == "train").select(
    ["breed", "images"]
)

train_stats = train_breed_counts.select(
    pl.col("images").mean().round(1).alias("media"),
    pl.col("images").median().alias("mediana"),
    pl.col("images").min().alias("mín"),
    pl.col("images").max().alias("máx"),
    pl.col("images").std().round(1).alias("desvío"),
    (pl.col("images").std() / pl.col("images").mean()).round(2).alias("coef. variación"),
)
mostrar(train_stats)

# %%
# Histograma de imágenes por raza para train: muestra cuán balanceadas están las 70 clases.
bin_step = 10
if not train_breed_counts.is_empty():
    hist_chart = (
        alt.Chart(inline(train_breed_counts))
        .mark_bar(opacity=0.75)
        .encode(
            x=alt.X(
                "images:Q",
                bin=alt.Bin(step=bin_step),
                title="Imágenes por raza (train)",
            ),
            y=alt.Y("count():Q", title="Razas"),
            color=alt.value(COLORS["primary"]),
            tooltip=[alt.Tooltip("count():Q", title="Razas")],
        )
        .properties(width=500, height=280, title="Desbalance de train")
    )
    display(hist_chart)
else:
    print("Sin datos para graficar la distribución de train.")

# %%
# Cargamos cada imagen una única vez y guardamos metadatos: tamaño, modo, formato, brillo,
# contraste y varianza del Laplaciano (la usamos después para detectar borrosas).
# `warnings` como error nos permite capturar archivos corruptos que PIL no termina de decodificar.
def compute_image_metadata(inventory: pl.DataFrame, dataset_dir: Path) -> pl.DataFrame:
    """Lee cada imagen y devuelve metadatos fotométricos + de decodificación."""
    records = []
    for row in inventory.iter_rows(named=True):
        path = dataset_dir / row["path"]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                with Image.open(path) as image:
                    image.load()
                    width, height = image.size
                    image_format, image_mode = image.format, image.mode
                    gray = np.asarray(image.convert("L"), dtype=np.float32)
                    brightness = float(gray.mean())
                    contrast = float(gray.std())
                    laplacian_var = float(
                        cv2.Laplacian(gray.astype(np.uint8), cv2.CV_64F).var()
                    )

            records.append(
                {
                    **row,
                    "width": width,
                    "height": height,
                    "aspect_ratio": round(width / height, 3),
                    "pixels": width * height,
                    "format": image_format,
                    "mode": image_mode,
                    "brightness": round(brightness, 2),
                    "contrast": round(contrast, 2),
                    "laplacian_var": round(laplacian_var, 2),
                    "readable": True,
                    "error": None,
                }
            )
        except (UnidentifiedImageError, OSError, ValueError, Warning) as error:
            records.append(
                {
                    **row,
                    "width": None,
                    "height": None,
                    "aspect_ratio": None,
                    "pixels": None,
                    "format": None,
                    "mode": None,
                    "brightness": None,
                    "contrast": None,
                    "laplacian_var": None,
                    "readable": False,
                    "error": type(error).__name__,
                }
            )

    return pl.from_dicts(
        records,
        schema={
            "split": pl.String,
            "breed": pl.String,
            "path": pl.String,
            "filename": pl.String,
            "suffix": pl.String,
            "width": pl.Int64,
            "height": pl.Int64,
            "aspect_ratio": pl.Float64,
            "pixels": pl.Int64,
            "format": pl.String,
            "mode": pl.String,
            "brightness": pl.Float64,
            "contrast": pl.Float64,
            "laplacian_var": pl.Float64,
            "readable": pl.Boolean,
            "error": pl.String,
        },
    )


image_meta = compute_image_metadata(inventory, DATASET_DIR)
print(f"Metadatos calculados para {image_meta.height} imágenes")
mostrar(image_meta.sample(5, seed=SEED))

# %%
# Formatos/modos y dimensiones: verificamos extensiones y que todo esté en 224×224.
readable = image_meta.filter(pl.col("readable"))
print(f"Legibles: {readable.height} | No legibles: {image_meta.height - readable.height}")

if not readable.is_empty():
    format_counts = (
        readable.group_by(["format", "mode"])
        .len()
        .rename({"len": "imágenes"})
        .sort("imágenes", descending=True)
    )
    mostrar(format_counts)

    dimension_counts = (
        readable.with_columns(
            pl.concat_str(
                pl.col("width").cast(pl.String),
                pl.lit("×"),
                pl.col("height").cast(pl.String),
            ).alias("dimensiones")
        )
        .group_by("dimensiones")
        .len()
        .rename({"len": "imágenes"})
        .sort("imágenes", descending=True)
    )
    mostrar(dimension_counts)
else:
    print("Sin imágenes legibles para resumir formatos/dimensiones.")

# %%
# Grilla visual de ejemplos por raza: sirve para inspeccionar encuadre, fondo, pose y oclusión.
def show_breed_grid(
    meta: pl.DataFrame,
    dataset_dir: Path,
    breeds: list[str] | None = None,
    n_breeds: int = 5,
    n_samples: int = 4,
    seed: int = SEED,
) -> None:
    """Muestra una grilla de ejemplos por raza (matplotlib, no Altair)."""
    if breeds is None:
        breeds = (
            meta["breed"]
            .unique()
            .sort()
            .sample(n=n_breeds, seed=seed, shuffle=True)
            .to_list()
        )

    fig, axes = plt.subplots(
        len(breeds), n_samples, figsize=(2.5 * n_samples, 2.5 * len(breeds)),
        squeeze=False,
    )
    for i, breed in enumerate(breeds):
        breed_meta = meta.filter(pl.col("breed") == breed)
        paths = (
            breed_meta.sample(n=n_samples, seed=seed)["path"]
            .to_list()
        )
        for j in range(n_samples):
            ax = axes[i, j]
            if j < len(paths):
                with Image.open(dataset_dir / paths[j]) as image:
                    ax.imshow(image.convert("RGB"))
            ax.axis("off")
            if j == 0:
                ax.set_ylabel(breed, fontsize=9, rotation=0, ha="right", va="center")

    fig.suptitle("Muestra visual por raza — encuadre, fondo, pose, oclusión", fontsize=12)
    plt.tight_layout()
    plt.show()


show_breed_grid(image_meta.filter(pl.col("split") == "train"), DATASET_DIR)

# %%
CONFUSABLE_BREEDS = ["Cockapoo", "Poodle", "Labradoodle"] #["American Hairless", "Chinese Crested", "Mex Hairless"]

# Muestra una comparación visual de razas fáciles de confundir.
def show_breed_comparison(
    meta: pl.DataFrame,
    dataset_dir: Path,
    breeds: list[str] = CONFUSABLE_BREEDS,
    n_samples: int = 5,
    seed: int | None = None,
) -> None:
    """Muestra una grilla comparando una lista arbitraria de razas."""
    readable_meta = meta.filter(pl.col("readable"))
    if readable_meta.is_empty():
        print("No hay imágenes legibles para mostrar.")
        return


    n_breeds = len(breeds)
    fig, axes = plt.subplots(
        n_breeds, n_samples, figsize=(2.5 * n_samples, 2.5 * n_breeds), squeeze=False
    )


    breed_meta = readable_meta.filter(pl.col("breed").is_in(breeds), pl.col("split") == "train")
    paths = (
        breed_meta.sample(n=n_samples * n_breeds, seed=seed)["path"]
        .to_list()
    )
    #random.shuffle(paths)

    print(len(paths))

    for i in range(n_breeds):
        for j in range(n_samples):
            ax = axes[i, j]
            with Image.open(dataset_dir / paths[i*n_samples + j]) as image:
                ax.imshow(image.convert("RGB"))
            ax.axis("off")

    fig.suptitle("Razas fáciles de confundir: {}".format(", ".join(breeds)), fontsize=12)
    plt.tight_layout()
    plt.show()


show_breed_comparison(image_meta, DATASET_DIR)

# %% [markdown]
#
# TODO redactar observaciones de las imagenes, a priori son todas fotos de frente, de un unico perro generalmente en exteriores
#

# %%
# Brillo y contraste como proxy de iluminación. Unpivotamos para graficar ambas métricas
# en un solo Altair facetado y detectar extremos de exposición.
if not readable.is_empty():
    photo_long = readable.unpivot(
        index=["split", "breed", "path"],
        on=["brightness", "contrast"],
        variable_name="metric",
        value_name="value",
    )

    photo_chart = (
        alt.Chart(inline(photo_long))
        .mark_bar(opacity=0.7)
        .encode(
            x=alt.X("value:Q", bin=alt.Bin(maxbins=40), title="Valor"),
            y=alt.Y("count():Q", title="Imágenes"),
            color=alt.Color(
                "metric:N",
                title=None,
                scale=alt.Scale(range=[COLORS["primary"], COLORS["secondary"]]),
            ),
            column=alt.Column("metric:N", title=None),
        )
        .properties(width=280, height=240, title="Distribución de brillo y contraste")
    )
    display(photo_chart)
else:
    print("Sin imágenes legibles para analizar brillo/contraste.")

# %% [markdown]
# ## 3.3 Control de calidad y filtrado
#
# Aplicamos reglas **explícitas y en orden**, y reportamos cuántas imágenes descarta cada
# una. El orden es eficiente: primero lo barato (legibilidad, tamaño), después lo que ya
# está calculado (borrosidad, exposición) y al final los duplicados.
#
# Reglas:
# 1. Archivo corrupto o que no se puede decodificar.
# 2. Lado mínimo de resolución < 96 px.
# 3. Varianza del Laplaciano < 100 (imagen borrosa).
# 4. Brillo extremo (fuera de [15, 240] en escala de grises).
# 5. Duplicados exactos (archivos idénticos, SHA-256).
#
# El filtrado se aplica a los tres splits por igual; en duplicados entre splits conservamos
# la copia de `train` y descartamos la de `valid`/`test` para evitar fuga de datos.

# %%
# Umbrales de filtrado
BLUR_THRESHOLD = 40
BRIGHTNESS_MIN = 15.0
BRIGHTNESS_MAX = 240.0

# %%
# Reglas visuales 1-3 en un único `when/then` encadenado: la primera que dispara queda
# registrada en `quality_rule`. Las que pasan pasan a la etapa de duplicados.
quality = image_meta.with_columns(
    pl.min_horizontal("width", "height").alias("min_side")
).with_columns(
    pl.when(~pl.col("readable"))
    .then(pl.lit("R1: No legible o corrupta"))
    .when(pl.col("laplacian_var") < BLUR_THRESHOLD)
    .then(pl.lit("R2: Borrosa (Laplaciano < {})".format(BLUR_THRESHOLD)))
    .when((pl.col("brightness") < BRIGHTNESS_MIN) | (pl.col("brightness") > BRIGHTNESS_MAX))
    .then(pl.lit("R3: {} < Brillo < {}".format(BRIGHTNESS_MIN, BRIGHTNESS_MAX)))
    .otherwise(pl.lit("Pasa reglas visuales"))
    .alias("quality_rule")
)


mostrar(
    quality.group_by(["quality_rule", "split"])
    .len()
    .rename({"len": "images"})
    .sort(["quality_rule", "split"])
)


# %% [markdown]
# Para duplicados usamos **SHA-256** (archivos exactamente idénticos). Como el mismo archivo
# puede aparecer en dos splits, conservamos la copia de valid/test y descartamos la de train:
# así no reducimos los conjuntos de evaluación, que son chicos y fijos.

# %%
# Marcamos duplicados exactos por SHA-256. Prioridad test > valid > train: si el mismo archivo
# está en train y en test/valid, descartamos la copia de train (conservamos la de evaluación).
def mark_duplicates(candidates: pl.DataFrame, dataset_dir: Path) -> pl.DataFrame:
    """Devuelve `candidates` con `duplicate_of`, `duplicate_of_split` e `is_duplicate`."""
    if candidates.is_empty():
        return candidates.with_columns(
            pl.lit(None, dtype=pl.String).alias("duplicate_of"),
            pl.lit(None, dtype=pl.String).alias("duplicate_of_split"),
            pl.lit(False).alias("is_duplicate"),
        )

    records = []
    for row in candidates.iter_rows(named=True):
        file_hash = hashlib.sha256((dataset_dir / row["path"]).read_bytes()).hexdigest()
        records.append({"path": row["path"], "split": row["split"], "file_hash": file_hash})

    hash_df = pl.from_dicts(records)

    # Menor número = mayor prioridad para conservar.
    priority = pl.DataFrame(
        {"split": ["test", "valid", "train"], "split_priority": [0, 1, 2]}
    )

    duplicates: dict[str, tuple[str, str]] = {}
    for file_hash in hash_df["file_hash"].unique().to_list():
        group = (
            hash_df.filter(pl.col("file_hash") == file_hash)
            .join(priority, on="split", how="left")
            .sort(["split_priority", "path"])
        )
        if group.height <= 1:
            continue
        paths = group["path"].to_list()
        splits = group["split"].to_list()
        kept, kept_split = paths[0], splits[0]
        for path in paths[1:]:
            duplicates[path] = (kept, kept_split)

    if duplicates:
        dup_df = pl.DataFrame(
            {
                "path": list(duplicates.keys()),
                "duplicate_of": [v[0] for v in duplicates.values()],
                "duplicate_of_split": [v[1] for v in duplicates.values()],
            }
        )
    else:
        dup_df = pl.DataFrame(
            {"path": [], "duplicate_of": [], "duplicate_of_split": []},
            schema={
                "path": pl.String,
                "duplicate_of": pl.String,
                "duplicate_of_split": pl.String,
            },
        )

    return candidates.join(dup_df, on="path", how="left").with_columns(
        pl.col("duplicate_of").is_not_null().alias("is_duplicate")
    )


visual_pass = quality.filter(pl.col("quality_rule") == "Pasa reglas visuales")
duplicate_flags = mark_duplicates(
    visual_pass.select(["path", "split", "breed"]), DATASET_DIR
)

# Combinamos la regla 4 con las reglas visuales: si es duplicado, pisa la regla anterior.
quality = quality.join(
    duplicate_flags.select(["path", "is_duplicate", "duplicate_of", "duplicate_of_split"]),
    on="path",
    how="left",
).with_columns(
    pl.when(pl.col("is_duplicate"))
    .then(pl.lit("R4: Duplicado exacto"))
    .otherwise(pl.col("quality_rule"))
    .alias("quality_rule")
)

# %%
# Resumen final del pipeline de calidad: imágenes descartadas por regla y total aceptado.
if not quality.is_empty():
    rule_counts = (
        quality.group_by(["quality_rule", "split"])
        .len()
        .rename({"len": "images"})
        .sort(["quality_rule", "split"])
    )
    mostrar(rule_counts)

    accepted = quality.filter(pl.col("quality_rule") == "Pasa reglas visuales")
    print(f"Aceptadas: {accepted.height} imágenes")
    accepted_by_split = accepted.group_by("split").len().rename({"len": "images"})
    mostrar(accepted_by_split.sort("split"))
else:
    print("Sin imágenes para resumir el filtrado.")

# %%
# Ejemplos descartados por cada regla: validan visualmente que el umbral elegido tiene sentido.
def show_rejected_examples(
    quality: pl.DataFrame, dataset_dir: Path, rules: list[str] | None = None, n: int = 3
) -> None:
    """Muestra hasta `n` imágenes rechazadas por cada regla de calidad."""
    rejected = quality.filter(~pl.col("quality_rule").str.starts_with("Pasa"))
    if rejected.is_empty():
        print("No hay imágenes descartadas para mostrar.")
        return

    if rules is None:
        rules = rejected["quality_rule"].unique().sort().to_list()

    for rule in rules:
        rule_examples = rejected.filter(pl.col("quality_rule") == rule)
        if rule_examples.is_empty():
            continue

        for split in sorted(rule_examples["split"].unique().to_list()):
            examples = rule_examples.filter(pl.col("split") == split).head(n)

            # Los duplicados se muestran de a pares: descartada vs. la copia conservada.
            if rule.startswith("R4:"):
                fig, axes = plt.subplots(
                    examples.height, 2, figsize=(8, 4 * examples.height), squeeze=False
                )
                for ax_pair, row in zip(axes, examples.iter_rows(named=True)):
                    for ax, path, title in zip(
                        ax_pair,
                        [row["path"], row["duplicate_of"]],
                        [f"Descartada ({row['split']})", f"Conservada ({row['duplicate_of_split']})"],
                    ):
                        with Image.open(dataset_dir / path) as image:
                            ax.imshow(image.convert("RGB"))
                        ax.set_title(f"{row['breed']}\n{title}", fontsize=9)
                        ax.axis("off")
                fig.suptitle(f"Rechazos {rule} — {split}", fontsize=12)
                plt.tight_layout()
                plt.show()
                continue

            fig, axes = plt.subplots(
                1, examples.height, figsize=(4 * examples.height, 4), squeeze=False
            )
            for ax, row in zip(axes.flat, examples.iter_rows(named=True)):
                try:
                    with Image.open(dataset_dir / row["path"]) as image:
                        ax.imshow(image.convert("RGB"))
                except (UnidentifiedImageError, OSError):
                    # Placeholder para archivos corruptos: los mostramos en gris para no romper la grilla.
                    ax.imshow(np.full((64, 64, 3), 128, dtype=np.uint8))
                ax.set_title(
                    f"{row['breed']}\n{row['width']}×{row['height']} px"
                    + (f"\nLaplaciano: {row['laplacian_var']}" if row["laplacian_var"] is not None else "")
                    + (f"\nBrillo: {row['brightness']}" if row["brightness"] is not None else ""),
                    fontsize=9,
                )
                ax.axis("off")

            fig.suptitle(f"Rechazos {rule} — {split}", fontsize=12)
            plt.tight_layout()
            plt.show()


show_rejected_examples(quality, DATASET_DIR)
