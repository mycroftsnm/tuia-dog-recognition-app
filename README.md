# TP2 - Sistema de Detección y Clasificación de Razas de Perros

Plantilla base para desarrollar un pipeline completo de Computer Vision:

**Embeddings → Búsqueda por similitud → Clasificación → Detección → Pipeline completo**

> 📄 El enunciado completo del trabajo práctico está en [ENUNCIADO.md](ENUNCIADO.md) (versión PDF: [Trabajo Práctico N°2.pdf](<Trabajo Práctico N°2.pdf>)).

La cátedra provee la infraestructura completa (Docker, aplicación Gradio, base de datos vectorial, API, scripts y herramientas de evaluación y visualización). El grupo hace un **fork** de este repositorio, trabaja en una **feature branch** y completa **únicamente** las funciones indicadas en cada etapa. Cualquier modificación fuera de esas funciones debe justificarse en el Pull Request.

## Objetivo del backend

API asincrónica en Python que permite:

- Búsqueda por similitud (`/search`): imagen consultada, Top-K imágenes similares y raza predicha.
- Clasificación supervisada (`/classify`): raza y score con el modelo entrenado seleccionado.
- Detección + clasificación (`/detect`): bounding boxes, raza y scores de confianza.
- Consulta del estado del procesamiento asincrónico (`/status/{job_id}`).
- Selección dinámica del modelo de embeddings (`/models` + campo `model` en `/search`).

La API responde `HTTP 202` con un `job_id` y luego permite consultar el resultado con estado:

```json
{
  "status": "done | inProgress | failed",
  "link": "url | none",
  "reason": "mensaje del error o null"
}
```

## Estructura

```text
tuia-dog-recognition-app/
├── 1-pre-processing.ipynb       # Parte 1: dataset y preprocesamiento
├── 2-train.ipynb                # Etapa 2: entrenamiento, evaluación (incl. NDCG@10 de la Etapa 1) y comparación
├── ENUNCIADO.md                 # Enunciado completo del TP
├── Trabajo Práctico N°2.pdf     # Enunciado en PDF
├── src/
│   ├── app/
│   │   └── main.py              # App FastAPI + /health
│   ├── frontend/                # UI Gradio provista por la cátedra (una pestaña por etapa)
│   │   ├── app.py
│   │   └── gradio_ui.py
│   └── lib/
│       ├── api.py               # Endpoints /upload, /search, /classify, /detect, /status, /models, /files
│       ├── bootstrap.py         # Construcción de los servicios (compartida por la API y los scripts)
│       ├── config.py            # Configuración vía .env
│       ├── files.py
│       ├── schemas.py
│       ├── evaluation/
│       │   └── metrics.py       # NDCG@k, precision/recall/F1, specificity
│       ├── visualization/
│       │   └── draw.py          # Dibujo de bounding boxes
│       ├── services/
│       │   ├── similarity_service.py   # ⚠️ Etapa 1: funciones a implementar
│       │   ├── classifier_service.py   # ⚠️ Etapa 2: funciones a implementar
│       │   ├── detection_service.py    # ⚠️ Etapa 3: funciones a implementar
│       │   └── task_manager.py
│       └── storage/
│           ├── base.py
│           ├── embedding_store.py      # Base vectorial en JSON
│           └── pgvector_store.py       # Base vectorial en PostgreSQL + pgvector
├── scripts/
│   ├── download_dataset.py      # Descarga el dataset de Kaggle
│   ├── build_index.py           # Indexa el dataset en la base vectorial
│   └── train_classifier.py      # Entrena/evalúa el clasificador (alternativa local a 2-train.ipynb)
├── data/
│   ├── dataset/                 # 70 Dog Breeds Image Dataset, no se versiona
│   └── embeddings.json          # Base vectorial JSON (si USE_PGVECTOR=false)
├── models/                      # Checkpoints entrenados (.pth / .onnx), no se versionan
├── output/                      # Imágenes subidas y resultados, no se versiona
├── requirements.txt
├── Dockerfile
├── Dockerfile.frontend
├── docker-compose.yml
├── .env.docker.example
└── .env.local.example
```

# Preparando el ambiente

## Requisitos

- Python 3.12
- Docker (Docker Desktop incluye Docker Compose)
- Git
- Acceso de lectura al ZIP de fotos independientes en Drive

Para probar la rama del trabajo desde una carpeta nueva, cloná el fork en esa rama.

```bash
git clone --branch feature/tp2-razas-perros https://github.com/mycroftsnm/tuia-dog-recognition-app.git
cd tuia-dog-recognition-app
```

Creá y activá un entorno de Python 3.12 como se indica en los pasos 2 a 4 de la opción local. Instalá las dependencias del proyecto y las que usan las notebooks en ese entorno.

```bash
uv pip install -r requirements.txt altair==6.3.0 polars==2.2.0 ipykernel==7.4.0
```

## Dataset

Dataset principal: [70 Dog Breeds Image Dataset (Kaggle)](https://www.kaggle.com/datasets/gpiosenka/70-dog-breedsimage-data-set).

```bash
python scripts/download_dataset.py
```

El script usa `kagglehub` para descargar el dataset público. Si Kaggle pide iniciar sesión, configurá tus [credenciales de Kaggle](https://www.kaggle.com/docs/api) o descargá el dataset desde la web y descomprimí `train/`, `valid/` y `test/` dentro de `data/dataset`.

Para evaluar con las fotos del equipo, descargá el ZIP con la cuenta de Drive que tiene acceso de lectura y extraelo en la raíz del repositorio, conservando las carpetas del ZIP. Incluye las 107 fotos en `data/dataset/independent/` y sus fuentes y licencias en `splits/independent_manifest.csv`. [Descargar fotos independientes y manifiesto](https://drive.google.com/file/d/1Mj_wpv2h8_Zr9gABHDXu_1_ZmsvkQcdW/view?usp=drive_link)

## Configurá tus modelos

Extraé el ZIP en la raíz del repositorio, conservando sus carpetas. Los clasificadores quedan en `models/resnet18_finetuned.pth` y `models/cnn_custom.pth`, donde los cargan la notebook de entrenamiento, el backend local y Docker. Los dos historiales quedan en `output/grupos-v1/`. La primera búsqueda con `baseline` descarga los pesos ImageNet si el indexador todavía no los descargó. La primera detección descarga `models/yolov8n.pt`. [Descargar modelos e historiales](https://drive.google.com/file/d/1IEZiJIOgRH36_3XirJiNXqvp3fCusKxU/view?usp=drive_link)

Después de subir una foto, elegí qué querés probar:

- **Búsqueda (`/search`).** Una ResNet18 preentrenada en ImageNet (`baseline`) genera un embedding de 512 dimensiones. Los diez vecinos más similares por coseno votan la raza, ponderados por su similitud. Si el mejor vecino queda por debajo de `0.674`, calibrado en validación, la app devuelve `unknown`. Con la misma galería para los tres extractores, la ResNet ajustada lideró en test (NDCG@10 de `0.8933`) y el baseline lideró en las 68 consultas independientes (`0.5156` frente a `0.4977` de la ResNet ajustada). Se priorizaron esas fotos por su cercanía al uso previsto. [Comparación de extractores en la notebook](2-train.ipynb#9-embeddings-de-los-modelos-entrenados).
- **Clasificación (`/classify`).** La ResNet18 ajustada predice la raza de la foto completa por defecto. Para comparar, seleccioná `cnn_custom` como modelo en la app o en la API. La ResNet obtuvo mayor F1 macro que la CNN tanto en test (`90,42 %` frente a `35,21 %`) como en el conjunto independiente (`57,19 %` frente a `16,00 %`). [Comparación y elección final en la notebook](2-train.ipynb#12-comparación-y-elección-final).
- **Detección (`/detect`).** YOLOv8n localiza los perros en la foto y la misma ResNet18 ajustada clasifica la raza de cada recorte. En las 24 fotos de validación, `0.15` localizó 10 de 21 perros, frente a 9 con `0.25` y 6 con `0.50`. [Comparación de umbrales en la notebook](2-train.ipynb#etapa-3--comparación-de-umbrales-del-detector-yolo).

Con docker compose, el entorno reinicia la aplicación y actualiza el código automáticamente al detectar cambios. Lo mismo ocurre en local si usan `--reload`.

El reinicio automático puede no funcionar en algunas versiones de Docker Desktop para *Windows*. En ese caso, vuelvan a correr los comandos del siguiente apartado para actualizar el código dentro de Docker.

## Opción 1 - Corriendo dentro de Docker

Es la forma en que la cátedra evalúa el trabajo.

### 1. Construí y corré la aplicación

Iniciá Docker Desktop o el motor de Docker antes de construir. Docker Compose usa `.env.docker.example` como configuración del backend. Desde la raíz del repositorio, corré estos comandos. La primera construcción descarga PyTorch y puede llevar varios minutos.

```bash
docker compose build
docker compose up -d
```

Abrí Gradio en `http://localhost:8080`. Verificá en `http://localhost:8000/health` que `status` sea `ok` y que ambos valores de `checkpoints` sean `true`. PostgreSQL queda disponible en `localhost:5432`.

Si el puerto 8080 está ocupado, podés ejecutar `$env:FRONTEND_PORT = "8081"` en PowerShell o `export FRONTEND_PORT=8081` en Bash antes de `docker compose up -d`. En ese caso abrí Gradio en `http://localhost:8081`.

### 2. Indexá la galería

Con el dataset en `data/dataset`, corré `1-pre-processing.ipynb` de principio a fin con el kernel del entorno de Python 3.12. También podés ejecutar el script desde la raíz del repositorio, con `MPLBACKEND=Agg` para que los gráficos no interrumpan la ejecución.

```powershell
$env:MPLBACKEND = "Agg"
python 1-pre-processing.py
```

En Bash, usá `MPLBACKEND=Agg python 1-pre-processing.py`. El preprocesamiento genera `splits/train_augmented.csv` y 4.107 copias en `data/dataset/augmented/train/`. Docker Compose monta el manifiesto y las fotos en el backend. [Registro de la partición](splits/README.md).

Indexá la galería con `EMBEDDING_MODEL=baseline` y `EMBEDDING_DIM=512`, tal como vienen en la configuración. Usá un índice vacío, ya que el comando agrega registros cada vez que se ejecuta. En CPU, la indexación completa puede llevar varios minutos. Desde la raíz del repositorio, ejecutá este comando.

```bash
docker compose exec backend python /scripts/build_index.py --split train
```

El comando debe terminar con `Total: 10500 embeddings almacenados (modelo: baseline)`. Podés confirmar la cantidad y el extractor guardado en la base de datos.

```bash
docker compose exec postgres psql -U dogs_user -d dogs -c "SELECT metadata->>'model' AS modelo, count(*) FROM embeddings GROUP BY 1;"
```

Para comparar con `resnet18_finetuned`, creá una base aparte con `docker compose exec postgres createdb -U dogs_user dogs_resnet_eval`. En `.env.docker.example`, configurá `POSTGRES_DB=dogs_resnet_eval`, `EMBEDDING_MODEL=resnet18_finetuned` y `EMBEDDING_DIM=512`. Ejecutá `docker compose restart backend` y repetí la indexación en esa base. Al terminar, restaurá `POSTGRES_DB=dogs` y `EMBEDDING_MODEL=baseline` y reiniciá el backend otra vez para volver a la galería original.

### 3. Probá las pestañas de Gradio

Cargá una foto de `data/dataset/independent/` y enviá la consulta con el botón de la etapa. Abrí **Estado y resultados**, donde ya figura el `job_id`, y presioná **Consultar**. En ventanas angostas, las pestañas que quedan fuera de la barra están en el menú **…**. Si aparece `en progreso`, volvé a presionar **Consultar**. Si la tarea falla, la pestaña muestra el motivo que devuelve `/status/{job_id}` en `reason`.

- **Búsqueda por similitud.** Seleccioná el extractor con el que indexaste la galería y dejá `Top K` en 10. El resultado muestra la consulta, los diez vecinos y la raza predicha.
- **Clasificación supervisada.** Probá `resnet18_finetuned` y `cnn_custom` sobre la misma foto. Compará la raza y el score después de consultar cada job.
- **Detección y clasificación.** Probá una foto con un perro, otra con perros superpuestos, una escena compleja y una foto sin perros. Las carpetas `escenas/` y `sin_perro/` del conjunto independiente incluyen esos casos. En el resultado aparecen las cajas sobre la foto y el detalle de detecciones en el JSON.

## Opción 2 - Configurando el ambiente local

### 1. Cloná tu fork

Usá el comando de clonación de [Requisitos](#requisitos).

### 2. Instalá uv

Para usuarios de Linux o Mac:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Para usuarios de Windows:

```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

[Documentación de uv](https://docs.astral.sh/uv/getting-started/installation/). También pueden usar `python3.12 -m venv .venv` y `pip`.

### 3. Configurá un ambiente virtual con Python 3.12

```bash
uv venv --python 3.12 .venv
```

### 4. Activá el ambiente virtual

```bash
# Linux / Mac
source .venv/bin/activate
```

```powershell
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

### 5. Instalá las dependencias

```bash
uv pip install -r requirements.txt altair==6.3.0 polars==2.2.0 ipykernel==7.4.0
```

### 6. Configurá las variables de entorno

Copiá `.env.local.example` a `src/.env`. El backend local y los scripts de `scripts/` leen la configuración desde ese archivo.

```bash
cp .env.local.example src/.env
```

### 7. Descargá el dataset

```bash
python scripts/download_dataset.py
```

### 8. Iniciá la base de datos

```bash
docker compose up postgres -d
```

Solo es necesaria con `USE_PGVECTOR=true` (por defecto). Con `USE_PGVECTOR=false` se usa la base vectorial JSON (`data/embeddings.json`) y no hace falta Docker para trabajar en local.

### 9. Indexá la galería

Generá el mismo manifiesto y las copias aumentadas indicados en el paso de indexación de Docker. Conservá `EMBEDDING_MODEL=baseline` y `EMBEDDING_DIM=512` en `src/.env` y corré este comando desde la raíz del repositorio.

```bash
python scripts/build_index.py --split train
```

### 10. Iniciá el backend

```bash
cd src
uvicorn app.main:app --reload --port 8000
```

### 11. Iniciá el frontend (en otra terminal, con el ambiente virtual activado)

```bash
cd src
uvicorn frontend.app:app --port 8080
```

### 12. Verificá que todo esté corriendo

- Frontend: `http://localhost:8080`
- Backend: `http://localhost:8000/docs` (Swagger) y `http://localhost:8000/health` (muestra el modelo seleccionado y los checkpoints encontrados)

## Entrenamiento local con CUDA en Windows

Con el driver NVIDIA y las dependencias del proyecto instalados, reemplazá PyTorch
y torchvision en la `.venv` por las versiones para CUDA 12.6. Ejecutá estos comandos en
PowerShell desde la raíz del repo.

```powershell
nvidia-smi
uv pip install --python .\.venv\Scripts\python.exe --index-url https://download.pytorch.org/whl/cu126 "torch==2.11.0+cu126" "torchvision==0.26.0+cu126"
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Las [versiones oficiales de PyTorch](https://pytorch.org/get-started/previous-versions/) incluyen el runtime CUDA. En `2-train.ipynb`, elegí el kernel de la `.venv` y reinicialo si estaba abierto. La primera celda indica el dispositivo detectado.

Para medir una época completa de cada modelo, incluida la validación, ejecutá estos comandos
en una terminal nueva desde la raíz. Usá un nombre de corrida nuevo para que el entrenamiento
genere otros checkpoints e historiales. El tamaño de lote es 32. Si CUDA se queda sin memoria,
podés probar con `$env:TRAIN_BATCH_SIZE = "16"` en una corrida nueva.

```powershell
$runId = "cuda-prueba-$(Get-Date -Format yyyyMMdd-HHmmss)"
$env:MODEL_PATH = Join-Path $PWD "models/$runId"
$env:OUTPUT_PATH = Join-Path $PWD "output/$runId"
$env:TORCH_HOME = Join-Path $PWD "models/torch"
$env:TRAIN_EPOCHS = "1"
$env:TRAIN_BATCH_SIZE = "32"
.\.venv\Scripts\python.exe scripts/train_classifier.py --model resnet18_finetuned --skip-eval
.\.venv\Scripts\python.exe scripts/train_classifier.py --model cnn_custom --skip-eval
```

El tiempo de cada época aparece en la terminal y en `history_<modelo>.json`. Para una corrida
completa, usá otro nombre de corrida y ajustá `TRAIN_EPOCHS` (12 por defecto). Si querés correr
la notebook con estas rutas y parámetros, iniciá Jupyter desde la misma terminal.

## Scripts provistos

Se ejecutan desde la raíz del repositorio con el ambiente local activo (y PostgreSQL corriendo si `USE_PGVECTOR=true`). Leen la configuración de `src/.env` si existe:

```bash
# Descargar el dataset de Kaggle a data/dataset
python scripts/download_dataset.py

# Indexar la galería en la base vectorial (requiere la Etapa 1 implementada)
python scripts/build_index.py --split train
# ... o lo mismo desde el contenedor del backend (con docker compose up -d corriendo)
docker compose exec backend python /scripts/build_index.py --split train
```

Para entrenar localmente con `scripts/train_classifier.py`, configurá un `MODEL_PATH` y un `OUTPUT_PATH` propios como en el ejemplo de entrenamiento con CUDA.

El indexador agrega registros a la base configurada. Para reconstruir una galería,
usá un índice vacío.

Con `USE_PGVECTOR=false`, la indexación del split completo es lenta: el JSON se reescribe en cada inserción. Para pruebas rápidas, usen `--limit` (máximo de imágenes por raza).

## Configuración

Los ejemplos de configuración local y de Docker incluyen los parámetros de búsqueda elegidos en la notebook.

```dotenv
EMBEDDING_MODEL=baseline
EMBEDDING_DIM=512
SIMILARITY_METRIC=cosine
TOP_K=10
SIMILARITY_THRESHOLD=0.674
```

La galería se guarda en PostgreSQL con pgvector por defecto. Con `USE_PGVECTOR=false`, se guarda en el archivo JSON indicado por `EMBEDDINGS_PATH`.

## Endpoints

- Backend: `http://localhost:8000` (documentación Swagger en `http://localhost:8000/docs`)
- PostgreSQL/pgvector: `localhost:5432`
- Frontend (provisto por la cátedra): `http://localhost:8080`

| Endpoint | Descripción |
|----------|-------------|
| `POST /upload` | Sube una imagen al servidor |
| `POST /search` | Etapa 1: búsqueda por similitud (acepta `model` y `top_k`) |
| `POST /classify` | Etapa 2: clasificación con el modelo entrenado (acepta `model`) |
| `POST /detect` | Etapa 3: detección + clasificación |
| `GET /status/{job_id}` | Estado del procesamiento asincrónico |
| `GET /models` | Modelos de embeddings disponibles |
| `GET /health` | Estado del backend y checkpoints encontrados |

## Pipeline

### Provisto por la cátedra

- API asincrónica (`/upload`, `/search`, `/classify`, `/detect`, `/status`, `/models`) y manejo de jobs.
- Orquestación de cada etapa: `search` e `index_image` (Etapa 1), `classify_image` (Etapa 2) y `predict` (Etapa 3: detección → recorte → clasificación → JSON).
- Selección dinámica del modelo de embeddings: `baseline`, `resnet18_finetuned` o `cnn_custom`.
- Carga de checkpoints (`.pth` / `.onnx`) desde `models/` y selección del modelo activo.
- Similitud configurable (`cosine` o `l2`) y manejo de desconocidos con `SIMILARITY_THRESHOLD`.
- Persistencia configurable en JSON o en PostgreSQL + pgvector (`USE_PGVECTOR`).
- Funciones auxiliares de evaluación (`lib/evaluation/metrics.py`) y de visualización (`lib/visualization/draw.py`).
- Frontend Gradio con una pestaña por etapa: Etapa 1 (imagen consultada, Top-K similares, raza predicha y selección de modelo), Etapa 2 (raza y score con el modelo entrenado) y Etapa 3 (bounding boxes, razas y scores).

### A implementar por el grupo

Estas funciones hoy lanzan `NotImplementedError`:

| Etapa | Función | Archivo |
|-------|---------|---------|
| 1 | `extract_embedding(image)` | `src/lib/services/similarity_service.py` |
| 1 | `search_similar_images(embedding, top_k)` | `src/lib/services/similarity_service.py` |
| 1 | `predict_breed_from_neighbors(results)` | `src/lib/services/similarity_service.py` |
| 2 | `train_classifier()` | `src/lib/services/classifier_service.py` |
| 2 | `evaluate_classifier()` | `src/lib/services/classifier_service.py` |
| 2 | `extract_custom_embedding(image)` | `src/lib/services/classifier_service.py` |
| 3 | `detect_dogs(image)` | `src/lib/services/detection_service.py` |
| 3 | `classify_detected_dog(crop)` | `src/lib/services/detection_service.py` |

En las Etapas 1 y 3 el trabajo se limita a completar las funciones. En la Etapa 2, además, el grupo entrena sus propios modelos (Modelo A: fine-tuning de ResNet18/ResNet50; Modelo B: CNN propia, **ambos obligatorios**) y hace un estudio de resultados.

Ver el contrato de cada función, recomendaciones y errores frecuentes en [ENUNCIADO.md](ENUNCIADO.md#24-funciones-a-implementar).

### Orden de trabajo sugerido

1. `1-pre-processing.ipynb`: EDA, filtrado, conjunto independiente, preprocesamiento, augmentation y splits.
2. Etapa 1 en `similarity_service.py`. Indexá la galería con `python scripts/build_index.py --split train` y probá la pestaña de la Etapa 1.
3. Etapa 2 en `classifier_service.py`, entrenando y comparando los modelos con `2-train.ipynb`. Para probar los pesos evaluados, seguí las rutas de la sección [Configurá tus modelos](#configurá-tus-modelos).
4. Etapa 3 en `detection_service.py` y validación end-to-end con `docker compose up`.

## Notebooks

| Notebook | Contenido |
|----------|-----------|
| `1-pre-processing.ipynb` | Dataset, EDA, control de calidad, conjunto independiente, preprocesamiento, data augmentation y splits |
| `2-train.ipynb` | Evaluación de la búsqueda (NDCG@10), entrenamiento del Modelo A y del Modelo B, métricas, análisis de errores, comparación y exportación de checkpoints |

Se pueden entregar commiteadas en el repositorio o en Google Colab, con el link a la notebook de Colab en una celda Markdown al inicio de la notebook del repositorio. En ambos casos deben estar **ejecutadas y con las salidas visibles**: una notebook sin salidas se considera **no entregada** (ver [ENUNCIADO.md](ENUNCIADO.md#82-notebooks-repositorio-o-google-colab)).

## Dataset

Documentar en la entrega final:

- Splits utilizados y cantidad de imágenes por raza en cada uno
- Reglas de filtrado y calidad, con la cantidad de imágenes descartadas por regla
- Conjunto independiente de evaluación: fuente, licencia, razas e imágenes por raza, y link de descarga
- Preprocesamiento y data augmentation aplicados

## Modelos y entrenamiento

Completar en la entrega final:

- Modelo de embeddings baseline (Etapa 1) y justificación
- Modelo A (fine-tuning) y Modelo B (CNN propia): arquitectura, estrategia y justificación
- Hiperparámetros y proceso de entrenamiento
- Umbral de similitud elegido y cómo se calibró
- Descarga e instalación de los checkpoints: [Configurá tus modelos](#configurá-tus-modelos).

### Evaluación de YOLO

La app usa `YOLO_CONF_THRESHOLD=0.15` para YOLOv8n (`models/yolov8n.pt`). Ese umbral tuvo menos omisiones en las 24 fotos de validación. Con el umbral elegido, la prueba independiente de 32 fotos registró una caja falsa sobre una persona y una caja duplicada. Las tres fotos sin perros devolvieron una lista vacía. [Comparación de umbrales del detector YOLO](2-train.ipynb#etapa-3--comparación-de-umbrales-del-detector-yolo).

## Resultados

Completar en la entrega final:

- Etapa 1: NDCG@10 por modelo de embeddings, sobre `test` y sobre el conjunto independiente
- Etapa 2: accuracy, precision, recall, specificity y F1 de ambos modelos, y matriz de confusión
- Comparación entre enfoques (búsqueda por similitud vs. clasificación supervisada; fine-tuning vs. CNN propia)
- Análisis de errores (falsos positivos y falsos negativos)

## Problemas encontrados y soluciones

Completar en la entrega final.

## Notas importantes

- Para usar `pgvector`, levanten `postgres` y definan `USE_PGVECTOR=true` en `.env`.
- `EMBEDDING_DIM` debe coincidir con la dimensión del embedding del modelo elegido. Si cambian `EMBEDDING_DIM`, la tabla `embeddings` se recrea y se borran los registros previos.
- La base vectorial guarda los embeddings de un modelo a la vez: al cambiar de modelo de embeddings hay que reindexar (`scripts/build_index.py`).
- Las rutas de la galería se guardan relativas a `DATA_PATH` (por ejemplo, `dataset/train/<raza>/<imagen>.jpg`): una base indexada desde el entorno local también sirve para el backend en Docker, porque comparten PostgreSQL y la carpeta `data/`.
- Los modelos (`*.pth`, `*.pt`, `*.onnx`) y los datos no se versionan (ver `.gitignore`). Para cargar los checkpoints evaluados, usá las rutas de la sección [Configurá tus modelos](#configurá-tus-modelos).
- Los pesos preentrenados (torchvision/timm y YOLO) se descargan en el primer uso, por lo que se necesita conexión a internet.

## Entrega

- Pull Request desde su feature branch hacia `main` **de su propio repositorio**, **sin mergear**, con el link subido al campus. No abran el PR contra el repositorio de la cátedra.
- Si el repositorio es privado, los docentes deben tener acceso.
- Notebooks `1-pre-processing.ipynb` y `2-train.ipynb` ejecutadas, con sus salidas.
- Para aprobar, el sistema debe andar sin errores con:

```bash
docker compose build
docker compose up
```

**Fecha de entrega:** a definir por la cátedra. Solo se consideran los commits hasta esa fecha y hora; se recomienda trabajar de forma incremental.

El detalle del contenido del PR, la plantilla de descripción y el checklist están en [ENUNCIADO.md](ENUNCIADO.md#8-entrega-pull-request).
