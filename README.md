# TP2 - Sistema de Detección y Clasificación de Razas de Perros

**Alejandro Armas y Gabriel Soda**

La aplicación permite subir una foto en Gradio para buscar perros parecidos, estimar la raza de un perro o localizar y clasificar a varios perros en la misma escena. [Enunciado del TP](ENUNCIADO.md) · [Versión PDF](<Trabajo Práctico N°2.pdf>).

## Qué hace cada pestaña

- **Búsqueda por similitud (`/search`).** La ResNet18 preentrenada en ImageNet (`baseline`) convierte la foto en un vector de 512 componentes. La aplicación recupera por coseno las diez imágenes más parecidas de la galería y predice la raza mediante un voto ponderado por similitud. Si el mejor vecino queda por debajo del umbral calibrado, devuelve `unknown`.
- **Clasificación supervisada (`/classify`).** El clasificador toma la foto completa y devuelve la raza más probable y su score. La ResNet18 ajustada es el modelo predeterminado y la CNN propia está disponible para comparar.
- **Detección y clasificación (`/detect`).** YOLOv8n localiza los perros. La aplicación recorta cada caja y pasa el recorte a la ResNet18 ajustada. El resultado muestra las cajas sobre la foto, la raza predicha por recorte y los scores de detección y clasificación.

La cátedra aportó la interfaz Gradio, la API FastAPI con procesamiento asincrónico, la orquestación de las etapas, el almacenamiento vectorial, Docker y las herramientas de métricas y visualización. [Alcance de la infraestructura provista](ENUNCIADO.md#23-lo-que-ya-está-resuelto).

Como grupo, preparamos los datos y la galería, implementamos la extracción de embeddings, la búsqueda y el voto de vecinos, entrenamos y evaluamos los dos clasificadores e integramos YOLO con la clasificación de recortes. [Servicio de similitud](src/lib/services/similarity_service.py) · [Servicio de clasificación](src/lib/services/classifier_service.py) · [Servicio de detección](src/lib/services/detection_service.py).

## Datos y partición

En las carpetas de origen de Kaggle encontramos versiones de una misma foto con cambios de escala o encuadre. Por ejemplo, `valid/Afghan/05.jpg` y `train/Afghan/012.jpg` tienen bytes distintos y repiten el perro, el fondo y detalles del borde. Si quedan en particiones distintas, el modelo puede aprender de una versión y ser evaluado con otra, lo que puede sobreestimar el rendimiento en fotos nuevas. [Ejemplo y auditoría visual](1-pre-processing.ipynb#37-partición-de-datos).

Después del filtro de calidad, agrupamos los originales vinculados por coincidencias confirmadas y asignamos cada grupo completo a una sola partición. Excluimos los grupos con etiquetas de raza discordantes. La auditoría final registró cero grupos confirmados repartidos entre particiones y cero candidatos pendientes. Recién después generamos aumentos a partir de los originales de entrenamiento. [Registro de la partición](splits/README.md).

- **Entrenamiento y galería.** Los 6.393 originales de `train` más 4.107 aumentos forman el conjunto de 10.500 imágenes que usamos para entrenar los clasificadores y construir la galería de búsqueda. [Manifiesto de originales](splits/train.csv) · [Manifiesto con aumentos](splits/train_augmented.csv).
- **Validación y test.** Cada uno contiene 1.371 originales. Usamos validación para elegir los checkpoints y calibrar los umbrales. Test mide el rendimiento en fotos reservadas del mismo dataset. [Validación](splits/valid.csv) · [Test](splits/test.csv).
- **Conjunto independiente.** Reunimos 107 fotos propias y de fuentes externas. Las 68 consultas de un solo perro permiten medir qué tan bien se recuperan vecinos de la raza correcta y se clasifica la foto completa fuera de Kaggle. La muestra de 32 fotos incluye perros solos, superpuestos, escenas complejas y fotos sin perros. Permite evaluar si YOLO localiza cada perro y si la ResNet acierta la raza de los perros localizados. [Manifiesto de fotos, fuentes y licencias](splits/independent_manifest.csv) · [Consultas de un perro](splits/independent_eval.csv) · [Muestra de detección](2-train.ipynb#detector-y-clasificador-en-el-conjunto-independiente).

## Cómo probar la aplicación

### Requisitos

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

### Descargá las fotos

Dataset principal: [70 Dog Breeds Image Dataset (Kaggle)](https://www.kaggle.com/datasets/gpiosenka/70-dog-breedsimage-data-set).

```bash
python scripts/download_dataset.py
```

El script usa `kagglehub` para descargar el dataset público. Si Kaggle pide iniciar sesión, configurá tus [credenciales de Kaggle](https://www.kaggle.com/docs/api) o descargá el dataset desde la web y descomprimí `train/`, `valid/` y `test/` dentro de `data/dataset`.

Para probar con las fotos del equipo, descargá el ZIP con la cuenta de Drive que tiene acceso de lectura y extraelo en la raíz del repositorio, conservando las carpetas del ZIP. Las fotos quedan en `data/dataset/independent/`. [Descargar fotos independientes y manifiesto](https://drive.google.com/file/d/1Mj_wpv2h8_Zr9gABHDXu_1_ZmsvkQcdW/view?usp=drive_link)

### Configurá tus modelos

Extraé el ZIP en la raíz del repositorio, conservando sus carpetas. Los clasificadores quedan en `models/resnet18_finetuned.pth` y `models/cnn_custom.pth`, donde los cargan la notebook de entrenamiento, el backend local y Docker. Los dos historiales quedan en `output/grupos-v1/`. La primera búsqueda con `baseline` descarga los pesos ImageNet si el indexador todavía no los descargó. La primera detección descarga `models/yolov8n.pt`. [Descargar modelos e historiales](https://drive.google.com/file/d/1IEZiJIOgRH36_3XirJiNXqvp3fCusKxU/view?usp=drive_link)

Con docker compose, el entorno reinicia la aplicación y actualiza el código automáticamente al detectar cambios. Lo mismo ocurre en local si usan `--reload`.

El reinicio automático puede no funcionar en algunas versiones de Docker Desktop para *Windows*. En ese caso, vuelvan a correr los comandos del siguiente apartado para actualizar el código dentro de Docker.

### Opción 1 - Corriendo dentro de Docker

Es la forma en que la cátedra evalúa el trabajo.

#### 1. Construí y corré la aplicación

Iniciá Docker Desktop o el motor de Docker antes de construir. Docker Compose usa `.env.docker.example` como configuración del backend. Desde la raíz del repositorio, corré estos comandos. La primera construcción descarga PyTorch y puede llevar varios minutos.

```bash
docker compose build
docker compose up -d
```

Abrí Gradio en `http://localhost:8080`. Verificá en `http://localhost:8000/health` que `status` sea `ok` y que ambos valores de `checkpoints` sean `true`. PostgreSQL queda disponible en `localhost:5432`.

Si el puerto 8080 está ocupado, podés ejecutar `$env:FRONTEND_PORT = "8081"` en PowerShell o `export FRONTEND_PORT=8081` en Bash antes de `docker compose up -d`. En ese caso abrí Gradio en `http://localhost:8081`.

#### 2. Indexá la galería

Con el dataset en `data/dataset`, corré `1-pre-processing.ipynb` de principio a fin con el kernel del entorno de Python 3.12. También podés ejecutar el script desde la raíz del repositorio, con `MPLBACKEND=Agg` para que los gráficos no interrumpan la ejecución.

```powershell
$env:MPLBACKEND = "Agg"
python 1-pre-processing.py
```

En Bash, usá `MPLBACKEND=Agg python 1-pre-processing.py`. El preprocesamiento genera `splits/train_augmented.csv` y las copias en `data/dataset/augmented/train/`. Docker Compose monta el manifiesto y las fotos en el backend. [Registro de la partición](splits/README.md).

Indexá la galería con `EMBEDDING_MODEL=baseline` y `EMBEDDING_DIM=512`, tal como vienen en la configuración. Usá un índice vacío, ya que el comando agrega registros cada vez que se ejecuta. En CPU, la indexación completa puede llevar varios minutos. Desde la raíz del repositorio, ejecutá este comando.

```bash
docker compose exec backend python /scripts/build_index.py --split train
```

El comando debe terminar con `Total: 10500 embeddings almacenados (modelo: baseline)`. Podés confirmar la cantidad y el extractor guardado en la base de datos.

```bash
docker compose exec postgres psql -U dogs_user -d dogs -c "SELECT metadata->>'model' AS modelo, count(*) FROM embeddings GROUP BY 1;"
```

Para comparar con `resnet18_finetuned`, creá una base aparte con `docker compose exec postgres createdb -U dogs_user dogs_resnet_eval`. En `.env.docker.example`, configurá `POSTGRES_DB=dogs_resnet_eval`, `EMBEDDING_MODEL=resnet18_finetuned` y `EMBEDDING_DIM=512`. Ejecutá `docker compose restart backend` y repetí la indexación en esa base. Al terminar, restaurá `POSTGRES_DB=dogs` y `EMBEDDING_MODEL=baseline` y reiniciá el backend otra vez para volver a la galería original.

#### 3. Probá las pestañas de Gradio

Cargá una foto de `data/dataset/independent/` y enviá la consulta con el botón de la etapa. Abrí **Estado y resultados**, donde ya figura el `job_id`, y presioná **Consultar**. En ventanas angostas, las pestañas que quedan fuera de la barra están en el menú **…**. Si aparece `en progreso`, volvé a presionar **Consultar**. Si la tarea falla, la pestaña muestra el motivo que devuelve `/status/{job_id}` en `reason`.

- **Búsqueda por similitud.** Seleccioná el extractor con el que indexaste la galería y dejá `Top K` en 10. El resultado muestra la consulta, los diez vecinos y la raza predicha.
- **Clasificación supervisada.** Probá `resnet18_finetuned` y `cnn_custom` sobre la misma foto. Compará la raza y el score después de consultar cada job.
- **Detección y clasificación.** Probá una foto con un perro, otra con perros superpuestos, una escena compleja y una foto sin perros. Las carpetas `escenas/` y `sin_perro/` del conjunto independiente incluyen esos casos. En el resultado aparecen las cajas sobre la foto y el detalle de detecciones en el JSON.

### Opción 2 - Configurando el ambiente local

#### 1. Cloná tu fork

Usá el comando de clonación de [Requisitos](#requisitos).

#### 2. Instalá uv

Para usuarios de Linux o Mac:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Para usuarios de Windows:

```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

[Documentación de uv](https://docs.astral.sh/uv/getting-started/installation/). También pueden usar `python3.12 -m venv .venv` y `pip`.

#### 3. Configurá un ambiente virtual con Python 3.12

```bash
uv venv --python 3.12 .venv
```

#### 4. Activá el ambiente virtual

```bash
# Linux / Mac
source .venv/bin/activate
```

```powershell
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

#### 5. Instalá las dependencias

```bash
uv pip install -r requirements.txt altair==6.3.0 polars==2.2.0 ipykernel==7.4.0
```

#### 6. Configurá las variables de entorno

Copiá `.env.local.example` a `src/.env`. El backend local y los scripts de `scripts/` leen la configuración desde ese archivo.

```bash
cp .env.local.example src/.env
```

#### 7. Descargá el dataset

```bash
python scripts/download_dataset.py
```

#### 8. Iniciá la base de datos

```bash
docker compose up postgres -d
```

Solo es necesaria con `USE_PGVECTOR=true` (por defecto). Con `USE_PGVECTOR=false` se usa la base vectorial JSON (`data/embeddings.json`) y no hace falta Docker para trabajar en local.

#### 9. Indexá la galería

Generá el mismo manifiesto y las copias aumentadas indicados en el paso de indexación de Docker. Conservá `EMBEDDING_MODEL=baseline` y `EMBEDDING_DIM=512` en `src/.env` y corré este comando desde la raíz del repositorio.

```bash
python scripts/build_index.py --split train
```

#### 10. Iniciá el backend

```bash
cd src
uvicorn app.main:app --reload --port 8000
```

#### 11. Iniciá el frontend (en otra terminal, con el ambiente virtual activado)

```bash
cd src
uvicorn frontend.app:app --port 8080
```

#### 12. Verificá que todo esté corriendo

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

# Indexar la galería en la base vectorial
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

Las tres operaciones de la aplicación devuelven `HTTP 202` con un `job_id`. La consulta de estado devuelve `inProgress`, `done` con el enlace al resultado o `failed` con el motivo del error.

## Notebooks

| Notebook | Contenido |
|----------|-----------|
| `1-pre-processing.ipynb` | Dataset, EDA, control de calidad, conjunto independiente, preprocesamiento, data augmentation y splits |
| `2-train.ipynb` | Evaluación de la búsqueda (NDCG@10), entrenamiento del Modelo A y del Modelo B, métricas, análisis de errores, comparación y exportación de checkpoints |

Se pueden entregar commiteadas en el repositorio o en Google Colab, con el link a la notebook de Colab en una celda Markdown al inicio de la notebook del repositorio. En ambos casos deben estar **ejecutadas y con las salidas visibles**: una notebook sin salidas se considera **no entregada** (ver [ENUNCIADO.md](ENUNCIADO.md#82-notebooks-repositorio-o-google-colab)).

## Modelos y entrenamiento

Completar en la entrega final:

- Modelo de embeddings baseline (Etapa 1) y justificación
- Modelo A (fine-tuning) y Modelo B (CNN propia): arquitectura, estrategia y justificación
- Hiperparámetros y proceso de entrenamiento
- Umbral de similitud elegido y cómo se calibró
- Descarga e instalación de los checkpoints: [Configurá tus modelos](#configurá-tus-modelos).

### Evaluación de YOLO

La app usa `YOLO_CONF_THRESHOLD=0.15` para YOLOv8n (`models/yolov8n.pt`). En las 24 fotos de validación, localizó 10 de los 21 perros, frente a 9 con `0.25` y 6 con `0.50`. Con el umbral elegido, la muestra independiente registró una caja falsa sobre una persona y una caja duplicada. Las tres fotos sin perros devolvieron una lista vacía. [Comparación de umbrales del detector YOLO](2-train.ipynb#etapa-3--comparación-de-umbrales-del-detector-yolo).

## Resultados

- **Búsqueda por similitud.** Con la misma galería de fotos para los tres extractores, la ResNet ajustada lideró en test (NDCG@10 de `0.8933`) y el baseline lideró en el independiente (`0.5156` frente a `0.4977` de la ResNet ajustada). Priorizamos esas fotos por su cercanía al uso previsto. [Comparación de extractores en la notebook](2-train.ipynb#9-embeddings-de-los-modelos-entrenados).
- **Clasificación supervisada.** La ResNet obtuvo mayor F1 macro que la CNN tanto en test (`90,42 %` frente a `35,21 %`) como en el conjunto independiente (`57,19 %` frente a `16,00 %`). [Comparación y elección final en la notebook](2-train.ipynb#12-comparación-y-elección-final).

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
