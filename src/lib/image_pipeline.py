"""Pipeline de imagen compartido entre la notebook de datos y los servicios de la app.

Es la **única** ruta de código para pasar de una imagen BGR uint8 (OpenCV) a la entrada
del modelo. Si la notebook entrena con un pipeline y el backend infiere con otro, el
modelo "anda en la notebook pero no en la app".

No confundir con `1-pre-processing.ipynb`: ese notebook hace el EDA y el filtrado del
dataset; este módulo define las transformaciones de imagen.

Convenciones:
  - Entrada: `np.ndarray` BGR uint8 (lo que devuelve `cv2.imread`).
  - Salida de `preprocess`: CHW float32 con normalización de ImageNet (RGB).

Que las imágenes de Kaggle ya sean 224×224 **no** vuelve innecesario el pipeline: el
letterbox es identidad para una imagen cuadrada de ese tamaño, pero BGR→RGB, el pasaje a
float32, la normalización de ImageNet y el orden HWC→CHW no son opcionales. Además, en
la Etapa 3 los recortes de YOLO y el conjunto independiente no son 224×224.
"""
from __future__ import annotations

import cv2
import numpy as np

IMAGE_SIZE = 224

# Estadísticas de ImageNet: las usan los modelos preentrenados de torchvision/timm.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """BGR (OpenCV) -> RGB. Escala de grises se replica a 3 canales; RGBA pierde alfa."""
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def letterbox(image: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    """Escala preservando aspecto y rellena con espejo hasta `size`x`size`.

    Es la identidad para una imagen ya cuadrada de `size`, así que no agrega nada a las
    imágenes de Kaggle pero sí a los recortes de YOLO. Se prefiere al resize directo
    (deforma el perro) y al center-crop (corta patas y orejas). El relleno es reflejo
    para no meter bordes negros que el modelo nunca vio.
    """
    height, width = image.shape[:2]
    if height == 0 or width == 0:
        raise ValueError("Imagen vacía")

    scale = size / max(height, width)
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    pad_top = (size - new_h) // 2
    pad_bottom = size - new_h - pad_top
    pad_left = (size - new_w) // 2
    pad_right = size - new_w - pad_left
    if pad_top or pad_bottom or pad_left or pad_right:
        resized = cv2.copyMakeBorder(
            resized, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT_101
        )
    return resized


def normalize(rgb: np.ndarray) -> np.ndarray:
    """float32 en [0, 1] y normalización de ImageNet (asume RGB)."""
    array = rgb.astype(np.float32) / 255.0
    return (array - IMAGENET_MEAN) / IMAGENET_STD


def to_chw(rgb: np.ndarray) -> np.ndarray:
    """HWC -> CHW, el orden que espera PyTorch."""
    return np.transpose(rgb, (2, 0, 1))


def preprocess(image_bgr: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    """Pipeline completo: BGR -> RGB -> letterbox -> normalización -> CHW float32."""
    return to_chw(normalize(letterbox(bgr_to_rgb(image_bgr), size))).astype(np.float32)


def to_batch(array_chw: np.ndarray) -> np.ndarray:
    """CHW -> 1CHW (dimensión de batch)."""
    return np.expand_dims(array_chw, axis=0)


def denormalize(array_chw: np.ndarray) -> np.ndarray:
    """Inversa de `preprocess`, para visualizar. Devuelve RGB uint8."""
    rgb = np.transpose(array_chw, (1, 2, 0)) * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)


def expand_box(
    box: tuple[int, int, int, int],
    height: int,
    width: int,
    ratio: float = 0.10,
) -> tuple[int, int, int, int]:
    """Agranda una caja (x1, y1, x2, y2) un `ratio` por lado, recortada a la imagen.

    Se usa con los recortes de YOLO (Etapa 3): el detector ajusta la caja al cuerpo y
    suele dejar afuera orejas, cola y patas, así que se le da un margen antes de
    clasificar. El margen imita el encuadre del dataset, donde el perro está completo.
    """
    x1, y1, x2, y2 = (int(v) for v in box)
    pad_x = int(round((x2 - x1) * ratio))
    pad_y = int(round((y2 - y1) * ratio))
    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(width, x2 + pad_x),
        min(height, y2 + pad_y),
    )
