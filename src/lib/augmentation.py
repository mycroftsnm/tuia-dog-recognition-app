"""Data augmentation para el entrenamiento (albumentations 2.x).

Se aplica **solo** sobre `train`. Las transformaciones están pensadas para fotos de
perros: la pose y el encuadre varían, el pelaje no cambia de color y la imagen puede
venir borrosa o subexpuesta.

Quedan afuera las transformaciones que no ocurren en la realidad o que cambian la raza
aparente: flip vertical, rotaciones de 90° y cambios de color/hue extremos.

La entrada es BGR uint8, igual que en el resto del proyecto: el pipeline solo aplica
transformaciones geométricas y fotométricas neutras al orden de canales. Después de
augmentar, la imagen pasa por `image_pipeline.preprocess` como cualquier otra.
"""
from __future__ import annotations

import albumentations as A
import cv2

from lib.image_pipeline import IMAGE_SIZE


def build_train_transform(image_size: int = IMAGE_SIZE, seed: int | None = None) -> A.Compose:
    """Pipeline de augmentation para entrenamiento.

    Cada transformación está justificada por lo que pasa en el dominio:
      - flip horizontal: el perro puede mirar a cualquier lado.
      - affine (escala, traslación, rotación ±15°): encuadre y pose variables; ±15° no
        cambia la raza aparente.
      - random resized crop (70–100 %): simula recortes más cerrados (los de YOLO) sin
        cortar al perro entero.
      - brillo/contraste ±20 %: interiores, exteriores y distintas exposiciones.
      - blur (gaussiano o de movimiento): fotos movidas o desenfocadas.
      - ruido gaussiano: compresión y sensores de baja calidad.
    """
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.Affine(
                scale=(0.9, 1.1),
                translate_percent=(-0.05, 0.05),
                rotate=(-15, 15),
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.5,
            ),
            A.RandomResizedCrop(
                size=(image_size, image_size),
                scale=(0.7, 1.0),
                ratio=(0.85, 1.15),
                p=0.3,
            ),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
            A.OneOf(
                [A.GaussianBlur(blur_limit=(3, 7)), A.MotionBlur(blur_limit=7)],
                p=0.2,
            ),
            A.GaussNoise(std_range=(0.02, 0.08), p=0.2),
        ],
        seed=seed,
    )
