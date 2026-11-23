"""Modelo B: CNN propia entrenada desde cero.

4 bloques convolucionales (Conv-BN-ReLU ×2 + MaxPool) → global average pooling →
capa de embedding → clasificador. La entrada es 224×224 RGB normalizada con ImageNet.

Igual que el Modelo A, la clase vive en un módulo importable porque el checkpoint se guarda
como modelo completo.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class ModeloBCNN(nn.Module):
    """CNN de 4 bloques con capa de embedding y clasificador de razas."""

    def __init__(self, num_classes: int, embedding_dim: int = 256, dropout: float = 0.3) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.features = nn.Sequential(
            _conv_block(3, 32),
            _conv_block(32, 64),
            _conv_block(64, 128),
            _conv_block(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, embedding_dim),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, num_classes),
        )
        self.class_names: list[str] = []

    def embed(self, images: torch.Tensor) -> torch.Tensor:
        """Vector de embedding (sin normalizar)."""
        return self.embedding(self.pool(self.features(images)))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.embed(images))

    def freeze_backbone(self) -> None:
        for parameter in self.features.parameters():
            parameter.requires_grad = False
