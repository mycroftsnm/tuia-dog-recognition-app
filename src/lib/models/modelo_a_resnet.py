"""Modelo A: ResNet18 preentrenada en ImageNet con cabeza de razas.

La clase tiene que vivir en un módulo importable: el backend carga los checkpoints con
`torch.load(path, weights_only=False)`, que necesita reconstruir el objeto completo.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class ModeloAResNet(nn.Module):
    """ResNet18 con `embed` (features de 512-D) y `forward` (logits de raza).

    - `embed(x)`: salida del *global average pooling* (512).
    - `forward(x)`: logits sobre las razas.
    - `class_names`: se completa al entrenar y viaja con el checkpoint.
    """

    def __init__(self, num_classes: int, pretrained: bool = True, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = models.resnet18(weights="DEFAULT" if pretrained else None)
        self.embedding_dim = self.net.fc.in_features
        self.net.fc = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.embedding_dim, num_classes),
        )
        self.class_names: list[str] = []

    def embed(self, images: torch.Tensor) -> torch.Tensor:
        """Vector de características (sin normalizar)."""
        return self.net(images)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.embed(images))

    def freeze_backbone(self) -> None:
        for parameter in self.net.parameters():
            parameter.requires_grad = False
