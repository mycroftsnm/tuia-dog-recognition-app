"""Arquitecturas entrenadas por el grupo: Modelo A (ResNet18) y Modelo B (CNN propia)."""
from lib.models.modelo_a_resnet import ModeloAResNet
from lib.models.modelo_b_cnn import ModeloBCNN

__all__ = ["ModeloAResNet", "ModeloBCNN"]
