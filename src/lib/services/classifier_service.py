from __future__ import annotations

import csv
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from lib.evaluation.metrics import precision_recall_f1, specificity
from lib.image_pipeline import preprocess
from lib.models import ModeloAResNet, ModeloBCNN

logger = logging.getLogger(__name__)


# Constantes de entrenamiento (documentadas en la notebook; no se tocan por corrida).
SEED = 42
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1
DROPOUT = 0.2


@dataclass
class TrainingConfig:
    """Hiperparámetros que sí cambian entre corridas (se configuran por `.env`)."""

    epochs: int = 12
    batch_size: int = 32
    learning_rate: float = 3e-4
    patience: int = 3
    strategy: str = "full"  # full | feature_extraction


class _ManifestDataset(Dataset):
    """Dataset de razas a partir de un manifiesto de `splits/`.

    Cada fila tiene `path` (relativo a `DATASET_PATH`) y `breed`. La imagen se lee con
    OpenCV (BGR uint8), se aumenta solo en train y se preprocesa con el pipeline compartido.
    """

    def __init__(
        self,
        rows: list[dict[str, str]],
        dataset_path: Path,
        class_index: dict[str, int],
        image_size: int,
        transform: Any | None = None,
    ) -> None:
        self.rows = rows
        self.dataset_path = dataset_path
        self.class_index = class_index
        self.image_size = image_size
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows[index]
        image = cv2.imread(str(self.dataset_path / row["path"]))
        if image is None:
            raise FileNotFoundError(f"No se pudo leer la imagen: {row['path']}")
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        tensor = torch.from_numpy(preprocess(image, self.image_size))
        return tensor, self.class_index[row["breed"]]


class ClassifierService:
    """Etapa 2: entrenamiento y comparacion de modelos de clasificacion.

    Funciones a implementar por el estudiante:
      - train_classifier()
      - evaluate_classifier()
      - extract_custom_embedding(image)

    La carga de checkpoints (.pth / .onnx) y la seleccion del modelo activo
    ya estan provistas.
    """

    def __init__(
        self,
        checkpoints: dict[str, Path],
        image_size: int,
        dataset_path: Path,
        output_path: Path,
        active_model: str = "resnet18_finetuned",
        splits_path: Path = Path("splits"),
        embedding_dim: int = 512,
        training: TrainingConfig | None = None,
        num_workers: int = 2,
    ) -> None:
        # checkpoints: nombre logico -> ruta del archivo (ej. resnet18_finetuned -> models/resnet18_finetuned.pth)
        self.checkpoints = checkpoints
        self.image_size = image_size
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.active_model_name = active_model
        self.splits_path = splits_path
        self.embedding_dim = embedding_dim
        self.training = training or TrainingConfig()
        self.num_workers = num_workers
        self._loaded: dict[str, Any] = {}
        # Última evaluación: la notebook saca de acá la matriz de confusión y las métricas por raza.
        self.last_evaluation: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Infraestructura provista
    # ------------------------------------------------------------------

    def set_active_model(self, name: str) -> None:
        """Define que checkpoint usan extract_custom_embedding y la clasificacion.

        Valores esperados: resnet18_finetuned | cnn_custom.
        """
        if name not in self.checkpoints:
            raise ValueError(f"Unknown model '{name}'. Expected one of: {sorted(self.checkpoints)}")
        self.active_model_name = name

    @property
    def active_checkpoint(self) -> Path:
        return self.checkpoints[self.active_model_name]

    def load_model(self, name: str | None = None) -> Any:
        """Carga (con cache) el checkpoint del modelo indicado o del activo.

        Soporta modelos PyTorch (.pth) y exportados a ONNX (.onnx).
        """
        key = name or self.active_model_name
        if key in self._loaded:
            return self._loaded[key]
        path = self.checkpoints[key]
        if not path.exists():
            raise ValueError(
                f"Checkpoint not found: {path}. Entrena el modelo (Etapa 2) y guardalo en esa ruta."
            )
        suf = path.suffix.lower()
        if suf == ".pth":
            model = torch.load(path, map_location="cpu", weights_only=False)
        elif suf == ".onnx":
            model = onnxruntime.InferenceSession(str(path))
        else:
            raise ValueError(f"Unsupported model format (expected .pth or .onnx): {path}")
        self._loaded[key] = model
        return model

    # ------------------------------------------------------------------
    # Helpers de entrenamiento
    # ------------------------------------------------------------------

    @staticmethod
    def _device() -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def _seed_everything(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _load_manifest(self, name: str) -> list[dict[str, str]]:
        """Lee `splits/<name>.csv` (una fila por imagen con `path` y `breed`)."""
        path = self.splits_path / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"No existe {path}. Ejecutá 1-pre-processing.py para generarlo."
            )
        with path.open(encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    def _class_names(self) -> list[str]:
        """Razas en el orden que usa el clasificador (una fila por raza en la matriz)."""
        return sorted({row["breed"] for row in self._load_manifest("train_augmented")})

    def _build_model(self, num_classes: int) -> nn.Module:
        if self.active_model_name == "resnet18_finetuned":
            model: nn.Module = ModeloAResNet(
                num_classes=num_classes, dropout=DROPOUT
            )
        else:
            model = ModeloBCNN(
                num_classes=num_classes,
                embedding_dim=self.embedding_dim,
                dropout=DROPOUT,
            )
        if self.training.strategy == "feature_extraction":
            model.freeze_backbone()
        return model

    def _make_loader(
        self, rows: list[dict[str, str]], class_index: dict[str, int], train: bool
    ) -> DataLoader:
        transform = None
        if train:
            from lib.augmentation import build_train_transform

            transform = build_train_transform(self.image_size, seed=SEED)
        dataset = _ManifestDataset(
            rows, self.dataset_path, class_index, self.image_size, transform
        )
        return DataLoader(
            dataset,
            batch_size=self.training.batch_size,
            shuffle=train,
            num_workers=self.num_workers,
            pin_memory=self._device().type == "cuda",
        )

    def _run_loader(
        self,
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> dict[str, Any]:
        """Una pasada completa: entrena si hay `optimizer`; si no, evalúa."""
        model.train(optimizer is not None)
        if optimizer is not None and self.training.strategy == "feature_extraction":
            # Con el backbone congelado, BatchNorm no debe actualizar sus estadísticas.
            for module in model.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()

        total_loss, total, correct = 0.0, 0, 0
        predictions: list[int] = []
        labels: list[int] = []
        with torch.set_grad_enabled(optimizer is not None):
            for images, targets in loader:
                images, targets = images.to(device), targets.to(device)
                logits = model(images)
                loss = criterion(logits, targets)
                if optimizer is not None:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                total_loss += float(loss.item()) * targets.size(0)
                total += targets.size(0)
                predicted = logits.argmax(dim=1)
                correct += int((predicted == targets).sum().item())
                predictions.extend(predicted.detach().cpu().tolist())
                labels.extend(targets.detach().cpu().tolist())
        return {
            "loss": total_loss / max(1, total),
            "accuracy": correct / max(1, total),
            "preds": predictions,
            "labels": labels,
        }

    @staticmethod
    def _metrics(
        labels: list[int], preds: list[int], num_classes: int
    ) -> tuple[dict[str, float], np.ndarray]:
        """Métricas macro (una clase = una raza) y matriz de confusión."""
        matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
        np.add.at(matrix, (np.asarray(labels), np.asarray(preds)), 1)

        total = int(matrix.sum())
        precisions, recalls, f1s, specificities = [], [], [], []
        for index in range(num_classes):
            tp = int(matrix[index, index])
            fp = int(matrix[:, index].sum()) - tp
            fn = int(matrix[index, :].sum()) - tp
            tn = total - tp - fp - fn
            precision, recall, f1 = precision_recall_f1(tp, fp, fn)
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
            specificities.append(specificity(tn, fp))

        metrics = {
            "accuracy": float(np.trace(matrix) / max(1, total)),
            "precision": float(np.mean(precisions)),
            "recall": float(np.mean(recalls)),
            "specificity": float(np.mean(specificities)),
            "f1": float(np.mean(f1s)),
        }
        return metrics, matrix

    # ------------------------------------------------------------------
    # Etapa 2: funciones a implementar
    # ------------------------------------------------------------------

    def train_classifier(self) -> None:
        """Entrena el modelo activo y guarda el mejor checkpoint según validación.

        Usa `train_augmented.csv` (originales + copias que balancean las clases) y
        `valid.csv` para early stopping. Guarda el historial por época en
        `output/history_<modelo>.json` para graficar las curvas en la notebook.
        """
        config = self.training
        self._seed_everything(SEED)

        train_rows = self._load_manifest("train_augmented")
        valid_rows = self._load_manifest("valid")
        class_names = self._class_names()
        class_index = {name: index for index, name in enumerate(class_names)}
        device = self._device()

        train_loader = self._make_loader(train_rows, class_index, train=True)
        valid_loader = self._make_loader(valid_rows, class_index, train=False)

        model = self._build_model(len(class_names)).to(device)
        model.class_names = class_names  # viaja con el checkpoint
        criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=config.learning_rate,
            weight_decay=WEIGHT_DECAY,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

        checkpoint = self.active_checkpoint
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        history: list[dict[str, float]] = []
        best_f1, best_epoch, epochs_without_improvement = -1.0, 0, 0

        logger.info(
            "Entrenando %s: %s train | %s valid | %s clases | %s (%s)",
            self.active_model_name,
            len(train_rows),
            len(valid_rows),
            len(class_names),
            device,
            config.strategy,
        )

        for epoch in range(1, config.epochs + 1):
            train_result = self._run_loader(model, train_loader, criterion, device, optimizer)
            valid_result = self._run_loader(model, valid_loader, criterion, device)
            valid_metrics, _ = self._metrics(
                valid_result["labels"], valid_result["preds"], len(class_names)
            )
            scheduler.step()

            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_result["loss"],
                    "train_accuracy": train_result["accuracy"],
                    "valid_loss": valid_result["loss"],
                    "valid_accuracy": valid_metrics["accuracy"],
                    "valid_f1": valid_metrics["f1"],
                }
            )
            logger.info(
                "época %02d | train loss %.4f acc %.3f | valid loss %.4f acc %.3f f1 %.3f",
                epoch,
                train_result["loss"],
                train_result["accuracy"],
                valid_result["loss"],
                valid_metrics["accuracy"],
                valid_metrics["f1"],
            )

            if valid_metrics["f1"] > best_f1:
                best_f1, best_epoch, epochs_without_improvement = valid_metrics["f1"], epoch, 0
                torch.save(model, checkpoint)
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= config.patience:
                    logger.info("Early stopping en la época %s (sin mejora en validación).", epoch)
                    break

        history_path = self.output_path / f"history_{self.active_model_name}.json"
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history_path.write_text(
            json.dumps(
                {
                    "model": self.active_model_name,
                    "strategy": config.strategy,
                    "best_epoch": best_epoch,
                    "best_valid_f1": best_f1,
                    "class_names": class_names,
                    "epochs": history,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        # El checkpoint cacheado quedó viejo: se recarga al evaluar.
        self._loaded.pop(self.active_model_name, None)
        logger.info("Historial guardado en %s", history_path)

    def evaluate_classifier(self) -> dict[str, float]:
        """Evalúa el modelo activo sobre `splits/test.csv`.

        Devuelve accuracy, precision, recall, specificity y F1 (macro) y deja en
        `self.last_evaluation` la matriz de confusión y las métricas por raza.
        """
        model = self.load_model()
        device = self._device()
        model.to(device)
        class_names = model.class_names
        rows = self._load_manifest("test")
        class_index = {name: index for index, name in enumerate(class_names)}

        loader = self._make_loader(rows, class_index, train=False)
        result = self._run_loader(model, loader, nn.CrossEntropyLoss(), device)
        metrics, matrix = self._metrics(result["labels"], result["preds"], len(class_names))

        support = matrix.sum(axis=1)
        self.last_evaluation = {
            "model": self.active_model_name,
            "metrics": metrics,
            "class_names": class_names,
            "confusion_matrix": matrix.tolist(),
            "per_class": [
                {
                    "breed": breed,
                    "support": int(support[index]),
                    "accuracy": float(matrix[index, index] / support[index])
                    if support[index]
                    else 0.0,
                }
                for index, breed in enumerate(class_names)
            ],
        }
        logger.info("Evaluación %s: %s", self.active_model_name, metrics)
        return metrics

    def extract_custom_embedding(self, image: np.ndarray) -> list[float]:
        """Embedding del modelo activo (capa previa al clasificador), L2-normalizado.
        """
        model = self.load_model()
        device = self._device()
        model.to(device)
        model.eval()
        tensor = torch.from_numpy(preprocess(image, self.image_size)).unsqueeze(0).to(device)
        with torch.no_grad():
            vector = model.embed(tensor).squeeze(0).cpu().numpy().astype(np.float32)
        vector = vector / max(float(np.linalg.norm(vector)), 1e-12)
        return vector.astype(float).tolist()
