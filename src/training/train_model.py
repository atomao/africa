from __future__ import annotations

from typing import Optional, Callable, Literal, Dict, Any

import torch
from torch import nn
from torch.optim import AdamW, Adam, SGD, Optimizer
from lightning.pytorch import LightningModule
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryPrecision,
    BinaryRecall,
    BinaryF1Score,
    BinaryJaccardIndex,
)


class SegmentationModel(LightningModule):
    """
    Generic LightningModule for binary segmentation using TorchMetrics MetricCollection.

    Any model must return logits: (B, 1, H, W)
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        optimizer_name: Literal["adam", "adamw", "sgd"] = "adamw",
        loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        threshold: float = 0.5,
        device: Any = None,
    ):
        super().__init__()

        self.model = model
        self.save_hyperparameters(ignore=["model", "loss_fn"])

        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer_name
        self.threshold = threshold

        # Use BCEWithLogitsLoss as default
        self.loss_fn = loss_fn or nn.BCEWithLogitsLoss()

        self.model_device = device if device else "cpu"
        # --- TorchMetrics with MetricCollection ---
        metrics = MetricCollection(
            {
                # "accuracy": BinaryAccuracy(threshold=threshold).to(self.model_device),
                # "precision": BinaryPrecision(threshold=threshold).to(self.model_device),
                # "recall": BinaryRecall(threshold=threshold).to(self.model_device),
                "f1": BinaryF1Score(threshold=threshold).to(self.model_device),
                "iou": BinaryJaccardIndex(threshold=threshold).to(self.model_device),
            }
        )

        # Clone for each stage
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

    @staticmethod
    def _ensure_channel_dim(t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 3:  # (B, H, W)
            return t.unsqueeze(1)
        return t

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _shared_step(self, batch: Any, stage: str) -> torch.Tensor:
        images, masks = batch
        masks = masks.float()
        masks = self._ensure_channel_dim(masks)

        logits = self(images)
        loss = self.loss_fn(logits, masks)

        metric_collection = (
            self.train_metrics
            if stage == "train"
            else self.val_metrics
            if stage == "val"
            else self.test_metrics
        )

        metric_logs = metric_collection(logits, masks)

        self.log(
            f"{stage}_loss",
            loss,
            prog_bar=True,
            on_step=(stage == "train"),
            on_epoch=True,
        )

        # log all metrics (prefixes already included from clone)
        self.log_dict(metric_logs, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        images, _ = batch
        logits = self(images)
        probs = torch.sigmoid(logits)
        return probs

    def configure_optimizers(self):
        if self.optimizer_name == "adamw":
            opt = AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer_name == "adam":
            opt = Adam(self.parameters(), lr=self.lr)
        elif self.optimizer_name == "sgd":
            opt = SGD(
                self.parameters(),
                lr=self.lr,
                momentum=0.9,
                weight_decay=self.weight_decay,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")

        return opt
