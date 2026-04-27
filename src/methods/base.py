"""Protocol every training method follows.

A method bundles: how to build the model, how to (optionally) preprocess the
data once before training, what one training step does, and how to compute
the final test-set metrics. The Hydra entry (`src/train.py`) treats every
method behind this same surface, so adding a new method = drop a file in
`src/methods/` and a YAML in `conf/method/`.

`vanilla` and `iceberg` defer model construction to `cfg.model`. `cg3` is a
coupled architecture+training recipe so it ignores `cfg.model` and builds
its own model in `build_model`.
"""

from __future__ import annotations

import abc
from typing import Any, Dict, Optional, Tuple

import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


class BaseMethod(abc.ABC):
    def __init__(self, cfg):
        self.cfg = cfg

    @abc.abstractmethod
    def build_model(self, in_channels: int, num_classes: int, *, data=None) -> torch.nn.Module:
        """Build and return the trainable model."""

    def prepare(self, model: torch.nn.Module, data) -> Any:
        """One-shot pre-training preprocessing. Default: delegate to the model
        if it has a `prepare` hook (e.g. Diff's feature propagation).
        Override for method-level preprocessing (e.g. CG3's hierarchy build)."""
        if hasattr(model, "prepare"):
            return model.prepare(data)
        return data

    def build_optimizer(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            model.parameters(),
            lr=float(self.cfg.method.lr),
            weight_decay=float(self.cfg.method.weight_decay),
        )

    @abc.abstractmethod
    def train_step(self, model: torch.nn.Module, data, optimizer: torch.optim.Optimizer,
                   epoch: int) -> Dict[str, float]:
        """One training step. Returns a dict of scalars to log
        (e.g. {'train_loss': ..., 'train_acc': ...}). Must call backward+step."""

    def predict_logits(self, model: torch.nn.Module, data) -> torch.Tensor:
        """Forward pass returning class logits. Override if the model's
        forward returns more than logits (e.g. CG3 returns a 4-tuple)."""
        model.eval()
        with torch.no_grad():
            return model(data.x, data.edge_index)

    def evaluate(self, model: torch.nn.Module, data, mask: Optional[torch.Tensor] = None) -> Tuple[float, float, float, float, float]:
        """5-tuple matching `eval/evaluation.py::evaluate` for parity with
        the existing summary tables."""
        if mask is None:
            mask = data.test_mask
        logits = self.predict_logits(model, data)
        pred = logits.argmax(dim=1)
        y_true = data.y[mask].cpu().numpy()
        y_pred = pred[mask].cpu().numpy()
        if len(y_true) == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
        recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
        f1_micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
        return acc, precision, recall, f1_macro, f1_micro

    def validate(self, model: torch.nn.Module, data) -> Dict[str, float]:
        """Per-epoch validation snapshot used for early stopping / W&B logging.
        Default: train_acc + val_acc/val_loss on the standard val_mask.
        Methods can override (e.g. iceberg uses (val_acc + val_f1) / 2)."""
        import torch.nn.functional as F
        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
            pred = logits.argmax(dim=1)
            out = {
                "train_acc": float((pred[data.train_mask] == data.y[data.train_mask]).float().mean().item()),
            }
            if hasattr(data, "val_mask") and data.val_mask.sum() > 0:
                val_loss = F.cross_entropy(logits[data.val_mask], data.y[data.val_mask]).item()
                val_acc = float((pred[data.val_mask] == data.y[data.val_mask]).float().mean().item())
                out["val_loss"] = val_loss
                out["val_acc"] = val_acc
                out["early_stop_metric"] = val_acc
            return out
