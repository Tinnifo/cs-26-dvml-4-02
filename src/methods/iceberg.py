"""IceBerg method (debiased self-training, WWW'25): pseudo-label unlabeled
nodes via the model's own confidence (mean-confidence threshold), then add
a balanced-softmax loss term on the pseudo-labeled set in addition to the
balanced-softmax loss on the true-labeled set ("double balancing").

Works as a plug-in on top of any backbone in `src/models/` — the model
config (`cfg.model`) is whatever you select with Hydra. The headline
configuration in the IceBerg paper pairs this method with the `Diff`
backbone, but `model=gcn method=iceberg` is also a valid recipe (and is
the cleanest A/B test against the existing `model=gcn method=vanilla` row).
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from sklearn.metrics import f1_score

from src.methods.base import BaseMethod


def _balanced_softmax_loss(logits: torch.Tensor, labels: torch.Tensor,
                           sample_per_class: torch.Tensor) -> torch.Tensor:
    """logits + log(class_count) before CE. Empty classes are clamped to 1
    so log() doesn't blow up at very small label budgets."""
    spc = sample_per_class.to(logits.device).type_as(logits).clamp(min=1.0)
    spc = spc.unsqueeze(0).expand(logits.shape[0], -1)
    return F.cross_entropy(logits + spc.log(), labels)


class IcebergMethod(BaseMethod):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.lamda = float(cfg.method.lamda)
        self.warmup = int(cfg.method.warmup)
        self.num_classes = None
        self.class_num_list = None

    def build_model(self, in_channels: int, num_classes: int, *, data=None) -> torch.nn.Module:
        return instantiate(
            self.cfg.model.arch,
            in_channels=in_channels,
            out_channels=num_classes,
        )

    def prepare(self, model: torch.nn.Module, data):
        data = super().prepare(model, data)
        self.num_classes = int(data.y.max().item()) + 1
        self.class_num_list = torch.tensor(
            [int((data.y[data.train_mask] == c).sum().item()) for c in range(self.num_classes)],
            device=data.x.device,
        )
        return data

    def _get_pseudo_labels(self, model: torch.nn.Module, data):
        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
            probs = F.softmax(logits, dim=1)
            confidence, pred_label = probs.max(dim=1)
        # True labels for training nodes — only pseudo-label the unlabeled
        pred_label[data.train_mask] = data.y[data.train_mask]
        unlabel_mask = ~data.train_mask
        if unlabel_mask.sum() == 0:
            return None, None
        threshold = confidence[unlabel_mask].mean().item()
        pseudo_mask = (confidence >= threshold) & unlabel_mask
        return pseudo_mask, pred_label

    def train_step(self, model: torch.nn.Module, data, optimizer: torch.optim.Optimizer,
                   epoch: int) -> Dict[str, float]:
        pseudo_mask, pseudo_label = (None, None)
        if epoch >= self.warmup:
            pseudo_mask, pseudo_label = self._get_pseudo_labels(model, data)

        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)

        loss = _balanced_softmax_loss(
            logits[data.train_mask], data.y[data.train_mask], self.class_num_list,
        )

        out = {"train_loss_supervised": float(loss.item())}
        if pseudo_mask is not None and pseudo_mask.sum() > 0:
            class_num_u = torch.tensor(
                [int((pseudo_label[pseudo_mask] == c).sum().item()) for c in range(self.num_classes)],
                device=logits.device,
            )
            loss_u = _balanced_softmax_loss(
                logits[pseudo_mask], pseudo_label[pseudo_mask], class_num_u,
            )
            loss = loss + self.lamda * loss_u
            out["train_loss_pseudo"] = float(loss_u.item())
            out["pseudo_count"] = int(pseudo_mask.sum().item())

        loss.backward()
        optimizer.step()
        out["train_loss"] = float(loss.item())
        return out

    def validate(self, model: torch.nn.Module, data) -> Dict[str, float]:
        """IceBerg uses (val_acc + val_f1)/2 as the early-stopping criterion."""
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
            y_true = data.y[data.val_mask].cpu().numpy()
            y_pred = pred[data.val_mask].cpu().numpy()
            val_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
            out["val_loss"] = val_loss
            out["val_acc"] = val_acc
            out["val_f1"] = val_f1
            out["early_stop_metric"] = (val_acc + val_f1) / 2.0
        return out
