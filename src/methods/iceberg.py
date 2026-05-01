from __future__ import annotations

from typing import Dict, Iterable, List

import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from sklearn.metrics import f1_score

from src.methods.base import BaseMethod


def _balanced_softmax(logits: torch.Tensor, labels: torch.Tensor,
                      sample_per_class: torch.Tensor) -> torch.Tensor:
    spc = sample_per_class.to(logits.device).type_as(logits).clamp(min=1.0)
    spc = spc.unsqueeze(0).expand(logits.shape[0], -1)
    return F.cross_entropy(logits + spc.log(), labels)


def _robust_balanced_softmax(logits: torch.Tensor, labels: torch.Tensor,
                             sample_per_class: torch.Tensor,
                             num_classes: int, beta: float) -> torch.Tensor:
    spc = sample_per_class.to(logits.device).type_as(logits).clamp(min=1.0)
    spc = spc.unsqueeze(0).expand(logits.shape[0], -1)
    adjusted = logits + spc.log()
    loss = F.cross_entropy(adjusted, labels)
    if beta > 0.0:
        pred = F.softmax(adjusted, dim=1).clamp(min=1e-7, max=1.0)
        one_hot = F.one_hot(labels, num_classes).float().to(labels.device).clamp(min=1e-4, max=1.0)
        rce = -(pred * one_hot.log()).sum(dim=1).mean()
        loss = loss + beta * rce
    return loss


def _resolve_params(group) -> List[torch.nn.Parameter]:
    if callable(group):
        group = group()
    return [p for p in group if p.requires_grad]


class IcebergMethod(BaseMethod):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.lamda = float(cfg.method.lamda)
        self.warmup = int(cfg.method.warmup)
        self.beta = float(cfg.method.beta)
        self.num_classes: int | None = None
        self.class_num_list: torch.Tensor | None = None

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

    def build_optimizer(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        lr = float(self.cfg.method.lr)
        wd = float(self.cfg.method.weight_decay)
        if hasattr(model, "reg_params") and hasattr(model, "non_reg_params"):
            reg = _resolve_params(model.reg_params)
            non_reg = _resolve_params(model.non_reg_params)
            if reg or non_reg:
                groups = []
                if reg:
                    groups.append({"params": reg, "weight_decay": wd})
                if non_reg:
                    groups.append({"params": non_reg, "weight_decay": 0.0})
                return torch.optim.Adam(groups, lr=lr)
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    def _pseudo_labels(self, model: torch.nn.Module, data):
        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
            probs = F.softmax(logits, dim=1)
            confidence, pred_label = probs.max(dim=1)
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
            pseudo_mask, pseudo_label = self._pseudo_labels(model, data)

        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)

        loss = _balanced_softmax(
            logits[data.train_mask], data.y[data.train_mask], self.class_num_list,
        )
        out: Dict[str, float] = {"train_loss_supervised": float(loss.item())}

        if pseudo_mask is not None and pseudo_mask.sum() > 0:
            class_num_u = torch.tensor(
                [int((pseudo_label[pseudo_mask] == c).sum().item()) for c in range(self.num_classes)],
                device=logits.device,
            )
            loss_u = _robust_balanced_softmax(
                logits[pseudo_mask], pseudo_label[pseudo_mask], class_num_u,
                self.num_classes, self.beta,
            )
            loss = loss + self.lamda * loss_u
            out["train_loss_pseudo"] = float(loss_u.item())
            out["pseudo_count"] = int(pseudo_mask.sum().item())

        loss.backward()
        optimizer.step()
        out["train_loss"] = float(loss.item())
        return out

    def validate(self, model: torch.nn.Module, data) -> Dict[str, float]:
        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
            pred = logits.argmax(dim=1)
        out: Dict[str, float] = {
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
