"""Vanilla supervised training: forward pass + cross-entropy on labeled nodes.
This is what the original cs-26 `src/run_experiments.py` did for every model."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from hydra.utils import instantiate

from src.methods.base import BaseMethod


class VanillaMethod(BaseMethod):
    def build_model(self, in_channels: int, num_classes: int, *, data=None) -> torch.nn.Module:
        return instantiate(
            self.cfg.model.arch,
            in_channels=in_channels,
            out_channels=num_classes,
        )

    def train_step(self, model: torch.nn.Module, data, optimizer: torch.optim.Optimizer,
                   epoch: int) -> Dict[str, float]:
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        return {"train_loss": float(loss.item())}
