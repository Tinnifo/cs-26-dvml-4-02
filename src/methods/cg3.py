"""CG3 method (contrastive graph-to-graph learning).

Architecture and loss are coupled, so this method ignores `cfg.model` and
builds its own model (`CG3Model`). Preprocessing builds the multi-level
hierarchy (`build_hierarchy_from_coarsen.build_hierarchy`) and pre-normalizes
the level-0 edge_index/edge_weight (since CG3's GCN convs use
`normalize=False` and rely on weights coming from `normalize_edge_index`).
Training step computes the multi-task loss inside `CG3Model.compute_loss`
with a staged mode schedule (`cls` -> `cls+cl` -> `full`).

The model's forward returns a 4-tuple `(z_gcn, z_hgcn, z, logits)`; we
override `predict_logits` to extract just `logits` so the shared eval path
in `BaseMethod` produces the same 5-tuple as for vanilla / iceberg.
"""

from __future__ import annotations

from typing import Dict

import torch

from src.methods.base import BaseMethod


class CG3Method(BaseMethod):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.hidden_dim = int(cfg.method.hidden_dim)
        self.warmup = int(cfg.method.warmup)
        self.full_start = int(cfg.method.full_start)
        self._train_idx = None
        self._pos_mask = None
        self._neg_mask = None

    def build_model(self, in_channels: int, num_classes: int, *, data=None) -> torch.nn.Module:
        from src.methods._cg3.cg3_model import CG3Model
        return CG3Model(
            in_dim=in_channels,
            hidden_dim=self.hidden_dim,
            num_classes=num_classes,
        )

    def prepare(self, model: torch.nn.Module, data):
        from torch_geometric.utils import to_scipy_sparse_matrix

        from src.methods._cg3.build_hierarchy import build_hierarchy, normalize_edge_index

        device = data.x.device
        adj = to_scipy_sparse_matrix(data.edge_index.cpu()).tocsr()
        edge_levels, c_matrices, _graphs = build_hierarchy(adj)

        # Move each `(ei, ew)` tuple to device. HGCN.forward unpacks both per
        # level — pass tuples through unchanged.
        edge_levels = [(ei.to(device), ew.to(device)) for ei, ew in edge_levels]
        model.hgcn.set_hierarchy(edge_levels, c_matrices)

        # Pre-normalize the level-0 edge_index for the local-view GCN convs
        # (CG3Model's gcn1/gcn2 use `normalize=False`). HGCN ignores the
        # function-arg edge_weight and uses its per-level weights from
        # `edge_levels`, so we only need to get the level-0 normalization
        # right here.
        ei0, ew0 = normalize_edge_index(
            data.edge_index, int(data.num_nodes), getattr(data, "edge_weight", None),
        )
        data.edge_index = ei0.to(device)
        data.edge_weight = ew0.to(device)

        self._refresh_supervised_masks(data)
        return data

    def _refresh_supervised_masks(self, data):
        self._train_idx = data.train_mask.nonzero(as_tuple=True)[0]
        labels = data.y[self._train_idx]
        pos = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        pos.fill_diagonal_(0)
        neg = 1 - pos
        neg.fill_diagonal_(0)
        self._pos_mask = pos
        self._neg_mask = neg

    def _stage_for(self, epoch: int) -> str:
        if epoch <= self.warmup:
            return "cls"
        if epoch <= self.full_start:
            return "cls+cl"
        return "full"

    def train_step(self, model: torch.nn.Module, data, optimizer: torch.optim.Optimizer,
                   epoch: int) -> Dict[str, float]:
        model.train()
        optimizer.zero_grad()
        z_gcn, z_hgcn, z, logits = model(data.x, data.edge_index, data.edge_weight)
        mode = self._stage_for(epoch)
        loss = model.compute_loss(
            z_gcn, z_hgcn, z, logits,
            data, self._train_idx, self._pos_mask, self._neg_mask,
            mode=mode,
        )
        loss.backward()
        optimizer.step()
        return {"train_loss": float(loss.item()), "stage": mode}

    def predict_logits(self, model: torch.nn.Module, data) -> torch.Tensor:
        model.eval()
        with torch.no_grad():
            _, _, _, logits = model(data.x, data.edge_index, data.edge_weight)
        return logits

    def validate(self, model: torch.nn.Module, data) -> Dict[str, float]:
        import torch.nn.functional as F
        model.eval()
        with torch.no_grad():
            _, _, _, logits = model(data.x, data.edge_index, data.edge_weight)
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
