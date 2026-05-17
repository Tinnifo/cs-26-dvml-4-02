"""CG3 method — faithful port of the snapshot/86b0818 CG3 pipeline.

Architecture and loss are coupled, so this method ignores `cfg.model` and
builds its own composite model:
  * a local-view `GraphConvolution` / `GraphAttention` two-layer classifier
    (selectable via `cfg.method.local_model = "gcn" | "gat"`).
  * a global-view hierarchical `HGCN` / `HGAT` (selectable via
    `cfg.method.global_model = "hgcn" | "hgat"`).
The four (local × global) combinations are the four experiments shown in
the snapshot's `CG3Method/main.py`.

`prepare` runs the graph-coarsening and supervised-contrastive bookkeeping
once at the start of training and stashes the per-data tensors on `data`
under `_cg3_*` attributes. Each forward call passes `(features, support,
labels, mask)` to the model, which returns `(outputs, loss, accuracy)` —
matching the snapshot's `forward` contract exactly.

The model adds an L2 weight-decay term inside its own loss, so the
optimizer is constructed with `weight_decay=0` to avoid double-counting.
"""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn.functional as F

from src.methods.base import BaseMethod



class CG3Method(BaseMethod):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.local_model = str(cfg.method.local_model)
        self.global_model_name = str(cfg.method.global_model)
        self.lr = float(cfg.method.lr)
        self.weight_decay = float(cfg.method.weight_decay)
        self.hidden_local = int(cfg.method.hidden_local)
        self.hidden_global = int(cfg.method.hidden_global)
        self.dropout = float(cfg.method.dropout)
        self.coarsen_level = int(cfg.method.coarsen_level)
        self.max_node_wgt = int(cfg.method.max_node_wgt)
        self.channel_num = int(cfg.method.channel_num)
        self.node_wgt_embed_dim = int(cfg.method.node_wgt_embed_dim)
        self._artifacts = None  # populated in build_model

    def build_model(self, in_channels: int, num_classes: int, *, data=None) -> torch.nn.Module:
        if data is None:
            raise RuntimeError(
                "CG3Method.build_model requires `data=` (it preprocesses the "
                "graph hierarchy before constructing the model)."
            )

        from src.methods._cg3.build_hierarchy import build_cg3_artifacts
        from src.methods._cg3.cg3_model import GNNModel
        from src.methods._cg3.hgcn import HGAT, HGCN

        artifacts = build_cg3_artifacts(
            data,
            coarsen_level=self.coarsen_level,
            max_node_wgt=self.max_node_wgt,
            channel_num=self.channel_num,
        )
        self._artifacts = artifacts

        if self.global_model_name == "hgcn":
            GlobalCls = HGCN
        elif self.global_model_name == "hgat":
            GlobalCls = HGAT
        else:
            raise ValueError(f"Unknown global_model: {self.global_model_name}")

        global_model = GlobalCls(
            input_dim=artifacts.input_dim,
            output_dim=artifacts.num_classes,
            hidden=self.hidden_global,
            transfer_list=artifacts.transfer_list,
            adj_list=artifacts.adj_list,
            node_wgt_list=artifacts.node_wgt_list,
            coarsen_level=self.coarsen_level,
            max_node_wgt=self.max_node_wgt,
            node_wgt_embed_dim=self.node_wgt_embed_dim,
            weight_decay=self.weight_decay,
            channel_num=self.channel_num,
            dropout=self.dropout,
        )

        model = GNNModel(
            num_classes=artifacts.num_classes,
            hidden=self.hidden_local,
            input_dim=artifacts.input_dim,
            global_model=global_model,
            train_idx=artifacts.train_idx_np,
            edge_pos=artifacts.edge_pos,
            mat01_tr_te=artifacts.mats_intra_inter,
            weight_decay=self.weight_decay,
            local_model=self.local_model,
            dropout=self.dropout,
            num_features_nonzero=artifacts.num_features_nonzero,
        )
        return model

    def prepare(self, model: torch.nn.Module, data):
        """Move CPU artifacts onto the data's device and stash them on
        `data._cg3_*`. Called once after `build_model().to(device)`."""
        device = data.x.device
        a = self._artifacts
        if a is None:
            raise RuntimeError("CG3Method.prepare called before build_model")

        data._cg3_feature_sp = a.feature_sp.to(device)
        data._cg3_support_sp = a.support_sp.to(device)
        data._cg3_y_train_oh = a.y_train_oh.to(device)
        data._cg3_train_mask_int = a.train_mask_int.to(device)
        data._cg3_y_val_oh = a.y_val_oh.to(device)
        data._cg3_val_mask_int = a.val_mask_int.to(device)
        data._cg3_y_test_oh = a.y_test_oh.to(device)
        data._cg3_test_mask_int = a.test_mask_int.to(device)
        return data

    def build_optimizer(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        # Snapshot adds the L2 term inside its own loss — avoid double-counting.
        return torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=0.0)

    def train_step(self, model: torch.nn.Module, data,
                   optimizer: torch.optim.Optimizer, epoch: int) -> Dict[str, float]:
        model.train()
        optimizer.zero_grad()
        outputs, loss, accuracy = model(
            data._cg3_feature_sp,
            data._cg3_support_sp,
            data._cg3_y_train_oh,
            data._cg3_train_mask_int,
        )
        loss.backward()
        optimizer.step()

        # --- detach everything once (clean logging) ---
        train_loss = loss.detach()

        loss_ce = getattr(model, "loss_ce", None)
        loss_gen = getattr(model, "loss_gen", None)
        loss_contrastive = getattr(model, "loss_contrastive", None)
        loss_total = getattr(model, "loss_total", loss)

        return {
            "train_loss": float(train_loss.item()),
            "train_acc": float(accuracy.detach().item()),

            "loss_ce": float(loss_ce.detach().item()) if loss_ce is not None else None,
            "loss_gen": float(loss_gen.detach().item()) if loss_gen is not None else None,
            "loss_contrastive": float(loss_contrastive.detach().item()) if loss_contrastive is not None else None,
            "loss_total": float(loss_total.detach().item()),
        }

    def predict_logits(self, model: torch.nn.Module, data) -> torch.Tensor:
        model.eval()
        with torch.no_grad():
            outputs, _, _ = model(
                data._cg3_feature_sp,
                data._cg3_support_sp,
                data._cg3_y_train_oh,
                data._cg3_train_mask_int,
            )
        return outputs

    def validate(self, model: torch.nn.Module, data) -> Dict[str, float]:
        model.eval()
        with torch.no_grad():
            outputs, _, _ = model(
                data._cg3_feature_sp,
                data._cg3_support_sp,
                data._cg3_y_val_oh,
                data._cg3_val_mask_int,
            )
            pred = outputs.argmax(dim=1)
        out: Dict[str, float] = {
            "train_acc": float(
                (pred[data.train_mask] == data.y[data.train_mask]).float().mean().item()
            ),
        }
        if hasattr(data, "val_mask") and data.val_mask.sum() > 0:
            val_loss = F.cross_entropy(outputs[data.val_mask], data.y[data.val_mask]).item()
            val_acc = float((pred[data.val_mask] == data.y[data.val_mask]).float().mean().item())
            out["val_loss"] = val_loss
            out["val_acc"] = val_acc
            out["early_stop_metric"] = val_acc
        return out
