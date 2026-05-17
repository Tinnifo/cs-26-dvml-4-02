"""CG3 GNNModel — local-view GCN/GAT fused with a global HGCN/HGAT view.

Faithful port of the snapshot/86b0818 `CG3Method/CG3Model.py`. Key
differences vs. the original:
- The optimizer is owned by Hydra's `BaseMethod.build_optimizer`, not by the
  model.
- `dp_fea0` (dropout-getter list) is replaced by an explicit `dropout: float`
  ctor kwarg, with `self.training` toggling it on/off.
- Numpy buffers from preprocessing (`edge_pos`, `train_idx`, `train_mat01`,
  `mat01_intra/inter`) are converted to tensors and registered as buffers
  here so `.to(device)` moves them.
- `forward(features, support, labels, mask)` returns the same
  `(outputs, loss, accuracy)` 3-tuple as the snapshot. Inference paths can
  pass any labels/mask and ignore the loss/accuracy outputs.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .cg3_layers import MLP, GraphAttention, GraphConvolution


def masked_softmax_cross_entropy(preds: torch.Tensor, labels: torch.Tensor,
                                 mask: torch.Tensor) -> torch.Tensor:
    """One-hot labels + int mask, matching the snapshot's TF formulation."""
    log_probs = F.log_softmax(preds, dim=1)
    loss = -(labels * log_probs).sum(dim=1)
    mask = mask.float()
    mean = mask.mean()
    if mean.item() == 0:
        return torch.zeros((), device=preds.device)
    mask = mask / mean
    loss = loss * mask
    return loss.mean()


def masked_accuracy(preds: torch.Tensor, labels: torch.Tensor,
                    mask: torch.Tensor) -> torch.Tensor:
    correct = torch.eq(torch.argmax(preds, 1), torch.argmax(labels, 1)).float()
    mask = mask.float()
    mean = mask.mean()
    if mean.item() == 0:
        return torch.zeros((), device=preds.device)
    mask = mask / mean
    return (correct * mask).mean()


class GNNModel(nn.Module):
    def __init__(self, *, num_classes: int, hidden: int, input_dim: int,
                 global_model: nn.Module, train_idx, edge_pos,
                 mat01_tr_te, weight_decay: float,
                 local_model: str, dropout: float, num_features_nonzero: int):
        super().__init__()

        self.weight_decay = float(weight_decay)
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden1 = hidden
        self.global_model = global_model
        self.dropout = float(dropout)

        if local_model == "gat":
            LocalLayer = GraphAttention
            hidden_act = F.elu
            hidden_dropout = self.dropout
            output_dropout = self.dropout
        elif local_model == "gcn":
            LocalLayer = GraphConvolution
            hidden_act = F.relu
            hidden_dropout = self.dropout
            output_dropout = 0.0
        else:
            raise ValueError(f"Unknown local_model: {local_model}")

        # Numpy preprocessing buffers — converted to tensors, registered so
        # `.to(device)` moves them with the model.
        self.register_buffer(
            "edge_pos_i", torch.from_numpy(np.asarray(edge_pos[:, 0]).astype("int64")),
        )
        self.register_buffer(
            "edge_pos_j", torch.from_numpy(np.asarray(edge_pos[:, 1]).astype("int64")),
        )
        self.register_buffer(
            "train_idx_buf", torch.from_numpy(np.asarray(train_idx).astype("int64")),
        )
        # train_mat01 (N×N) is registered as a buffer in the snapshot but
        # never read in forward — skipped here to avoid OOM on PubMed (~19K
        # nodes → ~1.5 GB float32). Drop without behavioral change.
        self.register_buffer(
            "mat01_intra", torch.from_numpy(mat01_tr_te[0].astype("float32")),
        )
        self.register_buffer(
            "mat01_inter", torch.from_numpy(mat01_tr_te[1].astype("float32")),
        )
        self.register_buffer(
            "mat01_intra_rowsum",
            torch.from_numpy(np.sum(mat01_tr_te[0], axis=1).astype("float32")),
        )
        self.train_idx_size = int(np.shape(train_idx)[0])

        # Two GNN class layers — the snapshot's classifier head.
        self.classlayers = nn.ModuleList()
        self.classlayers.append(LocalLayer(
            act=hidden_act,
            input_dim=self.input_dim,
            output_dim=self.hidden1,
            support=None,                       # set per forward
            sparse_inputs=True,
            isSparse=True,
            dropout=hidden_dropout,
            num_features_nonzero=num_features_nonzero,
            bias=True,
        ))
        self.classlayers.append(LocalLayer(
            act=(lambda x: x),
            input_dim=self.hidden1,
            output_dim=self.num_classes,
            support=None,
            sparse_inputs=False,
            isSparse=True,
            dropout=output_dropout,
            num_features_nonzero=num_features_nonzero,
            bias=True,
        ))

        # Edge generative MLP (p_e_xy decoder).
        self.p_e_yy_w_contra = MLP(
            act=(lambda x: x),
            input_dim=2 * self.num_classes,
            output_dim=1,
            sparse_inputs=False,
            isSparse=True,
            bias=True,
        )

        self.outputs: torch.Tensor | None = None
        self.concat_vec_local: torch.Tensor | None = None
        self.concat_vec_global: torch.Tensor | None = None
        self.loss = torch.tensor(0.0)
        self.accuracy = torch.tensor(0.0)
        self.p_e_xy = torch.tensor(0.0)

    @property
    def train_idx(self) -> torch.Tensor:
        return self.train_idx_buf

    def forward(self, features: torch.Tensor, support: torch.Tensor,
                labels: torch.Tensor, mask: torch.Tensor):
        # Class layer 1 → hidden
        self.classlayers[0].support = support
        self.classlayers[0].sparse_inputs = True
        h0 = self.classlayers[0](features)

        # Class layer 2 → num_classes
        self.classlayers[1].support = support
        self.classlayers[1].sparse_inputs = False
        h1 = self.classlayers[1](h0)

        self.concat_vec_local = F.normalize(h1, p=2, dim=1)

        # HGCN/HGAT global view — note: features must be the same sparse
        # input the local view consumes.
        global_out = self.global_model(features)
        self.concat_vec_global = F.normalize(global_out, p=2, dim=1)

        self.outputs = F.normalize(
            0.6 * self.concat_vec_local + 0.4 * self.concat_vec_global, p=2, dim=1,
        )

        loss_q_yobs_x_g = masked_softmax_cross_entropy(self.outputs, labels, mask)

        y_ei_local = self.concat_vec_local.index_select(0, self.edge_pos_i)
        y_ej_global = self.concat_vec_global.index_select(0, self.edge_pos_j)
        y_ei_global = self.concat_vec_global.index_select(0, self.edge_pos_i)
        y_ej_local = self.concat_vec_local.index_select(0, self.edge_pos_j)

        p_e_xy_1 = -torch.mean(torch.log(torch.sigmoid(
            self.p_e_yy_w_contra(torch.cat([y_ei_local, y_ej_global], dim=1))
        ).clamp(min=1e-8)))
        p_e_xy_2 = -torch.mean(torch.log(torch.sigmoid(
            self.p_e_yy_w_contra(torch.cat([y_ei_global, y_ej_local], dim=1))
        ).clamp(min=1e-8)))
        self.p_e_xy = p_e_xy_1 + p_e_xy_2

        # --- individual losses --- 
        loss_ce = loss_q_yobs_x_g
        loss_gen = self.p_e_xy
        loss_contrastive = self._contrastive_loss()
        
        
        total = (
            loss_ce
            + 0.4 * loss_gen
            + loss_contrastive
        )

        # Manual L2 on classlayer + p_e_yy_w_contra weights — matches snapshot.
        for i in range(2):
            for var in self.classlayers[i].vars.values():
                total = total + self.weight_decay * 0.5 * torch.sum(var ** 2)
        for var in self.p_e_yy_w_contra.vars.values():
            total = total + self.weight_decay * 0.5 * torch.sum(var ** 2)

        # Add the global model's own weight-decay term.
        total = total + self.global_model.loss

        self.loss = total
        self.accuracy = masked_accuracy(self.outputs, labels, mask)
        
        self.loss_ce = loss_ce.detach()
        self.loss_gen = loss_gen.detach()
        self.loss_contrastive = loss_contrastive.detach()
        self.loss_total = total.detach()

        return self.outputs, self.loss, self.accuracy

    def _contrastive_loss(self) -> torch.Tensor:
        device = self.concat_vec_local.device
        loss = torch.zeros((), device=device)

        # Eq. 4 — pairwise unsupervised between local & global (and reverse).
        cos_dist = torch.exp(torch.matmul(
            self.concat_vec_local, self.concat_vec_global.t(),
        ) / 0.5)
        neg = torch.mean(cos_dist, dim=1)
        diag_cos = torch.diagonal(cos_dist, 0)
        pos_neg1 = diag_cos / (neg + 1e-8)

        hp1 = 0.9

        cos_dist = torch.exp(torch.matmul(
            self.concat_vec_global, self.concat_vec_local.t(),
        ) / 0.5)
        neg = torch.mean(cos_dist, dim=1)
        diag_cos = torch.diagonal(cos_dist, 0)
        pos_neg2 = diag_cos / (neg + 1e-8)

        pos_neg3 = torch.cat([pos_neg1, pos_neg2], dim=0)
        loss = loss + (-hp1 * torch.mean(torch.log(pos_neg3.clamp(min=1e-8))))

        # Supervised contrastive (round 1).
        h1 = self.concat_vec_local.index_select(0, self.train_idx_buf)
        h2 = self.concat_vec_global.index_select(0, self.train_idx_buf)
        h_cos = torch.exp(torch.matmul(h1, h2.t()) / 0.5)
        sup_pos = torch.sum(h_cos * self.mat01_intra, dim=1)
        sup_neg = (torch.sum(h_cos * self.mat01_inter, dim=1) + sup_pos) / max(self.train_idx_size - 1, 1)
        sup_pos = sup_pos / (self.mat01_intra_rowsum + 1e-8)
        pos_neg_sup_1 = sup_pos / (sup_neg + 1e-8)

        # Supervised contrastive (round 2, swapped).
        h2_b = self.concat_vec_local.index_select(0, self.train_idx_buf)
        h1_b = self.concat_vec_global.index_select(0, self.train_idx_buf)
        h_cos = torch.exp(torch.matmul(h1_b, h2_b.t()) / 0.5)
        sup_pos = torch.sum(h_cos * self.mat01_intra, dim=1)
        sup_neg = (torch.sum(h_cos * self.mat01_inter, dim=1) + sup_pos) / max(self.train_idx_size - 1, 1)
        sup_pos = sup_pos / (self.mat01_intra_rowsum + 1e-8)
        pos_neg_sup_2 = sup_pos / (sup_neg + 1e-8)

        pos_neg_sup_3 = torch.cat([pos_neg_sup_1, pos_neg_sup_2], dim=0)
        loss = loss + (-hp1 * torch.mean(torch.log(pos_neg_sup_3.clamp(min=1e-8))))
        return loss
