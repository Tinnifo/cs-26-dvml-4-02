"""HGCN and HGAT global-view models for CG3.

Faithful port of the snapshot/86b0818 `CG3Method/models.py`. The TF-style
`placeholders` dict and the global `FLAGS` config are replaced by explicit
constructor kwargs (`input_dim`, `output_dim`, `hidden`, `coarsen_level`,
`max_node_wgt`, `node_wgt_embed_dim`, `weight_decay`, `channel_num`,
`dropout`).

The standalone `_accuracy` method is dropped — accuracy is computed by the
outer GNNModel against the integer `y` tensor that the Hydra pipeline owns.
The `_loss()` weight-decay term is preserved because GNNModel adds the
global model's `.loss` onto its total.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .hgcn_layers import HGCNGraphAttention, HGCNGraphConvolution


class _HierarchicalModel(nn.Module):
    """Common base for HGCN / HGAT — owns the per-level node-weight embedding
    table and the residual-skip schedule between encoder and decoder."""

    def __init__(self, input_dim: int, output_dim: int, hidden: int,
                 transfer_list, adj_list, node_wgt_list,
                 *, coarsen_level: int, max_node_wgt: int,
                 node_wgt_embed_dim: int, weight_decay: float,
                 channel_num: int, dropout: float):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden = hidden
        self.coarsen_level = coarsen_level
        self.max_node_wgt = max_node_wgt
        self.node_wgt_embed_dim = node_wgt_embed_dim
        self.weight_decay = weight_decay
        self.channel_num = channel_num
        self.dropout = float(dropout)

        self.transfer_list = transfer_list
        self.adj_list = adj_list
        self.node_wgt_list = node_wgt_list

        bound = math.sqrt(6.0 / (3.0 * node_wgt_embed_dim + 3.0 * input_dim))
        self.W_node_wgt = nn.Parameter(
            torch.empty(max_node_wgt, node_wgt_embed_dim).uniform_(-bound, bound)
        )

        for i, nw in enumerate(self.node_wgt_list):
            self.register_buffer(f"node_wgt_idx_{i}", torch.from_numpy(nw.astype("int64")))

        self.layers: nn.ModuleList = nn.ModuleList()
        self._build()

        self.outputs = None
        self.embed = None
        self.loss = torch.tensor(0.0)

    HIDDEN_ACT = staticmethod(F.relu)

    def _make_layer(self, **kwargs) -> nn.Module:
        raise NotImplementedError

    def _build(self):
        H = [self.hidden] * (2 * self.coarsen_level + 1)
        hidden_act = type(self).HIDDEN_ACT
        identity = (lambda x: x)

        # Input layer
        self.layers.append(self._make_layer(
            input_dim=self.input_dim,
            output_dim=H[0],
            support=self.adj_list[0] * self.channel_num,
            transfer=self.transfer_list[0],
            mod="input",
            layer_index=0,
            sparse_inputs=True,
            dropout=self.dropout,
            act=hidden_act,
        ))

        # Coarsen layers
        for i in range(self.coarsen_level - 1):
            self.layers.append(self._make_layer(
                input_dim=H[i],
                output_dim=H[i + 1],
                support=self.adj_list[i + 1] * self.channel_num,
                transfer=self.transfer_list[i + 1],
                mod="coarsen",
                layer_index=i + 1,
                sparse_inputs=False,
                dropout=self.dropout,
                act=hidden_act,
            ))

        # Refine layers
        for i in range(self.coarsen_level, self.coarsen_level * 2):
            self.layers.append(self._make_layer(
                input_dim=H[i - 1],
                output_dim=H[i],
                support=self.adj_list[2 * self.coarsen_level - i] * self.channel_num,
                transfer=self.transfer_list[2 * self.coarsen_level - 1 - i],
                mod="refine",
                layer_index=i,
                sparse_inputs=False,
                dropout=self.dropout,
                act=hidden_act,
            ))

        # Output layer (identity activation; the local view applies softmax/CE)
        self.layers.append(self._make_layer(
            input_dim=H[self.coarsen_level * 2 - 1],
            output_dim=self.output_dim,
            support=self.adj_list[0] * self.channel_num,
            transfer=self.transfer_list[0],
            mod="output",
            layer_index=self.coarsen_level * 2,
            sparse_inputs=False,
            dropout=self.dropout,
            act=identity,
        ))

    def _node_emb_for_layer(self, layer_idx: int):
        if layer_idx == 0:
            idx_buf = self.node_wgt_idx_0
        elif layer_idx < self.coarsen_level:
            idx_buf = getattr(self, f"node_wgt_idx_{layer_idx}")
        elif layer_idx < self.coarsen_level * 2:
            idx_buf = getattr(self, f"node_wgt_idx_{2 * self.coarsen_level - layer_idx}")
        else:
            return None
        return self.W_node_wgt[idx_buf]

    def forward(self, features):
        activations = [features]
        gnn_layers = []

        for i, layer in enumerate(self.layers):
            mod = layer.mod
            if mod in ("coarsen", "refine"):
                node_emb = self._node_emb_for_layer(i)
                hidden, pre = layer(activations[-1], node_emb=node_emb)
            else:
                hidden, pre = layer(activations[-1])
            gnn_layers.append(pre)
            if self.coarsen_level <= i < self.coarsen_level * 2:
                hidden = hidden + gnn_layers[self.coarsen_level * 2 - i - 1]
            activations.append(hidden)

        self.outputs = activations[-1]
        self.embed = activations[-2]
        self._loss()
        return self.outputs

    def _loss(self):
        loss = torch.tensor(0.0, device=self.W_node_wgt.device)
        for layer in self.layers:
            for var in layer.vars.values():
                loss = loss + self.weight_decay * 0.5 * torch.sum(var ** 2)
        self.loss = loss

    def predict(self):
        return F.softmax(self.outputs, dim=1)


class HGCN(_HierarchicalModel):
    """Hierarchical GCN global view."""

    HIDDEN_ACT = staticmethod(F.relu)

    def _make_layer(self, **kwargs):
        return HGCNGraphConvolution(
            num_features_nonzero=None,
            node_wgt_embed_dim=self.node_wgt_embed_dim,
            **kwargs,
        )


class HGAT(_HierarchicalModel):
    """Hierarchical GAT global view."""

    HIDDEN_ACT = staticmethod(F.elu)

    def _make_layer(self, **kwargs):
        return HGCNGraphAttention(
            num_features_nonzero=None,
            node_wgt_embed_dim=self.node_wgt_embed_dim,
            **kwargs,
        )
