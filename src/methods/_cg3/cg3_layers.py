"""Local-view layers for the CG3 GNNModel.

Faithful port of the snapshot/86b0818 `CG3Method/CG3Layer.py`
(`GraphConvolution`, `GraphAttention`, `MLP`) and its init/dropout helpers.
The TF-style `placeholders` dict and `dp_fea0` getter pattern are dropped;
the dropout rate and `num_features_nonzero` are constructor kwargs and the
layer uses `self.training` to switch dropout on/off.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def uniform(shape, scale=0.05):
    return nn.Parameter(torch.empty(*shape, dtype=torch.float32).uniform_(-scale, scale))


def zeros(shape):
    return nn.Parameter(torch.zeros(*shape, dtype=torch.float32))


def glorot(shape):
    init_range = math.sqrt(6.0 / (shape[0] + shape[1]))
    return nn.Parameter(torch.empty(*shape, dtype=torch.float32).uniform_(-init_range, init_range))


def sparse_dropout(x: torch.Tensor, keep_prob: float, noise_shape: int) -> torch.Tensor:
    if keep_prob >= 1.0:
        return x
    indices = x._indices()
    values = x._values()
    nnz = values.shape[0]
    random_tensor = keep_prob + torch.rand(nnz, device=values.device)
    dropout_mask = torch.floor(random_tensor).bool()
    new_indices = indices[:, dropout_mask]
    new_values = values[dropout_mask] * (1.0 / keep_prob)
    return torch.sparse_coo_tensor(new_indices, new_values, x.shape).coalesce()


def _dot(x, y, sparse: bool):
    if sparse:
        return torch.sparse.mm(x, y)
    return torch.matmul(x, y)


class GraphConvolution(nn.Module):
    """Single-channel GCN layer used by GNNModel's local view.

    Note: `support` (the normalized adjacency) is set once via the attribute
    by GNNModel before calling forward — this mirrors the snapshot.
    """

    def __init__(self, input_dim: int, output_dim: int, support=None,
                 num_features_nonzero: int | None = None,
                 act=F.softplus, bias: bool = False, sparse_inputs: bool = False,
                 isnorm: bool = False, isSparse: bool = False, dropout: float = 0.0):
        super().__init__()
        self.act = act
        self.support = support
        self.use_bias = bias
        self.isnorm = isnorm
        self.isSparse = isSparse
        self.sparse_inputs = sparse_inputs
        self.dropout = float(dropout)
        self.num_features_nonzero = num_features_nonzero

        self.vars: dict[str, nn.Parameter] = {}
        w = glorot([input_dim, output_dim])
        self.vars["weights_0"] = w
        self.register_parameter("weights_0", w)
        if self.use_bias:
            b = zeros([output_dim])
            self.vars["bias"] = b
            self.register_parameter("bias", b)

    def _dropout_rate(self) -> float:
        return self.dropout if self.training else 0.0

    def forward(self, inputs):
        x = inputs
        rate = self._dropout_rate()

        if self.sparse_inputs:
            x = sparse_dropout(x, 1.0 - rate, self.num_features_nonzero)
        else:
            x = F.dropout(x, p=rate, training=self.training)

        pre_sup = _dot(x, self.vars["weights_0"], sparse=self.sparse_inputs)
        output = _dot(self.support, pre_sup, sparse=self.isSparse)

        if self.use_bias:
            output = output + self.vars["bias"]
        if self.isnorm:
            output = F.normalize(output, p=2, dim=0)
        return self.act(output)


class GraphAttention(nn.Module):
    """Single-channel GAT-style layer used by GNNModel's local view.

    Uses scatter-add over `support`'s edge indices; supports sparse feature
    input on the very first layer (matching snapshot behavior).
    """

    def __init__(self, input_dim: int, output_dim: int, support=None,
                 num_features_nonzero: int | None = None,
                 act=F.softplus, bias: bool = False, sparse_inputs: bool = False,
                 isnorm: bool = False, isSparse: bool = True,
                 dropout: float = 0.0, alpha: float = 0.2):
        super().__init__()
        self.act = act
        self.support = support
        self.use_bias = bias
        self.isnorm = isnorm
        self.isSparse = isSparse
        self.sparse_inputs = sparse_inputs
        self.dropout = float(dropout)
        self.num_features_nonzero = num_features_nonzero
        self.alpha = alpha

        self.vars: dict[str, nn.Parameter] = {}
        self.vars["weights"] = glorot([input_dim, output_dim])
        self.register_parameter("weights", self.vars["weights"])
        self.vars["attn_l"] = glorot([output_dim, 1])
        self.vars["attn_r"] = glorot([output_dim, 1])
        self.register_parameter("attn_l", self.vars["attn_l"])
        self.register_parameter("attn_r", self.vars["attn_r"])

        # Treat all params under self.vars uniformly for the L2-on-vars used
        # by GNNModel's weight-decay term — keep "weights_0" alias too.
        self.vars["weights_0"] = self.vars["weights"]

        if self.use_bias:
            b = zeros([output_dim])
            self.vars["bias"] = b
            self.register_parameter("bias", b)

        self.leakyrelu = nn.LeakyReLU(alpha)

    def _dropout_rate(self) -> float:
        return self.dropout if self.training else 0.0

    def forward(self, inputs):
        x = inputs
        rate = self._dropout_rate()

        if self.sparse_inputs:
            x = sparse_dropout(x, 1.0 - rate, self.num_features_nonzero)
        else:
            x = F.dropout(x, p=rate, training=self.training)

        Wh = _dot(x, self.vars["weights"], sparse=self.sparse_inputs)

        adj = self.support.coalesce() if self.support.is_sparse else self.support
        indices = adj.indices()
        row, col = indices[0], indices[1]

        Wh_i = Wh[row]
        Wh_j = Wh[col]

        e = (
            torch.matmul(Wh_i, self.vars["attn_l"]).squeeze(-1)
            + torch.matmul(Wh_j, self.vars["attn_r"]).squeeze(-1)
        )
        e = self.leakyrelu(e)

        e = e - e.max()
        exp_e = torch.exp(e)

        denom = torch.zeros(Wh.size(0), device=Wh.device).scatter_add_(0, row, exp_e)
        alpha = exp_e / (denom[row] + 1e-9)

        alpha = F.dropout(alpha, p=rate, training=self.training)

        output = torch.zeros_like(Wh)
        output.index_add_(0, row, alpha.unsqueeze(1) * Wh_j)

        if self.use_bias:
            output = output + self.vars["bias"]
        if self.isnorm:
            output = F.normalize(output, p=2, dim=0)
        return self.act(output)


class MLP(nn.Module):
    """Single-matmul MLP layer (used as the edge-decoder W_edge in CG3)."""

    def __init__(self, input_dim: int, output_dim: int,
                 act=F.softplus, bias: bool = False,
                 sparse_inputs: bool = False, isnorm: bool = False, isSparse: bool = False):
        super().__init__()
        self.act = act
        self.use_bias = bias
        self.isnorm = isnorm
        self.isSparse = isSparse
        self.sparse_inputs = sparse_inputs

        self.vars: dict[str, nn.Parameter] = {}
        w = glorot([input_dim, output_dim])
        self.vars["weights_0"] = w
        self.register_parameter("weights_0", w)
        if self.use_bias:
            b = zeros([output_dim])
            self.vars["bias"] = b
            self.register_parameter("bias", b)

    def forward(self, inputs):
        x = inputs
        output = _dot(x, self.vars["weights_0"], sparse=self.sparse_inputs)
        if self.use_bias:
            output = output + self.vars["bias"]
        if self.isnorm:
            output = F.normalize(output, p=2, dim=0)
        return self.act(output)
