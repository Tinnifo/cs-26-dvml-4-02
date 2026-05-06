"""Layers for the HGCN / HGAT global-view models.

Faithful port of the snapshot/86b0818 `CG3Method/layers.py`. Differences vs.
the snapshot: (1) the TF-style `placeholders` dict is replaced by explicit
ctor kwargs (`dropout: float`, `num_features_nonzero: int`,
`node_wgt_embed_dim: int`); (2) `print(...)` debug statements removed; (3)
multi-channel `support` (a list of `(coords, values, shape)` tuples) is
turned into per-layer sparse buffers in `__init__`.
"""

from __future__ import annotations

import math

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import normalize

from .cg3_layers import glorot, sparse_dropout, zeros


def _scipy_to_torch_sparse(X) -> torch.Tensor:
    coo = X.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((coo.row, coo.col))).long()
    values = torch.from_numpy(coo.data)
    shape = torch.Size(coo.shape)
    return torch.sparse_coo_tensor(indices, values, shape).coalesce()


def _support_to_sparse(support_tuple) -> torch.Tensor:
    coords, values, shape = support_tuple
    rows = coords[:, 0]
    cols = coords[:, 1]
    sp_mat = sp.csr_matrix((values, (rows, cols)), shape=shape, dtype=np.float32)
    return _scipy_to_torch_sparse(sp_mat)


def _dot(x, y, sparse: bool):
    if sparse:
        return torch.sparse.mm(x, y)
    return torch.matmul(x, y)


class HGCNGraphConvolution(nn.Module):
    """Multi-channel, hierarchy-aware GCN layer for HGCN."""

    def __init__(self, input_dim: int, output_dim: int, support, transfer,
                 mod: str, layer_index: int, dropout: float = 0.0,
                 sparse_inputs: bool = False, act=F.relu, bias: bool = False,
                 featureless: bool = False, num_features_nonzero: int | None = None,
                 node_wgt_embed_dim: int = 5):
        super().__init__()
        self.dropout = float(dropout)
        self.act = act
        self.support = support
        self.transfer = transfer
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.use_bias = bias
        self.mod = mod
        self.layer_index = layer_index
        self.output_dim = output_dim
        self.num_features_nonzero = num_features_nonzero
        self.node_wgt_embed_dim = node_wgt_embed_dim

        self.vars: dict[str, nn.Parameter] = {}
        if self.mod in ("coarsen", "refine"):
            in_dim = input_dim + node_wgt_embed_dim
        else:
            in_dim = input_dim

        for i in range(len(self.support)):
            w = glorot([in_dim, output_dim])
            self.vars[f"weights_{i}"] = w
            self.register_parameter(f"weights_{i}", w)
        if self.use_bias:
            b = zeros([output_dim])
            self.vars["bias"] = b
            self.register_parameter("bias", b)

        for i in range(len(self.support)):
            self.register_buffer(f"support_tensor_{i}", _support_to_sparse(self.support[i]))

        if self.mod in ("coarsen", "input"):
            transfer_opo = normalize(self.transfer.T, norm="l2", axis=1).astype(np.float32)
            self.register_buffer("transfer_tensor", _scipy_to_torch_sparse(transfer_opo))
        elif self.mod == "refine":
            self.register_buffer(
                "transfer_tensor",
                _scipy_to_torch_sparse(self.transfer.astype(np.float32)),
            )
        else:
            self.transfer_tensor = None

        self.channel_combine = nn.Conv1d(
            in_channels=len(self.support), out_channels=1, kernel_size=1, bias=False,
        )

    def _dropout_rate(self) -> float:
        return self.dropout if self.training else 0.0

    def forward(self, inputs, node_emb=None):
        if self.mod in ("coarsen", "refine"):
            x = torch.cat([inputs, node_emb], dim=1)
        else:
            x = inputs

        rate = self._dropout_rate()
        if self.sparse_inputs:
            x = sparse_dropout(x, 1.0 - rate, self.num_features_nonzero)
        else:
            x = F.dropout(x, p=rate, training=self.training)

        supports = []
        for i in range(len(self.support)):
            if not self.featureless:
                pre_sup = _dot(x, self.vars[f"weights_{i}"], sparse=self.sparse_inputs)
            else:
                pre_sup = self.vars[f"weights_{i}"]
            sp_tensor = getattr(self, f"support_tensor_{i}")
            supports.append(_dot(sp_tensor, pre_sup, sparse=True))

        # (N, output_dim, num_channels) -> (N, num_channels, output_dim) -> Conv1d
        supports = torch.stack(supports, dim=2).permute(0, 2, 1)
        output = self.channel_combine(supports).squeeze(1)

        if self.use_bias:
            output = output + self.vars["bias"]
        output = self.act(output)

        gcn_output = output

        if self.mod == "output":
            return output, gcn_output
        if self.mod in ("coarsen", "input", "refine"):
            output = _dot(self.transfer_tensor, gcn_output, sparse=True)
        return output, gcn_output


class HGCNGraphAttention(nn.Module):
    """Multi-channel, hierarchy-aware GAT layer for HGAT."""

    def __init__(self, input_dim: int, output_dim: int, support, transfer,
                 mod: str, layer_index: int, dropout: float = 0.0,
                 sparse_inputs: bool = False, act=F.elu, bias: bool = False,
                 featureless: bool = False, num_features_nonzero: int | None = None,
                 node_wgt_embed_dim: int = 5, alpha: float = 0.2):
        super().__init__()
        self.dropout = float(dropout)
        self.act = act
        self.support = support
        self.transfer = transfer
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.use_bias = bias
        self.mod = mod
        self.layer_index = layer_index
        self.output_dim = output_dim
        self.num_features_nonzero = num_features_nonzero
        self.node_wgt_embed_dim = node_wgt_embed_dim
        self.alpha = alpha

        self.vars: dict[str, nn.Parameter] = {}
        if self.mod in ("coarsen", "refine"):
            in_dim = input_dim + node_wgt_embed_dim
        else:
            in_dim = input_dim

        # one (W, a) pair per channel
        for i in range(len(self.support)):
            w = glorot([in_dim, output_dim])
            self.vars[f"weights_{i}"] = w
            self.register_parameter(f"weights_{i}", w)
            a = glorot([2 * output_dim, 1])
            self.vars[f"attn_{i}"] = a
            self.register_parameter(f"attn_{i}", a)
        if self.use_bias:
            b = zeros([output_dim])
            self.vars["bias"] = b
            self.register_parameter("bias", b)

        for i in range(len(self.support)):
            self.register_buffer(f"support_tensor_{i}", _support_to_sparse(self.support[i]))

        if self.mod in ("coarsen", "input"):
            transfer_opo = normalize(self.transfer.T, norm="l2", axis=1).astype(np.float32)
            self.register_buffer("transfer_tensor", _scipy_to_torch_sparse(transfer_opo))
        elif self.mod == "refine":
            self.register_buffer(
                "transfer_tensor",
                _scipy_to_torch_sparse(self.transfer.astype(np.float32)),
            )
        else:
            self.transfer_tensor = None

        self.leakyrelu = nn.LeakyReLU(alpha)

    def _dropout_rate(self) -> float:
        return self.dropout if self.training else 0.0

    def _attention(self, Wh: torch.Tensor, attn: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        N = Wh.shape[0]
        indices = adj.coalesce().indices()
        row, col = indices[0], indices[1]

        Wh_i = Wh[row]
        Wh_j = Wh[col]
        edge_features = torch.cat([Wh_i, Wh_j], dim=1)

        e = self.leakyrelu(torch.matmul(edge_features, attn).squeeze(1))
        # stable softmax
        e = e - e.max()
        exp_e = torch.exp(e)

        denom = torch.zeros(N, device=Wh.device).scatter_add_(0, row, exp_e)
        alpha = exp_e / (denom[row] + 1e-16)

        out = torch.zeros_like(Wh)
        out = out.index_add(0, row, Wh_j * alpha.unsqueeze(1))
        return out

    def forward(self, inputs, node_emb=None):
        if self.mod in ("coarsen", "refine"):
            x = torch.cat([inputs, node_emb], dim=1)
        else:
            x = inputs

        rate = self._dropout_rate()
        if self.sparse_inputs:
            # The snapshot raises NotImplementedError here; HGAT is only
            # composed with dense intermediates (the input layer in HGAT is
            # 'input' mod which sets sparse_inputs from the constructor —
            # but in our pipeline we pass the sparse feature tensor only to
            # the first layer of the *local* GNNModel, never to HGAT).
            x = sparse_dropout(x, 1.0 - rate, self.num_features_nonzero)
        else:
            x = F.dropout(x, p=rate, training=self.training)

        outs = []
        for i in range(len(self.support)):
            Wh = _dot(x, self.vars[f"weights_{i}"], sparse=self.sparse_inputs)
            adj = getattr(self, f"support_tensor_{i}")
            outs.append(self._attention(Wh, self.vars[f"attn_{i}"], adj))

        out = torch.stack(outs, dim=2).mean(dim=2)
        if self.use_bias:
            out = out + self.vars["bias"]
        out = self.act(out)
        gat_output = out

        if self.mod == "output":
            return out, gat_output
        if self.mod in ("coarsen", "input", "refine"):
            out = torch.sparse.mm(self.transfer_tensor, gat_output)
        return out, gat_output
