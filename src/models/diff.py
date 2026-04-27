"""Diff backbone (from D2PT, used as IceBerg's headline backbone).

Architecture: a 2-layer MLP applied to *propagated* node features. Propagation
is `x_prop = (1-α) A x_prop + α x` repeated `T` times, where `A` is the
symmetric-normalized adjacency with self-loops. Propagation runs once via
`prepare(data)` and replaces `data.x`, so `forward(x, edge_index)` is then
just an MLP — keeping the call site identical to other backbones.
"""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base import BaseGNN


def _edge_index_to_sparse_adj(edge_index: torch.Tensor, num_nodes: int):
    edge_weight = np.ones(edge_index.shape[1], dtype=np.float32)
    adj = sp.csc_matrix(
        (edge_weight, (edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy())),
        shape=(num_nodes, num_nodes),
    ).tolil()
    return adj


def _normalize_adj(adj):
    rowsum = np.array(adj.sum(1))
    r_inv = np.power(rowsum, -1 / 2).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    return r_mat_inv.dot(adj).dot(r_mat_inv)


def _sparse_mx_to_torch(adj):
    adj = adj.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((adj.row, adj.col)).astype(np.int64))
    values = torch.from_numpy(adj.data)
    return torch.sparse_coo_tensor(indices, values, torch.Size(adj.shape))


def feature_propagation(x: torch.Tensor, edge_index: torch.Tensor, T: int, alpha: float) -> torch.Tensor:
    n = x.shape[0]
    adj = _edge_index_to_sparse_adj(edge_index, n)
    adj.setdiag(1)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = _normalize_adj(adj)
    adj_t = _sparse_mx_to_torch(adj).to(x.device)
    h = x.clone()
    for _ in range(T):
        h = torch.sparse.mm(adj_t, h)
        h = (1 - alpha) * h + alpha * x
    return h


class Diff(BaseGNN):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 dropout: float = 0.5, T: int = 20, alpha: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(in_channels, hidden_channels)
        self.linear2 = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout
        self.T = T
        self.alpha = alpha
        # Used by IceBerg's optimizer split — kept so methods that rely on it work.
        self.reg_params = self.linear1.parameters
        self.non_reg_params = self.linear2.parameters

    def prepare(self, data):
        with torch.no_grad():
            data.x = feature_propagation(data.x, data.edge_index, self.T, self.alpha)
        return data

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.linear1(x))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.linear2(h)
