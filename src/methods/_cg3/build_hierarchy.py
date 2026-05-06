"""Preprocessing for CG3 — port of the snapshot/86b0818 hierarchy pipeline.

Replaces the previous main-branch hierarchy builder. The artifacts produced
here mirror what `CG3Method/main.py` + `CG3Method/train.py` set up:
  * `feature_sp`   — sparse, row-normalized feature tensor.
  * `support_sp`   — D⁻⁰·⁵ (A+I) D⁻⁰·⁵ as a sparse tensor.
  * `support_tuple` — `(coords, values, shape)` tuple kept for `edge_pos`.
  * `transfer_list / adj_list / node_wgt_list` — per-level coarsening artifacts
    consumed by `HGCN` / `HGAT` (each level's `adj` is replicated `channel_num`
    times to mimic the snapshot's TF code).
  * `train_mat01`, `mats_intra_inter`, `tr_idx`, `y_train_oh`, `train_mask_int`,
    `y_val_oh`, `val_mask_int`, `y_test_oh`, `test_mask_int` — the supervised
    contrastive book-keeping and one-hot labels per split.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.utils import to_scipy_sparse_matrix

from .coarsen import create_coarse_graph, generate_hybrid_matching
from .graph import Graph


# ---------------------------------------------------------------------------
# helpers ported from CG3Method/funcCNN.py
# ---------------------------------------------------------------------------

def _normalize_features_csr(features: sp.spmatrix) -> sp.spmatrix:
    rowsum = np.asarray(features.sum(axis=1)).flatten()
    r_inv = np.power(rowsum.astype(float), -1, where=rowsum > 0)
    r_inv[~np.isfinite(r_inv)] = 0.0
    return sp.diags(r_inv).dot(features)


def _normalize_adj(adj: sp.spmatrix) -> sp.coo_matrix:
    adj = sp.coo_matrix(adj)
    rowsum = np.asarray(adj.sum(axis=1)).flatten()
    d_inv_sqrt = np.power(rowsum.astype(float), -0.5, where=rowsum > 0)
    d_inv_sqrt[~np.isfinite(d_inv_sqrt)] = 0.0
    d_mat = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat).transpose().dot(d_mat).tocoo()


def _sparse_to_tuple(mx: sp.spmatrix):
    if not sp.issparse(mx):
        mx = sp.coo_matrix(mx)
    else:
        mx = mx.tocoo()
    coords = np.vstack((mx.row, mx.col)).transpose()
    values = mx.data
    return coords, values, mx.shape


def _preprocess_features(features: sp.spmatrix):
    if not sp.issparse(features):
        features = sp.lil_matrix(features)
    features = _normalize_features_csr(features)
    if not sp.issparse(features):
        features = sp.lil_matrix(features)
    return _sparse_to_tuple(features)


def _preprocess_adj(adj: sp.spmatrix):
    if not sp.issparse(adj):
        adj = sp.csr_matrix(adj)
    adj_normalized = _normalize_adj(adj + sp.eye(adj.shape[0]))
    return _sparse_to_tuple(adj_normalized)


def _tuple_to_torch_sparse(sparse_tuple) -> torch.Tensor:
    coords, values, shape = sparse_tuple
    indices = torch.from_numpy(np.asarray(coords).T.astype("int64"))
    values_t = torch.from_numpy(np.asarray(values, dtype=np.float32))
    return torch.sparse_coo_tensor(indices, values_t, torch.Size(shape)).coalesce()


def _cal_class01_mat(y_train_oh: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    """Pairwise same-class indicator matrix; rows for unlabeled nodes are 0."""
    y = np.argmax(y_train_oh, axis=1)
    not_train = np.argwhere(~train_mask.astype(bool))
    y[not_train] = -1
    num_classes = int(np.max(y)) + 1
    mat01 = np.zeros([y_train_oh.shape[0], y_train_oh.shape[0]])
    for i in range(num_classes):
        pos = np.argwhere(y == i)
        if pos.size == 0:
            continue
        for j in range(pos.shape[0]):
            mat01[pos[j, 0], pos[:, 0]] = 1
    np.fill_diagonal(mat01, 0)
    return mat01


def _cal_intra_class_mat01(y: np.ndarray) -> list[np.ndarray]:
    """Returns [intra, inter] over the labeled-node × labeled-node block."""
    n = int(np.shape(y)[0])
    num_classes = int(np.max(y)) + 1
    intra = np.zeros([n, n])
    inter = np.ones([n, n])
    for c in range(num_classes):
        pos = np.argwhere(y == c)
        for k in range(pos.shape[0]):
            intra[pos[k, 0], pos[:, 0]] = 1
    inter -= intra
    intra -= np.eye(n)
    return [intra, inter]


# ---------------------------------------------------------------------------
# top-level entry point
# ---------------------------------------------------------------------------

@dataclass
class CG3Artifacts:
    # feature / adjacency
    input_dim: int
    num_classes: int
    num_nodes: int
    feature_sp: torch.Tensor
    support_sp: torch.Tensor
    support_tuple: tuple
    edge_pos: np.ndarray
    num_features_nonzero: int
    # hierarchy
    transfer_list: list = field(default_factory=list)
    adj_list: list = field(default_factory=list)
    node_wgt_list: list = field(default_factory=list)
    # labels / masks (per split)
    y_train_oh: torch.Tensor = None
    train_mask_int: torch.Tensor = None
    y_val_oh: torch.Tensor = None
    val_mask_int: torch.Tensor = None
    y_test_oh: torch.Tensor = None
    test_mask_int: torch.Tensor = None
    # supervised-contrastive book-keeping
    train_idx_np: np.ndarray = None
    train_mat01: np.ndarray = None
    mats_intra_inter: list = None


def build_cg3_artifacts(data, *, coarsen_level: int, max_node_wgt: int,
                         channel_num: int) -> CG3Artifacts:
    """Build everything CG3 / HGCN / HGAT need from a PyG Data object.

    Numpy-side preprocessing runs on CPU; the returned tensors are CPU tensors
    (the caller / `model.to(device)` move them to GPU)."""
    device_data = data
    edge_index_cpu = data.edge_index.detach().cpu()
    num_nodes = int(data.num_nodes)

    # ---- adjacency (scipy CSR) ----
    adj = to_scipy_sparse_matrix(edge_index_cpu, num_nodes=num_nodes).tocsr()

    # ---- features (scipy sparse) ----
    x_cpu = data.x.detach().cpu().numpy()
    features_sp = sp.csr_matrix(x_cpu.astype(np.float32))
    feature_tuple = _preprocess_features(features_sp)
    feature_sp = _tuple_to_torch_sparse(feature_tuple)
    input_dim = feature_tuple[2][1]
    num_features_nonzero = int(feature_tuple[1].shape[0])

    # ---- support (D^-0.5 (A+I) D^-0.5) ----
    support_tuple = _preprocess_adj(adj)
    support_sp = _tuple_to_torch_sparse(support_tuple)
    edge_pos = np.asarray(support_tuple[0])

    # ---- hierarchy via graph coarsening ----
    graph, _ = _read_graph_from_adj(adj)
    transfer_list = []
    adj_list = [copy.copy(graph.A)]
    node_wgt_list = [copy.copy(graph.node_wgt)]
    for _ in range(coarsen_level):
        match, coarse_size = generate_hybrid_matching(max_node_wgt, graph)
        coarse_graph = create_coarse_graph(graph, match, coarse_size)
        transfer_list.append(copy.copy(graph.C))
        graph = coarse_graph
        adj_list.append(copy.copy(graph.A))
        node_wgt_list.append(copy.copy(graph.node_wgt))
    adj_list = [[_preprocess_adj(a)] * channel_num for a in adj_list]

    # ---- labels & masks (per split, one-hot) ----
    y_int = data.y.detach().cpu().numpy().astype(np.int64)
    num_classes = int(y_int.max()) + 1
    one_hot_all = np.eye(num_classes, dtype=np.float32)[y_int]

    def _split_oh(mask_t: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        m = mask_t.detach().cpu().numpy().astype(bool)
        oh = np.zeros_like(one_hot_all)
        oh[m] = one_hot_all[m]
        return oh, m.astype(np.int32)

    y_train_oh_np, train_mask_np = _split_oh(data.train_mask)
    if hasattr(data, "val_mask") and data.val_mask is not None:
        y_val_oh_np, val_mask_np = _split_oh(data.val_mask)
    else:
        y_val_oh_np = np.zeros_like(one_hot_all)
        val_mask_np = np.zeros(num_nodes, dtype=np.int32)
    if hasattr(data, "test_mask") and data.test_mask is not None:
        y_test_oh_np, test_mask_np = _split_oh(data.test_mask)
    else:
        y_test_oh_np = np.zeros_like(one_hot_all)
        test_mask_np = np.zeros(num_nodes, dtype=np.int32)

    # ---- supervised-contrastive support matrices ----
    tr_idx = np.argwhere(train_mask_np > 0)[:, 0]
    train_mat01 = _cal_class01_mat(y_train_oh_np, train_mask_np)
    if tr_idx.size > 0:
        y_dim1 = np.argmax(y_train_oh_np, axis=1)
        mats = _cal_intra_class_mat01(y_dim1[tr_idx])
        num_labeled = int(tr_idx.size)
        mats[0] = mats[0] + np.eye(num_labeled)
    else:
        mats = [np.zeros((0, 0)), np.zeros((0, 0))]

    return CG3Artifacts(
        input_dim=int(input_dim),
        num_classes=int(num_classes),
        num_nodes=num_nodes,
        feature_sp=feature_sp,
        support_sp=support_sp,
        support_tuple=support_tuple,
        edge_pos=edge_pos,
        num_features_nonzero=num_features_nonzero,
        transfer_list=transfer_list,
        adj_list=adj_list,
        node_wgt_list=node_wgt_list,
        y_train_oh=torch.from_numpy(y_train_oh_np.astype(np.float32)),
        train_mask_int=torch.from_numpy(train_mask_np.astype(np.int32)),
        y_val_oh=torch.from_numpy(y_val_oh_np.astype(np.float32)),
        val_mask_int=torch.from_numpy(val_mask_np.astype(np.int32)),
        y_test_oh=torch.from_numpy(y_test_oh_np.astype(np.float32)),
        test_mask_int=torch.from_numpy(test_mask_np.astype(np.int32)),
        train_idx_np=tr_idx,
        train_mat01=train_mat01,
        mats_intra_inter=mats,
    )


def _read_graph_from_adj(adj: sp.spmatrix) -> tuple[Graph, dict]:
    """Build a `Graph` (the coarsening pipeline's input) from a scipy adj.

    Mirrors snapshot's `utils.read_graph_from_adj` for the case where we don't
    need a node-id mapping.
    """
    if not sp.isspmatrix_csr(adj):
        adj = adj.tocsr()
    node_num = adj.shape[0]
    edge_num = int(adj.nnz)
    graph = Graph(node_num, edge_num)
    edge_ptr = 0
    graph.adj_idx[0] = 0
    for i in range(node_num):
        neighbors = adj.indices[adj.indptr[i]:adj.indptr[i + 1]]
        weights = adj.data[adj.indptr[i]:adj.indptr[i + 1]]
        for j in range(len(neighbors)):
            neigh = int(neighbors[j])
            graph.adj_list[edge_ptr] = neigh
            graph.adj_wgt[edge_ptr] = float(weights[j])
            graph.degree[i] += float(weights[j])
            edge_ptr += 1
        graph.adj_idx[i + 1] = edge_ptr
        graph.node_wgt[i] = 1
    graph.A = adj
    return graph, {}
