import torch
import numpy as np

from coarsen import generate_hybrid_matching, create_coarse_graph
from graph import Graph

import scipy.sparse as sp


def cmap2C(cmap):
    node_num = len(cmap)
    i_arr = []
    j_arr = []
    data_arr = []

    for i in range(node_num):
        i_arr.append(i)
        j_arr.append(cmap[i])
        data_arr.append(1)

    return sp.csr_matrix((data_arr, (i_arr, j_arr)))

def convert_adj_to_graph(adj):
    """
    Convert scipy sparse adjacency matrix → Graph object
    EXACTLY like original HGCN pipeline expects.
    """

    # Ensure CSR format (same as TF code assumptions)
    if not sp.isspmatrix_csr(adj):
        adj = adj.tocsr()

    node_num = adj.shape[0]
    edge_num = adj.nnz   # counts directed edges (same as original)

    graph = Graph(node_num, edge_num)

    adj_idx = graph.adj_idx
    adj_list = graph.adj_list
    adj_wgt = graph.adj_wgt
    degree = graph.degree
    node_wgt = graph.node_wgt

    edge_ptr = 0
    adj_idx[0] = 0

    # -------------------------------------------------------
    # Build adjacency structure EXACTLY like original
    # -------------------------------------------------------
    for i in range(node_num):

        neighbors = adj.indices[adj.indptr[i]:adj.indptr[i + 1]]
        weights = adj.data[adj.indptr[i]:adj.indptr[i + 1]]

        for j in range(len(neighbors)):
            neigh = neighbors[j]

            adj_list[edge_ptr] = neigh
            adj_wgt[edge_ptr] = weights[j]

            degree[i] += weights[j]

            edge_ptr += 1

        adj_idx[i + 1] = edge_ptr

        # IMPORTANT (paper detail): all nodes start with weight = 1
        node_wgt[i] = 1

    # Store adjacency (used later in coarsening)
    graph.A = adj

    return graph


def build_hierarchy(adj, num_levels=3):
    """
    Builds multi-level graph hierarchy exactly like HGCN paper.
    """

    graphs = []
    C_matrices = []
    edge_levels = []

    # ----------------------
    # Level 0 (original graph)
    # ----------------------
    graph = convert_adj_to_graph(adj)
    graphs.append(graph)
    edge_levels.append(graph_to_edge_index(graph))

    # ----------------------
    # Coarsening loop
    # ----------------------
    for l in range(num_levels - 1):

        match, coarse_size = generate_hybrid_matching(
            max_node_wgt=50,
            graph=graph
        )

        coarse_graph = create_coarse_graph(
            graph,
            match,
            coarse_size
        )

        # C matrix (fine → coarse)
        C = cmap2C(graph.cmap).astype(np.float32)
        C = scipy_to_torch_sparse(C)
        C_matrices.append(C)

        graph = coarse_graph
        graphs.append(graph)
        edge_levels.append(graph_to_edge_index(graph))

    return edge_levels, C_matrices, graphs

def graph_to_edge_index(graph):
    row = []
    col = []
    weight = []

    for i in range(graph.node_num):
        start = graph.adj_idx[i]
        end = graph.adj_idx[i + 1]

        for idx in range(start, end):
            j = graph.adj_list[idx]

            row.append(i)
            col.append(j)
            weight.append(graph.adj_wgt[idx])   # IMPORTANT (paper detail)

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_weight = torch.tensor(weight, dtype=torch.float32)

    return edge_index, edge_weight


def scipy_to_torch_sparse(mat):
    mat = mat.tocoo()

    indices = torch.tensor(
        np.vstack((mat.row, mat.col)),
        dtype=torch.long
    )

    values = torch.tensor(mat.data, dtype=torch.float32)

    shape = torch.Size(mat.shape)
    
    return torch.sparse_coo_tensor(
    indices,
    values,
    shape
).coalesce()


