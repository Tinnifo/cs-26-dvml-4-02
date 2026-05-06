from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import copy

from utils import *
from models import HGCN, HGAT
from coarsen import *
from config import FLAGS


def HGCN_Model(placeholders, paras, adj):
    # -----------------------------
    # Safety check (IMPORTANT)
    # -----------------------------
    if adj is None:
        raise ValueError("Adjacency must be passed from main.py")

    # -----------------------------
    # Config
    # -----------------------------
    FLAGS.dataset = paras['dataset']
    FLAGS.model = 'hgcn'
    FLAGS.seed1 = paras.get('seed1', 123)
    FLAGS.seed2 = paras.get('seed2', 123)
    FLAGS.hidden = 32
    FLAGS.node_wgt_embed_dim = 5
    FLAGS.weight_decay = paras['weight_decay']
    FLAGS.coarsen_level = 4
    FLAGS.max_node_wgt = 50
    FLAGS.channel_num = 4

    # -----------------------------
    # Seed
    # -----------------------------
    np.random.seed(FLAGS.seed1)
    torch.manual_seed(FLAGS.seed2)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(FLAGS.seed2)

    # -----------------------------
    # preprocessing (NOW CORRECT)
    # -----------------------------
    features = placeholders['features']
    support = [preprocess_adj(adj)]

    model_func = HGCN

    graph, mapping = read_graph_from_adj(adj, FLAGS.dataset)
    print('total nodes:', graph.node_num)

    # -----------------------------
    # Graph coarsening
    # -----------------------------
    transfer_list = []
    adj_list = [copy.copy(graph.A)]
    node_wgt_list = [copy.copy(graph.node_wgt)]

    for i in range(FLAGS.coarsen_level):
        match, coarse_graph_size = generate_hybrid_matching(
            FLAGS.max_node_wgt, graph
        )
        coarse_graph = create_coarse_graph(graph, match, coarse_graph_size)

        transfer_list.append(copy.copy(graph.C))
        graph = coarse_graph

        adj_list.append(copy.copy(graph.A))
        node_wgt_list.append(copy.copy(graph.node_wgt))

        print(f'Coarsened level {i+1}: {graph.node_num} nodes')

    for i in range(len(adj_list)):
        adj_list[i] = [preprocess_adj(adj_list[i])]

    print("\nlayer_index 1")
    print("input shape:", features[-1])

    return model_func(
        placeholders,
        input_dim=features[2][1],
        logging=True,
        transfer_list=transfer_list,
        adj_list=adj_list,
        node_wgt_list=node_wgt_list
    )



def HGAT_Model(placeholders, paras, adj):
    # -----------------------------
    # Safety check
    # -----------------------------
    if adj is None:
        raise ValueError("Adjacency must be passed from main.py")

    # -----------------------------
    # Config
    # -----------------------------
    FLAGS.dataset = paras['dataset']
    FLAGS.model = 'hgat'
    FLAGS.seed1 = paras.get('seed1', 123)
    FLAGS.seed2 = paras.get('seed2', 123)
    FLAGS.hidden = 32
    FLAGS.node_wgt_embed_dim = 5
    FLAGS.weight_decay = paras['weight_decay']
    FLAGS.coarsen_level = 4
    FLAGS.max_node_wgt = 50
    FLAGS.channel_num = 4

    # -----------------------------
    # Seed
    # -----------------------------
    np.random.seed(FLAGS.seed1)
    torch.manual_seed(FLAGS.seed2)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(FLAGS.seed2)

    # -----------------------------
    # preprocessing
    # -----------------------------
    features = placeholders['features']
    support = [preprocess_adj(adj)]

    model_func = HGAT   # ✅ ONLY CHANGE vs HGCN

    graph, mapping = read_graph_from_adj(adj, FLAGS.dataset)
    print('total nodes:', graph.node_num)

    # -----------------------------
    # Graph coarsening
    # -----------------------------
    transfer_list = []
    adj_list = [copy.copy(graph.A)]
    node_wgt_list = [copy.copy(graph.node_wgt)]

    for i in range(FLAGS.coarsen_level):
        match, coarse_graph_size = generate_hybrid_matching(
            FLAGS.max_node_wgt, graph
        )
        coarse_graph = create_coarse_graph(graph, match, coarse_graph_size)

        transfer_list.append(copy.copy(graph.C))
        graph = coarse_graph

        adj_list.append(copy.copy(graph.A))
        node_wgt_list.append(copy.copy(graph.node_wgt))

        print(f'Coarsened level {i+1}: {graph.node_num} nodes')

    for i in range(len(adj_list)):
        adj_list[i] = [preprocess_adj(adj_list[i])]

    print("\nlayer_index 1")
    print("input shape:", features[-1])

    # -----------------------------
    # Build HGAT model
    # -----------------------------
    return model_func(
        placeholders,
        input_dim=features[2][1],
        logging=True,
        transfer_list=transfer_list,
        adj_list=adj_list,
        node_wgt_list=node_wgt_list
    )