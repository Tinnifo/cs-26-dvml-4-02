from __future__ import division
from __future__ import print_function

import time
import numpy as np
import torch

from utils import *
from models import HGCN
from coarsen import *
from config import FLAGS
import copy
import pickle as pkl


def HGCN_Model(placeholders, paras):
    # Settings (overrides via paras)
    FLAGS.dataset = paras['dataset']
    FLAGS.model = 'hgcn'
    FLAGS.seed1 = 123
    FLAGS.seed2 = 123
    FLAGS.hidden = 32
    FLAGS.node_wgt_embed_dim = 5
    FLAGS.weight_decay = paras['weight_decay']
    FLAGS.coarsen_level = 4
    FLAGS.max_node_wgt = 50
    FLAGS.channel_num = 4

    # Set random seed
    seed1 = FLAGS.seed1
    seed2 = FLAGS.seed2
    np.random.seed(seed1)
    torch.manual_seed(seed2)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed2)

    # Load data
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS.dataset)

    # Some preprocessing
    features = preprocess_features(features)
    support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = HGCN

    graph, mapping = read_graph_from_adj(adj, FLAGS.dataset)
    print('total nodes:', graph.node_num)

    # Step-1: Graph Coarsening.
    original_graph = graph
    transfer_list = []
    adj_list = [copy.copy(graph.A)]
    node_wgt_list = [copy.copy(graph.node_wgt)]
    for i in range(FLAGS.coarsen_level):
        match, coarse_graph_size = generate_hybrid_matching(FLAGS.max_node_wgt, graph)
        coarse_graph = create_coarse_graph(graph, match, coarse_graph_size)
        transfer_list.append(copy.copy(graph.C))
        graph = coarse_graph
        adj_list.append(copy.copy(graph.A))
        node_wgt_list.append(copy.copy(graph.node_wgt))
        print('There are %d nodes in the %d coarsened graph' % (graph.node_num, i + 1))

    print("\n")
    print('layer_index ', 1)
    print('input shape:   ', features[-1])

    for i in range(len(adj_list)):
        adj_list[i] = [preprocess_adj(adj_list[i])]

    return model_func(placeholders, input_dim=features[2][1], logging=True,
                      transfer_list=transfer_list, adj_list=adj_list, node_wgt_list=node_wgt_list)


if __name__ == "__main__":
    HGCNModel = HGCN_Model(placeholders, paras)