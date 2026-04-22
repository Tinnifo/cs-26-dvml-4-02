from cg3_model import CG3Model
from train import train, train_gcn
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


from build_hierarchy_from_coarsen import build_hierarchy, graph_to_edge_index
from torch_geometric.utils import to_scipy_sparse_matrix

import torch
import numpy as np
import random

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

class GCNBaseline(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_classes)

    def forward(self, x, edge_index, edge_weight):
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = self.conv2(x, edge_index, edge_weight)
        return x
    

if __name__ == "__main__":

    seeds = [0, 1, 2, 3, 4]

    cg3_results = []
    gcn_results = []

    for seed in seeds:
        print(f"\n===== SEED {seed} =====")
        set_seed(seed)

        dataset = Planetoid(
            root='/tmp/Cora',
            name='Cora',
            transform=T.NormalizeFeatures()
        )

        data = dataset[0]

        adj = to_scipy_sparse_matrix(data.edge_index).tocsr()
        edge_levels, C_matrices, graphs = build_hierarchy(adj)
        edge_index, edge_weight = graph_to_edge_index(graphs[0])

        data.edge_index = edge_index
        data.edge_weight = edge_weight

        # ----------------------
        # CG3
        # ----------------------
        model = CG3Model(
            in_dim=dataset.num_node_features,
            hidden_dim=64,
            num_classes=dataset.num_classes
        )

        model.hgcn.set_hierarchy(edge_levels, C_matrices)

        train(model, data)

        with torch.no_grad():
            _, _, _, logits = model(
                data.x,
                data.edge_index,
                data.edge_weight
            )
            pred = logits.argmax(dim=1)
            acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean().item()

        cg3_results.append(acc)

        # ----------------------
        # GCN
        # ----------------------
        gcn = GCNBaseline(
            in_dim=dataset.num_node_features,
            hidden_dim=64,
            num_classes=dataset.num_classes
        )

        train_gcn(gcn, data)

        with torch.no_grad():
            logits = gcn(data.x, data.edge_index, data.edge_weight)
            pred = logits.argmax(dim=1)
            acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean().item()

        gcn_results.append(acc)
    
print("\n===== FINAL RESULTS =====")
print(f"CG3: {np.mean(cg3_results):.4f} ± {np.std(cg3_results):.4f}")
print(f"GCN: {np.mean(gcn_results):.4f} ± {np.std(gcn_results):.4f}")