from torch_geometric.datasets import Planetoid
from torch_geometric.nn import (
    GCNConv,
    GATConv,
    SAGEConv,
    GINConv,
    TransformerConv
)



# =============================
# CONFIGURATION
# =============================

DATASETS = ["Cora", "CiteSeer"]
MODELS = ["GCN", "GAT", "GIN", "SAGE", "GT"]
BUDGETS = [1, 3, 5, 10]
SEEDS = [0, 1, 2, 3, 4]
EPOCHS = 200
HIDDEN_DIM = 16
LR = 0.01

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================
# MODELS
# =============================

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x


class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        nn1 = torch.nn.Sequential(
            torch.nn.Linear(in_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels),
        )
        nn2 = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, out_channels),
        )
        self.conv1 = GINConv(nn1)
        self.conv2 = GINConv(nn2)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x


class GraphTransformer(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4):
        super().__init__()
        self.conv1 = TransformerConv(in_channels, hidden_channels, heads=heads)
        self.conv2 = TransformerConv(hidden_channels * heads, out_channels, heads=1)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
... (106 linjer linjer tilbage)

message.txt
7 KB
﻿
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from torch_geometric.datasets import Planetoid
from torch_geometric.nn import (
    GCNConv,
    GATConv,
    SAGEConv,
    GINConv,
    TransformerConv
)

# =============================
# CONFIGURATION
# =============================

DATASETS = ["Cora", "CiteSeer"]
MODELS = ["GCN", "GAT", "GIN", "SAGE", "GT"]
BUDGETS = [1, 3, 5, 10]
SEEDS = [0, 1, 2, 3, 4]
EPOCHS = 200
HIDDEN_DIM = 16
LR = 0.01

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================
# MODELS
# =============================

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x


class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        nn1 = torch.nn.Sequential(
            torch.nn.Linear(in_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels),
        )
        nn2 = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, out_channels),
        )
        self.conv1 = GINConv(nn1)
        self.conv2 = GINConv(nn2)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x


class GraphTransformer(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4):
        super().__init__()
        self.conv1 = TransformerConv(in_channels, hidden_channels, heads=heads)
        self.conv2 = TransformerConv(hidden_channels * heads, out_channels, heads=1)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x


def get_model(name, dataset):
    if name == "GCN":
        return GCN(dataset.num_features, HIDDEN_DIM, dataset.num_classes)
    elif name == "GAT":
        return GAT(dataset.num_features, HIDDEN_DIM, dataset.num_classes)
    elif name == "GIN":
        return GIN(dataset.num_features, HIDDEN_DIM, dataset.num_classes)
    elif name == "SAGE":
        return GraphSAGE(dataset.num_features, HIDDEN_DIM, dataset.num_classes)
    elif name == "GT":
        return GraphTransformer(dataset.num_features, HIDDEN_DIM, dataset.num_classes)
    else:
        raise ValueError("Unknown model")


# =============================
# FEW-LABEL MASK
# =============================

def set_few_label_mask(data, num_labels_per_class, seed):
    torch.manual_seed(seed)
    num_classes = int(data.y.max()) + 1
    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)

    for c in range(num_classes):
        idx = (data.y == c).nonzero(as_tuple=True)[0]
        idx = idx[torch.randperm(idx.size(0))]
        selected = idx[:num_labels_per_class]
        train_mask[selected] = True

    data.train_mask = train_mask
    return data


# =============================
# EVALUATION
# =============================

def evaluate(model, data):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)

    y_true = data.y[data.test_mask].cpu().numpy()
    y_pred = pred[data.test_mask].cpu().numpy()

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)

    return acc, precision, recall, f1_macro, f1_micro


# =============================
# MAIN EXPERIMENT LOOP
# =============================

for dataset_name in DATASETS:
    print(f"\n========== DATASET: {dataset_name} ==========")

    dataset = Planetoid(root=f'data/{dataset_name}', name=dataset_name)
    base_data = dataset[0].to(device)

    for model_name in MODELS:
        print(f"\n---- Model: {model_name} ----")

        for budget in BUDGETS:
            all_metrics = []

            for seed in SEEDS:
                torch.manual_seed(seed)
                data = base_data.clone()
                data = set_few_label_mask(data, budget, seed)

                model = get_model(model_name, dataset).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=LR)

                # Training
                for epoch in range(EPOCHS):
                    model.train()
                    optimizer.zero_grad()
                    out = model(data.x, data.edge_index)
                    loss = F.cross_entropy(
                        out[data.train_mask],
                        data.y[data.train_mask]
                    )
                    loss.backward()
                    optimizer.step()

                metrics = evaluate(model, data)
                all_metrics.append(metrics)

            all_metrics = np.array(all_metrics)
            mean_metrics = np.mean(all_metrics, axis=0)
            std_metrics = np.std(all_metrics, axis=0)

            print(
                f"Budget {budget} | "
                f"Acc={mean_metrics[0]:.4f}±{std_metrics[0]:.4f} | "
                f"MacroF1={mean_metrics[3]:.4f}±{std_metrics[3]:.4f}"
            )
