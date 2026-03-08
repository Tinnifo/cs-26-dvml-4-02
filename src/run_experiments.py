import torch
import torch.nn.functional as F
import numpy as np
import sys
import os

# Add parent directory to path to allow imports from eval
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch_geometric.datasets import Planetoid
from src.GCN import GCN
from src.GAT import GAT
from src.SAGE import GraphSAGE
from src.GIN import GIN
from src.GT import GraphTransformer
from eval.evaluation import evaluate
from eval.Utils import set_few_label_mask

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

def get_model(name, num_features, num_classes, hidden_dim):
    if name == "GCN":
        return GCN(num_features, hidden_dim, num_classes)
    elif name == "GAT":
        return GAT(num_features, hidden_dim, num_classes)
    elif name == "GIN":
        return GIN(num_features, hidden_dim, num_classes)
    elif name == "SAGE":
        return GraphSAGE(num_features, hidden_dim, num_classes)
    elif name == "GT":
        return GraphTransformer(num_features, hidden_dim, num_classes)
    else:
        raise ValueError(f"Unknown model: {name}")

def main():
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

                    model = get_model(model_name, dataset.num_features, dataset.num_classes, HIDDEN_DIM).to(device)
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

if __name__ == "__main__":
    main()
