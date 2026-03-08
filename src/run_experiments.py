import torch
import torch.nn.functional as F
import numpy as np
import argparse
import sys
import os
import wandb

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
    parser = argparse.ArgumentParser(description='Run all GNN experiments')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='gnn-experiments', help='WandB project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='WandB entity (team or username)')
    parser.add_argument('--wandb_group', type=str, default='combined-runs', help='WandB group name')
    args = parser.parse_args()

    for dataset_name in DATASETS:
        print(f"\n========== DATASET: {dataset_name} ==========")

        dataset = Planetoid(root=f'data/{dataset_name}', name=dataset_name)
        base_data = dataset[0].to(device)

        for model_name in MODELS:
            print(f"\n---- Model: {model_name} ----")

            for budget in BUDGETS:
                if args.use_wandb:
                    run = wandb.init(
                        project=args.wandb_project,
                        entity=args.wandb_entity,
                        group=args.wandb_group,
                        name=f"{model_name}_{dataset_name}_budget{budget}",
                        config={
                            "dataset": dataset_name,
                            "model": model_name,
                            "budget": budget,
                            "epochs": EPOCHS,
                            "hidden_dim": HIDDEN_DIM,
                            "lr": LR,
                            "seeds": SEEDS
                        },
                        reinit=True
                    )

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

                    if args.use_wandb:
                        wandb.log({
                            f"seed_{seed}/accuracy": metrics[0],
                            f"seed_{seed}/macro_f1": metrics[3]
                        })

                all_metrics = np.array(all_metrics)
                mean_metrics = np.mean(all_metrics, axis=0)
                std_metrics = np.std(all_metrics, axis=0)

                print(
                    f"Budget {budget} | "
                    f"Acc={mean_metrics[0]:.4f}±{std_metrics[0]:.4f} | "
                    f"MacroF1={mean_metrics[3]:.4f}±{std_metrics[3]:.4f}"
                )

                if args.use_wandb:
                    wandb.log({
                        "mean_accuracy": mean_metrics[0],
                        "std_accuracy": std_metrics[0],
                        "mean_precision": mean_metrics[1],
                        "std_precision": std_metrics[1],
                        "mean_recall": mean_metrics[2],
                        "std_recall": std_metrics[2],
                        "mean_macro_f1": mean_metrics[3],
                        "std_macro_f1": std_metrics[3],
                        "mean_micro_f1": mean_metrics[4],
                        "std_micro_f1": std_metrics[4],
                    })
                    run.finish()

if __name__ == "__main__":
    main()
