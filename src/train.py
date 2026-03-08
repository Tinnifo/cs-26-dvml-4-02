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
    parser = argparse.ArgumentParser(description='Train GNN models on Planetoid datasets')
    parser.add_argument('--dataset', type=str, default='Cora', choices=['Cora', 'CiteSeer', 'PubMed'], help='Dataset name')
    parser.add_argument('--model', type=str, default='GCN', choices=['GCN', 'GAT', 'GIN', 'SAGE', 'GT'], help='Model name')
    parser.add_argument('--budget', type=int, default=20, help='Number of labels per class')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=16, help='Hidden dimension')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4], help='Random seeds for experiments')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='gnn-experiments', help='WandB project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='WandB entity (team or username)')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"\n========== DATASET: {args.dataset} | MODEL: {args.model} | BUDGET: {args.budget} ==========")

    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=f"{args.model}_{args.dataset}_budget{args.budget}",
            config=vars(args)
        )

    dataset = Planetoid(root=f'data/{args.dataset}', name=args.dataset)
    base_data = dataset[0].to(device)

    all_metrics = []

    for seed in args.seeds:
        torch.manual_seed(seed)
        data = base_data.clone()
        data = set_few_label_mask(data, args.budget, seed)

        model = get_model(args.model, dataset.num_features, dataset.num_classes, args.hidden_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        # Training
        for epoch in range(args.epochs):
            model.train()
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = F.cross_entropy(
                out[data.train_mask],
                data.y[data.train_mask]
            )
            loss.backward()
            optimizer.step()
            
            if args.use_wandb and len(args.seeds) == 1:
                wandb.log({"loss": loss.item(), "epoch": epoch})

        metrics = evaluate(model, data)
        all_metrics.append(metrics)
        
        if args.use_wandb and len(args.seeds) > 1:
            wandb.log({
                f"seed_{seed}/accuracy": metrics[0],
                f"seed_{seed}/precision": metrics[1],
                f"seed_{seed}/recall": metrics[2],
                f"seed_{seed}/macro_f1": metrics[3],
                f"seed_{seed}/micro_f1": metrics[4]
            })

    all_metrics = np.array(all_metrics)
    mean_metrics = np.mean(all_metrics, axis=0)
    std_metrics = np.std(all_metrics, axis=0)

    results = {
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
    }

    if args.use_wandb:
        wandb.log(results)
        wandb.finish()

    print(
        f"Results for {args.model} on {args.dataset} (Budget {args.budget}):\n"
        f"Accuracy:  {mean_metrics[0]:.4f} ± {std_metrics[0]:.4f}\n"
        f"Precision: {mean_metrics[1]:.4f} ± {std_metrics[1]:.4f}\n"
        f"Recall:    {mean_metrics[2]:.4f} ± {std_metrics[2]:.4f}\n"
        f"Macro F1:  {mean_metrics[3]:.4f} ± {std_metrics[3]:.4f}\n"
        f"Micro F1:  {mean_metrics[4]:.4f} ± {std_metrics[4]:.4f}"
    )

if __name__ == "__main__":
    main()
