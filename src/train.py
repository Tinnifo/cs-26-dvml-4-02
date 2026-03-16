import torch
import torch.nn.functional as F
import numpy as np
import argparse
import sys
import os
import wandb
import json

# Add parent directory to path to allow imports from eval
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch_geometric.datasets import Planetoid
from src.models import GCN, GAT, GraphSAGE, GIN, GraphTransformer
from eval.evaluation import evaluate
from eval.Utils import set_few_label_mask, set_budget_percent, set_seed

def get_model(name, num_features, num_classes, hidden_dim, dropout=0.5):
    if name == "GCN":
        return GCN(num_features, hidden_dim, num_classes, dropout=dropout)
    elif name == "GAT":
        return GAT(num_features, hidden_dim, num_classes, dropout=dropout)
    elif name == "GIN":
        return GIN(num_features, hidden_dim, num_classes, dropout=dropout)
    elif name == "SAGE":
        return GraphSAGE(num_features, hidden_dim, num_classes, dropout=dropout)
    elif name == "GT":
        return GraphTransformer(num_features, hidden_dim, num_classes, dropout=dropout)
    else:
        raise ValueError(f"Unknown model: {name}")

def main():
    parser = argparse.ArgumentParser(description='Train GNN models on Planetoid datasets')
    parser.add_argument('--dataset', type=str, default='Cora', choices=['Cora', 'CiteSeer', 'PubMed'], help='Dataset name')
    parser.add_argument('--model', type=str, default='GCN', choices=['GCN', 'GAT', 'GIN', 'SAGE', 'GT'], help='Model name')
    parser.add_argument('--budget', type=float, default=20, help='Number of labels per class OR fraction of total nodes')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=16, help='Hidden dimension')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--patience', type=int, default=50, help='Early stopping patience')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4], help='Random seeds for experiments')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='gnn-experiments', help='WandB project name')
    parser.add_argument('--wandb_entity', type=str, default='cs-26-dvml-4-02', help='WandB entity (team or username)')
    parser.add_argument('--config', type=str, default=None, help='Path to JSON config file')
    args = parser.parse_args()

    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Override args with config file if provided
    if args.config:
        with open(args.config, 'r') as f:
            config_data = json.load(f)
            for k, v in config_data.items():
                if hasattr(args, k):
                    setattr(args, k, v)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    budget_str = f"{int(args.budget)}" if args.budget >= 1 else f"{args.budget*100:.1f}%"
    print(f"\n========== DATASET: {args.dataset} | MODEL: {args.model} | BUDGET: {budget_str} ==========")

    if args.use_wandb:
        run_name = f"{args.model}_{args.dataset}_budget{budget_str}_{timestamp}"
        
        # Determine budget type
        budget_type = "per-class" if args.budget >= 1 else "percentage"
        
        # Get dataset specific budgets if available in config_data
        dataset_budgets = []
        if args.config:
            with open(args.config, 'r') as f:
                config_data = json.load(f)
                dataset_budgets = config_data.get('dataset_budgets', {}).get(args.dataset, [])

        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config={
                **vars(args),
                "budget_type": budget_type,
                "dataset_budgets": dataset_budgets
            }
        )

    # Use a consistent data directory
    dataset = Planetoid(root='data', name=args.dataset)
    base_data = dataset[0].to(device)

    all_metrics = []

    for seed in args.seeds:
        set_seed(seed)
        data = base_data.clone()
        
        # Handle both per-class budget and percentage budget
        if args.budget >= 1:
            data = set_few_label_mask(data, int(args.budget), seed)
        else:
            data = set_budget_percent(data, args.budget, seed)

        model = get_model(args.model, dataset.num_features, dataset.num_classes, args.hidden_dim, args.dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # Training
        best_val_acc = 0
        counter = 0

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
            
            # Validation step
            if hasattr(data, 'val_mask') and data.val_mask.sum() > 0:
                val_metrics = evaluate(model, data, mask=data.val_mask)
                val_acc = val_metrics[0]
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    counter = 0
                else:
                    counter += 1
                
                if args.use_wandb and len(args.seeds) == 1:
                    wandb.log({
                        "loss": loss.item(), 
                        "val_acc": val_acc,
                        "epoch": epoch
                    })
                
                if counter >= args.patience:
                    print(f"Seed {seed}: Early stopping at epoch {epoch}")
                    break
            else:
                if args.use_wandb and len(args.seeds) == 1:
                    wandb.log({"loss": loss.item(), "epoch": epoch})

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

    results = {
        "mean_accuracy": mean_metrics[0],
        "std_accuracy": std_metrics[0],
        "mean_macro_f1": mean_metrics[3],
        "std_macro_f1": std_metrics[3],
    }

    if args.use_wandb:
        wandb.log(results)
        wandb.finish()

    print(
        f"Results for {args.model} on {args.dataset} (Budget {budget_str}):\n"
        f"Accuracy:  {mean_metrics[0]:.4f} ± {std_metrics[0]:.4f}\n"
        f"Macro F1:  {mean_metrics[3]:.4f} ± {std_metrics[3]:.4f}"
    )

if __name__ == "__main__":
    main()
