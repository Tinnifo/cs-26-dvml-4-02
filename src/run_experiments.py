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
    parser = argparse.ArgumentParser(description='Run all GNN experiments')
    parser.add_argument('--config', type=str, default='src/config.json', help='Path to JSON config file')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='gnn-experiments-tinni', help='WandB project name')
    parser.add_argument('--wandb_entity', type=str, default='cs-26-dvml-4-02', help='WandB entity (team or username)')
    parser.add_argument('--wandb_group', type=str, default='combined-runs', help='WandB group name')
    args = parser.parse_args()

    import datetime

    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for dataset_name in config['datasets']:
        print(f"\n========== DATASET: {dataset_name} ==========")

        dataset = Planetoid(root='data', name=dataset_name)
        base_data = dataset[0].to(device)

        for model_name in config['models']:
            print(f"\n---- Model: {model_name} ----")

            # Combine per-class budgets and dataset-specific percentage budgets
            current_budgets = config['budgets'] + config['dataset_budgets'].get(dataset_name, [])
            
            for budget in current_budgets:
                # Format budget string for naming/display (use 3 decimal places for small percentages)
                if budget >= 1:
                    budget_str = f"{int(budget)}"
                else:
                    budget_str = f"{budget*100:.3f}%" if budget < 0.001 else f"{budget*100:.2f}%"
                
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                
                if args.use_wandb:
                    run_name = f"{model_name}_{dataset_name}_budget{budget_str}_{timestamp}"
                    run = wandb.init(
                        project=args.wandb_project,
                        entity=args.wandb_entity,
                        group=args.wandb_group,
                        name=run_name,
                        config={
                            "dataset": dataset_name,
                            "model": model_name,
                            "budget": budget,
                            **config
                        },
                        reinit=True
                    )

                all_metrics = []

                for seed in config['seeds']:
                    set_seed(seed)
                    data = base_data.clone()
                    
                    if budget >= 1:
                        data = set_few_label_mask(data, int(budget), seed)
                    else:
                        data = set_budget_percent(data, budget, seed)

                    model = get_model(model_name, dataset.num_features, dataset.num_classes, config['hidden_dim'], config['dropout']).to(device)
                    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

                    # Training
                    best_val_acc = 0
                    counter = 0

                    for epoch in range(config['epochs']):
                        model.train()
                        optimizer.zero_grad()
                        out = model(data.x, data.edge_index)
                        loss = F.cross_entropy(
                            out[data.train_mask],
                            data.y[data.train_mask]
                        )
                        loss.backward()
                        optimizer.step()

                        # Validation for early stopping
                        if hasattr(data, 'val_mask') and data.val_mask.sum() > 0:
                            val_metrics = evaluate(model, data, mask=data.val_mask)
                            val_acc = val_metrics[0]

                            if args.use_wandb:
                                wandb.log({"val_acc": val_acc, "epoch": epoch})

                            if val_acc > best_val_acc:
                                best_val_acc = val_acc
                                counter = 0
                            else:
                                counter += 1
                            if counter >= config['patience']:
                                break

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
                    f"Budget {budget_str} | "
                    f"Acc={mean_metrics[0]:.4f}±{std_metrics[0]:.4f} | "
                    f"MacroF1={mean_metrics[3]:.4f}±{std_metrics[3]:.4f}"
                )

                if args.use_wandb:
                    wandb.log({
                        "mean_accuracy": mean_metrics[0],
                        "std_accuracy": std_metrics[0],
                        "mean_macro_f1": mean_metrics[3],
                        "std_macro_f1": std_metrics[3],
                        "budget": budget,
                        "dataset": dataset_name,
                        "model": model_name,
                        "seed": seed,
                        "dataset_budgets": config['dataset_budgets'].get(dataset_name, []),
                        "epochs": config['epochs'],
                        "patience": config['patience'],
                        "lr": config['lr'],
                        "weight_decay": config['weight_decay'],
                        "hidden_dim": config['hidden_dim'],
                        "dropout": config['dropout'],
                    })
                    run.finish()

if __name__ == "__main__":
    main()
