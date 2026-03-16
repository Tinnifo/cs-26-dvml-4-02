import torch
import torch.nn.functional as F
import numpy as np
import argparse
import sys
import os
import wandb
import json
import datetime

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
    parser.add_argument('--wandb_project', type=str, default='gnn-experiments-test', help='WandB project name')
    parser.add_argument('--wandb_entity', type=str, default='cs-26-dvml-4-02', help='WandB entity (team or username)')
    parser.add_argument('--wandb_group', type=str, default='combined-runs', help='WandB group name')
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for dataset_name in config['datasets']:
        print(f"\n========== DATASET: {dataset_name} ==========")

        dataset = Planetoid(root='data', name=dataset_name)
        base_data = dataset[0].to(device)

        # Combine per-class budgets and dataset-specific percentage budgets
        current_budgets = config['budgets'] + config['dataset_budgets'].get(dataset_name, [])

        for budget in current_budgets:
            # Format budget string for naming/display (use 3 decimal places for small percentages)
            if budget >= 1:
                budget_str = f"{int(budget)}"
            else:
                budget_str = f"{budget*100:.3f}%" if budget < 0.001 else f"{budget*100:.2f}%"
            
            print(f"\n---- Budget: {budget_str} ----")
            results_for_table = []

            for model_name in config['models']:
                print(f"Model: {model_name}")
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                
                if args.use_wandb:
                    run_name = f"{model_name}_{dataset_name}_budget{budget_str}_{timestamp}"
                    run = wandb.init(
                        project=args.wandb_project,
                        entity=args.wandb_entity,
                        group=f"{args.wandb_group}_{dataset_name}_budget{budget_str}",
                        name=run_name,
                        config={
                            **config,
                            "dataset": dataset_name,
                            "model": model_name,
                            "budget": budget,
                            "budget_type": "per-class" if budget >= 1 else "percentage",
                            "dataset_budgets": config['dataset_budgets'].get(dataset_name, [])
                        },
                        reinit=True
                    )
                    
                    # x-axis options
                    run.define_metric("budget", hidden=True)
                    run.define_metric("dataset_budget", hidden=True)

                    # curves vs budget
                    run.define_metric("mean_accuracy", step_metric="budget")
                    run.define_metric("mean_macro_f1", step_metric="budget")

                    # curves vs dataset_budget
                    run.define_metric("mean_accuracy_vs_dataset_budget", step_metric="dataset_budget")
                    run.define_metric("mean_macro_f1_vs_dataset_budget", step_metric="dataset_budget")

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

                            if val_acc > best_val_acc:
                                best_val_acc = val_acc
                                counter = 0
                            else:
                                counter += 1
                            if counter >= config['patience']:
                                break

                    metrics = evaluate(model, data)
                    all_metrics.append(metrics)

                all_metrics = np.array(all_metrics)
                mean_metrics = np.mean(all_metrics, axis=0)
                std_metrics = np.std(all_metrics, axis=0)
                
                # Calculate margin of error (95% CI)
                n = len(config['seeds'])
                moe_acc = 1.96 * (std_metrics[0] / np.sqrt(n))
                moe_f1 = 1.96 * (std_metrics[3] / np.sqrt(n))

                # Store for the summary table
                results_for_table.append([
                    model_name,
                    mean_metrics[0], std_metrics[0], moe_acc,
                    mean_metrics[3], std_metrics[3], moe_f1
                ])

                if args.use_wandb:
                    ds_budgets = config['dataset_budgets'].get(dataset_name, [])
                    dataset_budget = budget if budget in ds_budgets else None
                    
                    log_dict = {
                        "budget": budget,
                        "mean_accuracy": mean_metrics[0],
                        "mean_macro_f1": mean_metrics[3],
                        "std_accuracy": std_metrics[0],
                        "std_macro_f1": std_metrics[3],
                        "moe_accuracy": moe_acc,
                        "moe_macro_f1": moe_f1,
                    }
                    if dataset_budget is not None:
                        log_dict.update({
                            "dataset_budget": dataset_budget,
                            "mean_accuracy_vs_dataset_budget": mean_metrics[0],
                            "mean_macro_f1_vs_dataset_budget": mean_metrics[3],
                        })
                    wandb.log(log_dict)
                    run.finish()

            # After all models for this budget, log the summary table
            if args.use_wandb and results_for_table:
                summary_run = wandb.init(
                    project=args.wandb_project,
                    entity=args.wandb_entity,
                    group=f"{args.wandb_group}_{dataset_name}_budget{budget_str}",
                    name=f"Summary_{dataset_name}_budget{budget_str}",
                    config={
                        **config,
                        "dataset": dataset_name,
                        "budget": budget,
                        "budget_type": "per-class" if budget >= 1 else "percentage",
                        "dataset_budgets": config['dataset_budgets'].get(dataset_name, [])
                    },
                    reinit=True
                )
                
                columns = [
                    "Model", 
                    "Mean Accuracy", "Std Accuracy", "MoE Accuracy", 
                    "Mean Macro F1", "Std Macro F1", "MoE Macro F1"
                ]
                summary_table = wandb.Table(columns=columns, data=results_for_table)
                summary_run.log({"results_table": summary_table})
                summary_run.finish()

if __name__ == "__main__":
    main()
