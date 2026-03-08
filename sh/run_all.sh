#!/bin/bash
#SBATCH --job-name=gnn-all-models
#SBATCH --partition=prioritized
#SBATCH --account=aau
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/run_all_%j.out

source .venv/bin/activate
WANDB_ENTITY=${WANDB_ENTITY:-""}

echo "Running all GNN experiments sequentially..."

for model in GCN GAT SAGE GIN GT
do
    echo "------------------------------------------------"
    echo "Running $model..."
    python3 src/train.py --model $model --dataset Cora --budget 20 --use_wandb ${WANDB_ENTITY:+--wandb_entity $WANDB_ENTITY}
done
