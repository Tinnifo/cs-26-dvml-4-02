#!/bin/bash
#SBATCH --job-name=gnn-gcn
#SBATCH --partition=prioritized
#SBATCH --account=aau
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/gcn_%j.out

# Load any necessary modules (adjust if needed for AI-LAB)
# module load cuda

# Activate virtual environment
source .venv/bin/activate

# Set WandB entity if provided as an environment variable, otherwise use default
WANDB_ENTITY=${WANDB_ENTITY:-""}

python3 src/train.py --model GCN --dataset Cora --budget 20 --use_wandb ${WANDB_ENTITY:+--wandb_entity $WANDB_ENTITY}
