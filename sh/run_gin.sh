#!/bin/bash
#SBATCH --job-name=gnn-gin
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/gin_%j.out

source .venv/bin/activate
WANDB_ENTITY=${WANDB_ENTITY:-""}

python3 src/train.py --model GIN --dataset Cora --budget 20 --use_wandb ${WANDB_ENTITY:+--wandb_entity $WANDB_ENTITY}
