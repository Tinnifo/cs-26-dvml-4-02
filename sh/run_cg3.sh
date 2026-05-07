#!/bin/bash
#SBATCH --job-name=cg3-grid
#SBATCH --partition=l4
#SBATCH --gres=gpu=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=logs/cg3_%j.out

set -e
source .venv/bin/activate

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export CUDA_VISIBLE_DEVICES=0

echo "[CG3] running combinations..."

python3 src/train.py --multirun +experiment=cg3_combinations device=cuda