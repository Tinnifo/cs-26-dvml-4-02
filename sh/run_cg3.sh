#!/bin/bash
#SBATCH --job-name=cg3_hgcn_sweep
#SBATCH --partition=l4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out

cd $SLURM_SUBMIT_DIR || exit 1
mkdir -p logs

source .venv/bin/activate

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export CUDA_LAUNCH_BLOCKING=1

python src/train.py --multirun \
method=cg3 \
method.local_model=gcn \
method.global_model=hgcn \
dataset=pubmed \
label_strategy.budget=1,3,5,10,20 \
method.max_node_wgt=500 \
device=cuda