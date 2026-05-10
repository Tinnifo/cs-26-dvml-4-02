#!/bin/bash
#!/bin/bash
#SBATCH --job-name=cg3_hgcn_sweep
#SBATCH --partition=l4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out

source .venv/bin/activate

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export CUDA_LAUNCH_BLOCKING=1

python src/train.py --multirun \
method=cg3 \
method.local_model=gcn,gat \
method.global_model=hgcn \
dataset=cora,citeseer,pubmed \
dataset.percentage_budgets='[0.005,0.01,0.02,0.03,0.04]','[0.005,0.01,0.015,0.02,0.03]','[0.0005,0.001,0.0015,0.002,0.0025]' \
label_strategy=percentage \
seeds=[0,1,2] \
device=cuda