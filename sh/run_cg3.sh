#!/bin/bash
#!/bin/bash
#SBATCH --job-name=cg3_hgcn_sweep
#SBATCH --partition=l4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out

source .venv/bin/activate

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export CUDA_LAUNCH_BLOCKING=1

python src/train.py --multirun \
method=cg3 \
method.local_model=gcn,gat \
method.global_model=hgcn \
dataset=cora,citeseer,pubmed \
label_strategy=percentage \
seeds=[0,1,2] \
device=cuda