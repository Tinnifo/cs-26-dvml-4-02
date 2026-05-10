#!/bin/bash
#!/bin/bash
#SBATCH --job-name=cg3_pubmed_hgcn
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
method.local_model=gcn \
method.global_model=hgcn \
dataset=pubmed \
label_strategy=per_class \
label_strategy.budget=1,3,5,10,20 \
seeds=[0,1,2] \
device=cuda