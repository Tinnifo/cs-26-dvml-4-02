#!/bin/bash
#SBATCH --job-name=pbgnn-baselines
#SBATCH --partition=l4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --array=0-2


# ============================================================
# 2. PERCENTAGE BUDGETS
# Different datasets use different percentage ranges
# ============================================================

# 1. Move to the directory where you submitted the job
set -euo pipefail

cd /ceph/home/student.aau.dk/ab10ix/cs-26-dvml-4-02 || exit 1
mkdir -p logs

# 2. Load necessary cluster modules (Ask your admin for the exact names)
# module load cuda/12.1

# 3. Activate environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "Error: .venv not found in $(pwd)"
    exit 1
fi


export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

# 4. Debug info - This will show up in your .out log
echo "Working directory: $(pwd)"
echo "Using python: $(which python)"
nvidia-smi

DATASETS=(cora citeseer pubmed)
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

echo "Running percentage sweep for: $DATASET"

if [ "$DATASET" = "cora" ]; then
    BUDGETS="0.005,0.01,0.02,0.03,0.04"
elif [ "$DATASET" = "citeseer" ]; then
    BUDGETS="0.005,0.01,0.015,0.02,0.03"
elif [ "$DATASET" = "pubmed" ]; then
    BUDGETS="0.0005,0.001,0.0015,0.002,0.0025"
fi

python3 src/train.py --multirun \
    model=gcn,gat,gin,sage,gt,diff \
    method=vanilla,iceberg \
    dataset=$DATASET \
    label_strategy=percentage \
    label_strategy.budget=$BUDGETS \
    device=cuda

echo "Done: $DATASET"