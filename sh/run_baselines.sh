#!/bin/bash
#SBATCH --job-name=gnn-baselines
#SBATCH --partition=l4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=15
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err


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

echo "Starting baseline sweeps..."

# ============================================================
# 1. PER-CLASS BUDGETS
# ============================================================
echo "[1/2] Running per-class sweeps..."

python3 src/train.py --multirun \
    model=gcn,gat,gin,sage,gt,diff \
    method=vanilla,iceberg \
    dataset=cora,citeseer,pubmed \
    label_strategy=per_class \
    label_strategy.budget=1,3,5,10,20 \
    device=cuda

# ============================================================
# 2. PERCENTAGE BUDGETS
# Different datasets use different percentage ranges
# ============================================================
echo "[2/2] Running percentage sweeps..."

declare -A PCT_BUDGETS=(
    [cora]="0.005,0.01,0.02,0.03,0.04"
    [citeseer]="0.005,0.01,0.015,0.02,0.03"
    [pubmed]="0.0005,0.001,0.0015,0.002,0.0025"
)

for ds in cora citeseer pubmed; do
    echo "Running percentage sweep for $ds"

    python3 src/train.py --multirun \
        model=gcn,gat,gin,sage,gt,diff \
        method=vanilla,iceberg \
        dataset=$ds \
        label_strategy=percentage \
        label_strategy.budget=${PCT_BUDGETS[$ds]} \
        device=cuda
done

echo "All baseline sweeps complete."

