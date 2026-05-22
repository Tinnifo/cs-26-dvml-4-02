#!/bin/bash
#SBATCH --job-name=pcgnn-baselines
#SBATCH --partition=l4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# ------------------------------------------------------------
# ALWAYS run from submission directory (VERY IMPORTANT FIX)
# ------------------------------------------------------------
cd "$SLURM_SUBMIT_DIR" || exit 1

mkdir -p logs

echo "Working directory: $(pwd)"
echo "SLURM submit dir: $SLURM_SUBMIT_DIR"

# ------------------------------------------------------------
# Activate environment (safe version)
# ------------------------------------------------------------
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    echo "ERROR: .venv not found in $(pwd)"
    echo "Contents:"
    ls -la
    exit 1
fi

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

# ------------------------------------------------------------
# Debug info
# ------------------------------------------------------------
echo "Using python: $(which python)"
nvidia-smi

# ------------------------------------------------------------
# Run experiment
# ------------------------------------------------------------
python3 src/train.py --multirun \
    model=gin \
    method=iceberg \
    dataset=pubmed \
    label_strategy=per_class \
    label_strategy.budget=1,20 \
    device=cuda