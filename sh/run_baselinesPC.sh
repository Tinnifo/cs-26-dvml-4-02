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
# ALWAYS go to submission directory
# ------------------------------------------------------------
cd "$SLURM_SUBMIT_DIR" || exit 1

echo "SLURM_SUBMIT_DIR: $SLURM_SUBMIT_DIR"
echo "Current dir: $(pwd)"

# ------------------------------------------------------------
# If submitted from subfolder (e.g. sh/), jump to repo root
# ------------------------------------------------------------
if [ ! -f "src/train.py" ]; then
    echo "src/train.py not found here, moving to repo root..."

    cd "$(dirname "$SLURM_SUBMIT_DIR")" || exit 1
fi

# final safety check
if [ ! -f "src/train.py" ]; then
    echo "ERROR: cannot find src/train.py"
    echo "Current dir: $(pwd)"
    ls -la
    exit 1
fi

mkdir -p logs

# ------------------------------------------------------------
# Activate environment
# ------------------------------------------------------------
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    echo "ERROR: .venv not found"
    ls -la
    exit 1
fi

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

# ------------------------------------------------------------
# Debug
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