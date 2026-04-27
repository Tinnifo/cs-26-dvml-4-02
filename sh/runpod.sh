#!/bin/bash
# Complete sweep for a RunPod-style single-A100 SXM instance (no Slurm).
#
# Target hardware:
#   GPU            A100 SXM 1x
#   vCPU           32 (AMD EPYC 7763)
#   Memory         250 GB
#   Container disk 20 GB
#
# Estimated runtime: ~3-5h for the full grid (~1950 training trials at
# ~5-15s each on an A100). TensorBoard logs (~50 MB) and Hydra multirun
# working dirs (~100 MB) sit comfortably inside the 20 GB container disk;
# the .venv install dominates (~4 GB for torch + torch-geometric).

set -e

cd "$(dirname "$0")/.."  # run from repo root regardless of cwd

# ─────────────────────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=0
# 32 vCPU, single training process — give torch a generous (but not maximal)
# intra-op pool so we don't oversubscribe the small models.
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_MAX_THREADS=16

# Activate venv if present; otherwise install into the container's Python.
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
else
    echo "No .venv found — installing requirements into container Python..."
    pip install -q --upgrade pip
    pip install -q -r requirements.txt
fi

python - <<'PY'
import torch
print(f"[runpod] torch={torch.__version__}  cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[runpod] device={torch.cuda.get_device_name(0)}")
PY

# ─────────────────────────────────────────────────────────────────────────────
# Sweeps — same four sections as sh/run.sh, no Slurm wrapping.
# ─────────────────────────────────────────────────────────────────────────────
declare -A PCT_BUDGETS=(
    [cora]="0.005,0.01,0.02,0.03,0.04"
    [citeseer]="0.005,0.01,0.015,0.02,0.03"
    [pubmed]="0.0005,0.001,0.0015,0.002,0.0025"
)

echo "[1/4] Standard backbones x {vanilla, iceberg} on per-class budgets..."
python src/train.py --multirun +experiment=full_grid

echo "[2/4] CG3 on per-class budgets..."
python src/train.py --multirun \
    model=cg3 method=cg3 \
    dataset=cora,citeseer,pubmed \
    label_strategy=per_class \
    label_strategy.budget=1,3,5,10,20

echo "[3/4] Standard backbones x {vanilla, iceberg} on percentage budgets..."
for ds in cora citeseer pubmed; do
    python src/train.py --multirun \
        model=gcn,gat,gin,sage,gt,diff \
        method=vanilla,iceberg \
        dataset=$ds \
        label_strategy=percentage \
        label_strategy.budget=${PCT_BUDGETS[$ds]}
done

echo "[4/4] CG3 on percentage budgets..."
for ds in cora citeseer pubmed; do
    python src/train.py --multirun \
        model=cg3 method=cg3 \
        dataset=$ds \
        label_strategy=percentage \
        label_strategy.budget=${PCT_BUDGETS[$ds]}
done

echo
echo "All sweeps complete. View TensorBoard:"
echo "  tensorboard --logdir runs --host 0.0.0.0 --port 6006"
echo "(in the RunPod UI, expose TCP port 6006 to reach the dashboard)"
