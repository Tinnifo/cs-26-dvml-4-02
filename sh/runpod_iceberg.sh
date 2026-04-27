#!/bin/bash
# IceBerg-only sweep for a RunPod-style single-A100 SXM instance.
# Drops vanilla and CG3 from sh/runpod.sh — keeps only `method=iceberg` runs
# across the 6 backbones, 3 datasets, and both label-budget regimes.
#
# Total: 6 backbones x 1 method x 3 datasets x 10 budgets = 180 Hydra runs
#        (= 900 training trials with 5 seeds).
# Estimated runtime on an A100 SXM: ~1.5-2 hours.
#
# Target hardware:
#   GPU            A100 SXM 1x
#   vCPU           32 (AMD EPYC 7763)
#   Memory         250 GB
#   Container disk 20 GB

set -e

cd "$(dirname "$0")/.."  # run from repo root regardless of cwd

# ─────────────────────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=0
# Per-process thread pools sized for n_jobs=8 parallel Hydra processes on a
# 32-vCPU box (8 * 4 = 32). See conf/config.yaml hydra.launcher.n_jobs.
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_MAX_THREADS=8

if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
else
    echo "No .venv found — installing requirements into container Python..."
    pip install -q --upgrade pip
    pip install -q -r requirements.txt
fi

python - <<'PY'
import torch
print(f"[runpod-iceberg] torch={torch.__version__}  cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[runpod-iceberg] device={torch.cuda.get_device_name(0)}")
PY

# ─────────────────────────────────────────────────────────────────────────────
# Sweeps — IceBerg only.
# ─────────────────────────────────────────────────────────────────────────────
declare -A PCT_BUDGETS=(
    [cora]="0.005,0.01,0.02,0.03,0.04"
    [citeseer]="0.005,0.01,0.015,0.02,0.03"
    [pubmed]="0.0005,0.001,0.0015,0.002,0.0025"
)

echo "[1/2] IceBerg on per-class budgets..."
python src/train.py --multirun \
    model=gcn,gat,gin,sage,gt,diff \
    method=iceberg \
    dataset=cora,citeseer,pubmed \
    label_strategy=per_class \
    label_strategy.budget=1,3,5,10,20

echo "[2/2] IceBerg on percentage budgets..."
for ds in cora citeseer pubmed; do
    python src/train.py --multirun \
        model=gcn,gat,gin,sage,gt,diff \
        method=iceberg \
        dataset=$ds \
        label_strategy=percentage \
        label_strategy.budget=${PCT_BUDGETS[$ds]}
done

echo
echo "All IceBerg sweeps complete. View TensorBoard:"
echo "  tensorboard --logdir runs --host 0.0.0.0 --port 6006"
echo "(in the RunPod UI, expose TCP port 6006 to reach the dashboard)"
