#!/bin/bash
# RunPod setup + sweep launcher for a single-A100 SXM instance (no Slurm).
#
# Usage:
#   sh/runpod.sh                       # run the hardcoded full sweep (default)
#   sh/runpod.sh full_grid             # run a single experiment yaml
#   sh/runpod.sh full_grid gnn_full_sweep   # run multiple experiment yamls
#   sh/runpod.sh -- model=gcn dataset=cora  # raw Hydra overrides after `--`
#
# Experiment names refer to files in conf/experiment/*.yaml (without the
# extension). They are passed as `+experiment=<name>` to src/train.py.
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
# torch>=2.6 flipped torch.load default to weights_only=True, which rejects
# PyG's pickled Planetoid cache. Force the pre-2.6 behaviour so dataset loads
# don't fail on the first run.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
# 32 vCPU box. Hydra's joblib launcher runs n_jobs=8 training processes in
# parallel (see conf/config.yaml), so per-process thread pools must stay small
# enough that 8 * threads <= 32 vCPUs — otherwise OMP context switching eats
# the parallelism benefit.
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_MAX_THREADS=8

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
# Dispatch
#   - no args  → run the hardcoded full sweep below
#   - args     → treat each arg as an experiment yaml name (conf/experiment/<name>.yaml)
#                or, after `--`, as raw Hydra overrides forwarded to a single multirun
# ─────────────────────────────────────────────────────────────────────────────
if [ "$#" -gt 0 ]; then
    # Split args into experiment names and (optional) raw overrides after `--`.
    experiments=()
    extra_overrides=()
    seen_sep=0
    for arg in "$@"; do
        if [ "$arg" = "--" ]; then
            seen_sep=1
            continue
        fi
        if [ "$seen_sep" -eq 0 ]; then
            experiments+=("$arg")
        else
            extra_overrides+=("$arg")
        fi
    done

    if [ "${#experiments[@]}" -eq 0 ] && [ "${#extra_overrides[@]}" -gt 0 ]; then
        echo "[runpod] raw multirun: ${extra_overrides[*]}"
        python src/train.py --multirun "${extra_overrides[@]}"
    else
        n=${#experiments[@]}
        i=1
        for exp in "${experiments[@]}"; do
            yaml="conf/experiment/${exp}.yaml"
            if [ ! -f "$yaml" ]; then
                echo "[runpod] error: experiment yaml not found: $yaml" >&2
                echo "Available experiments:" >&2
                ls conf/experiment/ >&2
                exit 1
            fi
            echo "[$i/$n] +experiment=$exp ${extra_overrides[*]}"
            python src/train.py --multirun "+experiment=$exp" "${extra_overrides[@]}"
            i=$((i + 1))
        done
    fi

    echo
    echo "Done. View TensorBoard:"
    echo "  tensorboard --logdir runs --host 0.0.0.0 --port 6006"
    exit 0
fi

# ─────────────────────────────────────────────────────────────────────────────
# Default: hardcoded full sweep (same four sections as sh/run.sh, no Slurm).
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
