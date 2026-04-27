#!/bin/bash
#SBATCH --job-name=gnn-grid
#SBATCH --partition=prioritized
#SBATCH --account=aau
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=15
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=logs/grid_%j.out

# Complete sweep: every (model, method) × every dataset × every budget × 5 seeds.
# Total: ~390 Hydra runs (= ~1950 training trials, since seeds are looped inside
# each run). Estimate ~30s per trial on GPU → ~16h. The 24h Slurm budget leaves
# headroom; trim sections below if you want a faster partial sweep.
#
# Splits into four multiruns because percentage budgets differ per dataset and
# can't be expressed in a single Hydra basic-sweeper expression.

set -e
source .venv/bin/activate

# TensorBoard logging is on by default (conf/config.yaml). All runs write to
# `runs/<dataset>/budget_<X>/<model>_<method>/`; view with `tensorboard --logdir runs`.

# ─────────────────────────────────────────────────────────────────────────────
# 1. Standard backbones × {vanilla, iceberg} — per-class budgets
# ─────────────────────────────────────────────────────────────────────────────
echo "[1/4] Standard backbones × {vanilla, iceberg} on per-class budgets..."
python3 src/train.py --multirun +experiment=full_grid
# ─────────────────────────────────────────────────────────────────────────────
# 2. CG3 — per-class budgets
# ─────────────────────────────────────────────────────────────────────────────
echo "[2/4] CG3 on per-class budgets..."
python3 src/train.py --multirun \
    model=cg3 method=cg3 \
    dataset=cora,citeseer,pubmed \
    label_strategy=per_class \
    label_strategy.budget=1,3,5,10,20 \
   
# ─────────────────────────────────────────────────────────────────────────────
# 3. Standard backbones × {vanilla, iceberg} — percentage budgets (per dataset)
# ─────────────────────────────────────────────────────────────────────────────
echo "[3/4] Standard backbones × {vanilla, iceberg} on percentage budgets..."
declare -A PCT_BUDGETS=(
    [cora]="0.005,0.01,0.02,0.03,0.04"
    [citeseer]="0.005,0.01,0.015,0.02,0.03"
    [pubmed]="0.0005,0.001,0.0015,0.002,0.0025"
)
for ds in cora citeseer pubmed; do
    python3 src/train.py --multirun \
        model=gcn,gat,gin,sage,gt,diff \
        method=vanilla,iceberg \
        dataset=$ds \
        label_strategy=percentage \
        label_strategy.budget=${PCT_BUDGETS[$ds]} \
       done

# ─────────────────────────────────────────────────────────────────────────────
# 4. CG3 — percentage budgets (per dataset)
# ─────────────────────────────────────────────────────────────────────────────
echo "[4/4] CG3 on percentage budgets..."
for ds in cora citeseer pubmed; do
    python3 src/train.py --multirun \
        model=cg3 method=cg3 \
        dataset=$ds \
        label_strategy=percentage \
        label_strategy.budget=${PCT_BUDGETS[$ds]} \
       done

echo "All sweeps complete."
