# GNN Benchmarking on Planetoid Datasets

Hydra-driven pipeline for benchmarking GNN backbones (GCN, GAT, GIN, SAGE, GT, Diff) under two label-budget regimes (per-class N, global %) on Cora / CiteSeer / PubMed, with three training methods plug-in: `vanilla` (CE), `iceberg` (debiased self-training: pseudo-label + balanced softmax), `cg3` (contrastive graph-to-graph multi-task). All metrics log to TensorBoard.

## 1. Setup

Python 3.10+ recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. How experiments are configured

Hydra composes one experiment from four config groups under `conf/`:

| Group | Files | Purpose |
|---|---|---|
| `model/` | `gcn`, `gat`, `gin`, `sage`, `gt`, `diff`, `cg3` | Backbone architecture |
| `method/` | `vanilla`, `iceberg`, `cg3` | Training recipe (loss, optional pseudo-labeling) |
| `dataset/` | `cora`, `citeseer`, `pubmed` | Planetoid loader + dataset-specific percentage budgets |
| `label_strategy/` | `per_class`, `percentage` | How the labeled set is sampled (uses `src/data/labels.py`) |

The top-level `conf/config.yaml` sets shared knobs (`epochs=200`, `patience=50`, `seeds=[0..4]`, TensorBoard log dir).

## 3. Running

### Single experiment
```bash
# vanilla GCN on Cora, 20 labels per class
python src/train.py model=gcn method=vanilla dataset=cora label_strategy=per_class label_strategy.budget=20

# GCN with the IceBerg trick — clean A/B test against the line above
python src/train.py model=gcn method=iceberg dataset=cora label_strategy=per_class label_strategy.budget=20

# Diff backbone + IceBerg (the paper's headline recipe)
python src/train.py model=diff method=iceberg dataset=cora label_strategy.budget=20

# CG3 — model knob ignored (the method bundles its own architecture)
python src/train.py model=cg3 method=cg3 dataset=cora label_strategy.budget=20

# Percentage label strategy
python src/train.py model=gcn method=iceberg dataset=cora label_strategy=percentage label_strategy.budget=0.01

# Disable TB logging if you just want stdout
python src/train.py model=gcn method=vanilla dataset=cora tensorboard.enable=false
```

### Sweeps (Hydra `--multirun`)
```bash
# All standard backbones × {vanilla, iceberg} on Cora @ budgets {1,3,5,10,20}
python src/train.py --multirun \
    model=gcn,gat,gin,sage,gt,diff method=vanilla,iceberg \
    dataset=cora label_strategy=per_class label_strategy.budget=1,3,5,10,20

# Use the bundled experiment recipe for the full grid
python src/train.py --multirun +experiment=full_grid
```

### HPC (Slurm)
```bash
sbatch sh/run.sh
```

## 4. Adding a new model or method

**New model**: drop a `src/models/foo.py` that subclasses `BaseGNN` (`forward(x, edge_index) → logits`), re-export it in `src/models/__init__.py`, and add `conf/model/foo.yaml` with `_target_: src.models.foo.Foo`. It now works with `vanilla` and `iceberg`.

**New method**: drop a `src/methods/foo.py` that subclasses `BaseMethod` (implementing `build_model` and `train_step`; optionally `prepare`, `validate`, `predict_logits`). Add it to `METHOD_REGISTRY` in `src/train.py` and write `conf/method/foo.yaml` with at least `name: foo`.

## 5. Project structure

```
conf/                     # Hydra config groups (model, method, dataset, label_strategy, experiment)
src/
├── train.py              # Hydra entry point — single run or `--multirun` sweep
├── models/               # BaseGNN + GCN, GAT, GIN, SAGE, GT, Diff
├── methods/
│   ├── base.py           # BaseMethod (build_model, prepare, train_step, evaluate, validate)
│   ├── vanilla.py
│   ├── iceberg.py
│   ├── cg3.py            # wraps the bundled CG3 code below
│   └── _cg3/             # CG3's bundled architecture (CG3Model, HGCN, hierarchy build)
└── data/
    ├── loader.py         # Planetoid loading + label-strategy dispatch
    └── labels.py         # set_few_label_mask, set_budget_percent, set_seed
sh/run.sh                 # Slurm wrapper around the Hydra multirun
data/                     # Auto-created by PyG (Planetoid downloads)
outputs/, multirun/       # Auto-created by Hydra (single-run / sweep working dirs)
runs/                     # TensorBoard event files
logs/                     # Slurm stdout/stderr
```

## 6. TensorBoard output

Each Hydra run = one `(model, method, dataset, label_strategy, budget)` config evaluated across all seeds. Each run gets its own log directory:

```
runs/<dataset>/budget_<X>/<model>_<method>/
```

Inside that directory:
- TB scalars — per-epoch curves (downsampled to every `epoch_log_every` epochs) under `seed_{seed}/{train_loss, val_acc, val_loss, ...}`; per-seed test metrics under `seed_{seed}/{test_accuracy, test_macro_precision, test_macro_recall, test_macro_f1, test_micro_f1, best_early_stop_metric}`; aggregate across seeds under `agg/{mean_accuracy, mean_macro_f1, std_accuracy, std_macro_f1, moe_accuracy, moe_macro_f1}`
- HParams entry (model, method, dataset, budget vs. final metrics) for the **HParams** tab
- `best_state_seed{seed}.pt` — best-val checkpoint per seed (toggle via `save_checkpoints` in `conf/config.yaml`). Reload with `model.load_state_dict(torch.load(...))`

View:
```bash
tensorboard --logdir runs
```

In the TB UI, the directory layout means runs for the same `(dataset, budget)` are siblings, so you can multi-select them to overlay curves for the (model, method) ablation.
