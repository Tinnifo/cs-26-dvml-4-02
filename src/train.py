"""Hydra entry point: trains one (model, method, dataset, label_strategy, budget)
configuration over a list of seeds and reports aggregated test metrics.

Single run:
  python src/train.py model=gcn method=iceberg dataset=cora label_strategy=per_class label_strategy.budget=20

Sweep (full grid):
  python src/train.py --multirun model=gcn,gat,gin,sage,gt,diff method=vanilla,iceberg \
                       dataset=cora,citeseer,pubmed label_strategy=per_class \
                       label_strategy.budget=1,3,5,10,20

CG3 (model knob ignored — its method bundles its own architecture):
  python src/train.py method=cg3 dataset=cora label_strategy=per_class label_strategy.budget=20

Logging: TensorBoard. Each Hydra run writes to
  <tensorboard.log_dir>/<dataset>/budget_<X>/<model>_<method>/
View with `tensorboard --logdir runs/`.
"""

from __future__ import annotations

import copy
import os
import sys

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

# Ensure project root is on the path so `src/` resolves.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.data.loader import apply_label_strategy, format_budget, load_dataset
from src.methods import CG3Method, IcebergMethod, VanillaMethod
from src.methods.base import BaseMethod

METHOD_REGISTRY = {
    "vanilla": VanillaMethod,
    "iceberg": IcebergMethod,
    "cg3": CG3Method,
}


def build_method(cfg: DictConfig) -> BaseMethod:
    name = cfg.method.name
    if name not in METHOD_REGISTRY:
        raise ValueError(f"Unknown method '{name}'. Add it to METHOD_REGISTRY in src/train.py.")
    return METHOD_REGISTRY[name](cfg)


def run_log_dir(cfg: DictConfig) -> str:
    """`runs/<dataset>/budget_<X>/<model>_<method>/` — shared by TensorBoard
    and the best-state checkpoint files so a downloaded run dir is
    self-contained."""
    return os.path.join(
        cfg.tensorboard.log_dir,
        cfg.dataset.name,
        f"budget_{format_budget(cfg.label_strategy.budget)}",
        f"{cfg.model.name}_{cfg.method.name}",
    )


def init_tensorboard(cfg: DictConfig):
    """One log dir per (dataset, budget, model, method) — same granularity as
    a single training run. All seeds for that config write into the same dir
    under `seed_{seed}/...` tags so they show as separate curves in TB."""
    from torch.utils.tensorboard import SummaryWriter

    return SummaryWriter(log_dir=run_log_dir(cfg))


def run_one_seed(cfg: DictConfig, method: BaseMethod, base_data, in_channels: int,
                 num_classes: int, seed: int, device: torch.device,
                 checkpoint_path: str | None = None):
    from src.data.labels import set_seed
    set_seed(seed)
    data = base_data.clone().to(device)
    data = apply_label_strategy(data, cfg.label_strategy.name, cfg.label_strategy.budget, seed)

    model = method.build_model(in_channels, num_classes, data=data).to(device)
    data = method.prepare(model, data)
    if cfg.compile_model:
        # `prepare` may swap params (e.g. CG3 replaces model.hgcn) — compile after.
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"  [warn] torch.compile failed ({e}); falling back to eager")
    optimizer = method.build_optimizer(model)

    best_metric = -float("inf")
    best_state = None
    counter = 0
    epoch_log = []

    for epoch in range(1, cfg.epochs + 1):
        train_out = method.train_step(model, data, optimizer, epoch)
        val_out = method.validate(model, data)
        epoch_log.append({"epoch": epoch, **train_out, **val_out})

        early = val_out.get("early_stop_metric")
        if early is not None:
            if early > best_metric:
                best_metric = float(early)
                best_state = copy.deepcopy(model.state_dict())
                counter = 0
            else:
                counter += 1
                if counter >= cfg.patience:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
        if checkpoint_path is not None:
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            # Strip `_orig_mod.` prefix introduced by torch.compile so the
            # checkpoint loads cleanly into either a compiled OR an eager model.
            clean_state = {k.removeprefix("_orig_mod."): v for k, v in best_state.items()}
            torch.save(clean_state, checkpoint_path)

    metrics = method.evaluate(model, data)
    return {
        "metrics": metrics,
        "epoch_log": epoch_log,
        "best_metric": best_metric,
        "stopped_at_epoch": epoch_log[-1]["epoch"] if epoch_log else 0,
    }


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> float:
    print(OmegaConf.to_yaml(cfg))

    if cfg.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.device)
    print(f"[train] device={device}")

    loaded = load_dataset(
        cfg.dataset.name,
        root=cfg.data_root,
        normalize_features=cfg.dataset.normalize_features,
    )
    base_data = loaded.data
    method = build_method(cfg)

    writer = None
    if cfg.tensorboard.enable:
        writer = init_tensorboard(cfg)

    seeds = list(cfg.seeds)
    all_metrics = []
    every = max(1, int(cfg.epoch_log_every))

    log_dir = run_log_dir(cfg)
    for seed in seeds:
        print(f"  [seed={seed}] training...")
        ckpt_path = (
            os.path.join(log_dir, f"best_state_seed{seed}.pt")
            if cfg.save_checkpoints else None
        )
        result = run_one_seed(cfg, method, base_data, loaded.in_channels,
                              loaded.num_classes, int(seed), device,
                              checkpoint_path=ckpt_path)
        m = result["metrics"]
        all_metrics.append(m)
        print(
            f"  [seed={seed}] stopped@{result['stopped_at_epoch']} "
            f"acc={m[0]:.4f} macroF1={m[3]:.4f}"
        )

        if writer is not None:
            for entry in result["epoch_log"]:
                if entry["epoch"] % every != 0 and entry["epoch"] != result["stopped_at_epoch"]:
                    continue
                step = int(entry["epoch"])
                for k, v in entry.items():
                    if k == "epoch":
                        continue
                    if isinstance(v, (int, float)):
                        writer.add_scalar(f"seed_{seed}/{k}", float(v), step)
            # Per-seed test metrics — single point each, step=0.
            writer.add_scalar(f"seed_{seed}/test_accuracy", float(m[0]), 0)
            writer.add_scalar(f"seed_{seed}/test_macro_precision", float(m[1]), 0)
            writer.add_scalar(f"seed_{seed}/test_macro_recall", float(m[2]), 0)
            writer.add_scalar(f"seed_{seed}/test_macro_f1", float(m[3]), 0)
            writer.add_scalar(f"seed_{seed}/test_micro_f1", float(m[4]), 0)
            writer.add_scalar(f"seed_{seed}/best_early_stop_metric", float(result["best_metric"]), 0)

    arr = np.array(all_metrics)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    n = len(seeds)
    moe_acc = 1.96 * std[0] / np.sqrt(n)
    moe_f1 = 1.96 * std[3] / np.sqrt(n)

    print(
        f"[summary {cfg.model.name}/{cfg.method.name} {cfg.dataset.name} "
        f"b={format_budget(cfg.label_strategy.budget)}] "
        f"acc={mean[0]:.4f}+-{moe_acc:.4f}  macroF1={mean[3]:.4f}+-{moe_f1:.4f}"
    )

    if writer is not None:
        # Aggregate across seeds.
        writer.add_scalar("agg/mean_accuracy", float(mean[0]), 0)
        writer.add_scalar("agg/mean_macro_f1", float(mean[3]), 0)
        writer.add_scalar("agg/std_accuracy", float(std[0]), 0)
        writer.add_scalar("agg/std_macro_f1", float(std[3]), 0)
        writer.add_scalar("agg/moe_accuracy", float(moe_acc), 0)
        writer.add_scalar("agg/moe_macro_f1", float(moe_f1), 0)

        # HParams entry — enables TB's HParams tab for cross-run comparison
        # (model/method/dataset/budget vs final metrics).
        writer.add_hparams(
            {
                "model": str(cfg.model.name),
                "method": str(cfg.method.name),
                "dataset": str(cfg.dataset.name),
                "label_strategy": str(cfg.label_strategy.name),
                "budget": float(cfg.label_strategy.budget),
                "epochs": int(cfg.epochs),
                "patience": int(cfg.patience),
                "seeds": str(list(cfg.seeds)),
            },
            {
                "hparam/mean_accuracy": float(mean[0]),
                "hparam/mean_macro_f1": float(mean[3]),
                "hparam/moe_accuracy": float(moe_acc),
            },
            run_name=".",
        )
        writer.flush()
        writer.close()

    return float(mean[0])


if __name__ == "__main__":
    main()
