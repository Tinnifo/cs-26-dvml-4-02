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
import logging
import os
import sys
import pandas as pd

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)

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
    import time
    
    
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
            log.warning(f"torch.compile failed ({e}); falling back to eager")
    optimizer = method.build_optimizer(model)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    best_metric = -float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = -1   
    counter = 0
    epoch_log = []

    for epoch in range(1, cfg.method.epochs + 1):
        train_out = method.train_step(model, data, optimizer, epoch)
        val_out = method.validate(model, data)
        epoch_log.append({"epoch": epoch, **train_out, **val_out})

        early = val_out.get("early_stop_metric")
        if early is not None:
            if early > best_metric:
                best_metric = float(early)
                best_epoch = epoch   
                best_state = copy.deepcopy(model.state_dict())
                counter = 0
            else:
                counter += 1

            if cfg.method.use_early_stopping and counter >= cfg.method.patience:
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
    # ---- aggregate losses over training ----
    loss_ce_vals = []
    loss_gen_vals = []
    loss_contrastive_vals = []
    loss_total_vals = []

    for e in epoch_log:
        if "loss_ce" in e and e["loss_ce"] is not None:
            loss_ce_vals.append(e["loss_ce"])
        if "loss_gen" in e and e["loss_gen"] is not None:
            loss_gen_vals.append(e["loss_gen"])
        if "loss_contrastive" in e and e["loss_contrastive"] is not None:
            loss_contrastive_vals.append(e["loss_contrastive"])
        if "loss_total" in e and e["loss_total"] is not None:
            loss_total_vals.append(e["loss_total"])
    
    loss_stats = {
        "loss_ce": float(np.mean(loss_ce_vals)) if loss_ce_vals else None,
        "loss_gen": float(np.mean(loss_gen_vals)) if loss_gen_vals else None,
        "loss_contrastive": float(np.mean(loss_contrastive_vals)) if loss_contrastive_vals else None,
        "loss_total": float(np.mean(loss_total_vals)) if loss_total_vals else None,
    }
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    runtime_sec = time.perf_counter() - start_time
            
    return {
        "metrics": metrics,
        "epoch_log": epoch_log,
        "best_metric": best_metric,
        "best_epoch": best_epoch,   
        "stopped_at_epoch": epoch_log[-1]["epoch"] if epoch_log else 0,
        "loss_stats": loss_stats,
        "runtime_sec": runtime_sec,
    }


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> float:
    log.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg))

    if cfg.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.device)
    log.info(f"device={device}")

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
    all_loss_stats = []
    all_runtimes = []
    best_epochs = []
    every = max(1, int(cfg.epoch_log_every))

    log_dir = run_log_dir(cfg)
    for seed in seeds:
        log.info(f"[seed={seed}] training...")
        ckpt_path = (
            os.path.join(log_dir, f"best_state_seed{seed}.pt")
            if cfg.save_checkpoints else None
        )
        result = run_one_seed(cfg, method, base_data, loaded.in_channels,
                              loaded.num_classes, int(seed), device,
                              checkpoint_path=ckpt_path)
        m = result["metrics"]
        all_metrics.append(m)
        all_loss_stats.append(result["loss_stats"])
        all_runtimes.append(result["runtime_sec"])
        best_epochs.append(result["best_epoch"])
        log.info(
            f"[seed={seed}] stopped@{result['stopped_at_epoch']} "
            f"best@{result['best_epoch']} "
            f"acc={m[0]:.4f} macroF1={m[3]:.4f}"
        )
        
        seed_result = {
            "seed": seed,
            "test_acc": float(m[0]),
            "macro_f1": float(m[3]),
        }
        seed_df = pd.DataFrame([seed_result])
        seed_csv = os.path.join(log_dir, "seed_results.csv")

        os.makedirs(log_dir, exist_ok=True)

        seed_df.to_csv(
            seed_csv,
            mode="a",
            header=not os.path.exists(seed_csv),
            index=False,
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

    loss_ce_mean = np.mean([x["loss_ce"] for x in all_loss_stats if x["loss_ce"] is not None])
    loss_gen_mean = np.mean([x["loss_gen"] for x in all_loss_stats if x["loss_gen"] is not None])
    loss_contrastive_mean = np.mean([x["loss_contrastive"] for x in all_loss_stats if x["loss_contrastive"] is not None])
    loss_total_mean = np.mean([x["loss_total"] for x in all_loss_stats if x["loss_total"] is not None])

    runtime_mean = float(np.mean(all_runtimes))
    runtime_std = float(np.std(all_runtimes))
    
    # ---------------- SAVE CSV HERE ----------------
    model_name = cfg.model.name

    # CG3 internally bundles architectures, so make model distinguishable
    if cfg.method.name == "cg3":
        model_name = f"{cfg.method.local_model}_{cfg.method.global_model}"

    results_data = {
        "model": model_name,
        "method": cfg.method.name,
        "dataset": cfg.dataset.name,
        "budget": cfg.label_strategy.budget,

        "mean_acc": round(float(mean[0]), 4),
        "std_acc": round(float(std[0]), 4),

        "mean_macro_f1": round(float(mean[3]), 4),
        "std_macro_f1": round(float(std[3]), 4),
        
        # losses
        #"l_ce": round(float(loss_ce_mean), 4),
        #"l_gen": round(float(loss_gen_mean), 4),
        #"l_con": round(float(loss_contrastive_mean), 4),
        #"l_total": round(float(loss_total_mean), 4),
        
        # runtime + epoch
        "rt_sec_mean": round(runtime_mean, 4),
        "best_epoch_mean": int(np.mean(best_epochs)) if best_epochs else None,
        "rt_sec_std": round(runtime_std, 4),

        # model hyperparams (safe access)
        "hidden_channels": getattr(cfg.model.arch, "hidden_channels", None),
        "dropout": getattr(cfg.model.arch, "dropout", None),
        
         # training setup
        "epochs": cfg.method.epochs,
        "patience": cfg.method.patience,
        "lr": cfg.method.lr,
        "weight_decay": cfg.method.weight_decay,
        
        # metadata
        "use_early_stopping": cfg.method.use_early_stopping,
        "seeds": str(list(cfg.seeds))
    }

    df_run = pd.DataFrame([results_data])

    os.makedirs(log_dir, exist_ok=True)
    run_csv_path = os.path.join(log_dir, "summary_results.csv")
    df_run.to_csv(
        run_csv_path,
        mode="a",
        header=not os.path.exists(run_csv_path),
        index=False
    )
    log.info(f"Run results saved to {run_csv_path}")

    from hydra.utils import get_original_cwd
    
    
    #master_csv_path = os.path.join(get_original_cwd(), "all_experimentsBaselinesPC.csv")    
    #master_csv_path = os.path.join(get_original_cwd(), "all_experimentsBaselinesPB.csv")    

    master_csv_path = os.path.join(get_original_cwd(), "all_experimentsPerClass.csv")
    #master_csv_path = os.path.join(get_original_cwd(), "all_experimentsCG3Percentage.csv")



    df_run.to_csv(
        master_csv_path,
        mode="a",
        header=not os.path.exists(master_csv_path),
        index=False
    )
    log.info(f"Appended results to {master_csv_path}")
    # ------------------------------------------------

    log.info(
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
                "epochs": int(cfg.method.epochs),
                "patience": int(cfg.method.patience),
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
