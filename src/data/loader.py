"""Dataset loading + label-strategy dispatch.

Wraps the helpers in `src/data/labels.py` (`set_few_label_mask`,
`set_budget_percent`) so per-class N or percentage budgets are sampled from
the original Planetoid train mask, with val/test masks left untouched.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid

from src.data.labels import set_budget_percent, set_few_label_mask, set_seed


@dataclass
class LoadedDataset:
    data: object
    name: str
    in_channels: int
    num_classes: int


def load_dataset(name: str, root: str = "data", normalize_features: bool = False) -> LoadedDataset:
    transform = T.NormalizeFeatures() if normalize_features else None
    dataset = Planetoid(root=root, name=name, transform=transform)
    data = dataset[0]
    return LoadedDataset(
        data=data,
        name=name,
        in_channels=int(dataset.num_features),
        num_classes=int(dataset.num_classes),
    )


def apply_label_strategy(data, strategy: str, budget, seed: int):
    """Build the train mask using cs-26's helpers. `strategy` is `per_class`
    (budget is N labels per class, int) or `percentage` (budget is fraction
    of all nodes, float). Val/test masks are untouched."""
    set_seed(seed)
    if strategy == "per_class":
        return set_few_label_mask(data, int(budget), seed)
    if strategy == "percentage":
        return set_budget_percent(data, float(budget), seed)
    raise ValueError(f"Unknown label strategy: {strategy}")


def format_budget(budget) -> str:
    """Match the budget-string formatting from `src/run_experiments.py` so
    W&B group / run names line up with the original cs-26 runs."""
    b = float(budget)
    if b >= 1:
        return f"{int(b)}"
    return f"{b * 100:.3f}%" if b < 0.001 else f"{b * 100:.2f}%"
