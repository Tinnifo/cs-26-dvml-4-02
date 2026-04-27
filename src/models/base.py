"""Protocol every standard GNN backbone in `src/models/` must follow.

The "standard" methods (`vanilla`, `iceberg`) call the model with
`forward(x, edge_index) -> logits` and call `prepare(data)` once before
training so backbones can do any one-shot precomputation (e.g. Diff's
feature propagation) by returning a modified data object.

CG3 is intentionally outside this protocol — its method (`methods/cg3.py`)
owns the model lifecycle directly because the architecture and loss are
inseparable from each other.
"""

from __future__ import annotations

import abc

import torch
import torch.nn as nn


class BaseGNN(nn.Module, abc.ABC):
    @abc.abstractmethod
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Return class logits with shape `[num_nodes, num_classes]`."""

    def prepare(self, data):
        """One-shot pre-training hook. Default: identity.

        Override when the backbone needs to swap in precomputed inputs
        (e.g. Diff replaces `data.x` with the propagated features so its
        `forward(x, edge_index)` call site stays unchanged).
        """
        return data
