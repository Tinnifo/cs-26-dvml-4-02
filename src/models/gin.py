import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv

from src.models.base import BaseGNN


class GIN(BaseGNN):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float = 0.5):
        super().__init__()
        nn1 = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        nn2 = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels),
        )
        self.conv1 = GINConv(nn1)
        self.conv2 = GINConv(nn2)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)
