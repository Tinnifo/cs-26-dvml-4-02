import torch
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv

from src.models.base import BaseGNN


class GT(BaseGNN):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 heads=4, dropout=0.5):
        super().__init__()
        self.conv1 = TransformerConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.conv2 = TransformerConv(hidden_channels * heads, out_channels,
                                     heads=1, concat=False, dropout=dropout)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)