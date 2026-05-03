import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

from src.models.base import BaseGNN


class GAT(BaseGNN):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 heads=8, dropout=0.6):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * heads, out_channels,
                             heads=1, concat=False, dropout=dropout)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)  # input dropout
        x = F.elu(self.conv1(x, edge_index))                      # ELU (paper correct)
        x = F.dropout(x, p=self.dropout, training=self.training)  # hidden dropout
        return self.conv2(x, edge_index)
