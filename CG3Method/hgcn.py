import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class HGCN_PyG(nn.Module):

    def __init__(self, in_dim, hidden_dim, num_levels=3):
        super().__init__()

        self.num_levels = num_levels

        # encoder / decoder
        self.enc_gcns = nn.ModuleList()
        self.dec_gcns = nn.ModuleList()
        # 3 layers
        for i in range(num_levels):
            self.enc_gcns.append(
                GCNConv(in_dim if i == 0 else hidden_dim, hidden_dim)
            )

            self.dec_gcns.append(
                GCNConv(hidden_dim, hidden_dim)
            )

        # bottleneck
        self.bottleneck = GCNConv(hidden_dim, hidden_dim)

        # hierarchy buffers
        self.edge_levels = None
        self.C_matrices = None

    def set_hierarchy(self, edge_levels, C_matrices):
        self.edge_levels = edge_levels
        self.C_matrices = C_matrices

    def forward(self, x, edge_index, edge_weight=None):

        if self.edge_levels is None:
            raise ValueError("Call set_hierarchy first")

        enc_feats = []
        h = x

        # ======================
        # ENCODER (coarsening)
        # ======================
        for l in range(self.num_levels):

            ei = self.edge_levels[l]

            h = self.enc_gcns[l](
                h, ei, edge_weight=edge_weight
            )
            h = F.relu(h)

            enc_feats.append(h)

            # coarsen
            if l < self.num_levels - 1:
                C = self.C_matrices[l].coalesce().to(h.device)
                h = torch.sparse.mm(C.t(), h)

        # ======================
        # BOTTLENECK
        # ======================
        h = self.bottleneck(
            h,
            self.edge_levels[-1],
            edge_weight=edge_weight
        )
        h = F.relu(h)

        # ======================
        # DECODER (refinement)
        # ======================
        for l in reversed(range(self.num_levels)):

            ei = self.edge_levels[l]

            if l > 0:
                C = self.C_matrices[l - 1].to(h.device)
                h = torch.sparse.mm(C, h)
                h = h + enc_feats[l - 1]

            h = self.dec_gcns[l](
                h, ei, edge_weight=edge_weight
            )
            h = F.relu(h)

        return F.normalize(h, dim=1)