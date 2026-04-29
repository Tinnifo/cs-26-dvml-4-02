import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from hgcn import HGCN_PyG as HGCN


class CG3Model(nn.Module):    
    def __init__(self, in_dim, hidden_dim, num_classes):
            super().__init__()

            # ======================
            # GCN (local view)
            # ======================
            self.gcn1 = GCNConv(in_dim, hidden_dim, normalize=False)
            self.gcn2 = GCNConv(hidden_dim, hidden_dim, normalize=False)

            # ======================
            # HGCN (global view)
            # ======================
            self.hgcn = HGCN(in_dim, hidden_dim, hidden_dim)
            
            # ======================
            # edge decoder (paper MLP)
            # ======================
            self.W_edge = nn.Linear(hidden_dim, hidden_dim, bias=False)

            # FIXED paper fusion
            self.alpha = nn.Parameter(torch.tensor(0.8))

            # classifier
            self.classifier = nn.Linear(hidden_dim, num_classes)

    # ---------------------
    # GCN encoder
    # ---------------------
    def encode_gcn(self, x, edge_index, edge_weight):
        x = F.relu(self.gcn1(x, edge_index, edge_weight))
        x = F.dropout(x, p=0.5, training=self.training)  # <-- ADD THIS
        x = self.gcn2(x, edge_index, edge_weight)
        return x

    # ---------------------
    # forward
    # ---------------------
    def forward(self, x, edge_index, edge_weight):

        z_gcn = self.encode_gcn(x, edge_index, edge_weight)
        z_hgcn = self.hgcn(x, edge_index, edge_weight)

        z_gcn = F.dropout(z_gcn, p=0.5, training=self.training)
        z_hgcn = F.dropout(z_hgcn, p=0.5, training=self.training)

        z_gcn = F.normalize(z_gcn, dim=1)
        z_hgcn = F.normalize(z_hgcn, dim=1)

        z = self.alpha * z_gcn + (1 - self.alpha) * z_hgcn
        z = F.normalize(z, dim=1)

        logits = self.classifier(z)

        return z_gcn, z_hgcn, z, logits
    
    def contrastive_loss(self, z_gcn, z_hgcn, train_idx, pos_mask, neg_mask, tau=0.5, hp1=0.9):
        """
        z_gcn   : [N, C]
        z_hgcn  : [N, C]
        train_idx : indices of labeled nodes
        pos_mask : [L, L] same-class mask
        neg_mask : [L, L] different-class mask
        """
     
    

        # ======================
        # UNSUPERVISED (paper exact)
        # ======================

        sim = torch.exp(torch.matmul(z_gcn, z_hgcn.t()) / tau)

        neg = sim.mean(dim=1)
        pos = sim.diag()

        unsup_1 = pos / (neg + 1e-8)


        sim_rev = torch.exp(torch.matmul(z_hgcn, z_gcn.t()) / tau)

        pos_rev = sim_rev.diag()
        denom_rev = sim_rev.sum(dim=1) - pos_rev + 1e-8

        unsup_2 = pos_rev / denom_rev

        unsup_loss = -hp1 * (
            torch.log(unsup_1 + 1e-8).mean() +
            torch.log(unsup_2 + 1e-8).mean()
        )


        # ======================
        # SUPERVISED (paper version)
        # ======================

        h1 = z_gcn[train_idx]
        h2 = z_hgcn[train_idx]
        h1 = F.normalize(h1, dim=1)
        h2 = F.normalize(h2, dim=1)
        
        sim = torch.matmul(h1, h2.t()) / tau


        # positive and negative separation
        pos_sum = (sim * pos_mask).sum(dim=1)
        neg_sum = (sim * neg_mask).sum(dim=1)

        # IMPORTANT: match TF normalization
        N = pos_mask.size(1)

        pos_mean = pos_sum / (pos_mask.sum(dim=1) + 1e-8)
        neg_mean = (neg_sum + pos_sum) / (N - 1 + 1e-8)

        sup_1 = pos_mean / (neg_mean + 1e-8)


        # reverse direction (HGCN → GCN)
        sim_rev = torch.matmul(h2, h1.t()) / tau

        pos_sum_rev = (sim_rev * pos_mask).sum(dim=1)
        neg_sum_rev = (sim_rev * neg_mask).sum(dim=1)

        pos_mean_rev = pos_sum_rev / (pos_mask.sum(dim=1) + 1e-8)
        neg_mean_rev = (neg_sum_rev + pos_sum_rev) / (N - 1 + 1e-8)

        sup_2 = pos_mean_rev / (neg_mean_rev + 1e-8)

        sup_loss = -hp1 * (
            torch.log(sup_1 + 1e-8).mean() +
            torch.log(sup_2 + 1e-8).mean()
        )
        
        return unsup_loss + sup_loss
    
    # ======================
    # EDGE GENERATIVE LOSS (PAPER MATCHED)
    # ======================
    def edge_loss(self, z_gcn, z_hgcn, edge_index):

        i, j = edge_index
        num_nodes = z_gcn.size(0)

        # -----------------------
        # POSITIVE
        # -----------------------
        pos_score_1 = (z_gcn[i] * self.W_edge(z_hgcn[j])).sum(dim=1)
        pos_score_2 = (z_hgcn[i] * self.W_edge(z_gcn[j])).sum(dim=1)

        pos_loss = (
            -torch.log(torch.sigmoid(pos_score_1) + 1e-8).mean()
            -torch.log(torch.sigmoid(pos_score_2) + 1e-8).mean()
        )

        # -----------------------
        # NEGATIVE SAMPLING
        # -----------------------
        neg_j = torch.randint(0, num_nodes, j.size(), device=j.device)

        # avoid sampling true edges
        mask = (neg_j == j)
        while mask.any():
            neg_j[mask] = torch.randint(0, num_nodes, (mask.sum(),), device=j.device)
            mask = (neg_j == j)

        neg_score_1 = (z_gcn[i] * self.W_edge(z_hgcn[neg_j])).sum(dim=1)
        neg_score_2 = (z_hgcn[i] * self.W_edge(z_gcn[neg_j])).sum(dim=1)

        neg_loss = (
            -torch.log(1 - torch.sigmoid(neg_score_1) + 1e-8).mean()
            -torch.log(1 - torch.sigmoid(neg_score_2) + 1e-8).mean()
        )

        return pos_loss + neg_loss
        
    def compute_loss(self, z_gcn, z_hgcn, z, logits, data,
                    train_idx, pos_mask, neg_mask,
                    mode="full"):

        loss_cls = F.cross_entropy(
            logits[data.train_mask],
            data.y[data.train_mask]
        )

        loss_cl = self.contrastive_loss(
            z_gcn, z_hgcn,
            train_idx, pos_mask, neg_mask
        )

        loss_edge = self.edge_loss(
            z_gcn, z_hgcn,
            data.edge_index
        )

        i, j = data.edge_index
        hgcn_smooth = ((z_hgcn[i] - z_hgcn[j])**2).sum(dim=1).mean()

        # ======================
        # CONTROL MODES
        # ======================
        if mode == "cls":
            return loss_cls

        elif mode == "cls+cl":
            return loss_cls + 0.05 * loss_cl

        elif mode == "cls+cl+edge":
            return loss_cls + 0.05 * loss_cl + 0.05 * loss_edge

        elif mode == "full":
            return (
                loss_cls
                + 0.25 * loss_cl
                + 0.05 * loss_edge
            )

        else:
            raise ValueError("Unknown loss mode")