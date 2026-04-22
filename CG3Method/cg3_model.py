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
            self.gcn1 = GCNConv(in_dim, hidden_dim)
            self.gcn2 = GCNConv(hidden_dim, hidden_dim)

            # ======================
            # HGCN (global view)
            # ======================
            self.hgcn = HGCN(in_dim, hidden_dim, hidden_dim)
            
            # ======================
            # edge decoder (paper MLP)
            # ======================
            self.W_edge = nn.Linear(hidden_dim, hidden_dim, bias=False)

            # FIXED paper fusion
            self.alpha = 0.6

            # classifier
            self.classifier = nn.Linear(hidden_dim, num_classes)

    # ---------------------
    # GCN encoder
    # ---------------------
    def encode_gcn(self, x, edge_index, edge_weight):
        x = F.relu(self.gcn1(x, edge_index, edge_weight))
        x = self.gcn2(x, edge_index, edge_weight)
        return F.normalize(x, dim=1)

    # ---------------------
    # forward
    # ---------------------
    def forward(self, x, edge_index, edge_weight):

        z_gcn = self.encode_gcn(x, edge_index, edge_weight)

        z_hgcn = self.hgcn(x, edge_index, edge_weight)

        # PAPER EXACT FUSION
        z = F.normalize(
            self.alpha * z_gcn + (1 - self.alpha) * z_hgcn,
            dim=1
        )

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

        pos = sim.diag()
        denom = sim.sum(dim=1) - pos + 1e-8   # exclude self

        unsup_1 = pos / denom


        sim_rev = torch.exp(torch.matmul(z_hgcn, z_gcn.t()) / tau)

        pos_rev = sim_rev.diag()
        denom_rev = sim_rev.sum(dim=1) - pos_rev + 1e-8

        unsup_2 = pos_rev / denom_rev

        unsup_loss = -hp1 * torch.log(
            torch.cat([unsup_1, unsup_2]) + 1e-8
        ).mean()


        # ======================
        # SUPERVISED (paper version)
        # ======================

        h1 = z_gcn[train_idx]
        h2 = z_hgcn[train_idx]

        sim = torch.exp(torch.matmul(h1, h2.t()) / tau)

        # positive and negative separation
        pos_sum = (sim * pos_mask).sum(dim=1)
        neg_sum = (sim * neg_mask).sum(dim=1)

        # IMPORTANT: match TF normalization
        N = pos_mask.size(1)

        pos_mean = pos_sum / (pos_mask.sum(dim=1) + 1e-8)
        neg_mean = (neg_sum + pos_sum) / (N - 1 + 1e-8)

        sup_1 = pos_mean / (neg_mean + 1e-8)


        # reverse direction (HGCN → GCN)
        sim_rev = torch.exp(torch.matmul(h2, h1.t()) / tau)

        pos_sum_rev = (sim_rev * pos_mask).sum(dim=1)
        neg_sum_rev = (sim_rev * neg_mask).sum(dim=1)

        pos_mean_rev = pos_sum_rev / (pos_mask.sum(dim=1) + 1e-8)
        neg_mean_rev = (neg_sum_rev + pos_sum_rev) / (N - 1 + 1e-8)

        sup_2 = pos_mean_rev / (neg_mean_rev + 1e-8)

        sup_loss = -hp1 * torch.log(torch.cat([sup_1, sup_2]) + 1e-8).mean()
        
        return unsup_loss + sup_loss
    
    # ======================
    # EDGE GENERATIVE LOSS (PAPER MATCHED)
    # ======================
    def edge_loss(self, z_gcn, z_hgcn, edge_index):

        i, j = edge_index

        # ======================
        # POSITIVE EDGES ONLY (paper exact)
        # ======================
        ei_gcn = z_gcn[i]
        ej_hgcn = z_hgcn[j]

        ei_hgcn = z_hgcn[i]
        ej_gcn = z_gcn[j]

        p1 = (ei_gcn * self.W_edge(ej_hgcn)).sum(dim=1)
        p2 = (ei_hgcn * self.W_edge(ej_gcn)).sum(dim=1)

        loss_1 = -torch.log(torch.sigmoid(p1) + 1e-8).mean()
        loss_2 = -torch.log(torch.sigmoid(p2) + 1e-8).mean()

        return loss_1 + loss_2
    
    def compute_loss(self, z_gcn, z_hgcn, z, logits, data, train_idx, pos_mask, neg_mask):
        
        # ======================
        # 1. Classification loss
        # ======================
        loss_cls = F.cross_entropy(
            logits[data.train_mask],
            data.y[data.train_mask]
        )

        # ======================
        # 2. Contrastive loss
        # ======================
        loss_cl = self.contrastive_loss(
            z_gcn, z_hgcn,
            train_idx, pos_mask, neg_mask
        )

        # ======================
        # 3. Edge loss
        # ======================
        loss_edge = self.edge_loss(
            z_gcn, z_hgcn,
            data.edge_index
        )

        # ======================
        # 4. HGCN smoothness
        # ======================
        i, j = data.edge_index
        hgcn_smooth = ((z_hgcn[i] - z_hgcn[j])**2).sum(dim=1).mean()

        # ======================
        # 5. Combine
        # ======================
        loss = (
            loss_cls
            + loss_cl
            + 0.4 * loss_edge
            + 0.1 * hgcn_smooth
        )

        return loss