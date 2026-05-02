import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from CG3Layer import GraphConvolution, MLP


def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    log_probs = F.log_softmax(preds, dim=1)
    loss = -(labels * log_probs).sum(dim=1)
    mask = mask.float()
    mask = mask / mask.mean()
    loss = loss * mask
    return loss.mean()


def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    correct_prediction = torch.eq(torch.argmax(preds, 1), torch.argmax(labels, 1))
    accuracy_all = correct_prediction.float()
    mask = mask.float()
    mask = mask / mask.mean()
    accuracy_all = accuracy_all * mask
    return accuracy_all.mean()


class GCNModel(nn.Module):
    def __init__(self, learning_rate, num_classes,
                 h, input_dim, HGCN,
                 train_idx, trtemask,
                 dp_fea0, edge_pos, train_mat01, mat01_tr_te, weight_decay):
        super(GCNModel, self).__init__()

        self.dp_fea0 = dp_fea0  # [dropout_getter_fn, num_features_nonzero_getter_fn]
        self.trtemask = trtemask
        self.weight_decay = weight_decay
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden1 = h
        self.HGCN = HGCN

        # Numpy-side data buffered as tensors.
        self.register_buffer('edge_pos_i', torch.from_numpy(np.asarray(edge_pos[:, 0]).astype('int64')))
        self.register_buffer('edge_pos_j', torch.from_numpy(np.asarray(edge_pos[:, 1]).astype('int64')))
        self.register_buffer('train_idx', torch.from_numpy(np.asarray(train_idx).astype('int64')))
        self.register_buffer('train_mat01', torch.from_numpy(train_mat01.astype('float32')))
        self.register_buffer('mat01_intra', torch.from_numpy(mat01_tr_te[0].astype('float32')))
        self.register_buffer('mat01_inter', torch.from_numpy(mat01_tr_te[1].astype('float32')))
        self.register_buffer('mat01_intra_rowsum',
                             torch.from_numpy(np.sum(mat01_tr_te[0], axis=1).astype('float32')))
        self.train_idx_size = int(np.shape(train_idx)[0])

        # Build classlayers (two GCN layers).
        self.classlayers = nn.ModuleList()
        self.classlayers.append(GraphConvolution(act=F.relu,
                                                 input_dim=self.input_dim,
                                                 output_dim=self.hidden1,
                                                 support=None,  # set per forward
                                                 sparse_inputs=True,
                                                 isSparse=True,
                                                 dropout=self.dp_fea0[0],
                                                 num_features_nonzero=self.dp_fea0[-1],
                                                 bias=True))

        self.classlayers.append(GraphConvolution(act=lambda x: x,
                                                 input_dim=self.hidden1,
                                                 output_dim=self.num_classes,
                                                 support=None,
                                                 sparse_inputs=False,
                                                 isSparse=True,
                                                 dropout=0,
                                                 num_features_nonzero=self.dp_fea0[-1],
                                                 bias=True))

        self.p_e_yy_w_contra = MLP(act=lambda x: x,
                                   input_dim=2 * self.num_classes,
                                   output_dim=1,
                                   sparse_inputs=False,
                                   isSparse=True,
                                   bias=True)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        # Forward-time results
        self.outputs = None
        self.concat_vec_DifGCN = None
        self.concat_vec_hgcn = None
        self.loss = torch.tensor(0.0)
        self.accuracy = torch.tensor(0.0)
        self.p_e_xy = torch.tensor(0.0)

    def forward(self, features, support, labels, mask):
        """Run a forward pass.

        features: torch sparse_coo_tensor of input features (already row-normalized).
        support : torch sparse_coo_tensor for D^{-0.5} (A+I) D^{-0.5}.
        labels  : tensor of one-hot labels.
        mask    : tensor of int/bool mask for the loss.
        """
        # Class layer 1 -> hidden
        self.classlayers[0].support = support
        self.classlayers[0].sparse_inputs = True
        h0 = self.classlayers[0](features)

        # Class layer 2 -> num_classes
        self.classlayers[1].support = support
        self.classlayers[1].sparse_inputs = False
        h1 = self.classlayers[1](h0)

        self.original_outputs = [h1]
        self.concat_vec_DifGCN = F.normalize(h1, p=2, dim=1)

        # HGCN forward
        hgcn_out = self.HGCN(features)
        self.original_outputs.append(hgcn_out)
        self.concat_vec_hgcn = F.normalize(hgcn_out, p=2, dim=1)

        self.outputs = F.normalize(0.6 * self.concat_vec_DifGCN + 0.4 * self.concat_vec_hgcn,
                                   p=2, dim=1)

        # ---- losses ----
        loss_q_yobs_x_g = masked_softmax_cross_entropy(self.outputs, labels, mask)

        y_ei_gcn = self.concat_vec_DifGCN.index_select(0, self.edge_pos_i)
        y_ej_hgcn = self.concat_vec_hgcn.index_select(0, self.edge_pos_j)
        y_ei_hgcn = self.concat_vec_hgcn.index_select(0, self.edge_pos_i)
        y_ej_gcn = self.concat_vec_DifGCN.index_select(0, self.edge_pos_j)

        p_e_xy_1 = -torch.mean(
            torch.log(torch.sigmoid(
                self.p_e_yy_w_contra(torch.cat([y_ei_gcn, y_ej_hgcn], dim=1))
            ))
        )
        p_e_xy_2 = -torch.mean(
            torch.log(torch.sigmoid(
                self.p_e_yy_w_contra(torch.cat([y_ei_hgcn, y_ej_gcn], dim=1))
            ))
        )
        self.p_e_xy = p_e_xy_1 + p_e_xy_2

        total = loss_q_yobs_x_g + 0.4 * self.p_e_xy
        total = total + self._contrastive_loss()

        # weight decay on the two classlayers and the MLP
        for i in range(2):
            for var in self.classlayers[i].vars.values():
                total = total + self.weight_decay * 0.5 * torch.sum(var ** 2)
        for var in self.p_e_yy_w_contra.vars.values():
            total = total + self.weight_decay * 0.5 * torch.sum(var ** 2)

        # Add HGCN's own (weight-decay) loss.
        total = total + self.HGCN.loss

        self.loss = total
        self.accuracy = masked_accuracy(self.outputs, labels, mask)
        return self.outputs, self.loss, self.accuracy

    def _contrastive_loss(self):
        loss = torch.tensor(0.0, device=self.concat_vec_DifGCN.device)

        # Eq. 4 — pairwise between concat_vec_DifGCN and concat_vec_hgcn (and reverse).
        cos_dist = torch.exp(torch.matmul(self.concat_vec_DifGCN,
                                          self.concat_vec_hgcn.t()) / 0.5)
        neg = torch.mean(cos_dist, dim=1)
        diag_cos = torch.diagonal(cos_dist, 0)
        positive_sum = diag_cos
        pos_neg1 = positive_sum / neg

        hp1 = 0.9

        cos_dist = torch.exp(torch.matmul(self.concat_vec_hgcn,
                                          self.concat_vec_DifGCN.t()) / 0.5)
        neg = torch.mean(cos_dist, dim=1)
        diag_cos = torch.diagonal(cos_dist, 0)
        positive_sum = diag_cos
        pos_neg2 = positive_sum / neg
        pos_neg3 = torch.cat([pos_neg1, pos_neg2], dim=0)
        loss = loss + (-hp1 * torch.mean(torch.log(pos_neg3)))

        # Supervised contrastive (round 1).
        h1 = self.concat_vec_DifGCN.index_select(0, self.train_idx)
        h2 = self.concat_vec_hgcn.index_select(0, self.train_idx)
        h_cos = torch.exp(torch.matmul(h1, h2.t()) / 0.5)
        sup_pos = torch.sum(h_cos * self.mat01_intra, dim=1)
        sup_neg = (torch.sum(h_cos * self.mat01_inter, dim=1) + sup_pos) / (self.train_idx_size - 1)
        sup_pos = sup_pos / self.mat01_intra_rowsum
        pos_neg_sup_1 = sup_pos / sup_neg

        # Supervised contrastive (round 2, swapped).
        h2_b = self.concat_vec_DifGCN.index_select(0, self.train_idx)
        h1_b = self.concat_vec_hgcn.index_select(0, self.train_idx)
        h_cos = torch.exp(torch.matmul(h1_b, h2_b.t()) / 0.5)
        sup_pos = torch.sum(h_cos * self.mat01_intra, dim=1)
        sup_neg = (torch.sum(h_cos * self.mat01_inter, dim=1) + sup_pos) / (self.train_idx_size - 1)
        sup_pos = sup_pos / self.mat01_intra_rowsum
        pos_neg_sup_2 = sup_pos / sup_neg

        pos_neg_sup_3 = torch.cat([pos_neg_sup_1, pos_neg_sup_2], dim=0)
        loss = loss + (-hp1 * torch.mean(torch.log(pos_neg_sup_3)))
        return loss