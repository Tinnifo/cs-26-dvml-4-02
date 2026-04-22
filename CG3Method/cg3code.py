import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dropout_edge
from torch_geometric.nn import TopKPooling

# ======================
# 1. Setup & Data
# ======================
device = torch.device('cpu') # Use CPU as we verified earlier
dataset = Planetoid(root='data/CiteSeer', name='CiteSeer')
data = dataset[0].to(device)

# ======================
# 2. Model Definitions
# ======================


class GCN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_classes)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x
# ======================
# HGCN-style model
# ======================
class HGCN_PyG(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes):
        super().__init__()

        # -------- Encoder (coarsening path) --------
        self.gcn1 = GCNConv(in_dim, hidden_dim)
        self.pool1 = TopKPooling(hidden_dim, ratio=0.7)

        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.pool2 = TopKPooling(hidden_dim, ratio=0.7)

        # -------- Bottleneck --------
        self.gcn3 = GCNConv(hidden_dim, hidden_dim)

        # -------- Decoder (refinement path) --------
        self.gcn4 = GCNConv(hidden_dim, hidden_dim)
        self.gcn5 = GCNConv(hidden_dim, hidden_dim)

        # -------- Output --------
        self.out = GCNConv(hidden_dim, num_classes)

    def forward(self, x, edge_index, batch=None):

        # If no batch given (single graph dataset)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # ======================
        # Level 0
        # ======================
        x0 = F.relu(self.gcn1(x, edge_index))
        x0_res = x0

        x1, edge1, _, batch1, _, _ = self.pool1(x0, edge_index, None, batch)

        # ======================
        # Level 1
        # ======================
        x1 = F.relu(self.gcn2(x1, edge1))
        x1_res = x1

        x2, edge2, _, batch2, _, _ = self.pool2(x1, edge1, None, batch1)

        # ======================
        # Bottleneck
        # ======================
        x2 = F.relu(self.gcn3(x2, edge2))

        # ======================
        # Refinement (reverse path)
        # ======================

        # up to level 1
        x1_up = x1_res + x2[batch2]  # skip fusion

        x1_up = F.relu(self.gcn4(x1_up, edge1))

        # up to level 0
        x0_up = x0_res + x1_up[batch1]

        x0_up = F.relu(self.gcn5(x0_up, edge_index))

        # output
        out = self.out(x0_up, edge_index)

        return out
    
class CG3_PyG(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes):
        super().__init__()
        # View 1
        self.gcn1 = GCNConv(in_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, num_classes)
        # View 2 (separate parameters for contrastive learning)
        self.gcn1_v2 = GCNConv(in_dim, hidden_dim)
        self.gcn2_v2 = GCNConv(hidden_dim, num_classes)

""""
1 model gcn
2 model hgcn
"""
    def forward(self, x, edge_index1, edge_index2):
        # Process View 1
        z1 = self.gcn2(x, edge_index1)
        # Process View 2
        z2 = self.gcn2_v2(x, edge_index2)
        return z1, z2

# ======================
# 3. CG3 Specific Losses
# ======================

def contrastive_loss(z1, z2, tau=0.5):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    logits = z1 @ z2.T / tau
    labels = torch.arange(z1.size(0)).to(z1.device)

    return F.cross_entropy(logits, labels)

def supervised_contrastive(z1, z2, labels, mask):
    z1, z2, labels = z1[mask], z2[mask], labels[mask]
    z1, z2 = F.normalize(z1, dim=1), F.normalize(z2, dim=1)
    sim = torch.exp((z1 @ z2.T) / 0.5)
    pos_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
    pos_mask.fill_diagonal_(0)
    neg_mask = 1 - pos_mask
    pos = (sim * pos_mask).sum(dim=1)
    neg = (sim * neg_mask).sum(dim=1)
    return -torch.log((pos + 1e-8) / (pos + neg + 1e-8)).mean()

def edge_loss(z1, z2, edge_index, num_nodes):
    i, j = edge_index
    pos_score = (z1[i] * z2[j]).sum(dim=1)
    neg_i = torch.randint(0, num_nodes, (i.size(0),), device=z1.device)
    neg_j = torch.randint(0, num_nodes, (j.size(0),), device=z1.device)
    neg_score = (z1[neg_i] * z2[neg_j]).sum(dim=1)
    return F.binary_cross_entropy_with_logits(torch.cat([pos_score, neg_score]), 
                                             torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)]))

# ======================
# 4. Training Functions
# ======================
""""

def train_gcn():
    model = GCN(dataset.num_features, 64, dataset.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
    model.eval()
    pred = model(data.x, data.edge_index).argmax(dim=1)
    acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean()
    return acc.item()

def train_hgcn():
    model = HGCN_PyG(dataset.num_features, 64, dataset.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    for epoch in range(200):
        model.train()
        optimizer.zero_grad()

        out = model(data.x, data.edge_index)

        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])

        loss.backward()
        optimizer.step()

    model.eval()
    pred = model(data.x, data.edge_index).argmax(dim=1)
    acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean()

    return acc.item()
""""


def train_cg3():
    model = CG3_PyG(dataset.num_features, 64, dataset.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
  
  
    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        z1, z2 = model(data.x, data.edge_index)
        
        l_cls = F.cross_entropy(z1[data.train_mask], data.y[data.train_mask])
        l_cl = contrastive_loss(z1, z2)
        l_sup = supervised_contrastive(z1, z2, data.y, data.train_mask)
        l_edge = edge_loss(z1, z2, data.edge_index, data.num_nodes)
        
        loss = l_cls + 0.2*l_cl + 0.1*l_sup + 0.1*l_edge
        loss.backward()
        optimizer.step()
    model.eval()
    z_eval = model(data.x, data.edge_index)
    pred = z_eval.argmax(dim=1)
    acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean()
    return acc.item()

# ======================
# 5. Execution
# ======================
print("Starting Standard GCN training...")
gcn_acc = train_gcn()
print(f"GCN Test Accuracy: {gcn_acc:.4f}\n")

print("Starting CG3 training...")
cg3_acc = train_cg3()
print(f"CG3 Test Accuracy: {cg3_acc:.4f}\n")

print("-" * 30)
print(f"Final Comparison on CiteSeer:")
print(f"GCN: {gcn_acc:.4f}")
print(f"CG3: {cg3_acc:.4f}")
print(f"Improvement: {((cg3_acc - gcn_acc) / gcn_acc) * 100:.2f}%")