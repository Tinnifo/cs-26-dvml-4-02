import torch
import torch.nn.functional as F
from build_hierarchy_from_coarsen import normalize_edge_index
import copy


def train(model, data, epochs=200, lr=0.005):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    data = data.to(device)

    edge_index, edge_weight = normalize_edge_index(
            data.edge_index,
            data.num_nodes,
            data.edge_weight
        )

    data.edge_index = edge_index
    data.edge_weight = edge_weight
    
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    
    best_acc = 0    
    best_state = None
    patience = 20
    counter = 0
    warmup = 30
    contrastive_start = 30
    full_start = 80

    for epoch in range(epochs):

        model.train()
        opt.zero_grad()

        z_gcn, z_hgcn, z, logits = model(
            data.x,
            data.edge_index,
            data.edge_weight
        )
        z_gcn = F.normalize(z_gcn, dim=1)
        z_hgcn = F.normalize(z_hgcn, dim=1)
        z = F.normalize(z, dim=1)
        #Contrastive learning WITHOUT normalization → cosine similarity becomes unstable → ~2–3% drop easily

        train_idx = data.train_mask.nonzero(as_tuple=True)[0]
        labels = data.y[train_idx]

        pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        pos_mask.fill_diagonal_(0)

        neg_mask = 1 - pos_mask
        neg_mask.fill_diagonal_(0)   # <-- ADD THIS
                
        
        if epoch < warmup:
            mode = "cls"
        elif epoch < full_start:
            mode = "cls+cl"
        else:
            mode = "full"

        loss = model.compute_loss(
            z_gcn, z_hgcn, z, logits,
            data, train_idx, pos_mask, neg_mask,
            mode=mode
        )

        loss.backward()
        opt.step()

        # accuracy
        model.eval()
        with torch.no_grad():
            _, _, _, logits = model(
                data.x,
                data.edge_index,
                data.edge_weight
            )

            pred = logits.argmax(dim=1)

            val_acc = (pred[data.val_mask] == data.y[data.val_mask]).float().mean()
            test_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean()

        print(f"{epoch} | loss {loss:.4f} | val {val_acc:.4f} | test {test_acc:.4f}")
        if val_acc > best_acc:
            best_state = copy.deepcopy(model.state_dict())
            best_acc = val_acc
            counter = 0
        else:
            counter += 1

        if counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
        model.eval()

        with torch.no_grad():
            _, _, _, logits = model(
                data.x,
                data.edge_index,
                data.edge_weight
            )

            pred = logits.argmax(dim=1)
            test_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean()

        print("FINAL TEST ACC:", test_acc.item())
        
def train_gcn(model, data, epochs=200, lr=0.01):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    data = data.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    for epoch in range(epochs):

        model.train()
        opt.zero_grad()

        logits = model(data.x, data.edge_index, data.edge_weight)

        loss = F.cross_entropy(
            logits[data.train_mask],
            data.y[data.train_mask]
        )

        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index, data.edge_weight)
            pred = logits.argmax(dim=1)

            acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean()

        print(f"[GCN] {epoch} | loss {loss:.4f} | acc {acc:.4f}")