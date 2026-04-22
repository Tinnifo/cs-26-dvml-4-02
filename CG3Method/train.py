import torch
import torch.nn.functional as F


def train(model, data, epochs=200, lr=0.01):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    data = data.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):

        model.train()
        opt.zero_grad()

        z_gcn, z_hgcn, z, logits = model(
            data.x,
            data.edge_index,
            data.edge_weight
        )

        train_idx = data.train_mask.nonzero(as_tuple=True)[0]
        labels = data.y[train_idx]

        pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        pos_mask.fill_diagonal_(0)

        neg_mask = 1 - pos_mask
        neg_mask.fill_diagonal_(0)   # <-- ADD THIS
                


        loss = model.compute_loss(
            z_gcn, z_hgcn, z, logits,
            data, train_idx, pos_mask, neg_mask
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

            acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean()

        print(f"{epoch} | loss {loss:.4f} | acc {acc:.4f}")
        
        
def train_gcn(model, data, epochs=200, lr=0.01):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    data = data.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)

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