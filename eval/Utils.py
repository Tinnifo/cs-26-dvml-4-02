import torch

def set_few_label_mask(data, num_labels_per_class, seed):
    torch.manual_seed(seed)
    num_classes = int(data.y.max()) + 1
    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)

    for c in range(num_classes):
        idx = (data.y == c).nonzero(as_tuple=True)[0]
        idx = idx[torch.randperm(idx.size(0))]
        selected = idx[:num_labels_per_class]
        train_mask[selected] = True

    data.train_mask = train_mask
    return data
