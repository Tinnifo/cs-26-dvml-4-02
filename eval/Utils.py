import torch

# Function that sets the train mask for a given dataset based on the number of labels per class
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


# Function that sets the train mask for a given dataset based on a fraction of nodes
def set_budget_percent(data, fraction, seed):
    torch.manual_seed(seed)
    # number of training nodes determined by label budget
    num_train = int(fraction * data.num_nodes)
    # randomly choose nodes
    idx = torch.randperm(data.num_nodes)[:num_train]

    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    train_mask[idx] = True

    data.train_mask = train_mask

    return data
