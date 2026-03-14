import torch
import numpy as np
import random

def set_seed(seed):
    """Sets random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Function that sets the train mask for a given dataset based on the number of labels per class
def set_few_label_mask(data, num_labels_per_class, seed):
    set_seed(seed)
    num_classes = int(data.y.max()) + 1
    
    # Initialize masks
    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    
    for c in range(num_classes):
        idx = (data.y == c).nonzero(as_tuple=True)[0]
        # Only select from nodes that are available (you might want to exclude certain nodes)
        # For Planetoid, we typically just pick from the whole set if we are re-masking
        idx = idx[torch.randperm(idx.size(0))]
        selected = idx[:num_labels_per_class]
        train_mask[selected] = True

    data.train_mask = train_mask
    
    # Ensure test_mask does not overlap with train_mask
    # If a node is in train_mask, it must not be in test_mask
    if hasattr(data, 'test_mask'):
        data.test_mask = data.test_mask & ~data.train_mask
    
    # Similarly for val_mask if it exists
    if hasattr(data, 'val_mask'):
        data.val_mask = data.val_mask & ~data.train_mask

    return data


# Function that sets the train mask for a given dataset based on a fraction of nodes
def set_budget_percent(data, fraction, seed):
    set_seed(seed)
    # number of training nodes determined by label budget
    num_train = int(fraction * data.num_nodes)
    # randomly choose nodes
    idx = torch.randperm(data.num_nodes)[:num_train]

    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    train_mask[idx] = True

    data.train_mask = train_mask
    
    # Ensure test_mask and val_mask do not overlap with train_mask
    if hasattr(data, 'test_mask'):
        data.test_mask = data.test_mask & ~data.train_mask
    if hasattr(data, 'val_mask'):
        data.val_mask = data.val_mask & ~data.train_mask

    return data
