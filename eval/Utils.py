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
    
    # Store the original train_mask to sample only from the original training nodes
    # This ensures consistency with standard benchmarks and avoids data leakage
    original_train_mask = data.train_mask.clone()
    
    # Initialize new train mask
    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    
    for c in range(num_classes):
        # Sample only from nodes that belong to class c AND are in the original training set
        idx = ((data.y == c) & original_train_mask).nonzero(as_tuple=True)[0]
        
        # Shuffle indices
        idx = idx[torch.randperm(idx.size(0))]
        
        # Select budget (or all available if budget > available)
        selected = idx[:num_labels_per_class]
        train_mask[selected] = True

    data.train_mask = train_mask
    
    # In standard Planetoid benchmarks, val and test masks are fixed.
    # By sampling only from the original train_mask, we ensure no overlap 
    # and maintain the original evaluation distributions.

    return data


# Function that sets the train mask for a given dataset based on a fraction of nodes
def set_budget_percent(data, fraction, seed):
    set_seed(seed)
    
    # Sample only from the original training nodes
    original_train_mask = data.train_mask.clone()
    original_train_indices = original_train_mask.nonzero(as_tuple=True)[0]
    
    # Total available training nodes in the standard benchmark
    num_available = len(original_train_indices)
    
    # Number of training nodes requested
    num_train = int(fraction * data.num_nodes)
    
    if num_train > num_available:
        # We restrict to the original training pool to follow standard benchmarks
        num_train = num_available

    # Randomly choose nodes from the training pool
    perm = torch.randperm(num_available)
    selected_indices = original_train_indices[perm[:num_train]]

    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    train_mask[selected_indices] = True

    data.train_mask = train_mask
    
    # Val and test masks remain fixed as per standard benchmark rules

    return data
