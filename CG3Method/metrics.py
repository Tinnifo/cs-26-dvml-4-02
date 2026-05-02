import torch
import torch.nn.functional as F


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