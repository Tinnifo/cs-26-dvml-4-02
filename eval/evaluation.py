import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate(model, data, mask=None):
    """
    Evaluates the model on the specified mask (defaults to test_mask).
    """
    model.eval()
    if mask is None:
        mask = data.test_mask
        
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)

    y_true = data.y[mask].cpu().numpy()
    y_pred = pred[mask].cpu().numpy()

    if len(y_true) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)

    return acc, precision, recall, f1_macro, f1_micro
