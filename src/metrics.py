"""
metrics.py

Evaluation metrics used throughout the project: accuracy, F1, AUROC, and the
loss-based generalization gap G = L_test - L_train.

All metrics are computed from raw logits + integer/float labels in {0,1}.
"""

import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score


@torch.no_grad()
def compute_loss(model, X: torch.Tensor, y: torch.Tensor) -> float:
    logits = model(X)
    loss = F.binary_cross_entropy_with_logits(logits, y)
    return loss.item()


@torch.no_grad()
def evaluate(model, X: torch.Tensor, y: torch.Tensor) -> dict:
    """
    Returns a dict with loss, accuracy, F1, and AUROC for the given split.
    """
    logits = model(X)
    loss = F.binary_cross_entropy_with_logits(logits, y).item()

    probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
    y_true = y.squeeze(1).cpu().numpy()
    y_pred = (probs >= 0.5).astype(int)

    acc = float((y_pred == y_true).mean())

    # F1 / AUROC can fail if a split is degenerate (single class); guard defensively.
    try:
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
    except ValueError:
        f1 = float("nan")
    try:
        auroc = float(roc_auc_score(y_true, probs))
    except ValueError:
        auroc = float("nan")

    return {"loss": loss, "accuracy": acc, "f1": f1, "auroc": auroc}


def generalization_gap(train_loss: float, test_loss: float) -> float:
    """G = L_test - L_train, the primary generalization measure used in the plan."""
    return test_loss - train_loss