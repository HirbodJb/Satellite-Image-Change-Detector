"""
src/metrics.py

Evaluation helpers for binary change-detection outputs.
The functions below work with either PyTorch tensors or NumPy arrays and
always convert predictions to a binary mask before computing metrics.
"""

import torch
import numpy as np


def _to_binary(preds, threshold=0.5):
    # Model outputs are usually logits, so tensors are passed through sigmoid
    # before thresholding. NumPy arrays are assumed to already contain scores
    # in [0, 1] or comparable probabilities.
    if isinstance(preds, torch.Tensor):
        return (torch.sigmoid(preds) > threshold).float()
    return (preds > threshold).astype(np.float32)


def iou_score(logits, targets, threshold=0.5, smooth=1e-6):
    # IoU measures the overlap between prediction and target masks.
    preds   = _to_binary(logits, threshold)
    inter   = (preds * targets).sum()
    union   = preds.sum() + targets.sum() - inter
    return ((inter + smooth) / (union + smooth)).item()


def f1_score(logits, targets, threshold=0.5, smooth=1e-6):
    # F1 balances precision and recall using the harmonic mean of both.
    preds = _to_binary(logits, threshold)
    tp    = (preds * targets).sum()
    fp    = preds.sum() - tp
    fn    = targets.sum() - tp
    return ((2 * tp + smooth) / (2 * tp + fp + fn + smooth)).item()


def precision_recall(logits, targets, threshold=0.5, smooth=1e-6):
    # Precision answers "how many predicted positives were correct?"
    # Recall answers "how many actual positives were recovered?"
    preds = _to_binary(logits, threshold)
    tp    = (preds * targets).sum()
    fp    = preds.sum() - tp
    fn    = targets.sum() - tp
    prec  = ((tp + smooth) / (tp + fp + smooth)).item()
    rec   = ((tp + smooth) / (tp + fn + smooth)).item()
    return prec, rec


def change_percentage(logits, threshold=0.5):
    """Return the percentage of pixels predicted as changed."""
    preds = _to_binary(logits, threshold)
    return (preds.sum() / preds.numel() * 100).item()