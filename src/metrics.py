"""
src/metrics.py
Evaluation metrics: IoU (Jaccard), F1, Precision, Recall.
All operate on binary tensors / numpy arrays.
"""

import torch
import numpy as np


def _to_binary(preds, threshold=0.5):
    if isinstance(preds, torch.Tensor):
        return (torch.sigmoid(preds) > threshold).float()
    return (preds > threshold).astype(np.float32)


def iou_score(logits, targets, threshold=0.5, smooth=1e-6):
    preds   = _to_binary(logits, threshold)
    inter   = (preds * targets).sum()
    union   = preds.sum() + targets.sum() - inter
    return ((inter + smooth) / (union + smooth)).item()


def f1_score(logits, targets, threshold=0.5, smooth=1e-6):
    preds = _to_binary(logits, threshold)
    tp    = (preds * targets).sum()
    fp    = preds.sum() - tp
    fn    = targets.sum() - tp
    return ((2 * tp + smooth) / (2 * tp + fp + fn + smooth)).item()


def precision_recall(logits, targets, threshold=0.5, smooth=1e-6):
    preds = _to_binary(logits, threshold)
    tp    = (preds * targets).sum()
    fp    = preds.sum() - tp
    fn    = targets.sum() - tp
    prec  = ((tp + smooth) / (tp + fp + smooth)).item()
    rec   = ((tp + smooth) / (tp + fn + smooth)).item()
    return prec, rec


def change_percentage(logits, threshold=0.5):
    """Return the fraction of pixels predicted as changed."""
    preds = _to_binary(logits, threshold)
    return (preds.sum() / preds.numel() * 100).item()