from typing import Dict

import numpy as np
import torch


def compute_confusion_matrix(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int = 255,
) -> torch.Tensor:
    """Vectorized confusion matrix for batched prediction/target tensors."""
    preds = preds.view(-1).to(torch.int64)
    targets = targets.view(-1).to(torch.int64)

    valid = (targets != ignore_index) & (targets >= 0) & (targets < num_classes)
    preds = preds[valid]
    targets = targets[valid]

    if preds.numel() == 0:
        return torch.zeros((num_classes, num_classes), dtype=torch.int64, device=targets.device)

    indices = targets * num_classes + preds
    confusion = torch.bincount(indices, minlength=num_classes * num_classes)
    return confusion.reshape(num_classes, num_classes)


def per_class_iou(confusion: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    confusion = confusion.to(torch.float64)
    true_positive = torch.diag(confusion)
    false_positive = confusion.sum(dim=0) - true_positive
    false_negative = confusion.sum(dim=1) - true_positive
    denom = true_positive + false_positive + false_negative
    return true_positive / (denom + eps)


def mean_iou(confusion: torch.Tensor) -> float:
    return float(per_class_iou(confusion).mean().item())


def pixel_accuracy(confusion: torch.Tensor, eps: float = 1e-7) -> float:
    confusion = confusion.to(torch.float64)
    correct = torch.diag(confusion).sum()
    total = confusion.sum()
    return float((correct / (total + eps)).item())


def metrics_from_confusion(confusion: torch.Tensor) -> Dict:
    iou = per_class_iou(confusion)
    return {
        "per_class_iou": iou.cpu().numpy(),
        "mean_iou": float(iou.mean().item()),
        "pixel_accuracy": pixel_accuracy(confusion),
    }


def confusion_to_numpy(confusion: torch.Tensor) -> np.ndarray:
    return confusion.detach().cpu().numpy().astype(np.int64)
