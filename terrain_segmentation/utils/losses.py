from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index: int = 255, class_weights: Optional[Sequence[float]] = None):
        super().__init__()
        self.ignore_index = ignore_index
        if class_weights is not None:
            self.register_buffer("class_weights", torch.tensor(class_weights, dtype=torch.float32))
        else:
            self.class_weights = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        weights = self.class_weights.to(logits.device) if self.class_weights is not None else None
        return F.cross_entropy(logits, targets, ignore_index=self.ignore_index, weight=weights)


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0, ignore_index: int = 255):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        num_classes = logits.shape[1]

        valid_mask = targets != self.ignore_index
        safe_targets = targets.clone()
        safe_targets[~valid_mask] = 0

        onehot = torch.zeros_like(logits).scatter_(1, safe_targets.unsqueeze(1), 1.0)
        valid_mask_exp = valid_mask.unsqueeze(1)

        probs = probs * valid_mask_exp
        onehot = onehot * valid_mask_exp

        intersection = (probs * onehot).sum(dim=(0, 2, 3))
        union = (probs + onehot).sum(dim=(0, 2, 3))

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class FocalLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 1.0,
        gamma: float = 2.0,
        ignore_index: int = 255,
        class_weights: Optional[Sequence[float]] = None,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        if class_weights is not None:
            self.register_buffer("class_weights", torch.tensor(class_weights, dtype=torch.float32))
        else:
            self.class_weights = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        weights = self.class_weights.to(logits.device) if self.class_weights is not None else None
        ce = F.cross_entropy(logits, targets, reduction="none", ignore_index=self.ignore_index, weight=weights)

        valid = targets != self.ignore_index
        ce_valid = ce[valid]
        if ce_valid.numel() == 0:
            return logits.new_tensor(0.0)

        pt = torch.exp(-ce_valid)
        focal = self.alpha * ((1.0 - pt) ** self.gamma) * ce_valid
        return focal.mean()


class HybridLoss(nn.Module):
    """Final loss = 0.5 * CE + 0.3 * Dice + 0.2 * Focal by default."""

    def __init__(
        self,
        ce_weight: float = 0.5,
        dice_weight: float = 0.3,
        focal_weight: float = 0.2,
        ignore_index: int = 255,
        class_weights: Optional[Sequence[float]] = None,
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

        self.ce = CrossEntropyLoss(ignore_index=ignore_index, class_weights=class_weights)
        self.dice = DiceLoss(ignore_index=ignore_index)
        self.focal = FocalLoss(ignore_index=ignore_index, class_weights=class_weights)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(logits, targets)
        dice_loss = self.dice(logits, targets)
        focal_loss = self.focal(logits, targets)
        return self.ce_weight * ce_loss + self.dice_weight * dice_loss + self.focal_weight * focal_loss
