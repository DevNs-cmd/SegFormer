import os
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np


def decode_segmentation(mask: np.ndarray, palette: List[List[int]]) -> np.ndarray:
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for idx, color in enumerate(palette):
        rgb[mask == idx] = np.array(color, dtype=np.uint8)
    return rgb


def overlay_mask(image_rgb: np.ndarray, mask_rgb: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    return cv2.addWeighted(image_rgb, 1.0 - alpha, mask_rgb, alpha, 0)


def error_map(gt_mask: np.ndarray, pred_mask: np.ndarray) -> np.ndarray:
    err = (gt_mask != pred_mask).astype(np.uint8) * 255
    return np.stack([err, np.zeros_like(err), np.zeros_like(err)], axis=-1)


def save_side_by_side(
    out_path: str,
    image_rgb: np.ndarray,
    gt_rgb: np.ndarray,
    pred_rgb: np.ndarray,
    err_rgb: np.ndarray,
) -> None:
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    panel = np.concatenate([image_rgb, gt_rgb, pred_rgb, err_rgb], axis=1)
    cv2.imwrite(out_path, cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))


def save_confusion_matrix_plot(confusion: np.ndarray, class_names: List[str], out_path: str) -> None:
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(confusion, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(im, ax=ax)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
