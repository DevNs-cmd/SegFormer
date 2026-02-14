import os
from typing import Dict, Iterable, Union

import cv2
import numpy as np


def _build_cost_lut(class_to_cost: Union[Dict, Iterable], num_classes: int) -> np.ndarray:
    if isinstance(class_to_cost, dict):
        lut = np.ones(num_classes, dtype=np.float32)
        for key, value in class_to_cost.items():
            cls_idx = int(key)
            if cls_idx < 0 or cls_idx >= num_classes:
                continue
            lut[cls_idx] = float(value)
        return lut

    values = list(class_to_cost)
    if len(values) < num_classes:
        raise ValueError("class_costs length must be >= num_classes")
    return np.asarray(values[:num_classes], dtype=np.float32)


def traversability_map(mask: np.ndarray, class_to_cost: Union[Dict, Iterable]) -> np.ndarray:
    num_classes = int(mask.max()) + 1
    lut = _build_cost_lut(class_to_cost, num_classes)
    cost_map = lut[mask]

    min_val = float(cost_map.min())
    max_val = float(cost_map.max())
    normalized = (cost_map - min_val) / (max_val - min_val + 1e-7)
    return (normalized * 255.0).astype(np.uint8)


def overlay_heatmap(image_bgr: np.ndarray, heatmap_gray: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    colored = cv2.applyColorMap(heatmap_gray, cv2.COLORMAP_JET)
    return cv2.addWeighted(image_bgr, 1.0 - alpha, colored, alpha, 0)


def save_traversability_outputs(
    out_dir: str,
    image_rgb: np.ndarray,
    mask: np.ndarray,
    class_to_cost: Union[Dict, Iterable],
    alpha: float = 0.5,
):
    os.makedirs(out_dir, exist_ok=True)
    heatmap = traversability_map(mask, class_to_cost)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    overlay = overlay_heatmap(image_bgr, heatmap, alpha=alpha)

    cv2.imwrite(os.path.join(out_dir, "traversability_heatmap.png"), heatmap)
    cv2.imwrite(os.path.join(out_dir, "traversability_overlay.png"), overlay)
    return heatmap, overlay
