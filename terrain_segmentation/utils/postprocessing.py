import cv2
import numpy as np


def morphological_closing(binary_mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)


def close_multiclass_mask(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    classes = np.unique(mask)
    closed_maps = []

    for cls in classes:
        binary = (mask == cls).astype(np.uint8)
        closed_maps.append(morphological_closing(binary, kernel_size=kernel_size).astype(bool))

    closed_stack = np.stack(closed_maps, axis=0)
    candidate_count = closed_stack.sum(axis=0)
    single_candidate = candidate_count == 1

    output = mask.copy()
    if np.any(single_candidate):
        winner_idx = np.argmax(closed_stack[:, single_candidate], axis=0)
        output[single_candidate] = classes[winner_idx]

    return output


def remove_small_regions(mask: np.ndarray, min_size: int = 100, background_class: int = 0) -> np.ndarray:
    output = mask.copy()
    for cls in np.unique(mask):
        binary = (output == cls).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for label_idx in range(1, num_labels):
            area = stats[label_idx, cv2.CC_STAT_AREA]
            if area < min_size:
                output[labels == label_idx] = background_class
    return output
