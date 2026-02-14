import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from torch.utils.data import Dataset


VALID_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


class SegmentationDataset(Dataset):
    """Reads image/mask pairs for train/val/test splits from config paths."""

    def __init__(self, cfg: Dict, split: str, transforms=None):
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be one of: train, val, test")

        self.cfg = cfg
        self.split = split
        self.transforms = transforms
        self.ignore_index = int(cfg["model"].get("ignore_index", 255))

        self.images_dir = cfg["paths"][f"{split}_images"]
        self.masks_dir = cfg["paths"].get(f"{split}_masks")

        if not os.path.isdir(self.images_dir):
            raise FileNotFoundError(f"Image directory does not exist: {self.images_dir}")

        self.samples = self._build_samples()
        if not self.samples:
            raise RuntimeError(f"No image files found in split '{split}' at: {self.images_dir}")

    def _build_samples(self) -> List[Tuple[str, Optional[str], str]]:
        image_names = sorted(
            [f for f in os.listdir(self.images_dir) if f.lower().endswith(VALID_IMAGE_EXTENSIONS)]
        )

        samples: List[Tuple[str, Optional[str], str]] = []
        for image_name in image_names:
            image_path = os.path.join(self.images_dir, image_name)
            stem = os.path.splitext(image_name)[0]

            mask_path = None
            if self.masks_dir and os.path.isdir(self.masks_dir):
                candidate_mask = os.path.join(self.masks_dir, f"{stem}.png")
                if os.path.isfile(candidate_mask):
                    mask_path = candidate_mask
                elif self.split != "test":
                    raise FileNotFoundError(f"Mask missing for image '{image_name}': {candidate_mask}")

            samples.append((image_path, mask_path, image_name))

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        image_path, mask_path, name = self.samples[idx]

        image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        mask = None
        if mask_path is not None:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(f"Failed to read mask: {mask_path}")
            mask = mask.astype(np.int64)

        if self.transforms is not None:
            if mask is None:
                transformed = self.transforms(image=image)
            else:
                transformed = self.transforms(image=image, mask=mask)
            image = transformed["image"]
            if mask is not None:
                mask = transformed["mask"].long()

        sample = {
            "image": image,
            "name": name,
            "image_path": image_path,
        }
        if mask is not None:
            sample["mask"] = mask
        return sample
