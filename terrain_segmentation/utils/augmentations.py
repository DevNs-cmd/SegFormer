from typing import Dict

import albumentations as A
from albumentations.pytorch import ToTensorV2


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _random_shadow_transform(enabled: bool):
    if not enabled:
        return None
    if hasattr(A, "RandomShadow"):
        return A.RandomShadow(p=0.25)
    return None


def build_transforms(is_train: bool, size: int, aug_cfg: Dict):
    transforms = []

    if is_train and bool(aug_cfg.get("enable", True)):
        if aug_cfg.get("random_scale", True):
            transforms.append(A.RandomScale(scale_limit=(-0.25, 0.25), p=0.5))

        if aug_cfg.get("perspective_transform", True):
            transforms.append(A.Perspective(scale=(0.03, 0.08), p=0.3))

        if aug_cfg.get("random_brightness_contrast", True):
            transforms.append(A.RandomBrightnessContrast(p=0.5))

        if aug_cfg.get("gaussian_noise", True):
            transforms.append(A.GaussNoise(std_range=(0.01, 0.05), p=0.35))

        if aug_cfg.get("motion_blur", True):
            transforms.append(A.MotionBlur(blur_limit=(3, 7), p=0.25))

        if aug_cfg.get("color_jitter", True):
            transforms.append(A.ColorJitter(p=0.35))

        shadow = _random_shadow_transform(aug_cfg.get("random_shadow", False))
        if shadow is not None:
            transforms.append(shadow)

        if aug_cfg.get("horizontal_flip", True):
            transforms.append(A.HorizontalFlip(p=0.5))

        transforms.append(A.PadIfNeeded(min_height=size, min_width=size, border_mode=0, fill=0, fill_mask=0))
        if aug_cfg.get("random_crop", True):
            transforms.append(A.RandomCrop(height=size, width=size))
        else:
            transforms.append(A.Resize(height=size, width=size))
    else:
        transforms.append(A.Resize(height=size, width=size))

    transforms.append(A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    transforms.append(ToTensorV2())

    return A.Compose(transforms)
