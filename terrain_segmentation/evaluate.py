import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.segformer import SegFormerModel
from utils.augmentations import IMAGENET_MEAN, IMAGENET_STD, build_transforms
from utils.dataset import SegmentationDataset
from utils.metrics import compute_confusion_matrix, confusion_to_numpy, metrics_from_confusion
from utils.visualization import decode_segmentation, error_map, save_confusion_matrix_plot, save_side_by_side



def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)



def denormalize_image(image_tensor: torch.Tensor) -> np.ndarray:
    image = image_tensor.permute(1, 2, 0).cpu().numpy()
    image = image * np.array(IMAGENET_STD, dtype=np.float32) + np.array(IMAGENET_MEAN, dtype=np.float32)
    image = (image * 255.0).clip(0, 255).astype(np.uint8)
    return image



def class_names_from_config(cfg: Dict) -> List[str]:
    names = []
    labels = cfg.get("labels", {})
    for idx in range(int(cfg["model"]["num_classes"])):
        names.append(labels.get(idx, labels.get(str(idx), f"class_{idx}")))
    return names



def evaluate_split(cfg: Dict, split: str) -> Dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = SegmentationDataset(
        cfg=cfg,
        split=split,
        transforms=build_transforms(False, cfg["train"]["image_size"], cfg["augmentation"]),
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )

    model = SegFormerModel(
        backbone=cfg["model"]["backbone"],
        num_classes=cfg["model"]["num_classes"],
        ignore_index=cfg["model"].get("ignore_index", 255),
        use_pretrained=False,
    ).to(device)

    checkpoint_path = os.path.join(cfg["paths"]["checkpoint_dir"], "best_model.pt")
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model"], strict=True)
    model.eval()

    num_classes = int(cfg["model"]["num_classes"])
    ignore_index = int(cfg["model"].get("ignore_index", 255))

    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)
    sample_results: List[Tuple[float, str, torch.Tensor, torch.Tensor, torch.Tensor]] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluate {split}"):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)

            logits = model(images)
            if logits.shape[-2:] != masks.shape[-2:]:
                logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)

            preds = torch.argmax(logits, dim=1)
            confusion += compute_confusion_matrix(preds, masks, num_classes, ignore_index=ignore_index)

            for i in range(preds.shape[0]):
                sample_conf = compute_confusion_matrix(
                    preds[i : i + 1], masks[i : i + 1], num_classes, ignore_index=ignore_index
                )
                sample_metric = metrics_from_confusion(sample_conf)
                sample_results.append(
                    (
                        float(sample_metric["mean_iou"]),
                        batch["name"][i],
                        batch["image"][i].cpu(),
                        masks[i].cpu(),
                        preds[i].cpu(),
                    )
                )

    aggregate_metrics = metrics_from_confusion(confusion)

    out_dir = os.path.join(cfg["paths"]["output_dir"], f"eval_{split}")
    os.makedirs(out_dir, exist_ok=True)

    save_confusion_matrix_plot(
        confusion_to_numpy(confusion),
        class_names_from_config(cfg),
        os.path.join(out_dir, "confusion_matrix.png"),
    )

    palette = cfg["visualization"]["palette"]
    worst_samples = sorted(sample_results, key=lambda item: item[0])[:10]

    for rank, (sample_iou, name, image_t, gt_t, pred_t) in enumerate(worst_samples):
        image = denormalize_image(image_t)
        gt = gt_t.numpy().astype(np.uint8)
        pred = pred_t.numpy().astype(np.uint8)

        gt_rgb = decode_segmentation(gt, palette)
        pred_rgb = decode_segmentation(pred, palette)
        err_rgb = error_map(gt, pred)

        base_name = os.path.splitext(name)[0]
        out_path = os.path.join(out_dir, f"worst_{rank + 1:02d}_{sample_iou:.4f}_{base_name}.png")
        save_side_by_side(out_path, image, gt_rgb, pred_rgb, err_rgb)

    metrics_yaml = {
        "split": split,
        "per_class_iou": [float(x) for x in aggregate_metrics["per_class_iou"]],
        "mean_iou": float(aggregate_metrics["mean_iou"]),
        "pixel_accuracy": float(aggregate_metrics["pixel_accuracy"]),
    }
    with open(os.path.join(out_dir, "metrics.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(metrics_yaml, f, sort_keys=False)

    return metrics_yaml



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    metrics = evaluate_split(cfg, args.split)
    print(yaml.safe_dump(metrics, sort_keys=False))


if __name__ == "__main__":
    main()
