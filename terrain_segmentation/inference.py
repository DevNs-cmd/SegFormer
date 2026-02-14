import argparse
import os
from typing import Dict, List

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from models.segformer import SegFormerModel
from utils.augmentations import build_transforms
from utils.postprocessing import close_multiclass_mask, remove_small_regions
from utils.traversability import save_traversability_outputs
from utils.visualization import decode_segmentation, overlay_mask



def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)



def list_input_images(path: str) -> List[str]:
    if os.path.isdir(path):
        valid_ext = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
        return [
            os.path.join(path, name)
            for name in sorted(os.listdir(path))
            if name.lower().endswith(valid_ext)
        ]
    if os.path.isfile(path):
        return [path]
    raise FileNotFoundError(f"Input path does not exist: {path}")



def preprocess_image(image_rgb: np.ndarray, size: int, cfg: Dict) -> torch.Tensor:
    tf = build_transforms(False, size=size, aug_cfg=cfg["augmentation"])
    transformed = tf(image=image_rgb)
    return transformed["image"].unsqueeze(0)



def tta_predict(model, image_rgb: np.ndarray, cfg: Dict, device: torch.device) -> np.ndarray:
    infer_cfg = cfg["inference"]
    if bool(infer_cfg.get("tta", True)):
        scales = infer_cfg.get("tta_scales", [cfg["train"]["image_size"]])
    else:
        scales = [cfg["train"]["image_size"]]

    orig_h, orig_w = image_rgb.shape[:2]
    prob_sum = None
    count = 0

    for scale in scales:
        with torch.no_grad():
            x = preprocess_image(image_rgb, int(scale), cfg).to(device)
            logits = model(x)
            logits = F.interpolate(logits, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
            probs = F.softmax(logits, dim=1)

            prob_sum = probs if prob_sum is None else prob_sum + probs
            count += 1

            if bool(infer_cfg.get("tta_flip", True)):
                flipped = np.ascontiguousarray(image_rgb[:, ::-1, :])
                fx = preprocess_image(flipped, int(scale), cfg).to(device)
                flogits = model(fx)
                flogits = F.interpolate(flogits, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
                flogits = torch.flip(flogits, dims=[3])
                fprobs = F.softmax(flogits, dim=1)

                prob_sum = prob_sum + fprobs
                count += 1

    probs = prob_sum / max(1, count)
    pred = torch.argmax(probs, dim=1)[0].cpu().numpy().astype(np.uint8)
    return pred



def postprocess_mask(mask: np.ndarray, cfg: Dict) -> np.ndarray:
    if not bool(cfg["inference"].get("postprocess", True)):
        return mask

    mask = close_multiclass_mask(mask, kernel_size=int(cfg["inference"].get("closing_kernel", 5)))
    mask = remove_small_regions(
        mask,
        min_size=int(cfg["inference"].get("min_region_size", 100)),
        background_class=int(cfg["inference"].get("background_class", 0)),
    )
    return mask



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--input", type=str, default=None, help="Image path or directory")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    cfg = load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    input_path = args.input if args.input is not None else cfg["paths"]["test_images"]
    output_dir = args.output if args.output is not None else os.path.join(cfg["paths"]["output_dir"], "inference")
    os.makedirs(output_dir, exist_ok=True)

    palette = cfg["visualization"]["palette"]
    class_costs = cfg["traversability"]["class_costs"]
    heatmap_alpha = float(cfg["traversability"].get("heatmap_alpha", 0.5))

    image_paths = list_input_images(input_path)
    for image_path in tqdm(image_paths, desc="Inference"):
        image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image_bgr is None:
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        pred = tta_predict(model, image_rgb, cfg, device)
        pred = postprocess_mask(pred, cfg)

        pred_color = decode_segmentation(pred, palette)
        overlay = overlay_mask(image_rgb, pred_color, alpha=float(cfg["visualization"].get("overlay_alpha", 0.5)))

        stem = os.path.splitext(os.path.basename(image_path))[0]
        cv2.imwrite(os.path.join(output_dir, f"{stem}_mask.png"), pred)
        cv2.imwrite(os.path.join(output_dir, f"{stem}_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        trav_dir = os.path.join(output_dir, f"{stem}_traversability")
        save_traversability_outputs(
            out_dir=trav_dir,
            image_rgb=image_rgb,
            mask=pred,
            class_to_cost=class_costs,
            alpha=heatmap_alpha,
        )


if __name__ == "__main__":
    main()
