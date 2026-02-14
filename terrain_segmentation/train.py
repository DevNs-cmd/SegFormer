import argparse
import os
import random
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.segformer import SegFormerModel
from utils.augmentations import build_transforms
from utils.dataset import SegmentationDataset
from utils.losses import HybridLoss
from utils.metrics import compute_confusion_matrix, metrics_from_confusion


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)



def ensure_dirs(cfg: Dict) -> None:
    os.makedirs(cfg["paths"]["output_dir"], exist_ok=True)
    os.makedirs(cfg["paths"]["log_dir"], exist_ok=True)
    os.makedirs(cfg["paths"]["checkpoint_dir"], exist_ok=True)



def build_loaders(cfg: Dict) -> Tuple[DataLoader, DataLoader]:
    train_tf = build_transforms(True, cfg["train"]["image_size"], cfg["augmentation"])
    val_tf = build_transforms(False, cfg["train"]["image_size"], cfg["augmentation"])

    train_ds = SegmentationDataset(cfg, split="train", transforms=train_tf)
    val_ds = SegmentationDataset(cfg, split="val", transforms=val_tf)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=pin_memory,
        drop_last=False,
    )
    return train_loader, val_loader



def get_scheduler(optimizer: torch.optim.Optimizer, cfg: Dict, steps_per_epoch: int):
    scheduler_name = str(cfg["train"].get("scheduler", "cosine")).lower()
    if scheduler_name == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=cfg["train"]["learning_rate"],
            epochs=cfg["train"]["epochs"],
            steps_per_epoch=max(1, steps_per_epoch),
            pct_start=0.1,
            div_factor=10.0,
            final_div_factor=100.0,
        )

    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg["train"]["epochs"],
        eta_min=cfg["train"]["learning_rate"] * 0.01,
    )



def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    loss_fn: nn.Module,
    device: torch.device,
    amp_enabled: bool,
    grad_clip: float,
    epoch: int,
    writer: SummaryWriter,
    scheduler_name: str,
) -> float:
    model.train()
    total_loss = 0.0

    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    pbar = tqdm(loader, desc=f"Train {epoch}")

    for step, batch in enumerate(pbar):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            logits = model(images)
            if logits.shape[-2:] != masks.shape[-2:]:
                logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            loss = loss_fn(logits, masks)

        scaler.scale(loss).backward()

        if grad_clip is not None and grad_clip > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()

        if scheduler_name == "onecycle":
            scheduler.step()

        total_loss += float(loss.item())
        pbar.set_postfix(loss=f"{loss.item():.4f}")

        global_step = epoch * max(1, len(loader)) + step
        writer.add_scalar("train/loss", float(loss.item()), global_step)
        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)

    return total_loss / max(1, len(loader))



def validate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    num_classes: int,
    ignore_index: int,
) -> Tuple[float, Dict]:
    model.eval()
    total_loss = 0.0
    confusion = torch.zeros((num_classes, num_classes), device=device, dtype=torch.int64)

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validate"):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)

            logits = model(images)
            if logits.shape[-2:] != masks.shape[-2:]:
                logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)

            loss = loss_fn(logits, masks)
            total_loss += float(loss.item())

            preds = torch.argmax(logits, dim=1)
            confusion += compute_confusion_matrix(preds, masks, num_classes, ignore_index=ignore_index)

    metrics = metrics_from_confusion(confusion)
    return total_loss / max(1, len(loader)), metrics



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    ensure_dirs(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(cfg["train"].get("amp", True) and device.type == "cuda")

    train_loader, val_loader = build_loaders(cfg)

    model = SegFormerModel(
        backbone=cfg["model"]["backbone"],
        num_classes=int(cfg["model"]["num_classes"]),
        ignore_index=int(cfg["model"].get("ignore_index", 255)),
        use_pretrained=bool(cfg["model"].get("use_pretrained", True)),
    ).to(device)

    loss_fn = HybridLoss(
        ce_weight=float(cfg["loss"].get("ce_weight", 0.5)),
        dice_weight=float(cfg["loss"].get("dice_weight", 0.3)),
        focal_weight=float(cfg["loss"].get("focal_weight", 0.2)),
        ignore_index=int(cfg["model"].get("ignore_index", 255)),
        class_weights=cfg["loss"].get("class_weights"),
    )

    optimizer = AdamW(
        model.parameters(),
        lr=float(cfg["train"]["learning_rate"]),
        weight_decay=float(cfg["train"].get("weight_decay", 0.0)),
    )
    scheduler = get_scheduler(optimizer, cfg, steps_per_epoch=len(train_loader))
    scheduler_name = str(cfg["train"].get("scheduler", "cosine")).lower()

    writer = SummaryWriter(log_dir=cfg["paths"]["log_dir"])

    best_miou = -1.0
    epochs_without_improvement = 0
    patience = int(cfg["train"].get("early_stopping_patience", 10))

    for epoch in range(int(cfg["train"]["epochs"])):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            device=device,
            amp_enabled=amp_enabled,
            grad_clip=float(cfg["train"].get("grad_clip", 0.0)),
            epoch=epoch,
            writer=writer,
            scheduler_name=scheduler_name,
        )

        val_loss, val_metrics = validate(
            model=model,
            loader=val_loader,
            loss_fn=loss_fn,
            device=device,
            num_classes=int(cfg["model"]["num_classes"]),
            ignore_index=int(cfg["model"].get("ignore_index", 255)),
        )

        if scheduler_name != "onecycle":
            scheduler.step()

        miou = float(val_metrics["mean_iou"])
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/miou", miou, epoch)
        writer.add_scalar("val/pixel_accuracy", float(val_metrics["pixel_accuracy"]), epoch)

        if miou > best_miou:
            best_miou = miou
            epochs_without_improvement = 0
            ckpt_path = os.path.join(cfg["paths"]["checkpoint_dir"], "best_model.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "best_miou": best_miou,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": cfg,
                },
                ckpt_path,
            )
        else:
            epochs_without_improvement += 1

        if not bool(cfg["train"].get("save_best_only", True)):
            epoch_path = os.path.join(cfg["paths"]["checkpoint_dir"], f"epoch_{epoch:03d}.pt")
            torch.save({"epoch": epoch, "model": model.state_dict(), "config": cfg}, epoch_path)

        if epochs_without_improvement >= patience:
            break

        print(
            f"Epoch {epoch + 1}/{cfg['train']['epochs']} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_mIoU={miou:.4f}"
        )

    writer.close()


if __name__ == "__main__":
    main()
