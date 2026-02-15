"""Tests for terrain segmentation project."""

import argparse
import sys
from typing import Dict

import numpy as np
import torch
import yaml
from torch import nn

from models.segformer import SegFormerModel
from utils.augmentations import build_transforms
from utils.dataset import SegmentationDataset
from utils.losses import HybridLoss
from utils.metrics import compute_confusion_matrix, metrics_from_confusion


def load_config(path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_config_loading() -> bool:
    """Test that configuration can be loaded."""
    print("Testing config loading...")
    try:
        cfg = load_config("configs/config.yaml")
        assert "model" in cfg
        assert "paths" in cfg
        assert "train" in cfg
        print("  ✓ Config loaded successfully")
        return True
    except Exception as e:
        print(f"  ✗ Config loading failed: {e}")
        return False


def test_model_creation(cfg: Dict) -> bool:
    """Test that model can be created."""
    print("Testing model creation...")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SegFormerModel(
            backbone=cfg["model"]["backbone"],
            num_classes=int(cfg["model"]["num_classes"]),
            ignore_index=int(cfg["model"].get("ignore_index", 255)),
            use_pretrained=bool(cfg["model"].get("use_pretrained", True)),
        ).to(device)
        
        # Test forward pass
        dummy_input = torch.randn(1, 3, 512, 512).to(device)
        with torch.no_grad():
            output = model(dummy_input)
        
        assert output is not None
        assert output.shape[1] == int(cfg["model"]["num_classes"])
        print(f"  ✓ Model created successfully, output shape: {output.shape}")
        return True
    except Exception as e:
        print(f"  ✗ Model creation failed: {e}")
        return False


def test_dataset_loading(cfg: Dict) -> bool:
    """Test that dataset can be loaded."""
    print("Testing dataset loading...")
    try:
        val_tf = build_transforms(False, cfg["train"]["image_size"], cfg.get("augmentation", {}))
        val_ds = SegmentationDataset(cfg, split="val", transforms=val_tf)
        
        assert len(val_ds) > 0
        sample = val_ds[0]
        
        assert "image" in sample
        assert sample["image"].shape[0] == 3  # RGB channels
        print(f"  ✓ Dataset loaded successfully, {len(val_ds)} samples, image shape: {sample['image'].shape}")
        return True
    except Exception as e:
        print(f"  ✗ Dataset loading failed: {e}")
        return False


def test_transforms(cfg: Dict) -> bool:
    """Test that transforms work correctly."""
    print("Testing transforms...")
    try:
        train_tf = build_transforms(True, cfg["train"]["image_size"], cfg.get("augmentation", {}))
        val_tf = build_transforms(False, cfg["train"]["image_size"], cfg.get("augmentation", {}))
        
        # Create dummy image and mask
        dummy_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        dummy_mask = np.random.randint(0, 4, (512, 512), dtype=np.int64)
        
        # Test train transforms (with augmentation)
        transformed = train_tf(image=dummy_image, mask=dummy_mask)
        assert transformed["image"].shape[-2:] == (cfg["train"]["image_size"], cfg["train"]["image_size"])
        
        # Test val transforms (without augmentation)
        transformed = val_tf(image=dummy_image, mask=dummy_mask)
        assert transformed["image"].shape[-2:] == (cfg["train"]["image_size"], cfg["train"]["image_size"])
        
        print("  ✓ Transforms work correctly")
        return True
    except Exception as e:
        print(f"  ✗ Transforms failed: {e}")
        return False


def test_loss_computation(cfg: Dict) -> bool:
    """Test that loss computation works."""
    print("Testing loss computation...")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        loss_fn = HybridLoss(
            ce_weight=float(cfg["loss"].get("ce_weight", 0.5)),
            dice_weight=float(cfg["loss"].get("dice_weight", 0.3)),
            focal_weight=float(cfg["loss"].get("focal_weight", 0.2)),
            ignore_index=int(cfg["model"].get("ignore_index", 255)),
            class_weights=cfg["loss"].get("class_weights"),
        )
        
        # Create dummy predictions and targets
        num_classes = int(cfg["model"]["num_classes"])
        dummy_logits = torch.randn(2, num_classes, 256, 256).to(device)
        dummy_targets = torch.randint(0, num_classes, (2, 256, 256)).to(device)
        
        loss = loss_fn(dummy_logits, dummy_targets)
        
        assert loss is not None
        assert loss.item() >= 0
        print(f"  ✓ Loss computation works, loss value: {loss.item():.4f}")
        return True
    except Exception as e:
        print(f"  ✗ Loss computation failed: {e}")
        return False


def test_metrics_computation(cfg: Dict) -> bool:
    """Test that metrics computation works."""
    print("Testing metrics computation...")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_classes = int(cfg["model"]["num_classes"])
        ignore_index = int(cfg["model"].get("ignore_index", 255))
        
        # Create dummy predictions and targets
        dummy_preds = torch.randint(0, num_classes, (2, 256, 256)).to(device)
        dummy_targets = torch.randint(0, num_classes, (2, 256, 256)).to(device)
        
        confusion = compute_confusion_matrix(dummy_preds, dummy_targets, num_classes, ignore_index=ignore_index)
        
        assert confusion.shape == (num_classes, num_classes)
        
        metrics = metrics_from_confusion(confusion)
        
        assert "mean_iou" in metrics
        assert "pixel_accuracy" in metrics
        print(f"  ✓ Metrics computation works, mIoU: {metrics['mean_iou']:.4f}, pixel accuracy: {metrics['pixel_accuracy']:.4f}")
        return True
    except Exception as e:
        print(f"  ✗ Metrics computation failed: {e}")
        return False


def test_full_pipeline(cfg: Dict) -> bool:
    """Test full inference pipeline."""
    print("Testing full pipeline...")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create model
        model = SegFormerModel(
            backbone=cfg["model"]["backbone"],
            num_classes=int(cfg["model"]["num_classes"]),
            ignore_index=int(cfg["model"].get("ignore_index", 255)),
            use_pretrained=bool(cfg["model"].get("use_pretrained", True)),
        ).to(device)
        model.eval()
        
        # Create transforms
        val_tf = build_transforms(False, cfg["train"]["image_size"], cfg.get("augmentation", {}))
        
        # Create dataset
        val_ds = SegmentationDataset(cfg, split="val", transforms=val_tf)
        
        # Get a sample
        sample = val_ds[0]
        image = sample["image"].unsqueeze(0).to(device)
        
        # Run inference
        with torch.no_grad():
            logits = model(image)
            pred = torch.argmax(logits, dim=1)
        
        assert pred.shape == (1, cfg["train"]["image_size"], cfg["train"]["image_size"])
        print(f"  ✓ Full pipeline works, prediction shape: {pred.shape}")
        return True
    except Exception as e:
        print(f"  ✗ Full pipeline failed: {e}")
        return False


def main() -> int:
    """Run all tests."""
    parser = argparse.ArgumentParser(description="Run tests for terrain segmentation project.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--skip-model", action="store_true", help="Skip model creation test (requires downloading pretrained weights)")
    args = parser.parse_args()
    
    print("=" * 50)
    print("Running terrain segmentation tests")
    print("=" * 50)
    
    # Load config
    cfg = load_config(args.config)
    
    results = []
    
    # Run tests
    results.append(("Config loading", test_config_loading()))
    results.append(("Transforms", test_transforms(cfg)))
    results.append(("Dataset loading", test_dataset_loading(cfg)))
    results.append(("Loss computation", test_loss_computation(cfg)))
    results.append(("Metrics computation", test_metrics_computation(cfg)))
    
    if not args.skip_model:
        results.append(("Model creation", test_model_creation(cfg)))
        results.append(("Full pipeline", test_full_pipeline(cfg)))
    else:
        print("\nSkipping model creation test (--skip-model flag)")
    
    # Print summary
    print("\n" + "=" * 50)
    print("Test Summary")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  {name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
