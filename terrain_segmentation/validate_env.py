import argparse
import importlib
import os
import sys
from typing import Dict, List

import yaml


REQUIRED_PACKAGES = [
    "torch",
    "torchvision",
    "transformers",
    "albumentations",
    "cv2",
    "numpy",
    "matplotlib",
    "tensorboard",
    "yaml",
    "tqdm",
]


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def check_packages() -> List[str]:
    missing = []
    for pkg in REQUIRED_PACKAGES:
        try:
            importlib.import_module(pkg)
        except Exception:
            missing.append(pkg)
    return missing


def check_dir(path: str, label: str, errors: List[str]) -> None:
    if not path:
        errors.append(f"Missing config value for {label}.")
        return
    if not os.path.isdir(path):
        errors.append(f"Directory not found for {label}: {path}")


def check_train_paths(cfg: Dict, errors: List[str]) -> None:
    check_dir(cfg["paths"].get("train_images"), "paths.train_images", errors)
    check_dir(cfg["paths"].get("train_masks"), "paths.train_masks", errors)
    check_dir(cfg["paths"].get("val_images"), "paths.val_images", errors)
    check_dir(cfg["paths"].get("val_masks"), "paths.val_masks", errors)


def check_eval_paths(cfg: Dict, split: str, errors: List[str]) -> None:
    check_dir(cfg["paths"].get(f"{split}_images"), f"paths.{split}_images", errors)
    check_dir(cfg["paths"].get(f"{split}_masks"), f"paths.{split}_masks", errors)

    ckpt_dir = cfg["paths"].get("checkpoint_dir")
    if not ckpt_dir or not os.path.isdir(ckpt_dir):
        errors.append(f"Checkpoint directory not found: {ckpt_dir}")
        return

    ckpt_path = os.path.join(ckpt_dir, "best_model.pt")
    if not os.path.isfile(ckpt_path):
        errors.append(f"Checkpoint missing: {ckpt_path}")


def check_inference_paths(cfg: Dict, errors: List[str]) -> None:
    check_dir(cfg["paths"].get("test_images"), "paths.test_images", errors)

    ckpt_dir = cfg["paths"].get("checkpoint_dir")
    if not ckpt_dir or not os.path.isdir(ckpt_dir):
        errors.append(f"Checkpoint directory not found: {ckpt_dir}")
        return

    ckpt_path = os.path.join(ckpt_dir, "best_model.pt")
    if not os.path.isfile(ckpt_path):
        errors.append(f"Checkpoint missing: {ckpt_path}")


def check_placeholder_paths(cfg: Dict, errors: List[str]) -> None:
    for key, value in cfg.get("paths", {}).items():
        if isinstance(value, str) and "/absolute/path/to/" in value:
            errors.append(f"Config path still placeholder at paths.{key}: {value}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Environment/config validation for terrain segmentation.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "evaluate", "inference"])
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    args = parser.parse_args()

    errors: List[str] = []

    if not os.path.isfile(args.config):
        print(f"ERROR: Config file not found: {args.config}")
        return 1

    cfg = load_config(args.config)

    missing_packages = check_packages()
    if missing_packages:
        errors.append("Missing Python packages: " + ", ".join(missing_packages))

    check_placeholder_paths(cfg, errors)

    output_dir = cfg["paths"].get("output_dir")
    log_dir = cfg["paths"].get("log_dir")
    checkpoint_dir = cfg["paths"].get("checkpoint_dir")

    if output_dir and not os.path.isdir(output_dir):
        print(f"WARN: output_dir does not exist yet (will be created by train): {output_dir}")
    if log_dir and not os.path.isdir(log_dir):
        print(f"WARN: log_dir does not exist yet (will be created by train): {log_dir}")
    if checkpoint_dir and not os.path.isdir(checkpoint_dir):
        print(f"WARN: checkpoint_dir does not exist yet (will be created by train): {checkpoint_dir}")

    if args.mode == "train":
        check_train_paths(cfg, errors)
    elif args.mode == "evaluate":
        check_eval_paths(cfg, args.split, errors)
    else:
        check_inference_paths(cfg, errors)

    if errors:
        print("\nValidation failed:")
        for err in errors:
            print(f"- {err}")
        return 1

    print("Validation passed: environment and config look ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
