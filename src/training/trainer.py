"""
Training loop with warm-up + fine-tuning phases, early stopping, and TensorBoard logging.

Usage:
    python src/training/trainer.py --config configs/resnet50_config.yaml
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.data.augmentation import get_train_transforms, get_val_transforms
from src.data.dataset import BBBC021Dataset, make_nsc_splits
from src.models.mobilenetv2 import MobileNetV2MoA
from src.models.resnet50 import ResNet50MoA


MODEL_REGISTRY = {
    "resnet50": ResNet50MoA,
    "mobilenetv2": MobileNetV2MoA,
}


def build_loaders(cfg: dict):
    meta = pd.read_csv(cfg["metadata_csv"])
    moa_df = pd.read_csv(cfg["moa_csv"])
    merged = meta.merge(moa_df, on="compound", how="inner")

    train_c, val_c, test_c = make_nsc_splits(
        merged, val_fraction=cfg["val_fraction"], test_fraction=cfg["test_fraction"], seed=cfg["seed"]
    )

    train_ds = BBBC021Dataset(cfg["processed_dir"], merged, get_train_transforms(cfg["image_size"]), train_c)
    val_ds = BBBC021Dataset(cfg["processed_dir"], merged, get_val_transforms(cfg["image_size"]), val_c)

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"], pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"], pin_memory=True)
    return train_loader, val_loader


def run_epoch(model, loader, criterion, optimiser, device, train: bool):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for images, labels in tqdm(loader, leave=False):
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)

            if train:
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def train(cfg: dict) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    train_loader, val_loader = build_loaders(cfg)
    num_classes = len(train_loader.dataset.class_to_idx)

    ModelClass = MODEL_REGISTRY[cfg["model"]]
    model = ModelClass(num_classes=num_classes, pretrained=True).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    writer = SummaryWriter(log_dir=Path(cfg["log_dir"]) / cfg["run_name"])
    best_val_acc, patience_counter = 0.0, 0
    checkpoint_path = Path(cfg["checkpoint_dir"]) / f"{cfg['run_name']}_best.pth"

    # ── Phase 1: warm-up (frozen backbone) ──────────────────────────────────
    model.freeze_backbone()
    optimiser = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg["warmup_lr"])
    print(f"Warm-up phase: {cfg['warmup_epochs']} epochs")
    for epoch in range(cfg["warmup_epochs"]):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimiser, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimiser, device, train=False)
        writer.add_scalars("loss", {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalars("acc", {"train": train_acc, "val": val_acc}, epoch)
        print(f"[WU {epoch+1:02d}] train_acc={train_acc:.4f}  val_acc={val_acc:.4f}")

    # ── Phase 2: full fine-tuning ────────────────────────────────────────────
    model.unfreeze_all()
    optimiser = AdamW(model.parameters(), lr=cfg["finetune_lr"], weight_decay=cfg["weight_decay"])
    scheduler = CosineAnnealingLR(optimiser, T_max=cfg["max_epochs"])
    print(f"\nFine-tuning phase: up to {cfg['max_epochs']} epochs")

    for epoch in range(cfg["max_epochs"]):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimiser, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimiser, device, train=False)
        scheduler.step()

        global_ep = cfg["warmup_epochs"] + epoch
        writer.add_scalars("loss", {"train": train_loss, "val": val_loss}, global_ep)
        writer.add_scalars("acc", {"train": train_acc, "val": val_acc}, global_ep)
        print(f"[FT {epoch+1:03d}] train_acc={train_acc:.4f}  val_acc={val_acc:.4f}  lr={scheduler.get_last_lr()[0]:.2e}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save(checkpoint_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg["patience"]:
                print(f"Early stopping at epoch {epoch+1}. Best val_acc={best_val_acc:.4f}")
                break

    writer.close()
    print(f"\nBest val accuracy: {best_val_acc:.4f}. Checkpoint saved to {checkpoint_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train(cfg)


if __name__ == "__main__":
    main()
