"""
Evaluation utilities: NSC accuracy, per-class F1, and confusion matrix.

Usage:
    python src/evaluation/metrics.py \
        --checkpoint outputs/models/resnet50_best.pth \
        --config configs/resnet50_config.yaml
"""

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from src.data.augmentation import get_val_transforms
from src.data.dataset import BBBC021Dataset, make_nsc_splits
from src.models.mobilenetv2 import MobileNetV2MoA
from src.models.resnet50 import ResNet50MoA

MODEL_REGISTRY = {"resnet50": ResNet50MoA, "mobilenetv2": MobileNetV2MoA}


def predict(model, loader, device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            preds = model(images).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    return np.array(all_preds), np.array(all_labels)


def nsc_accuracy(
    preds: np.ndarray,
    labels: np.ndarray,
    compound_ids: List[str],
) -> float:
    """
    Compound-level majority-vote accuracy (NSC protocol).
    Each compound gets one prediction = majority class among its images.
    """
    compound_votes: Dict[str, List[int]] = defaultdict(list)
    compound_true: Dict[str, int] = {}

    for pred, label, cid in zip(preds, labels, compound_ids):
        compound_votes[cid].append(int(pred))
        compound_true[cid] = int(label)

    correct = sum(
        max(set(votes), key=votes.count) == compound_true[cid]
        for cid, votes in compound_votes.items()
    )
    return correct / len(compound_votes)


def evaluate(cfg: dict) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    meta = pd.read_csv(cfg["metadata_csv"])
    moa_df = pd.read_csv(cfg["moa_csv"])
    merged = meta.merge(moa_df, on="compound", how="inner")
    _, _, test_compounds = make_nsc_splits(merged, seed=cfg["seed"])

    test_ds = BBBC021Dataset(
        cfg["processed_dir"], merged, get_val_transforms(cfg["image_size"]), test_compounds
    )
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])
    num_classes = len(test_ds.class_to_idx)

    ModelClass = MODEL_REGISTRY[cfg["model"]]
    model = ModelClass(num_classes=num_classes).to(device)
    model.load(Path(cfg["checkpoint"]), device)

    preds, labels = predict(model, test_loader, device)
    compound_ids = [s[0].parent.name + "_" + s[0].stem for s in test_ds.samples]

    acc = nsc_accuracy(preds, labels, compound_ids)
    print(f"\nNSC Accuracy: {acc:.4f} ({acc*100:.2f}%)")

    idx_to_class = {v: k for k, v in test_ds.class_to_idx.items()}
    target_names = [idx_to_class[i] for i in range(num_classes)]
    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=target_names))

    cm = confusion_matrix(labels, preds)
    print("Confusion matrix saved to outputs/figures/confusion_matrix.npy")
    np.save("outputs/figures/confusion_matrix.npy", cm)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg["checkpoint"] = str(args.checkpoint)

    evaluate(cfg)


if __name__ == "__main__":
    main()
