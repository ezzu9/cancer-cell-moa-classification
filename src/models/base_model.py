"""Abstract base class for all MoA classification models."""

from abc import ABC, abstractmethod
from pathlib import Path

import torch
import torch.nn as nn


class BaseMoAModel(ABC, nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.pretrained = pretrained

    @abstractmethod
    def get_cam_target_layer(self) -> nn.Module:
        """Return the convolutional layer used for Grad-CAM."""

    def freeze_backbone(self) -> None:
        for name, param in self.named_parameters():
            if "classifier" not in name and "fc" not in name:
                param.requires_grad = False

    def unfreeze_all(self) -> None:
        for param in self.parameters():
            param.requires_grad = True

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.state_dict(), "num_classes": self.num_classes}, path)

    def load(self, path: Path, device: torch.device) -> None:
        checkpoint = torch.load(path, map_location=device)
        self.load_state_dict(checkpoint["state_dict"])

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
