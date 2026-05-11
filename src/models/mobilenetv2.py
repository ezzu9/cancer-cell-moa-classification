"""MobileNetV2 transfer learning wrapper for MoA classification."""

import torch.nn as nn
from torchvision import models
from torchvision.models import MobileNet_V2_Weights

from src.models.base_model import BaseMoAModel


class MobileNetV2MoA(BaseMoAModel):
    def __init__(self, num_classes: int = 12, pretrained: bool = True, dropout: float = 0.3) -> None:
        super().__init__(num_classes, pretrained)
        weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.mobilenet_v2(weights=weights)

        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        in_features = backbone.classifier[1].in_features
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)

    def get_cam_target_layer(self) -> nn.Module:
        return self.features[-1][0]
