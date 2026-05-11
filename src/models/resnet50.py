"""ResNet50 transfer learning wrapper for MoA classification."""

import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights

from src.models.base_model import BaseMoAModel


class ResNet50MoA(BaseMoAModel):
    def __init__(self, num_classes: int = 12, pretrained: bool = True, dropout: float = 0.5) -> None:
        super().__init__(num_classes, pretrained)
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.resnet50(weights=weights)

        self.features = nn.Sequential(*list(backbone.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d(1)
        in_features = backbone.fc.in_features
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
        return self.features[-1][-1].conv3
