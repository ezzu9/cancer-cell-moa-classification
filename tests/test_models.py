"""Smoke tests for model forward passes and parameter counts."""

import torch
import pytest

from src.models.resnet50 import ResNet50MoA
from src.models.mobilenetv2 import MobileNetV2MoA


NUM_CLASSES = 12
BATCH = 2
IMAGE_SIZE = 224


@pytest.fixture
def dummy_batch():
    return torch.randn(BATCH, 3, IMAGE_SIZE, IMAGE_SIZE)


@pytest.mark.parametrize("ModelClass", [ResNet50MoA, MobileNetV2MoA])
def test_output_shape(ModelClass, dummy_batch):
    model = ModelClass(num_classes=NUM_CLASSES, pretrained=False)
    model.eval()
    with torch.no_grad():
        out = model(dummy_batch)
    assert out.shape == (BATCH, NUM_CLASSES)


@pytest.mark.parametrize("ModelClass", [ResNet50MoA, MobileNetV2MoA])
def test_freeze_reduces_trainable_params(ModelClass):
    model = ModelClass(num_classes=NUM_CLASSES, pretrained=False)
    total_before = model.count_parameters()
    model.freeze_backbone()
    total_after = model.count_parameters()
    assert total_after < total_before


@pytest.mark.parametrize("ModelClass", [ResNet50MoA, MobileNetV2MoA])
def test_cam_target_layer_exists(ModelClass):
    model = ModelClass(num_classes=NUM_CLASSES, pretrained=False)
    layer = model.get_cam_target_layer()
    assert layer is not None
