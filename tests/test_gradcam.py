"""Smoke tests for Grad-CAM output shape and value range."""

import numpy as np
import torch
import pytest

from src.models.resnet50 import ResNet50MoA
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


NUM_CLASSES = 12


def test_gradcam_output_shape():
    model = ResNet50MoA(num_classes=NUM_CLASSES, pretrained=False)
    model.eval()
    target_layer = model.get_cam_target_layer()
    dummy = torch.randn(1, 3, 224, 224)

    with GradCAM(model=model, target_layers=[target_layer]) as cam:
        targets = [ClassifierOutputTarget(0)]
        grayscale = cam(input_tensor=dummy, targets=targets)

    assert grayscale.shape == (1, 224, 224)


def test_gradcam_values_in_range():
    model = ResNet50MoA(num_classes=NUM_CLASSES, pretrained=False)
    model.eval()
    target_layer = model.get_cam_target_layer()
    dummy = torch.randn(1, 3, 224, 224)

    with GradCAM(model=model, target_layers=[target_layer]) as cam:
        grayscale = cam(input_tensor=dummy, targets=[ClassifierOutputTarget(0)])

    assert grayscale.min() >= 0.0
    assert grayscale.max() <= 1.0 + 1e-6
