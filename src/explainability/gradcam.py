"""
Grad-CAM and Guided Grad-CAM visualisation for MoA classification models.

Usage:
    python src/explainability/gradcam.py \
        --checkpoint outputs/models/resnet50_best.pth \
        --image_dir data/processed \
        --output_dir outputs/gradcam
"""

import argparse
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pytorch_grad_cam import GradCAM, GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from src.data.augmentation import get_val_transforms
from src.models.mobilenetv2 import MobileNetV2MoA
from src.models.resnet50 import ResNet50MoA

MODEL_REGISTRY = {"resnet50": ResNet50MoA, "mobilenetv2": MobileNetV2MoA}

MOA_CLASSES = [
    "Actin disruptors", "Aurora kinase inhibitors", "Cholesterol-lowering",
    "DNA damage", "DNA replication", "Eg5 kinesin inhibitors", "Epithelial",
    "Kinase inhibitors", "Microtubule destabilizers", "Microtubule stabilizers",
    "Protein degradation", "Protein synthesis",
]


def load_image(img_path: Path, image_size: int = 224):
    transform = get_val_transforms(image_size)
    rgb = np.array(Image.open(img_path).convert("RGB"))
    rgb_resized = cv2.resize(rgb, (image_size, image_size))
    tensor = transform(image=rgb_resized)["image"].unsqueeze(0)
    rgb_float = rgb_resized.astype(np.float32) / 255.0
    return tensor, rgb_float


def generate_gradcam(
    model,
    image_paths: List[Path],
    output_dir: Path,
    device: torch.device,
    target_class: Optional[int] = None,
    image_size: int = 224,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    target_layer = model.get_cam_target_layer()

    with GradCAM(model=model, target_layers=[target_layer]) as cam:
        for img_path in image_paths:
            tensor, rgb_float = load_image(img_path, image_size)
            tensor = tensor.to(device)

            with torch.no_grad():
                logits = model(tensor)
                pred_class = logits.argmax(1).item()
                confidence = F.softmax(logits, dim=1)[0, pred_class].item()

            targets = [ClassifierOutputTarget(target_class if target_class is not None else pred_class)]
            grayscale_cam = cam(input_tensor=tensor, targets=targets)[0]
            overlay = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)

            pred_name = MOA_CLASSES[pred_class].replace(" ", "_")
            out_name = f"{img_path.stem}_gradcam_{pred_name}_{confidence:.2f}.png"
            Image.fromarray(overlay).save(output_dir / out_name)
            print(f"  {img_path.name} → pred: {MOA_CLASSES[pred_class]} ({confidence:.1%})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", type=str, default="resnet50", choices=list(MODEL_REGISTRY))
    parser.add_argument("--image_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/gradcam"))
    parser.add_argument("--n_samples", type=int, default=5, help="Images per class to visualise")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ModelClass = MODEL_REGISTRY[args.model]
    model = ModelClass(num_classes=len(MOA_CLASSES)).to(device)
    model.load(args.checkpoint, device)
    model.eval()

    image_paths: List[Path] = []
    for class_dir in sorted(args.image_dir.iterdir()):
        if class_dir.is_dir():
            images = sorted(class_dir.glob("*.png"))[: args.n_samples]
            image_paths.extend(images)

    print(f"Generating Grad-CAM for {len(image_paths)} images...")
    generate_gradcam(model, image_paths, args.output_dir, device)
    print(f"\nSaved to {args.output_dir}")


if __name__ == "__main__":
    main()
