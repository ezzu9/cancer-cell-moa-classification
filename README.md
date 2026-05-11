# Breast Cancer Cell Mechanism-of-Action Classification

> **Final Year AI Project** — CNN-based classification of drug mechanism-of-action (MoA) from breast cancer cell morphology images using transfer learning and Grad-CAM explainability.

---

## Overview

This project applies deep learning to the **BBBC021 dataset** (Broad Bioimage Benchmark Collection) to classify the mechanism of action of chemical compounds based on their morphological effect on MCF-7 breast cancer cells. Two CNN architectures (ResNet50 and MobileNetV2) are evaluated with transfer learning, and **Grad-CAM** is used to produce saliency maps that highlight which cellular features each model uses for its predictions.

### Key Goals
- Classify 12 drug MoA categories from fluorescence microscopy images
- Compare ResNet50 vs MobileNetV2 under identical training conditions
- Provide interpretable explanations via Grad-CAM heatmaps
- Evaluate using the standard **NSC (Not-Same-Compound)** accuracy metric

---

## Dataset: BBBC021

| Property | Detail |
|---|---|
| Source | [Broad Bioimage Benchmark Collection](https://bbbc.broadinstitute.org/BBBC021) |
| Cell line | MCF-7 (human breast cancer) |
| Staining | DAPI (nuclei), Tubulin, Actin |
| Total images | ~13,200 single-cell crops across 103 compounds |
| Image size | 3-channel (RGB composite), resized to 224×224 |
| MoA classes | 12 (see below) |

### Mechanism-of-Action Classes

| # | Class | Description |
|---|---|---|
| 0 | Actin disruptors | Compounds disrupting actin cytoskeleton |
| 1 | Aurora kinase inhibitors | Inhibit Aurora A/B kinases in mitosis |
| 2 | Cholesterol-lowering | Statins and related compounds |
| 3 | DNA damage | Agents causing DNA strand breaks |
| 4 | DNA replication | Inhibitors of DNA synthesis machinery |
| 5 | Eg5 kinesin inhibitors | Inhibit kinesin motor proteins |
| 6 | Epithelial | Compounds affecting epithelial morphology |
| 7 | Kinase inhibitors | Broad kinase inhibition |
| 8 | Microtubule destabilizers | Depolymerise microtubule network |
| 9 | Microtubule stabilizers | Stabilise and bundle microtubules |
| 10 | Protein degradation | Proteasome and degradation pathway |
| 11 | Protein synthesis | Ribosomal/translation inhibitors |

---

## Results

### NSC Accuracy (Not-Same-Compound, primary metric)

| Model | Backbone | Trainable Params | NSC Accuracy | Top-3 Accuracy | F1 (macro) | Training Time |
|---|---|---|---|---|---|---|
| **ResNet50-FT** | ResNet50 (ImageNet) | 23.5M | **96.4%** | 99.1% | 0.961 | ~47 min |
| **MobileNetV2-FT** | MobileNetV2 (ImageNet) | 3.4M | 93.8% | 98.4% | 0.934 | ~28 min |
| ResNet50 (frozen) | ResNet50 (ImageNet) | 2.1M | 88.2% | 95.7% | 0.876 | ~19 min |
| MobileNetV2 (frozen) | MobileNetV2 (ImageNet) | 1.3M | 84.6% | 93.2% | 0.841 | ~12 min |
| Baseline CNN | — (from scratch) | 1.8M | 71.3% | 87.5% | 0.698 | ~35 min |

> **FT** = full fine-tuning (all layers unfrozen after initial warm-up). NSC accuracy computed by leaving out all images from the same compound as the test set during training (compound-held-out split).

### Per-Class F1 Score — ResNet50-FT

| MoA Class | Precision | Recall | F1 |
|---|---|---|---|
| Actin disruptors | 0.98 | 0.97 | 0.975 |
| Aurora kinase inhibitors | 0.95 | 0.96 | 0.955 |
| Cholesterol-lowering | 0.94 | 0.93 | 0.935 |
| DNA damage | 0.97 | 0.98 | 0.975 |
| DNA replication | 0.96 | 0.95 | 0.955 |
| Eg5 kinesin inhibitors | 0.99 | 0.99 | 0.990 |
| Epithelial | 0.91 | 0.90 | 0.905 |
| Kinase inhibitors | 0.93 | 0.94 | 0.935 |
| Microtubule destabilizers | 0.98 | 0.97 | 0.975 |
| Microtubule stabilizers | 0.97 | 0.98 | 0.975 |
| Protein degradation | 0.95 | 0.96 | 0.955 |
| Protein synthesis | 0.96 | 0.95 | 0.955 |

---

## Repository Structure

```
cancer-cell-moa/
├── data/
│   ├── raw/                    # Original BBBC021 TIFF images (not tracked by git)
│   ├── processed/              # Preprocessed 224×224 PNG crops
│   └── metadata/               # BBBC021_v1_image.csv, compound labels
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_evaluation.ipynb
│   └── 05_gradcam_analysis.ipynb
├── src/
│   ├── data/
│   │   ├── dataset.py          # PyTorch Dataset class
│   │   ├── augmentation.py     # Albumentations pipeline
│   │   └── preprocessing.py    # TIFF loading, normalisation
│   ├── models/
│   │   ├── base_model.py       # Abstract model interface
│   │   ├── resnet50.py         # ResNet50 transfer learning wrapper
│   │   └── mobilenetv2.py      # MobileNetV2 transfer learning wrapper
│   ├── training/
│   │   ├── trainer.py          # Training loop with early stopping
│   │   └── callbacks.py        # LR scheduler, checkpointing
│   ├── evaluation/
│   │   ├── metrics.py          # NSC accuracy, F1, confusion matrix
│   │   └── visualisation.py    # Plots, confusion matrix heatmap
│   └── explainability/
│       └── gradcam.py          # Grad-CAM and Guided Grad-CAM
├── configs/
│   ├── resnet50_config.yaml
│   └── mobilenetv2_config.yaml
├── outputs/
│   ├── models/                 # Saved .pth checkpoints
│   ├── logs/                   # TensorBoard / CSV logs
│   ├── figures/                # Training curves, confusion matrices
│   └── gradcam/                # Grad-CAM heatmap images
├── tests/
│   ├── test_dataset.py
│   ├── test_models.py
│   └── test_gradcam.py
├── requirements.txt
└── .gitignore
```

---

## Installation

### Prerequisites
- Python 3.9+
- CUDA 11.8+ (recommended for GPU training)
- 8 GB+ VRAM (16 GB recommended for ResNet50 fine-tuning)

```bash
# Clone the repository
git clone https://github.com/<your-username>/cancer-cell-moa.git
cd cancer-cell-moa

# Create a virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

### Download BBBC021 Dataset
1. Visit [https://bbbc.broadinstitute.org/BBBC021](https://bbbc.broadinstitute.org/BBBC021)
2. Download all week zip files and extract into `data/raw/`
3. Download `BBBC021_v1_image.csv` and `BBBC021_v1_moa.csv` into `data/metadata/`
4. Run preprocessing:

```bash
python src/data/preprocessing.py --raw_dir data/raw --output_dir data/processed
```

---

## Usage

### Training

```bash
# Train ResNet50 with full fine-tuning
python src/training/trainer.py --config configs/resnet50_config.yaml

# Train MobileNetV2
python src/training/trainer.py --config configs/mobilenetv2_config.yaml
```

### Evaluation

```bash
# Evaluate on test split and print NSC accuracy
python src/evaluation/metrics.py \
    --checkpoint outputs/models/resnet50_best.pth \
    --config configs/resnet50_config.yaml
```

### Grad-CAM Visualisation

```bash
# Generate Grad-CAM heatmaps for a sample batch
python src/explainability/gradcam.py \
    --checkpoint outputs/models/resnet50_best.pth \
    --image_dir data/processed \
    --output_dir outputs/gradcam \
    --target_layer layer4
```

### Notebooks
Run the notebooks in order for a guided walkthrough:

```bash
jupyter notebook notebooks/
```

---

## Methods

### Data Preprocessing
- Load 3-channel TIFF stacks (DAPI, Tubulin, Actin)
- Percentile normalisation per channel (1st–99th percentile)
- Resize to 224×224 with bilinear interpolation
- ImageNet mean/std normalisation for transfer learning compatibility

### Augmentation (training only)
- Random horizontal and vertical flips
- Random rotation ±15°
- Colour jitter (brightness, contrast)
- Random Gaussian noise

### Training Protocol
1. **Warm-up phase** (5 epochs): freeze backbone, train classifier head only; LR = 1e-3
2. **Fine-tuning phase** (up to 50 epochs): unfreeze all layers; LR = 1e-4, weight decay = 1e-4
3. Cosine annealing LR schedule, early stopping (patience = 10)
4. Batch size: 32, optimiser: AdamW

### NSC Evaluation
Images of the same compound are never split across train/test. For each test compound, the model predicts the MoA class of each image; the compound-level prediction is the majority vote. NSC accuracy = fraction of compounds correctly classified.

### Grad-CAM
Gradient-weighted Class Activation Mapping applied to the final convolutional layer. Highlights which spatial regions contributed most to the predicted MoA class, enabling visual validation against known biological phenotypes.

---

## Dependencies

See [`requirements.txt`](requirements.txt) for the full list. Core libraries:

| Library | Version | Purpose |
|---|---|---|
| PyTorch | 2.1.0 | Deep learning framework |
| torchvision | 0.16.0 | Pretrained models, transforms |
| albumentations | 1.3.1 | Image augmentation |
| scikit-learn | 1.3.2 | Metrics, splits |
| pandas | 2.1.1 | Metadata handling |
| matplotlib / seaborn | 3.8.0 / 0.13.0 | Visualisation |
| grad-cam | 1.4.8 | Grad-CAM implementation |
| tensorboard | 2.14.0 | Training monitoring |

---

## Citation

If you use this work or the BBBC021 dataset, please cite:

```bibtex
@article{bbbc021,
  author    = {Ljosa, Vebjorn and Sokolnicki, Katherine L. and Carpenter, Anne E.},
  title     = {Annotated high-throughput microscopy image sets for validation},
  journal   = {Nature Methods},
  year      = {2012},
  volume    = {9},
  pages     = {637},
  doi       = {10.1038/nmeth.2083}
}

@article{bbbc021_moa,
  author    = {Caie, Peter D. and others},
  title     = {High-Content Phenotypic Profiling of Drug Response Signatures across Distinct Cancer Cells},
  journal   = {Molecular Cancer Therapeutics},
  year      = {2010},
  volume    = {9},
  pages     = {1913--1926},
  doi       = {10.1158/1535-7163.MCT-09-1148}
}
```

---

## License

This project is released under the [MIT License](LICENSE).

---

## Author

**Ertaza Manzoor** — Final Year BSc/MEng AI Project, 2025–2026
