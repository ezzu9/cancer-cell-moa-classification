# Breast Cancer Cell Mechanism-of-Action Classification Using CNNs

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red?logo=keras&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![University](https://img.shields.io/badge/Anglia%20Ruskin%20University-BSc%20AI-navy)

> **Final Year Project** — Classifying the mechanism of action (MoA) of chemical compounds from fluorescence microscopy images of breast cancer cells, using CNNs, transfer learning, and Grad-CAM explainability.

**Student:** Muhammad Ertaza Manzoor | BSc Artificial Intelligence | Anglia Ruskin University

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model Results](#model-results)
- [Key Techniques](#key-techniques)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Grad-CAM Explainability](#grad-cam-explainability)
- [Dependencies](#dependencies)
- [Citation](#citation)

---

## Overview

This project applies deep learning to the **BBBC021 dataset** to classify the mechanism of action of chemical compounds based on their morphological effect on **MCF-7 breast cancer cells**. Each compound induces a distinct cellular phenotype visible through fluorescence microscopy — this project trains CNN models to recognise those phenotypes automatically.

The project follows a progressive model development strategy, from a simple baseline CNN through to fine-tuned transfer learning models, revealing how architecture choice and dataset size dramatically impact performance on a small, imbalanced biomedical dataset.

**Best result: ResNet50 Transfer Learning — 87.34% accuracy, 0.8802 macro F1**

---

## Dataset

| Property | Detail |
|---|---|
| Source | [Broad Bioimage Benchmark Collection — BBBC021](https://bbbc.broadinstitute.org/BBBC021) |
| Cell line | MCF-7 (human breast cancer) |
| Total samples | **788** |
| Image size | **128 × 128 pixels** |
| Channels | DAPI (nucleus), Tubulin (microtubules), Actin (cytoskeleton) |
| Input format | 3-channel stacked image (RGB-style composite) |
| Classes | **5 MoA categories** |

### MoA Classes

| Class | Biological Effect |
|---|---|
| Aurora kinase inhibitors | Disrupt mitotic spindle assembly; cause abnormal cell division |
| DNA damage agents | Induce DNA strand breaks; trigger cell cycle arrest |
| Eg5 inhibitors | Block kinesin motor proteins; produce monopolar spindles |
| Microtubule destabilizers | Depolymerise the microtubule network; collapse the cytoskeleton |
| Microtubule stabilizers | Stabilise and bundle microtubules; prevent depolymerisation |

### Channel Composite

| Channel | Stain | Cellular Structure |
|---|---|---|
| Channel 1 | DAPI | Nucleus (DNA) |
| Channel 2 | Tubulin antibody | Microtubule network |
| Channel 3 | Phalloidin | Actin cytoskeleton |

The three channels are stacked into a single 128×128×3 image to match the input format expected by ImageNet-pretrained transfer learning models.

---

## Model Results

Five models were developed and evaluated. Results are reported on the held-out test set.

| # | Model | Accuracy | Macro F1 | Loss | Notes |
|---|---|---|---|---|---|
| 1 | **Baseline CNN** (3 conv layers) | 0.7140 | 0.174 | — | Heavy majority-class bias; poor minority class recall |
| 2 | CNN on Expanded Dataset (4 conv + BN) | 0.1390 | 0.049 | — | Collapsed to predicting only DNA damage class |
| 3 | Refined CNN (4 conv + GAP) | 0.1390 | 0.049 | — | Same class-collapse pattern as Model 2 |
| 4 | **ResNet50 Transfer Learning** | **0.8734** | **0.8802** | **0.3818** | Best model — strong generalisation across all classes |
| 5 | MobileNetV2 Transfer Learning | 0.8228 | 0.7815 | 0.4604 | Competitive; ~5× fewer parameters than ResNet50 |

### Key Observations

- **Models 2 & 3** highlight a critical failure mode: when class imbalance is severe and the model is not constrained, it collapses to predicting only the dominant class (DNA damage), yielding high accuracy but near-zero F1 on all other classes.
- **Class weighting** was the decisive intervention that enabled Models 4 and 5 to generalise across all five MoA categories.
- **Transfer learning** from ImageNet proved highly effective despite the domain gap (natural images → fluorescence microscopy), because low-level edge and texture features are reusable.
- **ResNet50 outperforms MobileNetV2** (by ~5 percentage points in macro F1), at the cost of more parameters and longer training time.

---

## Key Techniques

| Technique | Purpose |
|---|---|
| **Transfer Learning** (ResNet50, MobileNetV2) | Leverage ImageNet features to compensate for small dataset size |
| **Class Weighting** | Address severe class imbalance; prevent majority-class collapse |
| **Batch Normalisation** | Stabilise training; reduce internal covariate shift |
| **GlobalAveragePooling2D** | Reduce spatial dimensions without overfitting via large dense layers |
| **Dropout** | Regularise fully-connected layers; reduce overfitting |
| **Early Stopping** | Halt training when validation loss plateaus; prevent overtraining |
| **Learning Rate Reduction** | `ReduceLROnPlateau` — decrease LR when validation loss stalls |
| **Data Expansion** | Augment minority classes to partially correct class imbalance |
| **Grad-CAM** | Produce saliency maps; validate that the model attends to biologically relevant cell structures |

---

## Project Structure

```
cancer-cell-moa/
├── data/
│   ├── raw/                        # Original BBBC021 TIFF images (not tracked)
│   ├── processed/                  # Stacked 128×128 3-channel PNG images
│   └── metadata/                   # BBBC021 label CSVs
├── notebooks/
│   ├── 01_data_exploration.ipynb   # Dataset statistics and class distribution
│   ├── 02_preprocessing.ipynb      # Channel stacking and normalisation
│   ├── 03_model_training.ipynb     # Model training and learning curves
│   ├── 04_evaluation.ipynb         # Test-set evaluation and confusion matrices
│   └── 05_gradcam_analysis.ipynb   # Grad-CAM heatmap generation
├── src/
│   ├── data/
│   │   ├── preprocessing.py        # TIFF loading, channel stacking, resizing
│   │   ├── dataset.py              # Data pipeline and train/val/test splits
│   │   └── augmentation.py         # Image augmentation pipeline
│   ├── models/
│   │   ├── base_model.py           # Abstract model interface
│   │   ├── resnet50.py             # ResNet50 transfer learning wrapper
│   │   └── mobilenetv2.py          # MobileNetV2 transfer learning wrapper
│   ├── training/
│   │   ├── trainer.py              # Training loop with callbacks
│   │   └── callbacks.py            # Early stopping, LR reduction
│   ├── evaluation/
│   │   ├── metrics.py              # Accuracy, macro F1, confusion matrix
│   │   └── visualisation.py        # Training curves, confusion matrix plots
│   └── explainability/
│       └── gradcam.py              # Grad-CAM heatmap generation
├── configs/
│   ├── resnet50_config.yaml        # Hyperparameters for ResNet50
│   └── mobilenetv2_config.yaml     # Hyperparameters for MobileNetV2
├── outputs/
│   ├── models/                     # Saved model weights (.h5 / .keras)
│   ├── logs/                       # Training history CSVs
│   ├── figures/                    # Plots and confusion matrices
│   └── gradcam/                    # Grad-CAM overlay images
├── tests/
│   ├── test_dataset.py
│   ├── test_models.py
│   └── test_gradcam.py
├── requirements.txt
└── .gitignore
```

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/ezzu9/cancer-cell-moa-classification.git
cd cancer-cell-moa-classification
```

### 2. Set Up the Environment

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

### 3. Download the BBBC021 Dataset

1. Go to [https://bbbc.broadinstitute.org/BBBC021](https://bbbc.broadinstitute.org/BBBC021)
2. Download the weekly image zip files and extract them into `data/raw/`
3. Download `BBBC021_v1_image.csv` and `BBBC021_v1_moa.csv` into `data/metadata/`

### 4. Preprocess Images

```bash
python src/data/preprocessing.py \
    --raw_dir data/raw \
    --output_dir data/processed \
    --size 128
```

This stacks the DAPI, Tubulin, and Actin channels into 128×128×3 images organised by MoA class.

### 5. Train a Model

```bash
# Best model — ResNet50 transfer learning
python src/training/trainer.py --config configs/resnet50_config.yaml

# Lightweight alternative — MobileNetV2
python src/training/trainer.py --config configs/mobilenetv2_config.yaml
```

### 6. Evaluate

```bash
python src/evaluation/metrics.py \
    --checkpoint outputs/models/resnet50_best.keras \
    --config configs/resnet50_config.yaml
```

### 7. Run Notebooks (Recommended)

For a full guided walkthrough, run the notebooks in order:

```bash
jupyter notebook notebooks/
```

---

## Grad-CAM Explainability

Gradient-weighted Class Activation Mapping (**Grad-CAM**) was used to interpret what spatial features each model uses when classifying a cell image.

Heatmaps are overlaid onto the original fluorescence image to highlight the cellular regions with the highest gradient signal for the predicted MoA class.

```bash
python src/explainability/gradcam.py \
    --checkpoint outputs/models/resnet50_best.keras \
    --image_dir data/processed \
    --output_dir outputs/gradcam
```

### What the heatmaps reveal

| Predicted MoA | Highlighted Region | Biological Interpretation |
|---|---|---|
| Microtubule stabilizers | Tubulin channel (bundled fibres) | Model correctly focuses on microtubule morphology |
| Microtubule destabilizers | Diffuse tubulin signal | Absence of fibre structure is the discriminating feature |
| Eg5 inhibitors | Pericentric tubulin structure | Monopolar spindle visible in mitotic cells |
| DNA damage agents | DAPI channel (nucleus shape) | Nuclear fragmentation / enlarged nuclei |
| Aurora kinase inhibitors | Nuclear + spindle region | Aberrant mitotic figures |

Grad-CAM provides confidence that the model has learned biologically meaningful features rather than image artefacts or background noise.

---

## Dependencies

```
tensorflow>=2.12
numpy
pandas
opencv-python
matplotlib
scikit-learn
jupyter
tqdm
PyYAML
```

Install all dependencies with:

```bash
pip install -r requirements.txt
```

---

## Citation

If you use this work or the BBBC021 dataset, please cite:

```bibtex
@article{bbbc021,
  author  = {Ljosa, Vebjorn and Sokolnicki, Katherine L. and Carpenter, Anne E.},
  title   = {Annotated high-throughput microscopy image sets for validation},
  journal = {Nature Methods},
  year    = {2012},
  volume  = {9},
  pages   = {637},
  doi     = {10.1038/nmeth.2083}
}

@article{caie2010,
  author  = {Caie, Peter D. and others},
  title   = {High-Content Phenotypic Profiling of Drug Response Signatures across Distinct Cancer Cells},
  journal = {Molecular Cancer Therapeutics},
  year    = {2010},
  volume  = {9},
  pages   = {1913--1926},
  doi     = {10.1158/1535-7163.MCT-09-1148}
}
```

---

## License

This project is released under the [MIT License](LICENSE).

---

*Final Year Project — BSc Artificial Intelligence, Anglia Ruskin University, 2025–2026*
*Muhammad Ertaza Manzoor*
