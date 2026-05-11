# Breast Cancer Cell Mechanism-of-Action Classification Using CNNs

<p align="center">
  <a href="https://breast-cancer-moa-classifier.streamlit.app" target="_blank">
    <img src="https://img.shields.io/badge/🚀%20Live%20Demo-Try%20the%20App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Live Demo"/>
  </a>
  &nbsp;
  <a href="https://huggingface.co/Ertaza/breast-cancer-moa-resnet50" target="_blank">
    <img src="https://img.shields.io/badge/🤗%20Hugging%20Face-ResNet50%20Model-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="Hugging Face Model"/>
  </a>
</p>

<p align="center">
  <a href="https://breast-cancer-moa-classifier.streamlit.app" target="_blank">
    <strong>👉 breast-cancer-moa-classifier.streamlit.app</strong>
  </a>
</p>

---

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19-orange?logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red?logo=keras&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![University](https://img.shields.io/badge/Anglia%20Ruskin%20University-BSc%20AI-navy)

> **Final Year Project** — Classifying the mechanism of action (MoA) of chemical compounds from fluorescence microscopy images of breast cancer cells, using CNNs, transfer learning, and Grad-CAM explainability.

**Student:** Muhammad Ertaza Manzoor | BSc Artificial Intelligence | Anglia Ruskin University

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Results](#results)
  - [Performance Summary](#performance-summary)
  - [Baseline CNN](#baseline-cnn)
  - [Expanded & Refined CNNs](#expanded--refined-cnns)
  - [ResNet50 Transfer Learning — Best Model](#resnet50-transfer-learning--best-model)
  - [MobileNetV2 Transfer Learning](#mobilenetv2-transfer-learning)
- [Grad-CAM Explainability](#grad-cam-explainability)
- [Key Techniques](#key-techniques)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
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

### Class Distribution

<p align="center">
  <img src="outputs/figures/final_moa_class_distribution.png" width="680" alt="MoA class distribution across the final dataset"/>
</p>

<p align="center"><em>Class distribution across the 788-sample dataset. DNA damage agents represent the dominant class, contributing to the class imbalance challenge addressed through class weighting.</em></p>

---

## Results

### Performance Summary

Five models were developed and evaluated. Results are reported on the held-out test set.

| # | Model | Accuracy | Macro F1 | Loss | Notes |
|---|---|---|---|---|---|
| 1 | **Baseline CNN** (3 conv layers) | 0.7140 | 0.174 | — | Heavy majority-class bias; poor minority class recall |
| 2 | CNN on Expanded Dataset (4 conv + BN) | 0.1390 | 0.049 | — | Collapsed to predicting only DNA damage class |
| 3 | Refined CNN (4 conv + GAP) | 0.1390 | 0.049 | — | Same class-collapse pattern as Model 2 |
| 4 | **ResNet50 Transfer Learning** ⭐ | **0.8734** | **0.8802** | **0.3818** | Best model — strong generalisation across all classes |
| 5 | MobileNetV2 Transfer Learning | 0.8228 | 0.7815 | 0.4604 | Competitive; ~5× fewer parameters than ResNet50 |

> **Key insight:** Models 2 & 3 demonstrate a critical failure mode — without class weighting, models collapse to predicting only the dominant class (DNA damage), yielding surface-level accuracy but near-zero F1 on all minority classes. Class weighting was the decisive fix that unlocked generalisation in the transfer learning models.

---

### Baseline CNN

The first model (3 convolutional layers, no class weighting) achieves 71.4% accuracy but with heavy bias toward the majority class. The confusion matrix reveals near-zero recall on minority MoA classes.

<p align="center">
  <img src="outputs/figures/confusion_matrix_baseline.png" width="520" alt="Baseline CNN confusion matrix"/>
</p>

<p align="center"><em>Baseline CNN confusion matrix — the model correctly classifies the dominant class but largely ignores minority MoA classes.</em></p>

<table align="center">
  <tr>
    <td align="center">
      <img src="outputs/figures/training_validation_accuracy_baseline.png" width="400" alt="Baseline CNN accuracy curves"/>
      <br/><em>Training vs validation accuracy</em>
    </td>
    <td align="center">
      <img src="outputs/figures/training_validation_loss_baseline.png" width="400" alt="Baseline CNN loss curves"/>
      <br/><em>Training vs validation loss</em>
    </td>
  </tr>
</table>

---

### Expanded & Refined CNNs

Models 2 and 3 (4 conv layers, batch normalisation, then GlobalAveragePooling2D) were trained on an expanded dataset but without class weighting. Both collapsed entirely to predicting a single class, producing near-identical degenerate confusion matrices.

<table align="center">
  <tr>
    <td align="center">
      <img src="outputs/figures/cnn_normalized_confusion_matrix.png" width="400" alt="Expanded CNN normalised confusion matrix"/>
      <br/><em>Expanded CNN — predicted only DNA damage class</em>
    </td>
    <td align="center">
      <img src="outputs/figures/refined_cnn_normalized_confusion_matrix.png" width="400" alt="Refined CNN normalised confusion matrix"/>
      <br/><em>Refined CNN — same class-collapse pattern</em>
    </td>
  </tr>
</table>

<table align="center">
  <tr>
    <td align="center">
      <img src="outputs/figures/cnn_training_validation_accuracy.png" width="400" alt="Expanded CNN accuracy"/>
      <br/><em>Expanded CNN — accuracy curves</em>
    </td>
    <td align="center">
      <img src="outputs/figures/refined_cnn_training_validation_accuracy.png" width="400" alt="Refined CNN accuracy"/>
      <br/><em>Refined CNN — accuracy curves</em>
    </td>
  </tr>
</table>

---

### ResNet50 Transfer Learning — Best Model

Fine-tuning a pre-trained ResNet50 backbone with class weighting and early stopping produced the best results: **87.34% accuracy and 0.8802 macro F1**. The model generalises well across all five MoA classes.

<p align="center">
  <img src="outputs/figures/resnet50_normalized_confusion_matrix.png" width="580" alt="ResNet50 normalised confusion matrix"/>
</p>

<p align="center"><em>ResNet50 normalised confusion matrix — strong diagonal signal across all 5 MoA classes, with only minor confusion between microtubule destabilizers and other classes.</em></p>

<table align="center">
  <tr>
    <td align="center">
      <img src="outputs/figures/resnet50_training_validation_accuracy.png" width="400" alt="ResNet50 accuracy curves"/>
      <br/><em>Training vs validation accuracy</em>
    </td>
    <td align="center">
      <img src="outputs/figures/resnet50_training_validation_loss.png" width="400" alt="ResNet50 loss curves"/>
      <br/><em>Training vs validation loss</em>
    </td>
  </tr>
</table>

<p align="center">
  <img src="outputs/figures/resnet50_sample_predictions.png" width="760" alt="ResNet50 sample predictions on test images"/>
</p>

<p align="center"><em>ResNet50 sample predictions on held-out test images — predicted class labels versus ground truth across all five MoA categories.</em></p>

---

### MobileNetV2 Transfer Learning

MobileNetV2 achieves **82.28% accuracy and 0.7815 macro F1** — competitive with ResNet50 at approximately one-fifth of the parameter count, making it a strong lightweight alternative.

<p align="center">
  <img src="outputs/figures/mobilenetv2_normalized_confusion_matrix.png" width="580" alt="MobileNetV2 normalised confusion matrix"/>
</p>

<p align="center"><em>MobileNetV2 normalised confusion matrix — good generalisation, though slightly weaker than ResNet50 on harder MoA classes.</em></p>

<table align="center">
  <tr>
    <td align="center">
      <img src="outputs/figures/mobilenetv2_training_validation_accuracy.png" width="400" alt="MobileNetV2 accuracy curves"/>
      <br/><em>Training vs validation accuracy</em>
    </td>
    <td align="center">
      <img src="outputs/figures/mobilenetv2_training_validation_loss.png" width="400" alt="MobileNetV2 loss curves"/>
      <br/><em>Training vs validation loss</em>
    </td>
  </tr>
</table>

<p align="center">
  <img src="outputs/figures/mobilenetv2_sample_predictions.png" width="760" alt="MobileNetV2 sample predictions on test images"/>
</p>

<p align="center"><em>MobileNetV2 sample predictions on held-out test images.</em></p>

---

## Grad-CAM Explainability

Gradient-weighted Class Activation Mapping (**Grad-CAM**) was applied to the final convolutional layer of the ResNet50 model to produce heatmaps that highlight which spatial regions of the cell image most influenced each prediction. This validates that the model has learned biologically meaningful features rather than image artefacts.

### Single Cell Example

<p align="center">
  <img src="outputs/figures/resnet50_gradcam_single_example.png" width="680" alt="Grad-CAM single cell example"/>
</p>

<p align="center"><em>Grad-CAM overlay on a single cell image — warm colours (red/yellow) indicate regions that most strongly activated the predicted MoA class.</em></p>

### Correctly Classified Samples

<p align="center">
  <img src="outputs/figures/resnet50_gradcam_correct_samples.png" width="760" alt="Grad-CAM on correctly classified samples"/>
</p>

<p align="center"><em>Grad-CAM heatmaps for correctly classified cells across multiple MoA classes. The model consistently attends to the relevant cellular compartment for each class — nucleus for DNA damage, microtubule network for tubulin-targeting compounds.</em></p>

### Random Sample Overview

<p align="center">
  <img src="outputs/figures/resnet50_gradcam_random_samples.png" width="760" alt="Grad-CAM on random test samples"/>
</p>

<p align="center"><em>Grad-CAM activations across a random sample of test images, showing consistent class-specific spatial attention.</em></p>

### Misclassified Samples

<p align="center">
  <img src="outputs/figures/resnet50_gradcam_misclassified_samples.png" width="760" alt="Grad-CAM on misclassified samples"/>
</p>

<p align="center"><em>Grad-CAM for misclassified samples — activations often highlight ambiguous or overlapping morphological features, providing interpretable insight into where the model goes wrong.</em></p>

### What the Heatmaps Reveal

| Predicted MoA | Highlighted Region | Biological Interpretation |
|---|---|---|
| Microtubule stabilizers | Tubulin channel (bundled fibres) | Model correctly focuses on microtubule morphology |
| Microtubule destabilizers | Diffuse tubulin signal | Absence of fibre structure is the discriminating feature |
| Eg5 inhibitors | Pericentric tubulin structure | Monopolar spindle visible in mitotic cells |
| DNA damage agents | DAPI channel (nucleus shape) | Nuclear fragmentation / enlarged nuclei |
| Aurora kinase inhibitors | Nuclear + spindle region | Aberrant mitotic figures |

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
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_cnn.ipynb
│   ├── 03_dataset_expansion.ipynb
│   ├── 04_data_preprocessing.ipynb
│   ├── 05a_cnn_training.ipynb
│   ├── 05b_cnn_training_refined.ipynb
│   ├── 06a_resnet50_transfer_learning.ipynb
│   ├── 06b_mobilenetv2_transfer_learning.ipynb
│   ├── 07_grad_cam.ipynb
│   └── 08_single_image_inference.ipynb
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
│   ├── resnet50_config.yaml
│   └── mobilenetv2_config.yaml
├── outputs/
│   ├── models/                     # Saved model weights (.keras)
│   ├── logs/                       # Training history
│   ├── figures/                    # All result plots (committed)
│   └── gradcam/                    # Grad-CAM overlay images
├── tests/
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
2. Download the weekly image zip files and extract into `data/raw/`
3. Download `BBBC021_v1_image.csv` and `BBBC021_v1_moa.csv` into `data/metadata/`

### 4. Run the Notebooks (Recommended)

Run notebooks in order for a complete walkthrough from raw data to Grad-CAM:

```bash
jupyter notebook notebooks/
```

### 5. Train a Model Directly

```bash
# Best model — ResNet50 transfer learning
python src/training/trainer.py --config configs/resnet50_config.yaml

# Lightweight alternative — MobileNetV2
python src/training/trainer.py --config configs/mobilenetv2_config.yaml
```

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
