# SNAP: Spectral Neuro-Symbolic Aided PCB Segmentation

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4.1-orange)](https://pytorch.org/)
[![Dataset: PCB-Vision](https://img.shields.io/badge/Dataset-PCB--Vision-green)](https://zenodo.org/records/10617721)

> **SNAP** is a neuro-symbolic fusion framework for accurate segmentation of Printed Circuit Board (PCB) components, combining deep learning-based RGB and hyperspectral image segmentation with deterministic spectral library matching to support automated, non-invasive electronic waste (e-waste) recycling.

---

## Table of Contents

- [Overview](#overview)
- [Motivation](#motivation)
- [Method](#method)
  - [RGB Segmentation: DeepLabV3+](#rgb-segmentation-deeplabv3)
  - [Hyperspectral Segmentation: HybridSSRNViT](#hyperspectral-segmentation-hybridssrnvit)
  - [Spectral Matching](#spectral-matching)
  - [Neuro-Symbolic Fusion](#neuro-symbolic-fusion)
  - [Data Augmentation](#data-augmentation)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Citation](#citation)
- [License](#license)

---

## Overview

Electronic waste is projected to reach **82 million tonnes by 2030**, making automated, non-invasive PCB component segmentation critical for efficient recycling pipelines. SNAP addresses this challenge through a multi-modal neuro-symbolic fusion strategy that achieves state-of-the-art performance on the PCB-Vision benchmark.

**Key results on the PCB-Vision benchmark:**

| Metric | SNAP (Ours) | Previous SOTA | Improvement |
|--------|------------|---------------|-------------|
| Mean F1-Score | **0.9437** | — | **+21.8%** |
| mIoU | **0.9267** | — | **+40.9%** |

---

## Motivation

Existing AI-driven, image-based PCB segmentation methods suffer from:

- Poor generalisation across RGB-only or HSI-only modalities
- Severe class imbalance in PCB component datasets
- Lack of multi-modal fusion that leverages complementary spectral and spatial information
- Structural hallucinations from naive generative augmentation

SNAP tackles all of these limitations with a principled neuro-symbolic architecture and a rigorous, quality-gated augmentation pipeline.

---

## Method

SNAP integrates three segmentation streams and fuses them using a neuro-symbolic strategy:

```
RGB Image ──────────► DeepLabV3+  ──────────────────────┐
                                                         │
HSI Data  ──────────► HybridSSRNViT ────────────────────►  Neuro-Symbolic Fusion ──► Final Segmentation Map
                                                         │
HSI Data  ──────────► Spectral Matching ────────────────┘
```

### RGB Segmentation: DeepLabV3+

**DeepLabV3+** is used for semantic segmentation of RGB imagery. Its **Atrous Spatial Pyramid Pooling (ASPP)** module enables multi-scale spatial feature extraction, capturing fine-grained component boundaries at varying receptive fields without loss of resolution.

### Hyperspectral Segmentation: HybridSSRNViT

**HybridSSRNViT** is a novel hybrid architecture that combines:

- **Spectral–Spatial Residual Networks (SSRN)** — for extracting local spectral-spatial patterns from HSI data cubes
- **Vision Transformers (ViT)** — for capturing long-range global dependencies across the hyperspectral image

This dual-path design bridges local spectral detail with holistic scene understanding, making it well-suited for discriminating visually similar PCB components that differ primarily in spectral signature.

### Spectral Matching

An auxiliary segmentation is derived from the HSI modality using a **deterministic spectral library matching** approach. This classical, physics-driven branch acts as a reliable symbolic prior, constraining the fusion and preventing deep learning components from overfitting to noisy training samples.

### Neuro-Symbolic Fusion

Segmentation maps from all three streams (DeepLabV3+, HybridSSRNViT, and Spectral Matching) are fused via a **neuro-symbolic fusion strategy** that combines the statistical power of neural predictions with the interpretability and determinism of spectral rule-based outputs. This fusion is the central novelty of SNAP and is responsible for the substantial gains over prior SOTA.

### Data Augmentation

To address **severe class imbalance** in the PCB-Vision dataset, three augmentation strategies are employed:

1. **Copy-Paste** — minority-class component instances are copied and pasted into training images
2. **CutMix** — random rectangular patches are mixed between images to improve boundary generalisation
3. **Three-Phase Generative Inpainting Pipeline** — a novel pipeline combining:
   - **Stable Diffusion** for high-fidelity image synthesis
   - **ControlNet** for structural conditioning
   - **LoRA fine-tuning** for domain-specific adaptation
   
   All generated samples are gated by a **multi-metric Generative Image Quality Assessment (GIQA)** module to prevent structural hallucinations and ensure only structurally valid augmentations enter training.

---

## Results

SNAP achieves a **mean F1-score of 0.9437** and **mIoU of 0.9267** on the PCB-Vision benchmark dataset — improvements of **+21.8%** and **+40.9%** respectively over the prior state of the art.

Detailed per-class metrics, training logs, and visualisations are available in the [`04_Logs_and_Results/`](./04_Logs_and_Results/) directory.

---

## Repository Structure

```
SNAP-Spectral-Neuro-Symbolic-Aided-PCB-Segmentation/
│
├── 02_Experiments/             # Ablation studies, baseline comparisons, and exploratory notebooks
│
├── 03_Core/                    # Core model implementations
│   ├── deeplabv3plus/          #   DeepLabV3+ RGB segmentation model
│   ├── hybridssrnvit/          #   HybridSSRNViT hyperspectral segmentation model
│   ├── spectral_matching/      #   Spectral library matching module
│   ├── fusion/                 #   Neuro-symbolic fusion logic
│   └── augmentation/           #   Copy-Paste, CutMix, and SD-LoRA augmentation pipelines
│
├── 04_Logs_and_Results/        # Training logs, evaluation metrics, and visualisations
│
├── 05_Docs/                    # Additional documentation and figures
│
├── GIQA/                       # Generative Image Quality Assessment gating module
│
├── color_folders.py            # Utility for colour-coded folder visualisations
├── requirements.txt            # Core Python dependencies
├── requirements_sdlora.txt     # Additional dependencies for SD1.5 + LoRA augmentation
├── Citation                    # Citation information for the PCB-Vision dataset
├── LICENSE                     # Apache 2.0 License
└── .gitignore
```

---

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- `git`

### Step 1 — Clone the Repository

```bash
git clone https://github.com/swatish434/SNAP-Spectral-Neuro-Symbolic-Aided-PCB-Segmentation-.git
cd SNAP-Spectral-Neuro-Symbolic-Aided-PCB-Segmentation-
```

### Step 2 — Install Core Dependencies

```bash
pip install -r requirements.txt
```

**Core packages include:**

| Package | Version |
|---------|---------|
| `torch` | 2.4.1 |
| `torchvision` | 0.19.1 |
| `numpy` | 1.24.4 |
| `opencv-python` | 4.12.0.88 |
| `scikit-learn` | 1.3.2 |
| `scikit-image` | 0.21.0 |
| `spectral` | 0.24 |
| `hylite` | 1.36 |
| `pillow` | 10.4.0 |
| `scipy` | 1.10.1 |
| `matplotlib` | 3.7.5 |
| `pandas` | 2.0.3 |

### Step 3 — Install SD + LoRA Dependencies (Optional — for Generative Augmentation)

If you wish to run the Stable Diffusion + LoRA augmentation pipeline:

```bash
pip install -r requirements_sdlora.txt
```

**Additional packages include:**

| Package | Version |
|---------|---------|
| `diffusers` | 0.26.0 |
| `transformers` | 4.37.0 |
| `accelerate` | 0.26.0 |
| `peft` | 0.8.0 |
| `huggingface_hub` | 0.20.0 |
| `xformers` | ≥0.0.23 |

> **Note:** Use your existing PyTorch installation; the SD-LoRA requirements file does not reinstall torch.

---

## Usage

### Training RGB Segmentation (DeepLabV3+)

```bash
python 03_Core/train_rgb.py \
    --model deeplabv3plus \
    --epochs 100 \
    --data_root /path/to/pcb_vision_dataset
```

### Training Hyperspectral Segmentation (HybridSSRNViT)

```bash
python 03_Core/train_hsi.py \
    --model hybridssrnvit \
    --epochs 100 \
    --data_root /path/to/pcb_vision_dataset
```

### Running Spectral Matching

```bash
python 03_Core/spectral_matching/run_matching.py \
    --hsi_dir /path/to/hsi_data \
    --library /path/to/spectral_library
```

### Running Neuro-Symbolic Fusion

```bash
python 03_Core/fusion/fuse.py \
    --rgb_preds /path/to/rgb_predictions \
    --hsi_preds /path/to/hsi_predictions \
    --spectral_preds /path/to/spectral_predictions \
    --output /path/to/output
```

### Running the GIQA-Gated Augmentation Pipeline

```bash
python 03_Core/augmentation/sd_lora_pipeline.py \
    --data_root /path/to/training_data \
    --giqa_threshold 0.85
```

### Evaluating Results

```bash
python 03_Core/evaluate.py \
    --preds /path/to/fusion_output \
    --gt /path/to/ground_truth \
    --output 04_Logs_and_Results/
```

---

## Dataset

SNAP is evaluated on the **PCB-Vision** benchmark — the first RGB-Hyperspectral benchmark segmentation dataset for printed circuit boards.

**Dataset contents:**

- **53 RGB images** captured with a Teledyne Dalsa C4020 high-resolution camera
- **53 Hyperspectral data cubes** captured with a Specim FX10 sensor (VNIR range)
- **Ground truth masks** for 4 classes: `Background`, `Component`, `IC`, `Connectors`
- Both `General` and `Monoseg` annotation variants

**Download:**

| Source | Link |
|--------|------|
| Zenodo (primary) | [https://zenodo.org/records/10617721](https://zenodo.org/records/10617721) |
| Rodare | [https://rodare.hzdr.de/record/2704](https://rodare.hzdr.de/record/2704) |
| Google Drive (organised) | [Drive link](https://drive.google.com/drive/folders/1RHmvNbwwPDYvDzqgMTqA6bv9tLyVOIOS?usp=drive_link) |

If you use the PCB-Vision dataset, please also cite the original dataset paper:

> Arbash, E., et al. *"PCB-Vision: A Multiscene RGB-Hyperspectral Benchmark Dataset of Printed Circuit Boards."* arXiv:2401.06528. [https://arxiv.org/abs/2401.06528](https://arxiv.org/abs/2401.06528)

---

## Citation

If you use SNAP in your research, please cite:

```bibtex
@article{snap_pcb_2025,
  title   = {SNAP: Spectral Neuro-Symbolic Aided PCB Segmentation},
  author  = {[Authors]},
  journal = {[Venue]},
  year    = {2025},
  url     = {https://github.com/swatish434/SNAP-Spectral-Neuro-Symbolic-Aided-PCB-Segmentation-}
}
```

> **Note:** Please update the citation with the full author list and venue details once the paper is published.

---

## Keywords

Deep learning · Electronic waste recycling · PCB segmentation · Hyperspectral imaging · RGB · Neuro-symbolic fusion · Spectral library matching · Generative augmentation · DeepLabV3+ · Vision Transformer · LoRA

---

## License

This project is licensed under the **Apache License 2.0**. See the [LICENSE](./LICENSE) file for details.

---

*For questions, issues, or contributions, please open a GitHub Issue or submit a Pull Request.*
