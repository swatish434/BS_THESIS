# PCB-Vision: A Multiscene RGB-Hyperspectral Benchmark Dataset of Printed Circuit Boards
[HZDR](https://hzdr.de) - [Hif_Exploration](https://www.iexplo.space/)

## Overview

Our primary focus is to enhance the non-invasive optical analysis of E-waste materials, specifically plastics and printed circuit boards (PCBs). We aim to develop a smart multisensor network that utilizes RGB cameras and hyperspectral imaging, along with other types of sensors, to improve the efficiency of the E-waste recycling industry.

![Workflow](https://github.com/Elias-Arbash/PCBVision/blob/main/images/workflow2.png)

## Research Paper

This GitHub repository corresponds to the research paper titled "PCB-Vision: A Multiscene RGB-Hyperspectral Benchmark Dataset of Printed Circuit Boards." The paper introduces the first RGB-Hyperspectral Imaging (HSI) benchmark segmentation dataset for PCBs. You can access the paper [here](https://arxiv.org/abs/2401.06528).

## Dataset Details

The dataset includes:
- **RGB images**: 53 PCBs scanned with a high-resolution RGB camera (Teledyne Dalsa C4020).
- **Hyperspectral Data**: 53 hyperspectral data cubes scanned with Specim FX10 in the VNIR range.
- **Ground Truth**: 'General' and 'Monoseg' masks for 4 classes: 'Background', 'Component', 'IC', 'Connectors'.

## Project Structure

This repository is organized as follows:

- `Results/`: **(New)** Stores all trained models, visualization images, and evaluation metrics.
- `models/`: Contains model definitions (UNet, DeepLabv3+, LinkNet, etc.).
- `train_rgb.py`: Main script for training RGB segmentation models.
- `train_hsi_pca.py`: Script for training HSI segmentation models using PCA features.
- `visualize_results.py`: Generates side-by-side comparisons and Grad-CAM visualizations.
- `evaluate_models.py`: Benchmarks all trained models and generates performance tables.
- `PCBVision_Explainer/`: Source code for the interactive results website.

## Installation

1.  Clone the repository.
2.  Install dependencies:
    ```bash
    pip install -r Requirements.txt
    ```
    *Note: Requires `torch`, `albumentations`, `spectral`, `matplotlib`, `opencv-python`, `pandas`.*

## Usage Guide

### 1. Training RGB Models
You can train various architectures by specifying the `--model` argument. The trained models will be saved to `Results/`.

```bash
# Train UNet (Default)
python train_rgb.py --model unet --epochs 100

# Train DeepLabv3+
python train_rgb.py --model deeplabv3+ --epochs 100

# Other options: linknet, attention_unet, resunet
```

### 2. Training HSI Models
To train the Hyperspectral model (using PCA-reduced data):
```bash
python train_hsi_pca.py
```

### 3. Visualization & Interpretation (Grad-CAM)
Generate side-by-side comparisons of your trained models against the Ground Truth and HSI model. This script also generates **Grad-CAM** heatmaps to explain *why* the model made its decision.

```bash
python visualize_results.py
```
**Output**: 
- Saves images to `Results/comparison_RGB_{MODEL}_HSI.png`.
- Automatically updates the Explainer Website if available.
- Features: Original RGB | GT | RGB Pred | HSI Pred | Grad-CAM (Components) | Grad-CAM (ICs).

### 4. Benchmarking & Evaluation
To evaluate all trained models found in the `Results/` directory and generate a comparison table:

```bash
python evaluate_models.py
```
**Output**: 
- `Results/evaluation_metrics.csv`: Detailed metrics.
- `Results/evaluation_metrics.png`: Visual table of performance.


## Data Access

To utilize the dataset, download it from the Rodare website: [Rodare](https://rodare.hzdr.de/record/2704), or from Zenodo: [Zenodo](https://zenodo.org/records/10617721).

## Citation

If you use this dataset or code, please cite:

Arbash, Elias, et al. (2024). PCB-Vision: A Multiscene RGB-Hyperspectral Benchmark Dataset of Printed Circuit Boards. arXiv preprint arXiv:2401.06528.

```latex
@article{arbash2024pcb,
  title={PCB-Vision: A Multiscene RGB-Hyperspectral Benchmark Dataset of Printed Circuit Boards},
  author={Arbash, Elias and Fuchs, Margret and Rasti, Behnood and Lorenz, Sandra and Ghamisi, Pedram and Gloaguen, Richard},
  journal={arXiv preprint arXiv:2401.06528},
  year={2024}
}
```

## License

Apache-2.0 License.
