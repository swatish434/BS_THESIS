# GIQA: Generated Image Quality Assessment for Augmented Images

## Overview

GIQA (Generated Image Quality Assessment) is a comprehensive toolkit for evaluating the quality of synthetic images generated through augmentation techniques like **Copy-Paste** and **CutMix**.

## Quality Metrics

### 1. No-Reference Metrics (Don't need original images)

| Metric | Description | Range | Interpretation |
|--------|-------------|-------|----------------|
| **NIQE** | Natural Image Quality Evaluator | 0-20 (lower is better) | Measures naturalness of image statistics |
| **BRISQUE** | Blind Referenceless Image Spatial Quality | 0-100 (lower is better) | Evaluates perceptual quality |
| **Inception Score** | Quality + Diversity measure | 0-∞ (higher is better) | Uses pre-trained CNN features |
| **Artifact Score** | Detection of augmentation artifacts | 0-1 (lower is better) | Custom detector for paste/cut boundaries |

### 2. Reference-Based Metrics (Need original images)

| Metric | Description | Range | Interpretation |
|--------|-------------|-------|----------------|
| **FID** | Frechet Inception Distance | 0-∞ (lower is better) | Distance between real and fake distributions |
| **LPIPS** | Learned Perceptual Image Patch Similarity | 0-1 (lower = more similar) | Perceptual similarity between images |

### 3. Custom Artifact Detection

| Artifact Type | Detection Method | Relevance |
|--------------|------------------|-----------|
| **Boundary Artifacts** | Edge density, gradient variance | Copy-Paste edges |
| **Texture Inconsistency** | Local texture variance, color histograms | Mixed regions |
| **CutMix Artifacts** | Rectangular contour detection, sharp transitions | CutMix boundaries |

## Installation

```bash
pip install torch torchvision numpy opencv-python scipy pillow tqdm matplotlib
```

## Quick Start

### Basic Usage

```bash
# Evaluate Copy-Paste augmented images
python giqa_evaluation.py \
    --augmented_dir ./augmented_copypaste \
    --augmentation_type copypaste \
    --output_dir ./giqa_results

# Evaluate CutMix augmented images with FID
python giqa_evaluation.py \
    --augmented_dir ./augmented_cutmix \
    --augmentation_type cutmix \
    --reference_dir ./original_images \
    --output_dir ./giqa_results
```

### Compare Multiple Augmentation Methods

```python
from giqa_evaluation import GIQAEvaluator

# Initialize evaluator
evaluator = GIQAEvaluator()

# Evaluate different methods
results = {
    'original': evaluator.evaluate_dataset('./original_images', 'none'),
    'copypaste': evaluator.evaluate_dataset('./augmented_copypaste', 'copypaste'),
    'cutmix': evaluator.evaluate_dataset('./augmented_cutmix', 'cutmix')
}

# Compare
comparison = evaluator.compare_augmentation_methods(results)
print(f"Best method: {comparison['best_method']}")
print(f"Best score: {comparison['best_score']:.2f}/100")
```

### Single Image Quality Assessment

```python
from giqa_evaluation import GIQAEvaluator
import numpy as np
from PIL import Image

# Load image
img = np.array(Image.open('augmented_image.png')) / 255.0

# Initialize evaluator
evaluator = GIQAEvaluator()

# Evaluate single image
results = evaluator.evaluate_single_image(img, aug_type='copypaste')

print(f"NIQE Score: {results['niqe']:.4f}")
print(f"BRISQUE Score: {results['brisque']:.4f}")
print(f"Artifact Score: {results['artifacts']['overall_artifact_score']:.4f}")
```

## Understanding the Results

### Overall Quality Score (0-100)

The overall quality score combines all metrics:

```
Score = 0.15 × NIQE_norm + 
         0.15 × BRISQUE_norm + 
         0.20 × IS_norm + 
         0.15 × LPIPS_norm + 
         0.20 × Artifact_norm + 
         0.15 × CLIP_norm
```

### Sample Output

```
======================================================================
GIQA EVALUATION RESULTS
======================================================================
Total images evaluated: 50

Quality Metrics:
  Overall Quality Score: 72.45/100
  NIQE (↓ better): 4.2341
  BRISQUE (↓ better): 28.5432
  Inception Score (↑ better): 12.8765
  Artifact Score (↓ better): 0.3421
  LPIPS Diversity: 0.2134
  FID vs Original: 45.6789

======================================================================
COMPARISON TABLE
======================================================================
| Metric            | original | copypaste | cutmix  |
|-------------------|----------|-----------|---------|
| Overall Score     | 78.32    | 72.45     | 68.91   |
| NIQE ↓            | 3.4521   | 4.2341    | 5.1234  |
| BRISQUE ↓         | 22.34    | 28.54     | 35.67   |
| Inception Score ↑ | 15.43    | 12.87     | 10.23   |
| Artifact Score ↓  | 0.12     | 0.34      | 0.45    |
| FID ↓             | 0.00     | 45.67     | 62.34   |
```

## Artifact Detection Details

### Copy-Paste Specific Artifacts

```python
artifacts = detector.detect_boundary_artifacts(img)
# Returns:
# {
#     'line_density': 2.34,        # Straight line density
#     'gradient_variance': 1234.5,  # Edge sharpness variance
#     'edge_density': 0.05          # Overall edge density
# }
```

### CutMix Specific Artifacts

```python
artifacts = detector.detect_cutmix_artifacts(img)
# Returns:
# {
#     'rectangular_contours': 3,    # Number of rectangular regions
#     'horizontal_transition': 45.2, # Sharpness of horizontal cut
#     'vertical_transition': 52.1,   # Sharpness of vertical cut
#     'has_cutmix_artifacts': True   # Binary detection
# }
```

### Texture Inconsistency

```python
artifacts = detector.detect_texture_inconsistency(img)
# Returns:
# {
#     'texture_consistency': 0.45,  # Lower = more consistent
#     'hue_peaks': 5,               # Number of dominant colors
#     'saturation_peaks': 3,        # Color saturation clusters
#     'mean_texture_var': 1234.5    # Average local texture variance
# }
```

## Best Practices

### 1. Use Multiple Metrics

```python
# Don't rely on a single metric
# Combine no-reference and reference-based metrics for complete picture

results = evaluator.evaluate_dataset(
    image_dir='./augmented',
    aug_type='copypaste',
    reference_dir='./original',  # Enables FID
    batch_size=32
)
```

### 2. Artifact Detection for Augmentation Tuning

```python
# Use artifact scores to tune augmentation parameters

for paste_prob in [0.3, 0.5, 0.7, 0.9]:
    augmented = generate_copypaste(prob=paste_prob)
    results = evaluator.evaluate_dataset(augmented, 'copypaste')
    print(f"Paste prob {paste_prob}: Artifact score = {results['metrics']['artifacts']['mean']:.4f}")
```

### 3. Compare Against Baseline

```python
# Always compare augmented images against original baseline

baseline = evaluator.evaluate_dataset('./original', 'none')
augmented = evaluator.evaluate_dataset('./augmented', 'copypaste')

print(f"Quality change: {augmented['overall_quality_score'] - baseline['overall_quality_score']:.2f}")
```

### 4. Batch Processing for Large Datasets

```python
# For large datasets, process in batches and save incremental results

import json

all_results = []
for batch in range(0, total_images, batch_size):
    batch_images = load_batch(batch, batch_size)
    for img in batch_images:
        result = evaluator.evaluate_single_image(img, 'copypaste')
        all_results.append(result)
    
    # Save intermediate results
    with open('results_partial.json', 'w') as f:
        json.dump(all_results, f)
```

## Interpreting Scores

### Good Quality Indicators

| Metric | Good Range | Excellent Range |
|--------|------------|-----------------|
| NIQE | < 5 | < 3 |
| BRISQUE | < 30 | < 20 |
| Inception Score | > 10 | > 20 |
| Artifact Score | < 0.3 | < 0.15 |
| FID | < 50 | < 20 |
| Overall | > 65 | > 80 |

### Common Issues and Solutions

| Issue | Symptoms | Solution |
|-------|----------|----------|
| Visible paste boundaries | High artifact score, high gradient variance | Blend edges, use Poisson blending |
| Texture mismatch | High texture inconsistency | Match texture statistics before pasting |
| Color inconsistency | Multiple hue/saturation peaks | Color harmonization post-processing |
| Low diversity | Low Inception Score, low LPIPS | Increase augmentation randomness |

## Example Workflow

```bash
# Step 1: Generate augmented images
python example_giqa_usage.py \
    --data_dir ./my_images \
    --output_dir ./giqa_output \
    --num_augmentations 50

# Step 2: Analyze results
# Check giqa_output/giqa_complete_results.json
# Check giqa_output/comparison_table.md

# Step 3: Visualize artifacts for specific images
# Check giqa_output/artifact_visualizations/

# Step 4: Tune augmentation based on findings
```

## Output Files

```
giqa_output/
├── giqa_complete_results.json    # Full results in JSON
├── comparison_table.md           # Markdown comparison table
├── artifact_visualizations/      # Visual analysis of artifacts
│   ├── artifact_analysis_img1.png
│   └── ...
├── augmented_copypaste/          # Generated Copy-Paste images
├── augmented_cutmix/             # Generated CutMix images
└── sample_original/              # Original images (if generated)
```

## Troubleshooting

### Out of Memory

```python
# Reduce batch size
results = evaluator.evaluate_dataset(
    image_dir='./images',
    batch_size=8  # Reduce from default 32
)
```

### Slow Processing

```python
# Use GPU and optimize data loading
device = torch.device('cuda')  # Ensure GPU
results = evaluator.evaluate_dataset(
    image_dir='./images',
    batch_size=64  # Larger batch for GPU
)
```

### Inconsistent Results

```python
# Ensure images are properly normalized (0-1 range)
img = np.array(Image.open(path)) / 255.0

# Check image format
print(f"Image shape: {img.shape}, range: [{img.min()}, {img.max()}]")
```

## Citation

If you use this GIQA toolkit, please cite the relevant metric papers:

- **NIQE**: Mittal et al., "Making a 'Completely Blind' Image Quality Analyzer", IEEE Signal Processing Letters, 2013
- **BRISQUE**: Mittal et al., "No-Reference Image Quality Assessment in the Spatial Domain", TIP, 2012
- **FID**: Heusel et al., "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium", NeurIPS, 2017
- **LPIPS**: Zhang et al., "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric", CVPR, 2018
