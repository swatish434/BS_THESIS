# Loss Functions Architecture - Complete Overview

## Overview
Your training pipeline uses multiple loss functions across different experiments. Here's a comprehensive breakdown:

---

## 1. Implemented Loss Functions

### Location: `utils/loss_functions.py`

#### A. **FocalLoss**
- **Purpose**: Address class imbalance by down-weighting easy examples
- **Formula**: `FL = -α(1 - p_t)^γ * log(p_t)`
- **Parameters**:
  - `alpha` (tensor): Per-class weights [C,] - Used for class balancing
  - `gamma` (float, default=2): Focusing parameter - Higher = more focus on hard examples
  - `reduction` (str, default='mean'): Loss reduction method
  - `ignore_index` (int, default=255): Ignore specific class index

**Key Features**:
- Automatically down-weights easy-to-classify pixels (high confidence)
- Up-weights hard examples (low confidence / misclassified)
- Supports class-specific weights via `alpha` parameter

**Use Case**: Severe class imbalance (your IC/Connector classes)

---

#### B. **DiceLoss**
- **Purpose**: Optimize IoU directly (better for segmentation than pixel-wise CE)
- **Formula**: `Dice = 1 - (2 * |X ∩ Y| + ε) / (|X| + |Y| + ε)`
- **Parameters**:
  - `smooth` (float, default=1e-6): Smoothing factor to avoid division by zero
  - `reduction` (str, default='mean'): Loss reduction method
  - `ignore_index` (int, default=255): Ignore specific class index

**Key Features**:
- Focuses on overlap (intersection over union)
- Less sensitive to class imbalance than CrossEntropy
- Directly optimizes segmentation metrics (Dice/F1)

**Use Case**: When IoU/Dice is the primary evaluation metric

---

#### C. **HybridLoss** (Focal + Dice)
- **Purpose**: Combine pixel-wise accuracy (Focal) with overlap optimization (Dice)
- **Formula**: `Hybrid = λ_focal * FocalLoss + λ_dice * DiceLoss`
- **Parameters**:
  - `alpha`, `gamma`: FocalLoss parameters
  - `smooth`: DiceLoss smoothing
  - `focal_weight` (default=0.5): Weight for Focal component
  - `dice_weight` (default=0.5): Weight for Dice component

**Key Features**:
- Balances classification accuracy and segmentation quality
- Can adjust weights to prioritize one component

**Use Case**: Best of both worlds - accuracy + segmentation quality

---

## 2. Actual Usage in Experiments

### RGB Experiments

| Model | Loss Function | Configuration | File |
|:---|:---|:---|:---|
| **RGB UNet** | CrossEntropyLoss | `weight=[1,5,5,5]` (class weights) | `RGB_Experiments/train_rgb.py:276` |
| **RGB DeepLabv3+ (Baseline)** | CrossEntropyLoss | `weight=[1,5,5,5]` | `RGB_Experiments/train_rgb.py:276` |
| **RGB DeepLabv3+ (Copy-Paste)** | CrossEntropyLoss | `weight=[1,5,5,5]` | `RGB_Experiments/train_rgb.py:276` |

**Why CrossEntropyLoss for RGB?**
- RGB has distinct visual features (color) that are easier to separate
- Simple class weights [1,5,5,5] provide sufficient balancing
- CrossEntropy is computationally efficient and stable

---

### HSI Experiments

| Model | Loss Function | Configuration | File |
|:---|:---|:---|:---|
| **HSI UNet (Baseline)** | CrossEntropyLoss | No weights | `train_hsi_overlap.py:198` |
| **HSI DeepLabv3+ (Baseline)** | CrossEntropyLoss | No weights | `train_hsi_overlap.py:198` |
| **HSI DeepLabv3+ (Overlap + Focal)** | FocalLoss | `alpha=[1,5,10,10]`, `gamma=2` | `HSI_Experiments/train_hsi.py:213` |

**Evolution**:
1. **Early experiments**: Basic CrossEntropyLoss (no class balancing)
2. **Current experiment**: Focal Loss with aggressive weights to combat severe imbalance

**Why Different from RGB?**
- HSI has **severe class imbalance** at patch level (very few IC/Connector pixels)
- RGB uses spatial context; HSI relies on spectral signatures (harder to distinguish)
- Need more aggressive loss function to force minority class learning

---

## 3. Loss Function Comparison

### CrossEntropyLoss (PyTorch Built-in)
```python
CE = -Σ w_c * log(p_c) * y_c
```
**Pros**:
- Fast, stable, well-tested
- Simple class weighting mechanism
- Works well when classes are relatively balanced

**Cons**:
- Treats all examples equally (easy + hard)
- Doesn't directly optimize segmentation metrics (IoU/Dice)
- Can be overwhelmed by easy examples in severe imbalance

**Used in**: All RGB experiments, Early HSI experiments

---

### FocalLoss (Custom Implementation)
```python
FL = -α(1 - p_t)^γ * CE
```
**Pros**:
- Automatically focuses on hard examples
- Reduces gradient contribution from easy examples
- Proven effective for object detection (RetinaNet paper)

**Cons**:
- Hyperparameter sensitive (`gamma`, `alpha`)
- Can be unstable if weights are too aggressive
- More computationally expensive than CE

**Used in**: HSI DeepLabv3+ (Focal Loss experiment)

**Result**: ❌ Degraded performance (-30% on minority classes)  
**Why it failed**: Overly aggressive class weights [1,5,10,10] caused precision collapse

---

### DiceLoss (Custom Implementation)
```python
Dice = 1 - 2|X∩Y| / (|X| + |Y|)
```
**Pros**:
- Directly optimizes IoU/F1 score
- More robust to class imbalance than pixel-wise CE
- Gentle learning signal (smooth gradients)

**Cons**:
- Can be unstable early in training
- Slower convergence than CE
- May struggle with very small objects

**Status**: ✅ Implemented but **NOT YET TESTED** in experiments

---

### HybridLoss (Custom Implementation)
```python
Hybrid = 0.5*Focal + 0.5*Dice
```
**Pros**:
- Combines benefits of classification and segmentation losses
- Used successfully in medical image segmentation
- More robust than either loss alone

**Cons**:
- Additional hyperparameter(weight ratio)
- Requires tuning of both component losses
- More complex to debug

**Status**: ✅ Implemented but **NOT YET TESTED** in experiments

---

## 4. Loss Function Selection Guide

### When to Use Each Loss:

| Scenario | Recommended Loss | Configuration |
|:---|:---|:---|
| **Balanced classes** | CrossEntropyLoss | No weights |
| **Moderate imbalance (10:1)** | CrossEntropyLoss | Class weights [1, 2-5] |
| **Severe imbalance (100:1)** | DiceLoss | `smooth=1e-6` |
| **Object detection-style** | FocalLoss | `gamma=2`, gentle `alpha` |
| **Production system** | HybridLoss | `focal:dice = 0.5:0.5` |
| **Direct IoU optimization** | DiceLoss | Primary metric is Dice/IoU |

---

## 5. Your Experimental Results Summary

### What Worked ✅
- **RGB + CrossEntropyLoss + Class Weights [1,5,5,5]**: 75% Mean IoU
- **Simple weighting was sufficient** for RGB's visual features

### What Didn't Work ❌
- **HSI + FocalLoss + Aggressive Weights [1,5,10,10]**: 40% Mean IoU (degraded)
- **Problem**: Over-penalized minority classes → precision collapse
- **Root cause**: Data scarcity, not loss function

### Not Yet Tested (Future Work)
- **DiceLoss** alone
- **HybridLoss** (Focal + Dice balanced)
- **DiceL Loss with gentler Focal** (`alpha=[1,2,3,3]`)

---

## 6. Implementation Details

### Class Weights Calculation (Example from your code)
```python
# RGB Training (train_rgb.py)
class_weights = torch.tensor([1.0, 5.0, 5.0, 5.0])  # Background, Component, IC, Connector
criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='mean')
```

### Focal Loss with Class Weights (HSI)
```python
# HSI Training (train_hsi.py)
class_weights = torch.tensor([1.0, 5.0, 10.0, 10.0]).to(device)
loss_fn = FocalLoss(alpha=class_weights, gamma=2, reduction='mean')
```

### Using HybridLoss (Not yet tested)
```python
from utils.loss_functions import HybridLoss

# Balanced hybrid
loss_fn = HybridLoss(
    alpha=class_weights,  # For Focal component
    gamma=2,
    smooth=1e-6,  # For Dice component
    focal_weight=0.5,  # Equal weighting
    dice_weight=0.5
)
```

---

## 7. Recommendations for Thesis

### Document the Following:
1. **Loss Function Evolution**:
   - Started with basic CE
   - Explored Focal Loss for class imbalance
   - Results showed data scarcity, not loss function, was the bottleneck

2. **Implementation Contributions**:
   - Custom FocalLoss, DiceLoss, HybridLoss implementations
   - Modular design allows easy experimentation
   - All loss functions support class weighting

3. **Future Work**:
   - Test DiceLoss (direct IoU optimization)
   - Explore HybridLoss for production systems
   - Investigate loss scheduling (CE → Dice over epochs)

---

## 8. Code References

| Loss Function | Implementation | Usage |
|:---|:---|:---|
| FocalLoss | `utils/loss_functions.py:6-51` | `HSI_Experiments/train_hsi.py:213` |
| DiceLoss | `utils/loss_functions.py:53-90` | Not yet used |
| HybridLoss | `utils/loss_functions.py:92-103` | Not yet used |
| CrossEntropyLoss | PyTorch built-in | `RGB_Experiments/train_rgb.py:276` |

---

## Summary

**Total Loss Functions**:
- ✅ **4 Implemented**: CrossEntropyLoss, FocalLoss, DiceLoss, HybridLoss
- ✅ **2 Tested**: CrossEntropyLoss (all experiments), FocalLoss (HSI)
- ⏳ **2 Available for Future Work**: DiceLoss, HybridLoss

**Best Performer**: **CrossEntropyLoss with moderate class weights** (RGB: 75% IoU)  
**Lesson Learned**: In data-scarce scenarios, **loss function alone cannot create missing information**
