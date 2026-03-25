# Advanced Data Augmentation Strategies for PCB Segmentation

To address the **class imbalance** problem (few ICs/Connectors vs. many background pixels) and improve model robustness, we can employ several advanced augmentation strategies beyond the standard rotation and flipping.

Here is a curated list of strategies applicable to your PCBVision project.

---

## ✅ Implemented: Smart Copy-Paste with Scale Jittering

**Status**: **ACTIVE** in `utils/augmentation_functions.py`

### Features Added
1. **Scale Jittering**: Components are randomly resized by ±10% before pasting
2. **Edge Blending**: Gaussian blur on edges creates smooth, realistic blending
3. **Bounds Checking**: Automatically skips pasting if scaled component exceeds image boundaries

### Usage
```python
from utils.augmentation_functions import CopyPasteAugmentation

# Initialize with scale jittering and edge blending enabled (default)
augmentor = CopyPasteAugmentation(
    minority_classes=(1, 2, 3),
    max_paste_per_class=3
)

# Build component bank
augmentor.build_bank(train_images, train_masks)

# Apply augmentation (scale_jitter=True, edge_blend=True by default)
aug_image, aug_mask = augmentor(image, mask)
```

### Expected Impact
- **Scale Invariance**: Model learns to recognize components at different sizes
- **Natural Appearance**: Edge blending prevents "pasted look" artifacts
- **Minority Class Boost**: ICs and Connectors see 3x more training examples

---

## 1. Smart Copy-Paste (Enhancement to Current Approach)
**Concept**: Your current implementation randomly places components. "Smart" Copy-Paste ensures placements are realistically positioned.
*   **Context-Awareness**: Only paste components onto "Background" (Class 0) regions vs. on top of other components.
*   **Scale Jittering**: Randomly resize the extracted component (e.g., ±10%) before pasting to teach the model scale invariance.
*   **Edge Blending**: Apply Gaussian blur to the edges of the pasted patch to remove "jagged" artifacts, preventing the model from just learning "sharp pixel transitions = object".

**Why for PCB?**
Essential for fixing the "Data Scarcity" of ICs. It directly generates new training samples for minority classes.

## 2. Mosaic Augmentation (YOLO Style)
**Concept**: Stitch 4 training images into a single image (2x2 grid) during training.
*   **Mechanism**: Take 4 random images, resize/crop them, and combine them. Adjust the masks accordingly.
*   **Benefit**: 
    1.  Increases the number of objects in a single batch norm step (better batch statistics).
    2.  Forces the model to recognize objects at different locations and relative contexts.
    
**Why for PCB?**
Very effective for **small object detection**. By varying the crop, it acts as a strong regularizer.

## 3. CutMix
**Concept**: Instead of pasting *specific objects*, cut a random rectangular region from Image A and paste it onto Image B.
*   **Label Mixing**: The label (mask) is also mixed.
*   **Difference from Copy-Paste**: Copy-Paste uses *segmented objects*. CutMix uses *random rectangles*.
    
**Why for PCB?**
Helps the model not to over-rely on the "PCB texture" context. If it sees a connector on top of a different PCB background, it learns to look at the *connector itself*.

## 4. GridMask / Cutout
**Concept**: Randomly set regions of the image to black (zeros).
*   **Cutout**: One large square removed.
*   **GridMask**: A grid of squares removed.
    
**Why for PCB?**
Simulates **occlusion** or **damage/glare**. It forces the network to recognize a component even if part of it is missing or obscured by a highlight (specular reflection).

## 5. Class-Balanced Oversampling (Simple but effective)
**Concept**: Instead of seeing every image once per epoch, show images containing minority classes *more often*.
*   **Implementation**: Create a `WeightedRandomSampler` in the PyTorch DataLoader. Assign higher weights to images containing Class 2 (IC) or Class 3 (Connector).
    
**Why for PCB?**
Guarantees the model sees enough examples of rare classes without needing synthetic data.

## 6. Geometric Transforms (Strict)
**Concept**: 
*   **Rotations**: 90°, 180°, 270° (Safe for PCBs).
*   **Flips**: Horizontal/Vertical.
*   **Shear**: *Avoid* or keep very low. PCBs are rigid manufacturing boards; they don't "shear" or warp like natural objects.
    
---

## Recommendation for Thesis

For the **Highest Impact** with the **Least Code Complexity**, implement this priority list:

1.  **Refine Copy-Paste** (Add Scale Jittering & Gaussian Edge Blending).
2.  **Class-Balanced Oversampling** (2-line change in DataLoader).
3.  **Mosaic** (If you have time, great for "Detection" robustness).

### Code Snippet: Scale Jittering for Copy-Paste
```python
# Inside your _paste function
scale = np.random.uniform(0.9, 1.1) # +/- 10%
new_h, new_w = int(ph * scale), int(pw * scale)
comp_data_resized = cv2.resize(comp.data, (new_w, new_h))
comp_mask_resized = cv2.resize(comp.mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
```
