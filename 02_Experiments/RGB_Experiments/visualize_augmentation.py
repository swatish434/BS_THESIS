import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from dataset_functions import read_dataset
from augmentation_functions import CopyPasteAugmentation

def main():
    # 1. Load Data
    print("Loading dataset...")
    dataset_path = "/home/bs_thesis/Documents/BS_THESIS/DATASETS/PCBDataset/"
    _, _, _, RGB, _, RGB_masks, _ = read_dataset(dataset_path)
    
    # Filter None
    RGB = [Pixel for Pixel in RGB if Pixel is not None]
    RGB_masks = [mask for mask in RGB_masks if mask is not None]
    
    print(f"Loaded {len(RGB)} valid images.")

    # 2. Initialize Augmentation
    print("Initializing Copy-Paste Augmentation...")
    augmentor = CopyPasteAugmentation(
        minority_classes=(1, 2, 3), # Components, ICs, Connectors
        max_paste_per_class=5,      # Aggressive pasting for visualization
        avoid_overlap=True
    )
    
    # 3. Build Bank
    print("Building Component Bank...")
    augmentor.build_bank(RGB, RGB_masks)
    
    # Check bank stats
    for cls, items in augmentor.component_bank.items():
        print(f"Class {cls}: {len(items)} components extracted.")

    # 4. Quantity Verification (Before vs After)
    print("\n" + "="*40)
    print("      Verification: Component Counts")
    print("="*40)
    print(f"{'Sample':<8} | {'Class':<12} | {'Before':<8} | {'After':<8} | {'Increase':<8}")
    print("-" * 60)

    from skimage.measure import label
    
    total_increase = {1:0, 2:0, 3:0}
    
    # Test on first 10 images
    test_indices = range(10)
    
    for idx in test_indices:
        original_mask = RGB_masks[idx].copy()
        if original_mask.ndim == 3: original_mask = original_mask.squeeze()
        
        # Apply Augmentation
        _, aug_mask = augmentor(RGB[idx], original_mask)
        if aug_mask.ndim == 3: aug_mask = aug_mask.squeeze()

        for cls_id, cls_name in [(1, 'Component'), (2, 'IC'), (3, 'Connector')]:
            # Count Before
            binary_before = (original_mask == cls_id).astype(np.uint8)
            count_before = label(binary_before, connectivity=2).max() # simple label count
            
            # Count After
            binary_after = (aug_mask == cls_id).astype(np.uint8)
            count_after = label(binary_after, connectivity=2).max()
            
            diff = count_after - count_before
            total_increase[cls_id] += diff
            
            if diff > 0:
                print(f"Img {idx:<4} | {cls_name:<12} | {count_before:<8} | {count_after:<8} | +{diff}")

    print("="*40)
    print("Total New Objects Added (in 10 images):")
    for cls_id, cls_name in [(1, 'Component'), (2, 'IC'), (3, 'Connector')]:
        print(f"  - {cls_name}: +{total_increase[cls_id]}")
    print("="*40)

    # 4. Generate Visualizations (Existing code...)
    print("Generating Visualizations...")
    
    # Pick a sample that has empty space (e.g., sample 0 or 5)
    indices = [0, 5, 10]
    
    fig, axes = plt.subplots(len(indices), 3, figsize=(15, 6 * len(indices)))
    
    for i, idx in enumerate(indices):
        original_img = RGB[idx].copy()
        original_mask = RGB_masks[idx].copy()
        
        # Apply Augmentation MULTIPLE times to ensure visibility
        # or just once but relying on the aggressive max_paste settings
        aug_img, aug_mask = augmentor(original_img, original_mask)
        
        # Plot Original
        axes[i, 0].imshow(original_img)
        axes[i, 0].set_title(f"Original RGB (Idx {idx})")
        axes[i, 0].axis('off')
        
        # Plot Augmented
        axes[i, 1].imshow(aug_img)
        axes[i, 1].set_title("Augmented RGB (Copy-Paste)")
        axes[i, 1].axis('off')

        # Plot Mask Difference (Highlight pasted areas)
        # 0 = Match, 1 = Difference
        diff = (original_mask != aug_mask).astype(int)
        axes[i, 2].imshow(diff, cmap='gray')
        axes[i, 2].set_title("Augmentation Delta (Mask)")
        axes[i, 2].axis('off')
        
    save_path = "Results/augmentation_check.png"
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved visualization to {save_path}")

if __name__ == "__main__":
    main()
