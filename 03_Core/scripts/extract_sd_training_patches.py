"""
Extract 256×256 patches from CutMix outputs for SD1.5 LoRA training

This script:
1. Loads RGB train images and masks
2. Applies CutMix augmentation
3. Extracts patches containing minority classes (Capacitor, IC, Connector)
4. Creates inpainting masks (dilated minority regions)
5. Generates class-specific prompts
6. Saves triplets: (image, inpaint_mask, prompt)

Output: data/cutmix_patches/{Capacitor,IC,Connector}/
"""

import os
import sys
import numpy as np
import cv2
from pathlib import Path
import argparse
from tqdm import tqdm
import json

# Add parent to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.dataset_functions import read_dataset
from utils.augmentation_functions import MultimodalCutMix

# Class mapping (FINAL - verified visually with user)
CLASS_NAMES = {
    0: "Others",
    1: "IC",
    2: "Capacitor",      # Round components
    3: "Connector"       # Rectangular components
}

# Include ALL minority classes for LoRA training (model needs to learn full PCB context)
# IC: 4.52% (common but needed for context)
# Capacitor: 0.49% (rarest - will get 5× augmentation in Stage C)
# Connector: 0.48% (rare - will get 5× augmentation in Stage C)
MINORITY_CLASSES = [1, 2, 3]  # IC, Capacitor, Connector

# Prompts for each class
PROMPTS = {
    1: "a close-up photo of a printed circuit board with integrated circuits, realistic, high detail",
    2: "a close-up photo of a printed circuit board with capacitors, realistic electronic components, sharp details",
    3: "a close-up photo of a printed circuit board with connectors, realistic electronic components"
}


def extract_patches_from_image(
    image, 
    mask, 
    patch_size=256,
    min_minority_ratio=0.02,  # 2% minority pixels
    max_minority_ratio=0.40,  # 40% minority pixels (not too crowded)
    stride=128,  # overlap patches
    dilation_kernel_size=7
):
    """
    Extract patches from a single CutMix image
    
    Returns:
        List of dicts with keys: image_patch, inpaint_mask, class_id
    """
    patches = []
    H, W = image.shape[:2]
    
    # Slide window
    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            # Extract patch
            img_patch = image[y:y+patch_size, x:x+patch_size]
            mask_patch = mask[y:y+patch_size, x:x+patch_size]
            
            # Check each minority class
            for class_id in MINORITY_CLASSES:
                # Count pixels of this class
                class_pixels = (mask_patch == class_id).sum()
                total_pixels = patch_size * patch_size
                ratio = class_pixels / total_pixels
                
                # Check if within target range
                if min_minority_ratio <= ratio <= max_minority_ratio:
                    # Create inpainting mask
                    # Binary mask: 1 where class_id exists
                    binary_mask = (mask_patch == class_id).astype(np.uint8)
                    
                    # Dilate to blend edges
                    kernel = np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8)
                    inpaint_mask = cv2.dilate(binary_mask, kernel, iterations=1)
                    
                    # Convert to 0-255
                    inpaint_mask = (inpaint_mask * 255).astype(np.uint8)
                    
                    patches.append({
                        'image_patch': img_patch.copy(),
                        'inpaint_mask': inpaint_mask,
                        'class_id': class_id,
                        'minority_ratio': ratio
                    })
    
    return patches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', default='/home/bs_thesis/Documents/BS_THESIS/DATASETS/PCBDataset/')
    parser.add_argument('--output-dir', default='data/cutmix_patches')
    parser.add_argument('--patch-size', type=int, default=256)
    parser.add_argument('--num-cutmix-per-image', type=int, default=5, 
                       help='Number of CutMix augmentations per image pair')
    parser.add_argument('--min-patches-per-class', type=int, default=500,
                       help='Minimum patches to extract per class')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    
    # Create output directories
    output_path = Path(args.output_dir)
    for class_id in MINORITY_CLASSES:
        class_dir = output_path / CLASS_NAMES[class_id]
        class_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("SD1.5 LoRA Patch Extraction for PCB Vision")
    print("="*60)
    print(f"\nOutput directory: {output_path}")
    print(f"Patch size: {args.patch_size}×{args.patch_size}")
    print(f"Target per class: {args.min_patches_per_class}+ patches\n")
    
    # Load dataset
    print("Loading RGB dataset...")
    _, _, _, RGB, _, RGB_masks, _ = read_dataset(args.dataset_path)
    
    # Filter out None and resize to common size
    IMG_SIZE = 640
    RGB_filtered = []
    masks_filtered = []
    for img, mask in zip(RGB, RGB_masks):
        if img is not None and mask is not None:
            # Resize to common size
            img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
            mask_resized = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
            RGB_filtered.append(img_resized)
            masks_filtered.append(mask_resized)
    
    print(f"Loaded {len(RGB_filtered)} valid RGB images\n")
    
    # Initialize CutMix
    cutmix = MultimodalCutMix(beta=1.0)
    
    # Patch counters
    patch_counts = {cid: 0 for cid in MINORITY_CLASSES}
    
    # Extract patches
    print("Extracting patches from CutMix outputs...\n")
    
    total_iterations = len(RGB_filtered) * args.num_cutmix_per_image
    pbar = tqdm(total=total_iterations, desc="Processing")
    
    for idx1 in range(len(RGB_filtered)):
        img1 = RGB_filtered[idx1]
        mask1 = masks_filtered[idx1]
        
        for _ in range(args.num_cutmix_per_image):
            # Random second image
            idx2 = np.random.randint(len(RGB_filtered))
            img2 = RGB_filtered[idx2]
            mask2 = masks_filtered[idx2]
            
            # Apply CutMix
            img_mixed, _, mask_mixed, _, _ = cutmix.cutmix(
                img1, None, mask1,
                img2, None, mask2
            )
            
            # Extract patches
            patches = extract_patches_from_image(
                img_mixed, 
                mask_mixed,
                patch_size=args.patch_size
            )
            
            # Save patches
            for patch_data in patches:
                class_id = patch_data['class_id']
                class_name = CLASS_NAMES[class_id]
                
                # Check if we need more patches for this class
                if patch_counts[class_id] >= args.min_patches_per_class:
                    continue
                
                # Generate filename
                patch_idx = patch_counts[class_id]
                filename_base = f"{patch_idx:05d}"
                
                # Save image
                img_path = output_path / class_name / f"{filename_base}_image.png"
                cv2.imwrite(str(img_path), cv2.cvtColor(patch_data['image_patch'], cv2.COLOR_RGB2BGR))
                
                # Save inpaint mask
                mask_path = output_path / class_name / f"{filename_base}_inpaint_mask.png"
                cv2.imwrite(str(mask_path), patch_data['inpaint_mask'])
                
                # Save prompt
                prompt_path = output_path / class_name / f"{filename_base}_prompt.txt"
                with open(prompt_path, 'w') as f:
                    f.write(PROMPTS[class_id])
                
                patch_counts[class_id] += 1
            
            pbar.update(1)
            
            # Check if all classes have enough patches
            if all(count >= args.min_patches_per_class for count in patch_counts.values()):
                break
        
        if all(count >= args.min_patches_per_class for count in patch_counts.values()):
            break
    
    pbar.close()
    
    # Save metadata
    metadata = {
        'patch_size': args.patch_size,
        'num_cutmix_per_image': args.num_cutmix_per_image,
        'patch_counts': {CLASS_NAMES[cid]: count for cid, count in patch_counts.items()},
        'prompts': {CLASS_NAMES[cid]: PROMPTS[cid] for cid in MINORITY_CLASSES},
        'seed': args.seed
    }
    
    with open(output_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("Extraction Complete!")
    print("="*60)
    for class_id in MINORITY_CLASSES:
        class_name = CLASS_NAMES[class_id]
        count = patch_counts[class_id]
        print(f"  {class_name:12s}: {count:5d} patches")
    print(f"\nSaved to: {output_path}")
    print("="*60)


if __name__ == "__main__":
    main()
