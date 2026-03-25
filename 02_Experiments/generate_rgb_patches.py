"""
generate_rgb_patches.py
───────────────────────
Generates 256x256 RGB patches from the 53 full RGB images + Monoseg masks.
Uses sliding window with stride=200 (overlapping).
Saves patches to a new folder: RGB_Patches_256/
  - Train_0.png, Train_0_mask.png
  - Val_0.png,   Val_0_mask.png
  - Test_0.png,  Test_0_mask.png

Usage:
    python generate_rgb_patches.py
"""

import os
import cv2
import numpy as np
import glob
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
RGB_DIR    = '/home/samiran_iiserb/asip_lab/UG/SWATISH/BS_THESIS/DATASETS/PCBDataset/RGB'
MASK_DIR   = '/home/samiran_iiserb/asip_lab/UG/SWATISH/BS_THESIS/DATASETS/PCBDataset/RGB/Monoseg'
OUTPUT_DIR = '/home/samiran_iiserb/asip_lab/UG/SWATISH/BS_THESIS/PCBVision/01_Data/RGB_Patches_256'

PATCH_SIZE  = 256
STRIDE      = 200
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# TEST_RATIO  = 0.15

MIN_FG_RATIO = 0.01  # skip patches with less than 1% foreground


def extract_patches(img, mask, patch_size, stride, min_fg_ratio=0.01):
    """Extract patches, skip background-only ones."""
    h, w    = img.shape[:2]
    patches = []

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            img_p  = img [y:y+patch_size, x:x+patch_size]
            mask_p = mask[y:y+patch_size, x:x+patch_size]

            # Pad if needed
            ph, pw = img_p.shape[:2]
            if ph < patch_size or pw < patch_size:
                img_p  = cv2.copyMakeBorder(img_p,  0, patch_size-ph,
                                             0, patch_size-pw,
                                             cv2.BORDER_REFLECT)
                mask_p = cv2.copyMakeBorder(mask_p, 0, patch_size-ph,
                                             0, patch_size-pw,
                                             cv2.BORDER_REFLECT)

            # Skip if too little foreground
            fg_ratio = (mask_p > 0).sum() / mask_p.size
            if fg_ratio < min_fg_ratio:
                continue

            patches.append((img_p, mask_p))

    return patches


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Find all paired RGB + Monoseg images
    rgb_files = sorted(glob.glob(os.path.join(RGB_DIR, '*.jpg')),
                       key=lambda x: int(Path(x).stem))
    print(f"Found {len(rgb_files)} RGB images")

    all_patches = []

    for rgb_path in rgb_files:
        img_id    = Path(rgb_path).stem
        mask_path = os.path.join(MASK_DIR, f"{img_id}.png")

        if not os.path.exists(mask_path):
            print(f"  Skipping {img_id} — no mask found")
            continue

        img  = cv2.imread(rgb_path)
        img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            print(f"  Skipping {img_id} — failed to load")
            continue

        # Resize mask to match image if needed
        if mask.shape[:2] != img.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                              interpolation=cv2.INTER_NEAREST)

        patches = extract_patches(img, mask, PATCH_SIZE, STRIDE, MIN_FG_RATIO)
        all_patches.extend(patches)
        print(f"  {img_id}.jpg ({img.shape[0]}x{img.shape[1]}) → {len(patches)} patches")

    total = len(all_patches)
    print(f"\nTotal patches with foreground: {total}")

    # Shuffle with fixed seed for reproducibility
    np.random.seed(42)
    indices = np.random.permutation(total)

    n_train = int(total * TRAIN_RATIO)
    n_val   = int(total * VAL_RATIO)

    splits = {
        'Train': indices[:n_train],
        'Val':   indices[n_train:n_train+n_val],
        'Test':  indices[n_train+n_val:],
    }

    print(f"Split — Train:{len(splits['Train'])}  "
          f"Val:{len(splits['Val'])}  Test:{len(splits['Test'])}")

    # Save patches
    for split_name, idxs in splits.items():
        print(f"Saving {split_name}...")
        for i, idx in enumerate(idxs):
            img_p, mask_p = all_patches[idx]
            base = os.path.join(OUTPUT_DIR, f"{split_name}_{i}")

            # Save RGB as PNG
            cv2.imwrite(base + '.png',
                        cv2.cvtColor(img_p, cv2.COLOR_RGB2BGR))
            # Save mask as PNG
            cv2.imwrite(base + '_mask.png', mask_p)

    print(f"\nDone! Patches saved to: {OUTPUT_DIR}")
    print(f"Train:{len(splits['Train'])}  Val:{len(splits['Val'])}  "
          f"Test:{len(splits['Test'])}")

    # Quick verification
    print("\nVerification:")
    for split_name in ['Train', 'Val', 'Test']:
        imgs  = glob.glob(os.path.join(OUTPUT_DIR, f"{split_name}_*.png"))
        imgs  = [f for f in imgs if '_mask' not in f]
        masks = glob.glob(os.path.join(OUTPUT_DIR, f"{split_name}_*_mask.png"))
        print(f"  {split_name}: {len(imgs)} images, {len(masks)} masks "
              f"{'OK' if len(imgs)==len(masks) else 'MISMATCH!'}")


if __name__ == '__main__':
    main()