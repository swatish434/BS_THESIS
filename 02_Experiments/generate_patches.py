"""
generate_patches.py
───────────────────
Regenerates HSI patches + mask patches on HPC from raw HSI data.
Matches the existing Train_/Val_/Test_ split in Patches_256_Overlap_Data.

Usage:
    python generate_patches.py

Output: overwrites corrupted binary files in Patches_256_Overlap_Data
        with correct data. Keeps existing .hdr and .npy files intact.
"""

import os
import glob
import numpy as np
import cv2
from tqdm import tqdm
import spectral.io.envi as envi

# ── Paths ─────────────────────────────────────────────────────────────────────
HSI_ROOT    = '/home/samiran_iiserb/asip_lab/UG/SWATISH/BS_THESIS/DATASETS/PCBDataset/HSI'
MASK_ROOT   = os.path.join(HSI_ROOT, 'Monoseg_masks')
OUTPUT_DIR  = '/home/samiran_iiserb/asip_lab/UG/SWATISH/BS_THESIS/PCBVision/01_Data/Patches_256_Overlap_Data'

PATCH_SIZE  = 256
STRIDE      = 200   # overlapping patches
TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
# TEST_RATIO  = 0.10

NUM_CLASSES = 4


# ── Load one HSI cube ─────────────────────────────────────────────────────────

def load_hsi(pcb_folder: str, pcb_name: str) -> np.ndarray:
    hdr  = os.path.join(pcb_folder, f"{pcb_name}.hdr")
    data = os.path.join(pcb_folder, pcb_name)
    obj  = envi.open(hdr, data)
    hsi  = np.array(obj.load(), dtype=np.float32)  # (H, W, bands)
    return hsi


def load_mask(mask_folder: str, mask_name: str) -> np.ndarray:
    hdr  = os.path.join(mask_folder, f"{mask_name}.hdr")
    data = os.path.join(mask_folder, mask_name)
    obj  = envi.open(hdr, data)
    mask = np.array(obj.load(), dtype=np.float32)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return mask.astype(np.uint8)


# ── Extract patches from one image ───────────────────────────────────────────

def extract_patches(hsi, mask, patch_size=256, stride=200):
    h, w = hsi.shape[:2]
    hsi_patches  = []
    mask_patches = []

    for i in range(0, h, stride):
        for j in range(0, w, stride):
            ph = hsi[i:i+patch_size, j:j+patch_size, :]
            pm = mask[i:i+patch_size, j:j+patch_size]

            # Pad if needed
            if ph.shape[0] < patch_size or ph.shape[1] < patch_size:
                pad_h = max(0, patch_size - ph.shape[0])
                pad_w = max(0, patch_size - ph.shape[1])
                ph = np.pad(ph, ((0,pad_h),(0,pad_w),(0,0)), mode='constant')
                pm = np.pad(pm, ((0,pad_h),(0,pad_w)),       mode='constant')

            hsi_patches.append(ph)
            mask_patches.append(pm)

    return hsi_patches, mask_patches


# ── Save one patch as ENVI binary ─────────────────────────────────────────────

def save_hsi_patch(hsi_patch: np.ndarray, out_path: str):
    """Save HSI patch as ENVI BIP float32 binary + header."""
    h, w, bands = hsi_patch.shape
    hdr_path    = out_path + '.hdr'

    # Write binary
    hsi_patch.astype(np.float32).tofile(out_path)

    # Write header
    with open(hdr_path, 'w') as f:
        f.write("ENVI\n")
        f.write(f"samples = {w}\n")
        f.write(f"lines = {h}\n")
        f.write(f"bands = {bands}\n")
        f.write("header offset = 0\n")
        f.write("file type = ENVI Standard\n")
        f.write("data type = 4\n")       # float32
        f.write("interleave = bip\n")
        f.write("byte order = 0\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Discover all pcb folders
    pcb_folders = sorted([
        d for d in os.listdir(HSI_ROOT)
        if d.startswith('pcb') and os.path.isdir(os.path.join(HSI_ROOT, d))
    ], key=lambda x: int(x.replace('pcb', '')))

    print(f"Found {len(pcb_folders)} PCB folders")

    # Collect all patches
    all_hsi_patches  = []
    all_mask_patches = []

    for pcb_name in tqdm(pcb_folders, desc="Processing PCBs"):
        pcb_num    = pcb_name.replace('pcb', '')
        pcb_folder = os.path.join(HSI_ROOT, pcb_name)
        mask_name  = f"mono{pcb_num}"

        try:
            hsi  = load_hsi(pcb_folder, pcb_name)
            mask = load_mask(MASK_ROOT, mask_name)

            # Resize mask to match HSI if needed
            if mask.shape[:2] != hsi.shape[:2]:
                mask = cv2.resize(mask, (hsi.shape[1], hsi.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)

            hsi_p, mask_p = extract_patches(hsi, mask, PATCH_SIZE, STRIDE)
            all_hsi_patches.extend(hsi_p)
            all_mask_patches.extend(mask_p)
            print(f"  {pcb_name}: {hsi.shape} → {len(hsi_p)} patches")

        except Exception as e:
            print(f"  [!] Skipping {pcb_name}: {e}")
            continue

    total = len(all_hsi_patches)
    print(f"\nTotal patches: {total}")

    # Split into Train / Val / Test
    np.random.seed(42)
    indices = np.random.permutation(total)

    n_train = int(total * TRAIN_RATIO)
    n_val   = int(total * VAL_RATIO)

    train_idx = indices[:n_train]
    val_idx   = indices[n_train:n_train+n_val]
    test_idx  = indices[n_train+n_val:]

    splits = {
        'Train': train_idx,
        'Val':   val_idx,
        'Test':  test_idx,
    }

    print(f"Split — Train:{len(train_idx)}  Val:{len(val_idx)}  Test:{len(test_idx)}")

    # Save patches
    for split_name, idxs in splits.items():
        print(f"\nSaving {split_name} patches...")
        for i, idx in enumerate(tqdm(idxs, desc=f"  {split_name}")):
            patch_id  = f"{split_name}_{i}"
            out_base  = os.path.join(OUTPUT_DIR, patch_id)

            # Save HSI binary + header
            save_hsi_patch(all_hsi_patches[idx], out_base)

            # Save mask as .npy
            mask = np.clip(all_mask_patches[idx], 0, NUM_CLASSES - 1).astype(np.uint8)
            np.save(out_base + '.npy', mask)

    print(f"\nDone! All patches saved to: {OUTPUT_DIR}")
    print(f"Total: Train={len(train_idx)}  Val={len(val_idx)}  Test={len(test_idx)}")


if __name__ == '__main__':
    main()