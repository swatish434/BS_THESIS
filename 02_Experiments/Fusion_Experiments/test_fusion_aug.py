
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from Fusion_Experiments.fusion_dataset import FusionDataset

DATA_DIR = "/home/bs_thesis/Documents/BS_THESIS/PCBVision/Fusion_Experiments/data/patches"

def test_augmentations():
    # 1. Load some training indices
    train_indices = list(range(10))
    
    print("Initializing Augmented Dataset...")
    # For CP bank, we need some data. Let's just use the first 10 scenes.
    ds_temp = FusionDataset(DATA_DIR, split_indices=train_indices)
    bank_imgs = []
    bank_masks = []
    for i in range(len(ds_temp)):
        bank_imgs.append(np.load(ds_temp.image_files[i]))
        bank_masks.append(np.load(ds_temp.mask_files[i]))
        if i > 20: break # Small bank for testing
    
    ds = FusionDataset(
        DATA_DIR, 
        split_indices=train_indices,
        augment_cp=True,
        augment_cutmix=True,
        component_bank_data=(bank_imgs, bank_masks)
    )
    
    print(f"Dataset ready. Items: {len(ds)}")
    
    # Grab a few samples and save visualizations
    save_dir = "test_aug_viz"
    os.makedirs(save_dir, exist_ok=True)
    
    for i in range(5):
        img_tensor, mask_tensor = ds[i]
        
        # tensor is (C, H, W) -> (H, W, C)
        img_np = img_tensor.permute(1, 2, 0).numpy()
        mask_np = mask_tensor.numpy()
        
        # Use RGB channels for visualization (indices 0, 1, 2 in our stack)
        rgb = img_np[:, :, :3].astype(np.uint8)
        
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(rgb)
        plt.title(f"Augmented RGB (Sample {i})")
        plt.subplot(1, 2, 2)
        plt.imshow(mask_np, cmap='jet')
        plt.title(f"Augmented Mask")
        plt.savefig(os.path.join(save_dir, f"aug_test_{i}.png"))
        plt.close()
        
    print(f"Visualizations saved to {save_dir}")

if __name__ == "__main__":
    test_augmentations()
