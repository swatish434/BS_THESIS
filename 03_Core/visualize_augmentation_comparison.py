
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

# Import models
from models.Unet import UNET
from models.DeepLabv3_plus import DeepLabv3_plus
from models.ResUnet import ResUnet
from models.Unet_Attention import AttU_Net
from utils.dataset_functions import PCBFullDataset 
# Note: trains_hsi_overlap uses HSIPatchesDataset. We should use similar loading 
# but for visualization on full images (PCBFullDataset) or patches?
# The request implies "HSI models", usually trained on patches but evaluated on patches or full?
# Let's use HSIPatchesDataset for consistency with training, or check eval_hsi_resunet.py

from train_hsi_overlap import HSIPatchesDataset, get_hsi_model

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_name, checkpoint_path, in_channels=214, out_channels=4):
    print(f"Loading {model_name} from {checkpoint_path}...")
    model = get_hsi_model(model_name, in_channels, out_channels)
    model.to(device)
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        # Handle if state_dict is inside a key or direct
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        return model
    else:
        print(f"Warning: Checkpoint {checkpoint_path} not found!")
        return None

def visualize_comparison(model_name, baseline_ckpt, copypaste_ckpt, output_dir="Results/comparison"):
    os.makedirs(output_dir, exist_ok=True)
    
    # Load Models
    baseline_model = load_model(model_name, baseline_ckpt)
    copypaste_model = load_model(model_name, copypaste_ckpt)
    
    if baseline_model is None or copypaste_model is None:
        print("One or both models failed to load. Aborting.")
        return

    # Load Dataset (Validation Split)
    DATA_DIR = "/home/bs_thesis/Documents/BS_THESIS/PCBVision/Patches_256_Overlap_Data/"
    # Use 'Val' split as in training
    dataset = HSIPatchesDataset(DATA_DIR, split='Val')
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # We want to find samples where CopyPaste performs better, or just random samples?
    # Let's pick a few distinct samples.
    # We'll select 5 random samples.
    indices = np.random.choice(len(dataset), 5, replace=False)
    
    print(f"Visualizing samples: {indices}")
    
    # Colors for segmentation (BG, Comp1, Comp2, Comp3) -> Black, Red, Green, Blue
    colors = np.array([
        [0, 0, 0],       # 0: BG
        [255, 0, 0],     # 1: Open
        [0, 255, 0],     # 2: Short
        [0, 0, 255]      # 3: Mousebite (or whatever classes are)
    ], dtype=np.uint8)
    
    # PCBVision classes: 0: Background, 1: Open, 2: Short, 3: Mousebite
    # Wait, check dataset_functions.py:
    # classes = {0:'black', 1:'red', 2:'green', 3:'blue', 4:'Yellow'} ?
    # train_hsi_overlap.py says OUT_CHANNELS = 4. So 0, 1, 2, 3.
    
    for i, idx in enumerate(indices):
        hsi, mask = dataset[idx] # hsi: (C, H, W), mask: (H, W)
        
        # Inference
        inputs = hsi.unsqueeze(0).to(device) # (1, C, H, W)
        
        with torch.no_grad():
            # Baseline
            out_base = baseline_model(inputs)
            pred_base = torch.argmax(torch.softmax(out_base, dim=1), dim=1).squeeze().cpu().numpy()
            
            # CopyPaste
            out_cp = copypaste_model(inputs)
            pred_cp = torch.argmax(torch.softmax(out_cp, dim=1), dim=1).squeeze().cpu().numpy()
            
        # Get Pseudo-RGB for HSI
        # Usually bands [30, 20, 10] or similar for RGB approximation
        # HSI is (C, H, W).
        hsi_np = hsi.permute(1, 2, 0).cpu().numpy() # (H, W, C)
        # Select 3 bands. 
        # H x W x C
        if hsi_np.shape[2] > 3:
            rgb_img = hsi_np[:, :, [29, 19, 9]] # Approx red, green, blue bands
            # Normalize to 0-255
            rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min() + 1e-8)
            rgb_img = (rgb_img * 255).astype(np.uint8)
        else:
            rgb_img = (hsi_np * 255).astype(np.uint8)

        # Create Color Masks
        def colorize(m):
            h, w = m.shape
            res = np.zeros((h, w, 3), dtype=np.uint8)
            for c in range(1, 4): # skip 0 (black)
                res[m == c] = colors[c]
            return res

        gt_color = colorize(mask.cpu().numpy())
        base_color = colorize(pred_base)
        cp_color = colorize(pred_cp)
        
        # Plot
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        axes[0].imshow(rgb_img)
        axes[0].set_title("HSI (Pseudo-RGB)")
        axes[0].axis('off')
        
        axes[1].imshow(gt_color)
        axes[1].set_title("Ground Truth")
        axes[1].axis('off')
        
        axes[2].imshow(base_color)
        axes[2].set_title(f"Baseline\n({os.path.basename(baseline_ckpt)})")
        axes[2].axis('off')
        
        axes[3].imshow(cp_color)
        axes[3].set_title(f"Copy-Paste\n({os.path.basename(copypaste_ckpt)})")
        axes[3].axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(output_dir, f"comparison_{model_name}_{idx}.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Saved {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Model name (attunet/resunet)')
    parser.add_argument('--baseline', type=str, required=True, help='Path to baseline checkpoint')
    parser.add_argument('--copypaste', type=str, required=True, help='Path to copypaste checkpoint')
    args = parser.parse_args()
    
    visualize_comparison(args.model, args.baseline, args.copypaste)
