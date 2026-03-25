import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import cv2

# ── Path setup ───────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
CORE_DIR = os.path.join(PROJECT_ROOT, '03_Core')
sys.path.insert(0, CORE_DIR)

from models.hsrn_segmentation import HybridSSRNSeg
from train_hsrn_experiments import RGBPatchDataset, HSIPCADataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Data Loading Helper ──────────────────────────────────────────────────────
def get_loader_and_model(args):
    data_root = os.path.join(PROJECT_ROOT, '01_Data')
    
    if args.data_type == 'rgb':
        rgb_root = args.dataset_root
        test_ds = RGBPatchDataset(dataset_root=rgb_root, split='Test', target_size=args.image_size)
        in_ch = 3
    else:
        hsi_dir = os.path.join(PROJECT_ROOT, 'Patches_256_Overlap_Data')
        if not os.path.isdir(hsi_dir):
            hsi_dir = os.path.join(data_root, 'Patches_256_Overlap_Data')
        test_ds = HSIPCADataset(data_dir=hsi_dir, split='Test', n_components=args.n_components, target_size=args.image_size)
        in_ch = args.n_components

    # We only need batch_size=1 for visualization
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    
    model = HybridSSRNSeg(
        in_channels=in_ch,
        num_classes=args.num_classes,
        image_size=args.image_size,
        patch_size=args.patch_size,
    ).to(device)
    
    # Load weights
    results_dir = os.path.join(SCRIPT_DIR, 'Results')
    exp_tag = f"hsrn_{args.data_type}_nc{in_ch}_{args.augment}"
    save_path = os.path.join(results_dir, f"{exp_tag}_best.pth")
    
    if not os.path.exists(save_path):
        raise FileNotFoundError(f"Weights not found at {save_path}. Please train the model first.")
        
    print(f"Loading weights from {save_path}")
    checkpoint = torch.load(save_path, map_location=device, weights_only=True)
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint) # fallback for old saves
        
    model.eval()
    return test_loader, model, in_ch

# ── Feature Extraction Hooks ─────────────────────────────────────────────────
# Dictionary to store intermediate features
features = {}

def get_features(name):
    def hook(model, input, output):
        # Move to CPU immediately to save GPU memory and detach from graph
        # Depending on the layer, output might be a tensor or a tuple. 
        if isinstance(output, torch.Tensor):
            features[name] = output.detach().cpu().numpy()
        else:
            features[name] = output[0].detach().cpu().numpy() # take first if tuple
    return hook

# ── Visualization Helpers ────────────────────────────────────────────────────
def visualize_sample(image, mask, pred, spectral_features, transformer_features, sample_idx, save_dir, args):
    """
    Plots the input image, intermediate features, GT mask, and predicted mask.
    """
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f"HSRN {args.data_type.upper()} ({args.augment}) - Sample {sample_idx}", fontsize=16)
    
    # 1. Input Image
    ax = axes[0, 0]
    if args.data_type == 'rgb':
         img_display = image[0].transpose(1, 2, 0) # (H,W,3)
         ax.imshow(img_display)
         ax.set_title("Input RGB")
    else:
         # For HSI, display a false-color composite using bands 0, len//2, -1
         C = image.shape[1]
         false_color = np.zeros((image.shape[2], image.shape[3], 3))
         false_color[:,:,0] = image[0, 0, :, :]
         false_color[:,:,1] = image[0, C//2, :, :]
         false_color[:,:,2] = image[0, C-1, :, :]
         # Normalize for display
         for i in range(3):
             b_min, b_max = false_color[:,:,i].min(), false_color[:,:,i].max()
             if b_max > b_min:
                 false_color[:,:,i] = (false_color[:,:,i] - b_min) / (b_max - b_min)
         ax.imshow(false_color)
         ax.set_title("Input False-Color HSI")
    ax.axis('off')

    # 2. Spectral Features (from SSRN Encoder)
    ax = axes[0, 1]
    # spectral_features shape is (B, embed_ch, H, W). We take the mean across embed_ch for visualization.
    spec_act = np.mean(spectral_features[0], axis=0)
    im = ax.imshow(spec_act, cmap='viridis')
    ax.set_title("Spectral Encoder Activation (Mean)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.axis('off')

    # 3. Transformer Features (ViT)
    ax = axes[0, 2]
    # Transformer features are (B, N+1, dim). 
    # N is h_p*w_p (e.g. 16*16=256). We ignore the class token [:, 0, :] and reshape the patch tokens.
    patch_tokens = transformer_features[0, 1:, :] # shape: (N, dim)
    # Mean activation across embedding dim for each patch
    trans_act = np.mean(patch_tokens, axis=-1)
    
    # Reshape back to spatial grid
    h_p = w_p = int(np.sqrt(patch_tokens.shape[0]))
    trans_act = trans_act.reshape(h_p, w_p)
    im = ax.imshow(trans_act, cmap='plasma')
    ax.set_title("Transformer Spatial Attention (Mean)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.axis('off')

    # 4. Ground Truth Mask
    ax = axes[1, 0]
    im = ax.imshow(mask[0], cmap='jet', vmin=0, vmax=args.num_classes-1)
    ax.set_title("Ground Truth Mask")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.axis('off')
    
    # 5. Predicted Mask
    ax = axes[1, 1]
    im = ax.imshow(pred[0], cmap='jet', vmin=0, vmax=args.num_classes-1)
    ax.set_title("Predicted Mask")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.axis('off')

    # 6. Error Map (Difference)
    ax = axes[1, 2]
    error_map = (mask[0] != pred[0]).astype(int)
    im = ax.imshow(error_map, cmap='Reds', vmin=0, vmax=1)
    ax.set_title("Error Map (Red=Error)")
    ax.axis('off')

    plt.tight_layout()
    # Save the figure
    filename = f"vis_{args.data_type}_{args.augment}_sample{sample_idx:03d}.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {filepath}")

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Visualize Intermediate Features and Predictions of HSRN')
    parser.add_argument('--data_type', type=str, required=True, choices=['rgb', 'hsi'], help='rgb or hsi')
    parser.add_argument('--augment', type=str, required=True, choices=['none', 'copypaste', 'cutmix'], help='none, copypaste, cutmix')
    parser.add_argument('--n_components', type=int, default=214, help='Number of HSI bands (default 214)')
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--dataset_root', type=str, default='/home/bs_thesis/Documents/BS_THESIS/DATASETS/PCBDataset')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples to visualize')
    args = parser.parse_args()

    # Create visualization output directory
    vis_dir = os.path.join(SCRIPT_DIR, 'Results', 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)

    print(f"--- Visualization: {args.data_type.upper()} with {args.augment} ---")
    
    # Load Data and Model
    test_loader, model, in_ch = get_loader_and_model(args)
    
    # Register hooks on intermediate layers
    # We want: 
    # 1. Output of Spectral Encoder (before patch tokenization)
    model.spectral_enc.register_forward_hook(get_features('spectral_enc'))
    # 2. Output of Transformer (before segmentation decoder)
    model.transformer.register_forward_hook(get_features('transformer'))
    
    model.eval()
    
    with torch.no_grad():
        for i, (images, masks) in enumerate(test_loader):
            if i >= args.num_samples:
                break
                
            # Forward Pass
            # images: (1, C, H, W)
            # masks: (1, H, W)
            images_input = images.to(device)
            # This triggers the hooks and populates the `features` dict
            outputs = model(images_input)
            
            # Get Predictions
            preds = outputs.argmax(dim=1).cpu().numpy()
            images_np = images.cpu().numpy()
            masks_np = masks.cpu().numpy()
            
            # Visualize
            visualize_sample(
                image=images_np, 
                mask=masks_np, 
                pred=preds, 
                spectral_features=features['spectral_enc'], 
                transformer_features=features['transformer'], 
                sample_idx=i, 
                save_dir=vis_dir,
                args=args
            )

if __name__ == '__main__':
    main()
