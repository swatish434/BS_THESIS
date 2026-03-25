import os
import torch
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from matplotlib.colors import ListedColormap

from hsrn_model import HybridSSRNSeg
from train_hsrn_experiments import PCBDataset, HSIPCADataset

def compute_ssim_map(pred, target, window_size=11, max_val=1.0):
    """
    Computes a pixel-wise SSIM error map (H, W) for visualization.
    """
    # Simplified SSIM parameters
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2
    
    # Needs float blocks (1, 1, H, W)
    if pred.dim() == 2:
        pred = pred.unsqueeze(0).unsqueeze(0).float()
    if target.dim() == 2:
        target = target.unsqueeze(0).unsqueeze(0).float()
        
    pred = pred / max_val
    target = target / max_val

    mu1 = F.avg_pool2d(pred, window_size, stride=1, padding=window_size//2)
    mu2 = F.avg_pool2d(target, window_size, stride=1, padding=window_size//2)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.avg_pool2d(pred * pred, window_size, stride=1, padding=window_size//2) - mu1_sq
    sigma2_sq = F.avg_pool2d(target * target, window_size, stride=1, padding=window_size//2) - mu2_sq
    sigma12 = F.avg_pool2d(pred * target, window_size, stride=1, padding=window_size//2) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.squeeze().cpu().numpy() # Return (H, W) array

def main():
    parser = argparse.ArgumentParser(description='Visualize GIQA Metrics (SSIM Maps)')
    parser.add_argument('--data_type', type=str, default='rgb', choices=['rgb', 'hsi'])
    parser.add_argument('--augment', type=str, default='none', choices=['none', 'copypaste', 'cutmix'])
    parser.add_argument('--num_samples', type=int, default=3)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading {args.data_type.upper()} model with {args.augment} augmentation...")

    # Paths
    base_dir = '/home/bs_thesis/Documents/BS_THESIS/PCBVision'
    if args.data_type == 'rgb':
        data_dir = os.path.join(base_dir, '01_Data', 'data')
        dataset = PCBDataset(data_dir=data_dir, split='test')
        nc = 3
    else:
        data_dir = os.path.join(base_dir, '01_Data', 'data_hsi')
        dataset = HSIPCADataset(data_dir=data_dir, split='test')
        nc = 214

    # The dataloader was causing IPC freezes so we are grabbing directly from the set
    # because we only need singular image extracts for visualization

    weight_path = os.path.join(base_dir, '02_Experiments', 'HSRN', 'Results', f'hsrn_{args.data_type}_nc{nc}_{args.augment}_best.pth')
    
    # Colors for PCB semantic classes
    colors = ['black', 'green', 'gold', 'silver']
    cmap = ListedColormap(colors)

    if not os.path.exists(weight_path):
        print(f"Weight path {weight_path} not found. Ensure the model has trained.")
        return

    # Load Model
    model = HybridSSRNSeg(in_channels=nc, num_classes=4, image_size=256).to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
    model.eval()

    save_dir = os.path.join(base_dir, '02_Experiments', 'HSRN', 'Results', 'visualizations', 'giqa')
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for i in range(args.num_samples):
            # Select random index
            idx = np.random.randint(0, len(dataset))
            img_tensor, mask_tensor = dataset[idx]
            
            img_input = img_tensor.unsqueeze(0).to(device)
            mask_input = mask_tensor.unsqueeze(0).to(device)
            
            # Predict
            logits = model(img_input)
            pred_mask = torch.argmax(logits, dim=1).squeeze(0) # (H, W)
            
            # Compute raw SSIM map 
            # We pass raw class indices (0..3) with max_val=3.0 so it maps cleanly
            ssim_heatmap = compute_ssim_map(pred_mask.cpu(), mask_tensor.cpu(), max_val=3.0)
            
            # Calculate overall metrics for display
            overall_ssim = ssim_heatmap.mean()
            mse = F.mse_loss((pred_mask.float()/3.0).cpu(), (mask_tensor.float()/3.0).cpu())
            psnr = 20 * math.log10(1.0) - 10 * math.log10(mse.item()) if mse.item() > 0 else 100.0

            # ---------------------------------------------------------
            # PLOT: 1x4 (Original, Ground Truth, Prediction, SSIM Map)
            # ---------------------------------------------------------
            fig, axes = plt.subplots(1, 4, figsize=(22, 5))
            
            # 1. Original Image representation
            if args.data_type == 'rgb':
                disp_img = img_tensor.permute(1, 2, 0).cpu().numpy()
                disp_img = (disp_img - disp_img.min()) / (disp_img.max() - disp_img.min() + 1e-8)
            else:
                # HSI False color (using arbitrary bands 10, 50, 100)
                disp_img = img_tensor[[10, 50, 100], :, :].permute(1, 2, 0).cpu().numpy()
                disp_img = (disp_img - disp_img.min()) / (disp_img.max() - disp_img.min() + 1e-8)

            axes[0].imshow(disp_img)
            axes[0].set_title(f"Input ({args.data_type.upper()})")
            axes[0].axis('off')

            # 2. Ground Truth
            gt_np = mask_tensor.cpu().numpy()
            axes[1].imshow(gt_np, cmap=cmap, vmin=0, vmax=3)
            axes[1].set_title("Ground Truth Mask")
            axes[1].axis('off')

            # 3. Model Prediction
            pred_np = pred_mask.cpu().numpy()
            axes[2].imshow(pred_np, cmap=cmap, vmin=0, vmax=3)
            axes[2].set_title(f"Predicted Mask\n(PSNR: {psnr:.2f} dB)")
            axes[2].axis('off')

            # 4. SSIM Heatmap
            # The heatmap generally floats between -1 and 1. High SSIM (1.0) = no error.
            im = axes[3].imshow(ssim_heatmap, cmap='inferno_r', vmin=0.5, vmax=1.0)
            axes[3].set_title(f"SSIM Error Heatmap\n(Mean SSIM: {overall_ssim:.4f})")
            axes[3].axis('off')
            fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

            plt.tight_layout()
            out_file = os.path.join(save_dir, f'giqa_{args.data_type}_{args.augment}_sample{i:03d}.png')
            plt.savefig(out_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Saved visualization to: {out_file}")

    print("GIQA Visualization Complete.")

if __name__ == '__main__':
    main()
