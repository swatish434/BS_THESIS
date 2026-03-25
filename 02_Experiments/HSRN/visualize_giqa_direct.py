import os
import torch
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from PIL import Image
from matplotlib.colors import ListedColormap

from hsrn_model import HybridSSRNSeg

def compute_ssim_map(pred, target, window_size=11, max_val=1.0):
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2
    
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
    return ssim_map.squeeze().cpu().numpy()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Creating direct GIQA visuals using {device}")

    base_dir = '/home/bs_thesis/Documents/BS_THESIS/PCBVision'
    weight_path = os.path.join(base_dir, '02_Experiments', 'HSRN', 'Results', 'hsrn_rgb_nc3_none_best.pth')
    
    # Randomly selected images from test set folder directly to bypass dataset loader
    img_dir = os.path.join(base_dir, '01_Data', 'data', 'images', 'test')
    mask_dir = os.path.join(base_dir, '01_Data', 'data', 'annotations', 'test')
    
    files = os.listdir(img_dir)
    sample_imgs = [files[0], files[5]]

    colors = ['black', 'green', 'gold', 'silver']
    cmap = ListedColormap(colors)

    model = HybridSSRNSeg(in_channels=3, num_classes=4, image_size=256).to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
    model.eval()

    save_dir = os.path.join(base_dir, '02_Experiments', 'HSRN', 'Results', 'visualizations', 'giqa')
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for i, f in enumerate(sample_imgs):
            img_path = os.path.join(img_dir, f)
            mask_path = os.path.join(mask_dir, f.replace('.jpg', '.png'))

            # Raw NumPy load
            img_pil = Image.open(img_path).convert('RGB')
            mask_pil = Image.open(mask_path).convert('L') # ensure grayscale indices

            img_np = np.array(img_pil) # (256, 256, 3)
            mask_np = np.array(mask_pil) # (256, 256)

            # Map pixel values to classes cleanly using standard RGB pipeline logic
            mask_np[mask_np > 3] = 0

            # Convert to tensors
            img_tensor = torch.from_numpy(img_np).float().permute(2, 0, 1) / 255.0
            mask_tensor = torch.from_numpy(mask_np).long()

            img_input = img_tensor.unsqueeze(0).to(device)
            mask_input = mask_tensor.unsqueeze(0).to(device)
            
            logits = model(img_input)
            pred_mask = torch.argmax(logits, dim=1).squeeze(0)
            
            ssim_heatmap = compute_ssim_map(pred_mask.cpu(), mask_tensor.cpu(), max_val=3.0)
            overall_ssim = ssim_heatmap.mean()
            mse = F.mse_loss((pred_mask.float()/3.0).cpu(), (mask_tensor.float()/3.0).cpu())
            psnr = 20 * math.log10(1.0) - 10 * math.log10(mse.item()) if mse.item() > 0 else 100.0

            fig, axes = plt.subplots(1, 4, figsize=(22, 5))
            
            disp_img = img_tensor.permute(1, 2, 0).cpu().numpy()
            disp_img = (disp_img - disp_img.min()) / (disp_img.max() - disp_img.min() + 1e-8)

            axes[0].imshow(disp_img)
            axes[0].set_title(f"Input RGB")
            axes[0].axis('off')

            axes[1].imshow(mask_np, cmap=cmap, vmin=0, vmax=3)
            axes[1].set_title("Ground Truth Mask")
            axes[1].axis('off')

            pred_np = pred_mask.cpu().numpy()
            axes[2].imshow(pred_np, cmap=cmap, vmin=0, vmax=3)
            axes[2].set_title(f"Predicted Mask\n(PSNR: {psnr:.2f} dB)")
            axes[2].axis('off')

            im = axes[3].imshow(ssim_heatmap, cmap='inferno_r', vmin=0.5, vmax=1.0)
            axes[3].set_title(f"SSIM Error Heatmap\n(Mean SSIM: {overall_ssim:.4f})")
            axes[3].axis('off')
            fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

            plt.tight_layout()
            out_file = os.path.join(save_dir, f'giqa_direct_rgb_sample{i:03d}.png')
            plt.savefig(out_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Saved: {out_file}")

if __name__ == '__main__':
    main()
