import os
import torch
import math
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from hsrn_model import HybridSSRNSeg
from train_hsrn_experiments import PCBDataset, HSIPCADataset

def compute_psnr(pred, target, max_val=1.0):
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return 100.0
    return 20 * math.log10(max_val) - 10 * math.log10(mse.item())

def compute_ssim(pred, target, window_size=11, max_val=1.0):
    # Simplified SSIM for single channel masks
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2
    
    mu1 = F.avg_pool2d(pred, window_size, stride=1, padding=window_size//2)
    mu2 = F.avg_pool2d(target, window_size, stride=1, padding=window_size//2)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.avg_pool2d(pred * pred, window_size, stride=1, padding=window_size//2) - mu1_sq
    sigma2_sq = F.avg_pool2d(target * target, window_size, stride=1, padding=window_size//2) - mu2_sq
    sigma12 = F.avg_pool2d(pred * target, window_size, stride=1, padding=window_size//2) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean().item()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    base_dir = '/home/bs_thesis/Documents/BS_THESIS/PCBVision'
    rgb_data_dir = os.path.join(base_dir, '01_Data', 'data')
    hsi_data_dir = os.path.join(base_dir, '01_Data', 'data_hsi')
    results_dir = os.path.join(base_dir, '02_Experiments', 'HSRN', 'Results')

    experiments = [
        {'name': 'Exp 01: RGB - None',      'type': 'rgb', 'aug': 'none',      'nc': 3,   'weights': 'hsrn_rgb_nc3_none_best.pth'},
        {'name': 'Exp 02: RGB - CopyPaste', 'type': 'rgb', 'aug': 'copypaste', 'nc': 3,   'weights': 'hsrn_rgb_nc3_copypaste_best.pth'},
        {'name': 'Exp 03: RGB - CutMix',    'type': 'rgb', 'aug': 'cutmix',    'nc': 3,   'weights': 'hsrn_rgb_nc3_cutmix_best.pth'},
        {'name': 'Exp 04: HSI - None',      'type': 'hsi', 'aug': 'none',      'nc': 214, 'weights': 'hsrn_hsi_nc214_none_best.pth'},
        {'name': 'Exp 05: HSI - CopyPaste', 'type': 'hsi', 'aug': 'copypaste', 'nc': 214, 'weights': 'hsrn_hsi_nc214_copypaste_best.pth'},
        {'name': 'Exp 06: HSI - CutMix',    'type': 'hsi', 'aug': 'cutmix',    'nc': 214, 'weights': 'hsrn_hsi_nc214_cutmix_best.pth'},
    ]

    print("\nStarting GIQA Evaluation...")
    print("=" * 60)
    print(f"{'Experiment':<25} | {'SSIM':<10} | {'PSNR':<10}")
    print("-" * 60)

    for exp in experiments:
        weight_path = os.path.join(results_dir, exp['weights'])
        
        if not os.path.exists(weight_path):
            print(f"{exp['name']:<25} | {'Pending':<10} | {'Pending':<10}")
            continue

        if exp['type'] == 'rgb':
            test_dataset = PCBDataset(data_dir=rgb_data_dir, split='test')
        else:
            test_dataset = HSIPCADataset(data_dir=hsi_data_dir, split='test')
            
        test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)

        model = HybridSSRNSeg(in_channels=exp['nc'], num_classes=4, image_size=256).to(device)
        model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
        model.eval()

        total_ssim = 0.0
        total_psnr = 0.0
        batches = 0

        with torch.no_grad():
            from tqdm import tqdm
            for images, masks in tqdm(test_loader, desc=exp['name']):
                images = images.to(device)
                outputs = model(images)  
                preds = torch.argmax(outputs, dim=1) 

                preds_norm = (preds.unsqueeze(1).float() / 3.0).to(device)
                masks_norm = (masks.unsqueeze(1).float() / 3.0).to(device)
                
                total_ssim += compute_ssim(preds_norm, masks_norm)
                total_psnr += compute_psnr(preds_norm, masks_norm)
                batches += 1
                if batches >= 5:
                    break

        final_ssim = total_ssim / batches
        final_psnr = total_psnr / batches
        print(f"{exp['name']:<25} | {final_ssim:.4f}     | {final_psnr:.4f}")
        
        print(f"{exp['name']:<25} | {final_ssim:.4f}     | {final_psnr:.4f}")

    print("=" * 60)

if __name__ == '__main__':
    main()
