#!/usr/bin/env python3
"""
Unified Evaluation Script for PCBVision Models
Computes: Pixel Acc, mIoU, Mean F1, Mean Precision, Mean Recall, Kappa
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score, f1_score, precision_score, recall_score, confusion_matrix
from spectral import envi
from einops import rearrange
from torch.utils.data import Dataset

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Imports
from models.DeepLabv3_plus import DeepLabv3_plus
from models.FusionModel import RGBHSIFusionModel
from utils.dataset_functions import read_dataset, PCBFullDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================
# ViT-MoE Classes (Copied to avoid import issues)
# ============================================================

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model=256, num_heads=8):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.proj = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, L, D)
        x = self.proj(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, d_model=256, num_heads=8, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, int(d_model * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(d_model * mlp_ratio), d_model)
        )
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class ViTExpert(nn.Module):
    def __init__(self, d_model=256, num_layers=2, num_heads=8):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads)
            for _ in range(num_layers)
        ])
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

class Router(nn.Module):
    def __init__(self, d_model=256, num_experts=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, num_experts)
        )
    def forward(self, x):
        logits = self.fc(x)
        weights = F.softmax(logits, dim=-1)
        return weights, logits

class MoELayer(nn.Module):
    def __init__(self, d_model=256, num_experts=4, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([
            ViTExpert(d_model=d_model) for _ in range(num_experts)
        ])
        self.router = Router(d_model=d_model, num_experts=num_experts)
        
    def forward(self, x):
        B, L, D = x.shape
        router_weights, router_logits = self.router(x)
        topk_weights, topk_indices = torch.topk(router_weights, self.top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        output = torch.zeros_like(x)
        for i in range(self.top_k):
            expert_idx = topk_indices[:, :, i]
            expert_weight = topk_weights[:, :, i:i+1]
            for e in range(self.num_experts):
                mask = (expert_idx == e)
                if mask.any():
                    expert_out = self.experts[e](x)
                    output += expert_out * expert_weight * mask.unsqueeze(-1)
        return output, router_weights, router_logits

class ViTMoESegmentation(nn.Module):
    def __init__(self, in_channels=214, num_classes=4, d_model=256, num_experts=4, num_layers=2, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        self.num_classes = num_classes
        self.patch_embed = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)
        self.moe_layers = nn.ModuleList([
            MoELayer(d_model=d_model, num_experts=num_experts, top_k=2)
            for _ in range(num_layers)
        ])
        self.seg_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, num_classes)
        )
        
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.patch_embed(x)
        _, D, H_p, W_p = x.shape
        x = rearrange(x, 'b d h w -> b (h w) d')
        all_router_weights = []
        for moe_layer in self.moe_layers:
            x, router_weights, _ = moe_layer(x)
            all_router_weights.append(router_weights)
        logits = self.seg_head(x)
        logits = rearrange(logits, 'b (h w) c -> b c h w', h=H_p, w=W_p)
        logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)
        return logits, all_router_weights

# Dataset class imported from utils.dataset_functions

# ============================================================
# Metric Computation
# ============================================================

def compute_all_metrics(preds, targets, num_classes=4):
    """Compute all required metrics"""
    preds_flat = preds.flatten()
    targets_flat = targets.flatten()
    
    # Pixel Accuracy
    acc = (preds_flat == targets_flat).mean()
    
    # Kappa
    kappa = cohen_kappa_score(targets_flat, preds_flat)
    
    # Macro metrics (Mean F1, Precision, Recall)
    # labels=[0,1,2,3] ensures all 4 classes are considered even if missing in batch
    f1 = f1_score(targets_flat, preds_flat, average='macro', labels=range(num_classes))
    precision = precision_score(targets_flat, preds_flat, average='macro', labels=range(num_classes), zero_division=0)
    recall = recall_score(targets_flat, preds_flat, average='macro', labels=range(num_classes), zero_division=0)
    
    # mIoU
    cm = confusion_matrix(targets_flat, preds_flat, labels=range(num_classes))
    intersection = np.diag(cm)
    union = cm.sum(axis=1) + cm.sum(axis=0) - intersection
    iou = intersection / (union + 1e-10)
    miou = np.nanmean(iou)
    
    return {
        'Pixel Accuracy': acc,
        'Mean IoU': miou,
        'Mean F1 Score': f1,
        'Mean Precision': precision,
        'Mean Recall': recall,
        'Kappa': kappa
    }

def print_metrics(name, metrics):
    print(f"\n--- {name} Results ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    print("-" * 30)

# ============================================================
# Evaluation Functions
# ============================================================

def evaluate_rgb_model(model_path, dataset_root, model_name="RGB Model", mean=None, std=None, indices=None):
    print(f"\nEvaluating {model_name}...")
    
    if not os.path.exists(model_path):
        print(f"Error: Checkpoint not found at {model_path}")
        return
        
    model = DeepLabv3_plus(nInputChannels=3, n_classes=4).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    except Exception as e:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'model_state' in checkpoint:
            model.load_state_dict(checkpoint['model_state'])
        else:
            print(f"Error loading weights: {e}")
            return

    model.eval()
    
    # Use provided indices or default to test set
    if indices is None:
        indices = [2, 5, 6, 7, 9, 10, 12, 13, 14, 15, 16, 19, 20, 21, 26, 27, 28, 29, 30, 31, 33, 36, 38, 39, 40, 41, 43, 46, 48, 51]
    
    # Note: Added trailing slash to ensure path concatenation works
    _, general_masks, _, rgb, _, _, _ = read_dataset(dataset_root if dataset_root.endswith('/') else dataset_root + '/')
    
    all_preds = []
    all_targets = []
    img_size = 512
    
    # Validation Normalization (0-255 -> (x - mean)/std -> Result)
    # But usually torchvision.transforms.Normalize takes 0-1 tensor.
    # Our mean/std are in 0-255 scale.
    # So: Tensor(0-1) * 255 - mean / std?
    # Or: Tensor(0-1) - (mean/255) / (std/255)
    
    norm_mean = torch.tensor(mean).view(3, 1, 1).to(device) if mean else None
    norm_std = torch.tensor(std).view(3, 1, 1).to(device) if std else None
    
    # Adjust for 0-1 range
    if norm_mean is not None:
        norm_mean = norm_mean / 255.0
        norm_std = norm_std / 255.0
    
    print(f"DEBUG: evaluating on {len(indices)} indices: {indices}")
    for idx in tqdm(indices):
        if idx >= len(rgb) or rgb[idx] is None:
            # print(f"Skipping index {idx}, data is None")
            continue
            
        img = rgb[idx]
        mask = general_masks[idx]
        
        img_resized = cv2.resize(img, (img_size, img_size))
        mask_resized = cv2.resize(mask.astype(np.uint8), (img_size, img_size), interpolation=cv2.INTER_NEAREST)
        
        # To tensor (0-1)
        img_tensor = torch.from_numpy(img_resized.transpose(2, 0, 1).astype(np.float32) / 255.0).unsqueeze(0).to(device)
        
        # Normalize if needed
        if norm_mean is not None and norm_std is not None:
            img_tensor = (img_tensor - norm_mean) / norm_std
        
        with torch.no_grad():
            output = model(img_tensor)
            pred = output.argmax(dim=1).squeeze().cpu().numpy()
            
        all_preds.append(pred)
        all_targets.append(mask_resized)
        
    if not all_preds:
        print("No predictions generated! Check dataset path.")
        return

    # Visualization of first 5 samples
    print("Saving visualizations for first 5 samples...")
    viz_dir = os.path.join(os.path.dirname(model_path), "predictions")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Class colors: Background (Black), Component (Red), Capacitor (Green), Connector (Blue)
    colors = np.array([
        [0, 0, 0],
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255]
    ], dtype=np.uint8)
    
    # We need to access individual samples again for visualization, 
    # but the loop above already processed them.
    # To avoid reloading, we can just reload the specific indices we want to visualize.
    
    for i in range(min(5, len(indices))):
        idx = indices[i]
        
        # Reload raw image and mask for visualization
        if idx >= len(rgb) or rgb[idx] is None: continue
            
        img = rgb[idx]
        mask = general_masks[idx]
        
        pred_mask = all_preds[i] # This is already the prediction for indices[i] because all_preds is appended in order
        
        # img is BGR or RGB? opencv reads BGR. 
        # But read_dataset uses what? 
        # dataset_functions.read_dataset uses cv2.imread -> BGR.
        # So we can just use it directly, but wait, we need to resize it to 512x512 like we did for inference
        img_resized = cv2.resize(img, (img_size, img_size))
        mask_resized = cv2.resize(mask.astype(np.uint8), (img_size, img_size), interpolation=cv2.INTER_NEAREST)
        
        # Colorize masks
        pred_color = colors[pred_mask]
        gt_color = colors[mask_resized]
        
        # img_resized is BGR (from cv2). colors are RGB (defined above).
        # We should convert colors to BGR or img to RGB. 
        # Let's convert valid masks to BGR for consistency with cv2.imwrite
        pred_color = cv2.cvtColor(pred_color, cv2.COLOR_RGB2BGR)
        gt_color = cv2.cvtColor(gt_color, cv2.COLOR_RGB2BGR)
        
        # Save side-by-side
        # RGB | GT | Pred
        combined = np.hstack((img_resized, gt_color, pred_color))
        cv2.imwrite(os.path.join(viz_dir, f"sample_{idx}_viz.png"), combined)
        
    print(f"Visualizations saved to {viz_dir}")

    all_preds_np = np.concatenate([p.flatten() for p in all_preds])
    all_targets_np = np.concatenate([t.flatten() for t in all_targets])
    
    metrics = compute_all_metrics(all_preds_np, all_targets_np)
    print_metrics(model_name, metrics)
    return metrics

def evaluate_vitmoe(model_path, dataset_root, model_name="ViT-MoE"):
    print(f"\nEvaluating {model_name}...")
    
    if not os.path.exists(model_path):
        print(f"Error: Checkpoint not found at {model_path}")
        return

    model = ViTMoESegmentation(
        in_channels=214,
        num_classes=4,
        d_model=256,
        num_experts=4,
        num_layers=2,
        patch_size=16
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()
    
    all_sample_ids = list(range(1, 54))
    # Use same random_state=42 as training for consistent split
    from sklearn.model_selection import train_test_split
    _, val_ids = train_test_split(all_sample_ids, test_size=0.2, random_state=42)
    
    dataset = PCBFullDataset(
        val_ids, dataset_root,
        target_size=(512, 512),
        augment=False, normalize=True,
        copy_paste_aug=None
    )
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    
    all_preds = []
    all_targets = []
    
    print(f"Evaluating on {len(dataloader)} HSI samples...")
    for batch in tqdm(dataloader):
        hsi = batch['hsi'].to(device)
        # Slice to 214 channels if input is 224
        if hsi.shape[1] > 214:
            hsi = hsi[:, :214, :, :]
        mask = batch['mask'].to(device)
        
        with torch.no_grad():
            logits, _ = model(hsi)
            pred = logits.argmax(dim=1).cpu().numpy()
            mask = mask.cpu().numpy()
            
        all_preds.append(pred)
        all_targets.append(mask)
        
    all_preds_np = np.concatenate([p.flatten() for p in all_preds])
    all_targets_np = np.concatenate([t.flatten() for t in all_targets])
    
    metrics = compute_all_metrics(all_preds_np, all_targets_np)
    print_metrics(model_name, metrics)
    return metrics

def evaluate_fusion_model(model_path, dataset_root, model_name="Fusion Model", indices=None):
    print(f"\nEvaluating {model_name}...")
    
    if not os.path.exists(model_path):
        print(f"Error: Checkpoint not found at {model_path}")
        return

    model = RGBHSIFusionModel(num_classes=4, hsi_channels=214).to(device)
    
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    except Exception as e:
        print(f"Error loading weights: {e}")
        return
        
    model.eval()
    
    # Use provided indices or default to test set
    if indices is None:
        indices = [2, 5, 6, 7, 9, 10, 12, 13, 14, 15, 16, 19, 20, 21, 26, 27, 28, 29, 30, 31, 33, 36, 38, 39, 40, 41, 43, 46, 48, 51]
    
    print(f"Evaluation Indices ({len(indices)}): {indices}")
    
    # We need PCBFullDataset logic but for specific indices
    dataset = PCBFullDataset(indices, dataset_root, augment=False, normalize=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    
    all_preds = []
    all_targets = []
    
    print(f"Evaluating on {len(dataloader)} samples...")
    for batch in tqdm(dataloader):
        rgb = batch['rgb'].to(device)
        hsi = batch['hsi'].to(device)
        mask = batch['mask'].to(device)
        
        # Slice HSI if needed
        if hsi.shape[1] > 214:
            hsi = hsi[:, :214, :, :]
            
        with torch.no_grad():
            output = model(rgb, hsi)
            pred = output.argmax(dim=1).cpu().numpy()
            mask = mask.cpu().numpy()
            
        all_preds.append(pred)
        all_targets.append(mask)
        
    # Visualization of first 5 samples
    print("Saving visualizations for first 5 samples...")
    viz_dir = os.path.join(os.path.dirname(model_path), "predictions")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Class colors: Background (Black), Component (Red), Capacitor (Green), Connector (Blue)
    colors = np.array([
        [0, 0, 0],
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255]
    ], dtype=np.uint8)
    
    for i in range(min(5, len(indices))):
        idx = indices[i]
        
        # Get raw data
        sample = dataset[i] # This returns dict with tensors
        rgb_tensor = sample['rgb']
        mask_tensor = sample['mask']
        
        # Pred and GT from lists (collected during loop)
        # Note: all_preds[i] is (1, H, W) or (H, W)?
        # In loop: pred = output.argmax(dim=1).cpu().numpy() -> (B, H, W) -> since batch=1, it is (1, H, W)
        pred_mask = all_preds[i].squeeze()
        gt_mask = all_targets[i].squeeze()
        
        # Un-normalize RGB for display
        rgb_img = rgb_tensor.permute(1, 2, 0).cpu().numpy()
        rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min() + 1e-8)
        rgb_img = (rgb_img * 255).astype(np.uint8)
        
        # Colorize masks
        pred_color = colors[pred_mask]
        gt_color = colors[gt_mask]
        
        # Save side-by-side
        # RGB | GT | Pred
        # Resize masks to match RGB if needed (they should match though)
        if pred_color.shape[:2] != rgb_img.shape[:2]:
             pred_color = cv2.resize(pred_color, (rgb_img.shape[1], rgb_img.shape[0]), interpolation=cv2.INTER_NEAREST)
             gt_color = cv2.resize(gt_color, (rgb_img.shape[1], rgb_img.shape[0]), interpolation=cv2.INTER_NEAREST)

        combined = np.hstack((rgb_img, gt_color, pred_color))
        cv2.imwrite(os.path.join(viz_dir, f"sample_{idx}_viz.png"), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
        
    print(f"Visualizations saved to {viz_dir}")

    all_preds_np = np.concatenate([p.flatten() for p in all_preds])
    all_targets_np = np.concatenate([t.flatten() for t in all_targets])
    
    metrics = compute_all_metrics(all_preds_np, all_targets_np)
    print_metrics(model_name, metrics)
    return metrics

# Add global split variable or pass it down. Better to pass it down but changing signatures is invasive.
# Simple hack: Use a global variable or modify the dataset initialization inside the functions.
# Since the functions are defined above, I need to modify them to accept indices or split name.

# Let's check define `check_split` helper
def get_indices(split_name):
    if split_name == 'test':
        return [2, 5, 6, 7, 9, 10, 12, 13, 14, 15, 16, 19, 20, 21, 26, 27, 28, 29, 30, 31, 33, 36, 38, 39, 40, 41, 43, 46, 48, 51]
    elif split_name == 'val':
        return [1, 23, 24, 42, 45]
    else:
        raise ValueError("Invalid split name")

# I need to update the evaluation functions to use this.
# Since I can't easily change all function signatures in one REPLACE block without targeting widespread lines, 
# I will use a global variable strategy at the top of the file or main block, BUT `evaluate_unified` functions use local `test_indices`.
# I must update `evaluate_fusion_model` and others to use the indices passed or global.

# I will update `evaluate_fusion_model` first as it's the target.

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Unified Evaluation Script")
    parser.add_argument('--model', type=str, choices=['sdlora', 'sim2real', 'vitmoe', 'fusion', 'cp', 'all'], default='all', help="Model to evaluate")
    parser.add_argument('--dataset_root', type=str, default="/home/bs_thesis/Documents/BS_THESIS/DATASETS/PCBDataset/", help="Dataset root directory")
    parser.add_argument('--split', type=str, choices=['test', 'val'], default='test', help="Data split to evaluate on")
    args = parser.parse_args()
    
    target_indices = get_indices(args.split)
    dataset_root = args.dataset_root
    
    # 1. RGB DeepLabv3+ (SD-LoRA)
    if args.model in ['sdlora', 'all']:
        sd_lora_path = "/home/bs_thesis/Documents/BS_THESIS/PCBVision/RGB_Experiments/runs/20260204_165437_rgb_deeplabv3+_sdlora/RGB_deeplabv3+_best.pth"
        mean_val = [49.378013076782224, 41.3470613861084, 28.656757125854494]
        std_val = [44.75258714091906, 39.29804389076184, 26.90310023915742]
        evaluate_rgb_model(sd_lora_path, dataset_root, f"RGB DeepLabv3+ (CutMix + SD-LoRA) ({args.split.upper()})", mean=mean_val, std=std_val, indices=target_indices)
    
    # 1b. RGB DeepLabv3+ (Copy-Paste)
    if args.model in ['cp', 'all']:
        cp_path = "/home/bs_thesis/Documents/BS_THESIS/PCBVision/RGB_Experiments/runs/20260202_172507_rgb_deeplabv3+_cp/best_val_loss.pt"
        # Same mean/std as SD-LoRA based on config
        mean_val = [49.378013076782224, 41.3470613861084, 28.656757125854494]
        std_val = [44.75258714091906, 39.29804389076184, 26.90310023915742]
        evaluate_rgb_model(cp_path, dataset_root, f"RGB DeepLabv3+ (Copy-Paste) ({args.split.upper()})", mean=mean_val, std=std_val, indices=target_indices)
    
    # 2. Sim2Real
    if args.model in ['sim2real', 'all']:
        sim2real_path = "/home/bs_thesis/Documents/BS_THESIS/PCBVision/Results/Sim2Real/stage2_real_finetuned.pth"
        # evaluate_rgb_model(sim2real_path, dataset_root, "RGB DeepLabv3+ (Sim2Real)")
        # Note: Sim2Real disabled by default as per previous instructions, uncomment if needed
        # Or add logic to force it if explicitly requested
        if args.model == 'sim2real':
            evaluate_rgb_model(sim2real_path, dataset_root, "RGB DeepLabv3+ (Sim2Real)")
    
    # 3. ViT-MoE
    if args.model in ['vitmoe', 'all']:
        vitmoe_path = "/home/bs_thesis/Documents/BS_THESIS/PCBVision/MambaMoE_Experiments/vitmoe_patches_results/checkpoints/best_model.pth"
        evaluate_vitmoe(vitmoe_path, dataset_root, "HSI ViT-MoE (Patches)")

    # 4. Fusion Model
    if args.model in ['fusion', 'all']:
        fusion_path = "/home/bs_thesis/Documents/BS_THESIS/PCBVision/Results/Fusion/fusion_best.pth"
        if os.path.exists(fusion_path):
            evaluate_fusion_model(fusion_path, dataset_root, f"RGB-HSI Late Fusion ({args.split.upper()})", indices=target_indices)
        else:
            print(f"Fusion model not found at {fusion_path}. Skipping.")
