"""
visualize_results.py
────────────────────
Post-training visualization script.
Generates GT vs Prediction comparisons and intermediate feature maps.

Usage:
    python visualize_results.py --exp_id 5 --model_name "Hybrid SSRN-ViT" \
                                 --modality RGB --num_samples 8
"""

import os
import sys
import glob
import argparse
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torch.utils.data import DataLoader

# ── Add project root to path ──────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from config  import Config
from dataset import PCB_Dataset
from models  import build_model

# ── Class colour map (4 classes) ─────────────────────────────────────────────
CLASS_COLORS = np.array([
    [0,   0,   0  ],   # 0 → Background  (black)
    [255, 165, 0  ],   # 1 → IC          (orange)
    [0,   255, 255],   # 2 → Connector   (cyan)
    [255, 0,   255],   # 3 → Capacitor   (magenta)
], dtype=np.uint8)

CLASS_NAMES = ['Background', 'IC', 'Connector', 'Capacitor']


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    """Convert (H, W) int mask → (H, W, 3) RGB."""
    rgb = CLASS_COLORS[np.clip(mask, 0, len(CLASS_COLORS) - 1)]
    return rgb


def get_legend_patches():
    return [
        mpatches.Patch(color=CLASS_COLORS[i] / 255.0, label=CLASS_NAMES[i])
        for i in range(len(CLASS_NAMES))
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 1. GT vs Prediction grid
# ─────────────────────────────────────────────────────────────────────────────

def visualize_predictions(model, loader, config, save_dir, num_samples=8, exp_tag=''):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    collected = 0
    fig_rows  = []

    with torch.no_grad():
        for imgs, masks in loader:
            if collected >= num_samples:
                break

            imgs  = imgs.to(config.DEVICE)
            preds = model(imgs).argmax(dim=1).cpu().numpy()
            imgs  = imgs.cpu().numpy()
            masks = masks.numpy()

            for i in range(imgs.shape[0]):
                if collected >= num_samples:
                    break

                # ── RGB image (denormalise) ───────────────────────────────────
                img = imgs[i][:3]                          # first 3 channels
                img = np.clip(img.transpose(1, 2, 0), 0, 1)

                gt_rgb   = mask_to_rgb(masks[i])   / 255.0
                pred_rgb = mask_to_rgb(preds[i])   / 255.0

                # ── Overlay: prediction on image ─────────────────────────────
                overlay = img.copy()
                non_bg  = preds[i] > 0
                overlay[non_bg] = overlay[non_bg] * 0.5 + pred_rgb[non_bg] * 0.5

                fig_rows.append((img, gt_rgb, pred_rgb, overlay))
                collected += 1

    # ── Plot grid ────────────────────────────────────────────────────────────
    n      = len(fig_rows)
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    col_titles = ['Input RGB', 'Ground Truth', 'Prediction', 'Overlay']
    for c, title in enumerate(col_titles):
        axes[0, c].set_title(title, fontsize=13, fontweight='bold', pad=8)

    for r, (img, gt, pred, overlay) in enumerate(fig_rows):
        for c, arr in enumerate([img, gt, pred, overlay]):
            axes[r, c].imshow(arr)
            axes[r, c].axis('off')
        axes[r, 0].set_ylabel(f'Sample {r+1}', fontsize=10, rotation=0,
                              labelpad=50, va='center')

    # Legend
    legend = get_legend_patches()
    fig.legend(handles=legend, loc='lower center', ncol=len(CLASS_NAMES),
               fontsize=11, framealpha=0.9, bbox_to_anchor=(0.5, -0.01))

    plt.suptitle(f'Component Segmentation — GT vs Prediction — {exp_tag}', fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()

    out_path = os.path.join(save_dir, f'{exp_tag}_gt_vs_pred.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [✓] GT vs Prediction saved → {out_path}')
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# 2. Intermediate feature maps
# ─────────────────────────────────────────────────────────────────────────────

class FeatureHook:
    """Registers a forward hook and stores the activation."""
    def __init__(self, module):
        self.activation = None
        self.hook = module.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        self.activation = output.detach().cpu()

    def remove(self):
        self.hook.remove()


def visualize_feature_maps(model, sample_img, save_dir, exp_tag='', max_channels=16):
    """Extract and visualise intermediate activations from key layers."""
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    # ── Register hooks on key sub-modules ────────────────────────────────────
    hooks      = {}
    hook_names = {}

    def try_hook(name, module):
        if module is not None:
            hooks[name]      = FeatureHook(module)
            hook_names[name] = name

    # ── DeepLabV3+ (smp) — hook ResNet encoder stages + decoder ─────────────
    if hasattr(model, 'encoder') and hasattr(model, 'decoder') and not hasattr(model, 'vit'):
        enc = model.encoder
        for lname in ['layer1', 'layer2', 'layer3', 'layer4']:
            if hasattr(enc, lname):
                try_hook(f'encoder.{lname}', getattr(enc, lname))
        dec = model.decoder
        if hasattr(dec, 'aspp'):
            try_hook('decoder.aspp', dec.aspp)
        for name, module in dec.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                try_hook(f'decoder.{name}', module)
                break  # first conv only

    # ── Hybrid SSRN-ViT — hook SpectralSpatialEncoder ────────────────────────
    elif hasattr(model, 'encoder') and hasattr(model, 'vit'):
        enc = model.encoder
        for lname in ['conv1', 'res1', 'res2', 'res3', 'res4', 'conv2', 'conv3']:
            if hasattr(enc, lname):
                try_hook(f'ssrn.{lname}', getattr(enc, lname))

    # ── MambaHSI — hook Conv2d layers in encoder/decoder sequentials ─────────
    else:
        for part_name in ['encoder', 'decoder']:
            if hasattr(model, part_name):
                part = getattr(model, part_name)
                for i, layer in enumerate(part):
                    if isinstance(layer, torch.nn.Conv2d):
                        try_hook(f'{part_name}.conv{i}', layer)

    if not hooks:
        print('  [!] No hookable layers found — skipping feature maps.')
        return

    # ── Forward pass ─────────────────────────────────────────────────────────
    with torch.no_grad():
        _ = model(sample_img)

    # ── Plot each hooked layer ────────────────────────────────────────────────
    for name, hook in hooks.items():
        act = hook.activation  # (1, C, H, W) or (1, C, H, W, D) for 3D
        hook.remove()

        if act is None:
            continue

        # Handle 3D conv outputs → average over spectral dim
        if act.dim() == 5:
            act = act.mean(dim=-1)   # (1, C, H, W)

        act = act[0]                 # (C, H, W)
        C   = act.shape[0]
        n   = min(C, max_channels)

        # Normalise each channel to [0, 1]
        act_norm = act[:n].clone()
        for c in range(n):
            cmin, cmax = act_norm[c].min(), act_norm[c].max()
            if cmax > cmin:
                act_norm[c] = (act_norm[c] - cmin) / (cmax - cmin + 1e-8)

        # Grid layout
        cols = min(8, n)
        rows = (n + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
        axes = np.array(axes).reshape(-1)

        for c in range(n):
            fmap = act_norm[c].numpy()
            axes[c].imshow(fmap, cmap='viridis')
            axes[c].set_title(f'ch {c}', fontsize=7)
            axes[c].axis('off')

        for c in range(n, len(axes)):
            axes[c].axis('off')

        safe_name = name.replace('.', '_').replace('/', '_')
        plt.suptitle(f'Feature maps — {exp_tag} — {name}  ({C} ch total, showing {n})',
                     fontsize=10, fontweight='bold')
        plt.tight_layout()

        out_path = os.path.join(save_dir, f'{exp_tag}_features_{safe_name}.png')
        fig.savefig(out_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f'  [✓] Feature map ({name}) saved → {out_path}')


# ─────────────────────────────────────────────────────────────────────────────
# 3. Per-class IoU bar chart
# ─────────────────────────────────────────────────────────────────────────────

def visualize_per_class_iou(model, loader, config, save_dir, exp_tag=''):
    from metrics import SegMetrics
    model.eval()
    seg = SegMetrics(config.NUM_CLASSES)

    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(config.DEVICE), masks.to(config.DEVICE)
            preds = model(imgs)
            seg.update(preds, masks)

    results = seg.compute()
    iou_per = results['per_class_iou']

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(CLASS_NAMES, iou_per,
                  color=[c / 255.0 for c in CLASS_COLORS],
                  edgecolor='black', linewidth=0.8)

    for bar, val in zip(bars, iou_per):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=11)

    ax.set_ylim(0, 1.05)
    ax.set_ylabel('IoU', fontsize=12)
    ax.set_title(f'Per-class IoU — {exp_tag}\nmIoU={results["miou"]:.4f}  PixAcc={results["pixel_acc"]:.4f}',
                 fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(save_dir, f'{exp_tag}_per_class_iou.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [✓] Per-class IoU saved → {out_path}')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_id',      type=int,   required=True,
                        help='Experiment ID (1-27)')
    parser.add_argument('--model_name',  type=str,   required=True,
                        choices=['DeepLabV3+', 'Hybrid SSRN-ViT', 'MambaHSI'])
    parser.add_argument('--modality',    type=str,   required=True,
                        choices=['RGB', 'HSI', 'RGB+HSI'])
    parser.add_argument('--num_samples', type=int,   default=8,
                        help='Number of samples for GT vs Pred grid')
    parser.add_argument('--max_feat_ch', type=int,   default=16,
                        help='Max feature channels to visualise per layer')
    parser.add_argument('--split',       type=str,   default='val',
                        choices=['train', 'val'],
                        help='Which split to visualise')
    args = parser.parse_args()

    config   = Config()
    vis_dir  = os.path.join(config.SAVE_DIR, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)

    exp_tag = f'Exp_{args.exp_id:02d}_{args.model_name.replace(" ", "_")}_{args.modality.replace("+", "_")}'

    # ── Load checkpoint ───────────────────────────────────────────────────────
    ckpt_path = os.path.join(
        config.SAVE_DIR,
        f'Exp_{args.exp_id:02d}_{args.model_name.replace(" ", "_")}_{args.modality.replace("+", "_")}_best.pth'
    )
    if not os.path.exists(ckpt_path):
        # Try glob as fallback
        pattern   = os.path.join(config.SAVE_DIR, f'Exp_{args.exp_id:02d}_*.pth')
        candidates = glob.glob(pattern)
        if not candidates:
            raise FileNotFoundError(f'No checkpoint found for Exp {args.exp_id}. '
                                    f'Looked for: {ckpt_path}')
        ckpt_path = candidates[0]

    print(f'\n  Loading checkpoint: {ckpt_path}')

    # ── Build model ───────────────────────────────────────────────────────────
    in_ch = 0
    if 'RGB' in args.modality: in_ch += 3
    if 'HSI' in args.modality: in_ch += config.HSI_CHANNELS

    model = build_model(args.model_name, in_ch, config.NUM_CLASSES, config.IMAGE_SIZE)
    model.load_state_dict(torch.load(ckpt_path, map_location=config.DEVICE))
    model = model.to(config.DEVICE)
    model.eval()
    print(f'  Model loaded: {args.model_name}  |  in_ch={in_ch}')

    # ── Build dataset ─────────────────────────────────────────────────────────
    search_pattern = os.path.join(config.RGB_DIR, '*.jpg')
    all_files      = sorted(glob.glob(search_pattern))
    if not all_files:
        search_pattern = os.path.join(config.RGB_DIR, '*.png')
        all_files      = sorted(glob.glob(search_pattern))

    img_ids = [os.path.splitext(os.path.basename(f))[0] for f in all_files]
    split   = int(0.8 * len(img_ids))
    val_ids = img_ids[split:]
    use_ids = img_ids[:split] if args.split == 'train' else val_ids

    dataset = PCB_Dataset(use_ids, config, args.modality, 'None', mode=args.split)
    loader  = DataLoader(dataset, batch_size=4, shuffle=False,
                         num_workers=0, pin_memory=True)

    print(f'  Dataset split={args.split}  samples={len(dataset)}')

    # ── 1. GT vs Prediction ───────────────────────────────────────────────────
    print('\n  [1/3] Generating GT vs Prediction visualizations...')
    visualize_predictions(model, loader, config, vis_dir,
                          num_samples=args.num_samples, exp_tag=exp_tag)

    # ── 2. Feature maps (one sample) ─────────────────────────────────────────
    print('\n  [2/3] Extracting intermediate feature maps...')
    sample_img, _ = dataset[0]
    sample_img = sample_img.unsqueeze(0).to(config.DEVICE)
    visualize_feature_maps(model, sample_img, vis_dir,
                           exp_tag=exp_tag, max_channels=args.max_feat_ch)

    # ── 3. Per-class IoU bar chart ────────────────────────────────────────────
    print('\n  [3/3] Computing per-class IoU...')
    visualize_per_class_iou(model, loader, config, vis_dir, exp_tag=exp_tag)

    print(f'\n  All visualizations saved to: {vis_dir}')
    print('  Done!')


if __name__ == '__main__':
    main()