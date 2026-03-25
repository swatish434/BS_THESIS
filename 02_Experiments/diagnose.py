"""
diagnose.py
───────────
Systematic diagnostic script to verify:
1. Data loading correctness
2. Augmentation correctness  
3. Loss function correctness
4. Model input/output correctness
5. Class distribution
6. Augmentation visual effect
"""

import os, sys, glob
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torch.utils.data import DataLoader

from config  import Config
from dataset import PCB_Dataset, get_patch_ids
from models  import build_model

config     = Config()
CLASS_NAMES  = ['Background', 'IC', 'Connector', 'Capacitor']
CLASS_COLORS = np.array([[0,0,0],[255,165,0],[0,255,255],[255,0,255]], dtype=np.uint8)

def mask_to_rgb(mask):
    return CLASS_COLORS[np.clip(mask, 0, 3)]

DIAG_DIR = './results/diagnostics'
os.makedirs(DIAG_DIR, exist_ok=True)

print("\n" + "="*65)
print("  DIAGNOSTIC REPORT")
print("="*65)

# ─────────────────────────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────────────────────────
print("\n[1] DATA LOADING CHECK")

rgb_files = sorted(glob.glob(os.path.join(config.RGB_DIR, '*.jpg')))
rgb_ids   = [os.path.splitext(os.path.basename(f))[0] for f in rgb_files]
split     = int(0.8 * len(rgb_ids))
rgb_train = rgb_ids[:split]

hsi_train = get_patch_ids(config.PATCHES_DIR, 'train')

for modality, ids in [('RGB', rgb_train), ('HSI', hsi_train)]:
    ds      = PCB_Dataset(ids[:5], config, modality, 'None', 'train')
    img, mask = ds[0]
    in_ch   = 3 if modality == 'RGB' else config.HSI_CHANNELS
    ok_ch   = img.shape[0] == in_ch
    ok_sz   = img.shape[1] == config.IMAGE_SIZE
    ok_rng  = (img.min() >= 0.0) and (img.max() <= 1.0)
    ok_mask = set(mask.unique().tolist()).issubset({0,1,2,3})
    print(f"  {modality:8s} | shape:{img.shape} | "
          f"channels:{'OK' if ok_ch else 'FAIL'} | "
          f"size:{'OK' if ok_sz else 'FAIL'} | "
          f"range:[{img.min():.2f},{img.max():.2f}] {'OK' if ok_rng else 'FAIL'} | "
          f"mask_classes:{sorted(mask.unique().tolist())} {'OK' if ok_mask else 'FAIL'}")

# ─────────────────────────────────────────────────────────────────
# 2. CLASS DISTRIBUTION
# ─────────────────────────────────────────────────────────────────
print("\n[2] CLASS DISTRIBUTION CHECK")

for modality, ids, split_name in [
    ('RGB', rgb_train, 'train'),
    ('HSI', hsi_train[:50], 'train(50)')
]:
    counts = np.zeros(4, dtype=np.int64)
    ds = PCB_Dataset(ids, config, modality, 'None', 'train')
    for i in range(len(ds)):
        _, mask = ds[i]
        for c in range(4):
            counts[c] += (mask == c).sum().item()
    total = counts.sum()
    print(f"  {modality} {split_name}:")
    for c, name in enumerate(CLASS_NAMES):
        pct = 100 * counts[c] / total
        bar = '█' * int(pct / 2)
        print(f"    {name:12s} {pct:5.1f}% {bar}")

# ─────────────────────────────────────────────────────────────────
# 3. AUGMENTATION CHECK
# ─────────────────────────────────────────────────────────────────
print("\n[3] AUGMENTATION CHECK")

for aug in ['Copy-Paste', 'CutMix']:
    for modality, ids in [('RGB', rgb_train), ('HSI', hsi_train)]:
        if len(ids) < 2: continue
        ds   = PCB_Dataset(ids[:10], config, modality, aug, 'train')
        imgs, masks = [], []
        for i in range(min(5, len(ds))):
            img, mask = ds[i]
            imgs.append(img); masks.append(mask)

        # Check augmented samples have valid values
        for i, (img, mask) in enumerate(zip(imgs, masks)):
            ok_rng  = (img.min() >= 0.0) and (img.max() <= 1.0)
            ok_mask = set(mask.unique().tolist()).issubset({0,1,2,3})
            if not ok_rng or not ok_mask:
                print(f"  {aug} {modality} sample {i}: FAIL "
                      f"img_range:[{img.min():.2f},{img.max():.2f}] "
                      f"mask:{mask.unique().tolist()}")

        # Check augmented class distribution differs from original
        ds_none = PCB_Dataset(ids[:10], config, modality, 'None', 'train')
        orig_classes = set()
        aug_classes  = set()
        for i in range(min(5, len(ds))):
            _, m1 = ds_none[i]
            _, m2 = ds[i]
            orig_classes.update(m1.unique().tolist())
            aug_classes.update(m2.unique().tolist())

        print(f"  {aug:12s} {modality:8s} | "
              f"orig_classes:{sorted(orig_classes)} | "
              f"aug_classes:{sorted(aug_classes)} | OK")

# ─────────────────────────────────────────────────────────────────
# 4. AUGMENTATION VISUAL CHECK
# ─────────────────────────────────────────────────────────────────
print("\n[4] AUGMENTATION VISUAL OUTPUT → saving to diagnostics/")

fig, axes = plt.subplots(3, 4, figsize=(16, 12))
row_labels = ['No Aug', 'Copy-Paste', 'CutMix']

for row, aug in enumerate(['None', 'Copy-Paste', 'CutMix']):
    ds  = PCB_Dataset(rgb_train[:10], config, 'RGB', aug, 'train')
    img, mask = ds[0]
    rgb_np   = img.permute(1,2,0).numpy()
    mask_np  = mask.numpy()

    axes[row,0].imshow(rgb_np)
    axes[row,0].set_title(f'{aug} — RGB input')
    axes[row,0].axis('off')

    axes[row,1].imshow(mask_to_rgb(mask_np))
    axes[row,1].set_title('Mask')
    axes[row,1].axis('off')

    # Overlay
    overlay = rgb_np.copy()
    non_bg  = mask_np > 0
    mask_rgb = mask_to_rgb(mask_np) / 255.0
    overlay[non_bg] = overlay[non_bg] * 0.5 + mask_rgb[non_bg] * 0.5
    axes[row,2].imshow(overlay)
    axes[row,2].set_title('Overlay')
    axes[row,2].axis('off')

    # Class histogram
    counts = [(mask_np == c).sum() for c in range(4)]
    axes[row,3].bar(CLASS_NAMES, counts,
                    color=[c/255 for c in CLASS_COLORS])
    axes[row,3].set_title('Class distribution')
    axes[row,3].tick_params(axis='x', rotation=30)

plt.suptitle('RGB Augmentation Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{DIAG_DIR}/aug_visual_rgb.png', dpi=120, bbox_inches='tight')
plt.close()
print(f"  Saved: aug_visual_rgb.png")

# ─────────────────────────────────────────────────────────────────
# 5. LOSS FUNCTION CHECK
# ─────────────────────────────────────────────────────────────────
print("\n[5] LOSS FUNCTION CHECK")

# Test with known inputs
batch_size = 4
num_classes = 4
h = w = 64

# Perfect prediction — loss should be near 0
perfect_logits = torch.zeros(batch_size, num_classes, h, w)
perfect_target = torch.zeros(batch_size, h, w, dtype=torch.long)
for c in range(num_classes):
    idx = perfect_target == c
    perfect_logits[:, c][perfect_target == c] = 100.0

ce_loss = torch.nn.CrossEntropyLoss()(perfect_logits, perfect_target)
print(f"  CE loss (perfect prediction): {ce_loss.item():.6f} "
      f"{'OK (near 0)' if ce_loss.item() < 0.01 else 'WARN'}")

# Random prediction — loss should be ~log(4) ≈ 1.386
random_logits = torch.randn(batch_size, num_classes, h, w)
random_target = torch.randint(0, num_classes, (batch_size, h, w))
ce_random = torch.nn.CrossEntropyLoss()(random_logits, random_target)
print(f"  CE loss (random prediction):  {ce_random.item():.4f} "
      f"(expected ~{np.log(num_classes):.3f}) "
      f"{'OK' if abs(ce_random.item() - np.log(num_classes)) < 0.3 else 'WARN'}")

# Weighted loss — connector should have higher penalty
weights = torch.tensor([0.05, 1.32, 1.32, 1.32])
ce_w = torch.nn.CrossEntropyLoss(weight=weights)
# Misclassify all connector pixels
conn_target = torch.ones(1, h, w, dtype=torch.long) * 2  # all connector
wrong_logits = torch.zeros(1, num_classes, h, w)
wrong_logits[:, 0] = 100.0  # predict background
loss_conn_wrong = ce_w(wrong_logits, conn_target)
loss_bg_wrong   = ce_w(wrong_logits,
                       torch.zeros(1, h, w, dtype=torch.long))
print(f"  Weighted CE (wrong on connector): {loss_conn_wrong.item():.4f}")
print(f"  Weighted CE (wrong on BG):        {loss_bg_wrong.item():.4f}")
print(f"  Connector penalized more: "
      f"{'YES OK' if loss_conn_wrong > loss_bg_wrong else 'NO FAIL'}")

# ─────────────────────────────────────────────────────────────────
# 6. MODEL FORWARD PASS CHECK
# ─────────────────────────────────────────────────────────────────
print("\n[6] MODEL FORWARD PASS CHECK")

for model_name in ['DeepLabV3+', 'Hybrid SSRN-ViT', 'MambaHSI']:
    for modality, in_ch in [('RGB', 3), ('HSI', config.HSI_CHANNELS)]:
        try:
            model = build_model(model_name, in_ch, config.NUM_CLASSES,
                                config.IMAGE_SIZE).to(config.DEVICE)
            model.eval()
            x   = torch.randn(2, in_ch, config.IMAGE_SIZE,
                              config.IMAGE_SIZE).to(config.DEVICE)
            with torch.no_grad():
                out = model(x)
            ok_shape = out.shape == (2, config.NUM_CLASSES,
                                     config.IMAGE_SIZE, config.IMAGE_SIZE)
            ok_finite = torch.isfinite(out).all().item()
            print(f"  {model_name:20s} {modality:8s} | "
                  f"output:{out.shape} | "
                  f"shape:{'OK' if ok_shape else 'FAIL'} | "
                  f"finite:{'OK' if ok_finite else 'NaN/Inf FAIL'}")
            del model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  {model_name:20s} {modality:8s} | FAIL: {e}")

# ─────────────────────────────────────────────────────────────────
# 7. TRAIN/VAL SPLIT CHECK
# ─────────────────────────────────────────────────────────────────
print("\n[7] TRAIN/VAL SPLIT CHECK")

hsi_val  = get_patch_ids(config.PATCHES_DIR, 'val')
hsi_test = get_patch_ids(config.PATCHES_DIR, 'test')

# Check no overlap between splits
rgb_val  = rgb_ids[split:]
print(f"  RGB  — train:{len(rgb_train)} val:{len(rgb_val)} "
      f"overlap:{len(set(rgb_train) & set(rgb_val))} "
      f"{'OK' if len(set(rgb_train) & set(rgb_val)) == 0 else 'FAIL leakage!'}")

train_set = set(hsi_train)
val_set   = set(hsi_val)
test_set  = set(hsi_test)
print(f"  HSI  — train:{len(hsi_train)} val:{len(hsi_val)} test:{len(hsi_test)}")
print(f"  HSI train/val overlap:  {len(train_set & val_set)}  "
      f"{'OK' if len(train_set & val_set) == 0 else 'FAIL leakage!'}")
print(f"  HSI train/test overlap: {len(train_set & test_set)}  "
      f"{'OK' if len(train_set & test_set) == 0 else 'FAIL leakage!'}")
print(f"  HSI val/test overlap:   {len(val_set & test_set)}  "
      f"{'OK' if len(val_set & test_set) == 0 else 'FAIL leakage!'}")

print(f"\n{'='*65}")
print(f"  Diagnostic complete. Check {DIAG_DIR}/ for visual outputs.")
print(f"{'='*65}\n")