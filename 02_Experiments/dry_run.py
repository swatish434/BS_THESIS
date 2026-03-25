"""
full_dry_run.py
───────────────
Complete end-to-end dry run for all 27 experiments.
Tests every stage:
  1. Data loading (input)
  2. Augmentation (visual check)
  3. Model forward pass
  4. Loss computation
  5. Backward pass (gradient check)
  6. GT vs Prediction visualization (1 sample per exp)

Usage:
    python full_dry_run.py
"""

import os, sys, glob, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from config  import Config
from dataset import PCB_Dataset, get_patch_ids
from models  import build_model
from augmentations import apply_augmentation

# ── Setup ─────────────────────────────────────────────────────────────────────
config      = Config()
DIAG_DIR    = './results/full_dry_run'
os.makedirs(DIAG_DIR, exist_ok=True)

CLASS_NAMES  = ['Background', 'IC', 'Connector', 'Capacitor']
CLASS_COLORS = np.array([[0,0,0],[255,165,0],[0,255,255],[255,0,255]], dtype=np.uint8)

MODALITIES    = ['RGB', 'HSI', 'RGB+HSI']
MODELS        = ['DeepLabV3+', 'Hybrid SSRN-ViT', 'MambaHSI']
AUGMENTATIONS = ['None', 'Copy-Paste', 'CutMix']

passed = []
failed = []

def mask_to_rgb(mask):
    return CLASS_COLORS[np.clip(mask, 0, 3)]

def log(msg, level='INFO'):
    prefix = {'INFO': '  ', 'OK': '  ✓', 'FAIL': '  ✗', 'WARN': '  !'}
    print(f"{prefix.get(level,'  ')} {msg}")

# ── Build ID lists ─────────────────────────────────────────────────────────────
rgb_files  = sorted(glob.glob(os.path.join(config.RGB_DIR, '*.jpg')))
if not rgb_files:
    rgb_files = sorted(glob.glob(os.path.join(config.RGB_DIR, '*.png')))
rgb_ids    = [os.path.splitext(os.path.basename(f))[0] for f in rgb_files]
split      = int(0.8 * len(rgb_ids))
rgb_train  = rgb_ids[:split]
rgb_val    = rgb_ids[split:]

hsi_train  = get_patch_ids(config.PATCHES_DIR, 'train')
hsi_val    = get_patch_ids(config.PATCHES_DIR, 'val')

print(f"\n{'='*65}")
print(f"  FULL DRY RUN — all 27 experiments")
print(f"  RGB  train:{len(rgb_train)} val:{len(rgb_val)}")
print(f"  HSI  train:{len(hsi_train)} val:{len(hsi_val)}")
print(f"{'='*65}")

exp_id = 1

for modality in MODALITIES:
    for model_name in MODELS:
        for aug in AUGMENTATIONS:

            tag = f"EXP {exp_id:02d} | {modality:8s} | {model_name:20s} | {aug}"
            print(f"\n  {tag}")
            errors = []

            try:
                # ── Select IDs ───────────────────────────────────────────────
                train_ids = rgb_train if modality == 'RGB' else hsi_train
                val_ids   = rgb_val   if modality == 'RGB' else hsi_val

                if len(train_ids) < 2:
                    raise ValueError(f"Not enough training samples: {len(train_ids)}")

                in_ch = 0
                if 'RGB' in modality: in_ch += 3
                if 'HSI' in modality: in_ch += config.HSI_CHANNELS

                # ── STAGE 1: Data loading ─────────────────────────────────────
                ds    = PCB_Dataset(train_ids[:4], config, modality, 'None', 'train')
                img, mask = ds[0]

                assert img.ndim  == 3, f"img should be 3D got {img.shape}"
                assert mask.ndim == 2, f"mask should be 2D got {mask.shape}"
                assert img.shape[0]  == in_ch, f"Expected {in_ch}ch got {img.shape[0]}"
                assert img.shape[1]  == config.IMAGE_SIZE
                assert img.shape[2]  == config.IMAGE_SIZE
                assert img.min()     >= 0.0, f"img min={img.min()}"
                assert img.max()     <= 1.0, f"img max={img.max()}"
                assert set(mask.unique().tolist()).issubset({0,1,2,3})
                log("Stage 1 — data loading OK", 'OK')

                # ── STAGE 2: Augmentation ─────────────────────────────────────
                if aug != 'None':
                    img_np   = img.permute(1,2,0).numpy()
                    mask_np  = mask.numpy().astype(np.int64)
                    img2, mask2 = ds[1]
                    img2_np  = img2.permute(1,2,0).numpy()
                    mask2_np = mask2.numpy().astype(np.int64)

                    aug_img, aug_mask = apply_augmentation(
                        img_np, mask_np, img2_np, mask2_np, aug)

                    assert aug_img.shape  == img_np.shape,  f"aug img shape mismatch"
                    assert aug_mask.shape == mask_np.shape, f"aug mask shape mismatch"
                    assert aug_img.min()  >= 0.0, f"aug img min={aug_img.min()}"
                    assert aug_img.max()  <= 1.0, f"aug img max={aug_img.max()}"
                    assert set(np.unique(aug_mask).tolist()).issubset({0,1,2,3})
                    log(f"Stage 2 — {aug} augmentation OK", 'OK')
                else:
                    log("Stage 2 — augmentation skipped (None)", 'OK')

                # ── STAGE 3: Model forward pass ───────────────────────────────
                model = build_model(model_name, in_ch, config.NUM_CLASSES,
                                    config.IMAGE_SIZE).to(config.DEVICE)
                model.train()

                batch_img  = torch.stack([img, ds[1][0]]).to(config.DEVICE)
                batch_mask = torch.stack([mask, ds[1][1]]).to(config.DEVICE)

                output = model(batch_img)
                assert output.shape == (2, config.NUM_CLASSES,
                                        config.IMAGE_SIZE, config.IMAGE_SIZE)
                assert torch.isfinite(output).all(), "NaN/Inf in output"
                log(f"Stage 3 — forward pass OK {output.shape}", 'OK')

                # ── STAGE 4: Loss computation ─────────────────────────────────
                criterion = nn.CrossEntropyLoss()
                loss      = criterion(output, batch_mask)
                assert torch.isfinite(loss), f"Loss is NaN/Inf: {loss}"
                assert loss.item() > 0, "Loss is exactly 0 — suspicious"
                log(f"Stage 4 — loss OK ({loss.item():.4f})", 'OK')

                # ── STAGE 5: Backward pass ────────────────────────────────────
                loss.backward()
                # Check at least some gradients are non-zero
                has_grad = any(
                    p.grad is not None and p.grad.abs().sum() > 0
                    for p in model.parameters()
                )
                assert has_grad, "No gradients computed — model not training"
                log("Stage 5 — backward pass OK (gradients exist)", 'OK')

                # ── STAGE 6: GT vs Prediction visualization ───────────────────
                model.eval()
                with torch.no_grad():
                    pred = model(batch_img[0:1]).argmax(dim=1)[0].cpu().numpy()

                # Get RGB for display (first 3 channels)
                img_np    = batch_img[0].cpu().permute(1,2,0).numpy()
                if img_np.shape[2] > 3:
                    img_np = img_np[:,:,:3]
                img_np    = np.clip(img_np, 0, 1)
                gt_np     = batch_mask[0].cpu().numpy()

                fig, axes = plt.subplots(1, 4, figsize=(16, 4))
                axes[0].imshow(img_np)
                axes[0].set_title('Input', fontsize=11)
                axes[0].axis('off')

                axes[1].imshow(mask_to_rgb(gt_np))
                axes[1].set_title('Ground Truth', fontsize=11)
                axes[1].axis('off')

                axes[2].imshow(mask_to_rgb(pred))
                axes[2].set_title('Prediction (untrained)', fontsize=11)
                axes[2].axis('off')

                # Overlay
                overlay = img_np.copy()
                non_bg  = pred > 0
                pred_rgb = mask_to_rgb(pred) / 255.0
                overlay[non_bg] = overlay[non_bg]*0.5 + pred_rgb[non_bg]*0.5
                axes[3].imshow(overlay)
                axes[3].set_title('Overlay', fontsize=11)
                axes[3].axis('off')

                # Legend
                patches = [mpatches.Patch(color=CLASS_COLORS[i]/255.0,
                           label=CLASS_NAMES[i]) for i in range(4)]
                fig.legend(handles=patches, loc='lower center', ncol=4,
                           fontsize=9, bbox_to_anchor=(0.5, -0.05))

                safe_name = f"Exp_{exp_id:02d}_{model_name.replace(' ','_')}" \
                            f"_{modality.replace('+','_')}_{aug.replace('-','')}"
                plt.suptitle(f"{tag}", fontsize=11, fontweight='bold')
                plt.tight_layout()
                out_path = os.path.join(DIAG_DIR, f"{safe_name}_dryrun.png")
                fig.savefig(out_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                log(f"Stage 6 — visualization saved", 'OK')

                del model
                torch.cuda.empty_cache()

                log(f"ALL STAGES PASSED", 'OK')
                passed.append(exp_id)

            except Exception as e:
                log(f"FAILED: {e}", 'FAIL')
                failed.append((exp_id, str(e)))
                try:
                    del model
                    torch.cuda.empty_cache()
                except: pass

            exp_id += 1

# ── Summary ────────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  DRY RUN COMPLETE")
print(f"  PASSED: {len(passed)}/27")
print(f"  FAILED: {len(failed)}/27")

if failed:
    print(f"\n  Failed experiments:")
    for eid, err in failed:
        print(f"    EXP {eid:02d}: {err}")
    print(f"\n  Fix these before running full experiments!")
    sys.exit(1)
else:
    print(f"\n  All 27 experiments passed all 6 stages!")
    print(f"  Visualizations saved to: {DIAG_DIR}/")
    print(f"  Safe to run: python run_matrix.py")

print(f"{'='*65}\n")