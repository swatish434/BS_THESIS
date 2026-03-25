"""
run_matrix_v2.py
────────────────
Fixed version with:
1. Weighted CrossEntropyLoss to handle class imbalance
2. Better augmentation strategy for small datasets
3. Auto class weight computation from training data
4. Dice loss combined with CE for better minority class learning
"""

import csv
import gc
import os
import time
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config, ensure_dir
from dataset import PCB_Dataset
from metrics import SegMetrics
from models import build_model


# ─────────────────────────────────────────────────────────────────────────────
# Combined Loss: Weighted CE + Dice
# ─────────────────────────────────────────────────────────────────────────────

class DiceLoss(nn.Module):
    def __init__(self, num_classes, smooth=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = F.softmax(preds, dim=1)
        dice  = 0.0
        for c in range(self.num_classes):
            pred_c   = preds[:, c]
            target_c = (targets == c).float()
            inter    = (pred_c * target_c).sum()
            union    = pred_c.sum() + target_c.sum()
            dice    += (2.0 * inter + self.smooth) / (union + self.smooth)
        return 1.0 - dice / self.num_classes


class CombinedLoss(nn.Module):
    def __init__(self, class_weights, num_classes, ce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.ce   = nn.CrossEntropyLoss(weight=class_weights)
        self.dice = DiceLoss(num_classes)
        self.ce_w   = ce_weight
        self.dice_w = dice_weight

    def forward(self, preds, targets):
        return self.ce_w * self.ce(preds, targets) + \
               self.dice_w * self.dice(preds, targets)


def compute_class_weights(config, img_ids, modality):
    """Compute inverse frequency class weights from training masks."""
    print("  Computing class weights from training data...")
    counts = np.zeros(config.NUM_CLASSES, dtype=np.float64)

    for img_id in img_ids[:20]:  # sample first 20 for speed
        try:
            mask_path = os.path.join(config.MASK_DIR, f"{img_id}.png")
            if not os.path.exists(mask_path):
                continue
            import cv2
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            for c in range(config.NUM_CLASSES):
                counts[c] += (mask == c).sum()
        except Exception:
            continue

    if counts.sum() == 0:
        print("  [!] Could not compute weights, using defaults")
        return torch.tensor([0.1, 2.0, 5.0, 3.0])

    total   = counts.sum()
    weights = total / (config.NUM_CLASSES * counts + 1e-8)
    weights = weights / weights.sum() * config.NUM_CLASSES  # normalise
    weights[0] *= 0.3   # further suppress background

    print(f"  Class weights: BG={weights[0]:.2f} IC={weights[1]:.2f} "
          f"Connector={weights[2]:.2f} Capacitor={weights[3]:.2f}")
    return torch.tensor(weights, dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

def get_dataloaders(config, modality, augmentation):
    search_pattern = os.path.join(config.RGB_DIR, "*.jpg")
    all_files = sorted(glob.glob(search_pattern))
    if not all_files:
        search_pattern = os.path.join(config.RGB_DIR, "*.png")
        all_files = sorted(glob.glob(search_pattern))
    if not all_files:
        raise FileNotFoundError(f"No image files found in {config.RGB_DIR}")

    img_ids = [os.path.splitext(os.path.basename(f))[0] for f in all_files]
    split   = int(0.8 * len(img_ids))
    train_ids, val_ids = img_ids[:split], img_ids[split:]

    train_ds = PCB_Dataset(train_ids, config, modality, augmentation, mode='train')
    val_ds   = PCB_Dataset(val_ids,   config, modality, 'None',       mode='val')

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE,
                              shuffle=True,  num_workers=config.NUM_WORKERS,
                              pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=config.BATCH_SIZE,
                              shuffle=False, num_workers=config.NUM_WORKERS,
                              pin_memory=True)
    return train_loader, val_loader, train_ids


# ─────────────────────────────────────────────────────────────────────────────
# Training & validation
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, scaler, config):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for step, (img, mask) in enumerate(tqdm(loader, desc="  Train", leave=False)):
        img, mask = img.to(config.DEVICE), mask.to(config.DEVICE)
        with autocast(enabled=config.USE_AMP):
            output = model(img)
            loss   = criterion(output, mask) / config.GRAD_ACCUM
        scaler.scale(loss).backward()
        if (step + 1) % config.GRAD_ACCUM == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        total_loss += loss.item() * config.GRAD_ACCUM
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def validate(model, loader, criterion, config):
    model.eval()
    seg_metrics = SegMetrics(config.NUM_CLASSES)
    total_loss  = 0.0
    for img, mask in tqdm(loader, desc="  Val", leave=False):
        img, mask = img.to(config.DEVICE), mask.to(config.DEVICE)
        with autocast(enabled=config.USE_AMP):
            output = model(img)
            loss   = criterion(output, mask)
        total_loss += loss.item()
        seg_metrics.update(output, mask)
    results = seg_metrics.compute()
    results['val_loss'] = total_loss / max(len(loader), 1)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# CSV logging
# ─────────────────────────────────────────────────────────────────────────────

CSV_HEADER = [
    'exp_id', 'modality', 'model', 'augmentation',
    'best_val_miou', 'best_val_pixel_acc', 'best_epoch',
    'total_epochs_run', 'duration_min', 'status'
]

def init_csv(log_path):
    if not os.path.exists(log_path):
        with open(log_path, 'w', newline='') as f:
            csv.writer(f).writerow(CSV_HEADER)

def log_result(log_path, row):
    with open(log_path, 'a', newline='') as f:
        csv.DictWriter(f, fieldnames=CSV_HEADER).writerow(row)


# ─────────────────────────────────────────────────────────────────────────────
# Single experiment
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(exp_id, modality, model_name, aug_strategy, config):
    t0 = time.time()
    print(f"\n{'='*65}")
    print(f"  EXP {exp_id:02d} | {modality:8s} | {model_name:20s} | {aug_strategy}")
    print(f"{'='*65}")

    in_ch = 0
    if 'RGB' in modality: in_ch += 3
    if 'HSI' in modality: in_ch += config.HSI_CHANNELS

    model = build_model(model_name, in_ch, config.NUM_CLASSES,
                        config.IMAGE_SIZE).to(config.DEVICE)

    train_loader, val_loader, train_ids = get_dataloaders(
        config, modality, aug_strategy)

    # ── Weighted + Dice loss ──────────────────────────────────────────────────
    class_weights = compute_class_weights(config, train_ids, modality)
    class_weights = class_weights.to(config.DEVICE)
    criterion = CombinedLoss(class_weights, config.NUM_CLASSES,
                             ce_weight=0.5, dice_weight=0.5)

    optimizer = optim.AdamW(model.parameters(),
                            lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    scaler    = GradScaler(enabled=config.USE_AMP)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    best_miou      = -1.0
    best_pixel_acc = 0.0
    best_epoch     = 0
    no_improve     = 0
    save_path = os.path.join(
        config.SAVE_DIR,
        f"Exp_{exp_id:02d}_{model_name.replace(' ','_')}"
        f"_{modality.replace('+','_')}_v2_best.pth"
    )

    for epoch in range(1, config.EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, criterion,
                                 optimizer, scaler, config)
        val_stats  = validate(model, val_loader, criterion, config)

        val_miou      = val_stats['miou']
        val_pixel_acc = val_stats['pixel_acc']
        val_loss      = val_stats['val_loss']

        scheduler.step(val_miou)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{config.EPOCHS} | "
                  f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                  f"val_mIoU={val_miou:.4f}  pixel_acc={val_pixel_acc:.4f}")

        if val_miou > best_miou:
            best_miou      = val_miou
            best_pixel_acc = val_pixel_acc
            best_epoch     = epoch
            no_improve     = 0
            torch.save(model.state_dict(), save_path)
        else:
            no_improve += 1

        if no_improve >= config.PATIENCE:
            print(f"  Early stop at epoch {epoch}")
            break

    total_epochs = epoch
    del model
    torch.cuda.empty_cache()
    gc.collect()

    duration_min = (time.time() - t0) / 60.0
    print(f"  Finished → best val mIoU={best_miou:.4f} at epoch {best_epoch}")

    log_result(config.LOG_CSV, {
        'exp_id':             exp_id,
        'modality':           modality,
        'model':              model_name,
        'augmentation':       aug_strategy,
        'best_val_miou':      round(best_miou,      4),
        'best_val_pixel_acc': round(best_pixel_acc, 4),
        'best_epoch':         best_epoch,
        'total_epochs_run':   total_epochs,
        'duration_min':       round(duration_min, 1),
        'status':             'OK',
    })


# ─────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    config = Config()
    config.LOG_CSV = './results/experiment_results_v2.csv'
    ensure_dir(config.SAVE_DIR)
    init_csv(config.LOG_CSV)

    matrix = []
    exp_id = 1
    for mod in Config.MODALITIES:
        for mdl in Config.MODELS:
            for aug in Config.AUGMENTATIONS:
                matrix.append(dict(id=exp_id, mod=mod, model=mdl, aug=aug))
                exp_id += 1

    total_start = time.time()

    for exp in matrix:
        try:
            run_experiment(exp['id'], exp['mod'], exp['model'],
                           exp['aug'], config)
        except Exception as e:
            print(f"\n!!! Experiment {exp['id']} FAILED: {e}")
            log_result(config.LOG_CSV, {
                'exp_id':             exp['id'],
                'modality':           exp['mod'],
                'model':              exp['model'],
                'augmentation':       exp['aug'],
                'best_val_miou':      'N/A',
                'best_val_pixel_acc': 'N/A',
                'best_epoch':         'N/A',
                'total_epochs_run':   'N/A',
                'duration_min':       'N/A',
                'status':             f'FAILED: {e}',
            })
            torch.cuda.empty_cache()
            gc.collect()

    total_hours = (time.time() - total_start) / 3600
    print(f"\n{'='*65}")
    print(f"  All experiments complete — wall time: {total_hours:.2f} h")
    print(f"  Results in: {config.LOG_CSV}")
    print(f"{'='*65}")