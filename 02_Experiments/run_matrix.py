"""
run_matrix.py  v3
─────────────────
27 experiments with correct data routing:
  RGB modality     → 53 raw images + Monoseg masks (80/20 split)
  HSI modality     → 135 HSI patches + npy masks
  RGB+HSI modality → 135 HSI patches + npy masks (RGB from HSI[:3])
"""
import csv, gc, os, time, glob, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

from config  import Config, ensure_dir
from dataset import PCB_Dataset, get_patch_ids
from metrics import SegMetrics
from models  import build_model


# ─────────────────────────────────────────────────────────────────────────────
# Combined Loss: Weighted CE + Dice
# ─────────────────────────────────────────────────────────────────────────────

class DiceLoss(nn.Module):
    def __init__(self, num_classes, smooth=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.smooth      = smooth

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
    def __init__(self, class_weights, num_classes):
        super().__init__()
        self.ce   = nn.CrossEntropyLoss(weight=class_weights)
        self.dice = DiceLoss(num_classes)

    def forward(self, preds, targets):
        return 0.5 * self.ce(preds, targets) + 0.5 * self.dice(preds, targets)


def compute_class_weights(mask_paths, num_classes):
    """Compute inverse-frequency class weights from mask files."""
    counts = np.zeros(num_classes, dtype=np.float64)
    for p in mask_paths[:50]:
        try:
            import cv2
            if p.endswith('.npy'):
                mask = np.load(p)
            else:
                mask = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if mask is None: continue
            for c in range(num_classes):
                counts[c] += (mask == c).sum()
        except Exception:
            continue

    if counts.sum() == 0:
        weights = np.array([0.5, 2.0, 5.0, 3.0])
    else:
        total   = counts.sum()
        weights = total / (num_classes * counts + 1e-8)
        weights = np.clip(weights, 0.3, 8.0)
        weights = weights / weights.sum() * num_classes

    print(f"  Weights → BG:{weights[0]:.2f} IC:{weights[1]:.2f} "
          f"Connector:{weights[2]:.2f} Capacitor:{weights[3]:.2f}")
    return torch.tensor(weights, dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Data routing
# ─────────────────────────────────────────────────────────────────────────────

def get_dataloaders(config, modality, augmentation):
    if modality == 'RGB':
        # Use 53 raw images with Monoseg masks
        all_files = sorted(glob.glob(os.path.join(config.RGB_DIR, '*.jpg')))
        if not all_files:
            all_files = sorted(glob.glob(os.path.join(config.RGB_DIR, '*.png')))
        if not all_files:
            raise FileNotFoundError(f"No RGB images in {config.RGB_DIR}")
        img_ids   = [os.path.splitext(os.path.basename(f))[0] for f in all_files]
        split     = int(0.8 * len(img_ids))
        train_ids = img_ids[:split]
        val_ids   = img_ids[split:]
        mask_paths = [os.path.join(config.MASK_DIR, f"{i}.png") for i in train_ids]
        print(f"  RGB mode — train:{len(train_ids)}  val:{len(val_ids)}")
    else:
        # Use HSI patches for HSI and RGB+HSI
        train_ids  = get_patch_ids(config.PATCHES_DIR, 'train')
        val_ids    = get_patch_ids(config.PATCHES_DIR, 'val')
        mask_paths = [os.path.join(config.PATCHES_DIR, f"{i}.npy") for i in train_ids]
        print(f"  Patch mode — train:{len(train_ids)}  val:{len(val_ids)}")

    train_ds = PCB_Dataset(train_ids, config, modality, augmentation, mode='train')
    val_ds   = PCB_Dataset(val_ids,   config, modality, 'None',       mode='val')

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE,
                              shuffle=True,  num_workers=config.NUM_WORKERS,
                              pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=config.BATCH_SIZE,
                              shuffle=False, num_workers=config.NUM_WORKERS,
                              pin_memory=True)
    return train_loader, val_loader, mask_paths


# ─────────────────────────────────────────────────────────────────────────────
# Train / validate
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, scaler, config):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    for step, (img, mask) in enumerate(tqdm(loader, desc="  Train", leave=False)):
        img, mask = img.to(config.DEVICE), mask.to(config.DEVICE)
        with autocast('cuda', enabled=config.USE_AMP):
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
        with autocast('cuda', enabled=config.USE_AMP):
            output = model(img)
            loss   = criterion(output, mask)
        total_loss += loss.item()
        seg_metrics.update(output, mask)
    results             = seg_metrics.compute()
    results['val_loss'] = total_loss / max(len(loader), 1)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# CSV logging
# ─────────────────────────────────────────────────────────────────────────────

CSV_HEADER = ['exp_id','modality','model','augmentation',
              'best_val_miou','best_val_pixel_acc','best_epoch',
              'total_epochs_run','duration_min','status']

def init_csv(path):
    if not os.path.exists(path):
        with open(path, 'w', newline='') as f:
            csv.writer(f).writerow(CSV_HEADER)

def log_result(path, row):
    with open(path, 'a', newline='') as f:
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

    train_loader, val_loader, mask_paths = get_dataloaders(
        config, modality, aug_strategy)

    print("  Computing class weights...")
    class_weights = compute_class_weights(mask_paths, config.NUM_CLASSES)
    class_weights = class_weights.to(config.DEVICE)
    criterion     = CombinedLoss(class_weights, config.NUM_CLASSES)
    optimizer     = optim.AdamW(model.parameters(),
                                lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    scaler        = GradScaler('cuda', enabled=config.USE_AMP)
    scheduler     = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    best_miou = -1.0; best_pixel_acc = 0.0; best_epoch = 0
    save_path = os.path.join(
        config.SAVE_DIR,
        f"Exp_{exp_id:02d}_{model_name.replace(' ','_')}"
        f"_{modality.replace('+','_')}_best.pth")

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
            best_miou = val_miou; best_pixel_acc = val_pixel_acc
            best_epoch = epoch
            torch.save(model.state_dict(), save_path)

    del model
    torch.cuda.empty_cache()
    gc.collect()

    duration_min = (time.time() - t0) / 60.0
    print(f"  Finished → best val mIoU={best_miou:.4f} at epoch {best_epoch}")

    log_result(config.LOG_CSV, {
        'exp_id': exp_id, 'modality': modality,
        'model': model_name, 'augmentation': aug_strategy,
        'best_val_miou': round(best_miou, 4),
        'best_val_pixel_acc': round(best_pixel_acc, 4),
        'best_epoch': best_epoch, 'total_epochs_run': epoch,
        'duration_min': round(duration_min, 1), 'status': 'OK',
    })


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    config = Config()
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
                'exp_id': exp['id'], 'modality': exp['mod'],
                'model': exp['model'], 'augmentation': exp['aug'],
                'best_val_miou': 'N/A', 'best_val_pixel_acc': 'N/A',
                'best_epoch': 'N/A', 'total_epochs_run': 'N/A',
                'duration_min': 'N/A', 'status': f'FAILED: {e}',
            })
            torch.cuda.empty_cache(); gc.collect()

    total_hours = (time.time() - total_start) / 3600
    print(f"\n  All experiments complete — wall time: {total_hours:.2f} h")
    print(f"  Results in: {config.LOG_CSV}")