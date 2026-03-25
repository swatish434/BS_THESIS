"""
evaluate_all.py
───────────────
Runs after training completes. Loads each saved checkpoint and computes:
  - Per-class: IoU, F1, Precision, Recall
  - Mean: mIoU, mF1, mPrecision, mRecall
  - Overall: Pixel Accuracy

Saves results to results/evaluation_report.csv and prints a formatted table.

Usage:
    python evaluate_all.py              # evaluate all completed experiments
    python evaluate_all.py --exp_id 1  # evaluate single experiment
"""

import os, sys, glob, argparse, csv
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config  import Config
from dataset import PCB_Dataset, get_patch_ids
from models  import build_model

CLASS_NAMES = ['Background', 'IC', 'Connector', 'Capacitor']

# ── Metrics ───────────────────────────────────────────────────────────────────

class DetailedMetrics:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.conf = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    @torch.no_grad()
    def update(self, preds, targets):
        p = preds.argmax(dim=1).cpu().numpy().flatten()
        t = targets.cpu().numpy().flatten()
        valid = (t >= 0) & (t < self.num_classes)
        p, t  = p[valid], t[valid]
        idx   = self.num_classes * t + p
        self.conf += np.bincount(idx, minlength=self.num_classes**2)\
                       .reshape(self.num_classes, self.num_classes)

    def compute(self):
        cm   = self.conf
        results = {}

        per_iou       = []
        per_f1        = []
        per_precision = []
        per_recall    = []

        for c in range(self.num_classes):
            tp = cm[c, c]
            fp = cm[:, c].sum() - tp
            fn = cm[c, :].sum() - tp
            union = tp + fp + fn

            iou       = tp / (union + 1e-10)         if union > 0  else 0.0
            precision = tp / (tp + fp + 1e-10)       if (tp+fp) > 0 else 0.0
            recall    = tp / (tp + fn + 1e-10)       if (tp+fn) > 0 else 0.0
            f1        = 2 * precision * recall / (precision + recall + 1e-10)

            per_iou.append(float(iou))
            per_f1.append(float(f1))
            per_precision.append(float(precision))
            per_recall.append(float(recall))

        # Only average over classes present in GT
        present = cm.sum(axis=1) > 0
        results['per_iou']       = per_iou
        results['per_f1']        = per_f1
        results['per_precision'] = per_precision
        results['per_recall']    = per_recall
        results['miou']          = float(np.mean(np.array(per_iou)[present]))
        results['mf1']           = float(np.mean(np.array(per_f1)[present]))
        results['mprecision']    = float(np.mean(np.array(per_precision)[present]))
        results['mrecall']       = float(np.mean(np.array(per_recall)[present]))
        results['pixel_acc']     = float(cm.diagonal().sum() / (cm.sum() + 1e-10))
        return results


# ── Inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_checkpoint(model, loader, config):
    model.eval()
    metrics = DetailedMetrics(config.NUM_CLASSES)
    for img, mask in tqdm(loader, desc="  Evaluating", leave=False):
        img, mask = img.to(config.DEVICE), mask.to(config.DEVICE)
        out = model(img)
        metrics.update(out, mask)
    return metrics.compute()


# ── Experiment matrix ─────────────────────────────────────────────────────────

def build_matrix():
    MODALITIES    = ['RGB', 'HSI', 'RGB+HSI']
    MODELS        = ['DeepLabV3+', 'Hybrid SSRN-ViT', 'MambaHSI']
    AUGMENTATIONS = ['None', 'Copy-Paste', 'CutMix']
    matrix = []
    exp_id = 1
    for mod in MODALITIES:
        for mdl in MODELS:
            for aug in AUGMENTATIONS:
                matrix.append(dict(id=exp_id, mod=mod, model=mdl, aug=aug))
                exp_id += 1
    return matrix


# ── CSV header ────────────────────────────────────────────────────────────────

def get_csv_header():
    header = ['exp_id', 'modality', 'model', 'augmentation',
              'pixel_acc', 'mIoU', 'mF1', 'mPrecision', 'mRecall']
    for cls in CLASS_NAMES:
        for metric in ['IoU', 'F1', 'Precision', 'Recall']:
            header.append(f"{cls}_{metric}")
    return header


def result_to_row(exp, results):
    row = {
        'exp_id':       exp['id'],
        'modality':     exp['mod'],
        'model':        exp['model'],
        'augmentation': exp['aug'],
        'pixel_acc':    round(results['pixel_acc'], 4),
        'mIoU':         round(results['miou'],      4),
        'mF1':          round(results['mf1'],       4),
        'mPrecision':   round(results['mprecision'], 4),
        'mRecall':      round(results['mrecall'],   4),
    }
    for i, cls in enumerate(CLASS_NAMES):
        row[f"{cls}_IoU"]       = round(results['per_iou'][i],       4)
        row[f"{cls}_F1"]        = round(results['per_f1'][i],        4)
        row[f"{cls}_Precision"] = round(results['per_precision'][i], 4)
        row[f"{cls}_Recall"]    = round(results['per_recall'][i],    4)
    return row


# ── Pretty print table ────────────────────────────────────────────────────────

def print_table(all_rows):
    print(f"\n{'='*120}")
    print(f"{'EXP':>4} {'Modality':>8} {'Model':>20} {'Aug':>12} "
          f"{'PixAcc':>7} {'mIoU':>6} {'mF1':>6} "
          f"{'BG_IoU':>7} {'IC_IoU':>7} {'Conn_IoU':>9} {'Cap_IoU':>8} "
          f"{'IC_F1':>6} {'Conn_F1':>8} {'Cap_F1':>7}")
    print(f"{'-'*120}")
    for r in all_rows:
        if 'mIoU' not in r: continue
        print(f"  {r['exp_id']:>2} {r['modality']:>8} {r['model']:>20} {r['augmentation']:>12} "
              f"  {r['pixel_acc']:.3f}  {r['mIoU']:.3f}  {r['mF1']:.3f} "
              f"   {r['Background_IoU']:.3f}   {r['IC_IoU']:.3f}     {r['Connector_IoU']:.3f}      {r['Capacitor_IoU']:.3f} "
              f"  {r['IC_F1']:.3f}    {r['Connector_F1']:.3f}    {r['Capacitor_F1']:.3f}")
    print(f"{'='*120}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_id', type=int, default=None,
                        help='Evaluate single experiment (default: all)')
    args   = parser.parse_args()

    config  = Config()
    matrix  = build_matrix()
    if args.exp_id:
        matrix = [e for e in matrix if e['id'] == args.exp_id]

    # Data loaders
    rgb_files = sorted(glob.glob(os.path.join(config.RGB_DIR, '*.jpg')))
    if not rgb_files:
        rgb_files = sorted(glob.glob(os.path.join(config.RGB_DIR, '*.png')))
    rgb_ids   = [os.path.splitext(os.path.basename(f))[0] for f in rgb_files]
    split     = int(0.8 * len(rgb_ids))
    rgb_val   = rgb_ids[split:]
    hsi_val   = get_patch_ids(config.PATCHES_DIR, 'val')

    # CSV setup
    report_path = os.path.join(config.SAVE_DIR, 'evaluation_report.csv')
    header      = get_csv_header()
    write_header = not os.path.exists(report_path)
    all_rows    = []

    print(f"\n{'='*65}")
    print(f"  Evaluation Report — {len(matrix)} experiments")
    print(f"{'='*65}")

    for exp in matrix:
        # Find checkpoint
        ckpt = os.path.join(
            config.SAVE_DIR,
            f"Exp_{exp['id']:02d}_{exp['model'].replace(' ','_')}"
            f"_{exp['mod'].replace('+','_')}_best.pth"
        )
        if not os.path.exists(ckpt):
            print(f"  EXP {exp['id']:02d} — checkpoint not found, skipping")
            continue

        print(f"\n  EXP {exp['id']:02d} | {exp['mod']:8s} | {exp['model']:20s} | {exp['aug']}")

        # Build model
        in_ch = 0
        if 'RGB' in exp['mod']: in_ch += 3
        if 'HSI' in exp['mod']: in_ch += config.HSI_CHANNELS

        model = build_model(exp['model'], in_ch, config.NUM_CLASSES,
                            config.IMAGE_SIZE)
        model.load_state_dict(torch.load(ckpt, map_location=config.DEVICE))
        model = model.to(config.DEVICE)

        # Build val loader
        val_ids = rgb_val if exp['mod'] == 'RGB' else hsi_val
        val_ds  = PCB_Dataset(val_ids, config, exp['mod'], 'None', mode='val')
        val_loader = DataLoader(val_ds, batch_size=4, shuffle=False,
                                num_workers=0, pin_memory=True)

        # Evaluate
        results = evaluate_checkpoint(model, val_loader, config)
        row     = result_to_row(exp, results)
        all_rows.append(row)

        # Print per-class summary
        print(f"    PixAcc={results['pixel_acc']:.4f}  "
              f"mIoU={results['miou']:.4f}  mF1={results['mf1']:.4f}")
        for i, cls in enumerate(CLASS_NAMES):
            print(f"    {cls:12s} → "
                  f"IoU={results['per_iou'][i]:.4f}  "
                  f"F1={results['per_f1'][i]:.4f}  "
                  f"Prec={results['per_precision'][i]:.4f}  "
                  f"Rec={results['per_recall'][i]:.4f}")

        # Save to CSV
        with open(report_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            if write_header:
                writer.writeheader()
                write_header = False
            writer.writerow(row)

        del model
        torch.cuda.empty_cache()

    # Print summary table
    print_table(all_rows)
    print(f"  Full report saved → {report_path}")


if __name__ == '__main__':
    main()