#!/usr/bin/env python3
"""
Quick evaluation script for HSI ResUNet model on Patches (Val Split)
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from models.ResUnet import ResUnet
from models.Unet_Attention import AttU_Net
from train_hsi_overlap import HSIPatchesDataset, evaluate
from utils.loss_functions import HybridLoss
from utils.dataset_functions import evaluate_segmentation
import os
import json

# Configuration
CHECKPOINT = "Results/hsi_overlap_resunet_best.pth"
DATA_DIR = "/home/bs_thesis/Documents/BS_THESIS/PCBVision/Patches_256_Overlap_Data/"
INPUT_CHANNELS = 214
NUM_CLASSES = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4

print("="*60)
print("HSI ResUNet Evaluation (CutMix Baseline)")
print("="*60)
print(f"Checkpoint: {CHECKPOINT}")
print(f"Device: {DEVICE}")

# Load model
print("Loading model...")
model = ResUnet(channel=INPUT_CHANNELS, out_channel=NUM_CLASSES)

if os.path.exists(CHECKPOINT):
    state_dict = torch.load(CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model = model.to(DEVICE)
    model.eval()
    print("Model loaded successfully")
else:
    print(f"Error: Checkpoint not found at {CHECKPOINT}")
    # exit(1) # Don't exit, maybe we just want to test the script logic? No, we need metrics.
    print("Cannot evaluate without checkpoint.")
    exit(1)

# Load Val Data
print("Loading Validation Dataset...")
try:
    val_ds = HSIPatchesDataset(DATA_DIR, split='Val')
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True, shuffle=False)
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit(1)

# Evaluate
print("Running evaluation...")
all_preds = []
all_masks = []

with torch.no_grad():
    from tqdm import tqdm
    for data, target in tqdm(val_loader):
        data = data.to(DEVICE)
        target = target.cpu().numpy()
        
        output = model(data)
        preds = torch.argmax(torch.softmax(output, dim=1), dim=1).cpu().numpy()
        
        all_preds.extend(preds)
        all_masks.extend(target)

print("Computing metrics...")
results_tuple = evaluate_segmentation(all_masks, all_preds, num_classes=NUM_CLASSES)
(confusion_matrix_sum, true_positive_sum, true_negative_sum, false_positive_sum,
 false_negative_sum, precision, recall, f1_score, pixel_accuracy_per_class,
 pixel_accuracy, iou, dice_coefficient, kappa) = results_tuple

print("="*60)
print("RESULTS")
print("="*60)
print(f"Overall Accuracy: {pixel_accuracy:.4f}")
print(f"Mean IoU: {np.nanmean(iou):.4f}")
if isinstance(f1_score, (list, np.ndarray)):
    print(f"Mean F1: {np.nanmean(f1_score):.4f}")
else:
    print(f"Mean F1: {f1_score:.4f}")
print(f"Kappa: {kappa:.4f}")
print()
print("Per-Class IoU:")
class_names = ["Background", "Capacitor", "IC", "Connector"]
for i, name in enumerate(class_names):
    val = iou[i] if isinstance(iou, (list, np.ndarray)) else 0
    print(f"  {name:12s}: {val:.4f}")
print("="*60)

# Save
output_file = "Results/hsi_resunet_cutmix_val_metrics.json"
results = {
    "model": "ResUNet",
    "augmentation": "CutMix",
    "mIoU": float(np.nanmean(iou)),
    "OA": float(pixel_accuracy),
    "per_class_iou": {n: float(iou[i]) for i, n in enumerate(class_names)}
}
with open(output_file, "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved to {output_file}")
