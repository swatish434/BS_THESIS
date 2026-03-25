
import sys
import os
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import cv2
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from utils.dataset_functions import read_dataset, evaluate_segmentation
from utils.repro import resolve_device
from models.factory import get_model

# Constants
DATASET_PATH = "/home/bs_thesis/Documents/BS_THESIS/DATASETS/PCBDataset/"
IMG_RES_RGB = 640
NUM_CLASSES = 4
RGB_MEAN = [49.378, 41.347, 28.657]
RGB_STD = [7.538, 7.043, 4.648]

def get_hsi_center_crop(hsi, mask, crop_size=256):
    h, w, c = hsi.shape
    cy, cx = h // 2, w // 2
    half = crop_size // 2
    y1, x1 = max(0, cy - half), max(0, cx - half)
    y2, x2 = min(h, y1 + crop_size), min(w, x1 + crop_size)
    
    patch_hsi = hsi[y1:y2, x1:x2, :]
    patch_mask = mask[y1:y2, x1:x2]
    
    ph, pw, _ = patch_hsi.shape
    if ph < crop_size or pw < crop_size:
        pad_h = crop_size - ph
        pad_w = crop_size - pw
        patch_hsi = np.pad(patch_hsi, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
        patch_mask = np.pad(patch_mask, ((0, pad_h), (0, pad_w)), mode='constant')
        
    return patch_hsi, patch_mask

def normalize_rgb(img):
    img = img.astype(np.float32)
    img = (img - RGB_MEAN) / RGB_STD
    return img

def clipping_neg_pos(hscubes):
    clipped = []
    for i in tqdm(range(len(hscubes)), desc="Clipping HSI"):
        if hscubes[i] is None:
            clipped.append(None)
            continue
        img = hscubes[i].copy()
        img[img < 0.0] = 0.0
        img[img > 1.0] = 1.0
        clipped.append(img)
    return clipped

def slicing(hscubes, x):
    sliced = []
    for i in tqdm(range(len(hscubes)), desc="Slicing HSI"):
        if hscubes[i] is None:
            sliced.append(None)
            continue
        img = hscubes[i].copy()
        img = img[:, :, x:]
        sliced.append(img)
    return sliced

def evaluate_model(model_name, model_path, modality, arch, test_indices, RGB, RGB_masks, HSI, HSI_masks, device):
    print(f"Evaluating {model_name} ({modality})...")
    
    in_channels = 3 if modality == 'RGB' else 214
    
    try:
        model = get_model(arch, in_channels=in_channels, out_channels=NUM_CLASSES)
        state = torch.load(model_path, map_location=device)
        if 'model_state' in state: state = state['model_state']
        model.load_state_dict(state)
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"Failed to load {model_name}: {e}")
        return None

    predicted_masks = []
    ground_truths = []

    with torch.no_grad():
        for idx in tqdm(test_indices):
            if modality == 'RGB':
                img = RGB[idx]
                mask = RGB_masks[idx]
                if img is None: continue
                
                img_resized = cv2.resize(img, (IMG_RES_RGB, IMG_RES_RGB), interpolation=cv2.INTER_LINEAR)
                mask_resized = cv2.resize(mask, (IMG_RES_RGB, IMG_RES_RGB), interpolation=cv2.INTER_NEAREST)
                
                img_norm = normalize_rgb(img_resized)
                tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
                
                output = model(tensor)
                pred = torch.argmax(output, dim=1).cpu().numpy()[0]
                
                predicted_masks.append(pred)
                ground_truths.append(mask_resized)
                
            elif modality == 'HSI':
                hsi = HSI[idx]
                mask = HSI_masks[idx]
                if hsi is None: continue
                
                patch_hsi, patch_mask = get_hsi_center_crop(hsi, mask, crop_size=256)
                
                tensor = torch.from_numpy(patch_hsi.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
                
                output = model(tensor)
                pred = torch.argmax(output, dim=1).cpu().numpy()[0]
                
                predicted_masks.append(pred)
                ground_truths.append(patch_mask)

    results = evaluate_segmentation(ground_truths, predicted_masks, NUM_CLASSES)
    return results

def main():
    device = resolve_device('auto')
    print("Loading Dataset...")
    HSI, HSI_general_masks, _, RGB, _, RGB_general_masks, _ = read_dataset(DATASET_PATH)
    
    print("Preprocessing HSI (Clipping and Slicing 10 bands)...")
    HSI = clipping_neg_pos(HSI)
    HSI = slicing(HSI, 10)
    
    valid_indices = []
    for i in range(len(RGB)):
        if RGB[i] is not None:
            valid_indices.append(i)
            
    train_idx, test_idx = train_test_split(valid_indices, test_size=0.2, random_state=123)
    print(f"Test Set Size: {len(test_idx)}")
    
    models = [
        {
            'name': 'RGB_DeepLabv3+_Baseline (No Aug)',
            'path': '/home/bs_thesis/Documents/BS_THESIS/PCBVision/RGB_Experiments/results/RGB_deeplabv3+_baseline_best.pth',
            'modality': 'RGB',
            'arch': 'deeplabv3+'
        },
        {
            'name': 'RGB_DeepLabv3+_CP (Copy-Paste)',
            'path': '/home/bs_thesis/Documents/BS_THESIS/PCBVision/RGB_Experiments/runs/20260202_172507_rgb_deeplabv3+_cp/RGB_deeplabv3+_best.pth',
            'modality': 'RGB',
             'arch': 'deeplabv3+'
        },
        {
            'name': 'RGB_DeepLabv3+_SDLoRA',
            'path': '/home/bs_thesis/Documents/BS_THESIS/PCBVision/RGB_Experiments/runs/20260204_165437_rgb_deeplabv3+_sdlora/RGB_deeplabv3+_best.pth',
            'modality': 'RGB',
             'arch': 'deeplabv3+'
        },
        {
            'name': 'RGB_UNet_Baseline',
             'path': '/home/bs_thesis/Documents/BS_THESIS/PCBVision/RGB_Experiments/results/RGB_unet_best.pth',
             'modality': 'RGB',
              'arch': 'unet'
        },
        {
             'name': 'HSI_UNet_Baseline',
             'path': '/home/bs_thesis/Documents/BS_THESIS/PCBVision/HSI_Experiments/results/hsi_patches_unet_best.pth',
             'modality': 'HSI',
             'arch': 'unet'
        },
        {
             'name': 'HSI_UNet_CP (Patches)',
             'path': '/home/bs_thesis/Documents/BS_THESIS/PCBVision/HSI_Experiments/results/hsi_patches_cp_best.pth',
             'modality': 'HSI',
             'arch': 'unet'
        },
        {
             'name': 'HSI_UNet_Overlap',
             'path': '/home/bs_thesis/Documents/BS_THESIS/PCBVision/HSI_Experiments/results/hsi_overlap_best.pth',
             'modality': 'HSI',
             'arch': 'unet'
        },
        {
             'name': 'HSI_UNet_CP_256',
             'path': '/home/bs_thesis/Documents/BS_THESIS/PCBVision/HSI_Experiments/results/hsi_256_cp_best.pth',
             'modality': 'HSI',
             'arch': 'unet'
        },
        {
             'name': 'HSI_AttnUNet_Overlap',
             'path': '/home/bs_thesis/Documents/BS_THESIS/PCBVision/HSI_Experiments/results/hsi_overlap_attunet_best.pth',
             'modality': 'HSI',
             'arch': 'attention_unet'
        },
         {
             'name': 'HSI_DeepLabv3+_Overlap',
             'path': '/home/bs_thesis/Documents/BS_THESIS/PCBVision/HSI_Experiments/results/hsi_overlap_deeplabv3+_best.pth',
             'modality': 'HSI',
             'arch': 'deeplabv3+'
        }
    ]
    
    rows = []
    
    for m in models:
        res = evaluate_model(m['name'], m['path'], m['modality'], m['arch'], test_idx, RGB, RGB_general_masks, HSI, HSI_general_masks, device)
        if res is None: continue
        
        _, true_positives, true_negatives, false_positives, false_negatives, precision, recall, f1, _, pixel_accuracy, iou, _, global_kappa = res
        
        # Calculate Per-Class Kappa
        total = true_positives + true_negatives + false_positives + false_negatives
        # Avoid division by zero
        po = (true_positives + true_negatives) / total
        pe_num = (true_positives + false_positives) * (true_positives + false_negatives) + \
                 (true_negatives + false_negatives) * (true_negatives + false_positives)
        pe = pe_num / (total**2)
        kappa_per_class = (po - pe) / (1 - pe)
        kappa_per_class = np.nan_to_num(kappa_per_class)

        # Calculate Means
        mIoU = np.nanmean(iou)
        mPrec = np.nanmean(precision)
        mRec = np.nanmean(recall)
        mF1 = np.nanmean(f1)
        mKappa = np.nanmean(kappa_per_class)
        
        row = {
            'Model': m['name'],
            'Global_Kappa': global_kappa,
            'Pixel_Acc': pixel_accuracy,
            'mIoU': mIoU, 'mPrec': mPrec, 'mRec': mRec, 'mF1': mF1, 'mKappa': mKappa,
            'IoU_Bg': iou[0], 'IoU_Comp': iou[1], 'IoU_IC': iou[2], 'IoU_Conn': iou[3],
            'Prec_Bg': precision[0], 'Prec_Comp': precision[1], 'Prec_IC': precision[2], 'Prec_Conn': precision[3],
            'Rec_Bg': recall[0], 'Rec_Comp': recall[1], 'Rec_IC': recall[2], 'Rec_Conn': recall[3],
            'F1_Bg': f1[0], 'F1_Comp': f1[1], 'F1_IC': f1[2], 'F1_Conn': f1[3],
            'Kap_Bg': kappa_per_class[0], 'Kap_Comp': kappa_per_class[1], 'Kap_IC': kappa_per_class[2], 'Kap_Conn': kappa_per_class[3],
        }
        rows.append(row)

    print("\n\n" + "="*80)
    print("TABLE 1: SUMMARIZED RESULTS AND METRICS")
    print("="*80)
    print("| Model | Pixel Accuracy | Mean IoU | Mean F1 Score | Mean Precision | Mean Recall | Kappa |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    for r in rows:
        print(f"| {r['Model']} | {r['Pixel_Acc']:.4f} | {r['mIoU']:.4f} | {r['mF1']:.4f} | {r['mPrec']:.4f} | {r['mRec']:.4f} | {r['Global_Kappa']:.4f} |")

    print("\n\n" + "="*80)
    print("TABLE V: CLASS-WISE METRICS")
    print("="*80)
    
    for r in rows:
        print(f"\n### {r['Model']}")
        print(f"**Global Kappa Score**: {r['Global_Kappa']:.4f}")
        print("| Metric | Mean | Background | Component | IC | Connector |")
        print("| :--- | :--- | :--- | :--- | :--- | :--- |")
        print(f"| IoU | {r['mIoU']:.4f} | {r['IoU_Bg']:.4f} | {r['IoU_Comp']:.4f} | {r['IoU_IC']:.4f} | {r['IoU_Conn']:.4f} |")
        print(f"| Precision | {r['mPrec']:.4f} | {r['Prec_Bg']:.4f} | {r['Prec_Comp']:.4f} | {r['Prec_IC']:.4f} | {r['Prec_Conn']:.4f} |")
        print(f"| Recall | {r['mRec']:.4f} | {r['Rec_Bg']:.4f} | {r['Rec_Comp']:.4f} | {r['Rec_IC']:.4f} | {r['Rec_Conn']:.4f} |")
        print(f"| F1 Score | {r['mF1']:.4f} | {r['F1_Bg']:.4f} | {r['F1_Comp']:.4f} | {r['F1_IC']:.4f} | {r['F1_Conn']:.4f} |")
        print(f"| Kappa | {r['Global_Kappa']:.4f} | {r['Kap_Bg']:.4f} | {r['Kap_Comp']:.4f} | {r['Kap_IC']:.4f} | {r['Kap_Conn']:.4f} |")

if __name__ == "__main__":
    main()
