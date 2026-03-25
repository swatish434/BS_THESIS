
import sys
import os
import torch
import numpy as np
import spectral.io.envi as envi
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from PCBVision.train_hsi_overlap import HSIPatchesDataset, get_hsi_model
from PCBVision.utils.loss_functions import HybridLoss
from PCBVision.utils.augmentation_functions import MultimodalCutMix

DATA_DIR = "/home/bs_thesis/Documents/BS_THESIS/PCBVision/Patches_256_Overlap_Data/" 
IN_CHANNELS = 214
OUT_CHANNELS = 4
IMG_RES = 256

import argparse

def evaluate_hsi():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='deeplabv3+', help='Model name')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load Val Set
    val_ds = HSIPatchesDataset(DATA_DIR, split='Val')
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    
    # Load Model
    model_path = args.checkpoint
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return

    print(f"Loading model: {args.model} from {model_path}")
    model = get_hsi_model(args.model, IN_CHANNELS, OUT_CHANNELS).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    total_confusion_matrix = np.zeros((OUT_CHANNELS, OUT_CHANNELS))
    
    print("Evaluating...")
    with torch.no_grad():
        for data, target in val_loader:
            data = data.to(device)
            target = target.cpu().numpy().flatten()
            
            output = model(data)
            pred = torch.argmax(torch.softmax(output, dim=1), dim=1).cpu().numpy().flatten()
            
            cm = confusion_matrix(target, pred, labels=range(OUT_CHANNELS))
            total_confusion_matrix += cm

    # Metrics
    pixel_acc = np.diag(total_confusion_matrix).sum() / total_confusion_matrix.sum()
    
    ious = []
    f1s = []
    precisions = []
    recalls = []
    
    for i in range(OUT_CHANNELS):
        tp = total_confusion_matrix[i, i]
        fp = total_confusion_matrix[:, i].sum() - tp
        fn = total_confusion_matrix[i, :].sum() - tp
        
        iou = tp / (tp + fp + fn + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        ious.append(iou)
        f1s.append(f1)
        precisions.append(precision)
        recalls.append(recall)

    print("\n" + "="*50)
    print(f"FINAL EVALUATION RESULTS ({args.model} HSI CutMix)")
    print("="*50)
    print(f"Pixel Accuracy: {pixel_acc:.4f}")
    print(f"Mean IoU: {np.mean(ious):.4f}")
    print(f"Class IoUs: {ious}")
    print(f"Mean F1: {np.mean(f1s):.4f}")
    print("="*50)

if __name__ == "__main__":
    evaluate_hsi()
