#!/usr/bin/env python3
"""
Evaluate Sim2Real Fine-tuned Model.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

from utils.dataset_functions import read_dataset
from models.DeepLabv3_plus import DeepLabv3_plus

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def compute_metrics(pred, target, num_classes=4):
    """Compute per-class IoU and other metrics."""
    pred = pred.flatten()
    target = target.flatten()
    
    # Confusion matrix
    cm = confusion_matrix(target, pred, labels=list(range(num_classes)))
    
    # Per-class IoU
    iou_per_class = []
    for i in range(num_classes):
        intersection = cm[i, i]
        union = cm[i, :].sum() + cm[:, i].sum() - intersection
        if union > 0:
            iou_per_class.append(intersection / union)
        else:
            iou_per_class.append(0.0)
    
    # Mean IoU
    mean_iou = np.mean(iou_per_class)
    
    # Overall accuracy
    accuracy = np.diag(cm).sum() / cm.sum()
    
    return {
        'accuracy': accuracy,
        'mean_iou': mean_iou,
        'iou_per_class': iou_per_class,
        'confusion_matrix': cm
    }

def main():
    model_path = "Results/Sim2Real/stage2_real_finetuned.pth"
    dataset_path = "/home/bs_thesis/Documents/BS_THESIS/DATASETS/PCBDataset/"
    img_size = 640
    
    print(f"=== Evaluating Sim2Real Model: {model_path} ===")
    print(f"Device: {device}")
    
    # Load model
    model = DeepLabv3_plus(nInputChannels=3, n_classes=4).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
        model.eval()
        print("Model loaded successfully")
    else:
        print(f"Error: Model file {model_path} not found!")
        return
    
    # Load test data (same test set as usual)
    test_indices = [2, 5, 6, 7, 9, 10, 12, 13, 14, 15, 16, 19, 20, 21, 26, 27, 28, 29, 30, 31, 33, 36, 38, 39, 40, 41, 43, 46, 48, 51]
    
    _, general_masks, _, rgb, _, _, _ = read_dataset(dataset_path)
    
    all_preds = []
    all_targets = []
    
    print(f"\nEvaluating on {len(test_indices)} test samples...")
    
    for idx in tqdm(test_indices):
        if idx >= len(rgb) or rgb[idx] is None:
            continue
            
        img = rgb[idx]
        mask = general_masks[idx]
        
        # Resize
        img = cv2.resize(img, (img_size, img_size))
        mask = cv2.resize(mask.astype(np.uint8), (img_size, img_size), interpolation=cv2.INTER_NEAREST)
        
        # To tensor
        img_tensor = torch.from_numpy(img.transpose(2, 0, 1).astype(np.float32) / 255.0).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            output = model(img_tensor)
            pred = output.argmax(dim=1).squeeze().cpu().numpy()
        
        all_preds.append(pred)
        all_targets.append(mask)
    
    # Compute metrics
    all_preds = np.concatenate([p.flatten() for p in all_preds])
    all_targets = np.concatenate([t.flatten() for t in all_targets])
    
    metrics = compute_metrics(all_preds, all_targets, num_classes=4)
    
    # Print results
    class_names = ['Background/Others', 'IC', 'Capacitor', 'Connector']
    
    print("\n" + "="*60)
    print("SIM2REAL EVALUATION RESULTS")
    print("="*60)
    print(f"\nOverall Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"Mean IoU: {metrics['mean_iou']*100:.2f}%")
    
    print("\nPer-Class IoU:")
    print("-"*40)
    for i, name in enumerate(class_names):
        print(f"  {name:20s}: {metrics['iou_per_class'][i]*100:.2f}%")
    
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
    
    # Save results
    results_file = "Results/Sim2Real/evaluation_results.txt"
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    with open(results_file, 'w') as f:
        f.write("=== Sim2Real DeepLabv3+ Evaluation ===\n\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Test samples: {len(test_indices)}\n\n")
        f.write(f"Overall Accuracy: {metrics['accuracy']*100:.2f}%\n")
        f.write(f"Mean IoU: {metrics['mean_iou']*100:.2f}%\n\n")
        f.write("Per-Class IoU:\n")
        for i, name in enumerate(class_names):
            f.write(f"  {name}: {metrics['iou_per_class'][i]*100:.2f}%\n")
    
    print(f"\nResults saved to: {results_file}")

if __name__ == "__main__":
    main()
