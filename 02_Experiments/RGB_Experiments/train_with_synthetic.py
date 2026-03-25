#!/usr/bin/env python3
"""
Train RGB models using Real + Synthetic data.

Combines real PCB images with synthetically generated layouts for improved
minority class (Capacitor, Connector) performance.

Usage:
    python3 RGB_Experiments/train_with_synthetic.py --synthetic_dir data/synthetic_1000
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import argparse
from glob import glob

from utils.dataset_functions import read_dataset
from utils.loss_functions import HybridLoss
from models.DeepLabv3_plus import DeepLabv3_plus

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================
# Datasets
# ============================================================

class RealPCBDataset(Dataset):
    """Real PCB Dataset (from PCBDataset folder)."""
    
    def __init__(self, dataset_path, indices, img_size=640):
        self.img_size = img_size
        self.samples = []
        
        # Load dataset
        _, general_masks, _, rgb, _, _, _ = read_dataset(dataset_path)
        
        for idx in indices:
            if idx < len(rgb) and rgb[idx] is not None:
                img = rgb[idx]
                mask = general_masks[idx]
                
                # Resize
                img = cv2.resize(img, (img_size, img_size))
                mask = cv2.resize(mask.astype(np.uint8), (img_size, img_size), 
                                 interpolation=cv2.INTER_NEAREST)
                
                self.samples.append((img, mask))
                
        print(f"Loaded {len(self.samples)} real samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img, mask = self.samples[idx]
        
        # To tensor
        img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img)
        mask_tensor = torch.from_numpy(mask.astype(np.int64))
        
        return img_tensor, mask_tensor


class SyntheticDataset(Dataset):
    """Synthetic PCB Dataset (from generate_layouts.py)."""
    
    def __init__(self, synthetic_dir, img_size=640):
        self.img_size = img_size
        self.samples = []
        
        # Find all synthetic images
        img_paths = sorted(glob(os.path.join(synthetic_dir, "syn_*.png")))
        
        for img_path in img_paths:
            mask_path = img_path.replace('.png', '_mask.npy')
            if os.path.exists(mask_path):
                self.samples.append((img_path, mask_path))
                
        print(f"Loaded {len(self.samples)} synthetic samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        
        # Load
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = np.load(mask_path)
        if mask.ndim == 3: mask = mask.squeeze()
        
        # Resize
        img = cv2.resize(img, (self.img_size, self.img_size))
        mask = cv2.resize(mask.astype(np.uint8), (self.img_size, self.img_size),
                         interpolation=cv2.INTER_NEAREST)
        
        # To tensor
        img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img)
        mask_tensor = torch.from_numpy(mask.astype(np.int64))
        
        return img_tensor, mask_tensor


# ============================================================
# Training
# ============================================================

def train_epoch(loader, model, optimizer, loss_fn):
    model.train()
    total_loss = 0
    for data, targets in tqdm(loader, desc="Training"):
        data, targets = data.to(device), targets.to(device)
        optimizer.zero_grad()
        preds = model(data)
        loss = loss_fn(preds, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(loader, model, loss_fn):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, targets in tqdm(loader, desc="Evaluating"):
            data, targets = data.to(device), targets.to(device)
            preds = model(data)
            loss = loss_fn(preds, targets)
            total_loss += loss.item()
            
            pred_classes = preds.argmax(dim=1)
            correct += (pred_classes == targets).sum().item()
            total += targets.numel()
    
    accuracy = correct / total * 100
    return total_loss / len(loader), accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--synthetic_dir', type=str, default='data/synthetic_1000')
    parser.add_argument('--dataset_path', type=str, 
                        default='/home/bs_thesis/Documents/BS_THESIS/DATASETS/PCBDataset/')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--img_size', type=int, default=640)
    args = parser.parse_args()
    
    print(f"=== Training with Real + Synthetic Data ===")
    print(f"Synthetic dir: {args.synthetic_dir}")
    print(f"Device: {device}")
    
    # Dataset splits (same as train_rgb.py)
    train_indices = [1,3,8,11,17,22,23,24,25,32,34,44,45,47,49,50,52,53]
    val_indices = [18, 37, 42]
    test_indices = [2, 5, 6, 7, 9, 10, 12, 13, 14, 15, 16, 19, 20, 21, 26, 27, 28, 29, 30, 31, 33, 36, 38, 39, 40, 41, 43, 46, 48, 51]
    
    # Create datasets
    real_train = RealPCBDataset(args.dataset_path, train_indices, args.img_size)
    real_val = RealPCBDataset(args.dataset_path, val_indices, args.img_size)
    real_test = RealPCBDataset(args.dataset_path, test_indices, args.img_size)
    synthetic = SyntheticDataset(args.synthetic_dir, args.img_size)
    
    # Combine real + synthetic for training
    combined_train = ConcatDataset([real_train, synthetic])
    
    # ===== BALANCED SAMPLING =====
    # Create weights: real samples get higher weight to balance with synthetic
    n_real = len(real_train)
    n_synthetic = len(synthetic)
    
    # Target: 50% real, 50% synthetic in each epoch
    # Weight for each sample = 1 / (number_of_samples_in_its_class)
    weight_real = n_synthetic / n_real if n_real > 0 else 1.0  # Higher weight for real
    weight_synthetic = 1.0
    
    weights = [weight_real] * n_real + [weight_synthetic] * n_synthetic
    
    from torch.utils.data import WeightedRandomSampler
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=2 * n_synthetic,  # Epoch size: 2x synthetic (so ~50% real, ~50% synthetic)
        replacement=True  # Must be True for oversampling real
    )
    
    print(f"Combined training set: {len(combined_train)} samples ({n_real} real + {n_synthetic} synthetic)")
    print(f"Balanced sampling: weight_real={weight_real:.1f}x, epoch_samples={2*n_synthetic}")
    
    # DataLoaders
    train_loader = DataLoader(combined_train, batch_size=args.batch_size, sampler=sampler, num_workers=2)
    val_loader = DataLoader(real_val, batch_size=1, shuffle=False)
    test_loader = DataLoader(real_test, batch_size=1, shuffle=False)
    
    # Model
    model = DeepLabv3_plus(nInputChannels=3, n_classes=4).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    loss_fn = HybridLoss(focal_weight=0.5, dice_weight=0.5)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training
    save_path = "Results/deeplabv3_real_synthetic_best.pth"
    os.makedirs("Results", exist_ok=True)
    
    best_loss = float('inf')
    patience = 15
    patience_counter = 0
    
    print(f"\nStarting training for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        train_loss = train_epoch(train_loader, model, optimizer, loss_fn)
        val_loss, val_acc = evaluate(val_loader, model, loss_fn)
        
        print(f"Epoch {epoch+1}/{args.epochs}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%")
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"  Saved best model")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping after {epoch+1} epochs")
                break
    
    # Final evaluation
    print("\n=== Final Evaluation on Real Test Set ===")
    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=False))
    test_loss, test_acc = evaluate(test_loader, model, loss_fn)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")
    
    print(f"\nTraining complete! Model saved to: {save_path}")


if __name__ == "__main__":
    main()
