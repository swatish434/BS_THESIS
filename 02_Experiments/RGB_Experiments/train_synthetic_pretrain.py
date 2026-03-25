#!/usr/bin/env python3
"""
Sim2Real Stage 1: Pre-train on Synthetic Data

Trains DeepLabv3+ from scratch using ONLY 1000 synthetic images.
Goal: Learn robust feature extractors for components (Shape, Color) independent of real-world noise.

Usage:
    python3 RGB_Experiments/train_synthetic_pretrain.py --synthetic_dir data/synthetic_1000
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
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

class SyntheticDataset(Dataset):
    """Synthetic PCB Dataset (from generate_layouts.py)."""
    
    def __init__(self, synthetic_dir, img_size=640, augment=True):
        self.img_size = img_size
        self.augment = augment
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
        
        # Simple Augmentation (Flip/Rotate)
        if self.augment:
            if np.random.rand() < 0.5:
                img = cv2.flip(img, 1)
                mask = cv2.flip(mask, 1)
            if np.random.rand() < 0.5:
                img = cv2.flip(img, 0)
                mask = cv2.flip(mask, 0)
        
        # To tensor
        img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img)
        mask_tensor = torch.from_numpy(mask.astype(np.int64))
        
        return img_tensor, mask_tensor

class RealPCBDataset(Dataset):
    """Real PCB Dataset (Validation only)."""
    
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
                
        print(f"Loaded {len(self.samples)} real validation samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img, mask = self.samples[idx]
        
        # To tensor
        img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img)
        mask_tensor = torch.from_numpy(mask.astype(np.int64))
        
        return img_tensor, mask_tensor

# ============================================================
# Training Loop
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
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--img_size', type=int, default=640)
    args = parser.parse_args()
    
    print(f"=== Sim2Real STAGE 1: Pre-training on Synthetic Data ===")
    print(f"Synthetic dir: {args.synthetic_dir}")
    print(f"Device: {device}")
    
    # Validation split (using real data to monitor generalization)
    val_indices = [18, 37, 42]
    
    # Create datasets
    train_dataset = SyntheticDataset(args.synthetic_dir, args.img_size, augment=True)
    real_val = RealPCBDataset(args.dataset_path, val_indices, args.img_size)
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(real_val, batch_size=1, shuffle=False)
    
    # Model
    model = DeepLabv3_plus(nInputChannels=3, n_classes=4).to(device)
    
    # Loss and optimizer
    loss_fn = HybridLoss(focal_weight=0.5, dice_weight=0.5)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training
    save_dir = "Results/Sim2Real"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "stage1_synthetic_pretrain.pth")
    
    best_loss = float('inf')
    
    print(f"\nStarting pre-training for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        train_loss = train_epoch(train_loader, model, optimizer, loss_fn)
        # Note: Validation on REAL data might be poor initially, that's expected
        val_loss, val_acc = evaluate(val_loader, model, loss_fn)
        
        print(f"Epoch {epoch+1}/{args.epochs}: Train Loss={train_loss:.4f}, Val Loss (Real)={val_loss:.4f}, Val Acc={val_acc:.2f}%")
        
        # Save every improvement on training loss (since we care about learning synthetic features first)
        # Or save best val loss if we hope for some generalization
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"  Saved best model")
            
    # Always save final model for fine-tuning
    final_path = os.path.join(save_dir, "stage1_synthetic_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"\nPre-training complete! Weights saved to: {final_path}")

if __name__ == "__main__":
    main()
