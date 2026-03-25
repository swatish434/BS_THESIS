#!/usr/bin/env python3
"""
Sim2Real Stage 2: Fine-tune on Real Data

Loads pre-trained synthetic model and fine-tunes on 16 real images.
Uses heavy augmentation and low learning rate to adapt to the real domain.

Usage:
    python3 RGB_Experiments/train_finetune_real.py --pretrained_path Results/Sim2Real/stage1_synthetic_final.pth
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
import random
import albumentations as A

from utils.dataset_functions import read_dataset
from utils.loss_functions import HybridLoss
from models.DeepLabv3_plus import DeepLabv3_plus

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================
# Datasets with Heavy Augmentation
# ============================================================

class RealPCBDataset(Dataset):
    """Real PCB Dataset with heavy augmentation."""
    
    def __init__(self, dataset_path, indices, img_size=640, augment=True):
        self.img_size = img_size
        self.augment = augment
        self.samples = []
        
        # Define heavy augmentations
        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.GaussNoise(p=0.2),
            # Resize is handled separately to ensure mask consistency
        ])
        
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
                
        print(f"Loaded {len(self.samples)} real samples (augment={augment})")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img, mask = self.samples[idx]
        
        if self.augment:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
            
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
    for data, targets in tqdm(loader, desc="Fine-tuning"):
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
    parser.add_argument('--pretrained_path', type=str, required=True, 
                        help="Path to synthetic pre-trained weights")
    parser.add_argument('--dataset_path', type=str, 
                        default='/home/bs_thesis/Documents/BS_THESIS/DATASETS/PCBDataset/')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-5) # Low LR for fine-tuning
    parser.add_argument('--img_size', type=int, default=640)
    args = parser.parse_args()
    
    print(f"=== Sim2Real STAGE 2: Fine-tuning on Real Data ===")
    print(f"Pre-trained weights: {args.pretrained_path}")
    print(f"Device: {device}")
    
    # Dataset splits
    train_indices = [1,3,8,11,17,22,23,24,25,32,34,44,45,47,49,50,52,53]
    val_indices = [18, 37, 42]
    
    # Create datasets
    real_train = RealPCBDataset(args.dataset_path, train_indices, args.img_size, augment=True)
    real_val = RealPCBDataset(args.dataset_path, val_indices, args.img_size, augment=False)
    
    # DataLoaders
    train_loader = DataLoader(real_train, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(real_val, batch_size=1, shuffle=False)
    
    # Model
    model = DeepLabv3_plus(nInputChannels=3, n_classes=4).to(device)
    
    # Load pre-trained weights
    if os.path.exists(args.pretrained_path):
        print(f"Loading weights from {args.pretrained_path}...")
        model.load_state_dict(torch.load(args.pretrained_path, map_location=device, weights_only=False))
    else:
        print(f"Error: Pre-trained path {args.pretrained_path} does not exist!")
        return
    
    # Loss and optimizer
    loss_fn = HybridLoss(focal_weight=0.5, dice_weight=0.5)
    
    # Freeze Encoder for fine-tuning
    print("Freezing encoder layers (ResNet + ASPP)...")
    
    # 1. Freeze ResNet Backbone
    for param in model.resnet_features.parameters():
        param.requires_grad = False
        
    # 2. Freeze ASPP module (components are separate attributes)
    aspp_modules = [model.aspp1, model.aspp2, model.aspp3, model.aspp4, 
                    model.global_avg_pool, model.conv1, model.bn1]
    for module in aspp_modules:
        for param in module.parameters():
            param.requires_grad = False
            
    # Train only Decoder (conv2, bn2, last_conv)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters: {len(trainable_params)}")
    
    optimizer = optim.Adam(trainable_params, lr=args.lr)
    
    # Training
    save_dir = "Results/Sim2Real"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "stage2_real_finetuned.pth")
    
    best_loss = float('inf')
    
    print(f"\nStarting fine-tuning for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        train_loss = train_epoch(train_loader, model, optimizer, loss_fn)
        val_loss, val_acc = evaluate(val_loader, model, loss_fn)
        
        print(f"Epoch {epoch+1}/{args.epochs}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%")
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"  Saved best model")
            
    print(f"\nFine-tuning complete! Model saved to: {save_path}")

if __name__ == "__main__":
    main()
