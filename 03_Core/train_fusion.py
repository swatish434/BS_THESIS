
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import argparse

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.FusionModel import RGBHSIFusionModel
from utils.dataset_functions import PCBFullDataset
from utils.loss_functions import HybridLoss
from utils.augmentation_functions import MultimodalCutMix

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(loader, desc="Training")
    for batch in pbar:
        rgb = batch['rgb'].to(device)
        hsi = batch['hsi'].to(device)
        mask = batch['mask'].to(device)
        
        # Slice HSI if needed (224 -> 214)
        if hsi.shape[1] > 214:
            hsi = hsi[:, :214, :, :]
            
        optimizer.zero_grad()
        outputs = model(rgb, hsi)
        loss = criterion(outputs, mask)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
        
    return running_loss / len(loader)

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    intersections = 0
    unions = 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
            rgb = batch['rgb'].to(device)
            hsi = batch['hsi'].to(device)
            mask = batch['mask'].to(device)
            
            if hsi.shape[1] > 214:
                hsi = hsi[:, :214, :, :]
                
            outputs = model(rgb, hsi)
            loss = criterion(outputs, mask)
            running_loss += loss.item()
            
            preds = torch.argmax(outputs, dim=1)
            
            # Metrics
            correct += (preds == mask).sum().item()
            total += mask.numel()
            
            # mIoU approximation
            # Simple per-batch IoU logging (full calc in eval script)
    
    return running_loss / len(loader), correct / total

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=2) # Reduced from 4
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lora_weights', type=str, default=None, help="Path to SD-LoRA RGB weights")
    parser.add_argument('--amp', action='store_true', default=True, help="Use Mixed Precision Training") # Added AMP
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # ... (Dataset setup same) ...
    dataset_root = "/home/bs_thesis/Documents/BS_THESIS/DATASETS/PCBDataset/"
    all_ids = list(range(1, 54))
    
    # Strictly exclude the test indices used in evaluation script to prevent leakage
    # Indices from evaluate_unified.py
    test_indices = [2, 5, 6, 7, 9, 10, 12, 13, 14, 15, 16, 19, 20, 21, 26, 27, 28, 29, 30, 31, 33, 36, 38, 39, 40, 41, 43, 46, 48, 51]
    
    # Training IDs = All IDs - Test Indices
    train_val_pool = [x for x in all_ids if x not in test_indices]
    
    from sklearn.model_selection import train_test_split
    # Split remaining pool into Train/Val
    train_ids, val_ids = train_test_split(train_val_pool, test_size=0.2, random_state=42)
    
    print(f"Total: {len(all_ids)}, Test(Held-out): {len(test_indices)}")
    print(f"Train+Val Pool: {len(train_val_pool)}")
    print(f"Train: {len(train_ids)}, Val: {len(val_ids)}")
    
    cutmix = MultimodalCutMix(beta=1.0)
    
    # ... (Dataset creation same) ...
    train_dataset = PCBFullDataset(train_ids, dataset_root, augment=True, copy_paste_aug=cutmix)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dataset = PCBFullDataset(val_ids, dataset_root, augment=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    
    # ... (Model setup same) ...
    model = RGBHSIFusionModel(num_classes=4, hsi_channels=214).to(device)
    
    if args.lora_weights:
        # ... (Loading weights same) ...
        print(f"Loading RGB weights from {args.lora_weights}")
        try:
            checkpoint = torch.load(args.lora_weights, map_location=device)
            state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
            
            # Filter keys to match model.rgb_model prefix
            # RGB weights are for DeepLabv3_plus directly, so we load into model.rgb_model
            model.rgb_model.load_state_dict(state_dict)
            print("Loaded RGB weights successfully.")
        except Exception as e:
            print(f"Warning: Could not strictly load RGB weights: {e}")
            
    criterion = HybridLoss(focal_weight=0.5, dice_weight=0.5)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # AMP Scaler
    scaler = torch.cuda.amp.GradScaler() if args.amp and device.type == 'cuda' else None
    
    save_dir = "Results/Fusion"
    os.makedirs(save_dir, exist_ok=True)
    
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        # Train Loop with AMP
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch in pbar:
            rgb = batch['rgb'].to(device)
            hsi = batch['hsi'].to(device)
            mask = batch['mask'].to(device)
            
            if hsi.shape[1] > 214:
                hsi = hsi[:, :214, :, :]
                
            optimizer.zero_grad()
            
            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(rgb, hsi)
                    loss = criterion(outputs, mask)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(rgb, hsi)
                loss = criterion(outputs, mask)
                loss.backward()
                optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        train_loss = running_loss / len(train_loader)
        
        # Validation Loop
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "fusion_best.pth"))
            print("Saved Best Model")
            
    torch.save(model.state_dict(), os.path.join(save_dir, "fusion_last.pth"))

if __name__ == "__main__":
    main()
