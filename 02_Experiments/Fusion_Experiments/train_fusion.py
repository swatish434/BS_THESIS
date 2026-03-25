
import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from Fusion_Experiments.fusion_dataset import FusionDataset
from models.factory import get_model
from utils.dataset_functions import evaluate_segmentation

# Constants
NUM_CLASSES = 4
DATA_DIR = "/home/bs_thesis/Documents/BS_THESIS/PCBVision/Fusion_Experiments/data/patches"

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, required=True, choices=['unet', 'deeplabv3+', 'attention_unet', 'resunet'], help='Model architecture')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--run_name', type=str, default='fusion_run')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--aug_cp', action='store_true', help='Use Copy-Paste augmentation')
    parser.add_argument('--aug_cutmix', action='store_true', help='Use CutMix augmentation')
    args = parser.parse_args()
    
    seed_everything(42)
    device = torch.device(args.device)
    
    # Define Split
    all_indices = list(range(53))
    train_indices = all_indices[:42] # First 42 scenes for training
    val_indices = all_indices[42:]   # Last 11 scenes for validation (Test set in metrics calc)
    
    print(f"Train Scenes: {len(train_indices)}, Val Scenes: {len(val_indices)}")
    
    # Pre-build Copy-Paste bank efficiently
    prebuilt_bank = None
    if args.aug_cp:
        print("Building Copy-Paste component bank from training set...")
        temp_ds = FusionDataset(DATA_DIR, split_indices=train_indices)
        from utils.augmentation_functions import ComponentExtractor
        extractor = ComponentExtractor()
        prebuilt_bank = {c: [] for c in (1, 2, 3)}
        
        for i in tqdm(range(len(temp_ds)), desc="Extracting Components"):
            img = np.load(temp_ds.image_files[i])
            mask = np.load(temp_ds.mask_files[i])
            for cls in (1, 2, 3):
                extracted = extractor.extract(img, mask, cls)
                prebuilt_bank[cls].extend(extracted)
            # Full images are not stored, only the small crops in prebuilt_bank
    
    # Datasets
    train_ds = FusionDataset(
        DATA_DIR, 
        split_indices=train_indices,
        augment_cp=args.aug_cp,
        augment_cutmix=args.aug_cutmix,
        prebuilt_bank=prebuilt_bank
    )
    val_ds = FusionDataset(DATA_DIR, split_indices=val_indices)
    
    print(f"Train Patches: {len(train_ds)}")
    print(f"Val Patches: {len(val_ds)}")
    
    # Loaders
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4 if not args.aug_cp else 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Model
    print(f"Creating model: Fusion_{args.arch} with 227 input channels...")
    try:
        model = get_model(args.arch, in_channels=227, out_channels=NUM_CLASSES)
    except Exception as e:
        print(f"Error creating model: {e}")
        raise e

    model = model.to(device)
    
    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Training Loop
    best_val_loss = float('inf')
    best_iou = 0.0
    save_dir = os.path.join("/home/bs_thesis/Documents/BS_THESIS/PCBVision/Fusion_Experiments/runs", args.run_name)
    os.makedirs(save_dir, exist_ok=True)
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        all_preds = []
        all_masks = []
        
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                masks_np = masks.cpu().numpy()
                
                for i in range(len(preds)):
                    all_preds.append(preds[i])
                    all_masks.append(masks_np[i])
                
        val_loss /= len(val_loader)
        
        # Calculate Metrics
        # evaluate_segmentation(ground_truth_masks, predicted_masks, num_classes)
        # Returns: cm, tp, tn, fp, fn, prec, recall, f1, pa_class, pa_global, iou, dice, kappa
        results = evaluate_segmentation(all_masks, all_preds, NUM_CLASSES)
        _, _, _, _, _, precision, recall, f1, _, val_acc, val_iou, _, kappa = results
        
        # Use Mean IoU (val_iou is per class array)
        if isinstance(val_iou, np.ndarray):
            mIoU = np.nanmean(val_iou)
        else:
            mIoU = val_iou
            
        scheduler.step(val_loss)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, mIoU: {mIoU:.4f}, LR: {current_lr:.6f}")
        
        # Save Best
        if mIoU > best_iou: 
            best_iou = mIoU
            print(f"New Best mIoU: {best_iou:.4f}! Saving...")
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_miou.pth'))
            
        # Save Best Loss model
        torch.save(model.state_dict(), os.path.join(save_dir, 'last.pth'))
        if epoch == 0 or val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"New Best Val Loss! Saving...")
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_loss.pth'))
            
    print("Training Complete.")

if __name__ == "__main__":
    best_val_loss = float('inf')
    main()
