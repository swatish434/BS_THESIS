#!/usr/bin/env python3
"""
Train HSI models on PCA-reduced data with augmentation support.

Usage:
    python train_hsi_pca_augmented.py --n_components 10 --model unet --augment cutmix
    python train_hsi_pca_augmented.py --n_components 20 --model deeplabv3+ --augment copypaste
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import spectral.io.envi as envi
from tqdm import tqdm
import argparse
import json

# Import models
from models.Unet import UNET
from models.DeepLabv3_plus import DeepLabv3_plus
from models.ResUnet import ResUnet
from models.Unet_Attention import AttU_Net

# Import augmentation and loss
from utils.augmentation_functions import MultimodalCutMix, CopyPasteAugmentation
from utils.loss_functions import HybridLoss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class HSIPCADataset(Dataset):
    """Dataset for PCA-reduced HSI data with augmentation support."""
    
    def __init__(self, data_dir, split='Train', n_components=10, 
                 cutmix_aug=None, copypaste_aug=None, target_size=256):
        self.data_dir = data_dir
        self.split = split
        self.n_components = n_components
        self.cutmix_aug = cutmix_aug
        self.copypaste_aug = copypaste_aug
        self.target_size = target_size
        
        # Find all samples for this split
        all_files = os.listdir(data_dir)
        self.headers = sorted([f for f in all_files if f.startswith(split) and f.endswith('.hdr')],
                              key=lambda x: int(x.split('_')[1].split('.')[0]))
        
        print(f"Found {len(self.headers)} {split} samples")
        
        # Preload all data for augmentation bank
        if copypaste_aug is not None and split == 'Train':
            self._build_copypaste_bank()
    
    def _build_copypaste_bank(self):
        """Build CopyPaste component bank from all training samples."""
        print("Building CopyPaste bank...")
        images = []
        masks = []
        for hdr in self.headers:
            hsi, mask = self._load_sample(hdr)
            if hsi is not None:
                images.append(hsi)
                masks.append(mask)
        self.copypaste_aug.build_bank(images, masks)
    
    def _load_sample(self, header_filename):
        """Load a single PCA sample."""
        base = header_filename.replace('.hdr', '')
        hdr_path = os.path.join(self.data_dir, header_filename)
        data_path = os.path.join(self.data_dir, base)
        mask_path = os.path.join(self.data_dir, base + '.npy')
        
        try:
            hsi_obj = envi.open(hdr_path, data_path)
            hsi = np.array(hsi_obj.load(), dtype=np.float32)
            mask = np.load(mask_path)
            if mask.ndim == 3:
                mask = mask.squeeze()
            return hsi, mask
        except Exception as e:
            print(f"Error loading {header_filename}: {e}")
            return None, None
    
    def __len__(self):
        return len(self.headers)
    
    def __getitem__(self, idx):
        header = self.headers[idx]
        hsi, mask = self._load_sample(header)
        
        if hsi is None:
            return torch.zeros(self.n_components, self.target_size, self.target_size), \
                   torch.zeros(self.target_size, self.target_size, dtype=torch.long)
        
        # Resize if needed
        import cv2
        if hsi.shape[0] != self.target_size or hsi.shape[1] != self.target_size:
            hsi = cv2.resize(hsi, (self.target_size, self.target_size))
            mask = cv2.resize(mask.astype(np.uint8), (self.target_size, self.target_size), 
                             interpolation=cv2.INTER_NEAREST)
        
        # CutMix Augmentation
        if self.cutmix_aug and self.split == 'Train' and np.random.rand() < 0.5:
            idx2 = np.random.randint(len(self))
            hsi2, mask2 = self._load_sample(self.headers[idx2])
            if hsi2 is not None:
                if hsi2.shape[0] != self.target_size:
                    hsi2 = cv2.resize(hsi2, (self.target_size, self.target_size))
                    mask2 = cv2.resize(mask2.astype(np.uint8), (self.target_size, self.target_size),
                                      interpolation=cv2.INTER_NEAREST)
                try:
                    _, hsi, mask, _, _ = self.cutmix_aug.cutmix(None, hsi, mask, None, hsi2, mask2)
                except:
                    pass
        
        # CopyPaste Augmentation
        if self.copypaste_aug and self.split == 'Train' and np.random.rand() < 0.5:
            try:
                hsi, mask = self.copypaste_aug.apply(hsi, mask)
            except:
                pass
        
        # Convert to tensors
        hsi = hsi.transpose(2, 0, 1)  # (H, W, C) -> (C, H, W)
        hsi_tensor = torch.from_numpy(hsi).float()
        mask_tensor = torch.from_numpy(mask.astype(np.int64)).long()
        
        return hsi_tensor, mask_tensor


def get_model(model_name, in_channels, out_channels):
    """Get model by name."""
    model_name = model_name.lower()
    if model_name == 'unet':
        return UNET(in_channels=in_channels, out_channels=out_channels)
    elif model_name in ['deeplabv3+', 'deeplabv3_plus']:
        return DeepLabv3_plus(nInputChannels=in_channels, n_classes=out_channels)
    elif model_name == 'resunet':
        return ResUnet(channel=in_channels, out_channel=out_channels)
    elif model_name in ['attunet', 'attention_unet']:
        return AttU_Net(img_ch=in_channels, output_ch=out_channels)
    else:
        raise ValueError(f"Unknown model: {model_name}")


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
    parser = argparse.ArgumentParser(description='Train HSI PCA models with augmentation')
    parser.add_argument('--n_components', type=int, default=10, help='Number of PCA components')
    parser.add_argument('--model', type=str, default='unet', 
                        choices=['unet', 'deeplabv3+', 'resunet', 'attunet'])
    parser.add_argument('--augment', type=str, default='none',
                        choices=['none', 'cutmix', 'copypaste'],
                        help='Augmentation method')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=15)
    args = parser.parse_args()
    
    # Data directory
    data_dir = f"/home/bs_thesis/Documents/BS_THESIS/PCBVision/PCA_Data_{args.n_components}/"
    
    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found: {data_dir}")
        print(f"Run: python generate_pca_data_configurable.py --n_components {args.n_components}")
        return
    
    print(f"=== Training HSI PCA-{args.n_components} {args.model.upper()} ===")
    print(f"Augmentation: {args.augment}")
    print(f"Device: {device}")
    
    # Setup augmentation
    cutmix_aug = None
    copypaste_aug = None
    
    if args.augment == 'cutmix':
        cutmix_aug = MultimodalCutMix(beta=1.0)
        print("CutMix augmentation enabled")
    elif args.augment == 'copypaste':
        copypaste_aug = CopyPasteAugmentation(minority_classes=(1, 2, 3))
        print("CopyPaste augmentation enabled")
    
    # Create datasets
    train_ds = HSIPCADataset(data_dir, 'Train', args.n_components, cutmix_aug, copypaste_aug)
    val_ds = HSIPCADataset(data_dir, 'Val', args.n_components)
    test_ds = HSIPCADataset(data_dir, 'Test', args.n_components)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    
    # Create model
    model = get_model(args.model, args.n_components, 4).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    loss_fn = HybridLoss(focal_weight=0.5, dice_weight=0.5)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training
    save_name = f"hsi_pca{args.n_components}_{args.model}_{args.augment}_best.pth"
    save_dir = "Results"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, save_name)
    
    best_loss = float('inf')
    patience_counter = 0
    
    print(f"\nStarting training for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        train_loss = train_epoch(train_loader, model, optimizer, loss_fn)
        val_loss, val_acc = evaluate(val_loader, model, loss_fn)
        
        print(f"Epoch {epoch+1}/{args.epochs}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%")
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"  Saved best model: {save_name}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping after {epoch+1} epochs")
                break
    
    # Final evaluation
    print("\n=== Final Evaluation ===")
    model.load_state_dict(torch.load(save_path, map_location=device))
    test_loss, test_acc = evaluate(test_loader, model, loss_fn)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")
    
    print(f"\nTraining complete! Model saved to: {save_path}")


if __name__ == "__main__":
    main()
