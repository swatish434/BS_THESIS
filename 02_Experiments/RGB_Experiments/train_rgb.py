"""
Module: RGB_Experiments/train_rgb.py
Purpose: Main training script for RGB-based PCB segmentation models.

This script implements the complete training pipeline for semantic segmentation on RGB PCB images:
    - Dataset loading and preprocessing
    - Copy-Paste augmentation with Scale Jittering (optional)
    - Model training (DeepLabv3+, UNet, ResUnet, LinkNet, Attention UNet)
    - Hybrid Loss (Focal + Dice) or vanilla CrossEntropy
    - Evaluation and feature visualization

Workflow:
    1. Load PCB dataset (53 images)
    2. Split: Train/Val/Test (64%/16%/20%)
    3. Build Copy-Paste component bank (optional)
    4. Create DataLoaders with on-the-fly augmentation
    5. Train model with validation monitoring
    6. Evaluate on test set with metrics
    7. Save best model + feature maps

Key Features:
    - Supports 5 architectures via --model flag
    - Copy-Paste + Scale Jittering augmentation (--augment flag)
    - Hybrid Loss for class imbalance (--loss hybrid)
    - Early stopping, LR scheduling, gradient clipping
    - Mixed precision training (--amp)
    - Reproducible training (--seed, --deterministic)

Usage:
    # Train DeepLabv3+ with Hybrid Loss + Copy-Paste (default)
    python3 RGB_Experiments/train_rgb.py

    # Train UNet baseline without augmentation
    python3 RGB_Experiments/train_rgb.py --model unet --augment False --loss crossentropy
    
    # Train with custom epochs and batch size
    python3 RGB_Experiments/train_rgb.py --epochs 50 --batch_size 8

Author: BS Thesis - PCB Vision Project
Date: January 2026
"""

import sys
import os

# Add parent directory to path to access utils and models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import albumentations as A
from sklearn.model_selection import train_test_split

from utils.repro import seed_everything, resolve_device
from utils.experiment import make_run_dir, save_config, save_json
from models.factory import get_model as build_model

# Import custom modules
from utils.dataset_functions import read_dataset, visualize, evaluate_segmentation
from utils.RGB_functions import calculate_mean_std, resize_segmentation_masks
from models.Unet import UNET

# ===================================================================
# Configuration: Default hyperparameters
# ===================================================================
DATASET_PATH = "/home/bs_thesis/Documents/BS_THESIS/DATASETS/PCBDataset/"
IMG_RES = 640  # Input image resolution (H=W=640)
NUM_CLASSES = 4  # Background, Component, IC, Connector
BATCH_SIZE = 4
LEARNING_RATE = 0.0001
NUM_EPOCHS = 100  # Default to 100 for full convergence
AUGMENTATIONS = 13  # Multiplier for virtual dataset expansion

# Default experiment configuration
DEFAULT_MODEL = 'deeplabv3+'  # Best-performing architecture
DEFAULT_LOSS = 'hybrid'  # Focal + Dice for class imbalance
DEFAULT_AUGMENT = True  # Enable Copy-Paste + Scale Jittering

def get_albumentations_transform():
    return A.Compose([
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=40, border_mode=cv2.BORDER_CONSTANT, p=0.4),
        A.Rotate(limit=-40, border_mode=cv2.BORDER_CONSTANT, p=0.4),
        A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.4),
        A.ColorJitter(brightness=0.5, contrast=0.4, saturation=0.4, hue = 0.2, p = .4),
        A.ChannelShuffle(p=0.4),
        A.Transpose(p=0.4),
        A.RandomSnow(p=0.4),
        A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=.4, border_mode=0),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0, p=1),
            A.RandomGamma(p=1),
        ], p=0.4),
        A.OneOf([
            A.Blur(blur_limit=3, p=1),
            A.MotionBlur(blur_limit=3, p=1),
        ], p=0.4),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0, contrast_limit=0.2, p=1),
            A.HueSaturationValue(p=1),
        ], p=0.4),
    ])

def augment_training(images, masks, n=13):
    """
    Augment training images and masks a given number of times.
    """
    print(f"Augmenting data {n} times...")
    transform = get_albumentations_transform()
    aug_images = []
    aug_masks = []

    for i in range(len(images)):
        # Add original
        aug_images.append(images[i])
        aug_masks.append(masks[i])

        for _ in range(n):
            transformed = transform(image=images[i], mask=masks[i])
            aug_images.append(transformed['image'])
            aug_masks.append(transformed['mask'])
    
    return aug_images, aug_masks


# Import new modules
from utils.augmentation_functions import MultimodalCutMix, SDLoRAAugmentation
from utils.loss_functions import HybridLoss
from models.DeepLabv3_plus import DeepLabv3_plus # Assuming this file is named DeepLabv3+.py we need to rename it or import carefully
# Rename check: files are DeepLabv3+.py. Python import doesn't like '+'.
# I need to rename DeepLabv3+.py to DeepLabv3_plus.py first? 
# Or use importlib? Better to rename.
from models.LinkNet import LinkNet
from models.Unet_Attention import AttU_Net
from models.ResUnet import ResUnet

class RGBDataset(Dataset):
    def __init__(self, images, masks, mean, std, albumentations_transform=None, transform_mask=None, num_classes=4, cutmix_aug=None, sdlora_aug=None, multiplier=1):
        self.images = images
        self.masks = masks
        self.mean = mean
        self.std = std
        self.albumentations_transform = albumentations_transform
        self.transform_mask = transform_mask # Keeps legacy logic for mask resizing if separate, but usually handled by albumentations
        self.num_classes = num_classes
        self.cutmix_aug = cutmix_aug
        self.sdlora_aug = sdlora_aug
        self.multiplier = multiplier

    def __len__(self):
        return len(self.images) * self.multiplier

    def __getitem__(self, index):
        # Virtual indexing for expansion
        real_index = index % len(self.images)
        
        image = self.images[real_index]
        mask = self.masks[real_index]

        # 1. Resize first to ensure uniform dimensions (required for CutMix)
        if mask.ndim == 3: mask = mask.squeeze()
        
        image_pil = Image.fromarray(image.astype(np.uint8))
        mask_pil = Image.fromarray(mask.astype(np.uint8))
        
        # Resize to target IMG_RES
        image_pil = image_pil.resize((IMG_RES, IMG_RES), Image.BILINEAR)
        mask_pil = mask_pil.resize((IMG_RES, IMG_RES), Image.NEAREST)
        
        image = np.array(image_pil)
        mask = np.array(mask_pil)

        # 2. SD-LoRA Augmentation (applies pre-generated refined patches)
        if self.sdlora_aug and np.random.rand() < 0.5:
            try:
                image, mask = self.sdlora_aug(image, mask)
            except Exception as e:
                print(f"SD-LoRA augmentation failed: {e}")

        # 3. CutMix Augmentation (on resized images)
        if self.cutmix_aug and np.random.rand() < 0.5:
             # Pick random second image and resize it too
             idx2 = np.random.randint(len(self.images))
             image2 = self.images[idx2]
             mask2 = self.masks[idx2]
             
             if mask2.ndim == 3: mask2 = mask2.squeeze()
             
             # Resize second image to same dimensions
             image2_pil = Image.fromarray(image2.astype(np.uint8))
             mask2_pil = Image.fromarray(mask2.astype(np.uint8))
             
             image2_pil = image2_pil.resize((IMG_RES, IMG_RES), Image.BILINEAR)
             mask2_pil = mask2_pil.resize((IMG_RES, IMG_RES), Image.NEAREST)
             
             image2 = np.array(image2_pil)
             mask2 = np.array(mask2_pil)
             
             try:
                 # cutmix(rgb1, hsi1, mask1, rgb2, hsi2, mask2)
                 image, _, mask, _, _ = self.cutmix_aug.cutmix(
                     image, None, mask,
                     image2, None, mask2
                 )
             except Exception as e:
                 print(f"CutMix failed: {e}")

        # 3. Geometric & Color Augmentation (Albumentations)
        if self.albumentations_transform:
            # Ensure types
            image = image.astype(np.uint8)
            mask = mask.astype(np.uint8)
            
            transformed = self.albumentations_transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']

        # 4. Standardization (Preprocessing)
        image_pil = Image.fromarray(image.astype(np.uint8))
        mask_pil = Image.fromarray(mask.astype(np.uint8))
        
        image_np = np.array(image_pil).astype(np.float32)
        mask_np = np.array(mask_pil) # Keep as int/long for labels

        # Normalize
        image_np = (image_np - self.mean) / self.std
        image_np = image_np.astype(np.float32)
        
        # To Channel First (C, H, W)
        image_tensor = torch.from_numpy(image_np.transpose(2, 0, 1))
        mask_tensor = torch.as_tensor(mask_np, dtype=torch.long)

        return image_tensor, mask_tensor

def get_model(model_name, num_classes):
    return build_model(model_name, in_channels=3, out_channels=num_classes, pretrained=True)

def main():
    global NUM_EPOCHS, BATCH_SIZE
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL, help='Model architecture: unet, deeplabv3+, linknet, attention_unet, resunet')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size')
    # Augmentation mode selection
    parser.add_argument('--augment-mode', type=str, default='cutmix', 
                        choices=['baseline', 'cutmix', 'sdlora'],
                        help='Augmentation strategy: baseline (none), cutmix, sdlora')
    parser.add_argument('--loss', type=str, default=DEFAULT_LOSS, help='Loss function: crossentropy, hybrid')
    parser.add_argument('--device', default='auto', help='cpu, cuda, or auto')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--grad-accum', type=int, default=1)
    parser.add_argument('--grad-clip', type=float, default=0.0)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--run-dir', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs'))
    parser.add_argument('--resume', default=None)
    args = parser.parse_args()
    
    # Update global config with args (or just use args directly in code)
    NUM_EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size

    seed_everything(args.seed, deterministic=args.deterministic)
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    # 1. Load Data
    print("Loading dataset...")
    _, _, _, RGB, _, RGB_general_masks, _ = read_dataset(DATASET_PATH)
    
    print(f"Loaded {len(RGB)} RGB images.")

    # Filter out None values
    RGB_filtered = []
    masks_filtered = []
    for img, mask in zip(RGB, RGB_general_masks):
        if img is not None and mask is not None:
            RGB_filtered.append(img)
            masks_filtered.append(mask)
    RGB = RGB_filtered
    RGB_general_masks = masks_filtered
    print(f"Valid RGB images: {len(RGB)}")

    # 2. Split Data
    images_train, images_test, masks_train, masks_test = train_test_split(RGB, RGB_general_masks, test_size=0.2, random_state=123)
    images_train, images_validation, masks_train, masks_validation = train_test_split(images_train, masks_train, test_size=0.2, random_state=123)

    print(f"Train: {len(images_train)}, Val: {len(images_validation)}, Test: {len(images_test)}")

    # 3. Calculate Mean/Std
    mean, std = calculate_mean_std(images_train)
    print(f"Mean: {mean}, Std: {std}")

    # Build Augmentation based on mode
    augment_mode = args.augment_mode
    print(f"Augmentation Mode: {augment_mode}")
    
    cutmix_aug = None
    sdlora_aug = None
    
    if augment_mode == 'cutmix':
        print("Initializing CutMix Augmentation...")
        cutmix_aug = MultimodalCutMix(beta=1.0)
    elif augment_mode == 'sdlora':
        print("Initializing SD-LoRA Augmentation...")
        sdlora_aug = SDLoRAAugmentation(
            aug_bank_path='data/aug_bank',
            minority_classes=(2, 3),  # Capacitor, Connector
            class_names={2: 'Capacitor', 3: 'Connector'},
            max_paste_per_class=2,
            paste_probability=0.5,
            patch_size=(128, 128)
        )
    else:
        print("Baseline Mode - No augmentation")

    # 4. Augment Training Data (Dynamic On-The-Fly)
    # Replaced static augment_training list with Dataset multiplier
    # training_images, training_masks = augment_training(images_train, masks_train, n=AUGMENTATIONS)
    # print(f"Total Training after standard augmentation: {len(training_images)}")
    
    # 5. Create Datasets and Loaders
    # Pass transform and multiplier
    
    # Train Dataset: Supports CopyPaste AND Albumentations (Geometric) AND Expansion
    train_dataset = RGBDataset(
        images_train, 
        masks_train, 
        mean, 
        std, 
        albumentations_transform=get_albumentations_transform(), 
        transform_mask=True, 
        num_classes=NUM_CLASSES, 
        cutmix_aug=cutmix_aug,
        sdlora_aug=sdlora_aug,
        multiplier=AUGMENTATIONS
    )
    print(f"Total Training Samples (Virtual Expansion): {len(train_dataset)}")

    # Valid/Test Dataset: No Augmentation (only resize/normalize via class logic when transform=None implies partial logic?
    # Wait, RGBDataset logic for transform=None vs albumentations_transform=None is slightly different now.
    # We need to ensure Valid/Test get resized/normalized correctly.
    # In my configured RGBDataset:
    # 3. Resize & Standardization happens ALWAYS at the end.
    # So valid/test just need None for albumentations and copy_paste.
    valid_dataset = RGBDataset(images_validation, masks_validation, mean, std, albumentations_transform=None, transform_mask=True, num_classes=NUM_CLASSES)
    test_dataset = RGBDataset(images_test, masks_test, mean, std, albumentations_transform=None, transform_mask=True, num_classes=NUM_CLASSES)

    pin_memory = device.type == 'cuda'
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=args.num_workers, pin_memory=pin_memory, persistent_workers=args.num_workers > 0)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory, persistent_workers=args.num_workers > 0)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory, persistent_workers=args.num_workers > 0)

    # 6. Model Setup
    print(f"Initializing Model: {args.model}")
    model = get_model(args.model, NUM_CLASSES)
    model.to(device)

    # Loss
    print(f"Using Loss Function: {args.loss}")
    if args.loss.lower() == 'hybrid':
        criterion = HybridLoss(focal_weight=0.5, dice_weight=0.5, ignore_index=255) # Hybrid Loss
    else:
        # Standard Cross Entropy
        # Original notebook used weights: [.1, .8, .85, .9]
        class_weights = torch.tensor([0.1, 0.8, 0.85, 0.9], dtype=torch.float).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='mean')
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    scaler = None
    if args.amp and device.type == 'cuda':
        scaler = torch.cuda.amp.GradScaler()

    # 7. Training
    # Create Results Directory
    # Create Results Directory
    # Keep legacy results dir for downstream scripts
    RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    suffix = ""
    if augment_mode == 'baseline':
        suffix += "_baseline"
    
    aug_suffix = f"_{augment_mode}" if augment_mode != 'baseline' else ""
    run_dir = make_run_dir(args.run_dir, run_name=f"rgb_{args.model}{suffix}{aug_suffix}")
    save_config(run_dir, args, extra={'mean': mean, 'std': std})
    metrics_path = os.path.join(run_dir, 'metrics.jsonl')

    best_path = os.path.join(run_dir, f'RGB_{args.model}{suffix}_best.pth')
    legacy_best_path = os.path.join(RESULTS_DIR, f'RGB_{args.model}{suffix}_best.pth')

    best_val_loss = float('inf')
    epochs_no_improve = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt.get('model_state', ckpt))
        if 'optimizer_state' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state'])
        if 'scheduler_state' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state'])
        if scaler is not None and 'scaler_state' in ckpt:
            scaler.load_state_dict(ckpt['scaler_state'])
        best_val_loss = float(ckpt.get('best_val_loss', best_val_loss))
        print(f"Resumed from {args.resume} (best_val_loss={best_val_loss})")

    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0

        optimizer.zero_grad(set_to_none=True)
        step = 0

        for images, masks in train_loader:
            step += 1

            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True) # (N, 1, H, W)
            masks = torch.squeeze(masks, dim=1) # (N, H, W)
            
            # One hot encoding for Custom Loss/Output matching if needed, 
            # BUT CrossEntropyLoss expects target as (N, H, W) LongTensor with class indices.
            # Notebook logic:
            # masks = torch.nn.functional.one_hot(masks, num_classes)
            # masks = masks.permute(0, 3, 1, 2)
            # masks = masks.type(torch.FloatTensor)
            # ... criterion(outputs, masks)
            # WARNING: CrossEntropyLoss normally expects indices. If 'masks' is one-hot float, it might fail or behave unexpectedly unless it's a specific loss implementation.
            # Checking notebook: criterion = nn.CrossEntropyLoss(...)
            # PyTorch CrossEntropyLoss inputs: (N, C, H, W) logits, and (N, H, W) Long indices.
            # OR (N, C, H, W) logits and (N, C, H, W) float probabilities (soft labels) - supported in newer PyTorch versions.
            # Given the notebook does one-hot, maybe strictly following it is safer?
            # "masks = torch.nn.functional.one_hot(masks, num_classes)" -> (N, H, W, C)
            # "permute(0, 3, 1, 2)" -> (N, C, H, W)
            # "type(torch.FloatTensor)"
            
            # I will follow notebook logic exactly to be safe.
            masks_onehot = torch.nn.functional.one_hot(masks, NUM_CLASSES)
            masks_onehot = masks_onehot.permute(0, 3, 1, 2).float().to(device)

            if scaler is not None:
                with torch.autocast(device_type=device.type, enabled=True):
                    outputs = model(images)
                    loss = criterion(outputs, masks_onehot) / max(1, args.grad_accum)
                scaler.scale(loss).backward()
            else:
                outputs = model(images)
                loss = criterion(outputs, masks_onehot) / max(1, args.grad_accum)
                loss.backward()

            train_loss += float(loss.item())

            if step % max(1, args.grad_accum) == 0:
                if args.grad_clip and args.grad_clip > 0:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            
        avg_train_loss = train_loss / max(1, len(train_loader))
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {avg_train_loss:.4f}")
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.inference_mode():
            for images, masks in valid_loader:
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                masks = torch.squeeze(masks, dim=1)
                
                masks_onehot = torch.nn.functional.one_hot(masks, NUM_CLASSES)
                masks_onehot = masks_onehot.permute(0, 3, 1, 2).float()
                masks_onehot = masks_onehot.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks_onehot)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(valid_loader)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Validation Loss: {avg_val_loss:.4f}")

        scheduler.step(avg_val_loss)
        lr = optimizer.param_groups[0]['lr']

        record = {
            'epoch': epoch + 1,
            'train_loss': float(avg_train_loss),
            'val_loss': float(avg_val_loss),
            'lr': float(lr),
        }
        with open(metrics_path, 'a', encoding='utf-8') as f:
            f.write(str(record) + "\n")

        ckpt = {
            'epoch': epoch + 1,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'scaler_state': scaler.state_dict() if scaler is not None else None,
            'best_val_loss': float(best_val_loss),
            'args': vars(args),
        }
        torch.save(ckpt, os.path.join(run_dir, 'last.pt'))
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_path)
            torch.save(model.state_dict(), legacy_best_path)
            print(f"Model Saved! Best Val Loss: {best_val_loss:.4f}")

            ckpt['best_val_loss'] = float(best_val_loss)
            torch.save(ckpt, os.path.join(run_dir, 'best_val_loss.pt'))
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if args.patience > 0 and epochs_no_improve >= args.patience:
                print(f"Early stopping (no val_loss improvement for {args.patience} epochs)")
                break

    print("Training Completed.")
    save_json(os.path.join(run_dir, 'final_summary.json'), {'best_val_loss': float(best_val_loss)})

    # 8. Evaluation
    print("Starting Testing...")

    # Load best model
    load_path = best_path if os.path.exists(best_path) else legacy_best_path
    model.load_state_dict(torch.load(load_path, map_location=device))
    model.eval()

    # Feature Visualization Hook
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    # Register hook based on model type
    # For UNet, we usually want to see bottleneck or encoder outputs
    if args.model.lower() == 'unet':
        # UNet usually has 'bottleneck' or 'd4' (depending on implementation in models/Unet.py)
        # Let's inspect models/Unet.py structure or try to attach to a known layer.
        # Assuming typical structure, let's target the last encoder layer or bottleneck
        # If unknown, we can try to attach to the module named 'bottleneck' if it exists, or just print(model) to debug.
        # For now, let's try to hook into the 'bottleneck' or similar. 
        # Inspecting Unet.py (from memory/previous views): it has self.bottleneck
        try:
             model.bottleneck.register_forward_hook(get_activation('bottleneck'))
        except AttributeError:
             print("Warning: Could not register hook for 'bottleneck'. Skipping feature viz.")
    elif args.model.lower() == 'deeplabv3+':
        # ResNet features?
        try:
             model.resnet_features.layer4.register_forward_hook(get_activation('backbone_out'))
        except AttributeError:
             pass

    print("Evaluating on Test Set with Feature Visualization...")
    # Note: evaluate_segmentation in dataset_functions might not support 'features' arg.
    # We should perform visualization here in the script for the first test image.
    
    # Custom Viz for first batch
    for i, (images, masks) in enumerate(test_loader):
        if i > 0: break # Only 1 sample
        
        images = images.to(device)
        masks = masks.to(device)
        
        outputs = model(images)
        
        # Save Features if captured
        if activation:
            import matplotlib.pyplot as plt
            for name, act in activation.items():
                # act is (N, C, H, W)
                # Visualize first 16 channels
                num_channels = min(16, act.shape[1])
                fig, axes = plt.subplots(4, 4, figsize=(12, 12))
                for c in range(num_channels):
                    ax = axes[c//4, c%4]
                    ax.imshow(act[0, c].cpu().numpy(), cmap='viridis')
                    ax.axis('off')
                plt.suptitle(f"Feature Map: {name}")
                save_path = os.path.join(RESULTS_DIR, f"feature_map_{name}.png")
                plt.savefig(save_path)
                print(f"Saved {save_path}")
        
    print("Evaluating metrics...")
    
    predicted_masks = []
    
    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            # masks not needed for inference loop except size
            
            output = model(images)
            output = torch.squeeze(output, dim=0)
            output = torch.nn.functional.softmax(output, dim=0)
            output = torch.argmax(output, dim=0)
            
            predicted_masks.append(output.cpu().numpy())
    
    # Resize and Evaluate
    predicted_masks_resized = []
    for i, m in enumerate(predicted_masks):
        # masks_test is list of numpy arrays from original split
        original_shape = masks_test[i].shape
        # Use simple cv2 resize nearest as in PCA_functions (or import it)
        # Note: resize_segmentation_masks in PCA_functions handles class values (0,1,2,3) carefully using nearest.
        resized = resize_segmentation_masks(m, original_shape)
        predicted_masks_resized.append(resized)
        
    print("Evaluating metrics...")
    results = evaluate_segmentation(masks_test, predicted_masks_resized, NUM_CLASSES)
    # unpack results
    confusion_matrix_sum, true_positive_sum, true_negative_sum, false_positive_sum, \
    false_negative_sum, precision, recall, f1_score, pixel_accuracy_per_class, \
    pixel_accuracy, iou, dice_coefficient, kappa = results

    print("\n--- Evaluation Results ---")
    print(f"Pixel Accuracy: {pixel_accuracy:.4f}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1_score}")
    print(f"IoU: {iou}")
    print(f"Kappa: {kappa}")

if __name__ == "__main__":
    main()
