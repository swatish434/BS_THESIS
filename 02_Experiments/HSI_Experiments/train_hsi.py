
import sys
import os
import argparse

# Add parent directory to path to access utils and models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import spectral.io.envi as envi
from tqdm import tqdm
from models.Unet import UNET
from utils.augmentation_functions import CopyPasteAugmentation
import random
import gc

from PCBVision.utils.repro import seed_everything, resolve_device
from PCBVision.utils.experiment import make_run_dir, save_config, save_json
from PCBVision.models.factory import get_model

device = torch.device('cpu')

class HSIPatchesDataset(Dataset):
    def __init__(self, data_dir, split='Train', copy_paste_aug=None):
        self.data_dir = data_dir
        self.split = split
        self.copy_paste_aug = copy_paste_aug
        
        all_files = os.listdir(data_dir)
        self.headers = [f for f in all_files if f.startswith(split) and f.endswith('.hdr')]
        self.headers.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
        
        print(f"Found {len(self.headers)} samples for split {split}")

    def __len__(self):
        return len(self.headers)

    def load_sample(self, idx):
        header_filename = self.headers[idx]
        base_filename = header_filename.replace('.hdr', '')
        
        header_path = os.path.join(self.data_dir, header_filename)
        data_path = os.path.join(self.data_dir, base_filename)
        mask_path = os.path.join(self.data_dir, base_filename + '.npy')
        
        try:
            hsi_obj = envi.open(header_path, data_path)
            hsi_cube = hsi_obj.load() 
            hsi_cube = np.array(hsi_cube, dtype=np.float32)
            
            mask = np.load(mask_path)
            if len(mask.shape) == 3: mask = mask.squeeze()
            
            return hsi_cube, mask
        except Exception as e:
            print(f"Error loading {header_filename}: {e}")
            return None, None

    def __getitem__(self, idx):
        hsi_cube, mask = self.load_sample(idx)
        if hsi_cube is None:
            return torch.zeros(214, 256, 256), torch.zeros(256, 256) # 256x256

        # Copy-Paste Augmentation
        if self.copy_paste_aug and np.random.rand() < 0.5:
             try:
                 hsi_cube, mask = self.copy_paste_aug(hsi_cube, mask)
             except Exception as e:
                 print(f"Augmentation failed: {e}")

        # Preprocessing
        hsi_cube = hsi_cube.transpose(2, 0, 1) 
        
        hsi_tensor = torch.from_numpy(hsi_cube).float()
        mask_tensor = torch.from_numpy(mask).long() 
        
        return hsi_tensor, mask_tensor

def build_bank_subset(dataset, cp_aug, num_samples=200):
    total = len(dataset)
    # If dataset is smaller than request, use all
    num_samples = min(total, num_samples)
    
    indices = np.random.choice(total, num_samples, replace=False)
    
    print(f"Building Copy-Paste Bank from {num_samples} random samples...")
    
    images = []
    masks = []
    
    for idx in tqdm(indices):
        img, mask = dataset.load_sample(idx)
        if img is not None:
            images.append(img)
            masks.append(mask)
            
    cp_aug.build_bank(images, masks)
    
    del images
    del masks
    gc.collect()

def train_one_epoch(
    loader,
    model,
    optimizer,
    loss_fn,
    *,
    scaler=None,
    grad_accum=1,
    grad_clip=0.0,
):
    model.train()
    loop = tqdm(loader)
    total_loss = 0.0

    optimizer.zero_grad(set_to_none=True)
    for step, (data, targets) in enumerate(loop, start=1):
        data = data.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if scaler is not None:
            with torch.autocast(device_type=device.type, enabled=True):
                predictions = model(data)
                loss = loss_fn(predictions, targets) / grad_accum
            scaler.scale(loss).backward()
        else:
            predictions = model(data)
            loss = loss_fn(predictions, targets) / grad_accum
            loss.backward()

        total_loss += float(loss.item())
        if step % grad_accum == 0:
            if grad_clip and grad_clip > 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        loop.set_postfix(loss=float(loss.item() * grad_accum))

    return total_loss / max(1, len(loader))

def evaluate(loader, model, loss_fn):
    model.eval()
    loop = tqdm(loader)
    total_loss = 0
    num_correct = 0
    num_pixels = 0
    
    with torch.inference_mode():
        for batch_idx, (data, targets) in enumerate(loop):
            data = data.to(device)
            targets = targets.to(device)
            
            predictions = model(data)
            loss = loss_fn(predictions, targets)
            total_loss += loss.item()
            
            preds = torch.argmax(torch.softmax(predictions, dim=1), dim=1)
            num_correct += (preds == targets).sum()
            num_pixels += torch.numel(preds)
            
    accuracy = (num_correct / num_pixels * 100) if num_pixels else 0.0
    print(f"Validation Accuracy: {accuracy:.2f}")
    return total_loss / max(1, len(loader)), float(accuracy)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--dry-run', action='store_true', help='Run 1 epoch validation')
    parser.add_argument('--model', type=str, default='unet', help='Model architecture [unet, deeplabv3+, resunet]')
    parser.add_argument('--device', default='auto', help='cpu, cuda, or auto')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--deterministic', action='store_true', help='Deterministic CUDNN (slower)')
    parser.add_argument('--data-dir', default="/home/bs_thesis/Documents/BS_THESIS/PCBVision/Patches_256_Overlap_Data/")
    parser.add_argument('--in-channels', type=int, default=214)
    parser.add_argument('--out-channels', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--amp', action='store_true', help='Enable mixed precision (CUDA only)')
    parser.add_argument('--grad-accum', type=int, default=1)
    parser.add_argument('--grad-clip', type=float, default=0.0)
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--run-dir', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs'))
    parser.add_argument('--resume', default=None, help='Path to checkpoint to resume')
    args = parser.parse_args()

    global device
    seed_everything(args.seed, deterministic=args.deterministic)
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    run_dir = make_run_dir(args.run_dir, run_name=f"hsi_{args.model}" + ("_dry" if args.dry_run else ""))
    save_config(run_dir, args)

    NUM_EPOCHS = 1 if args.dry_run else args.epochs
    PIN_MEMORY = device.type == 'cuda'
    
    print("Initializing Copy-Paste Augmentation...")
    cp_aug = CopyPasteAugmentation(minority_classes=(1, 2, 3))
    
    # Init Datasets
    train_ds = HSIPatchesDataset(args.data_dir, split='Train', copy_paste_aug=cp_aug)
    val_ds = HSIPatchesDataset(args.data_dir, split='Val')
    
    # Build Bank (Use 150 samples here as we have more data now)
    # Be careful with memory
    build_bank_subset(train_ds, cp_aug, num_samples=150) 
    
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=PIN_MEMORY,
        shuffle=False,
        persistent_workers=args.num_workers > 0,
    )
    

    
    print(f"Initializing Model: {args.model}")
    model = get_model(args.model, args.in_channels, args.out_channels, pretrained=False).to(device)
    
    # Focal Loss with Class Weights to combat severe imbalance
    # Class 0 (Background): weight 1.0 (no boost)
    # Class 1 (Component): weight 5.0 (moderate boost)
    # Class 2 (IC): weight 10.0 (heavy boost - very rare)
    # Class 3 (Connector): weight 10.0 (heavy boost - very rare)
    class_weights = torch.tensor([1.0, 5.0, 10.0, 10.0], dtype=torch.float32).to(device)
    loss_fn = FocalLoss(alpha=class_weights, gamma=2, reduction='mean')
    print(f"Using Focal Loss with gamma=2 and class weights: {class_weights.cpu().numpy()}")
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    scaler = None
    if args.amp and device.type == 'cuda':
        scaler = torch.cuda.amp.GradScaler()

    best_loss = float('inf')
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
        best_loss = float(ckpt.get('best_loss', best_loss))
        print(f"Resumed from {args.resume} (best_loss={best_loss})")
    
    metrics_path = os.path.join(run_dir, 'metrics.jsonl')
    print(f"Starting Training for {NUM_EPOCHS} epochs...")
    
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch: {epoch+1}/{NUM_EPOCHS}")
        train_loss = train_one_epoch(
            train_loader,
            model,
            optimizer,
            loss_fn,
            scaler=scaler,
            grad_accum=max(1, args.grad_accum),
            grad_clip=args.grad_clip,
        )
        val_loss, val_acc = evaluate(val_loader, model, loss_fn)
        scheduler.step(val_loss)
        
        lr = optimizer.param_groups[0]['lr']
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | LR: {lr:.2e}")

        record = {
            'epoch': epoch + 1,
            'train_loss': float(train_loss),
            'val_loss': float(val_loss),
            'val_acc': float(val_acc),
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
            'best_loss': float(best_loss),
            'args': vars(args),
        }
        torch.save(ckpt, os.path.join(run_dir, 'last.pt'))

        if val_loss < best_loss:
            best_loss = float(val_loss)
            epochs_no_improve = 0
            ckpt['best_loss'] = best_loss
            torch.save(ckpt, os.path.join(run_dir, 'best_loss.pt'))
            torch.save(model.state_dict(), os.path.join(run_dir, 'best_loss.pth'))
            print("Saved best-loss checkpoint")
        else:
            epochs_no_improve += 1
            if args.patience > 0 and epochs_no_improve >= args.patience:
                print(f"Early stopping (no val_loss improvement for {args.patience} epochs)")
                break
            
    print("Training Complete.")
    save_json(os.path.join(run_dir, 'final_summary.json'), {'best_loss': best_loss})

if __name__ == "__main__":
    main()
