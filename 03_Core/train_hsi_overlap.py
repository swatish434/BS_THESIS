
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import spectral.io.envi as envi
from tqdm import tqdm
from models.Unet import UNET
from utils.augmentation_functions import MultimodalCutMix, CopyPasteAugmentation
from utils.loss_functions import HybridLoss
import random
import sys
import gc

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class HSIPatchesDataset(Dataset):
    def __init__(self, data_dir, split='Train', augmentation=None, augment_type=None):
        self.data_dir = data_dir
        self.split = split
        self.augmentation = augmentation
        self.augment_type = augment_type
        
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

        # Normalization (Per-sample MinMax)
        # Ensure data is 0-1 range consistently
        # Handle potential outliers or zeros
        h_min = hsi_cube.min()
        h_max = hsi_cube.max()
        if h_max > h_min:
            hsi_cube = (hsi_cube - h_min) / (h_max - h_min)
        else:
            hsi_cube = np.zeros_like(hsi_cube)

        # Augmentation
        if self.augmentation and self.split == 'Train':
            if self.augment_type == 'cutmix' and np.random.rand() < 0.5:
                 # Load another random sample
                 idx2 = np.random.randint(len(self))
                 hsi_cube2, mask2 = self.load_sample(idx2)
                 
                 if hsi_cube2 is not None:
                     try:
                         # cutmix(rgb1, hsi1, mask1, rgb2, hsi2, mask2)
                         # pass None for RGB
                         _, hsi_cube, mask, _, _ = self.augmentation.cutmix(
                             None, hsi_cube, mask, 
                             None, hsi_cube2, mask2
                         )
                     except Exception as e:
                         pass # print(f"CutMix failed: {e}")
            elif self.augment_type == 'copypaste':
                # CopyPaste handles its own probability internally or we call it
                # It expects (H,W,C) and (H,W)
                try:
                    hsi_cube, mask = self.augmentation(hsi_cube, mask)
                except Exception as e:
                    pass # print(f"CopyPaste failed: {e}")

        # Preprocessing
        hsi_cube = hsi_cube.transpose(2, 0, 1) 
        
        hsi_tensor = torch.from_numpy(hsi_cube).float()
        mask_tensor = torch.from_numpy(mask).long() 
        
        return hsi_tensor, mask_tensor



def train_one_epoch(loader, model, optimizer, loss_fn):
    model.train()
    loop = tqdm(loader)
    total_loss = 0
    
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        predictions = model(data)
        loss = loss_fn(predictions, targets)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())
        
    return total_loss / len(loader)

def evaluate(loader, model, loss_fn):
    model.eval()
    loop = tqdm(loader)
    total_loss = 0
    num_correct = 0
    num_pixels = 0
    
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(loop):
            data = data.to(device)
            targets = targets.to(device)
            
            predictions = model(data)
            loss = loss_fn(predictions, targets)
            total_loss += loss.item()
            
            preds = torch.argmax(torch.softmax(predictions, dim=1), dim=1)
            num_correct += (preds == targets).sum()
            num_pixels += torch.numel(preds)
            
    accuracy = num_correct / num_pixels * 100
    print(f"Validation Accuracy: {accuracy:.2f}")
    accuracy = num_correct / num_pixels * 100
    print(f"Validation Accuracy: {accuracy:.2f}")
    return total_loss / len(loader), accuracy

from models.Unet import UNET
from models.DeepLabv3_plus import DeepLabv3_plus
from models.ResUnet import ResUnet
from models.Unet_Attention import AttU_Net

def build_copypaste_bank(dataset):
    print("Building CopyPaste Bank from dataset...")
    # Initialize Augmentor
    augmentor = CopyPasteAugmentation(minority_classes=[1, 2, 3])
    
    # Iterate through dataset to extract components
    # We use the dataset's load_sample method
    count = 0
    # Limit to first N samples to save time/memory if needed, but for 53 PCBs it's fine
    # Actually this is patch dataset? "Patches_256_Overlap_Data"
    # Loading thousands of patches might be slow.
    # Let's try loading a subset or robustly.
    
    print(f"Scanning {len(dataset)} samples for components...")
    for i in tqdm(range(len(dataset))):
        hsi, mask = dataset.load_sample(i)
        if hsi is None: continue
        
        # Norm 0-1 for extraction consistency
        h_min, h_max = hsi.min(), hsi.max()
        if h_max > h_min:
             hsi = (hsi - h_min) / (h_max - h_min)
             
        # Extract
        for cls in augmentor.minority_classes:
            extracted = augmentor.extractor.extract(hsi, mask, cls)
            augmentor.component_bank[cls].extend(extracted)
            count += len(extracted)
            
    print(f"Bank built with {count} components.")
    return augmentor

def get_hsi_model(model_name, in_channels, out_channels):
    model_name = model_name.lower()
    if model_name == 'unet':
        return UNET(in_channels=in_channels, out_channels=out_channels)
    elif model_name == 'deeplabv3+' or model_name == 'deeplabv3_plus':
        return DeepLabv3_plus(nInputChannels=in_channels, n_classes=out_channels)
    elif model_name == 'resunet':
        return ResUnet(channel=in_channels, out_channel=out_channels)
    elif model_name in ['attunet', 'attentionunet', 'attention_unet']:
        return AttU_Net(img_ch=in_channels, output_ch=out_channels)
    else:
        raise ValueError(f"Model {model_name} not supported for HSI or unknown.")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--dry-run', action='store_true', help='Run 1 epoch validation')
    parser.add_argument('--model', type=str, default='deeplabv3+', help='Model name')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--augment', type=str, default='cutmix', choices=['none', 'cutmix', 'copypaste'], help='Augmentation type')
    args = parser.parse_args()

    # Parameters
    LEARNING_RATE = 1e-4
    # Reduced batch size for 256x256 (4x pixels vs 128x128)
    # HSI is heavy (214 channels)
    BATCH_SIZE = 4 
    NUM_EPOCHS = 1 if args.dry_run else args.epochs
    NUM_WORKERS = 2 
    PIN_MEMORY = True
    DATA_DIR = "/home/bs_thesis/Documents/BS_THESIS/PCBVision/Patches_256_Overlap_Data/" 
    IN_CHANNELS = 214 
    OUT_CHANNELS = 4 
    
    # Init Augmentation
    augmentation = None
    if args.augment == 'cutmix':
        print("Initializing CutMix Augmentation...")
        augmentation = MultimodalCutMix(beta=1.0)
    elif args.augment == 'copypaste':
        print("Initializing CopyPaste Augmentation...")
        # We need a temporary dataset to build the bank
        temp_ds = HSIPatchesDataset(DATA_DIR, split='Train')
        augmentation = build_copypaste_bank(temp_ds)
        del temp_ds
        gc.collect()

    # Init Datasets
    train_ds = HSIPatchesDataset(DATA_DIR, split='Train', augmentation=augmentation, augment_type=args.augment)
    val_ds = HSIPatchesDataset(DATA_DIR, split='Val') 
     
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, shuffle=False)
    
    print(f"Initializing Model: {args.model}")
    model = get_hsi_model(args.model, IN_CHANNELS, OUT_CHANNELS).to(device)
    
    save_name = f"hsi_overlap_{args.model}_{args.augment}_best.pth" if not args.dry_run else f"dry_run_{args.model}_{args.augment}.pth"
    save_path = os.path.join("Results", save_name)
    
    if args.resume:
        if os.path.exists(save_path):
            print(f"Resuming from checkpoint: {save_path}")
            model.load_state_dict(torch.load(save_path, map_location=device))
        else:
            print(f"Checkpoint not found at {save_path}, starting from scratch.")

    loss_fn = HybridLoss(focal_weight=0.5, dice_weight=0.5, ignore_index=255)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # save_name definition moved up and fixed

    best_loss = float('inf')
    
    print(f"Starting Training on 256x256 Overlap Data for {NUM_EPOCHS} epochs...")
    
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch: {epoch+1}/{NUM_EPOCHS}")
        train_loss = train_one_epoch(train_loader, model, optimizer, loss_fn)
        val_loss, val_acc = evaluate(val_loader, model, loss_fn)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        if val_loss < best_loss:
            best_loss = val_loss
            os.makedirs("Results", exist_ok=True)
            torch.save(model.state_dict(), os.path.join("Results", save_name))
            print("Saved Best Model")
            
    print("Training Complete.")

if __name__ == "__main__":
    main()
