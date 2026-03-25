#!/usr/bin/env python3
"""
Simple runner for Exp 05 / 06 using the original HybridSSRNSeg from 03_Core.
Mirrors the logic used for Exp 04 (HSI None) which completed successfully.
"""
import os, sys, argparse, json
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
CORE_DIR     = os.path.join(PROJECT_ROOT, '03_Core')
sys.path.insert(0, CORE_DIR)

from models.hsrn_segmentation import HybridSSRNSeg
from utils.augmentation_functions import CopyPasteAugmentation, MultimodalCutMix
from utils.loss_functions import HybridLoss

import spectral.io.envi as envi

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class HSIDataset(Dataset):
    def __init__(self, data_dir, split='Train', cutmix_aug=None, copypaste_aug=None):
        self.data_dir      = data_dir
        self.split         = split
        self.cutmix_aug    = cutmix_aug
        self.copypaste_aug = copypaste_aug
        all_files = os.listdir(data_dir)
        self.headers = sorted(
            [f for f in all_files if f.startswith(split+'_') and f.endswith('.hdr')],
            key=lambda x: int(x.replace(split+'_','').replace('.hdr',''))
        )
        print(f"[HSI] {split}: {len(self.headers)} samples")
        if copypaste_aug is not None and split == 'Train':
            self._build_copypaste_bank()

    def _load_sample(self, hdr):
        base = hdr.replace('.hdr','')
        hsi_obj = envi.open(os.path.join(self.data_dir, hdr), os.path.join(self.data_dir, base))
        hsi = np.array(hsi_obj.load(), dtype=np.float32)   # (H, W, 214)
        mask = np.load(os.path.join(self.data_dir, base+'.npy'))
        if mask.ndim==3 and mask.shape[-1]==1: mask = mask[:,:,0]
        elif mask.ndim==3 and mask.shape[0]==1: mask = mask[0]
        h_min, h_max = hsi.min(), hsi.max()
        hsi = (hsi - h_min) / (h_max - h_min + 1e-8)
        return hsi, mask

    def _build_copypaste_bank(self):
        print("[HSI] Building CopyPaste bank…")
        imgs, masks = [], []
        for hdr in tqdm(self.headers, desc="Building bank"):
            try:
                h, m = self._load_sample(hdr)
                imgs.append(h); masks.append(m)
            except Exception: pass
        self.copypaste_aug.build_bank(imgs, masks)

    def __len__(self): return len(self.headers)

    def __getitem__(self, idx):
        try:
            hsi, mask = self._load_sample(self.headers[idx])
        except Exception:
            return (torch.zeros(214, 256, 256), torch.zeros(256, 256, dtype=torch.long))

        if self.cutmix_aug and self.split=='Train' and np.random.rand()<0.5:
            idx2 = np.random.randint(len(self))
            try:
                h2, m2 = self._load_sample(self.headers[idx2])
                _, hsi, mask, _, _ = self.cutmix_aug.cutmix(None,hsi,mask,None,h2,m2)
            except Exception: pass

        if self.copypaste_aug and self.split=='Train' and np.random.rand()<0.5:
            try:
                hsi, mask = self.copypaste_aug(hsi, mask)
            except Exception: pass

        return torch.from_numpy(hsi.transpose(2,0,1)).float(), torch.from_numpy(mask.astype(np.int64)).long()


def compute_metrics(preds, targets, n=4):
    p = preds.cpu().numpy(); t = targets.cpu().numpy()
    pa = (p==t).sum() / t.size
    iou = np.zeros(n); cnt = np.zeros(n)
    for c in range(n):
        inter = ((p==c)&(t==c)).sum(); union = ((p==c)|(t==c)).sum()
        if union>0: iou[c]+=inter/union; cnt[c]+=1
    return pa, np.mean([iou[c]/max(cnt[c],1) for c in range(n)])

def train_epoch(loader, model, opt, loss_fn):
    model.train(); tot=0
    for data,tgt in tqdm(loader, desc='  Train', leave=False):
        data,tgt = data.to(device), tgt.to(device)
        opt.zero_grad(); loss = loss_fn(model(data), tgt)
        loss.backward(); opt.step(); tot+=loss.item()
    return tot/len(loader)

def evaluate(loader, model, loss_fn, n=4):
    model.eval(); tot=0; preds=[]; tgts=[]
    with torch.no_grad():
        for data,tgt in tqdm(loader, desc='  Eval ', leave=False):
            data,tgt = data.to(device), tgt.to(device)
            out = model(data); tot+=loss_fn(out,tgt).item()
            preds.append(out.argmax(1)); tgts.append(tgt)
    pa,miou = compute_metrics(torch.cat(preds), torch.cat(tgts), n)
    return tot/len(loader), pa*100, miou*100

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--augment',    type=str,   default='none', choices=['none','copypaste','cutmix'])
    parser.add_argument('--epochs',     type=int,   default=100)
    parser.add_argument('--batch_size', type=int,   default=4)
    parser.add_argument('--lr',         type=float, default=1e-4)
    parser.add_argument('--patience',   type=int,   default=15)
    parser.add_argument('--data_dir',   type=str,   default='/home/bs_thesis/Documents/BS_THESIS/PCBVision/01_Data/Patches_256_Overlap_Data')
    args = parser.parse_args()

    print("="*60)
    print(f"  Hybrid SSRN ViT — HSI Exp (augment={args.augment})")
    print(f"  Device: {device}  |  Batch: {args.batch_size}")
    print("="*60)

    cutmix_aug    = MultimodalCutMix(beta=1.0)           if args.augment=='cutmix'    else None
    copypaste_aug = CopyPasteAugmentation((1,2,3))        if args.augment=='copypaste' else None

    train_ds = HSIDataset(args.data_dir, 'Train', cutmix_aug=cutmix_aug, copypaste_aug=copypaste_aug)
    val_ds   = HSIDataset(args.data_dir, 'Val')
    test_ds  = HSIDataset(args.data_dir, 'Test')

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=2,               shuffle=False, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=2,               shuffle=False, num_workers=0, pin_memory=True)

    model    = HybridSSRNSeg(in_channels=214, num_classes=4, image_size=256).to(device)
    loss_fn  = HybridLoss(focal_weight=0.5, dice_weight=0.5)
    opt      = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched    = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)

    results_dir = os.path.join(SCRIPT_DIR, 'Results')
    os.makedirs(results_dir, exist_ok=True)
    tag       = f"hsrn_hsi_nc214_{args.augment}"
    save_path = os.path.join(results_dir, f"{tag}_best.pth")
    log_path  = os.path.join(results_dir, f"{tag}_log.json")

    best_val = float('inf'); patience_count=0; history=[]
    print(f"\nStarting training — {args.epochs} epochs …\n")
    for epoch in range(1, args.epochs+1):
        tr_loss = train_epoch(train_loader, model, opt, loss_fn)
        vl_loss, vl_pa, vl_miou = evaluate(val_loader, model, loss_fn)
        sched.step()
        lr = opt.param_groups[0]['lr']
        rec = {'epoch':epoch,'train_loss':round(tr_loss,5),'val_loss':round(vl_loss,5),
               'val_pa':round(vl_pa,3),'val_miou':round(vl_miou,3),'lr':round(lr,7)}
        history.append(rec)
        print(f"Epoch {epoch:3d}/{args.epochs}  TrainLoss={tr_loss:.4f}  ValLoss={vl_loss:.4f}  "
              f"ValPA={vl_pa:.2f}%  ValMIoU={vl_miou:.2f}%  lr={lr:.2e}")
        if vl_loss < best_val:
            best_val=vl_loss; patience_count=0
            torch.save({'model':model.state_dict(),'epoch':epoch,'val_loss':vl_loss}, save_path)
            print(f"  ↑ Saved best → {save_path}")
        else:
            patience_count+=1
            if patience_count>=args.patience:
                print(f"\nEarly stopping at epoch {epoch}."); break
        with open(log_path,'w') as f:
            json.dump({'args':vars(args),'history':history}, f, indent=2)

    print(f"\nLog → {log_path}")
    print("\n=== Final Test Evaluation ===")
    model.load_state_dict(torch.load(save_path,map_location=device)['model'])
    tl, tp, tm = evaluate(test_loader, model, loss_fn)
    print(f"  TestLoss={tl:.4f}  TestPA={tp:.2f}%  TestMIoU={tm:.2f}%")
    print(f"\nDone!  Best model → {save_path}")

if __name__=='__main__':
    main()
