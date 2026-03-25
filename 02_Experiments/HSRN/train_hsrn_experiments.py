#!/usr/bin/env python3
"""
Unified training script for Hybrid SSRN ViT experiments on PCB Vision.

Supports 6 experiment configurations:
  --data_type rgb  --augment none      → Exp 01
  --data_type rgb  --augment copypaste → Exp 02
  --data_type rgb  --augment cutmix    → Exp 03
  --data_type hsi  --augment none      → Exp 04
  --data_type hsi  --augment copypaste → Exp 05
  --data_type hsi  --augment cutmix    → Exp 06

Usage examples:
  python train_hsrn_experiments.py --data_type rgb --augment none
  python train_hsrn_experiments.py --data_type hsi --augment copypaste --n_components 10
  python train_hsrn_experiments.py --data_type rgb --augment cutmix --epochs 100 --batch_size 8

Data Layout:
  RGB → <dataset_root>/RGB/<i>.jpg
         <dataset_root>/RGB/Monoseg/<i>.png   (class 0-3 monoseg mask)

  HSI → 01_Data/Patches_256_Overlap_Data/
         Train_<i>.hdr + Train_<i>  (ENVI cube), Train_<i>.npy (mask)
         Val/Test similarly.
"""

import os
import sys
import argparse
import json
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import warnings
import math
from typing import Optional, Tuple, List, Dict, Any

warnings.filterwarnings('ignore')

# ============================================================================
# Device Configuration
# ============================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Try to import spectral library for HSI support
try:
    import spectral.io.envi as envi
    SPECTRAL_AVAILABLE = True
except ImportError:
    SPECTRAL_AVAILABLE = False
    print("[WARN] 'spectral' library not installed. HSI mode will not be available.")


# ============================================================================
# Model Definition: Hybrid SSRN-ViT Segmentation
# ============================================================================

class ResidualBlock3D(nn.Module):
    """
    3D Residual block for spectral feature extraction.
    Fixed: Removed double ReLU, proper residual connection.
    """
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: tuple, padding: tuple, 
                 use_1x1conv: bool = False, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 
                               kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels,
                               kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        if use_1x1conv:
            self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.shortcut = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        if self.shortcut is not None:
            identity = self.shortcut(x)
        
        out = F.relu(out + identity)
        return out


class SpectralSpatialEncoder(nn.Module):
    """
    Spectral-Spatial encoder using 3D convolutions.
    Extracts features from hyperspectral or RGB input.
    """
    def __init__(self, in_channels: int, hidden_channels: int = 24):
        super().__init__()
        
        # Spectral convolution
        self.conv1 = nn.Conv3d(1, hidden_channels, kernel_size=(1, 1, 7), stride=(1, 1, 2))
        self.bn1 = nn.Sequential(
            nn.BatchNorm3d(hidden_channels),
            nn.ReLU(inplace=True)
        )
        
        # Spectral residual blocks
        self.res1 = ResidualBlock3D(hidden_channels, hidden_channels, (1, 1, 7), (0, 0, 3))
        self.res2 = ResidualBlock3D(hidden_channels, hidden_channels, (1, 1, 7), (0, 0, 3))
        
        # Spatial residual blocks
        self.res3 = ResidualBlock3D(hidden_channels, hidden_channels, (3, 3, 1), (1, 1, 0))
        self.res4 = ResidualBlock3D(hidden_channels, hidden_channels, (3, 3, 1), (1, 1, 0))
        
        # Spectral reduction
        kernel_3d = max(1, math.ceil((in_channels - 6) / 2))
        self.conv2 = nn.Conv3d(hidden_channels, 128, kernel_size=(1, 1, kernel_3d))
        self.bn2 = nn.Sequential(
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True)
        )
        
        # Spatial feature extraction
        self.conv3 = nn.Conv3d(1, hidden_channels, kernel_size=(3, 3, 128), padding=(0, 0, 0))
        self.bn3 = nn.Sequential(
            nn.BatchNorm3d(hidden_channels),
            nn.ReLU(inplace=True)
        )
        
        self.out_channels = hidden_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, H, W)
        Returns:
            Features of shape (B, hidden_channels, H-2, W-2)
        """
        B, C, H, W = x.shape
        
        # Reshape for 3D convolution: (B, 1, H, W, C)
        x = x.unsqueeze(1).permute(0, 1, 2, 3, 4)
        
        # Spectral processing
        x = self.bn1(self.conv1(x))
        x = self.res1(x)
        x = self.res2(x)
        
        # Spatial processing
        x = self.res3(x)
        x = self.res4(x)
        
        # Spectral reduction
        x = self.bn2(self.conv2(x))
        
        # Rearrange for spatial convolution
        x = x.permute(0, 4, 2, 3, 1)  # (B, 1, H, W, 128)
        
        # Spatial feature extraction
        x = self.bn3(self.conv3(x))
        
        # Reshape output: (B, hidden_channels, H-2, W-2)
        x = x.squeeze(4)  # Remove last dimension
        
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""
    
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim ** -0.5
        
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, n, _ = x.shape
        h = self.heads
        
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(b, n, h, -1).transpose(1, 2), qkv)
        
        dots = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            dots = dots.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(dots, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(b, n, -1)
        
        return self.to_out(out)


class TransformerBlock(nn.Module):
    """Transformer encoder block."""
    
    def __init__(self, dim: int, heads: int, dim_head: int, mlp_dim: int, dropout: float):
        super().__init__()
        self.attn = nn.Sequential(
            nn.LayerNorm(dim),
            MultiHeadAttention(dim, heads, dim_head, dropout)
        )
        self.ff = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        x = x + self.ff(x)
        return x


class ViTEncoder(nn.Module):
    """
    Vision Transformer encoder for spatial feature refinement.
    """
    
    def __init__(self, image_size: int, patch_size: int, dim: int, 
                 depth: int, heads: int, mlp_dim: int, 
                 channels: int = 24, dropout: float = 0.1):
        super().__init__()
        
        assert image_size % patch_size == 0, "Image size must be divisible by patch size"
        
        self.patch_size = patch_size
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        
        self.patch_embed = nn.Linear(patch_dim, dim)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(dropout)
        
        self.transformer = nn.ModuleList([
            TransformerBlock(dim, heads, dim // heads, mlp_dim, dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, H, W)
        Returns:
            Features of shape (B, num_patches, dim)
        """
        B, C, H, W = x.shape
        p = self.patch_size
        
        # Create patches
        x = x.unfold(2, p, p).unfold(3, p, p)  # (B, C, H/p, W/p, p, p)
        x = x.contiguous().view(B, C, -1, p, p)
        x = x.permute(0, 2, 3, 4, 1).contiguous()  # (B, num_patches, p, p, C)
        x = x.view(B, -1, p * p * C)  # (B, num_patches, patch_dim)
        
        # Patch embedding
        x = self.patch_embed(x)
        
        # Add class token and position embedding
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed[:, :x.size(1)]
        x = self.dropout(x)
        
        # Transformer blocks
        for block in self.transformer:
            x = block(x)
        
        return self.norm(x)


class Decoder(nn.Module):
    """
    Segmentation decoder with skip connections and upsampling.
    """
    
    def __init__(self, in_channels: int, num_classes: int, 
                 image_size: int, patch_size: int):
        super().__init__()
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches_side = image_size // patch_size
        
        # Feature projection
        self.proj = nn.Linear(in_channels, 64)
        
        # Decoder convolutions
        self.up1 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        
        self.up2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        
        self.up3 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, 1)
        )

    def forward(self, x: torch.Tensor, original_size: int) -> torch.Tensor:
        """
        Args:
            x: Features from ViT of shape (B, num_patches+1, dim)
            original_size: Original image size
        Returns:
            Segmentation logits of shape (B, num_classes, H, W)
        """
        B = x.size(0)
        
        # Remove CLS token and reshape
        x = x[:, 1:]  # (B, num_patches, dim)
        x = self.proj(x)  # (B, num_patches, 64)
        
        # Reshape to spatial
        x = x.view(B, self.num_patches_side, self.num_patches_side, -1)
        x = x.permute(0, 3, 1, 2)  # (B, 64, H/p, W/p)
        
        # Upsampling path
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        
        # Resize to original size
        x = F.interpolate(x, size=(original_size, original_size), 
                          mode='bilinear', align_corners=False)
        
        return x


class HybridSSRNSeg(nn.Module):
    """
    Hybrid SSRN-ViT model for semantic segmentation.
    Combines spectral-spatial feature extraction with transformer attention.
    """
    
    def __init__(self, in_channels: int, num_classes: int, 
                 image_size: int = 256, patch_size: int = 16,
                 dim: int = 512, depth: int = 6, heads: int = 8, 
                 mlp_dim: int = 1024, dropout: float = 0.1):
        super().__init__()
        
        self.image_size = image_size
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        # Spectral-Spatial encoder
        self.encoder = SpectralSpatialEncoder(in_channels, hidden_channels=24)
        
        # ViT encoder
        self.vit = ViTEncoder(
            image_size=image_size - 2,  # Account for encoder spatial reduction
            patch_size=min(patch_size, image_size - 2),
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            channels=24,
            dropout=dropout
        )
        
        # Decoder
        self.decoder = Decoder(
            in_channels=dim,
            num_classes=num_classes,
            image_size=image_size - 2,
            patch_size=min(patch_size, image_size - 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, H, W)
        Returns:
            Segmentation logits of shape (B, num_classes, H, W)
        """
        # Encode
        spatial_features = self.encoder(x)
        
        # ViT processing
        vit_features = self.vit(spatial_features)
        
        # Decode
        logits = self.decoder(vit_features, self.image_size)
        
        return logits


# ============================================================================
# Loss Functions
# ============================================================================

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, 
                 reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class DiceLoss(nn.Module):
    """Dice Loss for segmentation."""
    
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = inputs.size(1)
        
        # Apply softmax
        probs = F.softmax(inputs, dim=1)
        
        # One-hot encode targets
        targets_one_hot = F.one_hot(targets, num_classes).float()
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2)
        
        # Flatten
        probs_flat = probs.view(-1)
        targets_flat = targets_one_hot.view(-1)
        
        intersection = (probs_flat * targets_flat).sum()
        union = probs_flat.sum() + targets_flat.sum()
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice


class HybridLoss(nn.Module):
    """
    Combined loss for semantic segmentation.
    Fixed: Proper weight handling and device management.
    """
    
    def __init__(self, focal_weight: float = 0.5, dice_weight: float = 0.5,
                 focal_alpha: float = 0.25, focal_gamma: float = 2.0):
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dice_loss = DiceLoss()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        focal = self.focal_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        
        return self.focal_weight * focal + self.dice_weight * dice


# ============================================================================
# Augmentation Functions
# ============================================================================

class CopyPasteAugmentation:
    """
    Copy-Paste augmentation for semantic segmentation.
    Copies instances from minority classes to augment training data.
    """
    
    def __init__(self, minority_classes: Tuple[int, ...] = (1, 2, 3),
                 paste_probability: float = 0.5):
        self.minority_classes = minority_classes
        self.paste_probability = paste_probability
        self.bank: List[Tuple[np.ndarray, np.ndarray]] = []
    
    def build_bank(self, images: List[np.ndarray], masks: List[np.ndarray]) -> None:
        """Build bank of instances from minority classes."""
        self.bank = []
        
        for img, mask in zip(images, masks):
            for cls in self.minority_classes:
                class_mask = (mask == cls)
                if class_mask.sum() > 0:
                    # Find bounding box
                    rows = np.any(class_mask, axis=1)
                    cols = np.any(class_mask, axis=0)
                    if rows.any() and cols.any():
                        rmin, rmax = np.where(rows)[0][[0, -1]]
                        cmin, cmax = np.where(cols)[0][[0, -1]]
                        
                        # Extract instance
                        instance_img = img[rmin:rmax+1, cmin:cmax+1].copy()
                        instance_mask = mask[rmin:rmax+1, cmin:cmax+1].copy()
                        
                        self.bank.append((instance_img, instance_mask, cls))
        
        print(f"[CopyPaste] Built bank with {len(self.bank)} instances")
    
    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply copy-paste augmentation."""
        if not self.bank or np.random.rand() > self.paste_probability:
            return image, mask
        
        # Random instance from bank
        idx = np.random.randint(len(self.bank))
        instance_img, instance_mask, cls = self.bank[idx]
        
        # Random position
        h, w = image.shape[:2]
        ih, iw = instance_img.shape[:2]
        
        if ih >= h or iw >= w:
            return image, mask
        
        y = np.random.randint(0, h - ih)
        x = np.random.randint(0, w - iw)
        
        # Paste instance
        region = instance_mask > 0
        image[y:y+ih, x:x+iw][region] = instance_img[region]
        mask[y:y+ih, x:x+iw][region] = instance_mask[region]
        
        return image, mask


class MultimodalCutMix:
    """
    CutMix augmentation for segmentation.
    """
    
    def __init__(self, beta: float = 1.0, prob: float = 0.5):
        self.beta = beta
        self.prob = prob
    
    def cutmix(self, img1: Optional[np.ndarray], img2: np.ndarray,
               mask1: np.ndarray, mask2: Optional[np.ndarray],
               img2_aux: Optional[np.ndarray] = None,
               mask2_aux: Optional[np.ndarray] = None
               ) -> Tuple[Optional[np.ndarray], np.ndarray, np.ndarray, 
                          Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Apply CutMix between two samples.
        
        Returns:
            Tuple of (mixed_img1, mixed_img2, mixed_mask1, mixed_mask2, mixed_img2_aux)
        """
        if np.random.rand() > self.prob:
            return img1, img2, mask1, img2_aux, mask2_aux
        
        h, w = img2.shape[:2]
        
        # Sample bounding box
        lam = np.random.beta(self.beta, self.beta)
        cut_w = int(w * np.sqrt(1 - lam))
        cut_h = int(h * np.sqrt(1 - lam))
        
        cx = np.random.randint(0, w - cut_w + 1) if cut_w < w else 0
        cy = np.random.randint(0, h - cut_h + 1) if cut_h < h else 0
        
        # Create mixed image and mask
        mixed_img = img2.copy()
        mixed_mask = mask2.copy()
        
        # Apply cut region from first sample (if available)
        if img1 is not None and mask1 is not None:
            mixed_img[cy:cy+cut_h, cx:cx+cut_w] = img1[cy:cy+cut_h, cx:cx+cut_w]
            mixed_mask[cy:cy+cut_h, cx:cx+cut_w] = mask1[cy:cy+cut_h, cx:cx+cut_w]
        
        return None, mixed_img, mixed_mask, None, None


# ============================================================================
# Datasets
# ============================================================================

class RGBPatchDataset(Dataset):
    """
    Loads RGB images and segmentation masks for PCB defect segmentation.
    """
    
    TRAIN_IDS = list(range(1, 39))   # 1..38
    VAL_IDS   = list(range(39, 45))  # 39..44
    TEST_IDS  = list(range(45, 54))  # 45..53
    
    def __init__(self, dataset_root: str, split: str = 'Train',
                 cutmix_aug: Optional[MultimodalCutMix] = None,
                 copypaste_aug: Optional[CopyPasteAugmentation] = None,
                 target_size: int = 256):
        self.dataset_root = dataset_root
        self.split = split
        self.cutmix_aug = cutmix_aug
        self.copypaste_aug = copypaste_aug
        self.target_size = target_size
        
        # Select split IDs
        split_ids = {
            'Train': self.TRAIN_IDS,
            'Val':   self.VAL_IDS,
            'Test':  self.TEST_IDS,
        }[split]
        
        # Find valid samples
        self.sample_ids = []
        for sid in split_ids:
            img_path = os.path.join(dataset_root, 'RGB', f'{sid}.jpg')
            mask_path = os.path.join(dataset_root, 'RGB', 'Monoseg', f'{sid}.png')
            if os.path.isfile(img_path) and os.path.isfile(mask_path):
                self.sample_ids.append(sid)
        
        print(f"[RGB] {split}: {len(self.sample_ids)} samples")
        
        # Build CopyPaste bank for training
        if copypaste_aug is not None and split == 'Train':
            self._build_copypaste_bank()
    
    def _load_sample(self, sid: int) -> Tuple[np.ndarray, np.ndarray]:
        """Load and resize a sample."""
        img_path = os.path.join(self.dataset_root, 'RGB', f'{sid}.jpg')
        mask_path = os.path.join(self.dataset_root, 'RGB', 'Monoseg', f'{sid}.png')
        
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        
        # Load mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to load mask: {mask_path}")
        
        # Resize
        ts = self.target_size
        img = cv2.resize(img, (ts, ts))
        mask = cv2.resize(mask.astype(np.uint16), (ts, ts), 
                          interpolation=cv2.INTER_NEAREST)
        
        return img, mask
    
    def _build_copypaste_bank(self) -> None:
        """Build CopyPaste instance bank."""
        print("[RGB] Building CopyPaste bank...")
        imgs, masks = [], []
        for sid in self.sample_ids:
            img, mask = self._load_sample(sid)
            imgs.append(img)
            masks.append(mask)
        self.copypaste_aug.build_bank(imgs, masks)
    
    def __len__(self) -> int:
        return len(self.sample_ids)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sid = self.sample_ids[idx]
        img, mask = self._load_sample(sid)
        
        # Apply CutMix augmentation
        if self.cutmix_aug and self.split == 'Train' and np.random.rand() < 0.5:
            idx2 = np.random.randint(len(self))
            img2, mask2 = self._load_sample(self.sample_ids[idx2])
            try:
                _, img_m, mask_m, _, _ = self.cutmix_aug.cutmix(
                    img, img2, mask, mask2)
                img, mask = img_m, mask_m
            except Exception:
                pass
        
        # Apply CopyPaste augmentation
        if self.copypaste_aug and self.split == 'Train' and np.random.rand() < 0.5:
            try:
                img, mask = self.copypaste_aug(img, mask)
            except Exception:
                pass
        
        # Convert to tensors
        img_t = torch.from_numpy(img.transpose(2, 0, 1)).float()
        mask_t = torch.from_numpy(mask.astype(np.int64)).long()
        
        return img_t, mask_t


class HSIPCADataset(Dataset):
    """
    Loads HSI patches from ENVI format files.
    Fixed: Proper error handling and mask loading.
    """
    
    def __init__(self, data_dir: str, split: str = 'Train',
                 n_components: int = 214,
                 cutmix_aug: Optional[MultimodalCutMix] = None,
                 copypaste_aug: Optional[CopyPasteAugmentation] = None,
                 target_size: int = 256):
        
        if not SPECTRAL_AVAILABLE:
            raise ImportError("'spectral' library required for HSI mode. "
                            "Install with: pip install spectral")
        
        self.data_dir = data_dir
        self.split = split
        self.n_components = n_components
        self.cutmix_aug = cutmix_aug
        self.copypaste_aug = copypaste_aug
        self.target_size = target_size
        
        # Find header files
        all_files = os.listdir(data_dir)
        self.headers = sorted(
            [f for f in all_files
             if f.startswith(f'{split}_') and f.endswith('.hdr')],
            key=lambda x: int(x.replace(f'{split}_', '').replace('.hdr', ''))
        )
        
        print(f"[HSI] {split}: {len(self.headers)} samples")
        
        # Build CopyPaste bank for training
        if copypaste_aug is not None and split == 'Train':
            self._build_copypaste_bank()
    
    def _load_sample(self, hdr_filename: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Load HSI cube and mask from ENVI format."""
        base = hdr_filename.replace('.hdr', '')
        hdr_path = os.path.join(self.data_dir, hdr_filename)
        data_path = os.path.join(self.data_dir, base)
        mask_path = os.path.join(self.data_dir, f'{base}.npy')
        
        try:
            # Load HSI cube
            hsi_obj = envi.open(hdr_path, data_path)
            hsi = np.array(hsi_obj.load(), dtype=np.float32)
            
            # Load mask
            mask = np.load(mask_path)
            
            # Handle mask shape
            if mask.ndim == 3:
                if mask.shape[-1] == 1:
                    mask = mask[:, :, 0]
                elif mask.shape[0] == 1:
                    mask = mask[0, :, :]
            
            # Normalize HSI
            h_min, h_max = hsi.min(), hsi.max()
            if h_max > h_min:
                hsi = (hsi - h_min) / (h_max - h_min)
            else:
                hsi = np.zeros_like(hsi)
            
            # Resize if needed
            H, W, C = hsi.shape
            ts = self.target_size
            
            if H != ts or W != ts:
                hsi_r = np.zeros((ts, ts, C), dtype=np.float32)
                for c in range(C):
                    hsi_r[:, :, c] = cv2.resize(hsi[:, :, c], (ts, ts))
                hsi = hsi_r
                mask = cv2.resize(mask.astype(np.uint16), (ts, ts),
                                  interpolation=cv2.INTER_NEAREST)
            
            return hsi, mask
            
        except Exception as e:
            print(f"[WARN] Failed to load {hdr_filename}: {e}")
            return None, None
    
    def _build_copypaste_bank(self) -> None:
        """Build CopyPaste instance bank."""
        print("[HSI] Building CopyPaste bank...")
        imgs, masks = [], []
        for hdr in tqdm(self.headers, desc="Building bank"):
            hsi, mask = self._load_sample(hdr)
            if hsi is not None and mask is not None:
                imgs.append(hsi)
                masks.append(mask)
        if imgs:
            self.copypaste_aug.build_bank(imgs, masks)
    
    def __len__(self) -> int:
        return len(self.headers)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        hsi, mask = self._load_sample(self.headers[idx])
        
        # Handle load failure
        if hsi is None:
            return (torch.zeros(self.n_components, self.target_size, self.target_size),
                    torch.zeros(self.target_size, self.target_size, dtype=torch.long))
        
        # Apply CutMix augmentation
        if self.cutmix_aug and self.split == 'Train' and np.random.rand() < 0.5:
            idx2 = np.random.randint(len(self))
            hsi2, mask2 = self._load_sample(self.headers[idx2])
            if hsi2 is not None:
                try:
                    _, hsi_m, mask_m, _, _ = self.cutmix_aug.cutmix(
                        hsi, hsi2, mask, mask2)
                    hsi, mask = hsi_m, mask_m
                except Exception:
                    pass
        
        # Apply CopyPaste augmentation
        if self.copypaste_aug and self.split == 'Train' and np.random.rand() < 0.5:
            try:
                hsi, mask = self.copypaste_aug(hsi, mask)
            except Exception:
                pass
        
        # Convert to tensors
        hsi_t = torch.from_numpy(hsi.transpose(2, 0, 1)).float()
        mask_t = torch.from_numpy(mask.astype(np.int64)).long()
        
        return hsi_t, mask_t


# ============================================================================
# Metrics
# ============================================================================

def compute_metrics(preds: torch.Tensor, targets: torch.Tensor, 
                    num_classes: int = 4) -> Tuple[float, float]:
    """
    Compute pixel accuracy and mean IoU.
    Fixed: Proper handling of edge cases.
    """
    pred_np = preds.cpu().numpy()
    target_np = targets.cpu().numpy()
    
    # Pixel accuracy
    pa_sum = (pred_np == target_np).sum()
    total = target_np.size
    pa = pa_sum / total if total > 0 else 0.0
    
    # mIoU
    iou_per_class = []
    for c in range(num_classes):
        pred_c = (pred_np == c)
        target_c = (target_np == c)
        
        intersection = (pred_c & target_c).sum()
        union = (pred_c | target_c).sum()
        
        if union > 0:
            iou_per_class.append(intersection / union)
    
    miou = np.mean(iou_per_class) if iou_per_class else 0.0
    
    return pa, miou


# ============================================================================
# Training Functions
# ============================================================================

def train_epoch(loader: DataLoader, model: nn.Module, 
                optimizer: optim.Optimizer, loss_fn: nn.Module) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    for data, targets in tqdm(loader, desc='  Train', leave=False):
        data = data.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        preds = model(data)
        loss = loss_fn(preds, targets)
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(loader)


def evaluate(loader: DataLoader, model: nn.Module, 
             loss_fn: nn.Module, num_classes: int = 4) -> Tuple[float, float, float]:
    """Evaluate model on validation/test set."""
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for data, targets in tqdm(loader, desc='  Eval ', leave=False):
            data = data.to(device)
            targets = targets.to(device)
            
            preds = model(data)
            loss = loss_fn(preds, targets)
            total_loss += loss.item()
            
            all_preds.append(preds.argmax(dim=1))
            all_targets.append(targets)
    
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    
    pa, miou = compute_metrics(all_preds, all_targets, num_classes)
    
    return total_loss / len(loader), pa * 100, miou * 100


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Hybrid SSRN ViT experiments on PCB Vision')
    
    # Data arguments
    parser.add_argument('--data_type', type=str, default='rgb',
                        choices=['rgb', 'hsi'],
                        help='Input modality: rgb or hsi')
    parser.add_argument('--augment', type=str, default='none',
                        choices=['none', 'copypaste', 'cutmix'],
                        help='Augmentation strategy')
    parser.add_argument('--n_components', type=int, default=214,
                        help='Number of HSI bands')
    parser.add_argument('--dataset_root', type=str,
                        default='/home/bs_thesis/Documents/BS_THESIS/DATASETS/PCBDataset',
                        help='Dataset root directory')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=15)
    
    # Model arguments
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--dim', type=int, default=512)
    parser.add_argument('--depth', type=int, default=6)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--mlp_dim', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0.1)
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./Results',
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    # Validate data paths
    if args.data_type == 'rgb':
        if not os.path.isdir(args.dataset_root):
            print(f"[ERROR] Dataset root not found: {args.dataset_root}")
            sys.exit(1)
        rgb_dir = os.path.join(args.dataset_root, 'RGB')
        if not os.path.isdir(rgb_dir):
            print(f"[ERROR] RGB directory not found: {rgb_dir}")
            sys.exit(1)
        data_path = args.dataset_root
        in_channels = 3
    else:
        # HSI mode - check for data directory
        hsi_dir = args.dataset_root
        if not os.path.isdir(hsi_dir):
            print(f"[ERROR] HSI data directory not found: {hsi_dir}")
            sys.exit(1)
        data_path = hsi_dir
        in_channels = args.n_components
    
    # Print configuration
    print("=" * 70)
    print("  Hybrid SSRN ViT — PCB Vision Experiment")
    print("=" * 70)
    print(f"  Data type   : {args.data_type.upper()}")
    print(f"  Augmentation: {args.augment}")
    print(f"  Data path   : {data_path}")
    print(f"  Input chans : {in_channels}")
    print(f"  Device      : {device}")
    print(f"  Epochs      : {args.epochs}")
    print(f"  Batch size  : {args.batch_size}")
    print("=" * 70)
    
    # Setup augmentations
    cutmix_aug = MultimodalCutMix(beta=1.0) if args.augment == 'cutmix' else None
    copypaste_aug = (CopyPasteAugmentation(minority_classes=(1, 2, 3))
                     if args.augment == 'copypaste' else None)
    
    if args.augment != 'none':
        print(f"\n[AUG] {args.augment.capitalize()} augmentation enabled")
    
    # Create datasets
    print("\nLoading datasets...")
    if args.data_type == 'rgb':
        train_ds = RGBPatchDataset(
            dataset_root=data_path, split='Train',
            cutmix_aug=cutmix_aug, copypaste_aug=copypaste_aug,
            target_size=args.image_size)
        val_ds = RGBPatchDataset(
            dataset_root=data_path, split='Val',
            target_size=args.image_size)
        test_ds = RGBPatchDataset(
            dataset_root=data_path, split='Test',
            target_size=args.image_size)
    else:
        train_ds = HSIPCADataset(
            data_dir=data_path, split='Train',
            n_components=args.n_components,
            cutmix_aug=cutmix_aug, copypaste_aug=copypaste_aug,
            target_size=args.image_size)
        val_ds = HSIPCADataset(
            data_dir=data_path, split='Val',
            n_components=args.n_components,
            target_size=args.image_size)
        test_ds = HSIPCADataset(
            data_dir=data_path, split='Test',
            n_components=args.n_components,
            target_size=args.image_size)
    
    # Create data loaders
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, 
                              shuffle=True, num_workers=0, pin_memory=True,
                              drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False,
                            num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=2, shuffle=False,
                             num_workers=0, pin_memory=True)
    
    # Create model
    print("\nCreating model...")
    model = HybridSSRNSeg(
        in_channels=in_channels,
        num_classes=args.num_classes,
        image_size=args.image_size,
        patch_size=args.patch_size,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        mlp_dim=args.mlp_dim,
        dropout=args.dropout,
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    
    # Loss and optimizer
    loss_fn = HybridLoss(focal_weight=0.5, dice_weight=0.5)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    exp_tag = f"hsrn_{args.data_type}_nc{in_channels}_{args.augment}"
    save_path = os.path.join(args.output_dir, f"{exp_tag}_best.pth")
    log_path = os.path.join(args.output_dir, f"{exp_tag}_log.json")
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    history = []
    
    print(f"\nStarting training for {args.epochs} epochs...\n")
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss = train_epoch(train_loader, model, optimizer, loss_fn)
        
        # Validate
        val_loss, val_pa, val_miou = evaluate(val_loader, model, loss_fn, args.num_classes)
        
        # Update scheduler
        scheduler.step()
        
        # Get current learning rate
        lr_now = optimizer.param_groups[0]['lr']
        
        # Record history
        record = {
            'epoch': epoch,
            'train_loss': round(train_loss, 5),
            'val_loss': round(val_loss, 5),
            'val_pa': round(val_pa, 3),
            'val_miou': round(val_miou, 3),
            'lr': round(lr_now, 7),
        }
        history.append(record)
        
        # Print progress
        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"TrainLoss={train_loss:.4f}  ValLoss={val_loss:.4f}  "
              f"ValPA={val_pa:.2f}%  ValMIoU={val_miou:.2f}%  lr={lr_now:.2e}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Fixed: Remove weights_only=True for save (it's for loading)
            state_dict = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'args': vars(args),
            }
            torch.save(state_dict, save_path)
            print(f"  ↑ Saved best model → {save_path}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break
        
        # Save log
        with open(log_path, 'w') as f:
            json.dump({'args': vars(args), 'history': history}, f, indent=2)
    
    print(f"\nTraining log saved → {log_path}")
    
    # Final test evaluation
    print("\n" + "=" * 70)
    print("Final Test Evaluation")
    print("=" * 70)
    
    # Fixed: Use weights_only=False since checkpoint contains non-tensor data
    checkpoint = torch.load(save_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    
    test_loss, test_pa, test_miou = evaluate(test_loader, model, loss_fn, args.num_classes)
    
    print(f"  Test Loss  : {test_loss:.4f}")
    print(f"  Test PA    : {test_pa:.2f}%")
    print(f"  Test mIoU  : {test_miou:.2f}%")
    print(f"\nBest model saved at: {save_path}")
    
    # Save final results
    final_results = {
        'args': vars(args),
        'history': history,
        'test_results': {
            'test_loss': round(test_loss, 5),
            'test_pa': round(test_pa, 3),
            'test_miou': round(test_miou, 3),
        }
    }
    
    final_path = os.path.join(args.output_dir, f"{exp_tag}_final.json")
    with open(final_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"Final results saved → {final_path}")


if __name__ == '__main__':
    main()
