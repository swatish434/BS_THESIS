"""
dataset.py  v4
──────────────
Data routing:
  RGB      → RGB patches (256x256 from full images) + _mask.png
             img_ids = ['Train_0', 'Train_1', ...]
  HSI      → HSI patches (.hdr binary) + .npy masks
             img_ids = ['Train_0', 'Train_1', ...]
  RGB+HSI  → HSI patches + pseudo-RGB from HSI first 3 channels
             img_ids = ['Train_0', 'Train_1', ...]
"""

import os, random, cv2, glob
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import spectral.io.envi as envi
    SPECTRAL_AVAILABLE = True
except ImportError:
    SPECTRAL_AVAILABLE = False

from augmentations import apply_augmentation


# ─────────────────────────────────────────────────────────────────────────────
# ID discovery helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_patch_ids(patches_dir: str, split: str) -> list:
    """
    Returns sorted valid HSI patch IDs (both .hdr and .npy present,
    binary file > 1MB to filter corrupted patches).
    """
    prefix    = {'train': 'Train_', 'val': 'Val_', 'test': 'Test_'}[split]
    all_files = set(os.listdir(patches_dir))
    ids = []
    for fname in all_files:
        if fname.startswith(prefix) and fname.endswith('.hdr'):
            pid      = fname.replace('.hdr', '')
            npy_path = os.path.join(patches_dir, f"{pid}.npy")
            bin_path = os.path.join(patches_dir, pid)
            if not os.path.exists(npy_path): continue
            try:
                if os.path.getsize(bin_path) < 1000000: continue
                ids.append(pid)
            except Exception: continue
    ids = sorted(ids, key=lambda x: int(x.split('_')[1]))
    print(f"  [{split}] Found {len(ids)} valid HSI patches")
    return ids


def get_rgb_patch_ids(rgb_patches_dir: str, split: str) -> list:
    """
    Returns sorted RGB patch IDs from RGB_Patches_256 folder.
    Each patch has: Train_N.png + Train_N_mask.png
    """
    prefix = {'train': 'Train_', 'val': 'Val_', 'test': 'Test_'}[split]
    all_files = set(os.listdir(rgb_patches_dir))
    ids = []
    for fname in all_files:
        if fname.startswith(prefix) and fname.endswith('.png') and '_mask' not in fname:
            pid       = fname.replace('.png', '')
            mask_file = f"{pid}_mask.png"
            if mask_file in all_files:
                ids.append(pid)
    ids = sorted(ids, key=lambda x: int(x.split('_')[1]))
    print(f"  [{split}] Found {len(ids)} valid RGB patches")
    return ids


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class PCB_Dataset(Dataset):
    def __init__(self, img_ids, config, modality='RGB',
                 augmentation='None', mode='train'):
        self.img_ids      = img_ids
        self.config       = config
        self.modality     = modality
        self.augmentation = augmentation
        self.mode         = mode

    def __len__(self):
        return len(self.img_ids)

    # ── RGB patch loader ──────────────────────────────────────────────────────

    def _load_rgb_patch(self, patch_id: str) -> np.ndarray:
        """Load RGB patch from RGB_Patches_256 folder."""
        ts   = self.config.IMAGE_SIZE
        path = os.path.join(self.config.RGB_PATCHES_DIR, f"{patch_id}.png")
        img  = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"RGB patch not found: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[0] != ts or img.shape[1] != ts:
            img = cv2.resize(img, (ts, ts))
        return img.astype(np.float32) / 255.0

    def _load_mask_rgb_patch(self, patch_id: str) -> np.ndarray:
        """Load mask for RGB patch."""
        ts   = self.config.IMAGE_SIZE
        path = os.path.join(self.config.RGB_PATCHES_DIR, f"{patch_id}_mask.png")
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"RGB patch mask not found: {path}")
        if mask.shape[0] != ts or mask.shape[1] != ts:
            mask = cv2.resize(mask, (ts, ts), interpolation=cv2.INTER_NEAREST)
        return np.clip(mask, 0, self.config.NUM_CLASSES - 1).astype(np.int64)

    # ── HSI patch loader ──────────────────────────────────────────────────────

    def _load_hsi(self, patch_id: str) -> np.ndarray:
        """Load HSI patch from ENVI .hdr + binary file."""
        ts        = self.config.IMAGE_SIZE
        hdr_path  = os.path.join(self.config.HSI_DIR, f"{patch_id}.hdr")
        data_path = os.path.join(self.config.HSI_DIR, patch_id)
        if not (SPECTRAL_AVAILABLE and os.path.exists(hdr_path)):
            raise FileNotFoundError(f"HSI not found: {hdr_path}")
        hsi_obj = envi.open(hdr_path, data_path)
        hsi     = np.array(hsi_obj.load(), dtype=np.float32)
        h_min, h_max = hsi.min(), hsi.max()
        if h_max > h_min:
            hsi = (hsi - h_min) / (h_max - h_min + 1e-8)
        else:
            hsi = np.zeros_like(hsi)
        if hsi.shape[0] != ts or hsi.shape[1] != ts:
            hsi_r = np.zeros((ts, ts, hsi.shape[2]), dtype=np.float32)
            for c in range(hsi.shape[2]):
                hsi_r[:, :, c] = cv2.resize(hsi[:, :, c], (ts, ts))
            hsi = hsi_r
        return hsi

    def _load_rgb_from_hsi(self, patch_id: str) -> np.ndarray:
        """
        Derive pseudo-RGB from evenly spaced HSI bands.
        Uses bands 37, 112, 187 (short / mid / long wavelength)
        instead of bands 0,1,2 to avoid duplication with full HSI.
        """
        ts = self.config.IMAGE_SIZE
        try:
            hsi   = self._load_hsi(patch_id)
            n_bands = hsi.shape[2]
            # Pick 3 evenly spaced bands across full spectrum
            b_r = min(37,  n_bands - 1)
            b_g = min(112, n_bands - 1)
            b_b = min(187, n_bands - 1)
            rgb = np.stack([hsi[:, :, b_r],
                            hsi[:, :, b_g],
                            hsi[:, :, b_b]], axis=-1).copy()
            for c in range(3):
                mn, mx = rgb[:, :, c].min(), rgb[:, :, c].max()
                if mx > mn:
                    rgb[:, :, c] = (rgb[:, :, c] - mn) / (mx - mn + 1e-8)
            return rgb
        except Exception:
            return np.zeros((ts, ts, 3), dtype=np.float32)

    def _load_mask_patch(self, patch_id: str) -> np.ndarray:
        """Load .npy mask for HSI patch."""
        ts       = self.config.IMAGE_SIZE
        npy_path = os.path.join(self.config.PATCHES_DIR, f"{patch_id}.npy")
        if not os.path.exists(npy_path):
            raise FileNotFoundError(f"HSI mask not found: {npy_path}")
        mask = np.load(npy_path)
        if mask.ndim == 3:
            mask = mask.squeeze()
        if mask.shape[0] != ts or mask.shape[1] != ts:
            mask = cv2.resize(mask.astype(np.uint8), (ts, ts),
                              interpolation=cv2.INTER_NEAREST)
        return np.clip(mask, 0, self.config.NUM_CLASSES - 1).astype(np.int64)

    # ── Build one sample ──────────────────────────────────────────────────────

    def _build_image(self, img_id: str):
        if self.modality == 'RGB':
            return self._load_rgb_patch(img_id), self._load_mask_rgb_patch(img_id)
        elif self.modality == 'HSI':
            return self._load_hsi(img_id), self._load_mask_patch(img_id)
        elif self.modality == 'RGB+HSI':
            rgb = self._load_rgb_from_hsi(img_id)
            hsi = self._load_hsi(img_id)
            return np.concatenate([rgb, hsi], axis=-1), self._load_mask_patch(img_id)
        else:
            raise ValueError(f"Unknown modality: {self.modality}")

    # ── Main getter ───────────────────────────────────────────────────────────

    def __getitem__(self, idx):
        img_id          = self.img_ids[idx]
        img_data, mask  = self._build_image(img_id)
        if self.mode == 'train' and self.augmentation != 'None':
            donor_idx      = random.choice(
                [i for i in range(len(self.img_ids)) if i != idx])
            img_b, mask_b  = self._build_image(self.img_ids[donor_idx])
            img_data, mask = apply_augmentation(
                img_data, mask, img_b, mask_b, self.augmentation)
        return (torch.from_numpy(img_data).float().permute(2, 0, 1),
                torch.from_numpy(mask).long())