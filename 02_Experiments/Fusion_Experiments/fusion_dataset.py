
import os
import torch
import numpy as np
import glob
from torch.utils.data import Dataset

class FusionDataset(Dataset):
    def __init__(self, data_dir, split_indices=None, augment_cp=False, augment_cutmix=False, prebuilt_bank=None):
        """
        Args:
            data_dir (str): Path to directory containing 'images' and 'masks' subdirectories
            split_indices (list[int], optional): List of scene indices to include.
            augment_cp (bool): Whether to apply Copy-Paste augmentation.
            augment_cutmix (bool): Whether to apply CutMix augmentation.
            prebuilt_bank (dict, optional): Pre-built component bank mapping class_id to list of ComponentPatches.
        """
        self.image_dir = os.path.join(data_dir, 'images')
        self.mask_dir = os.path.join(data_dir, 'masks')
        self.augment_cp = augment_cp
        self.augment_cutmix = augment_cutmix
        
        # Get all mask files
        all_masks = sorted(glob.glob(os.path.join(self.mask_dir, "*.npy")))
        
        if split_indices is not None:
            self.mask_files = []
            for m in all_masks:
                filename = os.path.basename(m)
                try:
                    # Filename: patch_scene{idx}_{p_idx}.npy
                    scene_idx = int(filename.split('_scene')[1].split('_')[0])
                    if scene_idx in split_indices:
                        self.mask_files.append(m)
                except:
                    continue
        else:
            self.mask_files = all_masks
            
        self.image_files = [m.replace('masks', 'images') for m in self.mask_files]
        
        # Initialize Augmentors
        self.cp_augmentor = None
        if self.augment_cp:
            from utils.augmentation_functions import CopyPasteAugmentation
            self.cp_augmentor = CopyPasteAugmentation(minority_classes=(1, 2, 3), max_paste_per_class=3)
            if prebuilt_bank:
                self.cp_augmentor.component_bank = prebuilt_bank
        
        self.cutmix_augmentor = None
        if self.augment_cutmix:
            from utils.augmentation_functions import MultimodalCutMix
            self.cutmix_augmentor = MultimodalCutMix(beta=1.0)

        if split_indices is not None:
             print(f"FusionDataset: Loaded {len(self.image_files)} patches for {len(split_indices)} scenes. AugCP: {self.augment_cp}, AugCM: {self.augment_cutmix}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = np.load(self.image_files[idx]) # (H, W, 227)
        mask = np.load(self.mask_files[idx]).astype(np.int64) # (H, W)
        
        if image.dtype != np.float32:
            image = image.astype(np.float32)

        # 1. Apply CutMix
        if self.augment_cutmix and self.cutmix_augmentor and np.random.rand() < 0.5:
            idx2 = np.random.randint(len(self))
            image2 = np.load(self.image_files[idx2]).astype(np.float32)
            mask2 = np.load(self.mask_files[idx2]).astype(np.int64)
            _, image, mask, _, _ = self.cutmix_augmentor.cutmix(None, image, mask, None, image2, mask2)

        # 2. Apply Copy-Paste
        if self.augment_cp and self.cp_augmentor:
            image, mask = self.cp_augmentor(image, mask)
        
        # CHW conversion
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).long()
        
        return image, mask
