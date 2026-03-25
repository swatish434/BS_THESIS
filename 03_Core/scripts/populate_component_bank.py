#!/usr/bin/env python3
"""
Populate Component Bank from Real Dataset.
Extracts components (Capacitor, Connector) and saves them as RGBA PNGs.
"""

import sys
import os
import cv2
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.dataset_functions import read_dataset
from utils.augmentation_functions import ComponentExtractor

def main():
    dataset_path = '/home/bs_thesis/Documents/BS_THESIS/DATASETS/PCBDataset/'
    bank_dir = 'aug_bank'
    
    # Classes to extract
    # 2: Capacitor, 3: Connector
    classes = {2: 'Capacitor', 3: 'Connector'}
    
    # Create directories
    for name in classes.values():
        os.makedirs(os.path.join(bank_dir, name), exist_ok=True)
        
    print("Loading dataset...")
    # read_dataset returns tuple of lists
    _, _, _, rgb_list, _, general_masks_list, _ = read_dataset(dataset_path)
    
    extractor = ComponentExtractor(min_area=100, max_area=50000)
    
    counts = {k: 0 for k in classes.keys()}
    
    print("Extracting components...")
    
    for i in tqdm(range(len(rgb_list))):
        if rgb_list[i] is None: continue
        
        img = rgb_list[i] # RGB
        mask = general_masks_list[i]
        
        for cls_id, cls_name in classes.items():
            patches = extractor.extract(img, mask, cls_id)
            
            for p in patches:
                # p.data is patch RGB
                # p.mask is patch Mask (could contain other classes or background)
                
                # Create Alpha channel based on mask
                binary_mask = (p.mask == cls_id).astype(np.uint8)
                
                # Check for empty mask (shouldn't happen due to extractor logic but safe check)
                if binary_mask.sum() == 0: continue
                
                # Convert to RGBA
                rgba = cv2.cvtColor(p.data, cv2.COLOR_RGB2RGBA)
                rgba[:, :, 3] = binary_mask * 255
                
                # Save
                filename = f"{cls_name}_{counts[cls_id]:05d}.png"
                save_path = os.path.join(bank_dir, cls_name, filename)
                
                # Convert to BGR for opencv saving
                rgba_bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
                cv2.imwrite(save_path, rgba_bgr)
                
                counts[cls_id] += 1
                
    print(f"Extraction complete!")
    print(f"Capacitors: {counts[2]}")
    print(f"Connectors: {counts[3]}")

if __name__ == "__main__":
    main()
