#!/usr/bin/env python3
"""
Layout Generation Script (The "Frankenstein" Generator)

Generates varied synthetic PCB layouts by aggressively pasting component patches
onto background images. These layouts are intended to be "refined" by SD-LoRA later.

Features:
- Loads background images (mostly empty or IC-only).
- Loads component bank (Capacitors, Connectors).
- Pastes N components per image with scale jittering and rotation.
- Saves pairs of (Layout_RGB, Layout_Mask).

Usage:
    python scripts/generate_layouts.py --num_images 1000 --output_dir synthetic_layouts
"""

import os
import sys
import random
import numpy as np
import cv2
import argparse
from tqdm import tqdm
from glob import glob
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.augmentation_functions import ComponentPatch, ComponentExtractor

def load_backgrounds(data_dir, split='Train'):
    """Load images that serve as good backgrounds."""
    # Search for *.jpg in data_dir (RGB folder)
    image_paths = sorted(glob(os.path.join(data_dir, "*.jpg")))
    
    backgrounds = []
    print(f"Loading {len(image_paths)} background candidates from {data_dir}...")
    
    for img_path in tqdm(image_paths):
        try:
            # Load Image
            img = cv2.imread(img_path)
            if img is None: continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Construct Mask Path
            # .../RGB/1.jpg -> .../RGB/General/1.png
            basename = os.path.basename(img_path)
            mask_name = basename.replace('.jpg', '.png')
            mask_path = os.path.join(data_dir, 'General', mask_name)
            
            # Load Mask
            if os.path.exists(mask_path):
                # Load as grayscale (indices)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    # Fallback to PIL if cv2 issues
                    from PIL import Image
                    mask = np.array(Image.open(mask_path))
            else:
                print(f"Warning: Mask not found for {basename}, creating empty")
                h, w = img.shape[:2]
                mask = np.zeros((h, w), dtype=np.uint8)
                 
            backgrounds.append((img, mask))
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            
    return backgrounds

def load_component_bank(bank_dir):
    """Load pre-extracted components from aug_bank directory."""
    comp_bank = {2: [], 3: []} # Capacitor, Connector
    
    class_map = {'Capacitor': 2, 'Connector': 3}
    
    print(f"Loading component bank from {bank_dir}...")
    
    for cls_name, cls_id in class_map.items():
        cls_dir = os.path.join(bank_dir, cls_name)
        if not os.path.exists(cls_dir):
            print(f"Warning: {cls_dir} does not exist")
            continue
            
        api_paths = glob(os.path.join(cls_dir, "*.png"))
        for p in api_paths:
            # Load as Unchanged to keep Alpha channel
            img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
            if img is not None:
                if img.shape[2] == 4:
                    # RGBA
                    alpha = img[:, :, 3]
                    rgb = img[:, :, :3]
                    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                    
                    # Create mask from alpha
                    # High alpha = component
                    mask = np.where(alpha > 128, cls_id, 0).astype(np.uint8)
                    
                    # Wrap in ComponentPatch
                    patch = ComponentPatch(rgb, mask, cls_id)
                    comp_bank[cls_id].append(patch)
                else:
                    # Fallback for RGB only (should not happen with new extractor)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    h, w = img.shape[:2]
                    mask = np.full((h, w), cls_id, dtype=np.uint8)
                    patch = ComponentPatch(img, mask, cls_id)
                    comp_bank[cls_id].append(patch)
                
    counts = {k: len(v) for k, v in comp_bank.items()}
    print(f"Component Bank Stats: {counts}")
    return comp_bank

# Mock ComponentPatch if utils one not available or simpler
class ComponentPatch:
    def __init__(self, data, mask, class_id):
        self.data = data
        self.mask = mask
        self.class_id = class_id

class LayoutGenerator:
    def __init__(self, comp_bank, backgrounds):
        self.comp_bank = comp_bank
        self.backgrounds = backgrounds
        self.classes = [2, 3] 
        
    def generate_single(self, min_components=5, max_components=12):
        if not self.backgrounds:
            raise ValueError("No backgrounds loaded!")
            
        # 1. Pick random background
        bg_idx = random.randint(0, len(self.backgrounds)-1)
        base_img, base_mask = self.backgrounds[bg_idx]
        
        # Copy to avoid modifying original
        layout_img = base_img.copy()
        layout_mask = base_mask.copy()
        
        # 2. Determine how many items to paste
        num_pastes = random.randint(min_components, max_components)
        
        for _ in range(num_pastes):
            # Pick class
            cls = random.choice(self.classes)
            if not self.comp_bank[cls]: continue
            
            # Pick component
            comp = random.choice(self.comp_bank[cls])
            
            # Random Scale
            scale = random.uniform(0.8, 1.5)
            
            # Check for valid location
            loc = self._find_location(layout_mask, comp.mask.shape, scale)
            
            if loc:
                self._paste(layout_img, layout_mask, comp, loc, scale)
                
        return layout_img, layout_mask

    def _find_location(self, mask, comp_shape, scale, max_attempts=20):
        H, W = mask.shape
        ph = int(comp_shape[0] * scale)
        pw = int(comp_shape[1] * scale)
        
        if ph >= H or pw >= W: return None
        
        for _ in range(max_attempts):
            # Try to place anywhere, even if overlapping (frankestein!)
            # But prefer empty areas if possible
            y = random.randint(0, H - ph)
            x = random.randint(0, W - pw)
            return (y, x) # Just return first valid bounds for now to force density
                
        return None

    def _paste(self, img, mask, comp, loc, scale, edge_blend=True):
        y, x = loc
        ph = int(comp.mask.shape[0] * scale)
        pw = int(comp.mask.shape[1] * scale)
        
        # Resize component RGB
        comp_rgb = cv2.resize(comp.data, (pw, ph), interpolation=cv2.INTER_LINEAR)
        
        # Resize component Mask
        comp_mask_small = cv2.resize(comp.mask, (pw, ph), interpolation=cv2.INTER_NEAREST)
        binary_mask = (comp_mask_small == comp.class_id).astype(np.float32)
        
        # Edge blending: create soft alpha mask using Gaussian blur
        if edge_blend:
            # Erode then blur to create smooth transition at edges
            kernel = np.ones((5, 5), np.uint8)
            eroded = cv2.erode(binary_mask.astype(np.uint8), kernel, iterations=1)
            alpha_mask = cv2.GaussianBlur(eroded.astype(np.float32), (15, 15), 5.0)
            alpha_mask = np.clip(alpha_mask, 0, 1)
        else:
            alpha_mask = binary_mask
        
        # Paste with alpha blending
        for c in range(3):  # RGB channels
            target_region = img[y:y+ph, x:x+pw, c].astype(np.float32)
            comp_region = comp_rgb[:, :, c].astype(np.float32)
            blended = alpha_mask * comp_region + (1 - alpha_mask) * target_region
            img[y:y+ph, x:x+pw, c] = blended.astype(np.uint8)
        
        # Mask is still hard assignment (ground truth must be sharp)
        mask[y:y+ph, x:x+pw] = np.where(
            comp_mask_small == comp.class_id,
            comp.class_id,
            mask[y:y+ph, x:x+pw]
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_images', type=int, default=1000)
    parser.add_argument('--output_dir', type=str, default='data/synthetic_layouts')
    # Default to main PCBDataset path
    parser.add_argument('--data_dir', type=str, default='/home/bs_thesis/Documents/BS_THESIS/DATASETS/PCBDataset/RGB')
    parser.add_argument('--bank_dir', type=str, default='aug_bank') # Existing bank
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    vis_dir = os.path.join(args.output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # 1. Load Resources
    backgrounds = load_backgrounds(args.data_dir)
    comp_bank = load_component_bank(args.bank_dir)
    
    if not backgrounds:
        print("Error: No backgrounds found!")
        return

    generator = LayoutGenerator(comp_bank, backgrounds)
    
    print(f"Generating {args.num_images} synthetic layouts...")
    
    for i in tqdm(range(args.num_images)):
        img, mask = generator.generate_single()
        
        # Save
        img_path = os.path.join(args.output_dir, f"syn_{i}.png")
        mask_path = os.path.join(args.output_dir, f"syn_{i}_mask.npy")
        
        # Save RGB as PNG
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(img_path, img_bgr)
        
        # Save Mask
        np.save(mask_path, mask)
        
        # Visualize first 10
        if i < 10:
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1); plt.title("Synthetic Layout"); plt.imshow(img)
            plt.subplot(1, 2, 2); plt.title("Mask"); plt.imshow(mask * 50)
            plt.savefig(os.path.join(vis_dir, f"vis_{i}.png"))
            plt.close()

    print(f"Done! Layouts saved to {args.output_dir}")

if __name__ == "__main__":
    main()
