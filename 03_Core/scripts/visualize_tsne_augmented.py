
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.manifold import TSNE
from tqdm import tqdm
import argparse

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.dataset_functions import read_dataset
from utils.augmentation_functions import CopyPasteAugmentation, MultimodalCutMix

def plot_augmented_tsne(dataset_path, output_path=None, num_samples_per_class=50, aug_mode="combined"):
    
    if output_path is None:
        if aug_mode == "combined":
            output_path = "Evaluation/benchmark_results/tsne_augmented.png"
        else:
            output_path = f"Evaluation/benchmark_results/tsne_{aug_mode}.png"

    print(f"Loading dataset for {aug_mode} augmentation...")
    # read_dataset returns: HSI, HSI_seg, HSI_mono, RGB, RGB_mono, RGB_general, PCB_Masks
    # We need HSI and HSI_mono (or RGB_mono which is usually the detailed one)
    # dataset_functions.py: HSI_mono_masks seems to be what we want (class masks)
    
    # Note: read_dataset returns lists.
    HSI, _, HSI_mono_masks, RGB, _, RGB_mono_masks, _ = read_dataset(dataset_path)
    
    # Filter out None values
    valid_indices = [i for i, h in enumerate(HSI) if h is not None and HSI_mono_masks[i] is not None]
    HSI = [HSI[i] for i in valid_indices]
    Masks = [HSI_mono_masks[i] for i in valid_indices]
    RGB = [RGB[i] for i in valid_indices] # Needed for CutMix pairing if we wanted RGB, but CutMix handles None RGB
    
    print(f"Loaded {len(HSI)} valid samples.")

    # Initialize Augmentations
    print("Initializing Augmentations...")
    # Copy-Paste
    # Minority classes: 2 (IC/Cap?), 3 (Connector?) -> Check visualize_spectral_signatures.py
    # visualize_spectral_signatures.py says: 0: Background, 1: Component, 2: IC, 3: Connector
    # But augmentation_functions says default minority_classes=(1,2,3)
    # Let's use 1, 2, 3.
    cp_aug = CopyPasteAugmentation(minority_classes=(1, 2, 3), max_paste_per_class=3, avoid_overlap=True)
    if aug_mode in ["combined", "copypaste"]:
        print("Building Copy-Paste Bank...")
        cp_aug.build_bank(HSI, Masks)
    
    # CutMix
    cutmix = MultimodalCutMix(beta=1.0)
    
    # Data collection for t-SNE
    class_pixels = {0: [], 1: [], 2: [], 3: []}
    class_names = {0: 'Background', 1: 'Capacitor', 2: 'IC', 3: 'Connector'}
    # Colors matching "Before" plot:
    # Capacitor (1): Blue
    # IC (2): Green
    # Connector (3): Orange
    colors = {0: 'black', 1: 'tab:blue', 2: 'tab:green', 3: 'tab:orange'}
    
    print("Applying Augmentations and Sampling Pixels...")
    
    for i in tqdm(range(len(HSI))):
        hsi_img = HSI[i] # (H, W, Bands)
        mask_img = Masks[i] # (H, W) or (H, W, 1)
        rgb_img = RGB[i]
        
        # Helper to resize
        def resize_to_match(img, target_h, target_w, interpolation=cv2.INTER_LINEAR):
            if img is None: return None
            if img.ndim == 3:
                 return cv2.resize(img, (target_w, target_h), interpolation=interpolation)
            else:
                 return cv2.resize(img, (target_w, target_h), interpolation=interpolation)

        # Ensure RGB matches HSI dimensions (CutMix requires synced dimensions)
        h, w = hsi_img.shape[:2]
        if rgb_img is not None and (rgb_img.shape[0] != h or rgb_img.shape[1] != w):
            rgb_img = resize_to_match(rgb_img, h, w)
            
        # Ensure mask matches HSI (usually does, but to be safe)
        if mask_img.shape[0] != h or mask_img.shape[1] != w:
             mask_img = resize_to_match(mask_img, h, w, interpolation=cv2.INTER_NEAREST)

        # Initialize augmented vars
        hsi_aug = hsi_img
        mask_aug = mask_img

        # Apply Copy-Paste
        if aug_mode in ["combined", "copypaste"]:
            hsi_aug, mask_aug = cp_aug(hsi_aug, mask_aug)
        
        # Apply CutMix
        # If 'cutmix' mode: Always apply (100%)
        # If 'combined' mode: 50% chance
        apply_cutmix = False
        if aug_mode == "cutmix":
            apply_cutmix = True
        elif aug_mode == "combined" and np.random.rand() < 0.5:
            apply_cutmix = True
            
        if apply_cutmix: 
            idx2 = np.random.randint(len(HSI))
            hsi2 = HSI[idx2]
            mask2 = Masks[idx2]
            rgb2 = RGB[idx2]
            
            # Resize second image to match first image dimensions for CutMix
            hsi2 = resize_to_match(hsi2, h, w)
            mask2 = resize_to_match(mask2, h, w, interpolation=cv2.INTER_NEAREST)
            rgb2 = resize_to_match(rgb2, h, w)
            
            # cutmix args: rgb1, hsi1, mask1, rgb2, hsi2, mask2
            # We can pass None for RGB if we only care about HSI
            _, hsi_aug, mask_aug, _, _ = cutmix.cutmix(rgb_img, hsi_aug, mask_aug, rgb2, hsi2, mask2)

        # Sampling Pixels
        # Ensure mask is 2D
        if mask_aug.ndim == 3: mask_aug = mask_aug.squeeze()
        
        height, width, bands = hsi_aug.shape
        
        for c in [1, 2, 3]: # Skipping background (0) to reduce noise/compute if not needed
                           # User image shows Capacitor, Connector, IC. No Background.
            coords = np.where(mask_aug == c)
            if len(coords[0]) > 0:
                num_pix = len(coords[0])
                # Randomly select subset
                n_samples = min(num_pix, num_samples_per_class)
                indices = np.random.choice(num_pix, n_samples, replace=False)
                
                selected_y = coords[0][indices]
                selected_x = coords[1][indices]
                
                spectra = hsi_aug[selected_y, selected_x, :]
                class_pixels[c].extend(spectra)

    # Prepare data for t-SNE
    print("Running t-SNE...")
    all_data = []
    all_labels = []
    
    for c in [1, 2, 3]: # Only plotting these classes as per user image
        data = np.array(class_pixels[c])
        if len(data) > 0:
            # Subsample if too many points total (TSNE is slow)
            # Limit to e.g. 1000 points per class total
            if len(data) > 2000:
                indices = np.random.choice(len(data), 2000, replace=False)
                data = data[indices]
                
            all_data.append(data)
            all_labels.extend([c] * len(data))
            
    if not all_data:
        print("No pixels found for classes 1, 2, 3!")
        return

    X = np.vstack(all_data)
    y = np.array(all_labels)
    
    print(f"Total samples for t-SNE: {X.shape}")
    
    # Run t-SNE
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    X_embedded = tsne.fit_transform(X)
    
    # Plotting
    print("Plotting...")
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for c in [1, 2, 3]:
        mask = (y == c)
        points = X_embedded[mask]
        ax.scatter(points[:, 0], points[:, 1], label=class_names[c], s=10, alpha=0.6, color=colors[c])
        
        # Add red dashed circle around Capacitor (Class 1) if desired
        # Or maybe CutMix needs highlighting? 
        # For CutMix, typically we want to see if classes mix well or if there's structure.
        # Let's keep the circle for Capacitor as requested previously, or maybe make it optional?
        # User requested "after cutmix aug also", implies similar visualization.
        if c == 1 and len(points) > 0:
            from matplotlib.patches import Ellipse
            
            # Calculate centroid and covariance
            mean = np.mean(points, axis=0)
            cov = np.cov(points, rowvar=False)
            
            # Eigenvalues/vectors for ellipse orientation
            lambda_, v = np.linalg.eig(cov)
            lambda_ = np.sqrt(lambda_)
            
            # Use 2 standard deviations (covers ~95% of data)
            ell = Ellipse(xy=(mean[0], mean[1]),
                          width=lambda_[0]*4, height=lambda_[1]*4,
                          angle=np.degrees(np.arctan2(v[1, 0], v[0, 0])),
                          edgecolor='red', fc='None', lw=2, linestyle='--')
            ax.add_patch(ell)
            
            # Add annotation/text if needed? User just said "Red Circle"
            
    ax.set_title(f"Pixel-level t-SNE ({aug_mode.capitalize()})")
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    # Custom legend order or labels if needed?
    # Default legend is fine as long as colors match.
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"Saved plot to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default="/home/bs_thesis/Documents/BS_THESIS/DATASETS/PCBDataset/")
    parser.add_argument('--aug_mode', type=str, default="combined", choices=["combined", "cutmix", "copypaste"], help="Augmentation mode: combined, cutmix, or copypaste")
    args = parser.parse_args()
    
    plot_augmented_tsne(args.dataset_path, aug_mode=args.aug_mode)
