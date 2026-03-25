import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import spectral.io.envi as envi
from tqdm import tqdm
import argparse

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def load_sample(data_dir, filename):
    header_path = os.path.join(data_dir, filename)
    base_filename = filename.replace('.hdr', '')
    data_path = os.path.join(data_dir, base_filename)
    mask_path = os.path.join(data_dir, base_filename + '.npy')
    
    try:
        if os.path.exists(mask_path):
            mask = np.load(mask_path)
            if len(mask.shape) == 3: mask = mask.squeeze()
        else:
            return None, None

        hsi_obj = envi.open(header_path, data_path)
        hsi_cube = hsi_obj.load()
        return np.array(hsi_cube), mask
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None, None

def plot_signatures(data_dir, num_samples=50):
    print(f"Scanning {data_dir}...")
    all_files = os.listdir(data_dir)
    headers = [f for f in all_files if f.endswith('.hdr')]
    
    # Class containers
    # 0: Background, 1: Component, 2: IC, 3: Connector
    class_pixels = {0: [], 1: [], 2: [], 3: []}
    class_names = {0: 'Background (PCB)', 1: 'Component (Cap/Res)', 2: 'Integrated Circuit (IC)', 3: 'Connector'}
    colors = {0: 'black', 1: 'green', 2: 'red', 3: 'blue'}
    
    count = 0
    for hdr in tqdm(headers):
        if count >= num_samples: break
        
        img, mask = load_sample(data_dir, hdr)
        if img is None: continue
        
        # Normalize if needed (assuming raw data is roughly consistent or sufficient for viz)
        # Usually we want to see relative differences.
        
        height, width, bands = img.shape
        
        # Sample pixels to save memory (don't take all)
        for c in [0, 1, 2, 3]:
            # Find coordinates of this class
            coords = np.where(mask == c)
            if len(coords[0]) > 0:
                # Randomly select up to 50 pixels per class per image
                num_pix = len(coords[0])
                indices = np.random.choice(num_pix, min(num_pix, 50), replace=False)
                
                selected_y = coords[0][indices]
                selected_x = coords[1][indices]
                
                # Extract spectra
                spectra = img[selected_y, selected_x, :]
                class_pixels[c].extend(spectra)
                
        count += 1
        
    print("Plotting...")
    plt.figure(figsize=(10, 6))
    
    for c in [0, 1, 2, 3]:
        if len(class_pixels[c]) == 0:
            print(f"No pixels found for class {c}")
            continue
            
        data = np.array(class_pixels[c])
        mean_spectrum = np.mean(data, axis=0)
        std_spectrum = np.std(data, axis=0)
        
        # Plot Mean
        x_axis = np.arange(len(mean_spectrum))
        plt.plot(x_axis, mean_spectrum, label=class_names[c], color=colors[c], linewidth=2)
        
        # Optional: Plot STD (shade)
        plt.fill_between(x_axis, mean_spectrum - std_spectrum, mean_spectrum + std_spectrum, color=colors[c], alpha=0.1)
        
    plt.title("Mean Spectral Signatures by Class")
    plt.xlabel("Spectral Band Index (0-213)")
    plt.ylabel("Reflectance / Intensity")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_path = "Evaluation/benchmark_results/spectral_signatures.png"
    plt.savefig(output_path, dpi=300)
    print(f"Saved plot to {output_path}")

if __name__ == "__main__":
    DATA_DIR = "/home/bs_thesis/Documents/BS_THESIS/PCBVision/Patches_256_Overlap_Data/"
    plot_signatures(DATA_DIR)
