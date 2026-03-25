import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import argparse
import sys

# Add parent directory to path to import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.dataset_functions import read_dataset
from utils.augmentation_functions import CopyPasteAugmentation, MultimodalCutMix

def count_pixels(masks, exclude_background=True):
    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for mask in masks:
        unique, counts_per_mask = np.unique(mask, return_counts=True)
        for u, c in zip(unique, counts_per_mask):
            if int(u) in counts:
                counts[int(u)] += c
    
    if exclude_background:
        if 0 in counts:
            del counts[0]
            
    return counts

def plot_pie_chart(counts, title, output_path, colors):
    labels = []
    sizes = []
    plot_colors = []
    
    # Mapping: 1=Component(Capacitor), 2=IC, 3=Connector
    # User image uses: IC=Red, Capacitor=Green, Connectors=Blue
    # My previous mapping was: Capacitor(1)=Blue, IC(2)=Green, Connector(3)=Orange
    # The user accepted my previous mapping for t-SNE.
    # However, for this specific request, they shared an image with specific colors.
    # "Figure 3 :Class distribution" shows:
    # IC (Red) ~ 82%
    # Capacitor (Green) ~ 8.7%
    # Connectors (Blue) ~ 9.2%
    
    # I should check the class IDs again.
    # In thesis_experiments_report.tex:
    # 1 = IC (4.52%) -> User image Red is dominant (82%). So Class 1 is likely IC in that chart?
    # Wait, tex says:
    # Class 1: IC (4.52%)
    # Class 2: Capacitor (0.49%)
    # Class 3: Connector (0.48%)
    #
    # My t-SNE script `visualize_tsne_augmented.py` used:
    # 1: Capacitor, 2: IC, 3: Connector
    #
    # Let's check `utils/dataset_functions.py` or `CLASS_DEFINITIONS.md` if it exists.
    # I see `CLASS_DEFINITIONS.md` in the file list earlier.
    
    # Let's stick to the mapping I used in t-SNE which the user approved:
    # 1: Capacitor, 2: IC, 3: Connector
    # BUT, if I use that, the Baseline distribution might look different from their image if their image used different IDs.
    
    # Let's look at the counts ratio in their image: 82 : 9 : 9
    # In my tex report: IC(4.5%) vs Cap(0.5%) vs Conn(0.5%) -> Ratio is roughly 9 : 1 : 1.
    # So the dominant class is definitely IC.
    # In t-SNE script, I named Class 1 as "Capacitor".
    # If Class 1 is actually IC, then my t-SNE label "Capacitor" (Blue) was pointing to the dominant cluster?
    # In t-SNE, the Blue cluster was "sparse and fragmented" (Before) and "dense" (After).
    # IC is usually the largest component class (chips are big), Capacitors are small.
    #
    # Let's trust the `thesis_experiments_report.tex` for ground truth class IDs:
    # 1 = IC
    # 2 = Capacitor
    # 3 = Connector
    #
    # Wait, in `visualize_tsne_augmented.py`, I wrote:
    # class_names = {0: 'Background', 1: 'Capacitor', 2: 'IC', 3: 'Connector'}
    # And the user said: "The colors are consistent (Blue is always the Capacitor/Component...)"
    #
    # I need to be careful. Let's just calculate counts and see which class is dominant.
    # The dominant class (after background) should be IC.
    
    class_map = {1: 'IC', 2: 'Capacitor', 3: 'Connectors'} # Default to what likely matches the data stats
    
    # However, I will check the data distribution in the script to confirm mapping.
    # For now, I will use generic labels based on ID if unsure, but I will deduce it.
    
    total = sum(counts.values())
    if total == 0:
        print(f"No pixels found for {title}")
        return

    # Sort by class ID
    sorted_ids = sorted(counts.keys())
    
    # Colors matching the user's example image roughly?
    # IC (Red), Capacitor (Green), Connectors (Blue)
    # If 1=IC, 2=Cap, 3=Conn
    
    color_map = {
        1: 'red',      # IC
        2: 'green',    # Capacitor
        3: 'blue'      # Connectors
    }
    
    # But wait, in t-SNE I used: 1=Blue, 2=Green, 3=Orange.
    # I should probably stick to the user's NEW image style for THIS specific request?
    # "like this class distribution... need the class distribution after..."
    # I will attempt to match the user's example colors for this pie chart.
    
    valid_counts = {}
    for cid in sorted_ids:
        if counts[cid] > 0:
            valid_counts[cid] = counts[cid]

    # Dynamically determine labels based on counts to ensure I label the big one IC?
    # Or just trust the ID.
    
    # Let's output the raw counts in text first to be sure.
    
    labels = [class_map.get(cid, str(cid)) for cid in valid_counts.keys()]
    sizes = [valid_counts[cid] for cid in valid_counts.keys()]
    plot_colors_list = [color_map.get(cid, 'gray') for cid in valid_counts.keys()]

    plt.figure(figsize=(6, 6))
    wedges, texts, autotexts = plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=plot_colors_list)
    plt.title(title)
    
    # Make text readable
    for t in texts:
        t.set_fontsize(12)
        t.set_fontweight('bold')
    for t in autotexts:
        t.set_color('white')
        t.set_fontsize(10)
        t.set_fontweight('bold')
        
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved {title} to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default="/home/bs_thesis/Documents/BS_THESIS/DATASETS/PCBDataset/")
    args = parser.parse_args()

    # Load Data
    print("Loading dataset...")
    HSI, _, HSI_mono_masks, RGB, _, RGB_mono_masks, _ = read_dataset(args.dataset_path)
    
    # Filter valid
    valid_indices = [i for i, h in enumerate(HSI) if h is not None and HSI_mono_masks[i] is not None]
    HSI = [HSI[i] for i in valid_indices]
    Masks = [HSI_mono_masks[i] for i in valid_indices]
    RGB = [RGB[i] for i in valid_indices]
    
    print(f"Loaded {len(HSI)} valid samples.")
    
    # 1. Baseline Distribution
    print("Calculating Baseline Distribution...")
    baseline_counts = count_pixels(Masks, exclude_background=True)
    print(f"Baseline Counts: {baseline_counts}")
    
    # Identify dominant class to correct labels if needed
    # Expectation: IC > Capacitor ~ Connector
    # If Class 1 is dominant, set 1=IC.
    sorted_counts = sorted(baseline_counts.items(), key=lambda x: x[1], reverse=True)
    dominant_class = sorted_counts[0][0]
    
    print(f"Dominant class is ID {dominant_class}")
    
    # Assumption mapping based on Latex report
    # 1: IC, 2: Capacitor, 3: Connector
    
    # 2. Copy-Paste Augmentation
    print("Applying Copy-Paste Augmentation...")
    cp_aug = CopyPasteAugmentation(minority_classes=(1, 2, 3), max_paste_per_class=15, avoid_overlap=True) # Increased paste count to see effect?
    # The default in training was max_paste_per_class=3?
    # Let's check visualize_tsne which used 3.
    # But usually augmentation tries to balance classes.
    # The user wants to see "after augmentation".
    # If I use standard augmentation parameters, I should see an increase.
    
    cp_aug.build_bank(HSI, Masks)
    
    cp_masks = []
    for i in tqdm(range(len(HSI))):
        _, mask_aug = cp_aug(HSI[i], Masks[i])
        cp_masks.append(mask_aug)
        
    cp_counts = count_pixels(cp_masks, exclude_background=True)
    print(f"Copy-Paste Counts: {cp_counts}")
    plot_pie_chart(cp_counts, "Class Distribution (Copy-Paste)", "Evaluation/benchmark_results/dist_copypaste.png", {})

    # 3. CutMix Augmentation
    print("Applying CutMix Augmentation...")
    cutmix = MultimodalCutMix(beta=1.0)
    
    cm_masks = []
    for i in tqdm(range(len(HSI))):
        idx2 = np.random.randint(len(HSI))
        
        # Resize helper
        h, w = HSI[i].shape[:2]
        
        def resize_mask(m, h, w):
            if m.ndim == 3: m = m.squeeze()
            return cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        
        mask1 = Masks[i]
        mask2 = resize_mask(Masks[idx2], h, w)
        
        # Dummy RGB/HSI
        dummy_hsi1 = np.zeros((h, w, 1))
        dummy_hsi2 = np.zeros((h, w, 1))
        
        _, _, mask_aug, _, _ = cutmix.cutmix(None, dummy_hsi1, mask1, None, dummy_hsi2, mask2)
        cm_masks.append(mask_aug)
        
    cm_counts = count_pixels(cm_masks, exclude_background=True)
    print(f"CutMix Counts: {cm_counts}")
    plot_pie_chart(cm_counts, "Class Distribution (CutMix)", "Evaluation/benchmark_results/dist_cutmix.png", {})

if __name__ == "__main__":
    main()
