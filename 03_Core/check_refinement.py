import cv2
import os
import glob
import matplotlib.pyplot as plt

layout_dir = "data/synthetic_demo"
refined_dir = "data/synthetic_demo_refined"
output_dir = "data/synthetic_demo_comparison"
os.makedirs(output_dir, exist_ok=True)

layout_images = sorted(glob.glob(os.path.join(layout_dir, "syn_*.png")))

print(f"Generating comparisons for {len(layout_images)} images...")

for layout_path in layout_images:
    filename = os.path.basename(layout_path)
    refined_path = os.path.join(refined_dir, filename)
    
    if not os.path.exists(refined_path):
        continue
        
    img_layout = cv2.imread(layout_path)
    img_refined = cv2.imread(refined_path)
    
    # Concatenate side-by-side
    combined = cv2.hconcat([img_layout, img_refined])
    
    # Save
    save_path = os.path.join(output_dir, f"compare_{filename}")
    cv2.imwrite(save_path, combined)

print(f"Comparisons saved to {output_dir}")
