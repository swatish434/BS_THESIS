#!/usr/bin/env python3
"""
Generate a GIF visualization of the Synthetic Data Pipeline.

Shows the complete flow:
1. Background image (from dataset)
2. Component patches (from aug_bank)
3. Pasting process (step by step)
4. Final layout + mask overlay

Usage:
    python3 scripts/visualize_pipeline_gif.py
"""

import os
import sys
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from glob import glob
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def add_title(img, title, font_scale=1.5):
    """Add title text to image."""
    img = img.copy()
    h, w = img.shape[:2]
    
    # Add black bar at top
    bar_height = 60
    result = np.zeros((h + bar_height, w, 3), dtype=np.uint8)
    result[bar_height:] = img
    
    # Add text
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(title, font, font_scale, 2)[0]
    text_x = (w - text_size[0]) // 2
    text_y = 40
    cv2.putText(result, title, (text_x, text_y), font, font_scale, (255, 255, 255), 2)
    
    return result

def create_pipeline_gif(output_path="pipeline_visualization.gif"):
    """Generate the complete pipeline visualization GIF."""
    
    frames = []
    target_size = (512, 512)
    
    # ========== FRAME 1: Title ==========
    title_frame = np.zeros((target_size[1] + 60, target_size[0], 3), dtype=np.uint8)
    title_frame = add_title(np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8), 
                           "Synthetic Data Pipeline")
    cv2.putText(title_frame, "CutMix + Edge Blending", (120, 300), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 255, 100), 2)
    frames.append(title_frame)
    
    # ========== FRAME 2: Load Background ==========
    bg_paths = glob("/home/bs_thesis/Documents/BS_THESIS/DATASETS/PCBDataset/RGB/*.jpg")
    if bg_paths:
        bg = cv2.imread(random.choice(bg_paths))
        bg = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
        bg = cv2.resize(bg, target_size)
    else:
        bg = np.random.randint(20, 50, (target_size[1], target_size[0], 3), dtype=np.uint8)
    
    frame_bg = add_title(bg, "Step 1: Background Image")
    frames.append(frame_bg)
    
    # ========== FRAME 3: Component Bank ==========
    bank_dir = "data/aug_bank"
    cap_paths = glob(os.path.join(bank_dir, "Capacitor", "*.png"))[:6]
    conn_paths = glob(os.path.join(bank_dir, "Connector", "*.png"))[:6]
    
    # Create component collage
    collage = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
    collage[:] = (30, 30, 30)  # Dark background
    
    # Add capacitors (top row)
    x_offset = 20
    for i, p in enumerate(cap_paths[:3]):
        comp = cv2.imread(p)
        if comp is not None:
            comp = cv2.cvtColor(comp, cv2.COLOR_BGR2RGB)
            h, w = comp.shape[:2]
            scale = min(100/w, 100/h)
            comp = cv2.resize(comp, (int(w*scale), int(h*scale)))
            ch, cw = comp.shape[:2]
            y_offset = 100
            if y_offset + ch < target_size[1] and x_offset + cw < target_size[0]:
                collage[y_offset:y_offset+ch, x_offset:x_offset+cw] = comp
            x_offset += cw + 30
    
    # Add connectors (bottom row)
    x_offset = 20
    for i, p in enumerate(conn_paths[:3]):
        comp = cv2.imread(p)
        if comp is not None:
            comp = cv2.cvtColor(comp, cv2.COLOR_BGR2RGB)
            h, w = comp.shape[:2]
            scale = min(100/w, 100/h)
            comp = cv2.resize(comp, (int(w*scale), int(h*scale)))
            ch, cw = comp.shape[:2]
            y_offset = 280
            if y_offset + ch < target_size[1] and x_offset + cw < target_size[0]:
                collage[y_offset:y_offset+ch, x_offset:x_offset+cw] = comp
            x_offset += cw + 30
    
    # Labels
    cv2.putText(collage, "Capacitors", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)
    cv2.putText(collage, "Connectors", (20, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
    
    frame_bank = add_title(collage, "Step 2: Component Bank (aug_bank)")
    frames.append(frame_bank)
    
    # ========== FRAMES 4-7: Pasting Process ==========
    layout = bg.copy()
    mask = np.zeros((target_size[1], target_size[0]), dtype=np.uint8)
    
    # Pick components to paste
    all_comps = [(p, 2) for p in cap_paths[:2]] + [(p, 3) for p in conn_paths[:2]]
    random.shuffle(all_comps)
    
    for step, (comp_path, class_id) in enumerate(all_comps[:4]):
        comp = cv2.imread(comp_path)
        if comp is None:
            continue
        comp = cv2.cvtColor(comp, cv2.COLOR_BGR2RGB)
        
        # Random position
        h, w = comp.shape[:2]
        scale = random.uniform(0.5, 1.0)
        comp = cv2.resize(comp, (int(w*scale), int(h*scale)))
        ch, cw = comp.shape[:2]
        
        max_y = target_size[1] - ch - 10
        max_x = target_size[0] - cw - 10
        if max_y < 10 or max_x < 10:
            continue
            
        y = random.randint(10, max_y)
        x = random.randint(10, max_x)
        
        # Create alpha mask for smooth blending
        comp_mask = np.ones((ch, cw), dtype=np.float32)
        kernel = np.ones((5, 5), np.uint8)
        eroded = cv2.erode(comp_mask.astype(np.uint8), kernel, iterations=1)
        alpha = cv2.GaussianBlur(eroded.astype(np.float32), (15, 15), 5.0)
        
        # Paste with blending
        for c in range(3):
            target_region = layout[y:y+ch, x:x+cw, c].astype(np.float32)
            comp_region = comp[:, :, c].astype(np.float32)
            blended = alpha * comp_region + (1 - alpha) * target_region
            layout[y:y+ch, x:x+cw, c] = blended.astype(np.uint8)
        
        # Update mask
        mask[y:y+ch, x:x+cw] = class_id
        
        # Create frame showing this step
        class_name = "Capacitor" if class_id == 2 else "Connector"
        title = f"Step 3.{step+1}: Paste {class_name} (Gaussian Blend)"
        frame_paste = add_title(layout.copy(), title)
        frames.append(frame_paste)
    
    # ========== FRAME 8: Final Layout ==========
    frame_final = add_title(layout, "Step 4: Final Synthetic Layout")
    frames.append(frame_final)
    
    # ========== FRAME 9: Mask Visualization ==========
    # Color the mask
    mask_viz = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
    mask_viz[mask == 0] = (50, 50, 50)    # Background
    mask_viz[mask == 2] = (255, 200, 0)   # Capacitor - Yellow
    mask_viz[mask == 3] = (0, 200, 255)   # Connector - Cyan
    
    frame_mask = add_title(mask_viz, "Step 5: Ground Truth Mask")
    frames.append(frame_mask)
    
    # ========== FRAME 10: Overlay ==========
    overlay = cv2.addWeighted(layout, 0.6, mask_viz, 0.4, 0)
    frame_overlay = add_title(overlay, "Step 6: Layout + Mask Overlay")
    frames.append(frame_overlay)
    
    # ========== FRAME 11: Summary ==========
    summary = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
    summary[:] = (20, 20, 40)
    
    lines = [
        "Pipeline Summary:",
        "",
        "1. Load background from PCBDataset",
        "2. Load components from aug_bank", 
        "3. Paste with Gaussian edge blending",
        "4. Generate aligned mask automatically",
        "",
        "Result: Photorealistic synthetic data",
        "with perfect ground truth labels!"
    ]
    
    y_pos = 80
    for line in lines:
        cv2.putText(summary, line, (30, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (200, 255, 200), 1)
        y_pos += 45
    
    frame_summary = add_title(summary, "Synthetic Data Pipeline Complete!")
    frames.append(frame_summary)
    
    # ========== Save GIF ==========
    print(f"Creating GIF with {len(frames)} frames...")
    
    # Convert to PIL images
    pil_frames = [Image.fromarray(f) for f in frames]
    
    # Save
    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=1500,  # 1.5 seconds per frame
        loop=0
    )
    
    print(f"GIF saved to: {output_path}")
    return output_path

if __name__ == "__main__":
    # Save to artifacts directory
    output_dir = "/home/bs_thesis/.gemini/antigravity/brain/499198d4-56d3-4cee-84d9-5d3502d836be"
    output_path = os.path.join(output_dir, "pipeline_visualization.gif")
    
    create_pipeline_gif(output_path)
    print(f"\nVisualization complete! View at:\n{output_path}")
