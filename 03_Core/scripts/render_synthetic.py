#!/usr/bin/env python3
"""
Render Synthetic Data using SD-LoRA Refinement (Phase 2)

Refines "Frankenstein" CutMix layouts into photorealistic images using Stable Diffusion + LoRA.
Input: Synthetic Layouts (RGB + Mask)
Output: Refined Synthetic Images (RGB) + Original Masks

Usage:
    python scripts/render_synthetic.py --input_dir data/synthetic_layouts --output_dir data/synthetic_final --strength 0.3
"""

import os
import sys
import argparse
import torch
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from diffusers import StableDiffusionImg2ImgPipeline
from peft import PeftModel

# Project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_pipeline(lora_path, device='cuda'):
    print(f"Loading SD Pipeline with LoRA: {lora_path}")
    
    # Use Img2Img pipeline for full image refinement
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", # Base model matching LoRA 
        torch_dtype=torch.float16,
        safety_checker=None
    ).to(device)
    
    # Load LoRA
    pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_path)
    pipe.to(device)
    
    return pipe

def process_single(pipe, layout_path, output_path, strength=0.3, prompt="pcb board, electronic components, high quality, realistic"):
    # Load layout
    img = Image.open(layout_path).convert("RGB")
    
    # Resize to SD native resolution (512x512) for best quality, then resize back
    w, h = img.size
    img_512 = img.resize((512, 512), Image.LANCZOS)
    
    # Generate
    generator = torch.Generator(device=pipe.device).manual_seed(42)
    
    refined_512 = pipe(
        prompt=prompt,
        image=img_512,
        strength=strength, # Low strength preserves structure (layout), High strength hallucinates
        guidance_scale=7.5,
        num_inference_steps=30,
        generator=generator
    ).images[0]
    
    # Resize back to original
    refined = refined_512.resize((w, h), Image.LANCZOS)
    
    # Save
    refined.save(output_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True, help="Dir containing syn_X.png")
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--lora_path', type=str, default="lora_output/pcb_lora") # Verify path!
    parser.add_argument('--strength', type=float, default=0.35, help="Denoising strength (0.3-0.5 best)")
    parser.add_argument('--prompt', type=str, default="high resolution photo of a pcb circuit board with electronic components, capacitors, connectors, soldering, macro photography, 8k uhd")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check LoRA
    if not os.path.exists(args.lora_path):
        print(f"Error: LoRA path {args.lora_path} not found!")
        # Try finding it
        found_lora = [d for d in os.listdir('.') if 'lora' in d and os.path.isdir(d)]
        print(f"Available dirs: {found_lora}")
        return

    pipe = load_pipeline(args.lora_path)
    
    # Find inputs
    inputs = sorted([f for f in os.listdir(args.input_dir) if f.endswith('.png') and 'vis' not in f])
    print(f"Found {len(inputs)} layouts to refine.")
    
    for filename in tqdm(inputs):
        in_path = os.path.join(args.input_dir, filename)
        out_path = os.path.join(args.output_dir, filename)
        
        # We also need to copy the mask!
        mask_name = filename.replace('.png', '_mask.npy')
        mask_in = os.path.join(args.input_dir, mask_name)
        mask_out = os.path.join(args.output_dir, mask_name)
        
        if os.path.exists(mask_in):
            # Copy mask
            import shutil
            shutil.copy(mask_in, mask_out)
        
        process_single(pipe, in_path, out_path, strength=args.strength, prompt=args.prompt)
        
    print(f"Refinement complete! Saved to {args.output_dir}")

if __name__ == "__main__":
    main()
