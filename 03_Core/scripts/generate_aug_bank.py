"""
Generate Augmentation Bank using SD1.5 LoRA

Takes CutMix patches and generates refined versions using trained LoRA.
Supports both 1:1 refinement and 1:N variational generation.

Usage:
    # Conservative (1:1 refinement)
    python scripts/generate_aug_bank.py --variations 1
    
    # Aggressive (1:5 generation for class imbalance)
    python scripts/generate_aug_bank.py --variations 5
"""

import os
import argparse
import json
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import torch

from diffusers import StableDiffusionInpaintPipeline
from peft import PeftModel


def load_pipeline(lora_path, device='cuda'):
    """Load SD Inpainting pipeline with trained LoRA"""
    print("Loading Stable Diffusion Inpainting pipeline...")
    
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16,
        safety_checker=None
    ).to(device)
    
    # Load LoRA weights
    print(f"Loading LoRA weights from {lora_path}...")
    pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_path)
    
    # Enable memory optimizations
    pipe.enable_attention_slicing()
    
    print("Pipeline ready!")
    return pipe


def quality_check(image, mask, thresholds):
    """Filter out low-quality generations"""
    img_array = np.array(image)
    mask_array = np.array(mask)
    
    # Resize mask to match image dimensions if needed
    if img_array.shape[:2] != mask_array.shape[:2]:
        mask_array = cv2.resize(mask_array, (img_array.shape[1], img_array.shape[0]), 
                                 interpolation=cv2.INTER_NEAREST)
    
    # Get inpainted region
    inpaint_region = img_array[mask_array > 128]
    
    if len(inpaint_region) == 0:
        return False
    
    # Check 1: Sharpness (Laplacian variance)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < thresholds['min_sharpness']:
        return False
    
    # Check 2: Brightness
    mean_brightness = np.mean(inpaint_region)
    if mean_brightness < thresholds['min_brightness'] or mean_brightness > thresholds['max_brightness']:
        return False
    
    # Check 3: Variance (not uniform/empty)
    if np.std(inpaint_region) < thresholds['min_variance']:
        return False
    
    # Check 4: No extreme artifacts
    if np.any(inpaint_region == 0) or np.any(inpaint_region == 255):
        # Pure black or white pixels might be artifacts
        if np.mean(inpaint_region == 0) > 0.1 or np.mean(inpaint_region == 255) > 0.1:
            return False
    
    return True


def generate_variations(pipe, image, mask, prompt, num_variations, strength, args):
    """Generate N variations from one CutMix patch"""
    variations = []
    
    for var_idx in range(num_variations):
        seed = args.seed + var_idx
        generator = torch.Generator(device=pipe.device).manual_seed(seed)
        
        result = pipe(
            prompt=prompt,
            image=image,
            mask_image=mask,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            strength=strength,
            generator=generator
        ).images[0]
        
        variations.append(result)
    
    return variations


def main():
    parser = argparse.ArgumentParser()
    
    # Paths
    parser.add_argument('--cutmix-patches-dir', default='data/cutmix_patches')
    parser.add_argument('--lora-path', default='models/sd_lora_pcb/final')
    parser.add_argument('--output-dir', default='data/aug_bank')
    
    # Generation parameters
    parser.add_argument('--variations', type=int, default=1,
                       help='Number of variations per CutMix patch (1=refinement, 5+=generation)')
    parser.add_argument('--strength', type=float, default=0.7,
                       help='How much to vary from input (0.5=conservative, 0.8=diverse)')
    parser.add_argument('--num-inference-steps', type=int, default=50)
    parser.add_argument('--guidance-scale', type=float, default=7.5)
    
    # Quality filtering
    parser.add_argument('--min-sharpness', type=float, default=50.0)
    parser.add_argument('--min-brightness', type=float, default=30.0)
    parser.add_argument('--max-brightness', type=float, default=225.0)
    parser.add_argument('--min-variance', type=float, default=10.0)
    
    # Misc
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda')
    
    args = parser.parse_args()
    
    # Create output directories
    output_path = Path(args.output_dir)
    classes = ['Capacitor', 'IC', 'Connector']
    for class_name in classes:
        (output_path / class_name).mkdir(parents=True, exist_ok=True)
    
    # Quality thresholds
    thresholds = {
        'min_sharpness': args.min_sharpness,
        'min_brightness': args.min_brightness,
        'max_brightness': args.max_brightness,
        'min_variance': args.min_variance
    }
    
    print("="*60)
    print("SD1.5 LoRA Augmentation Bank Generation")
    print("="*60)
    print(f"CutMix patches: {args.cutmix_patches_dir}")
    print(f"LoRA weights: {args.lora_path}")
    print(f"Output: {args.output_dir}")
    print(f"Variations per patch: {args.variations}")
    print(f"Strength: {args.strength}")
    print(f"Quality filtering: {'Enabled' if args.min_sharpness > 0 else 'Disabled'}")
    print()
    
    # Load pipeline
    pipe = load_pipeline(args.lora_path, args.device)
    
    # Process each class
    stats = {cls: {'total': 0, 'passed': 0, 'failed': 0} for cls in classes}
    
    for class_name in classes:
        print(f"\nProcessing {class_name}...")
        
        class_dir = Path(args.cutmix_patches_dir) / class_name
        if not class_dir.exists():
            print(f"  Warning: {class_dir} not found, skipping")
            continue
        
        # Find all CutMix patches
        image_files = sorted(class_dir.glob("*_image.png"))
        print(f"  Found {len(image_files)} CutMix patches")
        
        output_idx = 0
        
        for img_path in tqdm(image_files, desc=f"  {class_name}"):
            base_name = img_path.stem.replace('_image', '')
            mask_path = class_dir / f"{base_name}_inpaint_mask.png"
            prompt_path = class_dir / f"{base_name}_prompt.txt"
            
            if not mask_path.exists() or not prompt_path.exists():
                print(f"  Warning: Missing mask/prompt for {base_name}, skipping")
                continue
            
            # Load inputs
            image = Image.open(img_path).convert('RGB')
            mask = Image.open(mask_path).convert('L')
            with open(prompt_path, 'r') as f:
                prompt = f.read().strip()
            
            # Generate variations
            variations = generate_variations(
                pipe, image, mask, prompt,
                args.variations, args.strength, args
            )
            
            # Quality filter and save
            for var_idx, variation in enumerate(variations):
                stats[class_name]['total'] += 1
                
                if quality_check(variation, mask, thresholds):
                    # Save to bank
                    save_name = f"{output_idx:05d}.png"
                    save_path = output_path / class_name / save_name
                    variation.save(save_path)
                    
                    stats[class_name]['passed'] += 1
                    output_idx += 1
                else:
                    stats[class_name]['failed'] += 1
        
        print(f"  Generated: {stats[class_name]['passed']} patches")
        print(f"  Filtered out: {stats[class_name]['failed']} patches")
        print(f"  Acceptance rate: {stats[class_name]['passed']/max(stats[class_name]['total'],1)*100:.1f}%")
    
    # Save metadata
    metadata = {
        'cutmix_source': args.cutmix_patches_dir,
        'lora_path': args.lora_path,
        'variations_per_patch': args.variations,
        'strength': args.strength,
        'num_inference_steps': args.num_inference_steps,
        'guidance_scale': args.guidance_scale,
        'quality_thresholds': thresholds,
        'statistics': stats,
        'seed': args.seed
    }
    
    with open(output_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Final summary
    print("\n" + "="*60)
    print("Generation Complete!")
    print("="*60)
    total_generated = sum(s['passed'] for s in stats.values())
    total_filtered = sum(s['failed'] for s in stats.values())
    
    for class_name in classes:
        print(f"{class_name:12s}: {stats[class_name]['passed']:4d} patches")
    
    print(f"\nTotal generated: {total_generated}")
    print(f"Total filtered: {total_filtered}")
    print(f"Overall acceptance: {total_generated/(total_generated+total_filtered)*100:.1f}%")
    print(f"\nSaved to: {output_path}")
    print("="*60)


if __name__ == "__main__":
    main()
