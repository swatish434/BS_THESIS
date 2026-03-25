#!/usr/bin/env python3
"""
Evaluate SD-LoRA refined images using GIQA metrics.
Compares:
1. Refined vs Originals (FID for distribution realism)
2. Refined vs Unrefined Layouts (SSIM/PSNR for structural preservation)
"""

import os
import json
import numpy as np
import cv2
from glob import glob
from giqa_evaluator import AugmentationQualityEvaluator, compute_ssim, compute_psnr, compute_histogram_correlation, compute_edge_density, extract_features, compute_fid

def main():
    # Paths
    original_dir = "/home/bs_thesis/Documents/BS_THESIS/PCBVision/02_Experiments/HSRN/Results/giqa/original"
    unrefined_dir = "/home/bs_thesis/Documents/BS_THESIS/PCBVision/01_Data/data/synthetic_demo"
    refined_dir = "/home/bs_thesis/Documents/BS_THESIS/PCBVision/01_Data/data/synthetic_demo_refined"
    output_dir = "/home/bs_thesis/Documents/BS_THESIS/PCBVision/02_Experiments/HSRN/Results/giqa"
    comparison_json = os.path.join(output_dir, "giqa_comparison.json")

    print(f"Evaluating SD-LoRA refined images...")
    evaluator = AugmentationQualityEvaluator()

    # 1. Distribution Comparison (FID)
    print("\nComputing Distribution Metrics (FID against Originals)...")
    orig_feats = extract_features(original_dir, evaluator.vgg, evaluator.device)
    refined_feats = extract_features(refined_dir, evaluator.vgg, evaluator.device)
    fid = compute_fid(orig_feats, refined_feats)
    print(f"  FID: {fid:.4f}")

    # 2. Structural Preservation (Comparison with Unrefined Layouts)
    print("\nComputing Structural Preservation (Refined vs Unrefined)...")
    refined_files = sorted(glob(os.path.join(refined_dir, "syn_*.png")))
    
    ssim_scores = []
    psnr_scores = []
    hist_corrs = []
    edge_densities = []
    orig_edge_densities = []

    paired_count = 0
    for ref_path in refined_files:
        base_name = os.path.basename(ref_path)
        unref_path = os.path.join(unrefined_dir, base_name)
        
        if os.path.exists(unref_path):
            img_ref = cv2.imread(ref_path)
            img_unref = cv2.imread(unref_path)
            
            # Ensure same size
            if img_ref.shape != img_unref.shape:
                img_ref = cv2.resize(img_ref, (img_unref.shape[1], img_unref.shape[0]))
            
            ssim_scores.append(compute_ssim(img_unref, img_ref))
            psnr_scores.append(compute_psnr(img_unref, img_ref))
            hist_corrs.append(compute_histogram_correlation(img_unref, img_ref))
            
            edge_densities.append(compute_edge_density(img_ref))
            orig_edge_densities.append(compute_edge_density(img_unref))
            paired_count += 1

    print(f"  Processed {paired_count} paired samples.")
    
    # Aggregate results for SD-LoRA
    sd_lora_results = {
        "augmentation_type": "SD-LoRA",
        "num_original": len(glob(os.path.join(original_dir, "*.png"))),
        "num_augmented": len(refined_files),
        "num_paired": paired_count,
        "fid": round(fid, 4),
        "ssim_mean": round(np.mean(ssim_scores), 4) if ssim_scores else 0,
        "ssim_std": round(np.std(ssim_scores), 4) if ssim_scores else 0,
        "psnr_mean": round(np.mean(psnr_scores), 2) if psnr_scores else 0,
        "psnr_std": round(np.std(psnr_scores), 2) if psnr_scores else 0,
        "hist_corr_mean": round(np.mean(hist_corrs), 4) if hist_corrs else 0,
        "hist_corr_std": round(np.std(hist_corrs), 4) if hist_corrs else 0,
        "edge_density_orig_mean": round(np.mean(orig_edge_densities), 6) if orig_edge_densities else 0,
        "edge_density_aug_mean": round(np.mean(edge_densities), 6) if edge_densities else 0,
        "edge_density_ratio": round(np.mean(edge_densities) / (np.mean(orig_edge_densities) + 1e-8), 4) if orig_edge_densities else 0,
        "pixel_mean_orig": 40.62,  # Baseline from GIQA comparison
        "pixel_mean_aug": round(np.mean([cv2.imread(f).mean() for f in refined_files]), 2),
        "pixel_std_orig": 41.52,   # Baseline from GIQA comparison
        "pixel_std_aug": round(np.mean([cv2.imread(f).std() for f in refined_files]), 2)
    }

    print("\nSD-LoRA Metrics Summary:")
    for k, v in sd_lora_results.items():
        print(f"  {k}: {v}")

    # Load existing results and append/update
    if os.path.exists(comparison_json):
        with open(comparison_json, 'r') as f:
            all_results = json.load(f)
    else:
        all_results = []
    
    # Remove existing SD-LoRA if present
    all_results = [r for r in all_results if r['augmentation_type'] != "SD-LoRA"]
    all_results.append(sd_lora_results)
    
    with open(comparison_json, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to {comparison_json}")

if __name__ == "__main__":
    main()
