#!/usr/bin/env python3
import json
import matplotlib.pyplot as plt
import numpy as np
import os

def main():
    json_path = "./Results/giqa/giqa_comparison.json"
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found.")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    # Insert Original baseline
    original_baseline = {
        "augmentation_type": "Original (GT)",
        "fid": 0.0,
        "ssim_mean": 1.0,
        # PSNR for identical images is inf, capping at 50 for visualization
        "psnr_mean": 50.0, 
        "hist_corr_mean": 1.0,
        "edge_density_ratio": 1.0
    }
    data.insert(0, original_baseline)

    # Extract methods
    methods = [d["augmentation_type"] for d in data]

    # Metrics to plot
    metrics = [
        {"key": "fid", "title": "FID (Lower is Better)", "ylabel": "Score"},
        {"key": "ssim_mean", "title": "SSIM (Higher is Better)", "ylabel": "Index [0,1]"},
        {"key": "psnr_mean", "title": "PSNR (Higher is Better)", "ylabel": "dB"},
        {"key": "hist_corr_mean", "title": "Histogram Correlation", "ylabel": "Correlation"},
        {"key": "edge_density_ratio", "title": "Edge Density Ratio", "ylabel": "Ratio (Aug/Orig)"}
    ]

    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    fig.suptitle('GIQA Comparison: Original vs Copy-Paste vs CutMix', fontsize=16, fontweight='bold', y=1.05)

    # Added a green color for the Original baseline
    colors = ['#2ca02c', '#1f77b4', '#ff7f0e']

    for i, metric in enumerate(metrics):
        ax = axes[i]
        values = [d[metric["key"]] for d in data]
        
        bars = ax.bar(methods, values, color=colors[:len(methods)], width=0.5)
        
        # Add labels on top of bars
        for idx, bar in enumerate(bars):
            yval = bar.get_height()
            
            # Special label for PSNR of Original
            if metric["key"] == "psnr_mean" and methods[idx] == "Original (GT)":
                label = "$\infty$"
            elif yval < 5 and metric["key"] != "fid":
                label = f"{yval:.4f}"
            else:
                label = f"{yval:.1f}"

            ax.text(bar.get_x() + bar.get_width()/2, yval + (0.02 * max(values)), 
                    label, 
                    ha='center', va='bottom', fontsize=10)

        ax.set_title(metric["title"], fontsize=12, pad=10)
        ax.set_ylabel(metric["ylabel"])
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # For SSIM and Hist Corr, set ylim to [0, 1.1]
        if "SSIM" in metric["title"] or "Correlation" in metric["title"]:
            ax.set_ylim(0, 1.1)

    plt.tight_layout()
    output_path = "./Results/giqa/giqa_metrics_plot.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to '{output_path}'")

if __name__ == "__main__":
    main()
