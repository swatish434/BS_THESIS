"""
augmentation_distribution.py
─────────────────────────────
Computes and visualizes class pixel distribution:
  Before augmentation vs After Copy-Paste vs After CutMix
For both RGB and HSI modalities.

Usage:
    python augmentation_distribution.py
"""

import os, glob, random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm

from config  import Config
from dataset import PCB_Dataset, get_patch_ids, get_rgb_patch_ids
from augmentations import apply_augmentation

config      = Config()
DIAG_DIR    = './results/diagnostics'
os.makedirs(DIAG_DIR, exist_ok=True)

CLASS_NAMES  = ['Background', 'IC', 'Connector', 'Capacitor']
CLASS_COLORS = ['#2C2C2A', '#EF9F27', '#1D9E75', '#D4537E']
N_SAMPLES    = 50   # samples to measure per strategy


def compute_distribution(ids, config, modality, augmentation, n=50):
    """Compute mean class pixel distribution over n samples."""
    ds     = PCB_Dataset(ids, config, modality, augmentation, 'train')
    counts = np.zeros(config.NUM_CLASSES, dtype=np.float64)
    actual = min(n, len(ds))

    for i in tqdm(range(actual), desc=f"  {augmentation:12s} {modality}", leave=False):
        _, mask = ds[i]
        for c in range(config.NUM_CLASSES):
            counts[c] += (mask == c).sum().item()

    total = counts.sum()
    return (counts / total * 100) if total > 0 else counts


def plot_comparison(results, modality, save_path):
    """
    Bar chart comparison: before vs after augmentations.
    results = dict of {strategy: array of % per class}
    """
    strategies = list(results.keys())
    x          = np.arange(config.NUM_CLASSES)
    width      = 0.25
    n_strats   = len(strategies)
    offsets    = np.linspace(-(n_strats-1)/2, (n_strats-1)/2, n_strats) * width

    fig, axes  = plt.subplots(1, 2, figsize=(16, 6))

    # ── Plot 1: All classes including background ──────────────────────────────
    ax = axes[0]
    strat_colors = ['#3C3489', '#0F6E56', '#993C1D']
    for i, (strat, dist) in enumerate(results.items()):
        bars = ax.bar(x + offsets[i], dist, width,
                      label=strat, color=strat_colors[i], alpha=0.85,
                      edgecolor='white', linewidth=0.5)
        for bar, val in zip(bars, dist):
            if val > 0.5:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                        f'{val:.1f}%', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, fontsize=11)
    ax.set_ylabel('Pixel percentage (%)', fontsize=11)
    ax.set_title(f'{modality} — All classes (incl. background)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    # ── Plot 2: Minority classes only (zoom in) ───────────────────────────────
    ax2      = axes[1]
    minority = [1, 2, 3]  # IC, Connector, Capacitor
    x2       = np.arange(len(minority))

    for i, (strat, dist) in enumerate(results.items()):
        minority_dist = dist[minority]
        bars = ax2.bar(x2 + offsets[i], minority_dist, width,
                       label=strat, color=strat_colors[i], alpha=0.85,
                       edgecolor='white', linewidth=0.5)
        for bar, val in zip(bars, minority_dist):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f'{val:.2f}%', ha='center', va='bottom', fontsize=8)

    ax2.set_xticks(x2)
    ax2.set_xticklabels([CLASS_NAMES[i] for i in minority], fontsize=11)
    ax2.set_ylabel('Pixel percentage (%)', fontsize=11)
    ax2.set_title(f'{modality} — Minority classes (zoomed)', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3)

    plt.suptitle(f'Class Distribution: Before vs After Augmentation — {modality}',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


def print_table(results, modality):
    """Print numeric comparison table."""
    print(f"\n  {'='*70}")
    print(f"  {modality} — Class distribution comparison (% of pixels)")
    print(f"  {'='*70}")
    print(f"  {'Strategy':15s} | {'Background':>12s} | {'IC':>10s} | {'Connector':>10s} | {'Capacitor':>10s}")
    print(f"  {'-'*70}")
    for strat, dist in results.items():
        print(f"  {strat:15s} | {dist[0]:>11.2f}% | {dist[1]:>9.2f}% | "
              f"{dist[2]:>9.2f}% | {dist[3]:>9.2f}%")
    print(f"  {'='*70}")

    # Change relative to None
    if 'None' in results:
        base = results['None']
        print(f"\n  Change relative to No Augmentation:")
        print(f"  {'Strategy':15s} | {'Background':>12s} | {'IC':>10s} | {'Connector':>10s} | {'Capacitor':>10s}")
        print(f"  {'-'*70}")
        for strat, dist in results.items():
            if strat == 'None': continue
            delta = dist - base
            def fmt(v): return f"{'+'if v>=0 else ''}{v:.2f}%"
            print(f"  {strat:15s} | {fmt(delta[0]):>12s} | {fmt(delta[1]):>10s} | "
                  f"{fmt(delta[2]):>10s} | {fmt(delta[3]):>10s}")
        print(f"  {'='*70}")


# ── Main ───────────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  Augmentation Distribution Analysis")
print(f"{'='*65}")

# ── RGB ───────────────────────────────────────────────────────────────────────
print("\n  [1/2] RGB modality...")
rgb_train = get_rgb_patch_ids(config.RGB_PATCHES_DIR, 'train')
rgb_results = {}
for aug in ['None', 'Copy-Paste', 'CutMix']:
    rgb_results[aug] = compute_distribution(
        rgb_train, config, 'RGB', aug, N_SAMPLES)

print_table(rgb_results, 'RGB')
plot_comparison(rgb_results, 'RGB',
                os.path.join(DIAG_DIR, 'aug_distribution_rgb.png'))

# ── HSI ───────────────────────────────────────────────────────────────────────
print("\n  [2/2] HSI modality...")
hsi_train = get_patch_ids(config.PATCHES_DIR, 'train')
hsi_results = {}
for aug in ['None', 'Copy-Paste', 'CutMix']:
    hsi_results[aug] = compute_distribution(
        hsi_train, config, 'HSI', aug, N_SAMPLES)

print_table(hsi_results, 'HSI')
plot_comparison(hsi_results, 'HSI',
                os.path.join(DIAG_DIR, 'aug_distribution_hsi.png'))

print(f"\n{'='*65}")
print(f"  Done! Saved to {DIAG_DIR}/")
print(f"    aug_distribution_rgb.png")
print(f"    aug_distribution_hsi.png")
print(f"{'='*65}\n")