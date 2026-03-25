"""
visualize_generative.py
────────────────────────
Visualizes the SD inpainting augmentation pipeline on N random patches.

Saves a grid PNG showing for each sample:
  Col 1 — Original RGB patch
  Col 2 — Ground truth mask (before aug)
  Col 3 — Inpainted result
  Col 4 — New GT mask (after aug)
  Col 5 — Difference overlay (what changed)

Usage:
    python visualize_generative.py --n_samples 4 --modality RGB
    python visualize_generative.py --n_samples 6 --modality HSI
"""

import os, sys, argparse, random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import Config
from generative_augmentation import generative_augment

# ── Colour map for 4 classes ──────────────────────────────────────────────────
CLASS_COLORS = np.array([
    [30,  30,  30],   # 0 background  — dark grey
    [255, 165,  0],   # 1 IC chip     — orange
    [0,   200, 255],  # 2 connector   — cyan
    [220,   0, 220],  # 3 capacitor   — magenta
], dtype=np.uint8)

CLASS_NAMES = ['Background', 'IC chip', 'Connector', 'Capacitor']


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    """Convert integer class mask → RGB colour image."""
    return CLASS_COLORS[mask.clip(0, 3)]


def diff_overlay(img_orig: np.ndarray, img_aug: np.ndarray,
                 mask_orig: np.ndarray, mask_aug: np.ndarray) -> np.ndarray:
    """
    Highlight pixels where the mask changed.
    Changed pixels → bright yellow tint over the augmented image.
    """
    changed = (mask_orig != mask_aug)                      # (H, W) bool
    overlay = img_aug.copy()
    overlay[changed] = (overlay[changed] * 0.4 +
                        np.array([1.0, 1.0, 0.0]) * 0.6)  # yellow tint
    overlay = np.clip(overlay, 0, 1)
    return overlay


def run_visualization(n_samples: int = 4, modality: str = 'RGB',
                      seed: int = 42, out_path: str = None):

    random.seed(seed); np.random.seed(seed)

    config = Config()
    os.makedirs('./results/diagnostics', exist_ok=True)

    # ── Load dataset ──────────────────────────────────────────────────────────
    if modality == 'RGB':
        from dataset import get_rgb_patch_ids, PCB_Dataset
        ids = get_rgb_patch_ids(config.RGB_PATCHES_DIR, 'train')
    else:
        from dataset import get_patch_ids, PCB_Dataset
        ids = get_patch_ids(config.PATCHES_DIR, 'train')

    print(f"  Found {len(ids)} {modality} training patches")
    sample_ids = random.sample(ids, min(n_samples, len(ids)))

    ds = PCB_Dataset(sample_ids, config, modality, 'None', 'train')

    # ── Layout: n_samples rows × 5 cols ──────────────────────────────────────
    n_cols = 5
    fig, axes = plt.subplots(
        n_samples, n_cols,
        figsize=(n_cols * 3.5, n_samples * 3.5),
        squeeze=False
    )

    col_titles = [
        'Original Patch',
        'GT Mask (before)',
        'Inpainted Result',
        'GT Mask (after)',
        'Changed Pixels (Δ)',
    ]
    for col, title in enumerate(col_titles):
        axes[0][col].set_title(title, fontsize=11, fontweight='bold', pad=8)

    print(f"\n  Running generative augmentation on {n_samples} samples...\n")

    for row, (img_t, mask_t) in enumerate(ds):
        img_np  = img_t.permute(1, 2, 0).numpy()   # (H, W, C) float32
        mask_np = mask_t.numpy()                    # (H, W) int64

        # Stats before
        orig_counts = {c: int((mask_np == c).sum()) for c in range(4)}
        print(f"  Sample {row+1}/{n_samples} — "
              f"classes present: {[c for c in range(4) if orig_counts[c] > 0]}")

        # Run augmentation
        aug_img, aug_mask = generative_augment(img_np, mask_np, n_instances=2)

        # Stats after
        aug_counts  = {c: int((aug_mask == c).sum()) for c in range(4)}
        added_px    = {c: aug_counts[c] - orig_counts[c] for c in range(1, 4)}
        added_str   = "  Added → " + "  ".join(
            f"{CLASS_NAMES[c]}: +{added_px[c]}" for c in range(1, 4) if added_px[c] > 0
        )
        print(added_str if any(v > 0 for v in added_px.values())
              else "  No new pixels added (no valid background region found)")

        rgb_orig = img_np[:, :, :3]                 # float32 [0,1]
        rgb_aug  = aug_img[:, :, :3]

        diff     = diff_overlay(rgb_orig, rgb_aug, mask_np, aug_mask)

        # ── Plot ──────────────────────────────────────────────────────────────
        axes[row][0].imshow(rgb_orig)
        axes[row][1].imshow(mask_to_rgb(mask_np))
        axes[row][2].imshow(rgb_aug)
        axes[row][3].imshow(mask_to_rgb(aug_mask))
        axes[row][4].imshow(diff)

        # Row label
        axes[row][0].set_ylabel(f"Sample {row+1}", fontsize=9,
                                 labelpad=4, rotation=90, va='center')

        # Pixel-change count on diff panel
        n_changed = int((mask_np != aug_mask).sum())
        axes[row][4].set_xlabel(f"{n_changed} px changed", fontsize=8,
                                 color='orange')

        for ax in axes[row]:
            ax.set_xticks([]); ax.set_yticks([])

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_patches = [
        mpatches.Patch(color=CLASS_COLORS[c] / 255.0, label=CLASS_NAMES[c])
        for c in range(4)
    ]
    legend_patches.append(
        mpatches.Patch(color=(1.0, 1.0, 0.0), label='Changed region (Δ)')
    )
    fig.legend(handles=legend_patches, loc='lower center',
               ncol=5, fontsize=10,
               bbox_to_anchor=(0.5, 0.01), frameon=True)

    plt.suptitle(
        f'SD Inpainting Augmentation — {modality} | {n_samples} samples',
        fontsize=14, fontweight='bold', y=1.01
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])

    if out_path is None:
        out_path = f'./results/diagnostics/generative_aug_viz_{modality}_{n_samples}samples.png'

    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  ✓ Saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_samples', type=int, default=4,
                        help='Number of patch samples to visualize (default: 4)')
    parser.add_argument('--modality',  type=str, default='RGB',
                        choices=['RGB', 'HSI'],
                        help='RGB or HSI patches (default: RGB)')
    parser.add_argument('--seed',      type=int, default=42,
                        help='Random seed for sample selection (default: 42)')
    parser.add_argument('--out',       type=str, default=None,
                        help='Custom output path (optional)')
    args = parser.parse_args()

    run_visualization(
        n_samples=args.n_samples,
        modality=args.modality,
        seed=args.seed,
        out_path=args.out,
    )