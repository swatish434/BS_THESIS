"""
giqa_evaluation.py
──────────────────
Evaluates augmented images from all 3 strategies using GIQA:
  - Distance-based (kNN in feature space)
  - GMM-based      (likelihood under Gaussian Mixture)
  - MBC-based      (Maximum Between-Class margin)

Compares:
  1. Real images (baseline)
  2. Copy-Paste augmented
  3. CutMix augmented
  4. No augmentation (control)

Usage:
    python giqa_evaluation.py
"""

import os, sys, random, glob
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from config  import Config
from dataset import PCB_Dataset, get_rgb_patch_ids
from augmentations import apply_augmentation

config   = Config()
DIAG_DIR = './results/giqa'
os.makedirs(DIAG_DIR, exist_ok=True)

N_REAL    = 50   # real images to fit GIQA on
N_AUG     = 50   # augmented images to evaluate
DEVICE    = 'cuda' if torch.cuda.is_available() else 'cpu'


# ─────────────────────────────────────────────────────────────────────────────
# Feature Extractor (InceptionV3)
# ─────────────────────────────────────────────────────────────────────────────

class FeatureExtractor:
    def __init__(self):
        self.model = models.inception_v3(weights='IMAGENET1K_V1',
                                          transform_input=False)
        self.model.fc = nn.Identity()
        self.model.eval().to(DEVICE)
        self.preprocess = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],
                                  [0.229,0.224,0.225])
        ])

    def extract(self, img_np: np.ndarray) -> np.ndarray:
        """img_np: (H,W,3) float32 [0,1] or uint8"""
        if img_np.dtype != np.uint8:
            img_np = (img_np * 255).astype(np.uint8)
        pil = Image.fromarray(img_np)
        t   = self.preprocess(pil).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            f = self.model(t)
        return f.cpu().numpy().flatten()

    def extract_batch(self, imgs, desc=''):
        return np.array([self.extract(im) for im in
                         tqdm(imgs, desc=f'  Features {desc}', leave=False)])


# ─────────────────────────────────────────────────────────────────────────────
# GIQA Methods
# ─────────────────────────────────────────────────────────────────────────────

class DistanceGIQA:
    def __init__(self, k=5):
        self.k  = k
        self.nn = None

    def fit(self, real_features):
        self.nn = NearestNeighbors(n_neighbors=self.k, metric='euclidean')
        self.nn.fit(real_features)

    def score(self, gen_features):
        dists, _ = self.nn.kneighbors(gen_features)
        avg_dist  = dists.mean(axis=1)
        scores    = 1.0 / (1.0 + avg_dist)
        return float(scores.mean()), float(scores.std())


class GMMGiqa:
    def __init__(self, n_components=5, n_components_pca=64):
        self.n_components     = n_components
        self.n_components_pca = n_components_pca
        self.gmm = None
        self.pca = None

    def fit(self, real_features):
        from sklearn.decomposition import PCA
        # Reduce 2048-dim features to 64-dim with PCA first
        # (GMM cannot fit full covariance on 50 samples x 2048 dims)
        n_comp = min(self.n_components_pca, real_features.shape[0] - 1)
        self.pca = PCA(n_components=n_comp, random_state=42)
        reduced  = self.pca.fit_transform(real_features.astype(np.float64))
        self.gmm = GaussianMixture(
            n_components  = self.n_components,
            covariance_type = "diag",   # diagonal avoids singularity
            reg_covar     = 1e-3,         # regularization
            random_state  = 42,
            max_iter      = 200,
        )
        self.gmm.fit(reduced)

    def score(self, gen_features):
        reduced = self.pca.transform(gen_features.astype(np.float64))
        ll      = self.gmm.score_samples(reduced)
        norm    = (ll - ll.min()) / (ll.max() - ll.min() + 1e-8)
        return float(norm.mean()), float(norm.std())


class MBCGiqa:
    def __init__(self, percentile=10):
        self.percentile  = percentile
        self.real_feats  = None

    def fit(self, real_features):
        self.real_feats = real_features

    def score(self, gen_features):
        # Distance to real distribution
        dists_to_real = cdist(gen_features, self.real_feats,
                              metric='euclidean').min(axis=1)
        # Margin: how much closer to real than to each other
        dists_to_gen  = cdist(gen_features, gen_features,
                              metric='euclidean')
        np.fill_diagonal(dists_to_gen, np.inf)
        dists_to_gen_min = dists_to_gen.min(axis=1)

        margin = dists_to_gen_min - dists_to_real
        # Normalize
        norm   = (margin - margin.min()) / (margin.max() - margin.min() + 1e-8)
        return float(norm.mean()), float(norm.std())


# ─────────────────────────────────────────────────────────────────────────────
# Load images
# ─────────────────────────────────────────────────────────────────────────────

def load_rgb_images(ids, config, augmentation, n):
    """Load n RGB images with given augmentation strategy."""
    ds   = PCB_Dataset(ids, config, 'RGB', augmentation, 'train')
    imgs = []
    for i in range(min(n, len(ds))):
        img, _ = ds[i]
        # Convert to (H,W,3) uint8
        arr = (img.permute(1,2,0).numpy() * 255).astype(np.uint8)
        imgs.append(arr)
    return imgs


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*65}")
    print(f"  GIQA Evaluation — Augmentation Quality Assessment")
    print(f"{'='*65}\n")

    # Load IDs
    train_ids = get_rgb_patch_ids(config.RGB_PATCHES_DIR, 'train')
    random.seed(42)
    random.shuffle(train_ids)

    # ── Load image sets ───────────────────────────────────────────────────────
    print("  Loading image sets...")
    real_imgs      = load_rgb_images(train_ids, config, 'None',       N_REAL)
    none_imgs      = load_rgb_images(train_ids, config, 'None',       N_AUG)
    copypaste_imgs = load_rgb_images(train_ids, config, 'Copy-Paste', N_AUG)
    cutmix_imgs    = load_rgb_images(train_ids, config, 'CutMix',     N_AUG)

    print(f"  Real:{len(real_imgs)}  None:{len(none_imgs)}  "
          f"Copy-Paste:{len(copypaste_imgs)}  CutMix:{len(cutmix_imgs)}")

    # ── Extract features ──────────────────────────────────────────────────────
    print("\n  Extracting InceptionV3 features...")
    fx = FeatureExtractor()
    real_feats      = fx.extract_batch(real_imgs,      'real')
    none_feats      = fx.extract_batch(none_imgs,      'none')
    copypaste_feats = fx.extract_batch(copypaste_imgs, 'copy-paste')
    cutmix_feats    = fx.extract_batch(cutmix_imgs,    'cutmix')

    # ── Fit GIQA on real images ───────────────────────────────────────────────
    print("\n  Fitting GIQA scorers on real images...")
    dist_giqa = DistanceGIQA(k=5)
    gmm_giqa  = GMMGiqa(n_components=5)
    mbc_giqa  = MBCGiqa(percentile=10)

    dist_giqa.fit(real_feats)
    gmm_giqa.fit(real_feats)
    mbc_giqa.fit(real_feats)

    # ── Score all strategies ──────────────────────────────────────────────────
    print("\n  Scoring augmentation strategies...")
    strategies = {
        'Real (baseline)': real_feats,
        'No Augmentation': none_feats,
        'Copy-Paste':      copypaste_feats,
        'CutMix':          cutmix_feats,
    }

    results = {}
    for name, feats in strategies.items():
        d_mean, d_std = dist_giqa.score(feats)
        g_mean, g_std = gmm_giqa.score(feats)
        m_mean, m_std = mbc_giqa.score(feats)
        results[name] = {
            'Distance': (d_mean, d_std),
            'GMM':      (g_mean, g_std),
            'MBC':      (m_mean, m_std),
            'Combined': ((d_mean + g_mean + m_mean) / 3,
                         (d_std  + g_std  + m_std)  / 3),
        }

    # ── Print results table ───────────────────────────────────────────────────
    print(f"\n{'='*75}")
    print(f"  GIQA Results (higher = better quality / closer to real distribution)")
    print(f"{'='*75}")
    print(f"  {'Strategy':20s} | {'Distance':>12s} | {'GMM':>12s} | "
          f"{'MBC':>12s} | {'Combined':>12s}")
    print(f"  {'-'*75}")
    for name, r in results.items():
        print(f"  {name:20s} | "
              f"{r['Distance'][0]:.4f}±{r['Distance'][1]:.3f} | "
              f"{r['GMM'][0]:.4f}±{r['GMM'][1]:.3f} | "
              f"{r['MBC'][0]:.4f}±{r['MBC'][1]:.3f} | "
              f"{r['Combined'][0]:.4f}±{r['Combined'][1]:.3f}")
    print(f"{'='*75}\n")

    # ── Plot ──────────────────────────────────────────────────────────────────
    metrics   = ['Distance', 'GMM', 'MBC', 'Combined']
    names     = list(results.keys())
    colors    = ['#3C3489', '#0F6E56', '#993C1D', '#633806']

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    for ax, metric in zip(axes, metrics):
        means = [results[n][metric][0] for n in names]
        stds  = [results[n][metric][1] for n in names]
        bars  = ax.bar(range(len(names)), means, yerr=stds,
                       color=colors, alpha=0.85, capsize=4,
                       edgecolor='white', linewidth=0.5)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=20, ha='right', fontsize=9)
        ax.set_title(f'{metric}-based GIQA', fontsize=11, fontweight='bold')
        ax.set_ylabel('Quality Score (↑ better)', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        for bar, val, std in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + std + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    plt.suptitle('GIQA: Augmentation Quality vs Real PCB Images',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    out_path = os.path.join(DIAG_DIR, 'giqa_results.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Plot saved → {out_path}")

    # Save CSV
    import csv
    csv_path = os.path.join(DIAG_DIR, 'giqa_results.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Strategy','Distance_mean','Distance_std',
                    'GMM_mean','GMM_std','MBC_mean','MBC_std',
                    'Combined_mean','Combined_std'])
        for name, r in results.items():
            w.writerow([name,
                        r['Distance'][0], r['Distance'][1],
                        r['GMM'][0],      r['GMM'][1],
                        r['MBC'][0],      r['MBC'][1],
                        r['Combined'][0], r['Combined'][1]])
    print(f"  CSV saved  → {csv_path}")
    print(f"\n{'='*65}\n")


if __name__ == '__main__':
    main()