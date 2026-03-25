#!/usr/bin/env python3
"""
GIQA-inspired Augmentation Quality Evaluator for PCBVision.

Evaluates the quality of Copy-Paste and CutMix augmented images against
originals using multiple metrics:
  1. SSIM (Structural Similarity) — per-image structural fidelity
  2. FID  (Frechet Inception Distance) — distribution-level realism
  3. LPIPS-style perceptual distance (via VGG features)
  4. Pixel-level statistics (mean, std, histogram correlation)
  5. Edge quality (Canny edge density preservation)

Usage:
    python giqa_evaluator.py --original_dir /path/to/originals \
        --copypaste_dir /path/to/copypaste --cutmix_dir /path/to/cutmix \
        --output_dir ./Results/giqa
"""

import os
import sys
import json
import argparse
import numpy as np
import cv2
from glob import glob
from collections import defaultdict

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from scipy import linalg
from tqdm import tqdm

# ═══════════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_ssim(img1, img2, C1=6.5025, C2=58.5225):
    """Compute SSIM between two grayscale or color images."""
    if img1.ndim == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if img2.ndim == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.GaussianBlur(img1 ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(ssim_map.mean())


def compute_psnr(img1, img2):
    """Peak Signal-to-Noise Ratio between two images."""
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return 100.0
    return 20 * np.log10(255.0 / np.sqrt(mse))


def compute_histogram_correlation(img1, img2):
    """Histogram correlation between two images."""
    if img1.ndim == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if img2.ndim == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    h1 = cv2.calcHist([img1], [0], None, [256], [0, 256]).flatten()
    h2 = cv2.calcHist([img2], [0], None, [256], [0, 256]).flatten()
    h1 = h1 / (h1.sum() + 1e-8)
    h2 = h2 / (h2.sum() + 1e-8)
    return float(cv2.compareHist(
        h1.astype(np.float32), h2.astype(np.float32), cv2.HISTCMP_CORREL))


def compute_edge_density(img):
    """Fraction of edge pixels (Canny)."""
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    edges = cv2.Canny(gray, 50, 150)
    return float(edges.sum() / 255.0 / edges.size)


# ═══════════════════════════════════════════════════════════════════════════════
# VGG Feature Extractor (for FID / Perceptual Distance)
# ═══════════════════════════════════════════════════════════════════════════════

class VGGFeatureExtractor(nn.Module):
    """Extract pool5 features from VGG16 for FID computation."""
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(pretrained=True)
        self.features = vgg.features
        self.avgpool = vgg.avgpool
        self.classifier = nn.Sequential(*list(vgg.classifier.children())[:4])  # up to relu6
        self.eval()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def extract_features(images_dir, model, device, max_images=500):
    """Extract VGG features from images in a directory."""
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    files = sorted(glob(os.path.join(images_dir, '*.png')) +
                   glob(os.path.join(images_dir, '*.jpg')))
    if not files:
        # Try subdirectories
        files = sorted(glob(os.path.join(images_dir, '**', '*.png'), recursive=True) +
                       glob(os.path.join(images_dir, '**', '*.jpg'), recursive=True))
    files = files[:max_images]
    if not files:
        return np.array([])

    features = []
    model.eval()
    with torch.no_grad():
        for f in tqdm(files, desc=f"  Extracting features", leave=False):
            img = cv2.imread(f)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            tensor = transform(img).unsqueeze(0).to(device)
            feat = model(tensor).cpu().numpy().flatten()
            features.append(feat)
    return np.array(features)


def compute_fid(feats1, feats2):
    """Compute FID between two sets of features."""
    if len(feats1) < 2 or len(feats2) < 2:
        return float('nan')
    mu1, sigma1 = feats1.mean(axis=0), np.cov(feats1, rowvar=False)
    mu2, sigma2 = feats2.mean(axis=0), np.cov(feats2, rowvar=False)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fid)


# ═══════════════════════════════════════════════════════════════════════════════
# Main Evaluator Class
# ═══════════════════════════════════════════════════════════════════════════════

class AugmentationQualityEvaluator:
    """Evaluate the quality of augmented images against originals."""

    def __init__(self, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[GIQA] Device: {self.device}")
        self.vgg = VGGFeatureExtractor().to(self.device)
        self.vgg.eval()

    def _load_images(self, img_dir, max_images=200):
        """Load images from a directory."""
        files = sorted(glob(os.path.join(img_dir, '*.png')) +
                       glob(os.path.join(img_dir, '*.jpg')))
        if not files:
            files = sorted(glob(os.path.join(img_dir, '**', '*.png'), recursive=True) +
                           glob(os.path.join(img_dir, '**', '*.jpg'), recursive=True))
        files = files[:max_images]
        images = []
        for f in files:
            img = cv2.imread(f)
            if img is not None:
                images.append((os.path.basename(f), img))
        return images

    def evaluate_augmentation_set(self, original_images_dir, augmented_images_dir,
                                   augmentation_type="Unknown", max_images=200):
        """
        Evaluate a set of augmented images against originals.
        Returns a dict of aggregate metrics.
        """
        print(f"\n{'='*60}")
        print(f"  Evaluating: {augmentation_type}")
        print(f"  Original : {original_images_dir}")
        print(f"  Augmented: {augmented_images_dir}")
        print(f"{'='*60}")

        orig_imgs = self._load_images(original_images_dir, max_images)
        aug_imgs  = self._load_images(augmented_images_dir, max_images)

        if not orig_imgs:
            print(f"[WARN] No original images found in {original_images_dir}")
            return {}
        if not aug_imgs:
            print(f"[WARN] No augmented images found in {augmented_images_dir}")
            return {}

        print(f"  Loaded {len(orig_imgs)} originals, {len(aug_imgs)} augmented")

        # ─── Per-image metrics (paired if possible, otherwise distribution) ───
        # Try to match by filename; if not, compare distributions only
        orig_dict = {name: img for name, img in orig_imgs}
        aug_dict  = {name: img for name, img in aug_imgs}
        paired = set(orig_dict.keys()) & set(aug_dict.keys())

        ssim_scores = []
        psnr_scores = []
        hist_corrs  = []

        if paired:
            print(f"  Found {len(paired)} paired images for per-image metrics")
            for name in tqdm(sorted(paired), desc="  Per-image metrics", leave=False):
                o = orig_dict[name]
                a = aug_dict[name]
                # Resize if needed
                if o.shape != a.shape:
                    a = cv2.resize(a, (o.shape[1], o.shape[0]))
                ssim_scores.append(compute_ssim(o, a))
                psnr_scores.append(compute_psnr(o, a))
                hist_corrs.append(compute_histogram_correlation(o, a))

        # ─── Edge density comparison ──────────────────────────────────────────
        orig_edge = [compute_edge_density(img) for _, img in orig_imgs]
        aug_edge  = [compute_edge_density(img) for _, img in aug_imgs]

        # ─── Distribution-level: FID ──────────────────────────────────────────
        print("  Computing FID (VGG features)...")
        orig_feats = extract_features(original_images_dir, self.vgg, self.device, max_images)
        aug_feats  = extract_features(augmented_images_dir, self.vgg, self.device, max_images)
        fid = compute_fid(orig_feats, aug_feats)

        # ─── Pixel statistics ─────────────────────────────────────────────────
        orig_means = [img.mean() for _, img in orig_imgs]
        orig_stds  = [img.std()  for _, img in orig_imgs]
        aug_means  = [img.mean() for _, img in aug_imgs]
        aug_stds   = [img.std()  for _, img in aug_imgs]

        results = {
            'augmentation_type': augmentation_type,
            'num_original': len(orig_imgs),
            'num_augmented': len(aug_imgs),
            'num_paired': len(paired) if paired else 0,
            'fid': round(fid, 4),
            'ssim_mean': round(np.mean(ssim_scores), 4) if ssim_scores else None,
            'ssim_std':  round(np.std(ssim_scores),  4) if ssim_scores else None,
            'psnr_mean': round(np.mean(psnr_scores), 2) if psnr_scores else None,
            'psnr_std':  round(np.std(psnr_scores),  2) if psnr_scores else None,
            'hist_corr_mean': round(np.mean(hist_corrs), 4) if hist_corrs else None,
            'hist_corr_std':  round(np.std(hist_corrs),  4) if hist_corrs else None,
            'edge_density_orig_mean': round(np.mean(orig_edge), 6),
            'edge_density_aug_mean':  round(np.mean(aug_edge),  6),
            'edge_density_ratio':     round(np.mean(aug_edge) / (np.mean(orig_edge) + 1e-8), 4),
            'pixel_mean_orig': round(np.mean(orig_means), 2),
            'pixel_mean_aug':  round(np.mean(aug_means),  2),
            'pixel_std_orig':  round(np.mean(orig_stds),  2),
            'pixel_std_aug':   round(np.mean(aug_stds),   2),
        }

        # Print summary
        print(f"\n  Results for {augmentation_type}:")
        print(f"    FID:  {results['fid']}")
        if results['ssim_mean'] is not None:
            print(f"    SSIM: {results['ssim_mean']:.4f} ± {results['ssim_std']:.4f}")
            print(f"    PSNR: {results['psnr_mean']:.2f} ± {results['psnr_std']:.2f}")
            print(f"    Hist Correlation: {results['hist_corr_mean']:.4f} ± {results['hist_corr_std']:.4f}")
        print(f"    Edge Density Ratio: {results['edge_density_ratio']:.4f}")
        print(f"    Pixel Mean: {results['pixel_mean_orig']:.2f} (orig) → {results['pixel_mean_aug']:.2f} (aug)")
        print(f"    Pixel Std:  {results['pixel_std_orig']:.2f} (orig) → {results['pixel_std_aug']:.2f} (aug)")

        return results

    def compare_augmentation_methods(self, results_list, output_dir=None):
        """Compare multiple augmentation methods and print a comparison table."""
        print(f"\n{'='*70}")
        print(f"  GIQA Comparison: Augmentation Quality")
        print(f"{'='*70}")

        header = f"{'Metric':<25}"
        for r in results_list:
            header += f"  {r['augmentation_type']:>15}"
        print(header)
        print("-" * len(header))

        metrics = [
            ('FID ↓',               'fid'),
            ('SSIM ↑',              'ssim_mean'),
            ('PSNR ↑',             'psnr_mean'),
            ('Hist Correlation ↑',  'hist_corr_mean'),
            ('Edge Density Ratio',  'edge_density_ratio'),
            ('Pixel Mean (Aug)',    'pixel_mean_aug'),
            ('Pixel Std (Aug)',     'pixel_std_aug'),
        ]

        for label, key in metrics:
            row = f"{label:<25}"
            for r in results_list:
                val = r.get(key)
                if val is None:
                    row += f"  {'N/A':>15}"
                elif isinstance(val, float):
                    row += f"  {val:>15.4f}"
                else:
                    row += f"  {str(val):>15}"
            print(row)

        print()

        # Save comparison
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, 'giqa_comparison.json'), 'w') as f:
                json.dump(results_list, f, indent=2)
            print(f"  Results saved to {output_dir}/giqa_comparison.json")

        return results_list


# ═══════════════════════════════════════════════════════════════════════════════
# Generate augmented samples for evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def generate_augmented_samples(dataset_root, output_dir, num_samples=50, target_size=256):
    """
    Generate CopyPaste and CutMix augmented RGB samples for GIQA evaluation.
    Saves originals and augmented versions to separate directories.
    """
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '03_Core'))
    from utils.augmentation_functions import CopyPasteAugmentation, MultimodalCutMix

    rgb_dir  = os.path.join(dataset_root, 'RGB')
    mask_dir = os.path.join(dataset_root, 'RGB', 'Monoseg')

    orig_out = os.path.join(output_dir, 'original')
    cp_out   = os.path.join(output_dir, 'copypaste')
    cm_out   = os.path.join(output_dir, 'cutmix')
    os.makedirs(orig_out, exist_ok=True)
    os.makedirs(cp_out, exist_ok=True)
    os.makedirs(cm_out, exist_ok=True)

    # Load training images
    train_ids = list(range(1, 39))
    images, masks = [], []
    valid_ids = []
    for sid in train_ids:
        img_path = os.path.join(rgb_dir, f'{sid}.jpg')
        msk_path = os.path.join(mask_dir, f'{sid}.png')
        if os.path.isfile(img_path) and os.path.isfile(msk_path):
            img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            msk = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (target_size, target_size))
            msk = cv2.resize(msk.astype(np.uint16), (target_size, target_size), interpolation=cv2.INTER_NEAREST)
            images.append(img)
            masks.append(msk)
            valid_ids.append(sid)

    print(f"Loaded {len(images)} training images for augmentation")

    # CopyPaste setup
    cp_aug = CopyPasteAugmentation(minority_classes=(1, 2, 3))
    cp_aug.build_bank(images, masks)

    # CutMix setup
    cm_aug = MultimodalCutMix(beta=1.0)

    num_samples = min(num_samples, len(images))

    for i in tqdm(range(num_samples), desc="Generating samples"):
        idx = i % len(images)
        img = images[idx]
        msk = masks[idx]
        fname = f"sample_{i:04d}.png"

        # Save original
        cv2.imwrite(os.path.join(orig_out, fname),
                    cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

        # CopyPaste
        try:
            cp_img, cp_msk = cp_aug(img.copy(), msk.copy())
            cv2.imwrite(os.path.join(cp_out, fname),
                        cv2.cvtColor((cp_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        except Exception as e:
            print(f"  CopyPaste failed for sample {i}: {e}")
            cv2.imwrite(os.path.join(cp_out, fname),
                        cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

        # CutMix
        try:
            idx2 = (idx + 1) % len(images)
            _, cm_img, cm_msk, _, _ = cm_aug.cutmix(None, img.copy(), msk.copy(),
                                                      None, images[idx2].copy(), masks[idx2].copy())
            cv2.imwrite(os.path.join(cm_out, fname),
                        cv2.cvtColor((cm_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        except Exception as e:
            print(f"  CutMix failed for sample {i}: {e}")
            cv2.imwrite(os.path.join(cm_out, fname),
                        cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

    print(f"\nGenerated {num_samples} samples each in:")
    print(f"  Original  → {orig_out}")
    print(f"  CopyPaste → {cp_out}")
    print(f"  CutMix    → {cm_out}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='GIQA Augmentation Quality Evaluator')
    parser.add_argument('--dataset_root', type=str,
                        default='/home/bs_thesis/Documents/BS_THESIS/DATASETS/PCBDataset',
                        help='Path to raw PCBDataset (for generating samples)')
    parser.add_argument('--output_dir', type=str, default='./Results/giqa',
                        help='Output directory for results and generated samples')
    parser.add_argument('--num_samples', type=int, default=38,
                        help='Number of samples to generate')
    parser.add_argument('--skip_generate', action='store_true',
                        help='Skip sample generation, use existing dirs')
    parser.add_argument('--original_dir', type=str, default=None,
                        help='Override: path to original images')
    parser.add_argument('--copypaste_dir', type=str, default=None,
                        help='Override: path to copy-paste augmented images')
    parser.add_argument('--cutmix_dir', type=str, default=None,
                        help='Override: path to cutmix augmented images')
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Generate augmented samples if not skipping
    orig_dir = args.original_dir or os.path.join(output_dir, 'original')
    cp_dir   = args.copypaste_dir or os.path.join(output_dir, 'copypaste')
    cm_dir   = args.cutmix_dir or os.path.join(output_dir, 'cutmix')

    if not args.skip_generate:
        print("Step 1: Generating augmented samples for evaluation...\n")
        generate_augmented_samples(args.dataset_root, output_dir, args.num_samples)
    else:
        print("Step 1: Skipping generation, using existing directories\n")

    # Step 2: Evaluate
    print("\nStep 2: Running GIQA evaluation...\n")
    evaluator = AugmentationQualityEvaluator()

    cp_results = evaluator.evaluate_augmentation_set(
        original_images_dir=orig_dir,
        augmented_images_dir=cp_dir,
        augmentation_type="Copy-Paste"
    )

    cm_results = evaluator.evaluate_augmentation_set(
        original_images_dir=orig_dir,
        augmented_images_dir=cm_dir,
        augmentation_type="CutMix"
    )

    # Step 3: Compare
    print("\nStep 3: Comparing augmentation methods...\n")
    evaluator.compare_augmentation_methods([cp_results, cm_results], output_dir)

    print("\nDone!")


if __name__ == '__main__':
    main()
