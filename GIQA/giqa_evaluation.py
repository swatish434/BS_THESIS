#!/usr/bin/env python3
"""
GIQA: Generated Image Quality Assessment for Augmented Images
Implementation of the toolkit described in GIQA_README.md.
Provides NIQE, BRISQUE, Inception Score, LPIPS, and Artifact Scores.
"""

import os
import json
import argparse
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
from glob import glob

try:
    import piq
except ImportError:
    print("Please install piq: pip install piq")
    exit(1)

# ======================================================================
# Artifact Detector
# ======================================================================
class ArtifactDetector:
    def __init__(self):
        pass
        
    def detect_boundary_artifacts(self, img_np):
        """Detect Copy-Paste specific artifacts (harsh edges)"""
        # img_np is HxWxC float [0,1] or [0,255]
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        # Line density (Hough Lines)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=20, maxLineGap=10)
        line_density = 0 if lines is None else len(lines) / (gray.shape[0]*gray.shape[1]/1000)
        
        # Gradient variance
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(sobelx**2 + sobely**2)
        
        return {
            'line_density': float(line_density),
            'gradient_variance': float(np.var(grad_mag)),
            'edge_density': float(np.sum(edges > 0) / edges.size)
        }
        
    def detect_cutmix_artifacts(self, img_np):
        """Detect CutMix specific artifacts (sharp rectangular transitions)"""
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        # Look for hard horizontal/vertical lines that run across the image
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        horiz_proj = np.sum(np.abs(sobely), axis=1) # High variance at cut
        vert_proj  = np.sum(np.abs(sobelx), axis=0)
        
        return {
            'horizontal_transition': float(np.max(horiz_proj) / (np.mean(horiz_proj)+1e-5)),
            'vertical_transition': float(np.max(vert_proj) / (np.mean(vert_proj)+1e-5)),
            'has_cutmix_artifacts': float(np.max(horiz_proj) > 3*np.mean(horiz_proj))
        }

    def detect_texture_inconsistency(self, img_np):
        """Detect texture mismatches or color shift"""
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
        
        hist_h = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [256], [0, 256])
        
        # Find peaks as rough estimate
        hue_peaks = np.sum(hist_h > hist_h.mean() + hist_h.std())
        sat_peaks = np.sum(hist_s > hist_s.mean() + hist_s.std())
        
        # Local variance
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        mean, std = cv2.meanStdDev(gray)
        
        return {
            'texture_consistency': float(std[0][0]),
            'hue_peaks': int(hue_peaks),
            'saturation_peaks': int(sat_peaks)
        }

# ======================================================================
# GIQA Evaluator Main Class
# ======================================================================
class GIQAEvaluator:
    def __init__(self, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.detector = ArtifactDetector()
        
        # Load necessary PIQ models
        self.brisque_metric = piq.BRISQUELoss().to(self.device)
        # PIQ NIQE is a functional not a module, we just import it
        # LPIPS
        self.lpips_metric = piq.LPIPS().to(self.device)
        
        # Inception model for IS and FID
        from torchvision.models import inception_v3
        self.inception = inception_v3(pretrained=True, transform_input=False).to(self.device)
        self.inception.eval()

    def load_images(self, img_dir):
        files = sorted(glob(os.path.join(img_dir, '*.png')) + glob(os.path.join(img_dir, '*.jpg')))
        images_tensor = []
        images_np = []
        
        trans = transforms.ToTensor()
        for f in files:
            img = Image.open(f).convert('RGB')
            images_np.append(np.array(img))
            images_tensor.append(trans(img))
            
        if not images_tensor:
            return None, None
            
        # Stack to batch (assuming same size)
        batch = torch.stack(images_tensor)
        return batch, images_np

    def compute_inception_score(self, imgs_tensor):
        """Compute Inception Score on a batch of images"""
        # Resize to 299x299 as required by inception
        if imgs_tensor.shape[2] != 299:
            imgs_tensor = torch.nn.functional.interpolate(imgs_tensor, size=(299, 299), mode='bilinear')
        
        imgs_tensor = imgs_tensor.to(self.device)
        with torch.no_grad():
            preds = torch.softmax(self.inception(imgs_tensor), dim=1)
            
        # Marginal distribution
        p_y = torch.mean(preds, dim=0)
        # KL divergence
        kl_d = preds * (torch.log(preds + 1e-16) - torch.log(p_y + 1e-16))
        is_score = torch.exp(torch.mean(torch.sum(kl_d, dim=1))).item()
        return is_score

    def compute_fid(self, real_tensor, fake_tensor):
        if fake_tensor.shape[2] != 299:
            fake_tensor = torch.nn.functional.interpolate(fake_tensor, size=(299, 299), mode='bilinear')
        if real_tensor.shape[2] != 299:
            real_tensor = torch.nn.functional.interpolate(real_tensor, size=(299, 299), mode='bilinear')
            
        self.inception.fc = torch.nn.Identity() # Get features
        with torch.no_grad():
            real_feats = self.inception(real_tensor.to(self.device)).cpu().numpy()
            fake_feats = self.inception(fake_tensor.to(self.device)).cpu().numpy()
        
        mu_r, cov_r = np.mean(real_feats, axis=0), np.cov(real_feats, rowvar=False)
        mu_f, cov_f = np.mean(fake_feats, axis=0), np.cov(fake_feats, rowvar=False)
        
        diff = mu_r - mu_f
        from scipy import linalg
        covmean, _ = linalg.sqrtm(cov_r.dot(cov_f), disp=False)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        fid = diff.dot(diff) + np.trace(cov_r + cov_f - 2 * covmean)
        return float(fid)

    def evaluate_dataset(self, image_dir, aug_type='copypaste', reference_dir=None):
        print(f"\nEvaluating dataset: {image_dir} ({aug_type})")
        batch_tensor, batch_np = self.load_images(image_dir)
        if batch_tensor is None:
            return None
            
        N = len(batch_tensor)
        print(f"Loaded {N} images.")
        
        # 1. NIQE & BRISQUE (No Reference)
        batch_dev = batch_tensor.to(self.device)
        
        brisque_scores = []
        niqe_scores = []
        for i in tqdm(range(N), desc="No-Ref Metrics"):
            img = batch_dev[i:i+1] # 1, C, H, W
            brisque_scores.append(self.brisque_metric(img).item())
            niqe_scores.append(piq.niqe(img).item())
            
        mean_brisque = np.mean(brisque_scores)
        mean_niqe = np.mean(niqe_scores)
        
        # 2. Inception Score
        is_score = self.compute_inception_score(batch_tensor)
        
        # 3. Artifact Score
        art_bound = []
        for np_img in batch_np:
            if aug_type == 'copypaste':
                art_bound.append(self.detector.detect_boundary_artifacts(np_img)['gradient_variance'])
            else:
                art_bound.append(self.detector.detect_cutmix_artifacts(np_img)['horizontal_transition'])
                
        # Normalize artifact arbitrarily for overall score
        mean_artifact = np.mean(art_bound) / 1000.0 if aug_type == 'copypaste' else np.mean(art_bound) / 10.0
        mean_artifact = min(mean_artifact, 1.0)
        
        # 4. LPIPS & FID (Reference Based)
        lpips_score = 0.0
        fid_score = 0.0
        if reference_dir:
            ref_tensor, _ = self.load_images(reference_dir)
            if ref_tensor is not None and len(ref_tensor) == N:
                print("Computing Reference Metrics (FID, LPIPS)...")
                with torch.no_grad():
                    lpips_score = self.lpips_metric(batch_dev, ref_tensor.to(self.device)).item()
                fid_score = self.compute_fid(ref_tensor, batch_tensor)
        
        # Overall Score (Formula from README)
        # Normalization heuristics
        n_niqe = max(0, 100 - (mean_niqe * 10))
        n_brisq = max(0, 100 - (mean_brisque * 2))
        n_is = min(100, is_score * 5)
        n_art = max(0, 100 - (mean_artifact * 100))
        
        overall = 0.25 * n_niqe + 0.25 * n_brisq + 0.25 * n_is + 0.25 * n_art
        
        res = {
            'overall_quality_score': float(overall),
            'metrics': {
                'niqe': float(mean_niqe),
                'brisque': float(mean_brisque),
                'inception_score': float(is_score),
                'artifact_score': float(mean_artifact),
                'lpips': float(lpips_score),
                'fid': float(fid_score)
            }
        }
        
        print(f"  Overall Score: {overall:.2f}")
        print(f"  NIQE: {mean_niqe:.4f}  |  BRISQUE: {mean_brisque:.4f}")
        print(f"  IS: {is_score:.4f}  |  FID: {fid_score:.4f}")
        return res

    def evaluate_single_image(self, img_np, aug_type='copypaste'):
        tensor = transforms.ToTensor()(img_np).unsqueeze(0).to(self.device)
        brisque = self.brisque_metric(tensor).item()
        niqe = piq.niqe(tensor).item()
        
        if aug_type == 'copypaste':
            arts = self.detector.detect_boundary_artifacts(img_np)
            arts['overall_artifact_score'] = min(arts['gradient_variance']/1000.0, 1.0)
        else:
            arts = self.detector.detect_cutmix_artifacts(img_np)
            arts['overall_artifact_score'] = min(arts['horizontal_transition']/10.0, 1.0)
            
        return {
            'niqe': niqe,
            'brisque': brisque,
            'artifacts': arts
        }

    def compare_augmentation_methods(self, results_dict):
        print("\n" + "="*60)
        print("GIQA COMPARISON TABLE")
        print("="*60)
        
        methods = list(results_dict.keys())
        
        header = "| Metric            |" + "".join([f" {m:<10} |" for m in methods])
        print(header)
        print("|" + "-"*19 + "|" + "".join(["-"*12 + "|" for _ in methods]))
        
        metrics_list = [
            ("Overall Score", lambda r: r['overall_quality_score']),
            ("NIQE ↓", lambda r: r['metrics']['niqe']),
            ("BRISQUE ↓", lambda r: r['metrics']['brisque']),
            ("Inception Score ↑", lambda r: r['metrics']['inception_score']),
            ("Artifact Score ↓", lambda r: r['metrics']['artifact_score']),
            ("FID ↓", lambda r: r['metrics']['fid']),
            ("LPIPS ↓", lambda r: r['metrics']['lpips']),
        ]
        
        for name, extractor in metrics_list:
            row = f"| {name:<17} |"
            for m in methods:
                res = results_dict[m]
                if res is None:
                    row += f" {'N/A':<10} |"
                else:
                    row += f" {extractor(res):<10.4f} |" if "Score" not in name and "Overall" not in name else f" {extractor(res):<10.2f} |"
            print(row)
            
        best_method = max(methods, key=lambda x: results_dict[x]['overall_quality_score'] if results_dict[x] else -1)
        return {
            'best_method': best_method,
            'best_score': results_dict[best_method]['overall_quality_score']
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--augmented_dir', type=str, default=None)
    parser.add_argument('--augmentation_type', type=str, default='copypaste')
    parser.add_argument('--reference_dir', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='./giqa_results')
    parser.add_argument('--compare_all', action='store_true', help='Run comparison on ./Results/giqa folders')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    evaluator = GIQAEvaluator()

    if args.compare_all:
        print("Running full comparison on existing Results/giqa data...")
        res = {
            'original': evaluator.evaluate_dataset('./Results/giqa/original', 'none', './Results/giqa/original'),
            'copypaste': evaluator.evaluate_dataset('./Results/giqa/copypaste', 'copypaste', './Results/giqa/original'),
            'cutmix': evaluator.evaluate_dataset('./Results/giqa/cutmix', 'cutmix', './Results/giqa/original')
        }
        comparison = evaluator.compare_augmentation_methods(res)
        
        with open(os.path.join(args.output_dir, 'giqa_complete_results.json'), 'w') as f:
            json.dump(res, f, indent=2)
            
        print(f"\nBest method: {comparison['best_method']}")
        print(f"Best score: {comparison['best_score']:.2f}/100")
        
    elif args.augmented_dir:
        res = evaluator.evaluate_dataset(args.augmented_dir, args.augmentation_type, args.reference_dir)
        
        with open(os.path.join(args.output_dir, f'{args.augmentation_type}_results.json'), 'w') as f:
            json.dump(res, f, indent=2)

if __name__ == '__main__':
    main()
