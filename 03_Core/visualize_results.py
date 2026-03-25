import argparse
import os
import sys
from dataclasses import dataclass

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch


# Allow running as a script from any working directory.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from PCBVision.models import AttU_Net, DeepLabv3_plus, LinkNet, ResUnet, UNET
from PCBVision.utils.dataset_functions import read_dataset
from PCBVision.utils.repro import resolve_device


@dataclass
class ModelSpec:
    name: str
    path: str
    kind: str


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='auto', help='cpu, cuda, or auto')
    p.add_argument('--dataset-path', default="/home/bs_thesis/Documents/BS_THESIS/DATASETS/PCBDataset/")
    p.add_argument('--results-dir', default="Evaluation/benchmark_results")
    p.add_argument('--target-index', type=int, default=2)
    p.add_argument('--rgb-glob', default="RGB_Experiments/results/RGB_*_best.pth")
    p.add_argument('--hsi-glob', default="HSI_Experiments/results/*overlap*best.pth")
    p.add_argument('--num-classes', type=int, default=4)
    return p.parse_args()


def _discover_models(pattern: str, kind: str) -> list[ModelSpec]:
    import glob

    paths = sorted(set(glob.glob(pattern)))
    out: list[ModelSpec] = []
    for p in paths:
        base = os.path.basename(p)
        name = base.replace("_best.pth", "")
        out.append(ModelSpec(name=name, path=p, kind=kind))
    return out


def _build_model_from_filename(filename: str, in_channels: int, out_channels: int) -> torch.nn.Module:
    low = filename.lower()
    if 'deeplab' in low:
        return DeepLabv3_plus(nInputChannels=in_channels, n_classes=out_channels)
    if 'linknet' in low:
        return LinkNet(num_classes=out_channels, num_channels=in_channels)
    if 'resunet' in low:
        return ResUnet(channel=in_channels, out_channel=out_channels)
    if 'attention' in low:
        # Attention U-Net expects args named differently
        return AttU_Net(img_ch=in_channels, output_ch=out_channels)
    # Default
    return UNET(in_channels=in_channels, out_channels=out_channels)


def _rgb_preprocess(img: np.ndarray) -> torch.Tensor:
    # Uses the constants already used in your scripts.
    mean_rgb = np.array([49.378, 41.347, 28.657], dtype=np.float32)
    std_rgb = np.array([7.538, 7.043, 4.648], dtype=np.float32)

    img_resized = cv2.resize(img, (640, 640), interpolation=cv2.INTER_NEAREST)
    img_norm = (img_resized.astype(np.float32) - mean_rgb) / std_rgb
    return torch.from_numpy(img_norm.transpose(2, 0, 1)).float().unsqueeze(0)


def _hsi_overlap_patch(hsi_full: np.ndarray, hsi_mask: np.ndarray, target_size: int = 256):
    # Training used 214 channels from a 224-band cube (slice out first 10 bands).
    if hsi_full.shape[2] >= 214 + 10:
        hsi_full = hsi_full[:, :, 10:]

    h, w, _ = hsi_full.shape
    half = target_size // 2
    cy, cx = h // 2, w // 2
    y0, x0 = max(0, cy - half), max(0, cx - half)
    y1, x1 = y0 + target_size, x0 + target_size

    patch = hsi_full[y0:y1, x0:x1, :]
    gt = hsi_mask[y0:y1, x0:x1]
    return patch, gt


def main(args=None):
    if args is None:
        args = parse_args()

    device = resolve_device(args.device)
    os.makedirs(args.results_dir, exist_ok=True)
    print(f"Using device: {device}")

    HSI, _, HSI_mono_masks, RGB, _, RGB_masks, _ = read_dataset(args.dataset_path)
    idx = args.target_index
    if idx < 0 or idx >= len(RGB) or RGB[idx] is None:
        raise ValueError(f"Invalid target index {idx}")

    rgb_img = RGB[idx]
    rgb_gt = RGB_masks[idx]

    rgb_models = _discover_models(args.rgb_glob, kind='rgb')
    if not rgb_models:
        print(f"No RGB models matched: {args.rgb_glob}")

    hsi_models = _discover_models(args.hsi_glob, kind='hsi')

    # Precompute inputs
    rgb_tensor = _rgb_preprocess(rgb_img).to(device)

    hsi_patch = None
    hsi_gt_patch = None
    if idx < len(HSI) and HSI[idx] is not None and HSI_mono_masks[idx] is not None:
        try:
            hsi_patch, hsi_gt_patch = _hsi_overlap_patch(HSI[idx], HSI_mono_masks[idx])
        except Exception:
            hsi_patch = None
            hsi_gt_patch = None

    with torch.inference_mode():
        # --- RGB loop ---
        for spec in rgb_models:
            model = _build_model_from_filename(spec.name, in_channels=3, out_channels=args.num_classes).to(device)
            state = torch.load(spec.path, map_location=device)
            model.load_state_dict(state)
            model.eval()

            out = model(rgb_tensor)
            pred = torch.argmax(out, dim=1).cpu().numpy()[0].astype(np.uint8)
            pred_orig = cv2.resize(pred, (rgb_img.shape[1], rgb_img.shape[0]), interpolation=cv2.INTER_NEAREST)

            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            axes[0].imshow(rgb_img)
            axes[0].set_title("RGB")
            axes[0].axis('off')

            axes[1].imshow(rgb_gt, interpolation='nearest', vmin=0, vmax=args.num_classes - 1)
            axes[1].set_title("GT")
            axes[1].axis('off')

            axes[2].imshow(pred_orig, interpolation='nearest', vmin=0, vmax=args.num_classes - 1)
            axes[2].set_title(spec.name)
            axes[2].axis('off')

            out_path = os.path.join(args.results_dir, f"viz_{spec.name}_idx{idx}.png")
            fig.tight_layout()
            fig.savefig(out_path, dpi=150)
            plt.close(fig)

        # --- HSI loop (optional) ---
        if hsi_models and hsi_patch is not None and hsi_gt_patch is not None:
            for spec in hsi_models:
                # Overlap models are 214-channel.
                model = _build_model_from_filename(spec.name, in_channels=hsi_patch.shape[2], out_channels=args.num_classes).to(device)
                state = torch.load(spec.path, map_location=device)
                model.load_state_dict(state)
                model.eval()

                x = torch.from_numpy(hsi_patch.astype(np.float32)).permute(2, 0, 1).unsqueeze(0).to(device)
                out = model(x)
                pred = torch.argmax(out, dim=1).cpu().numpy()[0].astype(np.uint8)

                # Pseudo RGB (safe indices)
                b = [min(i, hsi_patch.shape[2] - 1) for i in (29, 53, 77)]
                pseudo = hsi_patch[:, :, b].astype(np.float32)
                pseudo = (pseudo - pseudo.min()) / (pseudo.max() - pseudo.min() + 1e-8)

                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                axes[0].imshow(pseudo)
                axes[0].set_title("HSI pseudo-RGB")
                axes[0].axis('off')

                axes[1].imshow(hsi_gt_patch, interpolation='nearest', vmin=0, vmax=args.num_classes - 1)
                axes[1].set_title("GT patch")
                axes[1].axis('off')

                axes[2].imshow(pred, interpolation='nearest', vmin=0, vmax=args.num_classes - 1)
                axes[2].set_title(spec.name)
                axes[2].axis('off')

                out_path = os.path.join(args.results_dir, f"viz_{spec.name}_idx{idx}.png")
                fig.tight_layout()
                fig.savefig(out_path, dpi=150)
                plt.close(fig)
        elif hsi_models:
            print("HSI models found, but HSI sample/mask not available for this index.")


if __name__ == '__main__':
    main(parse_args())
