"""
generative_augmentation.py  v2
────────────────────────────────
Generative In-Painting Pipeline for PCB Augmentation.

Pipeline:
  Step 1 — Select background patch + random mask region + class label
  Step 2 — Stable Diffusion Inpainting conditioned on class label
  Step 3 — Output synthetic image + ground truth mask

Changes in v2 (Option B fixes):
  - region_size_range increased from (32,96) → (80,140) so the inpaint
    region is large enough for SD to synthesize meaningful texture at 512px
  - strength increased from 0.85 → 0.99 (force SD to fully redraw region)
  - num_inference_steps increased from 30 → 50 (better quality generation)
  - Mask dilation added before passing to SD (avoids hard boundary artefacts)
  - Mask padding: inpaint region is padded by 8px on each side when passed
    to SD so the model has context around the region boundary
  - Blending uses Poisson cloning (cv2.seamlessClone) instead of hard alpha
    composite → smoother integration of inpainted region into original image
  - Added per-sample debug save (optional, controlled by DEBUG_SAVE flag)
"""

import os, sys, random, argparse, glob
import numpy as np
import cv2
import torch
from pathlib import Path
from tqdm import tqdm

# ── Set True to save per-sample debug PNGs to ./results/diagnostics/debug/ ──
DEBUG_SAVE = False

# ── Class config ──────────────────────────────────────────────────────────────
CLASS_NAMES   = {0: 'background', 1: 'IC chip', 2: 'connector', 3: 'capacitor'}
CLASS_PROMPTS = {
    1: ("a PCB integrated circuit IC chip, black square component with metal pins, "
        "top-down macro photography, sharp focus, green PCB board background, "
        "photorealistic, 8k"),
    2: ("a PCB connector port, white rectangular plastic connector with gold pins, "
        "top-down macro photography, sharp focus, green PCB board background, "
        "photorealistic, 8k"),
    3: ("a PCB electrolytic capacitor, cylindrical aluminium capacitor component, "
        "top-down macro photography, sharp focus, green PCB board background, "
        "photorealistic, 8k"),
}
NEGATIVE_PROMPT = (
    "blurry, cartoon, drawing, painting, sketch, low quality, watermark, "
    "text, deformed, distorted, oversaturated, unrealistic, 3d render"
)

BACKGROUND_CLASS = 0
MINORITY_CLASSES = [1, 2, 3]


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Background patch + mask selector  (v2: larger regions)
# ─────────────────────────────────────────────────────────────────────────────

def select_inpaint_region(img: np.ndarray, mask: np.ndarray,
                           target_class: int,
                           patch_size: int = 256,
                           region_size_range: tuple = (80, 140)):  # ← v2: was (32,96)
    """
    Find a background region in the image to inpaint a new component into.

    Larger region_size_range ensures the inpaint area maps to a big enough
    canvas at SD's 512×512 resolution for meaningful texture synthesis.

    Returns:
        inpaint_mask : binary mask (255 = region to fill, 0 = keep)
        region_bbox  : (y1, x1, y2, x2) of the inpaint region
    """
    h, w = img.shape[:2]
    rmin, rmax = region_size_range

    inpaint_mask = np.zeros((h, w), dtype=np.uint8)

    for _ in range(50):
        rh = random.randint(rmin, rmax)
        rw = random.randint(rmin, rmax)
        y1 = random.randint(0, h - rh)
        x1 = random.randint(0, w - rw)
        y2, x2 = y1 + rh, x1 + rw

        region = mask[y1:y2, x1:x2]
        # Only place in background areas
        if (region == BACKGROUND_CLASS).mean() >= 0.85:
            inpaint_mask[y1:y2, x1:x2] = 255
            return inpaint_mask, (y1, x1, y2, x2)

    return None, None


def dilate_mask(mask: np.ndarray, kernel_size: int = 8) -> np.ndarray:
    """
    Dilate the inpaint mask by kernel_size pixels.
    Gives SD a little context around the region boundary, reducing
    hard-edge artefacts at the composite boundary.
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.dilate(mask, kernel, iterations=1)


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Stable Diffusion Inpainting  (v2: stronger, more steps)
# ─────────────────────────────────────────────────────────────────────────────

class SDInpainter:
    def __init__(self, model_id="runwayml/stable-diffusion-inpainting",
                 device=None):
        self.device   = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.pipe     = None
        self.model_id = model_id

    def load(self):
        """Lazy load — only load when needed."""
        if self.pipe is not None:
            return
        try:
            from diffusers import StableDiffusionInpaintPipeline
            print(f"  Loading SD inpainting model ({self.model_id})...")
            self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                safety_checker=None,
            ).to(self.device)
            self.pipe.set_progress_bar_config(disable=True)
            # Enable memory-efficient attention if xformers is available
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
                print("  xformers memory-efficient attention enabled")
            except Exception:
                pass
            print(f"  Model loaded on {self.device}")
        except ImportError:
            raise ImportError(
                "diffusers not installed. Run: pip install diffusers transformers accelerate")

    def inpaint(self, img_np: np.ndarray, inpaint_mask: np.ndarray,
                class_id: int,
                strength: float = 0.99,            # ← v2: was 0.85
                guidance_scale: float = 9.0,       # ← v2: was 7.5 (stronger guidance)
                num_inference_steps: int = 50,     # ← v2: was 30
                ) -> np.ndarray:
        """
        Run SD inpainting on a region.

        Args:
            img_np       : (H, W, 3) uint8 RGB image
            inpaint_mask : (H, W) uint8, 255=inpaint, 0=keep
            class_id     : target class to generate
        Returns:
            result       : (H, W, 3) uint8 inpainted image
        """
        from PIL import Image
        self.load()

        h, w = img_np.shape[:2]
        prompt   = CLASS_PROMPTS.get(class_id, CLASS_PROMPTS[1])

        # ── v2: dilate mask before passing to SD ─────────────────────────────
        dilated_mask = dilate_mask(inpaint_mask, kernel_size=8)

        pil_img  = Image.fromarray(img_np).resize((512, 512))
        pil_mask = Image.fromarray(dilated_mask).resize((512, 512))

        result = self.pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            image=pil_img,
            mask_image=pil_mask,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
        ).images[0]

        # Resize back to original resolution
        result = np.array(result.resize((w, h)))
        return result


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Compose output + generate GT mask  (v2: seamless cloning)
# ─────────────────────────────────────────────────────────────────────────────

def compose_result(original_img, original_mask,
                   inpainted_img, inpaint_mask,
                   region_bbox, target_class):
    """
    Blend inpainted region back into original image using Poisson seamless
    cloning (cv2.seamlessClone) for smooth boundary integration.

    Falls back to hard alpha composite if seamlessClone fails.

    Generate ground truth mask by assigning target_class to inpainted region.
    """
    y1, x1, y2, x2 = region_bbox

    result_img  = original_img.copy()
    result_mask = original_mask.copy()

    # ── v2: Poisson seamless cloning ─────────────────────────────────────────
    try:
        # seamlessClone needs:
        #   src    : the patch to insert (full image, same size as dst)
        #   dst    : the background image
        #   mask   : white=blend region, black=ignore  (must be 3-ch)
        #   center : centre of the region in dst coords
        mask_3ch = cv2.merge([inpaint_mask, inpaint_mask, inpaint_mask])
        cy = (y1 + y2) // 2
        cx = (x1 + x2) // 2

        # Convert RGB→BGR for OpenCV
        src_bgr = cv2.cvtColor(inpainted_img,  cv2.COLOR_RGB2BGR)
        dst_bgr = cv2.cvtColor(original_img,   cv2.COLOR_RGB2BGR)

        blended_bgr = cv2.seamlessClone(
            src_bgr, dst_bgr, mask_3ch, (cx, cy), cv2.NORMAL_CLONE)
        result_img = cv2.cvtColor(blended_bgr, cv2.COLOR_BGR2RGB)

    except Exception:
        # Fallback: hard alpha composite (same as v1)
        region_mask = inpaint_mask[y1:y2, x1:x2, np.newaxis] / 255.0
        result_img[y1:y2, x1:x2] = (
            inpainted_img[y1:y2, x1:x2] * region_mask +
            original_img [y1:y2, x1:x2] * (1 - region_mask)
        ).astype(np.uint8)

    # Assign target class to inpainted region in mask
    result_mask[y1:y2, x1:x2][inpaint_mask[y1:y2, x1:x2] > 0] = target_class

    return result_img, result_mask


# ─────────────────────────────────────────────────────────────────────────────
# Generative augmentation wrapper (for use in dataset.py)
# ─────────────────────────────────────────────────────────────────────────────

_inpainter = None

def get_inpainter():
    global _inpainter
    if _inpainter is None:
        _inpainter = SDInpainter()
        _inpainter.load()
    return _inpainter


def generative_augment(img_np: np.ndarray, mask_np: np.ndarray,
                        n_instances: int = 2) -> tuple:
    """
    Apply generative in-painting augmentation.
    Focuses on minority classes with lowest representation.

    Args:
        img_np  : (H, W, 3) float32 [0,1] or uint8
        mask_np : (H, W) int64
    Returns:
        aug_img, aug_mask
    """
    # Convert to uint8 for PIL / OpenCV
    if img_np.dtype != np.uint8:
        img_uint8 = (img_np[:, :, :3] * 255).astype(np.uint8)
    else:
        img_uint8 = img_np[:, :, :3].copy()

    mask_int  = mask_np.copy()
    inpainter = get_inpainter()

    # Focus on rarest class first
    class_counts = [(c, (mask_int == c).sum()) for c in MINORITY_CLASSES]
    class_counts.sort(key=lambda x: x[1])

    for i in range(n_instances):
        target_class = class_counts[0][0]

        inpaint_mask, bbox = select_inpaint_region(
            img_uint8, mask_int, target_class)

        if inpaint_mask is None:
            continue

        inpainted = inpainter.inpaint(img_uint8, inpaint_mask, target_class)

        # ── Optional debug save ───────────────────────────────────────────────
        if DEBUG_SAVE:
            _save_debug(img_uint8, inpaint_mask, inpainted,
                        target_class, instance=i)

        img_uint8, mask_int = compose_result(
            img_uint8, mask_int, inpainted, inpaint_mask, bbox, target_class)

        # Update counts for next iteration
        class_counts = [(c, (mask_int == c).sum()) for c in MINORITY_CLASSES]
        class_counts.sort(key=lambda x: x[1])

    # Convert back to float32
    result_img = img_uint8.astype(np.float32) / 255.0

    # Restore HSI channels if input had more than 3
    if img_np.ndim == 3 and img_np.shape[2] > 3:
        result_img = np.concatenate([result_img, img_np[:, :, 3:]], axis=-1)

    return result_img, mask_int.astype(np.int64)


def _save_debug(orig, mask, inpainted, cls, instance):
    """Save a side-by-side debug PNG for a single inpainting call."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    debug_dir = './results/diagnostics/debug'
    os.makedirs(debug_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(orig);       axes[0].set_title('Original');     axes[0].axis('off')
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title(f'Inpaint mask (class={CLASS_NAMES[cls]})')
    axes[1].axis('off')
    axes[2].imshow(inpainted);  axes[2].set_title('SD output');    axes[2].axis('off')
    plt.tight_layout()

    import time
    fname = os.path.join(debug_dir, f'debug_cls{cls}_inst{instance}_{int(time.time())}.png')
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Integration with augmentations.py dispatcher
# ─────────────────────────────────────────────────────────────────────────────

def apply_generative(img_a: np.ndarray, mask_a: np.ndarray) -> tuple:
    """Drop-in replacement for apply_augmentation with strategy='Generative'."""
    return generative_augment(img_a, mask_a, n_instances=2)


# ─────────────────────────────────────────────────────────────────────────────
# Offline generation mode
# ─────────────────────────────────────────────────────────────────────────────

def generate_offline(config, modality='RGB', n_per_image=3, output_dir=None):
    """Pre-generate augmented samples offline and save to disk."""
    from dataset import get_rgb_patch_ids, get_patch_ids, PCB_Dataset

    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(config.RGB_PATCHES_DIR),
            f'Generative_Augmented_{modality}')
    os.makedirs(output_dir, exist_ok=True)

    if modality == 'RGB':
        train_ids = get_rgb_patch_ids(config.RGB_PATCHES_DIR, 'train')
    else:
        train_ids = get_patch_ids(config.PATCHES_DIR, 'train')

    ds = PCB_Dataset(train_ids, config, modality, 'None', 'train')

    print(f"\n  Generating {n_per_image} augmented samples per image...")
    print(f"  Total: {len(train_ids) * n_per_image} new samples")
    print(f"  Output: {output_dir}")

    generated = 0
    for idx in tqdm(range(len(ds)), desc="Generating"):
        img_t, mask_t = ds[idx]
        img_np  = img_t.permute(1, 2, 0).numpy()
        mask_np = mask_t.numpy()

        for j in range(n_per_image):
            aug_img, aug_mask = generative_augment(img_np, mask_np)
            base      = os.path.join(output_dir, f"Gen_{idx}_{j}")
            img_uint8 = (aug_img[:, :, :3] * 255).astype(np.uint8)
            cv2.imwrite(base + '.png',
                        cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR))
            np.save(base + '_mask.npy', aug_mask)
            generated += 1

    print(f"\n  Generated {generated} samples → {output_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',        type=str, default='test',
                        choices=['generate', 'test'])
    parser.add_argument('--modality',    type=str, default='RGB',
                        choices=['RGB', 'HSI'])
    parser.add_argument('--n_per_image', type=int, default=3)
    parser.add_argument('--debug',       action='store_true',
                        help='Enable per-sample debug PNG saves')
    args = parser.parse_args()

    if args.debug:
        import generative_augmentation as _self
        _self.DEBUG_SAVE = True

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from config import Config
    config = Config()

    if args.mode == 'test':
        print("Testing generative augmentation v2 on one RGB sample...")
        from dataset import get_rgb_patch_ids, PCB_Dataset
        ids = get_rgb_patch_ids(config.RGB_PATCHES_DIR, 'train')
        ds  = PCB_Dataset(ids[:1], config, 'RGB', 'None', 'train')
        img_t, mask_t = ds[0]
        img_np  = img_t.permute(1, 2, 0).numpy()
        mask_np = mask_t.numpy()

        print(f"  Input  — img:{img_np.shape}  mask classes:{np.unique(mask_np)}")
        aug_img, aug_mask = generative_augment(img_np, mask_np)
        print(f"  Output — img:{aug_img.shape}  mask classes:{np.unique(aug_mask)}")

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        CLASS_COLORS = np.array([[0,0,0],[255,165,0],[0,255,255],[255,0,255]])
        os.makedirs('./results/diagnostics', exist_ok=True)

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        axes[0].imshow(img_np[:,:,:3]);         axes[0].set_title('Original');     axes[0].axis('off')
        axes[1].imshow(CLASS_COLORS[mask_np]);  axes[1].set_title('GT mask');      axes[1].axis('off')
        axes[2].imshow(aug_img[:,:,:3]);        axes[2].set_title('Inpainted v2'); axes[2].axis('off')
        axes[3].imshow(CLASS_COLORS[aug_mask]); axes[3].set_title('New mask');     axes[3].axis('off')
        plt.suptitle('Generative Augmentation v2 — single sample test', fontsize=12)
        plt.tight_layout()
        plt.savefig('./results/diagnostics/generative_test_v2.png',
                    dpi=150, bbox_inches='tight')
        print("  Saved: results/diagnostics/generative_test_v2.png")

    elif args.mode == 'generate':
        generate_offline(config, args.modality, args.n_per_image)