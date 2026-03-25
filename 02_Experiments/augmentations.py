"""
augmentations.py
────────────────
Copy-Paste: matches reference implementation from bs_thesis.py
  - Uses skimage connected components to extract individual instances
  - Builds component bank per class
  - Adaptive paste count: 4 if class missing, 2 if rare, 0 if sufficient
  - Avoids pasting on top of existing minority class regions

CutMix: cut region from PCB B, paste into PCB A at same location
  - Retries to ensure pasted region contains foreground
"""

import numpy as np
import random
import cv2
from dataclasses import dataclass
from typing import Tuple, List

try:
    from skimage.measure import label, regionprops
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

BACKGROUND_CLASS = 0
MINORITY_CLASSES = [1, 2, 3]  # IC, Connector, Capacitor


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ComponentPatch:
    img:      np.ndarray   # (H, W, C) — works for both RGB and HSI
    mask:     np.ndarray   # (H, W)
    class_id: int
    bbox:     Tuple[int, int, int, int]
    area:     int


# ─────────────────────────────────────────────────────────────────────────────
# Component extractor — uses connected components (skimage)
# ─────────────────────────────────────────────────────────────────────────────

class ComponentExtractor:
    def __init__(self, min_area=50, max_area=12000):
        self.min_area = min_area
        self.max_area = max_area

    def extract(self, img: np.ndarray, mask: np.ndarray,
                class_id: int) -> List[ComponentPatch]:
        """
        Extract individual connected component instances of class_id.
        Uses skimage label+regionprops for proper instance separation.
        Falls back to bounding box extraction if skimage not available.
        """
        components = []
        binary = (mask == class_id).astype(np.uint8)

        if SKIMAGE_AVAILABLE:
            labeled = label(binary, connectivity=2)
            regions = regionprops(labeled)
        else:
            # Fallback: treat all pixels of this class as one region
            ys, xs = np.where(binary)
            if len(ys) == 0:
                return components
            regions = [type('R', (), {
                'area': len(ys),
                'bbox': (ys.min(), xs.min(), ys.max()+1, xs.max()+1)
            })()]

        for r in regions:
            if r.area < self.min_area or r.area > self.max_area:
                continue

            if SKIMAGE_AVAILABLE:
                y0, x0, y1, x1 = r.bbox
            else:
                y0, x0, y1, x1 = r.bbox

            if y1 <= y0 or x1 <= x0:
                continue

            img_patch  = img[y0:y1, x0:x1]
            mask_patch = mask[y0:y1, x0:x1]

            if img_patch.size == 0:
                continue

            components.append(ComponentPatch(
                img=img_patch.copy(),
                mask=mask_patch.copy(),
                class_id=class_id,
                bbox=(y0, y1, x0, x1),
                area=r.area
            ))

        return components


# ─────────────────────────────────────────────────────────────────────────────
# Copy-Paste augmentation — matches reference bs_thesis.py
# ─────────────────────────────────────────────────────────────────────────────

class CopyPasteAugmenter:
    def __init__(self, minority_classes=(1, 2, 3), seed=42):
        self.minority_classes = minority_classes
        self.extractor        = ComponentExtractor()
        self.component_bank   = {c: [] for c in minority_classes}
        random.seed(seed)
        np.random.seed(seed)

    def build_bank(self, img: np.ndarray, mask: np.ndarray):
        """Add components from one image to the bank."""
        for cls in self.minority_classes:
            self.component_bank[cls].extend(
                self.extractor.extract(img, mask, cls)
            )

    def _class_ratio(self, mask, cls):
        return np.sum(mask == cls) / mask.size

    def _find_location(self, mask, ph, pw):
        """Find paste location with no existing minority class pixels."""
        H, W = mask.shape
        if ph + 10 >= H or pw + 10 >= W:
            return None
        for _ in range(50):
            y = np.random.randint(5, H - ph - 5)
            x = np.random.randint(5, W - pw - 5)
            region = mask[y:y+ph, x:x+pw]
            if not np.any(np.isin(region, list(self.minority_classes))):
                return y, x
        return None

    def _paste(self, img_t, mask_t, comp, loc):
        """Paste only foreground pixels of the component."""
        y, x   = loc
        ph, pw = comp.mask.shape
        fg     = (comp.mask == comp.class_id)

        # Paste image channels
        if img_t.ndim == 3:
            for c in range(img_t.shape[2]):
                img_t[y:y+ph, x:x+pw, c] = np.where(
                    fg, comp.img[:, :, c], img_t[y:y+ph, x:x+pw, c])
        else:
            img_t[y:y+ph, x:x+pw] = np.where(
                fg, comp.img, img_t[y:y+ph, x:x+pw])

        # Paste mask
        mask_t[y:y+ph, x:x+pw] = np.where(
            fg, comp.class_id, mask_t[y:y+ph, x:x+pw])

    def __call__(self, img: np.ndarray, mask: np.ndarray):
        img_aug  = img.copy()
        mask_aug = mask.copy()

        for cls in self.minority_classes:
            ratio = self._class_ratio(mask_aug, cls)
            # Adaptive paste count — matches reference logic
            if ratio == 0:
                n_paste = 4   # class completely missing → paste 4
            elif ratio < 0.005:
                n_paste = 2   # class very rare → paste 2
            else:
                n_paste = 0   # class sufficient → skip

            for _ in range(n_paste):
                if not self.component_bank[cls]:
                    continue
                comp = random.choice(self.component_bank[cls])
                loc  = self._find_location(mask_aug, *comp.mask.shape)
                if loc:
                    self._paste(img_aug, mask_aug, comp, loc)

        return img_aug, mask_aug


# ─────────────────────────────────────────────────────────────────────────────
# CutMix — cut region from PCB B, paste into PCB A
# ─────────────────────────────────────────────────────────────────────────────

def cutmix(img_a: np.ndarray, mask_a: np.ndarray,
           img_b: np.ndarray, mask_b: np.ndarray,
           min_fg_ratio: float = 0.03) -> tuple:
    """
    Cut a rectangular region from PCB B and paste into PCB A
    at the same spatial location.
    Retries up to 15 times to find a region with foreground in B.
    """
    h, w = img_a.shape[:2]

    if img_b.shape[:2] != (h, w):
        img_b  = cv2.resize(img_b,  (w, h))
        mask_b = cv2.resize(mask_b, (w, h), interpolation=cv2.INTER_NEAREST)

    mixed_img  = img_a.copy()
    mixed_mask = mask_a.copy()

    for attempt in range(15):
        cut_h = random.randint(int(h * 0.2), int(h * 0.6))
        cut_w = random.randint(int(w * 0.2), int(w * 0.6))
        y1    = random.randint(0, h - cut_h)
        x1    = random.randint(0, w - cut_w)
        y2, x2 = y1 + cut_h, x1 + cut_w

        region_mask = mask_b[y1:y2, x1:x2]
        fg_ratio    = (region_mask > BACKGROUND_CLASS).sum() / region_mask.size

        if fg_ratio >= min_fg_ratio or attempt == 14:
            mixed_img [y1:y2, x1:x2] = img_b [y1:y2, x1:x2]
            mixed_mask[y1:y2, x1:x2] = mask_b[y1:y2, x1:x2]
            break

    return mixed_img, mixed_mask


# ─────────────────────────────────────────────────────────────────────────────
# Global augmenter instance (built once per dataset)
# ─────────────────────────────────────────────────────────────────────────────

_augmenter = None

def _get_augmenter(img_ids, config, modality):
    """Build component bank from all training images once."""
    global _augmenter
    if _augmenter is not None:
        return _augmenter

    import os
    _augmenter = CopyPasteAugmenter(minority_classes=(1, 2, 3))

    print("  Building Copy-Paste component bank...")
    for img_id in img_ids:
        try:
            if modality == 'RGB':
                for ext in ['.jpg', '.png']:
                    path = os.path.join(config.RGB_DIR, f"{img_id}{ext}")
                    if os.path.exists(path):
                        import cv2 as _cv2
                        img = _cv2.imread(path)
                        img = _cv2.cvtColor(img, _cv2.COLOR_BGR2RGB)
                        img = _cv2.resize(img, (config.IMAGE_SIZE, config.IMAGE_SIZE))
                        img = img.astype(np.float32) / 255.0
                        mask_path = os.path.join(config.MASK_DIR, f"{img_id}.png")
                        if os.path.exists(mask_path):
                            mask = _cv2.imread(mask_path, _cv2.IMREAD_GRAYSCALE)
                            mask = _cv2.resize(mask, (config.IMAGE_SIZE, config.IMAGE_SIZE),
                                             interpolation=_cv2.INTER_NEAREST)
                            _augmenter.build_bank(img, mask.astype(np.int64))
                        break
        except Exception:
            continue

    bank_sizes = {c: len(_augmenter.component_bank[c])
                  for c in _augmenter.minority_classes}
    print(f"  Bank: IC={bank_sizes[1]} Connector={bank_sizes[2]} Capacitor={bank_sizes[3]}")
    return _augmenter


# ─────────────────────────────────────────────────────────────────────────────
# Public dispatcher
# ─────────────────────────────────────────────────────────────────────────────

def apply_augmentation(img_a: np.ndarray, mask_a: np.ndarray,
                       img_b: np.ndarray, mask_b: np.ndarray,
                       strategy: str) -> tuple:
    if strategy == 'CutMix':
        return cutmix(img_a, mask_a, img_b, mask_b)
    elif strategy == 'Copy-Paste':
        # Build bank from donor on the fly (simple version without persistent bank)
        augmenter = CopyPasteAugmenter(minority_classes=(1, 2, 3))
        augmenter.build_bank(img_b, mask_b)
        return augmenter(img_a, mask_a)
    else:
        return img_a, mask_a


# ─────────────────────────────────────────────────────────────────────────────
# Updated dispatcher — includes Generative
# ─────────────────────────────────────────────────────────────────────────────

def apply_augmentation(img_a: np.ndarray, mask_a: np.ndarray,
                       img_b: np.ndarray, mask_b: np.ndarray,
                       strategy: str) -> tuple:
    if strategy == 'CutMix':
        return cutmix(img_a, mask_a, img_b, mask_b)
    elif strategy == 'Copy-Paste':
        augmenter = CopyPasteAugmenter(minority_classes=(1, 2, 3))
        augmenter.build_bank(img_b, mask_b)
        return augmenter(img_a, mask_a)
    elif strategy == 'Generative':
        from generative_augmentation import apply_generative
        return apply_generative(img_a, mask_a)
    else:
        return img_a, mask_a