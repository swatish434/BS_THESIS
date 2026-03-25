
import numpy as np
import cv2
import random
from dataclasses import dataclass
from typing import Tuple, List, Dict
from skimage.measure import label, regionprops

# ============================================================
# Data structure
# ============================================================

@dataclass
class ComponentPatch:
    data: np.ndarray # Can be RGB (H,W,3) or HSI (H,W,C)
    mask: np.ndarray
    class_id: int
    bbox: Tuple[int, int, int, int]
    area: int

# ============================================================
# Component extractor
# ============================================================

class ComponentExtractor:
    def __init__(self, min_area=50, max_area=10000):
        self.min_area = min_area
        self.max_area = max_area

    def extract(self, image, mask, class_id):
        # image can be HSI or RGB. Shape (H, W, C)
        binary = (mask == class_id).astype(np.uint8)
        labeled = label(binary, connectivity=2)

        components = []
        for r in regionprops(labeled):
            if r.area < self.min_area or r.area > self.max_area:
                continue

            y_min, x_min, y_max, x_max = r.bbox
            if y_max <= y_min or x_max <= x_min:
                continue

            # Handle 2D image (rare, usually (H,W,C)) vs 3D
            if image.ndim == 3:
                patch = image[y_min:y_max, x_min:x_max, :].copy()
            else:
                patch = image[y_min:y_max, x_min:x_max].copy()
                
            if patch.size == 0:
                continue

            components.append(
                ComponentPatch(
                    data=patch,
                    mask=mask[y_min:y_max, x_min:x_max].copy(),
                    class_id=class_id,
                    bbox=(y_min, y_max, x_min, x_max),
                    area=r.area
                )
            )
        return components

# ============================================================
# Copy-Paste Augmentation
# ============================================================

class CopyPasteAugmentation:
    def __init__(
        self,
        minority_classes=(1, 2, 3), # Assuming 0 is background
        max_paste_per_class=3,
        avoid_overlap=True,
        random_seed=42
    ):
        self.minority_classes = minority_classes
        self.max_paste_per_class = max_paste_per_class
        self.avoid_overlap = avoid_overlap

        self.extractor = ComponentExtractor()
        self.component_bank = {c: [] for c in minority_classes}

        random.seed(random_seed)
        np.random.seed(random_seed)

    def build_bank(self, images, masks):
        """
        Builds the component bank from a dataset of images and masks.
        """
        print(f"Building Copy-Paste Component Bank from {len(images)} images...")
        count = 0
        for img, mask in zip(images, masks):
            # Ensure mask is suitable (H,W)
            # If mask is (H,W,1) squeeze it?
            if mask.ndim == 3:
                mask = mask.squeeze()
                
            for cls in self.minority_classes:
                extracted = self.extractor.extract(img, mask, cls)
                self.component_bank[cls].extend(extracted)
                count += len(extracted)
        print(f"Bank built with {count} components.")

    def _find_location(self, mask, ph, pw, max_attempts=50):
        H, W = mask.shape
        for _ in range(max_attempts):
            # Ensure patch fits
            if H - ph - 5 <= 5 or W - pw - 5 <= 5:
                 return None
                 
            y0 = np.random.randint(5, H - ph - 5)
            x0 = np.random.randint(5, W - pw - 5)

            if self.avoid_overlap:
                region = mask[y0:y0+ph, x0:x0+pw]
                # If region contains ANY class (non-background 0), skip? 
                # Or just check if it overlaps with minority classes?
                # User code: np.any(np.isin(region, self.minority_classes))
                if np.any(np.isin(region, self.minority_classes)):
                    continue

            return y0, x0
        return None

    def _paste(self, image_t, mask_t, comp, loc, scale_jitter=True, edge_blend=True):
        """
        Paste a component patch onto the target image and mask.
        
        Args:
            image_t: Target image
            mask_t: Target mask
            comp: ComponentPatch to paste
            loc: (y0, x0) location to paste at
            scale_jitter: If True, randomly scale the component ±10%
            edge_blend: If True, apply Gaussian blur to edges for smoother blending
        """
        y0, x0 = loc
        comp_data = comp.data.copy()
        comp_mask_binary = (comp.mask == comp.class_id)
        
        # Scale Jittering
        if scale_jitter:
            scale = np.random.uniform(0.9, 1.1)  # ±10% scale variation
            ph_orig, pw_orig = comp.mask.shape
            ph_new = max(1, int(ph_orig * scale))
            pw_new = max(1, int(pw_orig * scale))
            
            # Resize component data
            if comp_data.ndim == 3:  # RGB or HSI
                comp_data = cv2.resize(comp_data, (pw_new, ph_new), interpolation=cv2.INTER_LINEAR)
            else:  # 2D
                comp_data = cv2.resize(comp_data, (pw_new, ph_new), interpolation=cv2.INTER_LINEAR)
            
            # Resize mask (use NEAREST to preserve class labels)
            comp_mask_resized = cv2.resize(
                comp.mask.astype(np.uint8), 
                (pw_new, ph_new), 
                interpolation=cv2.INTER_NEAREST
            )
            comp_mask_binary = (comp_mask_resized == comp.class_id)
        else:
            comp_mask_resized = comp.mask
        
        ph, pw = comp_mask_binary.shape
        
        # Edge Blending (optional smooth transition)
        if edge_blend:
            # Create a soft alpha mask using Gaussian blur
            alpha_mask = comp_mask_binary.astype(np.float32)
            alpha_mask = cv2.GaussianBlur(alpha_mask, (5, 5), 1.0)
            alpha_mask = np.clip(alpha_mask, 0, 1)  # Ensure values in [0, 1]
        else:
            alpha_mask = comp_mask_binary.astype(np.float32)

        # Check bounds
        if y0 + ph > image_t.shape[0] or x0 + pw > image_t.shape[1]:
            return  # Skip if out of bounds
        
        # Paste Image Data with alpha blending (Vectorized for high-dimensional data)
        if image_t.ndim == 3:
            target_region = image_t[y0:y0+ph, x0:x0+pw, :].astype(np.float32)
            comp_region = comp_data.astype(np.float32)
            
            # Broadcast alpha mask to (ph, pw, 1)
            alpha_broadcast = alpha_mask[:, :, np.newaxis]
            
            blended = alpha_broadcast * comp_region + (1 - alpha_broadcast) * target_region
            image_t[y0:y0+ph, x0:x0+pw, :] = blended.astype(image_t.dtype)
        else:
            target_region = image_t[y0:y0+ph, x0:x0+pw].astype(np.float32)
            comp_region = comp_data.astype(np.float32)
            blended = alpha_mask * comp_region + (1 - alpha_mask) * target_region
            image_t[y0:y0+ph, x0:x0+pw] = blended.astype(image_t.dtype)

        # Paste Mask (hard assignment for ground truth)
        mask_t[y0:y0+ph, x0:x0+pw] = np.where(
            comp_mask_binary,
            comp.class_id,
            mask_t[y0:y0+ph, x0:x0+pw]
        )

    def __call__(self, image, mask):
        # Expects image (H, W, C) and mask (H, W) or (H, W, 1)
        image_aug = image.copy()
        mask_aug = mask.copy()
        
        if mask_aug.ndim == 3:
            mask_aug = mask_aug.squeeze()

        for cls in self.minority_classes:
            if not self.component_bank[cls]:
                continue
                
            num_pastes = np.random.randint(1, self.max_paste_per_class + 1)
            for _ in range(num_pastes):
                comp = random.choice(self.component_bank[cls])
                
                # Check if fits (sanity check, usually bank components fit if from same dataset resolution)
                # But if we did resizing differently, might be issue.
                
                loc = self._find_location(mask_aug, *comp.mask.shape)
                if loc:
                    self._paste(image_aug, mask_aug, comp, loc)
                    
        return image_aug, mask_aug


# ============================================================
# Multimodal CutMix Augmentation
# ============================================================

class MultimodalCutMix:
    """
    CutMix augmentation for paired RGB-HSI data with synchronized masks
    """
    def __init__(self, beta=1.0):
        """
        Args:
            beta: Beta distribution parameter (controls mix ratio)
                  beta=1.0: uniform distribution
                  beta=0.5: more extreme cuts
        """
        self.beta = beta
    
    def cutmix(self, rgb1, hsi1, mask1, rgb2, hsi2, mask2):
        """
        Apply CutMix to paired RGB-HSI data.
        Handles cases where rgb or hsi might be None (single modality).
        
        Args:
            rgb1, hsi1, mask1: First PCB (RGB image, HSI cube, segmentation mask)
            rgb2, hsi2, mask2: Second PCB
        
        Returns:
            rgb_mixed, hsi_mixed, mask_mixed, cut_bbox, lambda_mix
        """
        # Determine shape from whatever is available
        if rgb1 is not None:
            h, w = rgb1.shape[:2]
        elif hsi1 is not None:
            h, w = hsi1.shape[:2]
        else:
            raise ValueError("At least one of rgb1 or hsi1 must be provided")

        # Sample mixing ratio from Beta distribution
        lambda_mix = np.random.beta(self.beta, self.beta)
        
        # Calculate cut size (square root for area-based mixing)
        cut_ratio = np.sqrt(1 - lambda_mix)
        cut_h = int(h * cut_ratio)
        cut_w = int(w * cut_ratio)
        
        # Random center point for the cut
        cx = np.random.randint(w)
        cy = np.random.randint(h)
        
        # Bounding box (ensure within image bounds)
        x1 = np.clip(cx - cut_w // 2, 0, w)
        y1 = np.clip(cy - cut_h // 2, 0, h)
        x2 = np.clip(cx + cut_w // 2, 0, w)
        y2 = np.clip(cy + cut_h // 2, 0, h)
        
        # Apply cut to RGB
        rgb_mixed = None
        if rgb1 is not None and rgb2 is not None:
            rgb_mixed = rgb1.copy()
            rgb_mixed[y1:y2, x1:x2] = rgb2[y1:y2, x1:x2]
        
        # Apply same cut to HSI (synchronized)
        hsi_mixed = None
        if hsi1 is not None and hsi2 is not None:
            hsi_mixed = hsi1.copy()
            hsi_mixed[y1:y2, x1:x2, :] = hsi2[y1:y2, x1:x2, :]
        
        # Apply same cut to mask
        mask_mixed = None
        if mask1 is not None and mask2 is not None:
            mask_mixed = mask1.copy()
            mask_mixed[y1:y2, x1:x2] = mask2[y1:y2, x1:x2]
        
        # Calculate actual mixing ratio (may differ from lambda_mix due to clipping)
        actual_lambda = 1 - ((x2 - x1) * (y2 - y1)) / (h * w)
        
        return rgb_mixed, hsi_mixed, mask_mixed, (x1, y1, x2, y2), actual_lambda


# ============================================================
# SD-LoRA Augmentation (Pre-generated Patch Bank)
# ============================================================

class SDLoRAAugmentation:
    """
    Augmentation using pre-generated SD-LoRA refined patches.
    
    Loads patches from the aug_bank directory and applies them
    to training images using Copy-Paste style integration.
    
    Unlike CopyPasteAugmentation which extracts components on-the-fly,
    this uses pre-refined patches generated by Stable Diffusion + LoRA.
    """
    
    def __init__(
        self,
        aug_bank_path='data/aug_bank',
        minority_classes=(2, 3),  # Capacitor=2, Connector=3 (based on corrected mapping)
        class_names={2: 'Capacitor', 3: 'Connector'},
        max_paste_per_class=2,
        paste_probability=0.5,
        patch_size=(128, 128),  # Target size when pasting onto 640x640 images
        random_seed=42
    ):
        self.aug_bank_path = aug_bank_path
        self.minority_classes = minority_classes
        self.class_names = class_names
        self.max_paste_per_class = max_paste_per_class
        self.paste_probability = paste_probability
        self.patch_size = patch_size
        
        self.patch_bank = {c: [] for c in minority_classes}
        
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        self._load_bank()
    
    def _load_bank(self):
        """Load pre-generated patches from aug_bank directory."""
        import os
        from PIL import Image as PILImage
        
        total_loaded = 0
        for cls_id in self.minority_classes:
            class_name = self.class_names.get(cls_id, f'class_{cls_id}')
            class_dir = os.path.join(self.aug_bank_path, class_name)
            
            if not os.path.exists(class_dir):
                print(f"Warning: {class_dir} not found, skipping")
                continue
            
            patch_files = sorted([f for f in os.listdir(class_dir) if f.endswith('.png')])
            
            for f in patch_files:
                try:
                    img = PILImage.open(os.path.join(class_dir, f)).convert('RGB')
                    self.patch_bank[cls_id].append(np.array(img))
                    total_loaded += 1
                except Exception as e:
                    print(f"Failed to load {f}: {e}")
            
            print(f"  Loaded {len(self.patch_bank[cls_id])} patches for {class_name}")
        
        print(f"SD-LoRA Aug Bank loaded: {total_loaded} total patches")
    
    def _find_location(self, mask, ph, pw, max_attempts=50):
        """Find a valid location to paste the patch (avoid minority class regions)."""
        H, W = mask.shape
        for _ in range(max_attempts):
            if H - ph - 5 <= 5 or W - pw - 5 <= 5:
                return None
            
            y0 = np.random.randint(5, H - ph - 5)
            x0 = np.random.randint(5, W - pw - 5)
            
            # Check if region overlaps with existing minority classes
            region = mask[y0:y0+ph, x0:x0+pw]
            if np.any(np.isin(region, self.minority_classes)):
                continue
            
            return y0, x0
        return None
    
    def apply(self, image, mask):
        """
        Apply SD-LoRA augmentation to an image.
        
        Args:
            image: RGB image (H, W, 3)
            mask: Segmentation mask (H, W)
        
        Returns:
            Augmented image and mask
        """
        if np.random.rand() > self.paste_probability:
            return image, mask  # No augmentation
        
        image = image.copy()
        mask = mask.copy()
        
        # For each minority class, paste some patches
        for cls_id in self.minority_classes:
            if not self.patch_bank[cls_id]:
                continue
            
            num_paste = np.random.randint(0, self.max_paste_per_class + 1)
            
            for _ in range(num_paste):
                # Random patch from bank
                patch_idx = np.random.randint(len(self.patch_bank[cls_id]))
                patch = self.patch_bank[cls_id][patch_idx].copy()
                
                # Resize patch to target size
                ph, pw = self.patch_size
                patch_resized = cv2.resize(patch, (pw, ph), interpolation=cv2.INTER_LINEAR)
                
                # Find location
                loc = self._find_location(mask, ph, pw)
                if loc is None:
                    continue
                
                y0, x0 = loc
                
                # Create a soft circular mask for blending
                center = (pw // 2, ph // 2)
                radius = min(ph, pw) // 2 - 5
                alpha_mask = np.zeros((ph, pw), dtype=np.float32)
                cv2.circle(alpha_mask, center, radius, 1.0, -1)
                alpha_mask = cv2.GaussianBlur(alpha_mask, (15, 15), 5.0)
                
                # Paste with alpha blending
                for c in range(3):
                    image[y0:y0+ph, x0:x0+pw, c] = (
                        alpha_mask * patch_resized[:, :, c] +
                        (1 - alpha_mask) * image[y0:y0+ph, x0:x0+pw, c]
                    ).astype(np.uint8)
                
                # Update mask in pasted region (set to the class ID)
                # Use a smaller region for mask to avoid edge artifacts
                inner_margin = 10
                mask[y0+inner_margin:y0+ph-inner_margin, 
                     x0+inner_margin:x0+pw-inner_margin] = cls_id
        
        return image, mask
    
    def __call__(self, image, mask):
        """Allow using class as a callable."""
        return self.apply(image, mask)
