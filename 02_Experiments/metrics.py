"""
metrics.py
──────────
Segmentation evaluation helpers.
"""

import torch
import numpy as np


class SegMetrics:
    """ Accumulates confusion matrix across batches, then computes metrics."""

    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.conf_matrix = np.zeros(
            (self.num_classes, self.num_classes), dtype=np.int64
        )

    @torch.no_grad()
    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        Args:
            preds   : (B, C, H, W) raw logits
            targets : (B, H, W)    long labels
        """
        pred_labels = preds.argmax(dim=1).cpu().numpy().flatten()
        true_labels = targets.cpu().numpy().flatten()

        # Mask out-of-range labels (e.g. ignore index 255)
        valid = (true_labels >= 0) & (true_labels < self.num_classes)
        pred_labels = pred_labels[valid]
        true_labels = true_labels[valid]

        # Handle potential empty valid masks
        if len(true_labels) == 0:
            return

        indices = self.num_classes * true_labels + pred_labels
        matrix  = np.bincount(indices, minlength=self.num_classes ** 2)
        self.conf_matrix += matrix.reshape(self.num_classes, self.num_classes)

    def compute(self) -> dict:
        """ Returns dict with miou, per_class_iou, pixel_acc."""
        cm = self.conf_matrix
        intersection = np.diag(cm)
        union        = cm.sum(axis=1) + cm.sum(axis=0) - intersection
        
        # Avoid division by zero
        iou_per_cls  = np.where(union > 0, intersection / (union + 1e-10), 0)
        
        # Filter out classes that were never present in the ground truth for mean calculation
        present_mask = cm.sum(axis=1) > 0
        if present_mask.any():
            miou = float(np.mean(iou_per_cls[present_mask]))
        else:
            miou = 0.0
            
        pixel_acc    = float(intersection.sum() / (cm.sum() + 1e-10))

        return {
            'miou':          miou,
            'pixel_acc':     pixel_acc,
            'per_class_iou': iou_per_cls.tolist(),
        }
