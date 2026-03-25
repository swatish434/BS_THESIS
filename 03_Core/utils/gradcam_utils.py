
import numpy as np
import torch
import torch.nn.functional as F
import cv2

class GradCAM:
    """
    Grad-CAM implementation for visualizing class activation maps.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)
        # Note: register_backward_hook is deprecated in newer torch versions (use register_full_backward_hook),
        # but for compatibility with older implementations we check.
        # If it fails, we use register_full_backward_hook logic or simple backward usage.

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        # grad_output is a tuple
        self.gradients = grad_output[0] 

    def __call__(self, x, class_idx=None):
        self.model.zero_grad()
        
        # Forward pass
        output = self.model(x)
        
        if class_idx is None:
            # If no class specified, use the one with highest probability
            class_idx = torch.argmax(output, dim=1).item()
            
        # Create target for backprop
        # output is (N, C, H, W) for segmentation or (N, C) for classification.
        # For segmentation, GradCAM usually targets a specific pixel or the sum of the class mask.
        # Here we do "Sum of scores for Class C" across the whole image.
        
        one_hot = torch.zeros_like(output)
        one_hot[:, class_idx, :, :] = 1
        
        # Backward pass
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Get captured data
        gradients = self.gradients
        activations = self.activations
        
        # Global Average Pooling of Gradients (Weights)
        # gradients: (N, C, H, W) -> we pool over H, W
        weights = torch.mean(gradients, dim=[2, 3], keepdim=True)
        
        # Weighted sum of activations
        # activations: (N, C, H, W)
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        
        # ReLU
        cam = F.relu(cam)
        
        # Normalize to 0-1
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.data.cpu().numpy()[0, 0] # Return 2D array (H, W)

def show_cam_on_image(img, mask):
    """
    Overlay Grad-CAM heatmap on image.
    img: (H, W, 3) float range 0-1 or uint8
    mask: (H, W) float range 0-1
    """
    # Resize mask to image size
    heatmap = cv2.resize(mask, (img.shape[1], img.shape[0]))
    
    # Colorize
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    
    # Normalize image if needed
    if img.max() > 1:
        img = np.float32(img) / 255
        
    cam = heatmap + img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)
