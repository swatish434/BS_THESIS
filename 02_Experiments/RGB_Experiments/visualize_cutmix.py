
import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from PIL import Image

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from RGB_Experiments.train_rgb import RGBDataset, get_model, calculate_mean_std, DATASET_PATH, NUM_CLASSES, IMG_RES
from utils.dataset_functions import read_dataset
from utils.RGB_functions import resize_segmentation_masks
from utils.repro import resolve_device

def visualize_results():
    device = resolve_device('auto')
    print(f"Using device: {device}")

    # Load Data
    print("Loading dataset...")
    _, _, _, RGB, _, RGB_general_masks, _ = read_dataset(DATASET_PATH)
    
    RGB_filtered = []
    masks_filtered = []
    for img, mask in zip(RGB, RGB_general_masks):
        if img is not None and mask is not None:
            RGB_filtered.append(img)
            masks_filtered.append(mask)
    RGB = RGB_filtered
    RGB_general_masks = masks_filtered

    # Split Data
    images_train, images_test, masks_train, masks_test = train_test_split(RGB, RGB_general_masks, test_size=0.2, random_state=123)
    # Calibrate mean/std
    mean, std = calculate_mean_std(images_train)
    
    # Test Dataset
    test_dataset = RGBDataset(images_test, masks_test, mean, std, albumentations_transform=None, transform_mask=True, num_classes=NUM_CLASSES)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    # Load Model
    model_path = "/home/bs_thesis/Documents/BS_THESIS/PCBVision/RGB_Experiments/runs/20260202_172507_rgb_deeplabv3+_cp/RGB_deeplabv3+_best.pth"
    print(f"Loading model from {model_path}")
    
    model = get_model('deeplabv3+', NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Hook for features
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    # Hook layer4 of ResNet (Backbone output)
    # model.resnet_features.layer4
    try:
        model.resnet_features.layer4.register_forward_hook(get_activation('backbone'))
        print("Hook registered on resnet_features.layer4")
    except Exception as e:
        print(f"Error registering hook: {e}")

    # Visualize 3 samples
    OUTPUT_DIR = "visualizations_cutmix"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    mean_t = torch.tensor(mean).view(3, 1, 1).to(device)
    std_t = torch.tensor(std).view(3, 1, 1).to(device)

    print("Generating visualizations...")
    samples_to_viz = [0, 2, 5] # Visualize arbitrary interesting indices
    
    count = 0
    for i, (images, masks) in enumerate(test_loader):
        if i not in samples_to_viz:
            continue
            
        images = images.to(device)
        
        # Inference
        output = model(images) # captures features in 'activation'
        pred = torch.argmax(torch.softmax(output, dim=1), dim=1).cpu().numpy()[0] #(H,W)
        
        # Unnormalize image for visualization
        img_vis = images[0] * std_t + mean_t
        img_vis = img_vis.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        
        # Ground Truth
        gt_mask = masks[0].cpu().numpy()
        
        # Features
        features = activation.get('backbone', None)
        
        # Plotting
        # Layout: 2 Rows. Row 1: Image, GT, Pred. Row 2: Top 4 Feature Channels
        fig = plt.figure(figsize=(15, 8))
        gs = fig.add_gridspec(2, 4)
        
        # Row 1
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(img_vis)
        ax1.set_title("Input RGB")
        ax1.axis('off')
        
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(gt_mask, cmap='tab10', vmin=0, vmax=NUM_CLASSES-1)
        ax2.set_title("Ground Truth")
        ax2.axis('off')
        
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(pred, cmap='tab10', vmin=0, vmax=NUM_CLASSES-1)
        ax3.set_title("Prediction")
        ax3.axis('off')
        
        # Legend/Color reference could be nice but skipping for simplicity
        
        # Row 2: Features
        if features is not None:
            # features shape (1, 2048, H/16, W/16)
            f_map = features[0].cpu().numpy() # (2048, h, w)
            # Plot first 4 channels with high variance or just first 4
            # Usually deep features are abstract. Let's pick 4.
            for f_idx in range(4):
                ax_f = fig.add_subplot(gs[1, f_idx])
                ax_f.imshow(f_map[f_idx], cmap='viridis')
                ax_f.set_title(f"Feature Ch {f_idx}")
                ax_f.axis('off')
        else:
            print("No features captured.")
            
        plt.suptitle(f"Test Sample {i} - RGB DeepLabv3+ (CutMix)", fontsize=16)
        plt.tight_layout()
        save_path = f"{OUTPUT_DIR}/sample_{i}_viz.png"
        plt.savefig(save_path)
        print(f"Saved {save_path}")
        plt.close()
        
        count += 1
        if count >= len(samples_to_viz):
            break

if __name__ == "__main__":
    visualize_results()
