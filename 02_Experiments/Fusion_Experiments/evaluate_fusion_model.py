
import os
import sys
import argparse
import numpy as np
import torch
import cv2
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from Fusion_Experiments.fusion_dataset import FusionDataset
from models.factory import get_model
from utils.dataset_functions import evaluate_segmentation

# Constants
NUM_CLASSES = 4
DATA_DIR = "/home/bs_thesis/Documents/BS_THESIS/PCBVision/Fusion_Experiments/data/patches"

def visualize_prediction(image, mask, pred, save_path):
    # Image: (227, H, W) -> RGB visualization? Use first 3 channels.
    rgb = image[:3, :, :].cpu().numpy().transpose(1, 2, 0)
    # Norm? If it was 0-1, fine.
    
    mask = mask.cpu().numpy()
    pred = pred.cpu().numpy()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(rgb)
    axes[0].set_title("RGB Input")
    axes[0].axis('off')
    
    axes[1].imshow(mask, cmap='jet', vmin=0, vmax=NUM_CLASSES-1)
    axes[1].set_title("Ground Truth")
    axes[1].axis('off')
    
    axes[2].imshow(pred, cmap='jet', vmin=0, vmax=NUM_CLASSES-1)
    axes[2].set_title("Prediction")
    axes[2].axis('off')
    
    plt.savefig(save_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    device = torch.device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Test Split (Scenes 42-52)
    test_indices = list(range(42, 53))
    test_ds = FusionDataset(DATA_DIR, split_indices=test_indices)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    
    print(f"Loading checkpoint: {args.checkpoint}")
    try:
        model = get_model(args.arch, in_channels=227, out_channels=NUM_CLASSES)
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state)
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    all_preds = []
    all_masks = []
    
    print("Running Inference...")
    viz_count = 0
    with torch.no_grad():
        for idx, (images, masks) in enumerate(tqdm(test_loader)):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.append(preds.cpu().numpy()[0])
            all_masks.append(masks.cpu().numpy()[0])
            
            # Save first 10 visualizations
            if viz_count < 10:
                save_path = os.path.join(args.save_dir, f"viz_{idx}.png")
                visualize_prediction(images[0], masks[0], preds[0], save_path)
                viz_count += 1
                
    print("Calculating Metrics...")
    results = evaluate_segmentation(all_masks, all_preds, NUM_CLASSES)
    # Save metrics to text file
    with open(os.path.join(args.save_dir, "test_metrics.txt"), "w") as f:
        f.write(str(results))
        
    print("Evaluation Complete.")

if __name__ == "__main__":
    main()
