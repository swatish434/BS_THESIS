
import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from PCBVision.train_hsi_overlap import HSIPatchesDataset, get_hsi_model

DATA_DIR = "/home/bs_thesis/Documents/BS_THESIS/PCBVision/Patches_256_Overlap_Data/"
IN_CHANNELS = 214
OUT_CHANNELS = 4

def hsi_to_rgb(hsi_cube):
    """Convert HSI (C, H, W) tensor to RGB numpy image for visualization"""
    # Simply pick 3 channels: Red (~640nm), Green (~550nm), Blue (~460nm)
    # Assuming valid bands are roughly linear 400-1000nm over 214 bands
    # Red idx ~ 85, Green idx ~ 53, Blue idx ~ 21 (approx)
    
    # hsi_cube is (C, H, W)
    if torch.is_tensor(hsi_cube):
        hsi_cube = hsi_cube.cpu().numpy()
        
    r = hsi_cube[85, :, :]
    g = hsi_cube[53, :, :]
    b = hsi_cube[21, :, :]
    
    rgb = np.stack([r, g, b], axis=2)
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
    return rgb

def visualize_hsi():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='deeplabv3+', help='Model name')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    val_ds = HSIPatchesDataset(DATA_DIR, split='Val')
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    
    model_path = args.checkpoint
    print(f"Loading model: {args.model} from {model_path}")
    model = get_hsi_model(args.model, IN_CHANNELS, OUT_CHANNELS).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    OUTPUT_DIR = f"visualizations_hsi_{args.model}_cutmix"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Hook features
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    
    # Register hooks based on model type
    hook_registered = False
    try:
        if hasattr(model, 'resnet_features'):
            model.resnet_features.layer4.register_forward_hook(get_activation('backbone'))
            print("Hook registered on resnet_features.layer4")
            hook_registered = True
        elif hasattr(model, 'Conv5'):
            model.Conv5.register_forward_hook(get_activation('backbone'))
            print("Hook registered on AttU_Net Conv5 (bottleneck)")
            hook_registered = True
    except Exception as e:
        print(f"Error registering hook: {e}")
    
    print("Generating HSI visualizations...")
    samples_to_viz = [0, 5, 10]
    count = 0
    
    for i, (data, target) in enumerate(val_loader):
        if i not in samples_to_viz and count < 3:
             pass
        elif i in samples_to_viz:
             pass
        else:
             continue

        data = data.to(device)
        output = model(data)
        pred = torch.argmax(torch.softmax(output, dim=1), dim=1).cpu().numpy()[0]
        
        img_rgb = hsi_to_rgb(data[0])
        gt_mask = target[0].numpy()
        
        fig = plt.figure(figsize=(15, 8))
        gs = fig.add_gridspec(2, 4)
        
        # Row 1
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(img_rgb)
        ax1.set_title("Input HSI (Pseudo-RGB)")
        ax1.axis('off')
        
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(gt_mask, cmap='tab10', vmin=0, vmax=OUT_CHANNELS-1)
        ax2.set_title("Ground Truth")
        ax2.axis('off')
        
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(pred, cmap='tab10', vmin=0, vmax=OUT_CHANNELS-1)
        ax3.set_title("Prediction")
        ax3.axis('off')
        
        # Row 2 Features (if available)
        if hook_registered and 'backbone' in activation:
            features = activation['backbone'][0].cpu().numpy()
            for f_idx in range(4):
                ax_f = fig.add_subplot(gs[1, f_idx])
                ax_f.imshow(features[f_idx], cmap='viridis')
                ax_f.set_title(f"Feature Ch {f_idx}")
                ax_f.axis('off')
        else:
            # Just show empty or text
            ax_note = fig.add_subplot(gs[1, :])
            ax_note.text(0.5, 0.5, "Feature maps not available", ha='center', va='center')
            ax_note.axis('off')
            
        plt.suptitle(f"Test Sample {i} - HSI {args.model} (CutMix)", fontsize=16)
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/sample_{i}_viz.png")
        plt.close()
        print(f"Saved sample {i}")
        count += 1

if __name__ == "__main__":
    visualize_hsi()
