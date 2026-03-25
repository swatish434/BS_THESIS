import sys
import os
import argparse

# Allow running as a script from any working directory.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import spectral.io.envi as envi
import spectral as spi

from PCBVision.models import UNET, DeepLabv3_plus, LinkNet, AttU_Net, ResUnet
from PCBVision.models.factory import get_model
from PCBVision.utils.dataset_functions import evaluate_segmentation, read_dataset
from PCBVision.utils.PCA_functions import resize_hyperspectral_images, resize_segmentation_masks
from PCBVision.utils.repro import resolve_device

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='auto', help='cpu, cuda, or auto')
    p.add_argument('--dataset-path', default="/home/bs_thesis/Documents/BS_THESIS/DATASETS/PCBDataset/")
    p.add_argument('--pca-data-dir', default="/home/bs_thesis/Documents/BS_THESIS/PCBVision/PCA_Data/")
    p.add_argument('--output-dir', default="Evaluation/benchmark_results")
    p.add_argument('--num-classes', type=int, default=4)
    return p.parse_args()


device = resolve_device("auto")

def load_data_robust(data_dir, prefix, counts):
    # Reuse robust hsi loading
    pca_data = []
    masks = []
    for i in range(counts):
        header_file = os.path.join(data_dir, f"{prefix}_{i}.hdr")
        if not os.path.exists(header_file):
            continue 
        data_file = header_file[:-4]
        try:
            numpy_ndarr = envi.open(header_file, data_file)
            c = numpy_ndarr.load()
            pca_data.append(c)
            mask_file = os.path.join(data_dir, f"{prefix}_{i}.npy")
            if os.path.exists(mask_file):
                m = np.load(mask_file)
                masks.append(m)
            else:
                masks.append(None)
        except Exception as e:
            print(f"Error loading {header_file}: {e}")
    return pca_data, masks

def evaluate_model(model, model_type, test_images, test_masks_raw, num_classes=4):
    print(f"Evaluating {model_type} Model...")
    model.eval()
    predicted_masks = []
    
    # Preprocessing constants for RGB
    mean_rgb = np.array([49.378, 41.347, 28.657])
    std_rgb = np.array([7.538, 7.043, 4.648])
    
    with torch.inference_mode():
        for i, img in enumerate(test_images):
            if img is None:
                predicted_masks.append(None)
                continue
                
            if model_type == "RGB":
                # RGB Preprocessing
                img_resized = cv2.resize(img, (640, 640), interpolation=cv2.INTER_NEAREST)
                img_norm = (img_resized - mean_rgb) / std_rgb
                input_tensor = torch.from_numpy(img_norm.transpose(2,0,1)).float().unsqueeze(0).to(device)
            else:
                # HSI Preprocessing (Legacy 3-channel or Single Image)
                # But evaluate_models loop for HSI patches uses Precomputed path now.
                # This block is mainly for RGB.
                # If HSI falls here, it expects 3-channel or handled elsewhere.
                try:
                    resized_list, _ = resize_hyperspectral_images([img], [test_masks_raw[i]], 640)
                    img_resized = resized_list[0]
                    input_tensor = torch.from_numpy(img_resized).float().permute(2,0,1).unsqueeze(0).to(device)
                except:
                     # Fallback or error
                     input_tensor = torch.zeros(1, 3, 640, 640).to(device)

            output = model(input_tensor)
            pred_mask = torch.argmax(output, dim=1).cpu().numpy()[0]
            predicted_masks.append(pred_mask)
            
    # Resize predictions to original shape for evaluation
    final_preds = []
    valid_targets = []
    
    for i, pred in enumerate(predicted_masks):
        if pred is None or test_masks_raw[i] is None:
            continue
            
        target_shape = test_masks_raw[i].shape
        if test_masks_raw[i].ndim == 3:
            target_mask = test_masks_raw[i][:,:,0]
        else:
            target_mask = test_masks_raw[i]
            
        resized_pred = resize_segmentation_masks(pred, target_shape)
        final_preds.append(resized_pred)
        valid_targets.append(target_mask)
        
    # Metrics
    metrics = evaluate_segmentation(valid_targets, final_preds, num_classes)
    return metrics

def main(args=None):
    global device
    if args is None:
        args = parse_args()

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    dataset_path = args.dataset_path
    pca_data_dir = args.pca_data_dir
    RESULTS_DIR = "Results" # Keep reading from legacy Results if needed, but we prefer experiments folders
    
    OUTPUT_DIR = args.output_dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_csv = os.path.join(OUTPUT_DIR, "evaluation_metrics.csv")
    
    models_to_eval = []
    
    # 1. RGB Setup (Dynamic Discovery)
    # Find all RGB_*_best.pth files in local Results, Root, and new RGB_Experiments/results
    import glob
    model_files = []
    
    # New Standard Locations
    model_files.extend(glob.glob("RGB_Experiments/results/RGB_*_best.pth"))
    model_files.extend(glob.glob(os.path.join(RESULTS_DIR, "RGB_*_best.pth"))) # Legacy
    
    # Remove duplicates if any (by absolute path)
    model_files = list(set([os.path.abspath(p) for p in model_files]))

    if not model_files:
        if os.path.exists("RGB_unet_script.pth"): model_files.append("RGB_unet_script.pth")
    
    if model_files:
        print(f"Found RGB Models: {model_files}")
        
        # Load Test Data ONCE (Use split from train_rgb logic)
        from sklearn.model_selection import train_test_split
        indices = list(range(53))
        _, test_idx = train_test_split(indices, test_size=0.2, random_state=123)
        print("Loading RGB Data...")
        _, _, _, RGB, _, RGB_masks, _ = read_dataset(dataset_path)
        test_rgb = [RGB[i] for i in test_idx]
        test_rgb_masks = [RGB_masks[i] for i in test_idx]
        
        for m_path in model_files:
            try:
                # Extract model name from filename: RGB_{MODEL}_best.pth
                # e.g. Results/RGB_DeepLabv3+_best.pth -> DeepLabv3+
                basename = os.path.basename(m_path)
                m_name_raw = basename.replace("RGB_", "").replace("_best.pth", "")
                
                # Determine class args based on known architectures or just try
                # We need get_model logic here or replicate it. 
                # Simplest is to check name and instantiate correct class.
                print(f"Loading {m_path}...")
                
                # Dynamic instantiation based on filename token
                arch = m_name_raw.split('_')[0]
                try:
                    model = get_model(arch, in_channels=3, out_channels=args.num_classes, pretrained=False)
                except Exception:
                    print(f"Unknown architecture for {m_name_raw}, defaulting to UNET...")
                    model = UNET(in_channels=3, out_channels=args.num_classes)
                     
                model.to(device)
                model.load_state_dict(torch.load(m_path, map_location=device))
                
                models_to_eval.append({
                    "name": f"RGB_{m_name_raw}",
                    "model": model,
                    "type": "RGB",
                    "data": test_rgb,
                    "masks": test_rgb_masks
                })
            except Exception as e:
                print(f"Failed to load {m_path}: {e}")
    else:
        print("No RGB Models found.")

    # 2. HSI Setup
    # Check new HSI experiment folders
    hsi_files = glob.glob("HSI_Experiments/results/*overlap*best.pth")
    # Legacy fallback (skip if causes issues)
    # if os.path.exists(os.path.join(RESULTS_DIR, "HSI_unet_best.pth")):
    #    hsi_files.append(os.path.join(RESULTS_DIR, "HSI_unet_best.pth"))

    if hsi_files:
        print("Found HSI Models (Overlap/214-ch). preparing HSI Test Data...")
        from torch.utils.data import DataLoader
        # Define HSIDataset Locally or Import (Better to define simple wrapper if imports complex)
        # Reusing HSIDataset from train_hsi.py requires importing from HSI_Experiments.train_hsi
        # But that might trigger main execution.
        # Let's define a simple HSIDataset here or use spectral directly.
        
        class HSIDatasetEval(torch.utils.data.Dataset):
            def __init__(self, root_dir):
                self.root_dir = root_dir
                self.files = glob.glob(os.path.join(root_dir, "Test_*.hdr"))
                
            def __len__(self):
                return len(self.files)
            
            def __getitem__(self, idx):
                header = self.files[idx]
                base = header[:-4]
                try:
                    img = envi.open(header, base).load()
                    img = np.array(img, dtype=np.float32)
                    # No Augmentation for Eval
                    # Transpose (H, W, C) -> (C, H, W)
                    img = np.transpose(img, (2, 0, 1))
                    img_tensor = torch.from_numpy(img)
                    
                    mask_path = header.replace(".hdr", ".npy")
                    if os.path.exists(mask_path):
                        mask = np.load(mask_path)
                        mask = torch.tensor(mask, dtype=torch.long)
                    else:
                        mask = torch.zeros((img.shape[1], img.shape[2]), dtype=torch.long)
                        
                    return img_tensor, mask
                except Exception as e:
                    print(f"Error loading {header}: {e}")
                    return torch.zeros(214, 256, 256), torch.zeros(256, 256)

        test_hsi_dataset = HSIDatasetEval("/home/bs_thesis/Documents/BS_THESIS/PCBVision/Patches_256_Overlap_Data/")
        # Use Batch Size to avoid OOM
        test_hsi_loader = DataLoader(test_hsi_dataset, batch_size=8, shuffle=False)
        
        for hsi_path in hsi_files:
            print(f"Loading HSI Model: {hsi_path}")

            hsi_model_name = "HSI_UNet_Overlap"
            arch = "unet"
            if 'deeplab' in hsi_path.lower():
                arch = 'deeplabv3+'
                hsi_model_name = "HSI_DeepLabv3+_Overlap"
            elif 'resunet' in hsi_path.lower():
                arch = 'resunet'
                hsi_model_name = "HSI_ResUnet_Overlap"
            elif 'attunet' in hsi_path.lower():
                arch = 'attunet'
                hsi_model_name = "HSI_AttU_Net_Overlap"

            # Default to 214 channels for Overlap
            hsi_model = get_model(arch, in_channels=214, out_channels=args.num_classes, pretrained=False)

            hsi_model.to(device)
            hsi_model.load_state_dict(torch.load(hsi_path, map_location=device))
            hsi_model.eval()
            
            # Evaluate using DataLoader (Custom Logic needed because evaluate_segmentation expects lists)
            # We will accumulate predictions
            print(f"Evaluating {hsi_model_name} on HSI Patches...")
            all_preds = []
            all_masks = []
            
            with torch.no_grad():
                for img_batch, mask_batch in test_hsi_loader:
                     img_batch = img_batch.to(device)
                     outputs = hsi_model(img_batch)
                     preds = torch.argmax(outputs, dim=1).cpu().numpy()
                     
                     for i in range(len(preds)):
                         all_preds.append(preds[i])
                         all_masks.append(mask_batch[i].numpy())
            
            # Now we have lists, we can use the standard format
            models_to_eval.append({
                "name": hsi_model_name,
                "model": None, # Already evaluated
                "type": "HSI_Precomputed", # Special flag
                "predictions": all_preds,
                "masks": all_masks
            })
            
    # Legacy Block Removed
    if False:
        pass
        print("Loading HSI Data...")
        # HSI Test set is explicitly "Test_0" to "Test_14" in PCA_Data
        test_hsi, test_hsi_masks = load_data_robust(pca_data_dir, "Test", 15)
        
        hsi_model = UNET(in_channels=3, out_channels=4).to(device)
        hsi_model.load_state_dict(torch.load(hsi_path, map_location=device))
        
        models_to_eval.append({
            "name": "HSI_UNet",
            "model": hsi_model,
            "type": "HSI",
            "data": test_hsi,
            "masks": test_hsi_masks
        })
    else:
        print("HSI Model not found.")
        
    # Run Evaluation
    results = []
    
    for item in models_to_eval:
        if item.get('type') == 'HSI_Precomputed':
            # Use precomputed predictions
            print(f"Evaluating {item['name']} (Precomputed)...")
            metrics = evaluate_segmentation(item['masks'], item['predictions'], args.num_classes)
        else:
            metrics = evaluate_model(
                item['model'],
                item['type'],
                item['data'],
                item['masks'],
                num_classes=args.num_classes,
            )

        _, _, _, _, _, precision, recall, f1, _, pixel_acc, iou, dice, kappa = metrics
        
        print(f"Model: {item['name']}")
        print(f"Class-wise Precision: {precision}")
        print(f"Class-wise Recall: {recall}")
        print(f"Class-wise F1: {f1}")
        print(f"Class-wise IoU: {iou}")

        # Calculate Means for Arrays
        mean_iou = np.nanmean(iou)
        mean_f1 = np.nanmean(f1)
        mean_precision = np.nanmean(precision)
        mean_recall = np.nanmean(recall)
        
        
        # Append to results
        res_dict = {
            "Model": item['name'],
            "Pixel Accuracy": f"{pixel_acc:.4f}",
            "Mean IoU": f"{mean_iou:.4f}",
            "Background IoU": f"{iou[0]:.4f}",
            "Component IoU": f"{iou[1]:.4f}",
            "IC IoU": f"{iou[2]:.4f}",
            "Connector IoU": f"{iou[3]:.4f}",
            "Mean F1 Score": f"{mean_f1:.4f}",
            "Mean Precision": f"{mean_precision:.4f}",
            "Mean Recall": f"{mean_recall:.4f}",
            "Kappa": f"{np.mean(kappa):.4f}"
        }
        results.append(res_dict)

        
    # Save to CSV
    if results:
        df = pd.DataFrame(results)
        print("\n--- Final Results ---")
        print(df)
        df.to_csv(output_csv, index=False)
        print(f"\nSaved metrics to {output_csv}")
        
        # Save as Markdown
        md_path = output_csv.replace(".csv", ".md")
        df.to_markdown(md_path, index=False)
        print(f"Saved markdown table to {md_path}")
        
        # Save as PNG Image
        png_path = output_csv.replace(".csv", ".png")
        
        # --- PRETTY PRINTING & PLOTTING ---
        # Imports already at top
        
        # 1. Rename Rows for Readability
        name_mapping = {
            "RGB_DeepLabv3+": "RGB DeepLabv3+ (Copy-Paste)",
            "RGB_deeplabv3+_baseline": "RGB DeepLabv3+ (Baseline)",
            "RGB_unet": "RGB UNet (Baseline)",
            "HSI_DeepLabv3+_Overlap": "HSI DeepLabv3+ (Overlap + CP)",
            "HSI_UNet_Overlap": "HSI UNet (Overlap)",
            "RGB_deeplabv3+": "RGB DeepLabv3+ (Copy-Paste)" # Fallback if varying case
        }
        
        # Apply cleanup to Model column
        df['Model'] = df['Model'].apply(lambda x: name_mapping.get(x, x))
        
        # Create a plot with the table
        # Increased width to 14 to fit long names
        fig, ax = plt.subplots(figsize=(16, 2 + len(df) * 0.6)) 
        ax.axis('off')
        
        # Style the table
        # colWidths: First col wider for Model Name
        col_widths = [0.25] + [0.12] * (len(df.columns)-1)
        
        table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center', colWidths=col_widths)
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 2.5) # W, H scaling
        
        # Header & Best Value Highlighting
        # Identify columns to process (numeric ones)
        numeric_cols = df.columns[1:] # Skip 'Model'
        
        # Find max indices for each numeric column
        best_indices = {} # col_idx -> row_idx
        for col_idx, col_name in enumerate(numeric_cols):
            try:
                # Convert to float to be safe
                vals = df[col_name].astype(float)
                best_row = vals.idxmax()
                best_indices[col_idx + 1] = best_row # +1 because Model is col 0
            except:
                pass

        # Formatting Loop
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                # HEADER
                cell.set_text_props(weight='bold', color='white')
                cell.set_facecolor('#404652')
                cell.set_height(0.15)
            else:
                # DATA ROWS
                # Alternating Colors
                if row % 2 == 0:
                    cell.set_facecolor('#f2f2f2')
                else:
                    cell.set_facecolor('#ffffff')
                
                # Highlight Best Score
                if col in best_indices and best_indices[col] == (row - 1): # row-1 because table includes header
                    cell.set_text_props(weight='bold', color='#2e7d32') # Green Bold text
                    cell.set_facecolor('#e8f5e9') # Light Green bg
        
        plt.title("Model Performance Comparison (Best Score Highlighted)", fontsize=18, weight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        print(f"Saved visual table to {png_path}")
        
        # Link to Website Public
        website_path = "/home/bs_thesis/Documents/BS_THESIS/PCBVision_Explainer/public/evaluation_metrics.png"
        if os.path.exists(os.path.dirname(website_path)):
            plt.savefig(website_path, dpi=300, bbox_inches='tight')
            print(f"Updated Website Table at: {website_path}")
            
    else:
        print("No models evaluated.")

if __name__ == "__main__":
    main(parse_args())
