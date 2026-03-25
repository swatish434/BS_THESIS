
import sys
import os
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from RGB_Experiments.train_rgb import RGBDataset, get_model, calculate_mean_std, DATASET_PATH, NUM_CLASSES, IMG_RES
from utils.dataset_functions import read_dataset, evaluate_segmentation
from utils.RGB_functions import resize_segmentation_masks
from utils.repro import resolve_device

def evaluate_run():
    device = resolve_device('auto')
    print(f"Using device: {device}")

    # Load Data (Same splitting seed as training)
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

    # Split Data (Seed 123 from train_rgb.py)
    images_train, images_test, masks_train, masks_test = train_test_split(RGB, RGB_general_masks, test_size=0.2, random_state=123)
    # Train/Val split (we don't strictly need val here but for consistency of calculating mean/std on train)
    images_train, images_validation, masks_train, masks_validation = train_test_split(images_train, masks_train, test_size=0.2, random_state=123)

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

    print("Evaluating metrics...")
    predicted_masks = []
    
    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            output = model(images)
            output = torch.squeeze(output, dim=0)
            output = torch.nn.functional.softmax(output, dim=0)
            output = torch.argmax(output, dim=0)
            predicted_masks.append(output.cpu().numpy())
    
    # Resize and Evaluate
    predicted_masks_resized = []
    for i, m in enumerate(predicted_masks):
        original_shape = masks_test[i].shape
        resized = resize_segmentation_masks(m, original_shape)
        predicted_masks_resized.append(resized)

    results = evaluate_segmentation(masks_test, predicted_masks_resized, NUM_CLASSES)
    
    # Unpack results
    confusion_matrix_sum, true_positive_sum, true_negative_sum, false_positive_sum, \
    false_negative_sum, precision, recall, f1_score, pixel_accuracy_per_class, \
    pixel_accuracy, iou, dice_coefficient, kappa = results

    print("\n" + "="*50)
    print("FINAL EVALUATION RESULTS (RGB CutMix)")
    print("="*50)
    print(f"Pixel Accuracy: {pixel_accuracy:.4f}")
    print(f"Mean IoU: {iou}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1_score}")
    print(f"Kappa: {kappa}")
    print("="*50)

if __name__ == "__main__":
    evaluate_run()
