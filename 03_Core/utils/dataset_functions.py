# Importing libraries
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import spectral as spi
from scipy import io
import cv2
import sys
import matplotlib as mpl
import seaborn as sns
import timeit
import logging
import random
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import ToTensor, Resize
from PIL import Image
from time import sleep
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from matplotlib.legend_handler import HandlerBase
from spectral import *

def visualize(mask):
    """Mask visualization

    Show a PCB-vision mask in using specific colormap.
    
    Parameters:
        mask (numpy.ndarray): the 2D mask image.
    """
    
    colours = ['black','red', 'green', 'blue', 'Yellow']
    classes = {0:'black', 1:'red', 2:'green', 3:'blue', 4:'Yellow'}
    cmap = []
    for i,x in enumerate(np.unique(mask)):
        cmap.append(classes[x])
    cmap = mpl.colors.ListedColormap(cmap)
    colormap = plt.imshow(mask, cmap=cmap, interpolation="none")
    plt.axis('off')

def read_hsi_mask(datapath, GTpath):
    """Mask loading
    Reads an HSI and its corresponding mask from specified paths and 
    returns the mask as a NumPy array.

    Parameters:
        datapath (str): Path to the directory containing the HSI data.
        GTpath (str): Path to the corresponding mask file.

    Returns:
        mask (numpy.ndarray): NumPy array containing the loaded mask.
    """
    # Convert the GTpath to POSIX format for compatibility
    GTpath2 = GTpath.as_posix()

    # Strip the extension
    spectral_file = str(GTpath2[:-4])  

    # Open the HSI data using ENVI and access the mask band using 
    # bipfile
    numpy_ndarr = envi.open(GTpath, spectral_file)
    y = spi.io.bipfile.BipFile.open_memmap(numpy_ndarr)

    mask = y[:, :, 0]

    return mask

def read_hsi_cube(datapath, Cubepath):
    """
    Reads an HSI cube from the specified path and returns it as a NumPy
    array.

    Parameters:
        datapath (str): Path to the directory containing the HSI data.
        Cubepath (str): Path to the HSI cube file.

    Returns:
        hsi_cube (numpy.ndarray): NumPy array containing the loaded HSI 
        cube.
    """

    # Construct the paths to the header and spectral files
    header_file = str(datapath / Cubepath)
    # Remove extension
    spectral_file = str(datapath / Cubepath[:-4])  

    # Open the HSI data using ENVI
    img = envi.open(header_file, spectral_file)
    
    # Access the entire cube using open_memmap method of the image object
    try:
        hsi_cube = img.open_memmap()
    except Exception as e:
        # Fallback to load if memmap fails, or re-raise
        hsi_cube = img.load() 
        
    # Return the loaded HSI cube as a NumPy array
    return hsi_cube

    
def read_dataset(dataset_path):
    """This function reads the dataset.
    
    Parameters:
        - dataset_path (str): The path of the dataset folder.

    Returns:
        A tuple of 7 lists of 53 elements:
        
        - Hyperspectral Images (HSI) of the 53 PCB
        - HSI general segmentation masks
        - HSI mono segmentation masks
        - RGB images of the 53 PCB
        - RGB general segmentation masks
        - RGB mono segmentation masks
        - PCB Masks of the HSI
    """
    hsi_path = dataset_path + 'HSI/'
    rgb_path = dataset_path + 'RGB/'
    
    HSI = []
    HSI_seg_masks = []
    HSI_mono_masks = []
    RGB = []
    RGB_mono_masks = []
    RGB_general_masks = []
    PCB_Masks = []
    
    for i in tqdm(range(1,54)):
        sleep(0.001)
        try:
            datapath = Path( hsi_path + 'pcb' + str(i))
            Cubepath = "pcb" + str(i) + ".hdr"
            
            Maskpath = Path(hsi_path + 'General_masks/' + str(i) + ".HDR")
            
            # Temporary variables to ensure all parts load before appending
            hsi_temp = read_hsi_cube(datapath, Cubepath)
            hsi_seg_mask_temp = read_hsi_mask(datapath, Maskpath)
        
            Maskpath = Path(hsi_path + 'Monoseg_masks/mono' + str(i) + ".hdr")
            hsi_mono_mask_temp = read_hsi_mask(datapath, Maskpath)
        
            RGBpath = rgb_path + str(i) + '.jpg'
            rgb_temp = cv2.cvtColor(cv2.imread(RGBpath),cv2.COLOR_BGR2RGB)
        
            RGB_mono_masks_path = rgb_path + 'Monoseg/' + str(i) + '.png'
            rgb_mono_mask_temp = np.array(Image.open(RGB_mono_masks_path))
        
            RGB_general_masks_path = rgb_path + 'General/' + str(i) + '.png'
            rgb_general_mask_temp = np.array(Image.open(RGB_general_masks_path))
        
            maskspath = Path(hsi_path + 'PCB_Masks/'+str(i) + ".jpg")
            pcb_mask_temp = cv2.cvtColor(cv2.imread(str(maskspath) ),\
                                          cv2.COLOR_BGR2GRAY)
            
            # If we reached here, everything loaded successfully
            HSI.append(hsi_temp)
            HSI_seg_masks.append(hsi_seg_mask_temp)
            HSI_mono_masks.append(hsi_mono_mask_temp)
            RGB.append(rgb_temp)
            RGB_mono_masks.append(rgb_mono_mask_temp)
            RGB_general_masks.append(rgb_general_mask_temp)
            PCB_Masks.append(pcb_mask_temp)
            
        except Exception as e:
            print(f"Skipping sample {i} due to error: {e}")
            # Append None to all lists to maintain index alignment
            HSI.append(None)
            HSI_seg_masks.append(None)
            HSI_mono_masks.append(None)
            RGB.append(None)
            RGB_mono_masks.append(None)
            RGB_general_masks.append(None)
            PCB_Masks.append(None)
            continue

    print("Dataset loading is complete.")
    return HSI, HSI_seg_masks, HSI_mono_masks, RGB, RGB_mono_masks,\
           RGB_general_masks, PCB_Masks



# Function to train a model on a specific GPU
def set_gpu(gpu_id):
    """
    Function to set the current device to a specific GPU

    Parameters:
        gpu_id (int): The ID of the GPU to use

    Returns:
        torch.device: The selected GPU device
    """

    # Set the current device to the specified GPU ID
    torch.cuda.set_device(gpu_id)
    
    # Return the selected GPU device
    return torch.device("cuda")


def Generate_Training_data(training_list, HSI_cubes, seg_masks):
    """
    Generates augmented training data for the given HS cubes and their
    corresponding masks. The function for reading PCB-Vision HS
    training cubes and generating the training set.
    
    Parameters:
        training_list (list): A list of indices corresponding to PCB-Vision 
                              HS cubes and masks to be augmented.        
        HSI_cubes (list): A list of HS data cubes
        seg_masks (list): A list of the ground truth masks
        
    Returns:
        cubes (list): A list of the augmented HS cubes 
        masks (list): A list of the augmented masks
        
    """
    cubes = []
    masks = []
    for i, ii in enumerate(training_list):
        if HSI_cubes[ii-1] is None:
            continue
        cubes.append(HSI_cubes[ii-1])
        masks.append(seg_masks[ii-1])
        cube_aug, masks_aug = data_augmentation(HSI_cubes[ii-1], \
                                                seg_masks[ii-1])
        for j in range(len(cube_aug)):
            cubes.append(cube_aug[j])
            masks.append(masks_aug[j])
        
        del cube_aug, masks_aug
        
    return cubes, masks

def Generate_data(data_list, HSI_cubes, seg_masks):
    """
    Reading PCB-Vision validation and testing HS cubes.
    It does not perform any augmentation
    
    Args:
        data_list (list): A list of indices corresponding to the HS cubes 
                          in PCB-Vision to be read.
        HSI_cubes (list): A list of the augmented HS cubes 
        seg_masks (list): A list of the ground truth masks
        
    Returns:
        cubes (list): HS cubes
        masks (list): segmentation masks
    
    """
    cubes = []
    masks = []
    for i, ii in enumerate(data_list):
        if HSI_cubes[ii-1] is None:
            continue
        cubes.append(HSI_cubes[ii-1])
        masks.append(seg_masks[ii-1])
        
    return cubes, masks

def data_augmentation(hsi_cube, mask):
    """
    Augments a hyperspectral cube and its corresponding mask with different
    transformations.

    Args:
        hsi_cube (ndarray): A hyperspectral cube of shape (rows, columns, bands)
        mask (ndarray): A 2D mask of shape (rows, columns).

    Returns:
        A tuple of two lists:
        - The first list contains 7 augmented hyperspectral cubes, each of shape
          (rows, columns, bands).
        - The second list contains 7 augmented masks, each of shape (rows, columns).
    """
    # Initialize empty lists to store the augmented cubes and masks
    augmented_cubes = []
    augmented_masks = []

    # Get the number of rows and columns of the mask
    rows, cols = mask.shape
    #rows, cols = hsi_cube.shape[:2]

    # Random rotation clockwise
    angle = np.random.randint(1, 16)
    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    rotated_hsi = cv2.warpAffine(hsi_cube, M, (cols, rows))
    rotated_mask = cv2.warpAffine(mask, M, (cols, rows), flags=cv2.INTER_NEAREST)
    augmented_cubes.append(rotated_hsi)
    augmented_masks.append(rotated_mask)

    # Random rotation counter clockwise
    angle = np.random.randint(1, 16)
    M = cv2.getRotationMatrix2D((cols/2, rows/2), -angle, 1)
    rotated_hsi = cv2.warpAffine(hsi_cube, M, (cols, rows))
    rotated_mask = cv2.warpAffine(mask, M, (cols, rows), flags=cv2.INTER_NEAREST)
    augmented_cubes.append(rotated_hsi)
    augmented_masks.append(rotated_mask)

    # Random vertical translation
    if np.random.rand() < 0.5:
        v_trans = np.random.randint(-30, -9)
    else:
        v_trans = np.random.randint(10, 31)
    M = np.float32([[1, 0, 0], [0, 1, v_trans]])
    translated_hsi = cv2.warpAffine(hsi_cube, M, (cols, rows))
    translated_mask = cv2.warpAffine(mask, M, (cols, rows),flags=cv2.INTER_NEAREST)
    augmented_cubes.append(translated_hsi)
    augmented_masks.append(translated_mask)

    # Random horizontal translation
    if np.random.rand() < 0.5:
        h_trans = np.random.randint(-30, -9)
    else:
        h_trans = np.random.randint(10, 31)
    M = np.float32([[1, 0, h_trans], [0, 1, 0]])
    translated_hsi = cv2.warpAffine(hsi_cube, M, (cols, rows))
    translated_mask = cv2.warpAffine(mask, M, (cols, rows), flags=cv2.INTER_NEAREST)
    augmented_cubes.append(translated_hsi)
    augmented_masks.append(translated_mask)

    # Flip on the vertical axis
    flipped_hsi = np.flip(hsi_cube, axis=0)
    flipped_mask = np.flip(mask, axis=0)
    augmented_cubes.append(flipped_hsi)
    augmented_masks.append(flipped_mask)

    # Flip on the horizontal axis
    flipped_hsi = np.flip(hsi_cube, axis=1)
    flipped_mask = np.flip(mask, axis=1)
    augmented_cubes.append(flipped_hsi)
    augmented_masks.append(flipped_mask)

    return augmented_cubes, augmented_masks


def evaluate_segmentation(ground_truth_masks, predicted_masks, num_classes):
    # Initialize variables for aggregating evaluation metrics
    confusion_matrix_sum = np.zeros((num_classes, num_classes), dtype=np.int64)
    true_positive_sum = np.zeros(num_classes, dtype=np.int64)
    true_negative_sum = np.zeros(num_classes, dtype=np.int64)
    false_positive_sum = np.zeros(num_classes, dtype=np.int64)
    false_negative_sum = np.zeros(num_classes, dtype=np.int64)
    intersection_sum = np.zeros(num_classes, dtype=np.int64)
    union_sum = np.zeros(num_classes, dtype=np.int64)

    for gt_mask, pred_mask in zip(ground_truth_masks, predicted_masks):
        # Calculate confusion matrix
        cm = confusion_matrix(gt_mask.flatten(), pred_mask.flatten(), \
                              labels=list(range(num_classes)))
        confusion_matrix_sum += cm

        # Calculate true positive, true negative, false positive, false negative
        true_positive = np.diag(cm)
        true_positive_sum += true_positive

        false_positive = np.sum(cm, axis=0) - true_positive
        false_positive_sum += false_positive

        false_negative = np.sum(cm, axis=1) - true_positive
        false_negative_sum += false_negative

        # Calculate intersection and union for Intersection Over Union (IoU)
        intersection = true_positive
        union = np.sum(cm, axis=1) + np.sum(cm, axis=0) - true_positive
        intersection_sum += intersection
        union_sum += union

    with np.errstate(divide='ignore', invalid='ignore'):
        # Calculate pixel accuracy per class
        denom = (true_positive_sum + false_negative_sum)
        pixel_accuracy_per_class = np.divide(true_positive_sum, denom, out=np.full_like(true_positive_sum, np.nan, dtype=np.float64), where=denom != 0)

        # Calculate pixel accuracy
        total = np.sum(confusion_matrix_sum)
        pixel_accuracy = (np.sum(true_positive_sum) / total) if total != 0 else float('nan')

        # Calculate precision, recall, F1 score
        p_denom = (true_positive_sum + false_positive_sum)
        r_denom = (true_positive_sum + false_negative_sum)
        precision = np.divide(true_positive_sum, p_denom, out=np.full_like(true_positive_sum, np.nan, dtype=np.float64), where=p_denom != 0)
        recall = np.divide(true_positive_sum, r_denom, out=np.full_like(true_positive_sum, np.nan, dtype=np.float64), where=r_denom != 0)

        f_denom = (precision + recall)
        f1_score = np.divide(2 * precision * recall, f_denom, out=np.full_like(precision, np.nan, dtype=np.float64), where=f_denom != 0)

        # Calculate Intersection Over Union (IoU)
        iou = np.divide(intersection_sum, union_sum, out=np.full_like(intersection_sum, np.nan, dtype=np.float64), where=union_sum != 0)

        # Calculate Dice coefficient
        dice_denom = (np.sum(confusion_matrix_sum, axis=1) + np.sum(confusion_matrix_sum, axis=0))
        dice_coefficient = np.divide(2 * intersection_sum, dice_denom, out=np.full_like(intersection_sum, np.nan, dtype=np.float64), where=dice_denom != 0)

        # Calculate Kappa coefficient (scalar)
        if total == 0:
            kappa = float('nan')
        else:
            observed_accuracy = np.trace(confusion_matrix_sum) / total
            row_marginals = confusion_matrix_sum.sum(axis=1)
            col_marginals = confusion_matrix_sum.sum(axis=0)
            expected_accuracy = float((row_marginals @ col_marginals) / (total ** 2))
            kappa = (observed_accuracy - expected_accuracy) / (1.0 - expected_accuracy) if expected_accuracy != 1.0 else float('nan')

    # Return the calculated evaluation metrics
    return confusion_matrix_sum, true_positive_sum, true_negative_sum, false_positive_sum,\
           false_negative_sum, precision, recall, f1_score, pixel_accuracy_per_class, \
           pixel_accuracy, iou, dice_coefficient, kappa
class PCBFullDataset(Dataset):
    def __init__(self, sample_ids, dataset_root, target_size=(512, 512), augment=False, normalize=True, copy_paste_aug=None):
        self.sample_ids = sample_ids
        self.dataset_root = dataset_root
        self.target_size = target_size
        self.augment = augment
        self.normalize = normalize
        self.copy_paste_aug = copy_paste_aug
        
    def __len__(self):
        return len(self.sample_ids)
    
    def _load_hsi(self, sample_id):
        hsi_dir = f"{self.dataset_root}/HSI/pcb{sample_id}"
        hsi_files = list(os.path.join(hsi_dir, f) for f in os.listdir(hsi_dir) if f.endswith('.hdr'))
        if not hsi_files:
            raise ValueError(f"No .hdr files in {hsi_dir}")
        hdr_file = hsi_files[0]
        data_file = hdr_file.replace('.hdr', '')
        hsi_file = envi.open(hdr_file, data_file)
        hsi = np.array(hsi_file.open_memmap()).astype(np.float32)
        return hsi
    
    def _normalize_hsi(self, hsi):
        hsi_norm = np.zeros_like(hsi)
        for b in range(hsi.shape[2]):
            band = hsi[:, :, b]
            min_val = band.min()
            max_val = band.max()
            if max_val > min_val:
                hsi_norm[:, :, b] = (band - min_val) / (max_val - min_val)
        return np.clip(hsi_norm, 0, 1)
    
    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        try:
            hsi = self._load_hsi(sample_id)
            if hsi.ndim != 3: return self.__getitem__((idx + 1) % len(self))
            
            rgb_path = f"{self.dataset_root}/RGB/{sample_id}.jpg"
            rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
            
            mask_path = f"{self.dataset_root}/RGB/Monoseg/{sample_id}.png"
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            if mask is None: return self.__getitem__((idx + 1) % len(self))
            
            H_target, W_target = self.target_size
            num_bands = hsi.shape[2]
            hsi_resized = np.zeros((H_target, W_target, num_bands), dtype=np.float32)
            for b in range(num_bands):
                hsi_resized[:, :, b] = cv2.resize(hsi[:, :, b], (W_target, H_target))
            
            rgb_resized = cv2.resize(rgb, (W_target, H_target))
            mask_resized = cv2.resize(mask, (W_target, H_target), interpolation=cv2.INTER_NEAREST)
            
            if self.normalize:
                hsi_resized = self._normalize_hsi(hsi_resized)
            
            # Augmentation (CutMix)
            if self.augment and self.copy_paste_aug is not None and np.random.rand() < 0.5:
                # Load a second sample
                idx2 = np.random.randint(len(self))
                sample_id2 = self.sample_ids[idx2]
                try:
                    hsi2 = self._load_hsi(sample_id2)
                    rgb_path2 = f"{self.dataset_root}/RGB/{sample_id2}.jpg"
                    rgb2 = cv2.cvtColor(cv2.imread(rgb_path2), cv2.COLOR_BGR2RGB)
                    mask_path2 = f"{self.dataset_root}/RGB/Monoseg/{sample_id2}.png"
                    mask2 = cv2.imread(mask_path2, cv2.IMREAD_GRAYSCALE)
                    
                    if hsi2.ndim == 3 and mask2 is not None:
                        # Resize sample 2
                        hsi2_resized = np.zeros((H_target, W_target, num_bands), dtype=np.float32)
                        for b in range(num_bands):
                            hsi2_resized[:, :, b] = cv2.resize(hsi2[:, :, b], (W_target, H_target))
                        rgb2_resized = cv2.resize(rgb2, (W_target, H_target))
                        mask2_resized = cv2.resize(mask2, (W_target, H_target), interpolation=cv2.INTER_NEAREST)
                        
                        if self.normalize:
                            hsi2_resized = self._normalize_hsi(hsi2_resized)
                            
                        # Apply CutMix
                        # CutMix expects (H, W, C) for images
                        rgb_resized, hsi_resized, mask_resized, _, _ = self.copy_paste_aug.cutmix(
                            rgb_resized, hsi_resized, mask_resized,
                            rgb2_resized, hsi2_resized, mask2_resized
                        )
                except Exception as e:
                    # Ignore augmentation failure, proceed with original
                    pass

            hsi_tensor = torch.from_numpy(hsi_resized).permute(2, 0, 1).float()
            mask_tensor = torch.from_numpy(mask_resized).long()
            rgb_tensor = torch.from_numpy(rgb_resized).permute(2, 0, 1).float() / 255.0
            
            return {'hsi': hsi_tensor, 'mask': mask_tensor, 'rgb': rgb_tensor, 'sample_id': sample_id}
        
        except Exception as e:
            return self.__getitem__((idx + 1) % len(self))

