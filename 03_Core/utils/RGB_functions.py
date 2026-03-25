import numpy as np
from PIL import Image
import torch
import cv2


def _pil_bilinear_resample():
    # Pillow >= 9.1 uses Image.Resampling, older uses Image.BILINEAR
    resampling = getattr(Image, "Resampling", None)
    if resampling is not None:
        return resampling.BILINEAR
    # Pillow historically used integer constants; 2 corresponds to BILINEAR.
    return 2

def calculate_mean_std(image_list, target_resolution=(640, 640)):
    """
    Calculate mean and standard deviation across all images and channels.

    Parameters:
        image_list (list of numpy.ndarray): List of input images.
        target_resolution (tuple): Target resolution for resizing images.

    Returns:
        tuple: Mean and standard deviation values for each channel.
    """
    total_pixels = 0
    channel_sums = np.zeros(3, dtype=np.float64)
    channel_sumsq = np.zeros(3, dtype=np.float64)

    for image in image_list:
        if image is None:
            continue

        # Convert NumPy array to PIL Image
        pil_img = Image.fromarray(image)
        pil_img = pil_img.resize(target_resolution, _pil_bilinear_resample())
        arr = np.asarray(pil_img, dtype=np.float64)

        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"Expected RGB image (H,W,3), got shape {arr.shape}")

        total_pixels += arr.shape[0] * arr.shape[1]
        channel_sums += arr.reshape(-1, 3).sum(axis=0)
        channel_sumsq += (arr.reshape(-1, 3) ** 2).sum(axis=0)

    if total_pixels == 0:
        raise ValueError("No valid images provided to calculate_mean_std")

    means = channel_sums / total_pixels
    # Var(X) = E[X^2] - (E[X])^2
    vars_ = channel_sumsq / total_pixels - (means ** 2)
    stds = np.sqrt(np.maximum(vars_, 0.0))

    return tuple(means.tolist()), tuple(stds.tolist())

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

def resize_segmentation_masks(mask, new_shape):
    """
    Resize a list of segmentation masks while preserving class values to the 
    specified new shape (row x col).

    Parameters:
        masks (list of numpy arrays): A list of segmentation masks with values [0, 1, 2, 3].
        new_shape (tuple): A tuple representing the new shape (row, col) to which the masks
          should be resized.

    Returns:
        list of numpy arrays: A list of resized segmentation masks with preserved class values.
    """
    
    # Resize the mask to the new shape using OpenCV with nearest-neighbor interpolation
    resized_mask = cv2.resize(mask, (new_shape[1], new_shape[0]), interpolation=cv2.INTER_NEAREST)

    # Ensure the data type is integer
    resized_mask = resized_mask.astype(int)

    # Clip values to ensure they are in the [0, 3] range
    resized_mask = np.clip(resized_mask, 0, 3)

    return resized_mask
