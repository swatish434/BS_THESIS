# Run this quickly on HPC to check actual class values in your masks
import cv2
import numpy as np
import glob
import os

mask_dir = '/home/samiran_iiserb/asip_lab/UG/SWATISH/BS_THESIS/DATASETS/PCBDataset/RGB/Monoseg'
masks = glob.glob(os.path.join(mask_dir, '*.png'))

all_values = set()
for m in masks[:10]:
    mask = cv2.imread(m, cv2.IMREAD_GRAYSCALE)
    all_values.update(np.unique(mask).tolist())

print("Unique class values found in masks:", sorted(all_values))