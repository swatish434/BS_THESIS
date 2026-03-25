
import os
import numpy as np
import spectral.io.envi as envi
from glob import glob
from tqdm import tqdm

def main():
    data_dir = "/home/bs_thesis/Documents/BS_THESIS/PCBVision/Patches_256_Overlap_Data/"
    # Find headers
    headers = sorted(glob(os.path.join(data_dir, "Train_*.hdr")))
    
    print(f"Found {len(headers)} HSI patches.")
    
    mins = []
    maxs = []
    means = []
    
    # Check first 20 samples
    for h in tqdm(headers[:20]):
        try:
            d = h.replace('.hdr', '')
            obj = envi.open(h, d)
            data = obj.load()
            arr = np.array(data)
            
            mins.append(arr.min())
            maxs.append(arr.max())
            means.append(arr.mean())
        except Exception as e:
            print(f"Error: {e}")
            
    print(f"Min value range: {min(mins)} - {max(mins)}")
    print(f"Max value range: {min(maxs)} - {max(maxs)}")
    print(f"Global Mean estimate: {np.mean(means)}")
    
    # Check if int or float
    print(f"Data Type: {arr.dtype}")

if __name__ == "__main__":
    main()
