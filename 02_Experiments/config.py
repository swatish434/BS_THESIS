import torch
import os

class Config:
    # ── Path Settings ──────────────────────────────────────────────────────────
    # RGB patches — generated from 53 full images
    RGB_PATCHES_DIR = '/home/samiran_iiserb/asip_lab/UG/SWATISH/BS_THESIS/PCBVision/01_Data/RGB_Patches_256'

    # HSI patches — generated from raw HSI cubes
    PATCHES_DIR  = '/home/samiran_iiserb/asip_lab/UG/SWATISH/BS_THESIS/PCBVision/01_Data/Patches_256_Overlap_Data'

    # Raw paths (used by patch generator scripts only)
    DATASET_ROOT = '/home/samiran_iiserb/asip_lab/UG/SWATISH/BS_THESIS/DATASETS/PCBDataset'
    RGB_DIR      = os.path.join(DATASET_ROOT, 'RGB')
    MASK_DIR     = os.path.join(DATASET_ROOT, 'RGB', 'Monoseg')
    HSI_DIR      = PATCHES_DIR

    SAVE_DIR     = './results'
    LOG_CSV      = './results/experiment_results_v3.csv'

    # ── Hardware ───────────────────────────────────────────────────────────────
    DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    GPU_ID       = 0
    BATCH_SIZE   = 8
    NUM_WORKERS  = 4

    # ── Training ───────────────────────────────────────────────────────────────
    EPOCHS       = 100
    LR           = 1e-4
    WEIGHT_DECAY = 1e-4
    PATIENCE     = 15

    # ── Optimization ──────────────────────────────────────────────────────────
    USE_AMP      = True
    GRAD_ACCUM   = 1

    # ── Data ──────────────────────────────────────────────────────────────────
    HSI_CHANNELS = 224
    NUM_CLASSES  = 4
    IMAGE_SIZE   = 256

    # ── Experiment Matrix ──────────────────────────────────────────────────────
    MODALITIES    = ['RGB', 'HSI', 'RGB+HSI']
    MODELS        = ['DeepLabV3+', 'Hybrid SSRN-ViT', 'MambaHSI']
    AUGMENTATIONS = ['None', 'Copy-Paste', 'CutMix', 'Generative']


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)