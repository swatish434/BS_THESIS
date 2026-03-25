"""
SSRN-ViT Hyperspectral Image Classification - Corrected Version
Dataset: Indian Pines
All conceptual, logical, and technical errors have been fixed.
"""

import numpy as np
import torch
from operator import truediv
import torch.utils.data as Data
from sklearn import metrics, preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
import matplotlib.pyplot as plt
import scipy.io as sio
import os
import spectral
import cv2
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torchsummary import summary
from einops import rearrange, repeat
import torch_optimizer as optim2
from torch import optim
import time
import collections
import urllib.request
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ============================================================================
# SETUP AND DATA DOWNLOAD
# ============================================================================

# Create necessary directories (fixed: replaced Jupyter magic commands)
os.makedirs('./content/classification_maps', exist_ok=True)

def download_file(url, filepath):
    """Download file if it doesn't exist."""
    if not os.path.isfile(filepath):
        print(f"Downloading {filepath}...")
        try:
            urllib.request.urlretrieve(url, filepath)
            print(f"Downloaded {filepath} successfully.")
        except Exception as e:
            print(f"Error downloading {filepath}: {e}")

# Download Indian Pines dataset if not present
download_file(
    'http://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat',
    './content/Indian_pines_corrected.mat'
)
download_file(
    'http://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat',
    './content/Indian_pines_gt.mat'
)

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

global Dataset
Dataset = "IN"

def load_dataset(Dataset):
    """Load hyperspectral dataset."""
    mat_data = sio.loadmat('./content/Indian_pines_corrected.mat')
    mat_gt = sio.loadmat('./content/Indian_pines_gt.mat')
    data_hsi = mat_data['indian_pines_corrected']
    gt_hsi = mat_gt['indian_pines_gt']
    TOTAL_SIZE = 10249
    VALIDATION_SPLIT = 0.90
    TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)
    return data_hsi, gt_hsi, TOTAL_SIZE, TRAIN_SIZE, VALIDATION_SPLIT

# Load data
data_hsi, gt_hsi, TOTAL_SIZE, TRAIN_SIZE, VALIDATION_SPLIT = load_dataset(Dataset)

# Print dataset information
print("=" * 60)
print("DATASET INFORMATION")
print("=" * 60)
print(f"Data shape: {data_hsi.shape}")
print(f"Ground truth shape: {gt_hsi.shape}")

# Extract dimensions
image_x, image_y, BAND = data_hsi.shape
data = data_hsi.reshape(np.prod(data_hsi.shape[:2]), np.prod(data_hsi.shape[2:]))
gt = gt_hsi.reshape(np.prod(gt_hsi.shape[:2]),)

# Determine number of classes
CLASSES_NUM = int(max(gt))
print(f'Number of classes: {CLASSES_NUM}')

# Model parameters
print('\n----- Setting Parameters -----')
ITER = 1
PATCH_LENGTH = 4
lr, num_epochs, batch_size = 0.001, 100, 32
loss = torch.nn.CrossEntropyLoss()

# Calculate dimensions
img_rows = 2 * PATCH_LENGTH + 1
img_cols = 2 * PATCH_LENGTH + 1
img_channels = data_hsi.shape[2]
INPUT_DIMENSION = data_hsi.shape[2]
ALL_SIZE = data_hsi.shape[0] * data_hsi.shape[1]
VAL_SIZE = int(TRAIN_SIZE)
TEST_SIZE = TOTAL_SIZE - TRAIN_SIZE

# Initialize metrics storage
KAPPA = []
OA = []
AA = []
TRAINING_TIME = []
TESTING_TIME = []
ELEMENT_ACC = np.zeros((ITER, CLASSES_NUM))

# Scale and preprocess data
data = preprocessing.scale(data)
data_ = data.reshape(data_hsi.shape[0], data_hsi.shape[1], data_hsi.shape[2])
whole_data = data_
padded_data = np.lib.pad(whole_data, ((PATCH_LENGTH, PATCH_LENGTH), (PATCH_LENGTH, PATCH_LENGTH), (0, 0)),
                         'constant', constant_values=0)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================================================
# MODEL DEFINITIONS
# ============================================================================

class Residual(nn.Module):
    """
    Residual block with 3D convolutions.
    Fixed: Removed double ReLU activation issue.
    """
    def __init__(self, in_channels, out_channels, kernel_size, padding, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        # First convolution (no ReLU here to avoid double activation)
        self.conv1 = nn.Conv3d(in_channels, out_channels,
                               kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn1 = nn.BatchNorm3d(out_channels)
        
        # Second convolution
        self.conv2 = nn.Conv3d(out_channels, out_channels,
                               kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        # Optional 1x1 conv for dimension matching
        if use_1x1conv:
            self.conv3 = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None

    def forward(self, X):
        # Forward pass with single ReLU after BN
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y + X)


class SSRN_network(nn.Module):
    """
    Spectral-Spatial Residual Network for hyperspectral image classification.
    Fixed: Complete forward pass with classification layers.
    """
    def __init__(self, band, classes, patch_size=9):
        super(SSRN_network, self).__init__()
        self.name = 'SSRN'
        self.patch_size = patch_size
        
        # Initial spectral convolution
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=24,
                               kernel_size=(1, 1, 7), stride=(1, 1, 2))
        self.batch_norm1 = nn.Sequential(
            nn.BatchNorm3d(24, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(inplace=True)
        )

        # Spectral residual blocks
        self.res_net1 = Residual(24, 24, (1, 1, 7), (0, 0, 3))
        self.res_net2 = Residual(24, 24, (1, 1, 7), (0, 0, 3))
        
        # Spatial residual blocks
        self.res_net3 = Residual(24, 24, (3, 3, 1), (1, 1, 0))
        self.res_net4 = Residual(24, 24, (3, 3, 1), (1, 1, 0))

        # Calculate kernel size for spectral reduction
        kernel_3d = math.ceil((band - 6) / 2)

        # Spectral reduction convolution
        self.conv2 = nn.Conv3d(in_channels=24, out_channels=128, padding=(0, 0, 0),
                               kernel_size=(1, 1, kernel_3d), stride=(1, 1, 1))
        self.batch_norm2 = nn.Sequential(
            nn.BatchNorm3d(128, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(inplace=True)
        )
        
        # Spatial feature extraction
        self.conv3 = nn.Conv3d(in_channels=1, out_channels=24, padding=(0, 0, 0),
                               kernel_size=(3, 3, 128), stride=(1, 1, 1))
        self.batch_norm3 = nn.Sequential(
            nn.BatchNorm3d(24, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(inplace=True)
        )

        # Classification layers (fixed: these were defined but never used)
        # Calculate spatial size after all convolutions
        # After conv1 and res blocks: spatial size remains same (9x9)
        # After conv2: spectral dim reduces to 1
        # After conv3: we get 24 channels, spatial 5x5 (due to 3x3 conv no padding on spectral)
        
        self.spatial_size = patch_size - 2  # After 3x3 convolutions
        self.avg_pooling = nn.AvgPool2d(kernel_size=self.spatial_size)
        
        self.full_connection = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(24, classes)
        )

    def forward(self, X):
        """
        Fixed: Complete forward pass with proper tensor transformations.
        Input: (batch, 1, patch_size, patch_size, bands)
        Output: (batch, classes) - class logits
        """
        # Spectral feature extraction
        x1 = self.batch_norm1(self.conv1(X))
        
        # Spectral residual learning
        x2 = self.res_net1(x1)
        x2 = self.res_net2(x2)
        
        # Spatial residual learning
        x2 = self.res_net3(x2)
        x2 = self.res_net4(x2)
        
        # Spectral dimension reduction
        x2 = self.batch_norm2(self.conv2(x2))
        
        # Rearrange for spatial feature extraction
        # x2 shape: (batch, 128, H, W, 1) after conv2
        x2 = x2.permute(0, 4, 2, 3, 1)  # (batch, 1, H, W, 128)
        
        # Spatial feature extraction
        x2 = self.batch_norm3(self.conv3(x2))  # (batch, 24, H-2, W-2, 1)
        
        # Reshape for classification
        x2 = x2.view(x2.size()[0], x2.size()[1], x2.size()[2], x2.size()[3])
        # x2 shape: (batch, 24, H-2, W-2)
        
        # Global average pooling
        x2 = self.avg_pooling(x2)  # (batch, 24, 1, 1)
        x2 = x2.view(x2.size(0), -1)  # (batch, 24)
        
        # Classification
        x2 = self.full_connection(x2)  # (batch, classes)
        
        return x2


# ============================================================================
# VISION TRANSFORMER (ViT) COMPONENTS
# ============================================================================

MIN_NUM_PATCHES = 16

class Residual_1(nn.Module):
    """Residual wrapper for transformer layers."""
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    """Layer normalization before applying function."""
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    """Feed-forward network with GELU activation."""
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """Multi-head self-attention mechanism."""
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    """Transformer encoder with multiple layers."""
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual_1(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual_1(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x


class ViT(nn.Module):
    """
    Vision Transformer for hyperspectral classification.
    Takes features from SSRN and applies transformer attention.
    """
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, 
                 mlp_dim, channels=3, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        assert num_patches >= MIN_NUM_PATCHES, f'Number of patches ({num_patches}) too small. Decrease patch size.'

        self.patch_size = patch_size
        self.num_patches = num_patches

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.to_cls_token = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img, mask=None):
        p = self.patch_size

        # Rearrange image into patches
        x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        x = self.patch_to_embedding(x)
        
        b, n, _ = x.shape

        # Add class token and positional embedding
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        # Transformer encoding
        x = self.transformer(x, mask)

        # Extract class token for classification
        x = self.to_cls_token(x[:, 0])

        return self.mlp_head(x)


# ============================================================================
# HYBRID SSRN-ViT MODEL
# ============================================================================

class SSRN_ViT_Hybrid(nn.Module):
    """
    Hybrid model combining SSRN feature extraction with ViT classification.
    Fixed: Proper dimension handling between SSRN and ViT components.
    """
    def __init__(self, band, classes, patch_size=9, 
                 vit_dim=64, vit_depth=4, vit_heads=8, vit_mlp_dim=128,
                 vit_dropout=0.1, vit_emb_dropout=0.1):
        super(SSRN_ViT_Hybrid, self).__init__()
        
        self.patch_size = patch_size
        self.ssrn_channels = 24
        self.spatial_out = patch_size - 2  # After 3x3 convolutions
        
        # SSRN feature extractor (without final classification)
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=24,
                               kernel_size=(1, 1, 7), stride=(1, 1, 2))
        self.batch_norm1 = nn.Sequential(
            nn.BatchNorm3d(24, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(inplace=True)
        )

        self.res_net1 = Residual(24, 24, (1, 1, 7), (0, 0, 3))
        self.res_net2 = Residual(24, 24, (1, 1, 7), (0, 0, 3))
        self.res_net3 = Residual(24, 24, (3, 3, 1), (1, 1, 0))
        self.res_net4 = Residual(24, 24, (3, 3, 1), (1, 1, 0))

        kernel_3d = math.ceil((band - 6) / 2)
        self.conv2 = nn.Conv3d(in_channels=24, out_channels=128, padding=(0, 0, 0),
                               kernel_size=(1, 1, kernel_3d), stride=(1, 1, 1))
        self.batch_norm2 = nn.Sequential(
            nn.BatchNorm3d(128, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(inplace=True)
        )
        
        self.conv3 = nn.Conv3d(in_channels=1, out_channels=24, padding=(0, 0, 0),
                               kernel_size=(3, 3, 128), stride=(1, 1, 1))
        self.batch_norm3 = nn.Sequential(
            nn.BatchNorm3d(24, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(inplace=True)
        )
        
        # ViT classifier
        self.vit = ViT(
            image_size=self.spatial_out,
            patch_size=1,
            num_classes=classes,
            dim=vit_dim,
            depth=vit_depth,
            heads=vit_heads,
            mlp_dim=vit_mlp_dim,
            channels=self.ssrn_channels,
            dropout=vit_dropout,
            emb_dropout=vit_emb_dropout
        )
        
    def forward(self, X):
        # SSRN feature extraction
        x1 = self.batch_norm1(self.conv1(X))
        x2 = self.res_net1(x1)
        x2 = self.res_net2(x2)
        x2 = self.res_net3(x2)
        x2 = self.res_net4(x2)
        x2 = self.batch_norm2(self.conv2(x2))
        x2 = x2.permute(0, 4, 2, 3, 1)
        x2 = self.batch_norm3(self.conv3(x2))
        
        # Reshape for ViT: (batch, channels, height, width)
        x2 = x2.view(x2.size()[0], x2.size()[1], x2.size()[2], x2.size()[3])
        
        # ViT classification
        output = self.vit(x2)
        
        return output


# ============================================================================
# DATA SAMPLING AND LOADING FUNCTIONS
# ============================================================================

def sampling(proportion, ground_truth, bg=False):
    """
    Perform stratified sampling to split data into train and test sets.
    Fixed: Correct logic for train/test split based on proportion.
    
    Args:
        proportion (float): Proportion of data for TRAINING (e.g., 0.90 means 90% train, 10% test)
        ground_truth (numpy array): Array containing ground truth labels
        bg (bool): Flag indicating whether to consider background labels
    
    Returns:
        train_indexes (list): Indexes for training
        test_indexes (list): Indexes for testing
    """
    train = {}
    test = {}
    labels_loc = {}
    m = max(ground_truth)
    
    for i in range(m):
        # Adjust index range based on background flag
        a = 0 if bg else 1
        indexes = [j for j, x in enumerate(ground_truth.ravel().tolist()) if x == i + a]
        np.random.shuffle(indexes)
        labels_loc[i] = indexes
        
        if proportion != 1:
            # Fixed: nb_train is the number of training samples
            nb_train = int(proportion * len(indexes))
        else:
            nb_train = len(indexes)
        
        # Fixed: Correct assignment
        train[i] = indexes[:nb_train]
        test[i] = indexes[nb_train:]

    train_indexes = []
    test_indexes = []
    for i in range(m):
        train_indexes += train[i]
        test_indexes += test[i]

    np.random.shuffle(train_indexes)
    np.random.shuffle(test_indexes)

    return train_indexes, test_indexes


def index_assignment(index, row, col, pad_length):
    """Assign indices to row/column positions with padding offset."""
    indexed_assignments = {}
    for counter, value in enumerate(index):
        assign_0 = value // col + pad_length
        assign_1 = value % col + pad_length
        indexed_assignments[counter] = [assign_0, assign_1]
    return indexed_assignments


def select_patch(matrix, pos_row, pos_col, ex_len):
    """Extract a patch from the matrix centered at (pos_row, pos_col)."""
    rows = matrix[range(pos_row - ex_len, pos_row + ex_len + 1)]
    patch = rows[:, range(pos_col - ex_len, pos_col + ex_len + 1)]
    return patch


def select_small_cubic(data_size, data_indices, whole_data, patch_length, padded_data, dimension):
    """Extract cubic patches for each sample."""
    selected_patches = np.zeros((data_size, 2 * patch_length + 1, 2 * patch_length + 1, dimension))
    patch_assignments = index_assignment(data_indices, whole_data.shape[0], whole_data.shape[1], patch_length)
    for i in range(len(patch_assignments)):
        selected_patches[i] = select_patch(padded_data, patch_assignments[i][0], patch_assignments[i][1], patch_length)
    return selected_patches


def generate_iter(TRAIN_SIZE, train_indices, TEST_SIZE, test_indices, TOTAL_SIZE, total_indices,
                  TOTAL_SIZEBG, total_indicesbg, VAL_SIZE, whole_data, PATCH_LENGTH, padded_data,
                  INPUT_DIMENSION, batch_size, gt):
    """
    Generate data iterators for training, validation, and testing.
    Fixed: Correct validation split from training data.
    """
    # Extract labels
    background_labels = gt[total_indicesbg]
    gt_all = gt[total_indices] - 1
    y_train = gt[train_indices] - 1
    y_test = gt[test_indices] - 1

    # Select patches
    all_data = select_small_cubic(TOTAL_SIZE, total_indices, whole_data, PATCH_LENGTH, padded_data, INPUT_DIMENSION)
    background_data = select_small_cubic(TOTAL_SIZEBG, total_indicesbg, whole_data, PATCH_LENGTH, padded_data, INPUT_DIMENSION)
    train_data = select_small_cubic(TRAIN_SIZE, train_indices, whole_data, PATCH_LENGTH, padded_data, INPUT_DIMENSION)
    test_data = select_small_cubic(TEST_SIZE, test_indices, whole_data, PATCH_LENGTH, padded_data, INPUT_DIMENSION)

    # Reshape data
    training_data = train_data.reshape(train_data.shape[0], train_data.shape[1], train_data.shape[2], INPUT_DIMENSION)
    testing_data = test_data.reshape(test_data.shape[0], test_data.shape[1], test_data.shape[2], INPUT_DIMENSION)
    all_data = all_data.reshape(all_data.shape[0], all_data.shape[1], all_data.shape[2], INPUT_DIMENSION)
    background_data = background_data.reshape(background_data.shape[0], background_data.shape[1], 
                                               background_data.shape[2], INPUT_DIMENSION)

    # Fixed: Split training data for validation (not test data)
    # Use a portion of training data for validation
    val_split = 0.1  # 10% of training for validation
    val_size = int(TRAIN_SIZE * val_split)
    actual_train_size = TRAIN_SIZE - val_size

    # Split training data
    train_data_final = training_data[:-val_size]
    y_train_final = y_train[:-val_size]
    validation_data = training_data[-val_size:]
    y_val = y_train[-val_size:]

    # Convert to PyTorch tensors
    train_tensor_data = torch.from_numpy(train_data_final).type(torch.FloatTensor).unsqueeze(1)
    train_tensor_labels = torch.from_numpy(y_train_final).type(torch.LongTensor)
    train_dataset = Data.TensorDataset(train_tensor_data, train_tensor_labels)

    valid_tensor_data = torch.from_numpy(validation_data).type(torch.FloatTensor).unsqueeze(1)
    valid_tensor_labels = torch.from_numpy(y_val).type(torch.LongTensor)
    valid_dataset = Data.TensorDataset(valid_tensor_data, valid_tensor_labels)

    test_tensor_data = torch.from_numpy(testing_data).type(torch.FloatTensor).unsqueeze(1)
    test_tensor_labels = torch.from_numpy(y_test).type(torch.LongTensor)
    test_dataset = Data.TensorDataset(test_tensor_data, test_tensor_labels)

    all_tensor_data = torch.from_numpy(all_data).type(torch.FloatTensor).unsqueeze(1)
    all_tensor_labels = torch.from_numpy(gt_all).type(torch.LongTensor)
    all_dataset = Data.TensorDataset(all_tensor_data, all_tensor_labels)

    background_tensor_data = torch.from_numpy(background_data).type(torch.FloatTensor).unsqueeze(1)
    background_tensor_labels = torch.from_numpy(background_labels).type(torch.LongTensor)
    background_dataset = Data.TensorDataset(background_tensor_data, background_tensor_labels)

    # Create data loaders
    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_loader = Data.DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = Data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    all_loader = Data.DataLoader(dataset=all_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    background_loader = Data.DataLoader(dataset=background_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, valid_loader, test_loader, all_loader, background_loader


# ============================================================================
# TRAINING AND EVALUATION FUNCTIONS
# ============================================================================

def evaluate_accuracy(data_iter, net, loss, device):
    """Evaluate accuracy and loss on a dataset."""
    acc_sum, n = 0.0, 0
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for X, y in data_iter:
            X = X.to(device)
            y = y.to(device)
            net.eval()
            y_hat = net(X)
            l = loss(y_hat, y.long())
            acc_sum += (y_hat.argmax(dim=1) == y).float().sum().cpu().item()
            total_loss += l.item()
            num_batches += 1
            n += y.shape[0]
    
    net.train()
    return acc_sum / n, total_loss / num_batches


def train(net, train_iter, valid_iter, loss, optimizer, device, epochs, early_stopping=True, early_num=20):
    """
    Train the network with early stopping.
    Fixed: Correct early stopping logic (save when loss decreases).
    """
    train_losses = []
    valid_losses = []
    train_accuracies = []
    valid_accuracies = []
    
    best_valid_loss = float('inf')
    early_stopping_count = 0

    net = net.to(device)
    print(f"Training on {device}")
    start = time.time()

    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    for epoch in range(epochs):
        train_acc_sum, n = 0.0, 0
        train_l_sum = 0.0
        batch_count = 0
        epoch_start = time.time()
        
        net.train()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            
            y_hat = net(X)
            batch_loss = loss(y_hat, y.long())
            
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            
            train_l_sum += batch_loss.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        
        # Update learning rate
        lr_scheduler.step()
        
        # Evaluate on validation set
        valid_acc, valid_loss = evaluate_accuracy(valid_iter, net, loss, device)
        
        train_losses.append(train_l_sum / batch_count)
        train_accuracies.append(train_acc_sum / n)
        valid_losses.append(valid_loss)
        valid_accuracies.append(valid_acc)
        
        print(f'Epoch {epoch + 1:3d}: train loss {train_l_sum / batch_count:.6f}, '
              f'train acc {train_acc_sum / n:.3f}, valid loss {valid_loss:.6f}, '
              f'valid acc {valid_acc:.3f}, time {time.time() - epoch_start:.1f}s')
        
        # Fixed: Early stopping - save when validation loss DECREASES
        if early_stopping:
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                early_stopping_count = 0
                # Save best model
                torch.save(net.state_dict(), './content/best_model.pt')
            else:
                early_stopping_count += 1
                
            if early_stopping_count >= early_num:
                print(f'Early stopping at epoch {epoch + 1}')
                net.load_state_dict(torch.load('./content/best_model.pt', weights_only=True))
                break

    # Plot training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(121)
    plt.plot(train_accuracies, 'g-', label='Train Accuracy')
    plt.plot(valid_accuracies, 'b-', label='Valid Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    plt.subplot(122)
    plt.plot(train_losses, 'r-', label='Train Loss')
    plt.plot(valid_losses, 'orange', label='Valid Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('./content/training_history.png', dpi=150)
    plt.show()
    
    print(f'\nTraining completed in {time.time() - start:.1f}s')
    return train_losses, valid_losses, train_accuracies, valid_accuracies


def aa_and_each_accuracy(confusion_matrix):
    """Calculate per-class accuracy and average accuracy."""
    diagonal = np.diag(confusion_matrix)
    row_sums = np.sum(confusion_matrix, axis=1)
    accuracy_per_class = np.nan_to_num(diagonal / row_sums)
    average_accuracy = np.mean(accuracy_per_class)
    return accuracy_per_class, average_accuracy


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def list_to_colormap(x_list, num_classes=16):
    """Convert label list to RGB colormap for visualization."""
    # Define color palette for classes
    colors = [
        [255, 0, 0],      # Class 0: Red
        [0, 255, 0],      # Class 1: Green
        [0, 0, 255],      # Class 2: Blue
        [255, 255, 0],    # Class 3: Yellow
        [0, 255, 255],    # Class 4: Cyan
        [255, 0, 255],    # Class 5: Magenta
        [192, 192, 192],  # Class 6: Silver
        [128, 128, 128],  # Class 7: Gray
        [128, 0, 0],      # Class 8: Maroon
        [128, 128, 0],    # Class 9: Olive
        [0, 128, 0],      # Class 10: Dark Green
        [128, 0, 128],    # Class 11: Purple
        [0, 128, 128],    # Class 12: Teal
        [0, 0, 128],      # Class 13: Navy
        [255, 165, 0],    # Class 14: Orange
        [255, 215, 0],    # Class 15: Gold
        [0, 0, 0],        # Background: Black
    ]
    
    colormap = np.zeros((x_list.shape[0], 3))
    for index, item in enumerate(x_list):
        item_int = int(item)
        if 0 <= item_int < len(colors):
            colormap[index] = np.array(colors[item_int]) / 255.
        else:
            colormap[index] = np.array([0, 0, 0]) / 255.
    return colormap


def classification_map(map_data, ground_truth, dpi, save_path):
    """Generate and save classification map."""
    fig = plt.figure(frameon=False)
    fig.set_size_inches(ground_truth.shape[1] * 2.0 / dpi, ground_truth.shape[0] * 2.0 / dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)
    ax.imshow(map_data)
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def generate_classification_maps(all_iter, net, gt_hsi, dataset_name, device, total_indices):
    """Generate and save classification maps."""
    net.eval()
    predicted_labels = []
    
    with torch.no_grad():
        for X, _ in all_iter:
            X = X.to(device)
            y_hat = net(X)
            predicted_labels.extend(y_hat.cpu().argmax(axis=1).numpy())
    
    gt = gt_hsi.flatten()
    label_map = np.zeros(gt.shape)
    gt_labels = gt[total_indices] - 1
    label_map[total_indices] = predicted_labels
    
    # Generate colormaps
    colormap_predicted = list_to_colormap(label_map)
    colormap_gt = list_to_colormap(gt - 1)
    
    # Handle background
    colormap_modified = []
    for pred_label, gt_label in zip(label_map, gt - 1):
        if gt_label == -1:  # Background
            colormap_modified.append(16)
        else:
            colormap_modified.append(int(pred_label))
    colormap_modified = np.array(colormap_modified)
    
    # Reshape for visualization
    reshaped_predicted = np.reshape(colormap_predicted, (gt_hsi.shape[0], gt_hsi.shape[1], 3))
    reshaped_gt = np.reshape(colormap_gt, (gt_hsi.shape[0], gt_hsi.shape[1], 3))
    reshaped_modified = np.reshape(list_to_colormap(colormap_modified), (gt_hsi.shape[0], gt_hsi.shape[1], 3))
    
    # Save maps
    save_dir = f'./content/classification_maps/'
    os.makedirs(save_dir, exist_ok=True)
    
    classification_map(reshaped_predicted, gt_hsi, 150, f'{save_dir}{dataset_name}_predicted.png')
    classification_map(reshaped_gt, gt_hsi, 150, f'{save_dir}{dataset_name}_ground_truth.png')
    classification_map(reshaped_modified, gt_hsi, 150, f'{save_dir}{dataset_name}_modified.png')
    
    print(f"Classification maps saved to {save_dir}")
    
    # Calculate accuracies
    valid_indices = gt.flatten() > 0
    acc_with_bg = np.mean(label_map[valid_indices] == (gt[valid_indices] - 1))
    print(f"Classification accuracy: {acc_with_bg * 100:.2f}%")


def record_output(overall_accuracies, average_accuracies, kappas, element_accuracies,
                  training_times, testing_times, confusion_matrix, path):
    """Record evaluation results to a text file."""
    with open(path, 'w') as f:
        f.write('=' * 60 + '\n')
        f.write('SSRN-ViT HYPERSPECTRAL CLASSIFICATION RESULTS\n')
        f.write('=' * 60 + '\n\n')
        
        f.write(f'OAs for each iteration: {overall_accuracies}\n')
        f.write(f'AAs for each iteration: {average_accuracies}\n')
        f.write(f'KAPPAs for each iteration: {kappas}\n\n')
        
        f.write(f'mean_OA ± std_OA: {np.mean(overall_accuracies):.4f} ± {np.std(overall_accuracies):.4f}\n')
        f.write(f'mean_AA ± std_AA: {np.mean(average_accuracies):.4f} ± {np.std(average_accuracies):.4f}\n')
        f.write(f'mean_KAPPA ± std_KAPPA: {np.mean(kappas):.4f} ± {np.std(kappas):.4f}\n\n')
        
        f.write(f'Total training time: {np.sum(training_times):.2f}s\n')
        f.write(f'Total testing time: {np.sum(testing_times):.2f}s\n\n')
        
        f.write('Per-class accuracies:\n')
        f.write(f'Mean: {np.mean(element_accuracies, axis=0)}\n')
        f.write(f'Std: {np.std(element_accuracies, axis=0)}\n\n')
        
        f.write('Confusion Matrix:\n')
        f.write(str(confusion_matrix) + '\n')
    
    print(f"Results saved to {path}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Random seeds for reproducibility
    seeds = [1331, 1332, 1333, 1334, 1335, 1336, 1337, 1338, 1339, 1340, 1341]
    
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    
    for iteration_index in range(ITER):
        print(f'\n{"="*60}')
        print(f'ITERATION {iteration_index + 1}/{ITER}')
        print(f'{"="*60}')
        
        # Set random seeds
        np.random.seed(seeds[iteration_index])
        torch.manual_seed(seeds[iteration_index])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seeds[iteration_index])
        
        # Initialize hybrid model
        model = SSRN_ViT_Hybrid(
            band=BAND,
            classes=CLASSES_NUM,
            patch_size=img_rows,
            vit_dim=64,
            vit_depth=4,
            vit_heads=8,
            vit_mlp_dim=128,
            vit_dropout=0.1,
            vit_emb_dropout=0.1
        )
        model = model.to(device)
        
        # Print model summary
        print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        
        # Data sampling - Fixed: proper train/test split
        train_indices, test_indices = sampling(VALIDATION_SPLIT, gt)
        _, total_indices = sampling(1, gt)
        _, total_indicesbg = sampling(1, gt, bg=True)
        TOTAL_SIZEBG = 21025
        
        train_size = len(train_indices)
        test_size = len(test_indices)
        
        print(f'\nDataset split:')
        print(f'  Training samples: {train_size}')
        print(f'  Test samples: {test_size}')
        print(f'  Total labeled samples: {TOTAL_SIZE}')
        
        # Generate data loaders
        print('\nCreating data loaders...')
        train_iter, valid_iter, test_iter, all_iter, all_iter_bg = generate_iter(
            train_size, train_indices, test_size, test_indices,
            TOTAL_SIZE, total_indices, TOTAL_SIZEBG, total_indicesbg,
            VAL_SIZE, whole_data, PATCH_LENGTH, padded_data,
            INPUT_DIMENSION, batch_size, gt
        )
        
        # Train model
        print('\nStarting training...')
        training_start = time.time()
        train(model, train_iter, valid_iter, loss, optimizer, device, epochs=num_epochs)
        training_end = time.time()
        
        # Test model
        print('\nEvaluating on test set...')
        testing_start = time.time()
        pred_test = []
        
        model.eval()
        with torch.no_grad():
            for X, y in test_iter:
                X = X.to(device)
                y_hat = model(X)
                pred_test.extend(y_hat.cpu().argmax(axis=1).numpy())
        testing_end = time.time()
        
        # Calculate metrics
        ground_truth_test = gt[test_indices] - 1
        overall_accuracy = metrics.accuracy_score(pred_test, ground_truth_test)
        conf_matrix = metrics.confusion_matrix(ground_truth_test, pred_test)
        each_class_accuracy, average_accuracy = aa_and_each_accuracy(conf_matrix)
        cohen_kappa = metrics.cohen_kappa_score(pred_test, ground_truth_test)
        
        # Print results
        print(f'\n{"="*60}')
        print('RESULTS')
        print(f'{"="*60}')
        print(f'Overall Accuracy (OA): {overall_accuracy:.4f}')
        print(f'Average Accuracy (AA): {average_accuracy:.4f}')
        print(f"Cohen's Kappa: {cohen_kappa:.4f}")
        print(f'Training time: {training_end - training_start:.2f}s')
        print(f'Testing time: {testing_end - testing_start:.2f}s')
        print(f'\nPer-class accuracies:')
        for i, acc in enumerate(each_class_accuracy):
            print(f'  Class {i + 1:2d}: {acc:.4f}')
        
        # Save model
        model_path = f'./content/SSRN_ViT_IN_iter{iteration_index}.pt'
        torch.save(model.state_dict(), model_path)
        print(f'\nModel saved to {model_path}')
        
        # Store metrics
        KAPPA.append(cohen_kappa)
        OA.append(overall_accuracy)
        AA.append(average_accuracy)
        TRAINING_TIME.append(training_end - training_start)
        TESTING_TIME.append(testing_end - testing_start)
        ELEMENT_ACC[iteration_index, :] = each_class_accuracy
    
    # Save final results
    print('\n' + '=' * 60)
    print('SAVING0)
    
    # Save final model
    torch.save(model.state_dict(), './content/SSRN_ViT_Hyperspectral_IN_final.pt')
    print('Final model saved to ./content/SSRN_ViT_Hyperspectral_IN_final.pt')
    
    # Record results
    record_output(
        OA, AA, KAPPA, ELEMENT_ACC, TRAINING_TIME, TESTING_TIME, conf_matrix,
        f'./content/results_{img_rows}_{Dataset}_split_{VALIDATION_SPLIT}_lr_{lr}.txt'
    )
    
    # Generate classification maps
    print('\nGenerating classification maps...')
    generate_classification_maps(all_iter_bg, model, gt_hsi, Dataset, device, total_indicesbg)
    
    print('\n' + '=' * 60)
    print('COMPLETED SUCCE FINAL RESULTS')
    print('=' * 6SSFULLY')
    print('=' * 60)
