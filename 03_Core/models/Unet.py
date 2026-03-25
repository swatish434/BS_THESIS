"""
Module: models/Unet.py
Purpose: U-Net architecture for semantic segmentation (baseline model).

U-Net is a classic encoder-decoder architecture with skip connections, originally
designed for biomedical image segmentation. Used as baseline comparison for DeepLabv3+.

Architecture: Symmetric encoder-decoder with skip connections
    - Encoder: 4 levels, downsampling with max pooling
    - Bottleneck: Deepest features (1024 channels)
    - Decoder: 4 levels, upsampling with transpose conv + skip connections

Performance:
    - RGB PCB Dataset: ~67% Mean IoU (baseline)
    - Faster training than DeepLabv3+ but lower accuracy

Code Credits:
    1. Ronneberger et al., 2015
       "U-Net: Convolutional Networks for Biomedical Image Segmentation"
       https://arxiv.org/abs/1505.04597
    2. Aladdin Persson
       Adapted binary segmentation to multiclass.
       https://github.com/aladdinpersson/Machine-Learning-Collection/

Author: BS Thesis - PCB Vision Project
Date: January 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """
    Double convolution block: Conv→BN→ReLU→Conv→BN→ReLU.
    
    Basic building block of U-Net applied at each encoder/decoder level.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
    
    Returns:
        torch.Tensor: Output features (N, out_channels, H, W)
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        # Two 3×3 convolutions with padding=1 to maintain spatial size
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):
    """
    U-Net architecture for semantic segmentation.
    
    Classic symmetric encoder-decoder with skip connections.
    
    Args:
        in_channels (int): Input channels. Default: 3 (RGB)
        out_channels (int): Number of classes. Default: 4 (PCB classes)
        features (list): Feature channels at each level. Default: [64,128,256,512]
    
    Returns:
        torch.Tensor: Segmentation logits (N, out_channels, H, W)
    
    Example:
        >>> model = UNET(in_channels=3, out_channels=4)
        >>> out = model(torch.randn(1, 3, 640, 640))  # (1, 4, 640, 640)
    """
    def __init__(self, in_channels=3, out_channels=4, features=[64, 128, 256, 512]):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Halves resolution

        # ENCODER: Build downsampling path
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # DECODER: Build upsampling path
        for feature in reversed(features):
            # Transpose conv: upsampling (doubles H,W)
            self.ups.append(
                nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)
            )
            # After concat with skip: feature*2 channels
            self.ups.append(DoubleConv(feature*2, feature))

        # BOTTLENECK: Deepest part of U
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        # FINAL: 1×1 conv to produce class logits
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        in_h, in_w = x.shape[-2:]
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=False)

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        x = self.final_conv(x)
        if x.shape[-2:] != (in_h, in_w):
            x = F.interpolate(x, size=(in_h, in_w), mode='bilinear', align_corners=False)
        return x
