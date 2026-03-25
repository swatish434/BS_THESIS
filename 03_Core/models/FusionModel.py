
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.DeepLabv3_plus import DeepLabv3_plus

class SimpleHSIEncoder(nn.Module):
    """
    Lightweight CNN encoder for HSI data.
    Input: (B, 214, H, W)
    Output: (B, 256, H/4, W/4) - Matching DeepLab low-level features
    """
    def __init__(self, in_channels=214, out_channels=256):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2) # H/2
        self.bn2 = nn.BatchNorm2d(128)
        
        self.conv3 = nn.Conv2d(128, out_channels, kernel_size=3, padding=1, stride=2) # H/4
        self.bn3 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        return x

class RGBHSIFusionModel(nn.Module):
    def __init__(self, num_classes=4, hsi_channels=214):
        super().__init__()
        
        # 1. RGB Stream (DeepLabv3+)
        # We reuse the DeepLabv3+ model but will intercept its features
        self.rgb_model = DeepLabv3_plus(nInputChannels=3, n_classes=num_classes)
        
        # 2. HSI Stream
        self.hsi_encoder = SimpleHSIEncoder(in_channels=hsi_channels, out_channels=256)
        
        # 3. Fusion Block
        # RGB Low-level features are 256 channels (from DeepLab resnet layer1) via projection
        # We want to fuse HSI features (256) with RGB features.
        # DeepLab decoder expects low_level_features (256) and encoder_output (2048 from ASPP)
        
        # Strategy: 
        # Fuse HSI features into the "low_level_features" of DeepLabv3+
        # Original DeepLab uses projection: layer1 -> conv1x1 -> 48 channels (or 256 depending on impl)
        # Let's check DeepLab implementation.
        
        # In our DeepLabv3_plus.py:
        # x_low = self.resnet_features.layer1(x) [256ch]
        # output = self.aspp(...)
        # x_low = self.conv2(x_low) [48ch] -> Check this in forward
        
        # We will Concatenate HSI features (256) with RGB layer1 (256) -> 512
        # Then project to 48 channels for the decoder
        
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(512, 48, kernel_size=1, bias=False), # Fuse 256(RGB) + 256(HSI) -> 48
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        
        # We override the DeepLabv3+ forward method logic here
        
    def forward(self, rgb, hsi):
        input_shape = rgb.shape[-2:]
        
        # --- RGB Stream ---
        # Encoder (ResNet) returns (high_level, low_level)
        # high_level: (B, 2048, H/16, W/16)
        # low_level:  (B, 256, H/4, W/4)
        x_high_rgb, x_low_rgb = self.rgb_model.resnet_features(rgb)
        
        # ASPP (High-level features)
        x1 = self.rgb_model.aspp1(x_high_rgb)
        x2 = self.rgb_model.aspp2(x_high_rgb)
        x3 = self.rgb_model.aspp3(x_high_rgb)
        x4 = self.rgb_model.aspp4(x_high_rgb)
        x5 = self.rgb_model.global_avg_pool(x_high_rgb)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        
        x_aspp = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x_aspp = self.rgb_model.conv1(x_aspp)
        x_aspp = self.rgb_model.bn1(x_aspp)
        x_aspp = self.rgb_model.relu(x_aspp)
        
        # Upsample ASPP features to match low-level features
        x_aspp = F.interpolate(x_aspp, size=x_low_rgb.size()[2:], mode='bilinear', align_corners=True)
        
        # --- HSI Stream ---
        x_low_hsi = self.hsi_encoder(hsi) # (B, 256, H/4, W/4)
        
        # --- Fusion ---
        # Resize HSI if dimensions differ slightly (e.g. padding)
        if x_low_hsi.size() != x_low_rgb.size():
            x_low_hsi = F.interpolate(x_low_hsi, size=x_low_rgb.size()[2:], mode='bilinear', align_corners=True)
            
        # Concatenate Low-level features (256 RGB + 256 HSI)
        x_low_fused = torch.cat([x_low_rgb, x_low_hsi], dim=1) 
        x_low_fused = self.fusion_conv(x_low_fused) # -> 48 channels
        
        # --- Decoder ---
        x_dec = torch.cat([x_aspp, x_low_fused], dim=1)
        x_dec = self.rgb_model.last_conv(x_dec)
        
        # Output
        x_out = F.interpolate(x_dec, size=input_shape, mode='bilinear', align_corners=True)
        return x_out
