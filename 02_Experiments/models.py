"""
models.py  v2
─────────────
All three models updated to properly handle arbitrary input channels
including RGB+HSI (227 channels) while preserving pretrained weights.

Key fix: DeepLabV3+ now uses pretrained ImageNet weights for ALL input sizes
by modifying only the first conv layer and initializing extra channels
with the mean of the pretrained RGB weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import math


# ─────────────────────────────────────────────────────────────────────────────
# 1. DeepLabV3+ — pretrained backbone for any channel count
# ─────────────────────────────────────────────────────────────────────────────

def get_deeplabv3(in_channels: int, num_classes: int) -> nn.Module:
    """
    DeepLabV3+ with ResNet-50 backbone.

    For in_channels == 3: standard ImageNet pretrained weights.
    For in_channels != 3: load pretrained 3-ch model, then replace
    first conv to accept in_channels. Extra channels beyond 3 are
    initialized with the mean of the pretrained RGB weights so the
    model starts from a sensible point rather than random noise.

    This preserves ALL deeper pretrained weights (layer1-4, ASPP,
    decoder) and only requires learning the channel projection.
    """
    # Always load pretrained 3-channel model
    model = smp.DeepLabV3Plus(
        encoder_name    = "resnet50",
        encoder_weights = "imagenet",
        in_channels     = 3,
        classes         = num_classes,
    )

    if in_channels != 3:
        old_conv = model.encoder.conv1  # pretrained (64, 3, 7, 7)

        new_conv = nn.Conv2d(
            in_channels, 64,
            kernel_size=7, stride=2, padding=3, bias=False
        )

        with torch.no_grad():
            # Copy exact pretrained RGB weights for first 3 channels
            new_conv.weight[:, :3, :, :] = old_conv.weight.clone()

            if in_channels > 3:
                # Initialize extra channels with mean of RGB weights
                # This gives a sensible spectral baseline rather than
                # random initialization
                mean_w = old_conv.weight.mean(dim=1, keepdim=True)
                new_conv.weight[:, 3:, :, :] = mean_w.expand(
                    -1, in_channels - 3, -1, -1).clone()

        model.encoder.conv1 = new_conv
        print(f"  DeepLabV3+ first conv: 3 → {in_channels} channels "
              f"(pretrained RGB + mean-init for extra channels)")

    return model


# ─────────────────────────────────────────────────────────────────────────────
# 2. Hybrid SSRN-ViT — already handles arbitrary channels natively
# ─────────────────────────────────────────────────────────────────────────────

class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 padding, use_1x1conv=False, stride=1):
        super().__init__()
        self.conv1    = nn.Conv3d(in_channels, out_channels,
                                  kernel_size=kernel_size, padding=padding,
                                  stride=stride)
        self.bn1      = nn.BatchNorm3d(out_channels)
        self.conv2    = nn.Conv3d(out_channels, out_channels,
                                  kernel_size=kernel_size, padding=padding,
                                  stride=stride)
        self.bn2      = nn.BatchNorm3d(out_channels)
        self.shortcut = nn.Conv3d(in_channels, out_channels,
                                   kernel_size=1, stride=stride) \
                        if use_1x1conv else None

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.shortcut is not None:
            identity = self.shortcut(x)
        return F.relu(out + identity)


class SpectralSpatialEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 24):
        super().__init__()
        self.conv1 = nn.Conv3d(1, hidden_channels,
                               kernel_size=(1, 1, 7), stride=(1, 1, 2))
        self.bn1   = nn.Sequential(nn.BatchNorm3d(hidden_channels),
                                   nn.ReLU(inplace=True))
        self.res1  = ResidualBlock3D(hidden_channels, hidden_channels,
                                     (1,1,7), (0,0,3))
        self.res2  = ResidualBlock3D(hidden_channels, hidden_channels,
                                     (1,1,7), (0,0,3))
        self.res3  = ResidualBlock3D(hidden_channels, hidden_channels,
                                     (3,3,1), (1,1,0))
        self.res4  = ResidualBlock3D(hidden_channels, hidden_channels,
                                     (3,3,1), (1,1,0))
        kernel_3d  = max(1, math.ceil((in_channels - 6) / 2))
        self.conv2 = nn.Conv3d(hidden_channels, 128,
                               kernel_size=(1, 1, kernel_3d))
        self.bn2   = nn.Sequential(nn.BatchNorm3d(128), nn.ReLU(inplace=True))
        self.conv3 = nn.Conv3d(1, hidden_channels,
                               kernel_size=(3, 3, 128), padding=(0,0,0))
        self.bn3   = nn.Sequential(nn.BatchNorm3d(hidden_channels),
                                   nn.ReLU(inplace=True))

    def forward(self, x):
        B, C, H, W = x.shape
        if C < 7:
            pad = torch.zeros(B, 7-C, H, W, device=x.device)
            x   = torch.cat([x, pad], dim=1)
        x = x.unsqueeze(1)
        x = x.permute(0, 1, 3, 4, 2)
        x = self.bn1(self.conv1(x))
        x = self.res1(x); x = self.res2(x)
        x = self.res3(x); x = self.res4(x)
        x = self.bn2(self.conv2(x))
        x = x.permute(0, 4, 2, 3, 1)
        x = self.bn3(self.conv3(x))
        return x.squeeze(4)


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.ln1  = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads,
                                          dropout=dropout, batch_first=True)
        self.ln2  = nn.LayerNorm(dim)
        self.mlp  = nn.Sequential(
            nn.Linear(dim, mlp_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        res = x; x = self.ln1(x)
        x, _ = self.attn(x, x, x); x = res + x
        res = x; x = self.ln2(x)
        x = self.mlp(x); return res + x


class ViTEncoder(nn.Module):
    def __init__(self, image_size, patch_size, dim, depth, heads,
                 mlp_dim, channels=24, dropout=0.1):
        super().__init__()
        num_patches      = (image_size // patch_size) ** 2
        patch_dim        = channels * patch_size ** 2
        self.patch_size  = patch_size
        self.patch_embed = nn.Linear(patch_dim, dim)
        self.pos_embed   = nn.Parameter(torch.randn(1, num_patches+1, dim))
        self.cls_token   = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout     = nn.Dropout(dropout)
        self.transformer = nn.ModuleList([
            TransformerBlock(dim, heads, dim//heads, mlp_dim, dropout)
            for _ in range(depth)])
        self.norm        = nn.LayerNorm(dim)

    def forward(self, x):
        B, C, H, W = x.shape; p = self.patch_size
        x = x.unfold(2,p,p).unfold(3,p,p)
        x = x.contiguous().view(B, C, -1, p, p)
        x = x.permute(0,2,3,4,1).contiguous()
        x = x.view(B, -1, p*p*C)
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)
        x   = x + self.pos_embed[:, :x.size(1)]
        x   = self.dropout(x)
        for blk in self.transformer: x = blk(x)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, in_channels, num_classes, image_size, patch_size):
        super().__init__()
        self.num_patches_side = image_size // patch_size
        self.proj = nn.Linear(in_channels, 64)
        self.up1  = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        self.up2  = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        self.up3  = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32),
            nn.ReLU(inplace=True), nn.Conv2d(32, num_classes, 1))

    def forward(self, x, original_size):
        B = x.size(0); x = x[:, 1:]
        x = self.proj(x)
        x = x.view(B, self.num_patches_side, self.num_patches_side, -1)
        x = x.permute(0,3,1,2)
        x = self.up1(x); x = self.up2(x); x = self.up3(x)
        return F.interpolate(x, size=(original_size, original_size),
                             mode='bilinear', align_corners=False)


class HybridSSRNSeg(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, image_size: int = 256):
        super().__init__()
        self.image_size = image_size
        self.encoder    = SpectralSpatialEncoder(in_channels, hidden_channels=24)
        enc_out_size    = image_size - 2
        self.vit        = ViTEncoder(image_size=enc_out_size, patch_size=14,
                                     dim=512, depth=6, heads=8, mlp_dim=1024)
        self.decoder    = Decoder(in_channels=512, num_classes=num_classes,
                                  image_size=enc_out_size, patch_size=14)

    def forward(self, x):
        x = self.encoder(x)
        x = self.vit(x)
        return self.decoder(x, self.image_size)


# ─────────────────────────────────────────────────────────────────────────────
# 3. MambaHSI — updated with channel attention for better multi-channel fusion
# ─────────────────────────────────────────────────────────────────────────────

class ChannelAttention(nn.Module):
    """
    Squeeze-and-Excitation channel attention.
    Helps the model learn which of the 227 input channels are most
    informative for each class — especially useful for RGB+HSI fusion.
    """
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        mid = max(in_channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.fc(x).unsqueeze(-1).unsqueeze(-1)
        return x * w


class MambaHSI_Wrapper(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        # Channel attention to weight RGB vs HSI channels
        self.channel_attn = ChannelAttention(in_channels)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, 1),
        )

    def forward(self, x):
        x = self.channel_attn(x)   # learn channel importance
        return self.decoder(self.encoder(x))


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_model(model_name: str, in_channels: int,
                num_classes: int, image_size: int = 256) -> nn.Module:
    if model_name == 'DeepLabV3+':
        return get_deeplabv3(in_channels, num_classes)
    elif model_name == 'Hybrid SSRN-ViT':
        return HybridSSRNSeg(in_channels, num_classes, image_size)
    elif model_name == 'MambaHSI':
        return MambaHSI_Wrapper(in_channels, num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")