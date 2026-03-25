"""
Module: models/DeepLabv3_plus.py
Purpose: DeepLabv3+ architecture with ResNet101 backbone for semantic segmentation of PCB images.

This module implements the DeepLabv3+ model (Chen et al., 2018) optimized for PCB component
segmentation into 4 classes: Background, Component, IC, and Connector.

Architecture Overview:
    ┌──────────────┐
    │ Input Image  │ (H×W×C)
    └──────┬───────┘
           │
    ┌──────▼─────────┐
    │ ResNet101      │ Pretrained encoder with atrous convolutions
    │  - layer1      │ → Low-level features (H/4×W/4×256)
    │  - layer4      │ → High-level features (H/16×W/16×2048)
    └──────┬─────────┘
           │
    ┌──────▼─────────┐
    │ ASPP Module    │ Multi-scale context aggregation
    │  - 5 branches  │ Parallel: 1×1, 3×3(d=6), 3×3(d=12), 3×3(d=18), Global Pool
    └──────┬─────────┘
           │
    ┌──────▼─────────┐
    │ Decoder        │ Fuse high-level + low-level features
    │  - Upsample 4× │ 
    │  - Concat      │
    │  - Conv        │
    │  - Upsample 4× │
    └──────┬─────────┘
           │
    ┌──────▼─────────┐
    │ Segmentation   │ (H×W×num_classes)
    └────────────────┘

Key Components:
    - Bottleneck: ResNet building block with atrous (dilated) convolution support
    - ResNet: Modified ResNet101 backbone with configurable output stride (OS)
    - ASPP_module: Atrous Spatial Pyramid Pooling for multi-scale feature extraction
    - DeepLabv3_plus: Full encoder-decoder architecture

Performance:
    - RGB PCB Dataset: 75% Mean IoU (baseline), 78%+ with Copy-Paste + Scale Jittering
    - HSI PCB Dataset: 40% Mean IoU (limited by data scarcity)

Code Credits:
    1. Chen et al., 2018
       "Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation"
       Source: https://arxiv.org/abs/1802.02611
    
    2. doiken23
       Inspired by: https://github.com/doiken23/DeepLab_pytorch/blob/master/DeepLab_v3_plus.py

Author: BS Thesis - PCB Vision Project
Date: January 2026
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    """
    ResNet Bottleneck block with optional atrous (dilated) convolution.
    
    This is the fundamental building block of ResNet, implementing the identity mapping
    with residual connections. For DeepLabv3+, we use atrous convolutions to control
    the receptive field without downsampling, which preserves spatial resolution.
    
    Architecture:
        Input (inplanes channels)
            ↓
        1×1 Conv (reduce to 'planes' channels)  ← Dimensionality reduction
            ↓
        3×3 Conv (dilated if rate>1)            ← Spatial feature extraction
            ↓
        1×1 Conv (expand to 'planes×4')         ← Dimensionality expansion
            ↓
        Add Residual (+ input or downsampled input)
            ↓
        ReLU → Output (planes×4 channels)
    
    Atrous Convolution:
        - Standard: rate=1 (normal 3×3 convolution)
        - Atrous: rate>1 (dilated convolution with wider receptive field)
        - Effective receptive field = (kernel_size-1) × rate + 1
        - Example: rate=2 → 3×3 conv sees 5×5 area without pooling
    
    Attributes:
        expansion (int): Channel expansion factor (always 4 for Bottleneck)
        conv1 (nn.Conv2d): 1×1 reduction layer
        bn1 (nn.BatchNorm2d): Batch norm after conv1
        conv2 (nn.Conv2d): 3×3 atrous convolution layer
        bn2 (nn.BatchNorm2d): Batch norm after conv2
        conv3 (nn.Conv2d): 1×1 expansion layer
        bn3 (nn.BatchNorm2d): Batch norm after conv3
        relu (nn.ReLU): ReLU activation
        downsample (nn.Module): Optional projection for residual connection
        stride (int): Stride for spatial downsampling
        rate (int): Dilation rate for atrous convolution
    
    Args:
        inplanes (int): Number of input channels
        planes (int): Number of bottleneck channels (output will be planes×4)
        stride (int): Stride for downsampling. Default: 1 (no downsampling)
        rate (int): Dilation rate for atrous convolution. Default: 1 (standard conv)
        downsample (nn.Module): Module to match dimensions for residual. Default: None
    
    Returns:
        torch.Tensor: Output feature map of shape (N, planes×4, H', W')
            where H' = H/stride, W' = W/stride
    
    Example:
        >>> block = Bottleneck(inplanes=64, planes=64, stride=1, rate=2)
        >>> x = torch.randn(1, 64, 56, 56)
        >>> out = block(x)  # Shape: (1, 256, 56, 56)  [256 = 64×4]
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, rate=1, downsample=None):
        super(Bottleneck, self).__init__()
        
        # 1×1 convolution: Reduce dimensionality
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        
        # 3×3 atrous convolution: Spatial feature extraction with controlled receptive field
        # - dilation=rate controls the spacing between kernel elements
        # - padding=rate ensures output size matches input (when stride=1)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               dilation=rate, padding=rate, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        # 1×1 convolution: Expand dimensionality by 4×
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample  # Used when input/output dimensions mismatch
        self.stride = stride
        self.rate = rate

    def forward(self, x):
        """
        Forward pass through the bottleneck block.
        
        Algorithm Steps:
            1. Save input as residual for skip connection
            2. Apply 1×1 conv → BN → ReLU (dimensionality reduction)
            3. Apply 3×3 atrous conv → BN → ReLU (spatial features)
            4. Apply 1×1 conv → BN (dimensionality expansion, no ReLU yet)
            5. If necessary, downsample residual to match output dimensions
            6. Add residual connection: output = conv_output + residual
            7. Apply final ReLU
        
        Args:
            x (torch.Tensor): Input tensor of shape (N, inplanes, H, W)
        
        Returns:
            torch.Tensor: Output tensor of shape (N, planes×4, H', W')
        
        Shape:
            - Input: (N, inplanes, H, W)
            - Output: (N, planes×4, H/stride, W/stride)
        """
        # Store input for residual connection
        residual = x

        # Step 1: 1×1 conv (reduce channels)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # Step 2: 3×3 atrous conv (extract spatial features)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        # Step 3: 1×1 conv (expand channels)
        out = self.conv3(out)
        out = self.bn3(out)

        # Step 4: Match residual dimensions if needed
        # (happens when stride>1 or when input/output channels differ)
        if self.downsample is not None:
            residual = self.downsample(x)

        # Step 5: Add residual connection (the key to ResNet's success)
        out += residual
        
        # Step 6: Final activation
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """
    Modified ResNet101 backbone for DeepLabv3+ with atrous convolutions.
    
    This implements a ResNet101 encoder that replaces standard convolutions with
    atrous convolutions in later layers to maintain spatial resolution while increasing
    receptive field. This is crucial for dense prediction tasks like segmentation.
    
    Key Modifications from Standard ResNet101:
        1. Configurable Output Stride (OS): Controls final feature map resolution
           - OS=16: Features are 16× smaller than input (H/16 × W/16)
           - OS=8: Features are 8× smaller (better resolution, more compute)
        
        2. Atrous Convolutions: Replace pooling with dilated convolutions
           - Preserves spatial information
           - Increases receptive field without adding parameters
        
        3. Multi-Grid (MG): layer4 uses different dilation rates [1,2,4]
           - Provides multi-scale features within a single layer
    
    Architecture:
        Input (H×W×C)
            ↓
        conv1 (7×7, stride=2) → (H/2×W/2×64)
            ↓
        maxpool (3×3, stride=2) → (H/4×W/4×64)
            ↓
        layer1 [3 blocks] → (H/4×W/4×256)
            ↓
        layer2 [4 blocks] → (H/8×W/8×512) if OS=8, else (H/8×W/8×512)
            ↓
        layer3 [23 blocks] → (H/16 or H/8)
            ↓
        layer4 [3 blocks, MG] → (H/16 or H/8×2048) ← Used by ASPP
    
    Args:
        nInputChannels (int): Number of input channels (3 for RGB, 214 for HSI)
        block (class): Bottleneck block class
        layers (list): Number of blocks in each layer [3, 4, 23, 3] for ResNet101
        os (int): Output stride (8 or 16). Default: 16
        pretrained (bool): Load ImageNet pretrained weights. Default: False
    
    Attributes:
        inplanes (int): Current number of feature channels (tracks layer progression)
        conv1: Initial 7×7 convolution
        bn1: Batch normalization after conv1
        relu: ReLU activation
        maxpool: 3×3 max pooling
        layer1-4: ResNet layers with increasing depth
    
    Returns:
        tuple: (high_level_features, low_level_features)
            - high_level_features: (N, 2048, H/OS, W/OS) from layer4
            - low_level_features: (N, 256, H/4, W/4) from layer1
    
    Example:
        >>> model = ResNet(nInputChannels=3, block=Bottleneck, 
        ...                layers=[3,4,23,3], os=16, pretrained=True)
        >>> x = torch.randn(1, 3, 640, 640)
        >>> high_feat, low_feat = model(x)
        >>> print(high_feat.shape, low_feat.shape)
        torch.Size([1, 2048, 40, 40]) torch.Size([1, 256, 160, 160])
    """

    def __init__(self, nInputChannels, block, layers, os=16, pretrained=False):
        self.inplanes = 64  # Track current channel count as we build layers
        super(ResNet, self).__init__()
        
        # Configure strides, dilation rates, and multi-grid based on output stride
        # Output Stride (OS) determines final resolution: H_out = H_in / OS
        if os == 16:
            # Standard configuration: downsample in layer2 and layer3
            strides = [1, 2, 2, 1]  # layer1, layer2, layer3, layer4
            rates = [1, 1, 1, 2]    # Use atrous conv (rate=2) only in layer4
            blocks = [1, 2, 4]      # Multi-grid for layer4: rates × [1,2,4]
        elif os == 8:
            # Higher resolution: downsample only in layer2, use atrous in layer3+layer4
            strides = [1, 2, 1, 1]  # Stop downsampling after layer2
            rates = [1, 1, 2, 2]    # Use atrous conv in layer3 and layer4
            blocks = [1, 2, 1]      # Multi-grid for layer4
        else:
            raise NotImplementedError(f"Output stride {os} not supported. Use 8 or 16.")

        # Initial layers: Reduce spatial resolution by 4× (H/4 × W/4)
        # 7×7 conv with stride=2 → H/2 × W/2
        # maxpool with stride=2 → H/4 × W/4
        self.conv1 = nn.Conv2d(nInputChannels, 64, kernel_size=7, stride=2, padding=3,
                                bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Build ResNet layers
        # layer1: (H/4 × W/4 × 256) - Low-level features for decoder
        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], rate=rates[0])
        # layer2: (H/8 × W/8 × 512)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], rate=rates[1])
        # layer3: (H/OS × W/OS × 1024) - OS-dependent resolution
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], rate=rates[2])
        # layer4: (H/OS × W/OS × 2048) - High-level features with multi-grid atrous convolutions
        self.layer4 = self._make_MG_unit(block, 512, blocks=blocks, stride=strides[3], rate=rates[3])

        self._init_weight()

        if pretrained:
            self._load_pretrained_model()

    def _make_layer(self, block, planes, blocks, stride=1, rate=1):
        """
        Build a ResNet layer consisting of multiple Bottleneck blocks.
        
        Args:
            block: Bottleneck class
            planes (int): Number of intermediate channels
            blocks (int): Number of blocks in this layer
            stride (int): Stride for first block. Default: 1
            rate (int): Dilation rate for atrous convolution. Default: 1
        
        Returns:
            nn.Sequential: Layer with 'blocks' number of Bottleneck blocks
        """
        downsample = None
        # Create projection shortcut if dimensions change
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        # First block may downsample and/or have atrous convolution
        layers.append(block(self.inplanes, planes, stride, rate, downsample))
        self.inplanes = planes * block.expansion
        # Remaining blocks have stride=1 and same dilation rate
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_MG_unit(self, block, planes, blocks=[1,2,4], stride=1, rate=1):
        """
        Build Multi-Grid layer with varying dilation rates.
        
        Multi-Grid applies different dilation rates within a single layer,
        providing multi-scale features without additional depth.
        
        Example: blocks=[1,2,4], rate=2
            - Block 1: dilation = 1×2 = 2
            - Block 2: dilation = 2×2 = 4  
            - Block 3: dilation = 4×2 = 8
        
        Args:
            block: Bottleneck class
            planes (int): Number of intermediate channels
            blocks (list): List of dilation multipliers. Default: [1,2,4]
            stride (int): Stride for first block. Default: 1
            rate (int): Base dilation rate. Default: 1
        
        Returns:
            nn.Sequential: Multi-grid layer
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        # First block: dilation = blocks[0] × rate
        layers.append(block(self.inplanes, planes, stride, rate=blocks[0]*rate, downsample=downsample))
        self.inplanes = planes * block.expansion
        # Subsequent blocks: dilation = blocks[i] × rate
        for i in range(1, len(blocks)):
            layers.append(block(self.inplanes, planes, stride=1, rate=blocks[i]*rate))

        return nn.Sequential(*layers)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        low_level_feat = x
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x, low_level_feat

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _load_pretrained_model(self):
        pretrain_dict = torch.hub.load_state_dict_from_url(
            'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
            progress=True,
        )
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

def ResNet101(nInputChannels=3, os=16, pretrained=False):
    model = ResNet(nInputChannels, Bottleneck, [3, 4, 23, 3], os, pretrained=pretrained)
    return model


class ASPP_module(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (ASPP) module.
    
    ASPP applies parallel atrous convolutions with different dilation rates
    to capture multi-scale contextual information. This is one of the key
    innovations of DeepLabv3/v3+.
    
    Function:
        - rate=1: Captures local features (small receptive field)
        - rate=6,12,18: Captures context at increasing scales
        - Global pooling: Captures image-level context
    
    Args:
        inplanes (int): Number of input channels (typically 2048 from ResNet)
        planes (int): Number of output channels (typically 256)
        rate (int): Dilation rate for atrous convolution
    
    Note:
        When rate=1, uses 1×1 conv (no dilation). Otherwise uses 3×3 conv
        with dilation=rate to maintain same receptive field as standard conv.
    """
    def __init__(self, inplanes, planes, rate):
        super(ASPP_module, self).__init__()
        if rate == 1:
            kernel_size = 1
            padding = 0
        else:
            kernel_size = 3
            padding = rate  # padding = rate maintains spatial dimensions
        self.atrous_convolution = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=rate, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class DeepLabv3_plus(nn.Module):
    """
    DeepLabv3+ complete architecture for semantic segmentation.
    
    This is the full encoder-decoder model that combines:
    - ResNet101 encoder for feature extraction
    - ASPP for multi-scale context aggregation
    - Decoder for sharp boundary recovery via skip connections
    
    Architecture Pipeline:
        1. ENCODER: Input → ResNet101 → (high_features, low_features)
        2. ASPP: high_features → 5 parallel branches → concatenate → 256 channels
        3. DECODER:
           - Upsample ASPP output 4× (H/16 → H/4)
           - Refine low_features to 48 channels
           - Concatenate: [256 + 48] = 304 channels
           - Conv blocks → n_classes channels
           - Upsample 4× to original resolution (H/4 → H)
    
    Key Features:
        - Skip connections preserve spatial detail
        - Multi-scale ASPP captures context at multiple receptive fields
        - Lightweight decoder for efficiency
    
    Args:
        nInputChannels (int): Number of input channels. Default: 3 (RGB)
        n_classes (int): Number of output classes. Default: 4 (PCB classes)
        os (int): Output stride (8 or 16). Default: 16
        pretrained (bool): Load ImageNet pretrained ResNet. Default: False
        _print (bool): Print initialization info. Default: True
    
    Attributes:
        resnet_features: ResNet101 encoder
        aspp1-5: ASPP module branches
        global_avg_pool: Global context branch
        conv1, bn1: ASPP output reduction layer
        conv2, bn2: Low-level feature refinement
        last_conv: Final decoder convolution block
    
    Returns:
        torch.Tensor: Segmentation logits of shape (N, n_classes, H, W)
    
    Example:
        >>> model = DeepLabv3_plus(nInputChannels=3, n_classes=4, pretrained=True)
        >>> x = torch.randn(1, 3, 640, 640)
        >>> out = model(x)  # Shape: (1, 4, 640, 640)
        >>> pred = torch.argmax(out, dim=1)  # Class predictions
    """
    def __init__(self, nInputChannels=3, n_classes=4, os=16, pretrained=False, _print=True):
        if _print:
            print("Constructing DeepLabv3+ model...")
            print("Number of classes: {}".format(n_classes))
            print("Output stride: {}".format(os))
            print("Number of Input Channels: {}".format(nInputChannels))
        super(DeepLabv3_plus, self).__init__()

        if _print and pretrained and nInputChannels != 3:
            print("Note: pretrained ResNet101 weights are for 3-channel inputs; conv1 will be randomly initialized.")

        # ENCODER: ResNet101 backbone
        self.resnet_features = ResNet101(nInputChannels, os, pretrained=pretrained)

        # ASPP: Configure dilation rates based on output stride
        if os == 16:
            rates = [1, 6, 12, 18]  # Standard dilation rates for OS=16
        elif os == 8:
            rates = [1, 12, 24, 36]  # Wider dilation for OS=8 (need larger receptive fields)
        else:
            raise NotImplementedError(f"Output stride {os} not supported")

        # ASPP branches: 4 parallel atrous convolutions
        self.aspp1 = ASPP_module(2048, 256, rate=rates[0])  # 1×1 conv (local features)
        self.aspp2 = ASPP_module(2048, 256, rate=rates[1])  # 3×3 atrous (medium context)
        self.aspp3 = ASPP_module(2048, 256, rate=rates[2])  # 3×3 atrous (large context)
        self.aspp4 = ASPP_module(2048, 256, rate=rates[3])  # 3×3 atrous (very large context)

        self.relu = nn.ReLU()

        # ASPP branch 5: Global average pooling (image-level features)
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(2048, 256, 1, stride=1, bias=False),
                                             nn.BatchNorm2d(256),
                                             nn.ReLU())

        # ASPP output: Reduce concatenated features (5×256=1280) to 256 channels
        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)

        # DECODER: Low-level feature refinement (256 → 48 channels)
        self.conv2 = nn.Conv2d(256, 48, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(48)

        # DECODER: Final convolution block (304 → 256 → n_classes)
        # Input: Concatenated features [256 from ASPP + 48 from low-level] = 304
        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.Conv2d(256, n_classes, kernel_size=1, stride=1))

        self._init_weight()

    def forward(self, input):
        x, low_level_features = self.resnet_features(input)
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = F.interpolate(
            x,
            size=(int(math.ceil(input.size()[-2] / 4)), int(math.ceil(input.size()[-1] / 4))),
            mode='bilinear',
            align_corners=True,
        )

        low_level_features = self.conv2(low_level_features)
        low_level_features = self.bn2(low_level_features)
        low_level_features = self.relu(low_level_features)


        x = torch.cat((x, low_level_features), dim=1)
        x = self.last_conv(x)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def _init_weight(self):
        # Initialize only the decoder/head layers in this module.
        # (The ResNet encoder and ASPP blocks already perform their own initialization.
        #  If pretrained=True, ResNet weights are loaded after its own init.)
        for m in [self.conv1, self.bn1, self.conv2, self.bn2, self.last_conv]:
            for mm in m.modules() if isinstance(m, nn.Module) else []:
                if isinstance(mm, nn.Conv2d):
                    torch.nn.init.kaiming_normal_(mm.weight)
                    if mm.bias is not None:
                        nn.init.zeros_(mm.bias)
                elif isinstance(mm, nn.BatchNorm2d):
                    mm.weight.data.fill_(1)
                    mm.bias.data.zero_()


def print_model_summary(model, input_size=(3, 256, 256)):
    """
    Print detailed model summary including layer-wise parameters.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (C, H, W)
    
    Usage:
        model = DeepLabv3_plus(nInputChannels=3, n_classes=4)
        print_model_summary(model, input_size=(3, 256, 256))
    """
    try:
        from torchinfo import summary  # type: ignore
        print("\nDetailed Model Summary (using torchinfo):\n")
        summary(model, input_size=(1, *input_size), 
                col_names=["input_size", "output_size", "num_params", "trainable"],
                depth=4)
    except ImportError:
        print("Note: Install torchinfo for detailed summary: pip install torchinfo\n")
        print("Using basic parameter count...\n")
        
        # Calculate total parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params
        
        print("=" * 80)
        print("PARAMETER SUMMARY")
        print("=" * 80)
        print(f"Total Parameters:       {total_params:,}")
        print(f"Trainable Parameters:   {trainable_params:,}")
        print(f"Non-trainable Params:   {non_trainable_params:,}")
        print(f"Model Size (MB):        {total_params * 4 / (1024**2):.2f}")
        print("=" * 80)
        
        # Layer-wise breakdown
        print("\nLAYER-WISE PARAMETER COUNT:")
        print("-" * 80)
        print(f"{'Layer Name':<40} {'Total Params':>15} {'Trainable':>15}")
        print("-" * 80)
        for name, module in model.named_children():
            num_params = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            print(f"{name:<40} {num_params:>15,} {trainable:>15,}")
        print("-" * 80)


# Example usage when running as script
if __name__ == "__main__":
    print("\n" + "="*80)
    print("RGB DeepLabv3+ Model (3 channels → 4 classes)")
    print("="*80)
    model_rgb = DeepLabv3_plus(nInputChannels=3, n_classes=4)
    print_model_summary(model_rgb, input_size=(3, 256, 256))
    
    print("\n\n" + "="*80)
    print("HSI DeepLabv3+ Model (214 channels → 4 classes)")
    print("="*80)
    model_hsi = DeepLabv3_plus(nInputChannels=214, n_classes=4)
    print_model_summary(model_hsi, input_size=(214, 256, 256))
