"""
Code Credits:

1. Chaurasia et al., 2017
   - LinkNet: Exploiting Encoder Representations for Efficient Semantic Segmentation
   - Source: https://arxiv.org/abs/1707.03718

2. TOYGAR TANYEL
   - LinkNet Image Segmentation from scratch PyTorch
   - Source: https://www.kaggle.com/code/toygarr/linknet-image-segmentation-from-scratch-pytorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet as resnet_module

try:
    from torchvision.models import resnet18, resnet34
    from torchvision.models import ResNet18_Weights, ResNet34_Weights
except Exception:  # pragma: no cover
    resnet18 = None
    resnet34 = None
    ResNet18_Weights = None
    ResNet34_Weights = None
nonlinearity = nn.ReLU

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super().__init__()

        # B, C, H, W -> B, C/4, H, W
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity(inplace=True)

        # B, C/4, H, W -> B, C/4, H, W
        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3,
                                          stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity(inplace=True)

        # B, C/4, H, W -> B, C, H, W
        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


class LinkNet(nn.Module):
    def __init__(self, num_classes, num_channels=3, encoder='resnet34', pretrained=True):
        super().__init__()
        assert encoder in ['resnet18', 'resnet34']

        filters = [64, 128, 256, 512]

        if encoder == 'resnet18':
            if resnet18 is not None and ResNet18_Weights is not None:
                weights = ResNet18_Weights.DEFAULT if pretrained else None
                res = resnet18(weights=weights)
            else:
                res = resnet_module.resnet18(pretrained=pretrained)
        else:
            if resnet34 is not None and ResNet34_Weights is not None:
                weights = ResNet34_Weights.DEFAULT if pretrained else None
                res = resnet34(weights=weights)
            else:
                res = resnet_module.resnet34(pretrained=pretrained)

        # Support arbitrary input channel counts by adapting the first conv.
        if num_channels != 3:
            old_conv = res.conv1
            new_conv = nn.Conv2d(
                num_channels,
                old_conv.out_channels,
                kernel_size=(old_conv.kernel_size[0], old_conv.kernel_size[1]),
                stride=(old_conv.stride[0], old_conv.stride[1]),
                padding=(int(old_conv.padding[0]), int(old_conv.padding[1])),
                bias=False,
            )
            if pretrained:
                with torch.no_grad():
                    w = old_conv.weight  # (out, 3, k, k)
                    if num_channels == 1:
                        new_conv.weight.copy_(w.mean(dim=1, keepdim=True))
                    else:
                        new_conv.weight.zero_()
                        c = min(num_channels, w.size(1))
                        new_conv.weight[:, :c].copy_(w[:, :c])
                        if num_channels > w.size(1):
                            extra = new_conv.weight[:, w.size(1):]
                            extra.copy_(w.mean(dim=1, keepdim=True).expand_as(extra))
            res.conv1 = new_conv
        
        self.firstconv = res.conv1
        self.firstbn = res.bn1
        self.firstrelu = res.relu
        self.firstmaxpool = res.maxpool
        self.encoder1 = res.layer1
        self.encoder2 = res.layer2
        self.encoder3 = res.layer3
        self.encoder4 = res.layer4

        # Decoder
        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        # Final Classifier
        self.finaldeconv1 = nn.ConvTranspose2d(
            filters[0],
            32,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )
        self.finalrelu1 = nonlinearity(inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.finalrelu2 = nonlinearity(inplace=True)
        self.finalconv3 = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        in_h, in_w = x.shape[-2:]
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder with Skip Connections
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        # Final Classification
        x = self.finaldeconv1(d1)
        x = self.finalrelu1(x)
        x = self.finalconv2(x)
        x = self.finalrelu2(x)
        x = self.finalconv3(x)
        if x.shape[-2:] != (in_h, in_w):
            x = F.interpolate(x, size=(in_h, in_w), mode='bilinear', align_corners=False)
        return x
