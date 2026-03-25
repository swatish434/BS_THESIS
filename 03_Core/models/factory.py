from __future__ import annotations

from typing import Optional

import torch.nn as nn

from .DeepLabv3_plus import DeepLabv3_plus
from .LinkNet import LinkNet
from .ResUnet import ResUnet
from .Unet import UNET
from .Unet_Attention import AttU_Net


def get_model(name: str, in_channels: int, out_channels: int, *, pretrained: bool = False) -> nn.Module:
    """Create a segmentation model by name.

    Naming is forgiving to keep CLI simple.
    """
    n = (name or "unet").lower().replace("-", "_").replace(" ", "")
    if n in {"unet", "u_net"}:
        return UNET(in_channels=in_channels, out_channels=out_channels)
    if n in {"attunet", "attention_unet", "unet_attention"}:
        return AttU_Net(img_ch=in_channels, output_ch=out_channels)
    if n in {"resunet", "res_unet"}:
        return ResUnet(channel=in_channels, out_channel=out_channels)
    if n in {"deeplabv3+", "deeplabv3plus", "deeplabv3_plus", "deeplabv3"}:
        return DeepLabv3_plus(nInputChannels=in_channels, n_classes=out_channels, pretrained=pretrained, _print=False)
    if n in {"linknet"}:
        return LinkNet(num_classes=out_channels, num_channels=in_channels, pretrained=pretrained)

    raise ValueError(f"Unknown model name: {name}")
