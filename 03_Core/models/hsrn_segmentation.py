"""
Hybrid SSRN + ViT Segmentation Model (HybridSSRNSeg)
=====================================================
Adapted from the patch-classification Hybrid SSRN ViT notebook in the HSRN folder.
This version performs dense semantic segmentation (pixel-wise prediction) suitable
for the PCB Vision dataset.

Architecture (Phase 3 – TransUNet + Swin-Style Window Attention):
  1. Spectral Bottleneck : 1×1 Conv  → compress N bands to 64 "super-bands"
  2. CNN Encoder         : 4-stage downsampling to H/16, saving skip maps at each stage
  3. Window Attention    : custom shifted-window self-attention on the H/16 token grid
                           (same principle as Swin but with fully controlled dims)
  4. TransUNet Decoder   : progressive ConvTranspose2d + skip-connection upsampling
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Basic Transformer helpers
# ─────────────────────────────────────────────────────────────────────────────

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim), nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Custom Window Attention  (Swin-style shifted-window, no torchvision dep)
# ─────────────────────────────────────────────────────────────────────────────

class WindowAttention(nn.Module):
    """
    Local window self-attention.  Every 2nd layer the windows are shifted by
    (win//2, win//2) to enable cross-window communication — same idea as Swin.

    Input / output: (B, C, H, W)   — pure NCHW throughout.
    """
    def __init__(self, dim, window_size=4, heads=8, dropout=0.):
        super().__init__()
        self.win  = window_size
        self.heads = heads
        dim_head  = max(dim // heads, 1)
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5

        self.norm  = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
        self.ff_norm = nn.LayerNorm(dim)
        self.ff      = FeedForward(dim, dim * 4, dropout=dropout)

    def _window_attn(self, x_bhwc):
        B, H, W, C = x_bhwc.shape
        win = self.win
        # pad so H,W divisible by win
        ph = (win - H % win) % win
        pw = (win - W % win) % win
        x  = F.pad(x_bhwc, (0, 0, 0, pw, 0, ph))   # pad W then H
        Hp, Wp = x.shape[1], x.shape[2]

        # partition → (nW*B, win, win, C)
        x = x.view(B, Hp//win, win, Wp//win, win, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(-1, win*win, C)

        h = self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = dots.softmax(dim=-1)
        out  = torch.einsum('bhij,bhjd->bhid', attn, v)
        out  = rearrange(out, 'b h n d -> b n (h d)')
        out  = self.to_out(out)

        # reverse partition
        out = out.view(B, Hp//win, Wp//win, win, win, C)
        out = out.permute(0, 1, 3, 2, 4, 5).contiguous()
        out = out.view(B, Hp, Wp, C)
        # remove padding
        if ph or pw:
            out = out[:, :H, :W, :].contiguous()
        return out

    def forward(self, feat, shift=False):
        """feat: (B, C, H, W), shift: bool"""
        B, C, H, W = feat.shape
        x = feat.permute(0, 2, 3, 1)           # NHWC for attention

        # optional cyclic shift
        if shift:
            s = self.win // 2
            x = torch.roll(x, shifts=(-s, -s), dims=(1, 2))

        # attention
        x = x + self._window_attn(self.norm(x))
        x = x + self.ff(self.ff_norm(x))

        # reverse shift
        if shift:
            x = torch.roll(x, shifts=(s, s), dims=(1, 2))

        return x.permute(0, 3, 1, 2)           # back to NCHW


class WindowTransformer(nn.Module):
    """Stack of alternating regular / shifted window attention layers."""
    def __init__(self, dim, depth=4, window_size=4, heads=8, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList(
            [WindowAttention(dim, window_size, heads, dropout) for _ in range(depth)]
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x, shift=(i % 2 == 1))   # alternate regular / shifted
        return x


# ─────────────────────────────────────────────────────────────────────────────
# 3.  CNN Encoder  (4× max-pool  →  H/16 feature map + 4 skip maps)
# ─────────────────────────────────────────────────────────────────────────────

class CNNEncoder(nn.Module):
    """
    Progressive CNN downsampler.
    Input  : (B, 64, H,    W   )   ← after spectral bottleneck
    Output : (B, dim, H/16, W/16) + skip list [f1,f2,f3,f4]
    """
    def __init__(self, in_channels=64, dim=512):
        super().__init__()
        base = max(64, dim // 8)   # e.g. dim=512 → base=64

        def _block(ic, oc, double=True):
            layers = [nn.Conv2d(ic, oc, 3, padding=1), nn.BatchNorm2d(oc), nn.ReLU(True)]
            if double:
                layers += [nn.Conv2d(oc, oc, 3, padding=1), nn.BatchNorm2d(oc), nn.ReLU(True)]
            return nn.Sequential(*layers)

        self.enc1 = _block(in_channels, base,   double=True)
        self.enc2 = _block(base,        base*2, double=False)
        self.enc3 = _block(base*2,      base*4, double=False)
        self.enc4 = _block(base*4,      dim,    double=False)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        f1 = self.enc1(x);  x = self.pool(f1)   # H   → H/2
        f2 = self.enc2(x);  x = self.pool(f2)   # H/2 → H/4
        f3 = self.enc3(x);  x = self.pool(f3)   # H/4 → H/8
        f4 = self.enc4(x);  x = self.pool(f4)   # H/8 → H/16
        return x, [f1, f2, f3, f4]


# ─────────────────────────────────────────────────────────────────────────────
# 4.  TransUNet Decoder
# ─────────────────────────────────────────────────────────────────────────────

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(True),
            nn.Conv2d(out_ch, out_ch,            3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class TransUNetDecoder(nn.Module):
    """
    4-stage progressive upsampling with skip connections.
    dim  → half at each step, guided by encoder skip maps.
    """
    def __init__(self, dim, num_classes):
        super().__init__()
        base = max(64, dim // 8)
        # skip channels match CNNEncoder output: [base, base*2, base*4, dim]
        self.up1 = DecoderBlock(dim,    dim,    base*4)   # H/16 → H/8
        self.up2 = DecoderBlock(base*4, base*4, base*2)   # H/8  → H/4
        self.up3 = DecoderBlock(base*2, base*2, base)     # H/4  → H/2
        self.up4 = DecoderBlock(base,   base,   base)     # H/2  → H
        self.head = nn.Conv2d(base, num_classes, 1)

    def forward(self, x, skips):
        f1, f2, f3, f4 = skips
        x = self.up1(x, f4)
        x = self.up2(x, f3)
        x = self.up3(x, f2)
        x = self.up4(x, f1)
        return self.head(x)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Main Model
# ─────────────────────────────────────────────────────────────────────────────

class HybridSSRNSeg(nn.Module):
    """
    Hybrid SSRN + Swin-style Window-Attention Segmentation model for PCB Vision.

    Args:
        in_channels  : Input channels  (3=RGB, N=HSI bands / PCA components).
        num_classes  : Output classes  (4 for PCBVision).
        image_size   : H=W of input   (256).
        patch_size   : Kept for API compat; window_size=4 governs attention windows.
        dim          : CNN/Attention feature dimension. Default 512.
        depth        : Number of Window-Attention layers. Default 4.
        heads        : Attention heads. Default 8.
        mlp_dim      : Ignored (kept for API compat).
        dim_head     : Ignored (kept for API compat).
        dropout      : Dropout in attention & FF layers.
        embed_ch     : Ignored (kept for API compat).
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 4,
        image_size:  int = 256,
        patch_size:  int = 16,   # API compat only
        dim:         int = 512,
        depth:       int = 4,
        heads:       int = 8,
        mlp_dim:     int = 1024, # API compat only
        dim_head:    int = 64,   # API compat only
        dropout:     float = 0.1,
        embed_ch:    int = 64,   # API compat only
    ):
        super().__init__()
        self.image_size = image_size

        # ── A. Spectral Bottleneck ─────────────────────────────────────────
        # Learnable 1×1 PCA: compress any N input bands → 64 "super-bands"
        self.spectral_bottleneck = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )

        # ── B. CNN Encoder ─────────────────────────────────────────────────
        # (B,64,H,W) → (B,dim,H/16,W/16) + four skip maps
        self.cnn_encoder = CNNEncoder(in_channels=64, dim=dim)

        # ── C. Window Attention Transformer (Swin-style) ───────────────────
        # Operates on the (B,dim,H/16,W/16) feature map;
        # window_size=4 on a 16×16 grid = 4×4 local windows.
        self.window_transformer = WindowTransformer(
            dim=dim, depth=depth, window_size=4, heads=heads, dropout=dropout
        )

        # ── D. TransUNet Decoder ───────────────────────────────────────────
        self.decoder = TransUNetDecoder(dim=dim, num_classes=num_classes)

    def forward(self, x):
        # A: spectral compression
        x = self.spectral_bottleneck(x)       # (B, 64, H, W)

        # B: CNN encoding + collect skip maps
        feat, skips = self.cnn_encoder(x)     # (B, dim, H/16, W/16)

        # C: shifted-window attention for global context
        feat = self.window_transformer(feat)  # (B, dim, H/16, W/16)

        # D: progressive upsampling with skip connections
        logits = self.decoder(feat, skips)    # (B, num_classes, H, W)
        return logits


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_rgb = HybridSSRNSeg(in_channels=3,  num_classes=4, image_size=256).to(device)
    model_hsi = HybridSSRNSeg(in_channels=10, num_classes=4, image_size=256).to(device)

    x_rgb = torch.randn(2,  3, 256, 256).to(device)
    x_hsi = torch.randn(2, 10, 256, 256).to(device)

    out_rgb = model_rgb(x_rgb)
    out_hsi = model_hsi(x_hsi)

    print(f"RGB  → input {tuple(x_rgb.shape)}  output {tuple(out_rgb.shape)}")
    print(f"HSI  → input {tuple(x_hsi.shape)}  output {tuple(out_hsi.shape)}")
    print(f"RGB params: {sum(p.numel() for p in model_rgb.parameters()):,}")
    print(f"HSI params: {sum(p.numel() for p in model_hsi.parameters()):,}")
