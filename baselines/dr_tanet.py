"""DR-TANet: Dual-Resolution Spatio-Temporal Attention Network for Scene Change Detection.

Paper: "Spatial-Temporal Attention-Based Method and a New Dataset for Remote Sensing Image Change Detection"
IEEE Transactions on Image Processing (IEEE TIP 2020) / CVPRW.

Key Architecture Components:
1. Dual-Resolution Siamese CNN feature pyramid encoder (capturing high-res spatial details
   and low-res deep semantics).
2. Spatial-Temporal Attention Module (STAM):
   Jointly models spatial self-attention and bi-temporal cross-attention across stacked
   temporal tokens (2 * H * W), enabling long-range contextual reasoning between t1 and t2.
3. Temporal Channel Attention Module (TCAM) for inter-temporal channel recalibration.
4. Dual-resolution feature fusion and progressive upsampling decoder.
5. Strict support for arbitrary rectangular resolutions (480x640, 224x1024).
6. Returns unified baseline contract: {'logits': [B, 1, H, W], 'prob': [B, 1, H, W]}.
"""

from typing import Dict, List, Optional, Tuple, Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class ResBlock(nn.Module):
    """Standard Residual Block with 2 3x3 Convolutions."""

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                conv1x1(in_planes, planes, stride=stride),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + res
        return self.relu(out)


class DualResolutionEncoder(nn.Module):
    """Dual-Resolution Siamese Encoder extracting multi-scale feature pyramids."""

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
    ):
        super().__init__()
        # Initial stem downsampling to stride 4
        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Stage 1: High-resolution detail stream [B, base, H/4, W/4]
        self.stage1 = nn.Sequential(
            ResBlock(base_channels, base_channels, stride=1),
            ResBlock(base_channels, base_channels, stride=1),
        )

        # Stage 2: Intermediate stream [B, 2*base, H/8, W/8]
        self.stage2 = nn.Sequential(
            ResBlock(base_channels, base_channels * 2, stride=2),
            ResBlock(base_channels * 2, base_channels * 2, stride=1),
        )

        # Stage 3: Low-resolution deep semantic stream [B, 4*base, H/16, W/16]
        self.stage3 = nn.Sequential(
            ResBlock(base_channels * 2, base_channels * 4, stride=2),
            ResBlock(base_channels * 4, base_channels * 4, stride=1),
        )

    def forward_single(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        f_high = self.stage1(x)       # [B, base, H/4, W/4]
        f_mid = self.stage2(f_high)   # [B, 2*base, H/8, W/8]
        f_low = self.stage3(f_mid)    # [B, 4*base, H/16, W/16]
        return f_high, f_mid, f_low

    def forward(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        return self.forward_single(img1), self.forward_single(img2)


class SpatialTemporalAttentionModule(nn.Module):
    """Spatial-Temporal Attention Module (STAM).
    
    Stacks bi-temporal features across time (T=2), generating a unified spatial-temporal
    token representation (2 * H * W). Computes full inter-temporal and intra-temporal
    attention, allowing any location in image 1 to attend directly to all locations
    in image 2 and vice-versa.
    """

    def __init__(self, in_channels: int, inter_channels: Optional[int] = None):
        super().__init__()
        self.in_channels = in_channels
        self.inter_channels = inter_channels or in_channels // 2

        self.query_conv = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for STAM on bi-temporal feature pair.
        
        Args:
            x1: Feature map at t1 [B, C, H, W]
            x2: Feature map at t2 [B, C, H, W]
            
        Returns:
            Attended features y1, y2 [B, C, H, W]
        """
        B, C, H, W = x1.shape
        HW = H * W

        # 1. Project Query, Key, Value for both time steps
        q1 = self.query_conv(x1).view(B, self.inter_channels, HW)  # [B, C_mid, HW]
        q2 = self.query_conv(x2).view(B, self.inter_channels, HW)
        k1 = self.key_conv(x1).view(B, self.inter_channels, HW)
        k2 = self.key_conv(x2).view(B, self.inter_channels, HW)
        v1 = self.value_conv(x1).view(B, C, HW)                    # [B, C, HW]
        v2 = self.value_conv(x2).view(B, C, HW)

        # 2. Stack tokens across temporal dimension (T=2) -> token length = 2 * HW
        # Q: [B, 2*HW, C_mid], K: [B, C_mid, 2*HW], V: [B, 2*HW, C]
        Q = torch.cat([q1, q2], dim=2).permute(0, 2, 1)  # [B, 2*HW, C_mid]
        K = torch.cat([k1, k2], dim=2)                   # [B, C_mid, 2*HW]
        V = torch.cat([v1, v2], dim=2).permute(0, 2, 1)  # [B, 2*HW, C]

        # 3. Spatial-Temporal Attention Matrix [B, 2*HW, 2*HW]
        scale = 1.0 / math.sqrt(self.inter_channels)
        energy = torch.bmm(Q, K) * scale
        attn = F.softmax(energy, dim=-1)

        # 4. Multiply with Value tokens -> [B, 2*HW, C] -> [B, C, 2*HW]
        out = torch.bmm(attn, V).permute(0, 2, 1)

        # 5. Split back into time step 1 and time step 2
        out1 = out[:, :, :HW].view(B, C, H, W)
        out2 = out[:, :, HW:].view(B, C, H, W)

        # 6. Residual connection with learnable scalar gamma
        y1 = self.gamma * out1 + x1
        y2 = self.gamma * out2 + x2
        return y1, y2


class TemporalChannelAttentionModule(nn.Module):
    """Temporal Channel Attention Module (TCAM) for bi-temporal channel weighting."""

    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels * 2, max(8, in_channels // reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(8, in_channels // reduction), in_channels * 2, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, H, W = x1.shape
        # Global average pooling
        gap1 = x1.mean(dim=[-2, -1])  # [B, C]
        gap2 = x2.mean(dim=[-2, -1])  # [B, C]
        gap_cat = torch.cat([gap1, gap2], dim=1)  # [B, 2C]

        weights = self.fc(gap_cat).view(B, 2 * C, 1, 1)
        w1 = weights[:, :C, :, :]
        w2 = weights[:, C:, :, :]

        return x1 * (1.0 + w1), x2 * (1.0 + w2)


class DRTANetDecoder(nn.Module):
    """Dual-Resolution Decoder fusing STAM-attended semantics and high-res details."""

    def __init__(
        self,
        c_high: int = 64,
        c_mid: int = 128,
        c_low: int = 256,
        dec_dim: int = 128,
    ):
        super().__init__()
        # STAM difference fusion at low resolution (stride 16)
        self.diff_low = nn.Sequential(
            nn.Conv2d(c_low * 3, dec_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dec_dim),
            nn.ReLU(inplace=True),
        )

        # Mid-scale difference fusion (stride 8)
        self.diff_mid = nn.Sequential(
            nn.Conv2d(c_mid * 3, dec_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dec_dim),
            nn.ReLU(inplace=True),
        )

        # High-resolution detail difference (stride 4)
        self.diff_high = nn.Sequential(
            nn.Conv2d(c_high * 3, dec_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dec_dim),
            nn.ReLU(inplace=True),
        )

        # Progressive fusion layers
        self.fuse_low_mid = nn.Sequential(
            nn.Conv2d(dec_dim * 2, dec_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dec_dim),
            nn.ReLU(inplace=True),
        )
        self.fuse_mid_high = nn.Sequential(
            nn.Conv2d(dec_dim * 2, dec_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dec_dim),
            nn.ReLU(inplace=True),
        )

        # Final prediction head
        self.head = nn.Sequential(
            nn.ConvTranspose2d(dec_dim, dec_dim // 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dec_dim // 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(dec_dim // 2, dec_dim // 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dec_dim // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(dec_dim // 4, 1, kernel_size=3, padding=1),
        )

    def forward(
        self,
        f_high_1: torch.Tensor,
        f_high_2: torch.Tensor,
        f_mid_1: torch.Tensor,
        f_mid_2: torch.Tensor,
        y_low_1: torch.Tensor,
        y_low_2: torch.Tensor,
        target_size: Tuple[int, int],
    ) -> torch.Tensor:
        # Low-res difference [B, dec, H/16, W/16]
        d_low = self.diff_low(torch.cat([y_low_1, y_low_2, torch.abs(y_low_1 - y_low_2)], dim=1))

        # Mid-res difference [B, dec, H/8, W/8]
        d_mid = self.diff_mid(torch.cat([f_mid_1, f_mid_2, torch.abs(f_mid_1 - f_mid_2)], dim=1))

        # Upsample low to mid and fuse
        d_low_up = F.interpolate(d_low, size=d_mid.shape[-2:], mode='bilinear', align_corners=False)
        fused_mid = self.fuse_low_mid(torch.cat([d_low_up, d_mid], dim=1))

        # High-res difference [B, dec, H/4, W/4]
        d_high = self.diff_high(torch.cat([f_high_1, f_high_2, torch.abs(f_high_1 - f_high_2)], dim=1))

        # Upsample mid to high and fuse
        fused_mid_up = F.interpolate(fused_mid, size=d_high.shape[-2:], mode='bilinear', align_corners=False)
        fused_high = self.fuse_mid_high(torch.cat([fused_mid_up, d_high], dim=1))

        # Predict logits and ensure target resolution
        logits = self.head(fused_high)
        if logits.shape[-2:] != target_size:
            logits = F.interpolate(logits, size=target_size, mode='bilinear', align_corners=False)

        return logits


class DRTANet(nn.Module):
    """Dual-Resolution Spatio-Temporal Attention Network (DR-TANet).
    
    Args:
        in_channels: Input image channel count (default 3).
        base_channels: Base channel count for ResNet backbone (default 64).
        dec_dim: Channel dimension for decoder feature fusion (default 128).
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        dec_dim: int = 128,
    ):
        super().__init__()
        self.encoder = DualResolutionEncoder(
            in_channels=in_channels,
            base_channels=base_channels,
        )

        c_high = base_channels
        c_mid = base_channels * 2
        c_low = base_channels * 4

        # Spatial-Temporal Attention Module at semantic level
        self.stam = SpatialTemporalAttentionModule(in_channels=c_low, inter_channels=dec_dim // 2)
        # Temporal Channel Attention Module
        self.tcam = TemporalChannelAttentionModule(in_channels=c_low)

        # Dual-Resolution Decoder
        self.decoder = DRTANetDecoder(
            c_high=c_high,
            c_mid=c_mid,
            c_low=c_low,
            dec_dim=dec_dim,
        )

    def forward(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for DR-TANet.
        
        Args:
            img1: Image at time 1 [B, 3, H, W]
            img2: Image at time 2 [B, 3, H, W]
            
        Returns:
            Dictionary matching unified baseline contract:
                'logits': Raw change logits [B, 1, H, W]
                'prob': Sigmoid change probability [B, 1, H, W] in [0, 1]
        """
        target_size = (img1.shape[-2], img1.shape[-1])

        # 1. Dual-resolution feature extraction
        (f_high_1, f_mid_1, f_low_1), (f_high_2, f_mid_2, f_low_2) = self.encoder(img1, img2)

        # 2. Spatio-temporal and temporal-channel attention
        y_low_1, y_low_2 = self.stam(f_low_1, f_low_2)
        y_low_1, y_low_2 = self.tcam(y_low_1, y_low_2)

        # 3. Dual-resolution decoding to full rectangular resolution
        logits = self.decoder(
            f_high_1=f_high_1,
            f_high_2=f_high_2,
            f_mid_1=f_mid_1,
            f_mid_2=f_mid_2,
            y_low_1=y_low_1,
            y_low_2=y_low_2,
            target_size=target_size,
        )
        prob = torch.sigmoid(logits)

        return {
            'logits': logits,
            'prob': prob,
        }
