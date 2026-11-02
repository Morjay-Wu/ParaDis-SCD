"""Changer: Feature Interaction is What You Need for Change Detection (ICCV 2023).

Official Source: https://github.com/likyoo/open-cd (opencd/models/decode_heads/changer.py)
Paper: S. Fang, K. Li, and Z. Li,
       "Changer: Feature Interaction is What You Need for Change Detection,"
       IEEE/CVF International Conference on Computer Vision (ICCV), 2023.

Architecture:
1. Interactive Siamese ResNet-18 Backbone with bi-temporal ChannelExchange
   and SpatialExchange layers across stages.
2. Multi-scale feature pyramid projection and aggregation.
3. FDAF (Flow Dual-Alignment Fusion): estimates bi-directional dense offset flows
   (f1, f2) via depthwise separable 5x5 convolution and aligns bi-temporal features
   via F.grid_sample before differencing (warp(x1, f1) - x2, warp(x2, f2) - x1).
4. MixFFN depthwise-enhanced feed-forward discriminator head.
"""

from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class ChannelExchange(nn.Module):
    """Parameter-free bi-temporal Channel Exchange (ICCV 2023 Changer)."""

    def __init__(self, p: int = 2):
        super().__init__()
        self.p = p

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _, c, _, _ = x1.shape
        mask = (torch.arange(c, device=x1.device) % self.p == 0).view(1, c, 1, 1)
        out_x1 = torch.where(mask, x2, x1)
        out_x2 = torch.where(mask, x1, x2)
        return out_x1, out_x2


class SpatialExchange(nn.Module):
    """Parameter-free bi-temporal Spatial Exchange (ICCV 2023 Changer)."""

    def __init__(self, p: int = 2):
        super().__init__()
        self.p = p

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _, _, _, w = x1.shape
        mask = (torch.arange(w, device=x1.device) % self.p == 0).view(1, 1, 1, w)
        out_x1 = torch.where(mask, x2, x1)
        out_x2 = torch.where(mask, x1, x2)
        return out_x1, out_x2


class FDAF(nn.Module):
    """Flow Dual-Alignment Fusion Module (FDAF) from Changer (ICCV 2023)."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels
        kernel_size = 5
        self.flow_make = nn.Sequential(
            nn.Conv2d(
                in_channels * 2,
                in_channels * 2,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2,
                bias=True,
                groups=in_channels * 2,
            ),
            nn.InstanceNorm2d(in_channels * 2),
            nn.GELU(),
            nn.Conv2d(in_channels * 2, 4, kernel_size=1, padding=0, bias=False),
        )

    @staticmethod
    def warp(x: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        n, _, h, w = x.size()
        norm = torch.tensor([[[[max(1, w), max(1, h)]]]], dtype=x.dtype, device=x.device)
        col = torch.linspace(-1.0, 1.0, h, dtype=x.dtype, device=x.device).view(-1, 1).repeat(1, w)
        row = torch.linspace(-1.0, 1.0, w, dtype=x.dtype, device=x.device).repeat(h, 1)
        grid = torch.cat((row.unsqueeze(2), col.unsqueeze(2)), dim=2)
        grid = grid.unsqueeze(0).repeat(n, 1, 1, 1) + flow.permute(0, 2, 3, 1) / norm
        return F.grid_sample(x.float(), grid.float(), mode="bilinear", padding_mode="border", align_corners=True).to(x.dtype)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        output = torch.cat([x1, x2], dim=1)
        flow = self.flow_make(output)
        f1, f2 = torch.chunk(flow, 2, dim=1)
        x1_feat = self.warp(x1, f1) - x2
        x2_feat = self.warp(x2, f2) - x1
        return torch.cat([x1_feat, x2_feat], dim=1)


class MixFFN(nn.Module):
    """MixFFN discriminator head from Changer (ICCV 2023)."""

    def __init__(self, embed_dims: int, feedforward_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(embed_dims, feedforward_channels, kernel_size=1, bias=True),
            nn.Conv2d(
                feedforward_channels,
                feedforward_channels,
                kernel_size=3,
                padding=1,
                groups=feedforward_channels,
                bias=True,
            ),
            nn.GELU(),
            nn.Conv2d(feedforward_channels, embed_dims, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.layers(x)


class Changer(nn.Module):
    """Changer (ICCV 2023): Interactive ResNet-18 + FDAF Flow Dual-Alignment Network."""

    def __init__(self, pretrained: bool = False, embed_dims: int = 128):
        super().__init__()
        backbone = resnet18(weights=None if not pretrained else "DEFAULT")
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1  # 64
        self.layer2 = backbone.layer2  # 128
        self.layer3 = backbone.layer3  # 256
        self.layer4 = backbone.layer4  # 512

        # Bi-temporal interaction modules across stages (Official Changer config)
        self.interact1 = ChannelExchange(p=2)
        self.interact2 = SpatialExchange(p=2)
        self.interact3 = ChannelExchange(p=2)

        in_channels = [64, 128, 256, 512]
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, embed_dims, kernel_size=1, bias=False),
                nn.BatchNorm2d(embed_dims),
                nn.ReLU(inplace=True),
            )
            for ch in in_channels
        ])
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(embed_dims * 4, embed_dims, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dims),
            nn.ReLU(inplace=True),
        )

        self.neck_layer = FDAF(in_channels=embed_dims)
        self.discriminator = MixFFN(
            embed_dims=embed_dims * 2,
            feedforward_channels=embed_dims * 2,
        )
        self.cls_seg = nn.Conv2d(embed_dims * 2, 1, kernel_size=1)

    def _base_forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        target_hw = inputs[0].shape[2:]
        outs = []
        for idx, x in enumerate(inputs):
            proj = self.convs[idx](x)
            if proj.shape[2:] != target_hw:
                proj = F.interpolate(proj, size=target_hw, mode="bilinear", align_corners=False)
            outs.append(proj)
        return self.fusion_conv(torch.cat(outs, dim=1))

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> Dict[str, torch.Tensor]:
        target_size = (img1.shape[-2], img1.shape[-1])

        x1 = self.stem(img1)
        x2 = self.stem(img2)

        c1_1 = self.layer1(x1)
        c1_2 = self.layer1(x2)
        c1_1, c1_2 = self.interact1(c1_1, c1_2)

        c2_1 = self.layer2(c1_1)
        c2_2 = self.layer2(c1_2)
        c2_1, c2_2 = self.interact2(c2_1, c2_2)

        c3_1 = self.layer3(c2_1)
        c3_2 = self.layer3(c2_2)
        c3_1, c3_2 = self.interact3(c3_1, c3_2)

        c4_1 = self.layer4(c3_1)
        c4_2 = self.layer4(c3_2)

        out1 = self._base_forward([c1_1, c2_1, c3_1, c4_1])
        out2 = self._base_forward([c1_2, c2_2, c3_2, c4_2])

        aligned = self.neck_layer(out1, out2)
        feat = self.discriminator(aligned)
        logits = self.cls_seg(feat)

        if logits.shape[-2:] != target_size:
            logits = F.interpolate(logits, size=target_size, mode="bilinear", align_corners=False)

        prob = torch.sigmoid(logits)
        return {
            "logits": logits,
            "prob": prob,
        }
