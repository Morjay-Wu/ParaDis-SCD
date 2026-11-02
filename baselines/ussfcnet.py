"""USSFC-Net: Ultralightweight Spatial-Spectral Feature Cooperation Network (IEEE TGRS 2023).

Official Source: https://github.com/SUST-reynole/USSFC-Net
Paper: T. Lei, X. Geng, H. Ning, Z. Lv, M. Gong, Y. Jin, and A. K. Nandi,
       "Ultralightweight Spatial-Spectral Feature Cooperation Network for Change Detection
       in Remote Sensing Images," IEEE Transactions on Geoscience and Remote Sensing,
       vol. 61, pp. 1-14, 2023.

Key Modules:
1. CMConv: Cascade Multi-scale Dilated Convolution with channel-shifted group receptive fields.
2. SSFC: Parameter-free 3D Spatial-Spectral Feature Cooperation statistical attention.
3. MSDConv_SSFC: Multi-scale Decoupled Convolution coupled with SSFC.
4. USSFCNet: 5-stage ultralightweight Siamese encoder-decoder with arbitrary rectangular resolution support.
"""

from typing import Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SSFC(nn.Module):
    """Spatial-Spectral Feature Cooperation (SSFC) attention module."""

    def __init__(self, in_ch: int):
        super().__init__()
        self.in_ch = in_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.size()
        q = x.mean(dim=[2, 3], keepdim=True)
        k = x
        square = (k - q).pow(2)
        sigma = square.sum(dim=[2, 3], keepdim=True) / max(1, h * w)
        att_score = square / (2.0 * sigma + 1e-6) + 0.5
        att_weight = torch.sigmoid(att_score)
        return x * att_weight


class CMConv(nn.Module):
    """Cascade Multi-scale Convolution (CMConv) from USSFC-Net."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 3,
        groups: int = 1,
        dilation_set: int = 4,
        bias: bool = False,
    ):
        super().__init__()
        groups = max(1, groups)
        self.prim = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size,
            stride,
            padding=dilation,
            dilation=dilation,
            groups=groups * dilation_set,
            bias=bias,
        )
        self.prim_shift = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size,
            stride,
            padding=2 * dilation,
            dilation=2 * dilation,
            groups=groups * dilation_set,
            bias=bias,
        )
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, groups=groups, bias=bias)

        mask = torch.zeros_like(self.conv.weight, dtype=torch.bool)
        _in_channels = in_ch // (groups * dilation_set)
        _out_channels = out_ch // (groups * dilation_set)
        for i in range(dilation_set):
            for j in range(groups):
                r1_start = (i + j * groups) * _out_channels
                r1_end = (i + j * groups + 1) * _out_channels
                c_start = i * _in_channels
                c_end = (i + 1) * _in_channels
                if r1_end <= mask.shape[0] and c_end <= mask.shape[1]:
                    mask[r1_start:r1_end, c_start:c_end, :, :] = True

                r2_idx = (i + dilation_set // 2) % dilation_set + j * groups
                r2_start = r2_idx * _out_channels
                r2_end = (r2_idx + 1) * _out_channels
                if r2_end <= mask.shape[0] and c_end <= mask.shape[1]:
                    mask[r2_start:r2_end, c_start:c_end, :, :] = True

        with torch.no_grad():
            self.conv.weight.data[mask] = 0.0
        self.register_buffer("mask", mask, persistent=False)

        def backward_hook(grad: torch.Tensor) -> torch.Tensor:
            out = grad.clone()
            out[self.mask] = 0.0
            return out

        self.conv.weight.register_hook(backward_hook)
        self.groups = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_split = (z.chunk(2, dim=1) for z in x.chunk(self.groups, dim=1))
        x_merge = torch.cat(tuple(torch.cat((x2, x1), dim=1) for (x1, x2) in x_split), dim=1)
        x_shift = self.prim_shift(x_merge)
        return self.prim(x) + self.conv(x) + x_shift


class MSDConv_SSFC(nn.Module):
    """Multi-scale Decoupled Convolution with SSFC (MSDConv_SSFC)."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        ratio: int = 2,
        aux_k: int = 3,
        dilation: int = 3,
    ):
        super().__init__()
        self.out_ch = out_ch
        native_ch = math.ceil(out_ch / ratio)
        aux_ch = native_ch * (ratio - 1)

        self.native = nn.Sequential(
            nn.Conv2d(in_ch, native_ch, kernel_size, stride, padding=padding, dilation=1, bias=False),
            nn.BatchNorm2d(native_ch),
            nn.ReLU(inplace=True),
        )

        self.aux = nn.Sequential(
            CMConv(
                native_ch,
                aux_ch,
                aux_k,
                1,
                padding=1,
                groups=max(1, int(native_ch / 4)),
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(aux_ch),
            nn.ReLU(inplace=True),
        )

        self.att = SSFC(aux_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.native(x)
        x2 = self.att(self.aux(x1))
        out = torch.cat([x1, x2], dim=1)
        return out[:, : self.out_ch, :, :]


class First_DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.Conv = nn.Sequential(
            MSDConv_SSFC(in_ch, out_ch, dilation=3),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            MSDConv_SSFC(out_ch, out_ch, dilation=3),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.Conv(x)


class USSFCNet(nn.Module):
    """USSFC-Net (IEEE TGRS 2023) adapted for unified SCD dictionary contract."""

    def __init__(self, in_ch: int = 3, out_ch: int = 1, ratio: float = 0.5):
        super().__init__()
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1_1 = First_DoubleConv(in_ch, int(64 * ratio))
        self.Conv1_2 = First_DoubleConv(in_ch, int(64 * ratio))
        self.Conv2_1 = DoubleConv(int(64 * ratio), int(128 * ratio))
        self.Conv2_2 = DoubleConv(int(64 * ratio), int(128 * ratio))
        self.Conv3_1 = DoubleConv(int(128 * ratio), int(256 * ratio))
        self.Conv3_2 = DoubleConv(int(128 * ratio), int(256 * ratio))
        self.Conv4_1 = DoubleConv(int(256 * ratio), int(512 * ratio))
        self.Conv4_2 = DoubleConv(int(256 * ratio), int(512 * ratio))
        self.Conv5_1 = DoubleConv(int(512 * ratio), int(1024 * ratio))
        self.Conv5_2 = DoubleConv(int(512 * ratio), int(1024 * ratio))

        self.Up5 = nn.ConvTranspose2d(int(1024 * ratio), int(512 * ratio), 2, stride=2)
        self.Up_conv5 = DoubleConv(int(1024 * ratio), int(512 * ratio))

        self.Up4 = nn.ConvTranspose2d(int(512 * ratio), int(256 * ratio), 2, stride=2)
        self.Up_conv4 = DoubleConv(int(512 * ratio), int(256 * ratio))

        self.Up3 = nn.ConvTranspose2d(int(256 * ratio), int(128 * ratio), 2, stride=2)
        self.Up_conv3 = DoubleConv(int(256 * ratio), int(128 * ratio))

        self.Up2 = nn.ConvTranspose2d(int(128 * ratio), int(64 * ratio), 2, stride=2)
        self.Up_conv2 = DoubleConv(int(128 * ratio), int(64 * ratio))

        self.Conv_1x1 = nn.Conv2d(int(64 * ratio), out_ch, kernel_size=1, stride=1, padding=0)

    @staticmethod
    def _align_cat(skip: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        if up.shape[-2:] != skip.shape[-2:]:
            up = F.interpolate(up, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat((skip, up), dim=1)

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> Dict[str, torch.Tensor]:
        target_size = (img1.shape[-2], img1.shape[-1])

        c1_1 = self.Conv1_1(img1)
        c1_2 = self.Conv1_2(img2)
        x1 = torch.abs(c1_1 - c1_2)

        c2_1 = self.Conv2_1(self.Maxpool(c1_1))
        c2_2 = self.Conv2_2(self.Maxpool(c1_2))
        x2 = torch.abs(c2_1 - c2_2)

        c3_1 = self.Conv3_1(self.Maxpool(c2_1))
        c3_2 = self.Conv3_2(self.Maxpool(c2_2))
        x3 = torch.abs(c3_1 - c3_2)

        c4_1 = self.Conv4_1(self.Maxpool(c3_1))
        c4_2 = self.Conv4_2(self.Maxpool(c3_2))
        x4 = torch.abs(c4_1 - c4_2)

        c5_1 = self.Conv5_1(self.Maxpool(c4_1))
        c5_2 = self.Conv5_2(self.Maxpool(c4_2))
        x5 = torch.abs(c5_1 - c5_2)

        d5 = self.Up_conv5(self._align_cat(x4, self.Up5(x5)))
        d4 = self.Up_conv4(self._align_cat(x3, self.Up4(d5)))
        d3 = self.Up_conv3(self._align_cat(x2, self.Up3(d4)))
        d2 = self.Up_conv2(self._align_cat(x1, self.Up2(d3)))

        logits = self.Conv_1x1(d2)
        if logits.shape[-2:] != target_size:
            logits = F.interpolate(logits, size=target_size, mode="bilinear", align_corners=False)

        prob = torch.sigmoid(logits)
        return {
            "logits": logits,
            "prob": prob,
        }
