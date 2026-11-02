"""CGNet: Change Guiding Network for Change Detection (IEEE J-STARS / TGRS 2023).

Official Source: https://github.com/ChengxiHAN/CGNet-CD
Paper: C. Han, C. Wu, H. Guo, M. Hu, J. Li and H. Chen,
       "Change Guiding Network: Incorporating Change Prior to Guide Change Detection
       in Remote Sensing Imagery," IEEE J-STARS, vol. 16, pp. 8395-8407, 2023.

Architecture:
1. Siamese VGG16-BN 5-stage encoder extracting multi-scale features.
2. Bi-temporal concatenation and 3x3 conv channel reduction across stages 1-4.
3. Initial change map prior prediction from deep semantic stage 4.
4. ChangeGuideModule (CGM) incorporating the sigmoid change prior map (1 + guiding_map)
   to modulate query/key/value spatial attention across decoder stages 4, 3, and 2.
5. Supports arbitrary rectangular resolutions with bounded attention memory.
"""

from typing import Dict, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class BasicConv2d(nn.Module):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.cuda.amp.autocast(enabled=False):
            x_fp32 = x.float()
            x_fp32 = self.conv(x_fp32)
            x_fp32 = self.bn(x_fp32)
            x_fp32 = self.relu(x_fp32)
        return x_fp32


class ChangeGuideModule(nn.Module):
    """Change Guide Module (CGM) from CGNet (ChengxiHAN/CGNet-CD).

    Incorporates the coarse change prior map to guide spatial non-local attention.
    To support high-resolution rectangular SCD images (e.g. 480x640, 224x1024)
    within strict GPU memory budgets, keys and values are adaptively pooled when
    spatial tokens exceed max_kv_tokens while preserving full-resolution queries.
    """

    def __init__(self, in_dim: int, max_kv_side: int = 32):
        super().__init__()
        self.chanel_in = in_dim
        self.max_kv_side = max_kv_side
        inter_dim = max(in_dim // 8, 8)
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=inter_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=inter_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, guiding_map0: torch.Tensor) -> torch.Tensor:
        m_batchsize, C, height, width = x.size()
        guiding_map0 = F.interpolate(guiding_map0, size=(height, width), mode="bilinear", align_corners=False)
        guiding_map = torch.sigmoid(guiding_map0)

        modulated = x * (1.0 + guiding_map)
        query = self.query_conv(modulated)
        key = self.key_conv(modulated)
        value = self.value_conv(modulated)

        if height > self.max_kv_side or width > self.max_kv_side:
            kh = min(height, self.max_kv_side)
            kw = min(width, self.max_kv_side)
            key = F.adaptive_avg_pool2d(key, (kh, kw))
            value = F.adaptive_avg_pool2d(value, (kh, kw))
        else:
            kh, kw = height, width

        with torch.cuda.amp.autocast(enabled=False):
            proj_query = query.float().reshape(m_batchsize, -1, width * height).permute(0, 2, 1)  # [B, H*W, C']
            proj_key = key.float().reshape(m_batchsize, -1, kw * kh)  # [B, C', kh*kw]

            scale = 1.0 / math.sqrt(max(1, proj_key.size(1)))
            energy = torch.bmm(proj_query, proj_key) * scale
            attention = self.softmax(energy)

            proj_value = value.float().reshape(m_batchsize, -1, kw * kh)  # [B, C, kh*kw]
            out = torch.bmm(proj_value, attention.permute(0, 2, 1))
            out = out.reshape(m_batchsize, C, height, width)
            out = self.gamma.float() * out + x.float()
        return out.to(x.dtype)


class CGNet(nn.Module):
    """Change Guiding Network (CGNet, IEEE J-STARS 2023)."""

    def __init__(self, pretrained: bool = False):
        super().__init__()
        vgg16_bn = models.vgg16_bn(weights=None if not pretrained else "DEFAULT")
        for m in vgg16_bn.features.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = False
        self.inc = vgg16_bn.features[:6]       # 64, stride 1
        self.down1 = vgg16_bn.features[6:13]   # 128, stride 2
        self.down2 = vgg16_bn.features[13:23]  # 256, stride 4
        self.down3 = vgg16_bn.features[23:33]  # 512, stride 8
        self.down4 = vgg16_bn.features[33:43]  # 512, stride 16

        self.conv_reduce_1 = BasicConv2d(128 * 2, 128, 3, 1, 1)
        self.conv_reduce_2 = BasicConv2d(256 * 2, 256, 3, 1, 1)
        self.conv_reduce_3 = BasicConv2d(512 * 2, 512, 3, 1, 1)
        self.conv_reduce_4 = BasicConv2d(512 * 2, 512, 3, 1, 1)

        self.decoder = nn.Sequential(
            BasicConv2d(512, 64, 3, 1, 1),
            nn.Conv2d(64, 1, 3, 1, 1),
        )

        self.decoder_final = nn.Sequential(
            BasicConv2d(128, 64, 3, 1, 1),
            nn.Conv2d(64, 1, 1),
        )

        self.cgm_2 = ChangeGuideModule(256)
        self.cgm_3 = ChangeGuideModule(512)
        self.cgm_4 = ChangeGuideModule(512)

        self.decoder_module4 = BasicConv2d(1024, 512, 3, 1, 1)
        self.decoder_module3 = BasicConv2d(768, 256, 3, 1, 1)
        self.decoder_module2 = BasicConv2d(384, 128, 3, 1, 1)

    @staticmethod
    def _upsample_to(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=ref.shape[2:], mode="bilinear", align_corners=False)

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> Dict[str, torch.Tensor]:
        size = (img1.shape[-2], img1.shape[-1])

        layer1_pre_A = self.inc(img1)
        layer1_A = self.down1(layer1_pre_A)
        layer2_A = self.down2(layer1_A)
        layer3_A = self.down3(layer2_A)
        layer4_A = self.down4(layer3_A)

        layer1_pre_B = self.inc(img2)
        layer1_B = self.down1(layer1_pre_B)
        layer2_B = self.down2(layer1_B)
        layer3_B = self.down3(layer2_B)
        layer4_B = self.down4(layer3_B)

        with torch.cuda.amp.autocast(enabled=False):
            layer1 = self.conv_reduce_1(torch.cat((layer1_B.float(), layer1_A.float()), dim=1))
            layer2 = self.conv_reduce_2(torch.cat((layer2_B.float(), layer2_A.float()), dim=1))
            layer3 = self.conv_reduce_3(torch.cat((layer3_B.float(), layer3_A.float()), dim=1))
            layer4 = self.conv_reduce_4(torch.cat((layer4_B.float(), layer4_A.float()), dim=1))

            change_map = self.decoder(layer4)

            layer4 = self.cgm_4(layer4, change_map)
            feature4 = self.decoder_module4(torch.cat([self._upsample_to(layer4, layer3), layer3], dim=1))

            layer3 = self.cgm_3(feature4, change_map)
            feature3 = self.decoder_module3(torch.cat([self._upsample_to(layer3, layer2), layer2], dim=1))

            layer2 = self.cgm_2(feature3, change_map)
            layer1 = self.decoder_module2(torch.cat([self._upsample_to(layer2, layer1), layer1], dim=1))

            change_map = F.interpolate(change_map, size=size, mode="bilinear", align_corners=False)
            final_map = self.decoder_final(layer1)
            final_map = F.interpolate(final_map, size=size, mode="bilinear", align_corners=False)

        prob = torch.sigmoid(final_map)
        return {
            "logits": final_map,
            "prob": prob,
            "aux_logits": change_map,
        }
