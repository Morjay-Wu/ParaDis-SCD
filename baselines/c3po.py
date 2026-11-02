"""C-3PO: Combine 3 Possible Change Types for Scene Change Detection (CVPR 2023 / PR).

Official Source: https://github.com/DoctorKey/C-3PO
Paper: "Reduce, Reuse, Recycle: A New Paradigm for Scene Change Detection" /
       "How to Reduce Change Detection to Semantic Segmentation" (Wang et al., 2023).

Architecture:
1. Shared ResNet-18 multi-scale feature extractor (5 scales: stride 4, 4, 8, 16, 32).
2. MTF (Merge Temporal Features) module explicitly modeling 3 change types:
   - Informative sum: info = Conv(f0 + f1)
   - Disappearance:   disappear = Conv(ReLU(f0 - f1))
   - Appearance:      appear = Conv(ReLU(f1 - f0))
   - Exchange:        exchange = Conv(max(f0, f1) - min(f0, f1))
   - Combined:        ReLU(info + disappear + appear + exchange)
3. MSF (Multi-Scale Feature Fusion) top-down FPN with MTF at each scale.
4. DeepLabV3 ASPP segmentation head adapted for arbitrary rectangular resolutions
   and batch size B >= 1.
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class MTF(nn.Module):
    """Merge Temporal Features (MTF) module from C-3PO (DoctorKey/C-3PO).

    Separately learns the three possible change types (Disappear, Appear, Exchange)
    alongside shared temporal context (Info).
    """

    def __init__(self, channel: int, mode: str = "iade", kernel_size: int = 3):
        super().__init__()
        assert mode in [
            "i", "a", "d", "e", "ia", "id", "ie", "iae", "ide", "iad",
            "iade", "i2ade", "iad2e", "i2ad2e", "i2d",
        ]
        self.mode = mode
        self.channel = channel
        self.relu = nn.ReLU(inplace=True)
        padding = 1 if kernel_size == 3 else 0

        if "i2" in mode:
            self.i0 = nn.Conv2d(channel, channel, kernel_size, padding=padding, stride=1, bias=False)
            self.i1 = nn.Conv2d(channel, channel, kernel_size, padding=padding, stride=1, bias=False)
        else:
            self.conv = nn.Conv2d(channel, channel, kernel_size, padding=padding, stride=1, bias=False)

        if "ad2" in mode:
            self.app = nn.Conv2d(channel, channel, kernel_size, padding=padding, stride=1, bias=False)
            self.dis = nn.Conv2d(channel, channel, kernel_size, padding=padding, stride=1, bias=False)
        else:
            self.res = nn.Conv2d(channel, channel, kernel_size, padding=padding, stride=1, bias=False)

        self.exchange = nn.Conv2d(channel, channel, kernel_size, padding=padding, stride=1, bias=False)

    def forward(self, f0: torch.Tensor, f1: torch.Tensor) -> torch.Tensor:
        if "i2" in self.mode:
            info = self.i0(f0) + self.i1(f1)
        else:
            info = self.conv(f0 + f1)

        if "d" in self.mode:
            if "ad2" in self.mode:
                disappear = self.dis(self.relu(f0 - f1))
            else:
                disappear = self.res(self.relu(f0 - f1))
        else:
            disappear = 0

        if "a" in self.mode:
            if "ad2" in self.mode:
                appear = self.app(self.relu(f1 - f0))
            else:
                appear = self.res(self.relu(f1 - f0))
        else:
            appear = 0

        if "e" in self.mode:
            exchange = self.exchange(torch.max(f0, f1) - torch.min(f0, f1))
        else:
            exchange = 0

        if self.mode == "i":
            f = info
        elif self.mode == "a":
            f = appear
        elif self.mode == "d":
            f = disappear
        elif self.mode == "e":
            f = exchange
        elif self.mode == "ia":
            f = info + 2 * appear
        elif self.mode in ["id", "i2d"]:
            f = info + 2 * disappear
        elif self.mode == "ie":
            f = info + 2 * exchange
        elif self.mode == "iae":
            f = info + appear + exchange
        elif self.mode == "ide":
            f = info + disappear + exchange
        elif self.mode == "iad":
            f = info + disappear + appear
        else:
            f = info + disappear + appear + exchange

        return self.relu(f)


class MSF(nn.Module):
    """Multi-Scale Feature (MSF) fusion module from C-3PO (DoctorKey/C-3PO)."""

    def __init__(
        self,
        channels: List[int],
        total_f: int = 5,
        fpn_channel: int = 256,
        with_bn: bool = True,
        mode: str = "iade",
    ):
        super().__init__()
        self.num_f = len(channels)
        self.total_f = total_f
        assert 0 < self.num_f <= self.total_f

        cf_list = []
        cf_inner = []
        cf_layer = []
        for i in range(self.num_f):
            cf_list.append(MTF(channels[i], mode, kernel_size=3))
            cf_inner.append(self._make_layer(channels[i], fpn_channel, 1, with_bn))
            cf_layer.append(self._make_layer(fpn_channel, fpn_channel, 3, with_bn))

        self.cfs = nn.ModuleList(cf_list)
        self.cf_inners = nn.ModuleList(cf_inner)
        self.cf_layers = nn.ModuleList(cf_layer)

        self.reduce = nn.Conv2d(fpn_channel * self.num_f, fpn_channel, 3, padding=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(fpn_channel)
        self.relu = nn.ReLU(inplace=True)

    @staticmethod
    def _make_layer(in_channel: int, out_channel: int, kernel: int, with_bn: bool) -> nn.Sequential:
        layers: List[nn.Module] = []
        if kernel == 1:
            layers.append(nn.Conv2d(in_channel, out_channel, 1, bias=not with_bn))
        elif kernel == 3:
            layers.append(nn.Conv2d(in_channel, out_channel, 3, padding=1, bias=not with_bn))
        else:
            raise ValueError(f"Unsupported kernel size: {kernel}")

        if with_bn:
            layers.append(nn.BatchNorm2d(out_channel))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, t0_fs: List[torch.Tensor], t1_fs: List[torch.Tensor]) -> torch.Tensor:
        cfs = []
        for i in range(self.num_f):
            k = i + self.total_f - self.num_f
            cfs.append(self.cfs[i](t0_fs[k], t1_fs[k]))

        resize_shape = cfs[0].shape[-2:]
        final_list = []
        last_inner = None
        for i in range(self.num_f - 1, -1, -1):
            cf = self.cf_inners[i](cfs[i])
            if last_inner is None:
                last_inner = cf
            else:
                inner_top_down = F.interpolate(last_inner, size=cf.shape[-2:], mode="nearest")
                last_inner = cf + inner_top_down
            cf_layer = self.cf_layers[i](last_inner)
            final_list.append(cf_layer)

        final_list = [
            F.interpolate(cf_layer, size=resize_shape, mode="bilinear", align_corners=False)
            for cf_layer in final_list
        ]
        cf = torch.cat(final_list, dim=1)
        cf = self.relu(self.bn(self.reduce(cf)))
        return cf


class _ASPPConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int):
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class _ASPPPooling(nn.Module):
    """Global context pooling branch for ASPP that safely handles B=1 during training."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(num_groups=min(32, out_channels), num_channels=out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        out = self.pool(x)
        out = self.conv(out)
        out = self.norm(out)
        out = self.relu(out)
        return F.interpolate(out, size=size, mode="bilinear", align_corners=False)


class _DeepLabHead(nn.Module):
    """DeepLabV3 ASPP + Classifier Head used in C-3PO."""

    def __init__(
        self,
        in_channels: int = 256,
        aspp_channels: int = 256,
        num_classes: int = 1,
        atrous_rates: Tuple[int, int, int] = (12, 24, 36),
    ):
        super().__init__()
        modules: List[nn.Module] = [
            nn.Sequential(
                nn.Conv2d(in_channels, aspp_channels, 1, bias=False),
                nn.BatchNorm2d(aspp_channels),
                nn.ReLU(inplace=True),
            )
        ]
        for rate in atrous_rates:
            modules.append(_ASPPConv(in_channels, aspp_channels, rate))
        modules.append(_ASPPPooling(in_channels, aspp_channels))

        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * aspp_channels, aspp_channels, 1, bias=False),
            nn.BatchNorm2d(aspp_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(aspp_channels, aspp_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(aspp_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(aspp_channels, num_classes, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = [conv(x) for conv in self.convs]
        x = torch.cat(res, dim=1)
        x = self.project(x)
        return self.classifier(x)


class C3PONet(nn.Module):
    """C-3PO (Combine 3 POssible change types) Network for Scene Change Detection.

    Default configuration matches the official C-3PO ResNet-18 + MTF(iade) + MSF(4) + DeepLabV3
    architecture evaluated on PCD, VL-CMU-CD, and ChangeSim.
    """

    def __init__(
        self,
        pretrained: bool = False,
        mtf_mode: str = "iade",
        msf_num: int = 4,
        fpn_channel: int = 256,
    ):
        super().__init__()
        backbone = resnet18(weights=None if not pretrained else "DEFAULT")
        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        all_channels = [64, 64, 128, 256, 512]
        self.msf = MSF(
            channels=all_channels[-msf_num:],
            total_f=len(all_channels),
            fpn_channel=fpn_channel,
            with_bn=True,
            mode=mtf_mode,
        )
        self.head = _DeepLabHead(
            in_channels=fpn_channel,
            aspp_channels=fpn_channel,
            num_classes=1,
        )

    def _encode_single(self, x: torch.Tensor) -> List[torch.Tensor]:
        f0 = self.layer0(x)
        f1 = self.layer1(f0)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        return [f0, f1, f2, f3, f4]

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> Dict[str, torch.Tensor]:
        target_size = (img1.shape[-2], img1.shape[-1])
        t0_fs = self._encode_single(img1)
        t1_fs = self._encode_single(img2)

        fused = self.msf(t0_fs, t1_fs)
        logits = self.head(fused)
        if logits.shape[-2:] != target_size:
            logits = F.interpolate(logits, size=target_size, mode="bilinear", align_corners=False)

        prob = torch.sigmoid(logits)
        return {
            "logits": logits,
            "prob": prob,
        }
