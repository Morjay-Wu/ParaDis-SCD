"""CSNet: ChangeSim Official Baseline for Scene Change Detection (ICRA 2021).

Paper: "ChangeSim: Towards End-to-End Online Scene Change Detection in Industrial Indoor Environments"
IEEE International Conference on Robotics and Automation (ICRA 2021).

Key Characteristics:
1. Siamese ResNet encoder sharing weights across time steps t1 and t2.
2. Multi-scale feature difference extraction combining concats and absolute difference:
   D_i = Conv([F_1^i, F_2^i, |F_1^i - F_2^i|]).
3. Progressive deconvolution decoder with lateral skip connections.
4. Native support for arbitrary rectangular image resolutions (e.g. 480x640, 224x1024).
5. Returns unified dictionary contract: {'logits': [B, 1, H, W], 'prob': [B, 1, H, W]}.
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """3x3 convolution with symmetric padding."""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution."""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        bias=False,
    )


class ResNetBlock(nn.Module):
    """Residual building block for CSNet Siamese encoder."""

    expansion = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)
        return out


class SiameseResNetEncoder(nn.Module):
    """Siamese ResNet Backbone supporting arbitrary rectangular resolutions."""

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        layers: Tuple[int, int, int, int] = (2, 2, 2, 2),
    ):
        super().__init__()
        self.in_planes = base_channels

        # Stem: stride 4 initial downsampling
        self.conv1 = nn.Conv2d(
            in_channels,
            base_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 4 Residual Stages: strides 4, 8, 16, 32
        self.layer1 = self._make_layer(base_channels, layers[0], stride=1)
        self.layer2 = self._make_layer(base_channels * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(base_channels * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(base_channels * 8, layers[3], stride=2)

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_planes != planes * ResNetBlock.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_planes, planes * ResNetBlock.expansion, stride=stride),
                nn.BatchNorm2d(planes * ResNetBlock.expansion),
            )

        layers = []
        layers.append(
            ResNetBlock(
                self.in_planes,
                planes,
                stride=stride,
                downsample=downsample,
            )
        )
        self.in_planes = planes * ResNetBlock.expansion
        for _ in range(1, blocks):
            layers.append(ResNetBlock(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward_single(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract multi-scale pyramid features for a single time step."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # stride 4: [B, base, H/4, W/4]

        c1 = self.layer1(x)   # stride 4: [B, base, H/4, W/4]
        c2 = self.layer2(c1)  # stride 8: [B, 2*base, H/8, W/8]
        c3 = self.layer3(c2)  # stride 16: [B, 4*base, H/16, W/16]
        c4 = self.layer4(c3)  # stride 32: [B, 8*base, H/32, W/32]
        return [c1, c2, c3, c4]

    def forward(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Shared Siamese forward pass on image pair."""
        feats1 = self.forward_single(img1)
        feats2 = self.forward_single(img2)
        return feats1, feats2


class DifferenceFusionBlock(nn.Module):
    """Multi-Scale Difference Module.
    
    Computes genuine feature discrepancy combining bi-temporal representations:
    D = Conv([F1, F2, |F1 - F2|]).
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # Concatenation of [F1, F2, |F1 - F2|] has 3 * in_channels
        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels * 3, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, f1: torch.Tensor, f2: torch.Tensor) -> torch.Tensor:
        diff = torch.abs(f1 - f2)
        cat = torch.cat([f1, f2, diff], dim=1)
        return self.fuse(cat)


class ProgressiveDeconvDecoder(nn.Module):
    """Progressive Deconvolution / Upsampling Decoder for CSNet."""

    def __init__(self, channels: List[int], dec_channels: int = 128):
        super().__init__()
        c1, c2, c3, c4 = channels

        # Difference modules at 4 scales
        self.diff4 = DifferenceFusionBlock(c4, dec_channels)
        self.diff3 = DifferenceFusionBlock(c3, dec_channels)
        self.diff2 = DifferenceFusionBlock(c2, dec_channels)
        self.diff1 = DifferenceFusionBlock(c1, dec_channels)

        # Progressive deconv / fusion stages
        # Stage 4 to 3 (stride 32 -> 16)
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(dec_channels, dec_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dec_channels),
            nn.ReLU(inplace=True),
        )
        self.fuse3 = nn.Sequential(
            nn.Conv2d(dec_channels * 2, dec_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dec_channels),
            nn.ReLU(inplace=True),
        )

        # Stage 3 to 2 (stride 16 -> 8)
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(dec_channels, dec_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dec_channels),
            nn.ReLU(inplace=True),
        )
        self.fuse2 = nn.Sequential(
            nn.Conv2d(dec_channels * 2, dec_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dec_channels),
            nn.ReLU(inplace=True),
        )

        # Stage 2 to 1 (stride 8 -> 4)
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(dec_channels, dec_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dec_channels),
            nn.ReLU(inplace=True),
        )
        self.fuse1 = nn.Sequential(
            nn.Conv2d(dec_channels * 2, dec_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dec_channels),
            nn.ReLU(inplace=True),
        )

        # Final upsampling from stride 4 to original resolution (stride 1)
        self.up_final = nn.Sequential(
            nn.ConvTranspose2d(dec_channels, dec_channels // 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dec_channels // 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(dec_channels // 2, dec_channels // 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dec_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(dec_channels // 4, 1, kernel_size=3, padding=1),
        )

    def forward(
        self,
        feats1: List[torch.Tensor],
        feats2: List[torch.Tensor],
        target_size: Tuple[int, int],
    ) -> torch.Tensor:
        c1_1, c2_1, c3_1, c4_1 = feats1
        c1_2, c2_2, c3_2, c4_2 = feats2

        # 1. Multi-scale feature differences
        d4 = self.diff4(c4_1, c4_2)  # [B, dec, H/32, W/32]
        d3 = self.diff3(c3_1, c3_2)  # [B, dec, H/16, W/16]
        d2 = self.diff2(c2_1, c2_2)  # [B, dec, H/8, W/8]
        d1 = self.diff1(c1_1, c1_2)  # [B, dec, H/4, W/4]

        # 2. Progressive deconvolution with spatial matching
        u4 = self.up4(d4)
        if u4.shape[-2:] != d3.shape[-2:]:
            u4 = F.interpolate(u4, size=d3.shape[-2:], mode='bilinear', align_corners=False)
        m3 = self.fuse3(torch.cat([u4, d3], dim=1))

        u3 = self.up3(m3)
        if u3.shape[-2:] != d2.shape[-2:]:
            u3 = F.interpolate(u3, size=d2.shape[-2:], mode='bilinear', align_corners=False)
        m2 = self.fuse2(torch.cat([u3, d2], dim=1))

        u2 = self.up2(m2)
        if u2.shape[-2:] != d1.shape[-2:]:
            u2 = F.interpolate(u2, size=d1.shape[-2:], mode='bilinear', align_corners=False)
        m1 = self.fuse1(torch.cat([u2, d1], dim=1))

        # 3. Final deconvolution to full resolution
        logits = self.up_final(m1)
        if logits.shape[-2:] != target_size:
            logits = F.interpolate(logits, size=target_size, mode='bilinear', align_corners=False)

        return logits


class CSNet(nn.Module):
    """ChangeSim Official Baseline Network (CSNet).
    
    Args:
        in_channels: Input image channel count (default 3).
        base_channels: Base channel width of ResNet backbone (default 64).
        dec_channels: Decoder hidden feature channel dimension (default 128).
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        dec_channels: int = 128,
    ):
        super().__init__()
        self.encoder = SiameseResNetEncoder(
            in_channels=in_channels,
            base_channels=base_channels,
            layers=(2, 2, 2, 2),
        )
        channels = [
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
        ]
        self.decoder = ProgressiveDeconvDecoder(channels=channels, dec_channels=dec_channels)

    def forward(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for CSNet.
        
        Args:
            img1: Image at time 1 [B, 3, H, W]
            img2: Image at time 2 [B, 3, H, W]
            
        Returns:
            Dictionary matching unified baseline contract:
                'logits': Raw change logits [B, 1, H, W]
                'prob': Sigmoid change probability [B, 1, H, W] in [0, 1]
        """
        target_size = (img1.shape[-2], img1.shape[-1])
        feats1, feats2 = self.encoder(img1, img2)
        logits = self.decoder(feats1, feats2, target_size=target_size)
        prob = torch.sigmoid(logits)
        return {
            'logits': logits,
            'prob': prob,
        }
