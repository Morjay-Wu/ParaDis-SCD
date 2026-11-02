"""Dynamic Rectangular Feature Extractor for ParaDis-SCD.

Supports arbitrary rectangular image resolutions without square-grid assumptions.
Provides multi-scale pyramid features (c2, c3, c4, shape), reflective padding to
stride multiples (16/32), and dynamic 2D token spatialization.
"""

from typing import Dict, List, Optional, Tuple, Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except ImportError:
    timm = None

try:
    import torchvision
except ImportError:
    torchvision = None


class SSFCGate(nn.Module):
    """Parameter-free Spatial-Spectral Feature Cooperation (SSFC) statistical attention gate."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        q = x.mean(dim=(-2, -1), keepdim=True)
        sq = (x - q).pow(2)
        sigma = sq.sum(dim=(-2, -1), keepdim=True) / max(1, h * w)
        att = torch.sigmoid(sq / (2.0 * sigma + 1e-5) + 0.5)
        return x * att


class ConvNeXtBlock(nn.Module):
    """Multi-Scale Dilated ConvNeXt Block with depthwise 7x7 + dilated 3x3 conv and SSFC attention."""

    def __init__(self, dim: int, drop_path: float = 0.0):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.dwconv_dilated = nn.Conv2d(dim, dim, kernel_size=3, padding=2, dilation=2, groups=dim)
        self.norm = nn.GroupNorm(1, dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.ssfc = SSFCGate()
        self.gamma = nn.Parameter(0.25 * torch.ones((dim,)), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x) + self.dwconv_dilated(x)
        x = self.norm(x)
        # Permute for channel-last linear operations: (B, C, H, W) -> (B, H, W, C)
        x = x.permute(0, 2, 3, 1)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        return shortcut + self.ssfc(x)


class NativeConvNeXtBackbone(nn.Module):
    """Pure PyTorch native Multi-Scale ConvNeXt feature extractor with zero external dependencies.
    
    Guarantees 100% offline self-contained operation on arbitrary rectangular resolutions.
    Extracts high-resolution stride-2 stem feature (c1) plus stride 4, 8, 16, 32 pyramid stages.
    """

    def __init__(
        self,
        in_chans: int = 3,
        depths: Tuple[int, ...] = (2, 2, 4, 2),
        dims: Tuple[int, ...] = (64, 128, 256, 512),
        c1_dim: int = 32,
    ):
        super().__init__()
        self.dims = dims
        self.c1_dim = c1_dim

        # High-resolution stride-2 stem preserving crisp object boundaries
        self.stem_c1 = nn.Sequential(
            nn.Conv2d(in_chans, c1_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(min(8, c1_dim), c1_dim, eps=1e-6),
            nn.GELU(),
            nn.Conv2d(c1_dim, c1_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(min(8, c1_dim), c1_dim, eps=1e-6),
            nn.GELU(),
            SSFCGate(),
        )

        self.downsample_layers = nn.ModuleList()

        # Stem downsample from stride 2 to stride 4
        stem_s4 = nn.Sequential(
            nn.Conv2d(c1_dim, dims[0], kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(min(8, dims[0]), dims[0], eps=1e-6),
            nn.GELU(),
        )
        self.downsample_layers.append(stem_s4)

        # Intermediate downsamples: 2x downsamples (strides 8, 16, 32)
        for i in range(3):
            down = nn.Sequential(
                nn.GroupNorm(1, dims[i], eps=1e-6),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(down)

        self.stages = nn.ModuleList()
        for i in range(4):
            stage = nn.Sequential(*[ConvNeXtBlock(dim=dims[i]) for _ in range(depths[i])])
            self.stages.append(stage)

    def forward_with_c1(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        c1 = self.stem_c1(x)
        features = []
        curr = c1
        for i in range(4):
            curr = self.downsample_layers[i](curr)
            curr = self.stages[i](curr)
            features.append(curr)
        return c1, features

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        _, features = self.forward_with_c1(x)
        return features


class DynamicBackbone(nn.Module):
    """Dynamic Rectangular Feature Extractor for ParaDis-SCD.

    Extracts multi-scale pyramid features ('c2', 'c3', 'c4') with reflective padding
    to stride 16 (or 32), supporting arbitrary rectangular image shapes (e.g. 480x640,
    224x1024, 512x682).

    Args:
        name: Backbone architecture name:
              - 'convnext_tiny' (timm or native ConvNeXt)
              - 'resnet18' / 'resnet34' / 'resnet50'
              - 'vit_small_patch16_224' (with dynamic_img_size=True)
              - 'native_convnext' (built-in offline ConvNeXt)
        pretrained: Whether to load pretrained weights (default False).
        out_channels: Standardized channel dimension for primary semantic feature c4 (default 256).
        freeze: Whether to freeze backbone parameters.
    """

    def __init__(
        self,
        name: str = 'convnext_tiny',
        pretrained: bool = False,
        out_channels: int = 256,
        freeze: bool = False,
    ):
        super().__init__()
        self.name = name.lower()
        self.pretrained = pretrained
        self.out_channels = out_channels
        self.freeze = freeze
        self.align_stride = 16

        # Target channel dimensions for pyramid features
        self.c2_dim = 64
        self.c3_dim = 128
        self.c4_dim = out_channels

        self._build_backbone()

        # Projections to standardize multi-scale channels
        self.proj_c2 = nn.Sequential(
            nn.Conv2d(self.raw_dims[0], self.c2_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.c2_dim),
            nn.ReLU(inplace=True),
        )
        self.proj_c3 = nn.Sequential(
            nn.Conv2d(self.raw_dims[1], self.c3_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.c3_dim),
            nn.ReLU(inplace=True),
        )
        self.proj_c4 = nn.Sequential(
            nn.Conv2d(self.raw_dims[2], self.c4_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.c4_dim),
            nn.ReLU(inplace=True),
        )

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

    def _build_backbone(self):
        if 'vit' in self.name:
            self.arch_type = 'vit'
            self._build_vit()
        elif 'resnet' in self.name:
            self.arch_type = 'resnet'
            self._build_resnet()
        elif self.name == 'native_convnext' or timm is None:
            self.arch_type = 'native_convnext'
            self.backbone = NativeConvNeXtBackbone(dims=(64, 128, 256, 512))
            self.raw_dims = [64, 128, 256, 512]
            self.align_stride = 16
        else:
            self.arch_type = 'convnext'
            self._build_convnext()

    def _build_vit(self):
        if timm is None:
            raise ImportError("timm is required for ViT backbone.")
        self.patch_size = 16
        self.align_stride = 16
        self.backbone = timm.create_model(
            self.name,
            pretrained=self.pretrained,
            dynamic_img_size=True,
            num_classes=0,
        )
        embed_dim = self.backbone.embed_dim
        self.num_prefix = getattr(self.backbone, 'num_prefix_tokens', 1)
        # ViT produces single scale at patch_size (stride 16); construct pseudo-pyramid
        self.raw_dims = [embed_dim, embed_dim, embed_dim]

    def _build_convnext(self):
        try:
            self.backbone = timm.create_model(
                self.name,
                pretrained=self.pretrained,
                features_only=True,
            )
            feature_channels = self.backbone.feature_info.channels()
            self.raw_dims = [feature_channels[0], feature_channels[1], feature_channels[2]]
            self.align_stride = 16
        except Exception:
            # Fallback to native ConvNeXt
            self.arch_type = 'native_convnext'
            self.backbone = NativeConvNeXtBackbone(dims=(64, 128, 256, 512))
            self.raw_dims = [64, 128, 256, 512]
            self.align_stride = 16

    def _build_resnet(self):
        self.align_stride = 16
        if torchvision is not None and hasattr(torchvision.models, self.name):
            weights = getattr(torchvision.models, f"{self.name.capitalize()}_Weights").DEFAULT if self.pretrained else None
            net = getattr(torchvision.models, self.name)(weights=weights)
            self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
            self.layer1 = net.layer1
            self.layer2 = net.layer2
            self.layer3 = net.layer3
            self.layer4 = getattr(net, 'layer4', None)
            self.backbone = nn.Module()
            self.backbone.stem = self.stem
            self.backbone.layer1 = self.layer1
            self.backbone.layer2 = self.layer2
            self.backbone.layer3 = self.layer3
            if self.layer4 is not None:
                self.backbone.layer4 = self.layer4
            
            is_large = ('50' in self.name or '101' in self.name)
            self.raw_dims = [256, 512, 1024] if is_large else [64, 128, 256]
        elif timm is not None:
            self.backbone = timm.create_model(self.name, pretrained=self.pretrained, features_only=True)
            info = self.backbone.feature_info.channels()
            self.raw_dims = [info[1], info[2], info[3]]
        else:
            # Fallback to native ConvNeXt
            self.arch_type = 'native_convnext'
            self.backbone = NativeConvNeXtBackbone(dims=(64, 128, 256, 512))
            self.raw_dims = [64, 128, 256, 512]

    def _pad_to_alignment(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
        """Reflectively pad tensor so H and W are multiples of align_stride.

        Returns:
            padded_x: Tensor with dimensions divisible by align_stride.
            padding: (pad_left, pad_right, pad_top, pad_bottom)
        """
        B, C, H, W = x.shape
        pad_h = (self.align_stride - (H % self.align_stride)) % self.align_stride
        pad_w = (self.align_stride - (W % self.align_stride)) % self.align_stride

        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        if pad_h > 0 or pad_w > 0:
            if H > max(pad_top, pad_bottom) and W > max(pad_left, pad_right):
                x_pad = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode='reflect')
            else:
                x_pad = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode='replicate')
        else:
            x_pad = x

        return x_pad, (pad_left, pad_right, pad_top, pad_bottom)

    def forward(self, x: torch.Tensor) -> Dict[str, Union[torch.Tensor, Tuple[int, int], Tuple[int, int, int, int]]]:
        """Forward pass supporting arbitrary rectangular input resolutions.

        Args:
            x: Input image tensor [B, 3, H, W]

        Returns:
            Dict containing:
                'c2': Feature map at stride 4 [B, 64, H/4, W/4]
                'c3': Feature map at stride 8 [B, 128, H/8, W/8]
                'c4': Feature map at stride 16 [B, out_channels, H/16, W/16]
                'shape': Original image resolution (H, W)
                'pad': Applied padding tuple (left, right, top, bottom)
        """
        B, C, H, W = x.shape
        x_pad, padding = self._pad_to_alignment(x)
        B, C, H_pad, W_pad = x_pad.shape

        if self.arch_type == 'vit':
            # Dynamic 2D rectangular unrolling without square-grid assumptions
            H_p = H_pad // self.patch_size
            W_p = W_pad // self.patch_size

            tokens = self.backbone.forward_features(x_pad)  # [B, N_prefix + H_p * W_p, D]
            spatial_tokens = tokens[:, self.num_prefix : self.num_prefix + H_p * W_p, :]
            # Explicit (H_p, W_p) 2D spatialization
            feat_c4_raw = spatial_tokens.reshape(B, H_p, W_p, -1).permute(0, 3, 1, 2).contiguous()
            
            # Interpolate to create hierarchical representations for c2, c3
            feat_c3_raw = F.interpolate(feat_c4_raw, size=(H_pad // 8, W_pad // 8), mode='bilinear', align_corners=False)
            feat_c2_raw = F.interpolate(feat_c4_raw, size=(H_pad // 4, W_pad // 4), mode='bilinear', align_corners=False)

        elif self.arch_type == 'resnet':
            if hasattr(self, 'stem'):
                s1 = self.stem(x_pad)     # stride 4
                c2_raw = self.layer1(s1)  # stride 4
                c3_raw = self.layer2(c2_raw)  # stride 8
                c4_raw = self.layer3(c3_raw)  # stride 16
            else:
                feats = self.backbone(x_pad)
                c2_raw, c3_raw, c4_raw = feats[1], feats[2], feats[3]
            feat_c2_raw, feat_c3_raw, feat_c4_raw = c2_raw, c3_raw, c4_raw

        elif self.arch_type == 'native_convnext':
            c1, feats = self.backbone.forward_with_c1(x_pad)  # c1: stride 2, feats: strides [4, 8, 16, 32]
            feat_c2_raw, feat_c3_raw, feat_c4_raw = feats[0], feats[1], feats[2]

        else:  # 'convnext' via timm
            feats = self.backbone(x_pad)
            feat_c2_raw, feat_c3_raw, feat_c4_raw = feats[0], feats[1], feats[2]

        # Standardize channels
        c2 = self.proj_c2(feat_c2_raw)
        c3 = self.proj_c3(feat_c3_raw)
        c4 = self.proj_c4(feat_c4_raw)
        if self.arch_type != 'native_convnext':
            c1 = F.interpolate(c2[:, :32], size=(H_pad // 2, W_pad // 2), mode='bilinear', align_corners=False)

        return {
            'c1': c1,
            'c2': c2,
            'c3': c3,
            'c4': c4,
            'shape': (H, W),
            'pad': padding,
        }
