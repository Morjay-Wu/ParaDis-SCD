"""Zero-Sensor Monocular Depth & Surface Normal Fields (MG-Priors).

Estimates continuous relative monocular depth maps and unit surface normal fields
in S^2 from single RGB images without requiring physical depth sensors.
Surface normals are strictly normalized to unit length: ||n||_2 = 1.0.
"""

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class ASPPModule(nn.Module):
    """Atrous Spatial Pyramid Pooling for multi-scale geometric contextual receptive field."""

    def __init__(self, in_channels: int, out_channels: int, dilations: Tuple[int, ...] = (1, 3, 6)):
        super().__init__()
        self.branches = nn.ModuleList()
        # 1x1 conv
        self.branches.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.GroupNorm(4, out_channels),
                nn.GELU(),
            )
        )
        # Dilated 3x3 convs
        for d in dilations:
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=d, dilation=d, bias=False),
                    nn.GroupNorm(4, out_channels),
                    nn.GELU(),
                )
            )
        # Global average pooling branch
        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(4, out_channels),
            nn.GELU(),
        )
        total_in = out_channels * (len(dilations) + 2)
        self.project = nn.Sequential(
            nn.Conv2d(total_in, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(4, out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        outs = [b(x) for b in self.branches]
        g = F.interpolate(self.global_branch(x), size=(H, W), mode='bilinear', align_corners=False)
        outs.append(g)
        return self.project(torch.cat(outs, dim=1))


class MGPriorsOutput(dict):
    """Container supporting both dictionary indexing and tuple unpacking."""

    def __iter__(self):
        yield self['depth']
        yield self['normal']


class MGPriors(nn.Module):
    """Zero-Sensor Monocular Depth & Surface Normal Fields (MG-Priors).

    Generates relative scalar depth fields zeta in R+ and unit surface normal fields
    n in S^2 from single RGB images supporting arbitrary rectangular resolutions.

    Args:
        in_channels: Input image channels (default 3).
        base_channels: Base channel capacity of encoder (default 32).
        depth_min: Minimum positive depth offset (default 0.1).
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        depth_min: float = 0.1,
    ):
        super().__init__()
        self.depth_min = depth_min
        C = base_channels

        # Encoder stem: 2x downsample
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, C, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(4, C),
            nn.GELU(),
            nn.Conv2d(C, C, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, C),
            nn.GELU(),
        )

        # Stage 1: 4x downsample
        self.stage1 = nn.Sequential(
            nn.Conv2d(C, 2 * C, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 2 * C),
            nn.GELU(),
            nn.Conv2d(2 * C, 2 * C, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 2 * C),
            nn.GELU(),
        )

        # Stage 2: 8x downsample
        self.stage2 = nn.Sequential(
            nn.Conv2d(2 * C, 4 * C, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 4 * C),
            nn.GELU(),
            nn.Conv2d(4 * C, 4 * C, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 4 * C),
            nn.GELU(),
        )

        # Context ASPP at stride 8
        self.aspp = ASPPModule(4 * C, 2 * C, dilations=(1, 2, 4))

        # Progressive Decoder
        # 8x -> 4x
        self.up2 = nn.Sequential(
            nn.Conv2d(2 * C + 2 * C, 2 * C, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 2 * C),
            nn.GELU(),
        )
        # 4x -> 2x
        self.up1 = nn.Sequential(
            nn.Conv2d(2 * C + C, C, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, C),
            nn.GELU(),
        )
        # 2x -> 1x
        self.final_up = nn.Sequential(
            nn.Conv2d(C, C, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, C),
            nn.GELU(),
        )

        # Head 1: Relative Depth Head
        self.depth_head = nn.Sequential(
            nn.Conv2d(C, C // 2, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(2, C // 2),
            nn.GELU(),
            nn.Conv2d(C // 2, 1, kernel_size=1),
        )

        # Head 2: Unit Surface Normal Head (S^2 manifold)
        self.normal_head = nn.Sequential(
            nn.Conv2d(C, C // 2, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(2, C // 2),
            nn.GELU(),
            nn.Conv2d(C // 2, 3, kernel_size=1),
        )

        # Initial normal bias: point towards camera along z-axis [0, 0, 1]
        with torch.no_grad():
            self.normal_head[-1].bias.zero_()
            self.normal_head[-1].bias[2] = 1.0

    def forward(self, img: torch.Tensor) -> MGPriorsOutput:
        """Forward pass estimating relative depth and unit surface normals.

        Args:
            img: Input image tensor [B, 3, H, W]

        Returns:
            MGPriorsOutput dict containing:
                'depth': Scalar depth map [B, 1, H, W] in R+
                'normal': Unit surface normal field [B, 3, H, W] in S^2 (||n||_2 = 1.0)
        """
        B, C_in, H, W = img.shape

        # Multi-scale encoder features
        f_stem = self.stem(img)       # [B, C, H/2, W/2]
        f_s1 = self.stage1(f_stem)    # [B, 2C, H/4, W/4]
        f_s2 = self.stage2(f_s1)      # [B, 4C, H/8, W/8]

        # Context pooling
        f_ctx = self.aspp(f_s2)       # [B, 2C, H/8, W/8]

        # Progressive upsampling with skips
        up_s2 = F.interpolate(f_ctx, size=f_s1.shape[-2:], mode='bilinear', align_corners=False)
        f_dec1 = self.up2(torch.cat([up_s2, f_s1], dim=1))  # [B, 2C, H/4, W/4]

        up_s1 = F.interpolate(f_dec1, size=f_stem.shape[-2:], mode='bilinear', align_corners=False)
        f_dec0 = self.up1(torch.cat([up_s1, f_stem], dim=1))  # [B, C, H/2, W/2]

        up_s0 = F.interpolate(f_dec0, size=(H, W), mode='bilinear', align_corners=False)
        f_feat = self.final_up(up_s0)  # [B, C, H, W]

        # Relative Depth estimation (strictly positive)
        raw_depth = self.depth_head(f_feat)
        depth = F.softplus(raw_depth) + self.depth_min

        # Couple spatial depth gradients (-dz/dx, -dz/dy, 1) to prevent constant [0, 0, 1] normal collapse
        dz_dx = depth[:, :, :, 1:] - depth[:, :, :, :-1]
        dz_dx = F.pad(dz_dx, (0, 1, 0, 0), mode='replicate')
        dz_dy = depth[:, :, 1:, :] - depth[:, :, :-1, :]
        dz_dy = F.pad(dz_dy, (0, 0, 0, 1), mode='replicate')
        geom_normal = torch.cat([-2.0 * dz_dx, -2.0 * dz_dy, torch.ones_like(depth)], dim=1)

        # Surface Normal estimation with strict S^2 unit-length projection
        raw_normal = self.normal_head(f_feat) + 0.5 * geom_normal
        normal_len = torch.norm(raw_normal, p=2, dim=1, keepdim=True).clamp(min=1e-7)
        normal = raw_normal / normal_len

        return MGPriorsOutput({
            'depth': depth,
            'normal': normal,
            'feature': f_feat,
        })

    @staticmethod
    def downsample_priors(
        depth: torch.Tensor,
        normal: torch.Tensor,
        size: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Resamples depth and unit normal to feature scale (H_p, W_p) with S^2 re-normalization.

        Args:
            depth: Full-res depth [B, 1, H, W]
            normal: Full-res normal [B, 3, H, W]
            size: Target feature size (H_p, W_p)

        Returns:
            depth_p: Feature-scale depth [B, 1, H_p, W_p]
            normal_p: Feature-scale unit normal [B, 3, H_p, W_p] (||n||_2 = 1.0)
        """
        depth_p = F.interpolate(depth, size=size, mode='bilinear', align_corners=False)
        normal_interp = F.interpolate(normal, size=size, mode='bilinear', align_corners=False)
        norm_len = torch.norm(normal_interp, p=2, dim=1, keepdim=True).clamp(min=1e-7)
        normal_p = normal_interp / norm_len
        return depth_p, normal_p
