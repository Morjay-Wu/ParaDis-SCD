"""Parallax-Suppressed Multi-Scale Progressive Decoder.

Progressively restores full rectangular image resolution (H, W) through hierarchical
residual upsampling blocks (stride 16 -> 8 -> 4 -> 2 -> 1) with skip connections from
multi-scale backbone features (c2, c3). Supports arbitrary rectangular resolutions,
reflective padding reversal, and outputs 1-channel raw change logits and probabilities.
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderBlock(nn.Module):
    """Multi-scale residual upsampling block with dilated perception, SSFC, GroupNorm, and GELU."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.act1 = nn.GELU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.conv2_dilated = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=2, dilation=2, groups=out_channels, bias=False
        )
        self.norm2 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.act2 = nn.GELU()

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.GroupNorm(min(8, out_channels), out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    @staticmethod
    def _ssfc(x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        q = x.mean(dim=(-2, -1), keepdim=True)
        sq = (x - q).pow(2)
        sigma = sq.sum(dim=(-2, -1), keepdim=True) / max(1, h * w)
        att = torch.sigmoid(sq / (2.0 * sigma + 1e-5) + 0.5)
        return x * att

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.act1(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out) + self.conv2_dilated(out))
        out = self._ssfc(out)
        out = self.act2(out + res)
        return out


class BiTemporalParallaxSkipFusion(nn.Module):
    """Aligns and fuses bi-temporal multi-scale skip features using epipolar flow & disocclusion gate."""

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.warp_logit = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.occ_weight = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 4, channels, kernel_size=1, bias=False),
            nn.GroupNorm(min(8, channels), channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(8, channels), channels),
            nn.GELU(),
        )

    def forward(
        self,
        f1: torch.Tensor,
        f2: Optional[torch.Tensor] = None,
        warp_grid_12: Optional[torch.Tensor] = None,
        g_occ: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, C, H, W = f1.shape
        if f2 is None:
            f2 = torch.zeros_like(f1)

        d_raw = torch.abs(f1 - f2)

        if warp_grid_12 is not None:
            grid_up = F.interpolate(warp_grid_12, size=(H, W), mode='bilinear', align_corners=False)
            grid_sample_coords = grid_up.permute(0, 2, 3, 1)
            f2_warp = F.grid_sample(
                f2,
                grid_sample_coords,
                mode='bilinear',
                padding_mode='border',
                align_corners=False,
            )
            # Pixel-wise adaptive alignment selection: only blend f2_warp where it reduces discrepancy
            with torch.no_grad():
                err_raw = d_raw.mean(dim=1, keepdim=True)
                err_warp = torch.abs(f1 - f2_warp).mean(dim=1, keepdim=True)
                gain_gate = torch.sigmoid((err_raw - err_warp) * 6.0)
            w = torch.sigmoid(self.warp_logit).to(dtype=f1.dtype) * gain_gate
            f2_align = (1.0 - w) * f2 + w * f2_warp
        else:
            w = torch.sigmoid(self.warp_logit).to(dtype=f1.dtype)
            f2_align = f2 * (0.5 + 0.5 * w)

        d_align = torch.abs(f1 - f2_align)
        d_persist = torch.minimum(d_raw, d_align)

        if g_occ is not None:
            g_up = F.interpolate(g_occ, size=(H, W), mode='bilinear', align_corners=False)
            occ_w = 0.25 * torch.sigmoid(self.occ_weight).to(dtype=f1.dtype)
            d_pure = d_persist * (1.0 - occ_w * g_up)
        else:
            occ_w = 0.25 * torch.sigmoid(self.occ_weight).to(dtype=f1.dtype)
            d_pure = d_persist * (0.75 + 0.25 * occ_w)

        cat_feat = torch.cat([f1, f2_align, d_raw, d_pure], dim=1)
        return self.fuse(cat_feat)


class PanoramicChangeGuide(nn.Module):
    """Lightweight panoramic change-guided long-range strip & prototype context aggregation."""

    def __init__(self, channels: int):
        super().__init__()
        self.strip_h = nn.Conv2d(
            channels, channels, kernel_size=(1, 7), padding=(0, 3), groups=channels, bias=False
        )
        self.strip_v = nn.Conv2d(
            channels, channels, kernel_size=(7, 1), padding=(3, 0), groups=channels, bias=False
        )
        self.proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(min(8, channels), channels),
            nn.GELU(),
        )
        self.ctx_scale = nn.Parameter(torch.tensor(0.15, dtype=torch.float32))

    def forward(self, feat: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        # Prior-weighted global change prototype & anisotropic strip context
        weighted = feat * (0.5 + prior)
        strip_context = self.strip_h(weighted) + self.strip_v(weighted)
        proto_num = (feat * prior).sum(dim=(-2, -1), keepdim=True)
        proto_den = prior.sum(dim=(-2, -1), keepdim=True).clamp(min=1e-4)
        proto = proto_num / proto_den
        # Channel-wise affinity with global change prototype
        sim = torch.sigmoid((feat * proto).mean(dim=1, keepdim=True))
        ctx = self.proj(strip_context + feat * sim)
        return feat + self.ctx_scale.to(dtype=feat.dtype) * ctx


class DecoderOutput(dict):
    """Output container supporting both dictionary key access and tuple unpacking.

    Unpacking order:
        logits, prob = decoder(...)
    """

    def __iter__(self):
        yield self['logits']
        yield self['prob']


class ProgressiveDecoder(nn.Module):
    """Parallax-Suppressed Multi-Scale Progressive Decoder.

    Args:
        in_channels: Input channels from P-DAG fused representation (default 256).
        c3_channels: Channels of backbone stride 8 skip feature (default 128).
        c2_channels: Channels of backbone stride 4 skip feature (default 64).
        c1_channels: Channels of backbone stride 2 skip feature (default 32).
        out_channels: Final change logits channels (default 1).
    """

    def __init__(
        self,
        in_channels: int = 256,
        c3_channels: int = 128,
        c2_channels: int = 64,
        out_channels: Optional[int] = None,
        num_classes: int = 1,
        c1_channels: int = 32,
    ):
        super().__init__()
        if out_channels is None:
            out_channels = num_classes
        self.in_channels = in_channels
        self.c3_channels = c3_channels
        self.c2_channels = c2_channels
        self.c1_channels = c1_channels
        self.out_channels = out_channels
        self.num_classes = out_channels

        # Coarse change prior prediction head at stride 16 (deep supervision + spatial guidance)
        self.aux_head = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, out_channels, kernel_size=1),
        )
        self.guide_gamma = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        self.cgm_s16 = PanoramicChangeGuide(in_channels)
        self.cgm_s8 = PanoramicChangeGuide(128)

        # Bi-temporal parallax-aware skip fusion modules across scales
        self.skip_align_c3 = BiTemporalParallaxSkipFusion(c3_channels)
        self.skip_align_c2 = BiTemporalParallaxSkipFusion(c2_channels)
        self.skip_align_c1 = BiTemporalParallaxSkipFusion(c1_channels)

        # Stage 1: stride 16 -> stride 8
        # in_channels (256) + c3_channels (128) = 384 -> 128
        self.stage1_fuse = DecoderBlock(in_channels + c3_channels, 128)

        # Stage 2: stride 8 -> stride 4
        # 128 + c2_channels (64) = 192 -> 64
        self.stage2_fuse = DecoderBlock(128 + c2_channels, 64)

        # Stage 3: stride 4 -> stride 2
        # 64 + c1_channels (32) = 96 -> 32
        self.stage3_block = DecoderBlock(64 + c1_channels, 32)

        # Stage 4: stride 2 -> stride 1
        # 32 -> 16
        self.stage4_block = DecoderBlock(32, 16)

        # Final prediction head
        self.head = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, 16),
            nn.GELU(),
            nn.Conv2d(16, out_channels, kernel_size=1),
        )

    def _apply_prior_guidance(
        self,
        feat: torch.Tensor,
        aux_prob_s16: torch.Tensor,
        g_occ: Optional[torch.Tensor] = None,
        cgm: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        target_hw = feat.shape[-2:]
        prior = F.interpolate(aux_prob_s16, size=target_hw, mode='bilinear', align_corners=False)
        gamma = self.guide_gamma.to(dtype=feat.dtype)
        guided = feat * (1.0 + gamma * prior)
        if cgm is not None:
            guided = cgm(guided, prior)
        return guided

    def forward(
        self,
        fused_feat: torch.Tensor,
        skip_c3: Optional[torch.Tensor] = None,
        skip_c2: Optional[torch.Tensor] = None,
        target_size: Optional[Tuple[int, int]] = None,
        pad: Optional[Tuple[int, int, int, int]] = None,
        skip_c3_2: Optional[torch.Tensor] = None,
        skip_c2_2: Optional[torch.Tensor] = None,
        skip_c1: Optional[torch.Tensor] = None,
        skip_c1_2: Optional[torch.Tensor] = None,
        warp_grid_12: Optional[torch.Tensor] = None,
        g_occ: Optional[torch.Tensor] = None,
    ) -> DecoderOutput:
        """Forward pass for progressive multi-scale decoding.

        Args:
            fused_feat: Primary deep fused feature at stride 16 [B, in_channels, H_p, W_p]
            skip_c3: Backbone feature at stride 8 (view 1) [B, c3_channels, H_3, W_3]
            skip_c2: Backbone feature at stride 4 (view 1) [B, c2_channels, H_2, W_2]
            target_size: Target original image resolution (H, W)
            pad: Optional padding tuple (pad_left, pad_right, pad_top, pad_bottom)
            skip_c3_2: Optional backbone feature at stride 8 (view 2)
            skip_c2_2: Optional backbone feature at stride 4 (view 2)
            skip_c1: Optional backbone feature at stride 2 (view 1)
            skip_c1_2: Optional backbone feature at stride 2 (view 2)
            warp_grid_12: Optional dense forward warp grid [B, 2, H_p, W_p]
            g_occ: Optional disocclusion gate [B, 1, H_p, W_p]

        Returns:
            DecoderOutput dict containing:
                'logits': Raw change logits [B, 1, H, W]
                'prob': Sigmoid change probability [B, 1, H, W] in [0, 1]
                'features': Decoded high-resolution feature [B, 16, H, W]
                'aux_logits': Coarse deep change logits [B, 1, H, W]
        """
        B, C, H_p, W_p = fused_feat.shape

        # Step 0: Deep coarse change prior at stride 16
        aux_logits_s16 = self.aux_head(fused_feat)
        aux_prob_s16 = torch.sigmoid(aux_logits_s16)
        fused_guided = self._apply_prior_guidance(fused_feat, aux_prob_s16, g_occ, cgm=self.cgm_s16)

        # Step 1: Upsample stride 16 -> stride 8
        if skip_c3 is not None:
            size_8 = skip_c3.shape[-2:]
        else:
            size_8 = (H_p * 2, W_p * 2)

        up1 = F.interpolate(fused_guided, size=size_8, mode='bilinear', align_corners=False)
        if skip_c3 is None:
            skip_c3 = torch.zeros((B, self.c3_channels, size_8[0], size_8[1]), dtype=fused_feat.dtype, device=fused_feat.device)
        s3_fused = self.skip_align_c3(skip_c3, skip_c3_2, warp_grid_12=warp_grid_12, g_occ=g_occ)
        f1 = self.stage1_fuse(torch.cat([up1, s3_fused], dim=1))
        f1 = self._apply_prior_guidance(f1, aux_prob_s16, g_occ, cgm=self.cgm_s8)

        # Step 2: Upsample stride 8 -> stride 4
        if skip_c2 is not None:
            size_4 = skip_c2.shape[-2:]
        else:
            size_4 = (size_8[0] * 2, size_8[1] * 2)

        up2 = F.interpolate(f1, size=size_4, mode='bilinear', align_corners=False)
        if skip_c2 is None:
            skip_c2 = torch.zeros((B, self.c2_channels, size_4[0], size_4[1]), dtype=fused_feat.dtype, device=fused_feat.device)
        s2_fused = self.skip_align_c2(skip_c2, skip_c2_2, warp_grid_12=warp_grid_12, g_occ=g_occ)
        f2 = self.stage2_fuse(torch.cat([up2, s2_fused], dim=1))
        f2 = self._apply_prior_guidance(f2, aux_prob_s16, g_occ)

        # Step 3: Upsample stride 4 -> stride 2
        if skip_c1 is not None:
            size_2 = skip_c1.shape[-2:]
        else:
            size_2 = (size_4[0] * 2, size_4[1] * 2)
        up3 = F.interpolate(f2, size=size_2, mode='bilinear', align_corners=False)
        if skip_c1 is None:
            skip_c1 = torch.zeros((B, self.c1_channels, size_2[0], size_2[1]), dtype=fused_feat.dtype, device=fused_feat.device)
        s1_fused = self.skip_align_c1(skip_c1, skip_c1_2, warp_grid_12=warp_grid_12, g_occ=g_occ)
        f3 = self.stage3_block(torch.cat([up3, s1_fused], dim=1))
        f3 = self._apply_prior_guidance(f3, aux_prob_s16, g_occ)

        # Step 4: Upsample stride 2 -> stride 1 (padded full resolution)
        size_1 = (size_2[0] * 2, size_2[1] * 2)
        up4 = F.interpolate(f3, size=size_1, mode='bilinear', align_corners=False)
        f4 = self.stage4_block(up4)  # [B, 16, H_pad, W_pad]
        aux_full = F.interpolate(aux_logits_s16, size=size_1, mode='bilinear', align_corners=False)

        # Step 5: Invert reflective padding to restore exact unpadded dimensions
        if pad is not None:
            pad_left, pad_right, pad_top, pad_bottom = pad
            H_pad, W_pad = f4.shape[-2:]
            end_h = H_pad - pad_bottom if pad_bottom > 0 else H_pad
            end_w = W_pad - pad_right if pad_right > 0 else W_pad
            f4 = f4[:, :, pad_top:end_h, pad_left:end_w]
            aux_full = aux_full[:, :, pad_top:end_h, pad_left:end_w]

        # Ensure exact match with target_size if specified
        if target_size is not None and f4.shape[-2:] != target_size:
            f4 = F.interpolate(f4, size=target_size, mode='bilinear', align_corners=False)
            aux_full = F.interpolate(aux_full, size=target_size, mode='bilinear', align_corners=False)

        # Step 6: Final change logits (residual refinement with deep change prior)
        logits = self.head(f4) + 0.15 * aux_full  # [B, 1, H, W]
        prob = torch.sigmoid(logits)

        return DecoderOutput({
            'logits': logits,
            'prob': prob,
            'features': f4,
            'aux_logits': aux_full,
        })


# Alias
ChangeDecoder = ProgressiveDecoder
