"""Dual-Stream Background-Anchored Illumination Alignment (D-BAIA).

Decomposes arbitrary rectangular images into low-frequency ambient illumination
and high-frequency albedo via differentiable edge-preserving guided/bilateral filtering.
Estimates spatial affine gain alpha(x, y) and bias beta(x, y) strictly anchored on
static background regions to neutralize diurnal/shadow variations without canceling
genuine foreground scene changes.
"""

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class FastGuidedFilter(nn.Module):
    """Differentiable Fast Guided Filter in pure PyTorch for arbitrary rectangular inputs.
    
    Acts as an edge-preserving low-pass decomposition filter in O(N) linear time.
    """

    def __init__(self, radius: int = 7, eps: float = 1e-3):
        super().__init__()
        self.radius = radius
        self.eps = eps

    def _box_filter(self, x: torch.Tensor, r: int) -> torch.Tensor:
        """Applies 2D box filter via average pooling with reflective padding."""
        k_size = 2 * r + 1
        x_pad = F.pad(x, (r, r, r, r), mode='reflect')
        return F.avg_pool2d(x_pad, kernel_size=k_size, stride=1, padding=0)

    def forward(self, p: torch.Tensor, guidance: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Filters input p guided by guidance (or p itself if guidance is None).
        
        Args:
            p: Input image tensor [B, C, H, W]
            guidance: Guidance tensor [B, 1, H, W] (default: grayscale intensity of p)
            
        Returns:
            Filtered smooth low-frequency tensor [B, C, H, W]
        """
        if guidance is None:
            # Grayscale luminance: 0.299 R + 0.587 G + 0.114 B
            guidance = 0.299 * p[:, 0:1] + 0.587 * p[:, 1:2] + 0.114 * p[:, 2:3]

        r = self.radius
        mean_I = self._box_filter(guidance, r)
        mean_p = self._box_filter(p, r)
        mean_Ip = self._box_filter(guidance * p, r)
        cov_Ip = mean_Ip - mean_I * mean_p

        mean_II = self._box_filter(guidance * guidance, r)
        var_I = mean_II - mean_I * mean_I

        a = cov_Ip / (var_I + self.eps)
        b = mean_p - a * mean_I

        mean_a = self._box_filter(a, r)
        mean_b = self._box_filter(b, r)

        q = mean_a * guidance + mean_b
        return q


class DBAIAOutput(dict):
    """Custom dictionary output that also supports tuple unpacking."""

    def __iter__(self):
        yield self['img1_norm']
        yield self['img2_norm']
        yield self['light_field1']
        yield self['light_field2']


class DBAIA(nn.Module):
    """Dual-Stream Background-Anchored Illumination Alignment (D-BAIA).
    
    Decomposes input images into low-frequency ambient light fields and high-frequency
    reflectance (albedo). Spatial affine normalization fields (alpha, beta) are estimated
    over background regions to eliminate illumination drift between multi-temporal views.

    Args:
        in_channels: Number of color channels (default 3).
        filter_radius: Radius of the edge-preserving filter (default 7).
        filter_eps: Regularization epsilon for the filter (default 1e-3).
        sim_threshold: Cosine similarity threshold for background anchor mask (default 0.5).
        temp: Temperature parameter for soft sigmoid masking (default 0.1).
    """

    def __init__(
        self,
        in_channels: int = 3,
        filter_radius: int = 7,
        filter_eps: float = 1e-3,
        sim_threshold: float = 0.5,
        temp: float = 0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.filter = FastGuidedFilter(radius=filter_radius, eps=filter_eps)
        self.sim_threshold = sim_threshold
        self.temp = temp

        # Predictor for spatial affine gain and bias fields
        # Input: Light1 (3) + Light2 (3) + Albedo1 (3) + Albedo2 (3) + Background_Mask (1) = 13 channels
        self.affine_head = nn.Sequential(
            nn.Conv2d(in_channels * 4 + 1, 32, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, 32),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(4, 32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, in_channels * 2, kernel_size=3, padding=1),
        )

        # Zero-initialize the final conv layer for identity warm-start (alpha=1.0, beta=0.0)
        nn.init.zeros_(self.affine_head[-1].weight)
        nn.init.zeros_(self.affine_head[-1].bias)

        # Learnable residual blend weight to preserve sharp raw RGB textures early in training
        self.blend_logit = nn.Parameter(torch.tensor(-1.0, dtype=torch.float32))

    def decompose(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decomposes image into low-frequency light field and high-frequency albedo.
        
        Args:
            img: Image tensor [B, 3, H, W] in [0, 1]
            
        Returns:
            light: Low-frequency ambient illumination [B, 3, H, W]
            albedo: High-frequency surface reflectance [B, 3, H, W]
        """
        # Ensure non-negative bounds
        img_clamped = torch.clamp(img, min=0.0, max=1.0)
        light = self.filter(img_clamped)
        light = torch.clamp(light, min=1e-4, max=1.0)
        albedo = img_clamped / light
        return light, albedo

    def compute_background_mask(
        self,
        albedo1: torch.Tensor,
        albedo2: torch.Tensor,
    ) -> torch.Tensor:
        """Computes soft static background mask M_bg in [0, 1] using albedo cosine similarity."""
        a1_norm = F.normalize(albedo1, p=2, dim=1, eps=1e-6)
        a2_norm = F.normalize(albedo2, p=2, dim=1, eps=1e-6)
        cos_sim = (a1_norm * a2_norm).sum(dim=1, keepdim=True)  # [B, 1, H, W]
        bg_mask = torch.sigmoid((cos_sim - self.sim_threshold) / self.temp)
        return bg_mask

    def forward(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
    ) -> DBAIAOutput:
        """Forward pass for dual-stream illumination alignment.

        Args:
            img1: Image at time 1 [B, 3, H, W]
            img2: Image at time 2 [B, 3, H, W]

        Returns:
            DBAIAOutput dict containing:
                'img1_norm': Illumination-aligned image 1 [B, 3, H, W]
                'img2_norm': Normalized image 2 [B, 3, H, W]
                'light_field1': Single-channel light field 1 [B, 1, H, W]
                'light_field2': Single-channel light field 2 [B, 1, H, W]
        """
        B, C, H, W = img1.shape

        # Step 1: Retinex decomposition into light and albedo
        light1, albedo1 = self.decompose(img1)
        light2, albedo2 = self.decompose(img2)

        # Step 2: Background similarity mask
        bg_mask = self.compute_background_mask(albedo1, albedo2)

        # Step 3: Spatial affine parameter estimation
        feat_in = torch.cat([light1, light2, albedo1, albedo2, bg_mask], dim=1)
        affine_params = self.affine_head(feat_in)  # [B, 2*C, H/4, W/4]

        # Upsample affine parameters to full resolution
        if affine_params.shape[-2:] != (H, W):
            affine_params = F.interpolate(affine_params, size=(H, W), mode='bilinear', align_corners=False)

        delta_alpha = affine_params[:, :C, :, :]
        delta_beta = affine_params[:, C:, :, :]

        # Identity warm-start with bounded gain/bias to prevent dark-scene overfitting
        alpha = 1.0 + 0.35 * torch.tanh(delta_alpha)
        beta = 0.20 * torch.tanh(delta_beta)

        # Modulate light1 exclusively anchored on static background
        alpha_eff = 1.0 + bg_mask * (alpha - 1.0)
        beta_eff = bg_mask * beta

        light1_aligned = torch.clamp(alpha_eff * light1 + beta_eff, min=1e-4, max=1.0)
        img1_retinex = torch.clamp(light1_aligned * albedo1, min=0.0, max=1.0)
        img1_clamped = torch.clamp(img1, min=0.0, max=1.0)
        w_blend = torch.sigmoid(self.blend_logit).to(dtype=img1.dtype)
        img1_norm = torch.clamp((1.0 - w_blend) * img1_clamped + w_blend * img1_retinex, min=0.0, max=1.0)
        img2_norm = torch.clamp(img2, min=0.0, max=1.0)

        # Single-channel ambient scalar light fields
        light_field1 = light1.mean(dim=1, keepdim=True)
        light_field2 = light2.mean(dim=1, keepdim=True)

        return DBAIAOutput({
            'img1_norm': img1_norm,
            'img2_norm': img2_norm,
            'light_field1': light_field1,
            'light_field2': light_field2,
            'bg_mask': bg_mask,
            'alpha': alpha_eff,
            'beta': beta_eff,
        })


# Alias for backward compatibility
RetinexAlign = DBAIA
