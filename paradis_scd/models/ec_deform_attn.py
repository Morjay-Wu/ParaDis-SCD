"""Epipolar-Constrained Deformable Cross-Attention (EC-DCA).

Implements linear O(NK) epipolar cross-attention with K depth hypotheses:
p_k(x_1) = H_gnd * x_1 + d_k * e_12(x_1) + Delta p_k

Enforces the geometric uncertainty bound:
|Delta p^perp| <= epsilon * exp(-lambda * C_F)
where C_F is the epipolar line certainty metric.

Generates high-fidelity warped feature F_{2->1}, forward dense coordinate grid W_{1->2},
and backward dense coordinate grid W_{2->1}. Supports arbitrary rectangular resolutions
and full FP16/FP32 AMP numerical stability.
"""

from typing import Dict, List, Optional, Tuple, Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ECDCAOutput(dict):
    """Output container supporting both dictionary key access and tuple unpacking.

    Unpacking order matches Interface Contract 5:
        feat2_warped, warp_grid_12 = ec_dca(feat1, feat2, F_12, depth1, scale_shift)
    """

    def __iter__(self):
        yield self['feat2_warped']
        yield self['warp_grid_12']


class ECDCA(nn.Module):
    """Epipolar-Constrained Deformable Cross-Attention (EC-DCA).

    Args:
        in_channels: Input feature channels (default 256).
        num_hypotheses: Number of depth hypotheses K along the epipolar line (default 16).
        num_heads: Number of attention heads (default 4).
        max_offset: Maximum perpendicular offset epsilon in normalized [-1, 1] coordinates (default 0.05).
        lambda_geom: Uncertainty decay factor lambda (default 2.0).
        depth_offset_range: Tuple of (min_offset, max_offset) for hypothesis offsets (default (-0.35, 0.35)).
    """

    def __init__(
        self,
        in_channels: int = 256,
        num_hypotheses: int = 16,
        num_heads: int = 4,
        max_offset: float = 0.05,
        lambda_geom: float = 2.0,
        depth_offset_range: Tuple[float, float] = (-0.35, 0.35),
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_hypotheses = num_hypotheses
        self.num_heads = num_heads
        self.max_offset = max_offset
        self.lambda_geom = lambda_geom
        self.head_dim = in_channels // num_heads

        # Projections for Query, Key, and Value
        self.q_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.k_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.v_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.out_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(8, in_channels)

        # Deformable offset generator subnet
        # Input: feat1 (C) + feat2 (C) + depth (1) = 2C + 1
        self.offset_net = nn.Sequential(
            nn.Conv2d(in_channels * 2 + 1, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, 64),
            nn.GELU(),
            nn.Conv2d(64, num_hypotheses * 2, kernel_size=1),
        )

        # Initialize offset generator to zero for identity warm-start
        nn.init.zeros_(self.offset_net[-1].weight)
        nn.init.zeros_(self.offset_net[-1].bias)

        # Learnable depth hypothesis offsets along the epipolar ray
        init_offsets = torch.linspace(depth_offset_range[0], depth_offset_range[1], num_hypotheses)
        self.depth_offsets = nn.Parameter(init_offsets)

        # Scale factor for disparity prior
        self.disparity_scale = nn.Parameter(torch.tensor(0.08, dtype=torch.float32))

        # Learnable attention residual weight so warped feature preserves target semantics
        self.attn_gamma = nn.Parameter(torch.tensor(0.25, dtype=torch.float32))

    @staticmethod
    def _create_coordinate_grid(
        batch_size: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Creates normalized 2D coordinate grid in [-1, 1] with shape [B, 3, H, W] in homogeneous coordinates."""
        y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')  # [H, W]
        ones = torch.ones_like(grid_x)
        homo_grid = torch.stack([grid_x, grid_y, ones], dim=0)  # [3, H, W]
        return homo_grid.unsqueeze(0).repeat(batch_size, 1, 1, 1)  # [B, 3, H, W]

    def _sample_epipolar_warp(
        self,
        feat_src: torch.Tensor,
        feat_tgt: torch.Tensor,
        F_mat: torch.Tensor,
        depth_src: torch.Tensor,
        H_gnd: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Samples target feature using epipolar-constrained deformable cross-attention.

        Args:
            feat_src: Source query feature map [B, C, H, W]
            feat_tgt: Target key/value feature map [B, C, H, W]
            F_mat: Fundamental matrix from src to tgt [B, 3, 3] (x_tgt^T * F * x_src = 0)
            depth_src: Calibrated metric/relative depth of source [B, 1, H, W]
            H_gnd: Ground plane homography [B, 3, 3] in SL(3)

        Returns:
            feat_warped: Aligned feature map [B, C, H, W]
            warp_grid: Expected 2D coordinate grid in [-1, 1] [B, 2, H, W]
            attn_weights: Hypothesis attention distribution [B, K, H, W]
        """
        B, C, H, W = feat_src.shape
        K = self.num_hypotheses
        orig_dtype = feat_src.dtype

        # 1. Homogeneous source coordinates [B, 3, H, W] in FP32 for numerical stability
        homo_grid_f = self._create_coordinate_grid(B, H, W, feat_src.device, torch.float32)
        homo_flat_f = homo_grid_f.reshape(B, 3, H * W)

        # 2. Base ground homography mapping p_base = H_gnd * x_src in FP32 with bounded tangent damping
        x_src = homo_grid_f[:, 0:2, :, :]
        if H_gnd is not None:
            # Flatten spatial dimensions for matrix multiplication: [B, 3, H*W]
            p_tgt_h = torch.bmm(H_gnd.float(), homo_flat_f).reshape(B, 3, H, W)
            z_safe = p_tgt_h[:, 2:3, :, :]
            z_sign = torch.sign(z_safe)
            z_sign = torch.where(z_sign == 0, torch.ones_like(z_sign), z_sign)
            z_denom = torch.where(z_safe.abs() < 1e-4, z_sign * 1e-4, z_safe)
            p_homo_x = p_tgt_h[:, 0:1, :, :] / z_denom
            p_homo_y = p_tgt_h[:, 1:2, :, :] / z_denom
            p_homo = torch.cat([p_homo_x, p_homo_y], dim=1)
            eta_max = 0.20
            p_base = (x_src + eta_max * torch.tanh((p_homo - x_src) / eta_max)).to(dtype=orig_dtype)  # [B, 2, H, W]
        else:
            p_base = x_src.to(dtype=orig_dtype)  # [B, 2, H, W]

        # 3. Epipolar line in target image: l_tgt = F_mat @ x_src = [a, b, c]^T in FP32
        l_tgt = torch.bmm(F_mat.float(), homo_flat_f).reshape(B, 3, H, W)
        a = l_tgt[:, 0:1, :, :]  # [B, 1, H, W]
        b = l_tgt[:, 1:2, :, :]  # [B, 1, H, W]

        # Epipolar line direction vector e_dir = [-b, a]^T / sqrt(a^2 + b^2 + eps)
        # Perpendicular normal vector n_perp = [a, b]^T / sqrt(a^2 + b^2 + eps)
        # Clamping norm square min to 1e-6 prevents derivative NaN near epipole in FP16/AMP
        grad_sq = (a * a + b * b).clamp(min=1e-6)
        grad_norm = torch.sqrt(grad_sq)
        e_dir_x = -b / grad_norm
        e_dir_y = a / grad_norm
        e_dir = torch.cat([e_dir_x, e_dir_y], dim=1).to(dtype=orig_dtype)  # [B, 2, H, W]

        n_perp_x = a / grad_norm
        n_perp_y = b / grad_norm
        n_perp = torch.cat([n_perp_x, n_perp_y], dim=1).to(dtype=orig_dtype)  # [B, 2, H, W]

        # Epipolar line certainty metric C_F
        C_F = grad_norm.to(dtype=orig_dtype)  # [B, 1, H, W]

        # 4. Monocular disparity center d_center = 1.0 / Z_src
        d_center = 1.0 / torch.clamp(depth_src, min=0.1, max=20.0)  # [B, 1, H, W]
        scaled_disp = self.disparity_scale * d_center  # [B, 1, H, W]

        # 5. Predict deformable offsets from feature pair and depth
        feat_pair = torch.cat([feat_src, feat_tgt, depth_src], dim=1)  # [B, 2C+1, H, W]
        raw_offsets = self.offset_net(feat_pair)  # [B, 2K, H, W]
        raw_offsets = raw_offsets.reshape(B, K, 2, H, W)  # [B, K, 2, H, W]

        # Geometric uncertainty bound: |Delta p^perp| <= epsilon * exp(-lambda * C_F)
        # High epipolar certainty -> narrow corridor along epipolar ray
        bound_perp = self.max_offset * torch.exp(-self.lambda_geom * C_F)  # [B, 1, H, W]

        # 6. Construct K hypothesis sample coordinates
        # Expand e_dir, n_perp, p_base to [B, K, 2, H, W]
        p_base_k = p_base.unsqueeze(1).repeat(1, K, 1, 1, 1)  # [B, K, 2, H, W]
        e_dir_k = e_dir.unsqueeze(1).repeat(1, K, 1, 1, 1)    # [B, K, 2, H, W]
        n_perp_k = n_perp.unsqueeze(1).repeat(1, K, 1, 1, 1)  # [B, K, 2, H, W]

        # Offsets along and perpendicular to epipolar line
        offset_par = raw_offsets[:, :, 0:1, :, :]  # [B, K, 1, H, W]
        offset_perp_raw = raw_offsets[:, :, 1:2, :, :]  # [B, K, 1, H, W]
        offset_perp = torch.tanh(offset_perp_raw) * bound_perp.unsqueeze(1)  # [B, K, 1, H, W]

        delta_p = offset_par * e_dir_k + offset_perp * n_perp_k  # [B, K, 2, H, W]

        # Epipolar disparity displacement for hypothesis k
        # depth_offsets: [K] -> [1, K, 1, 1, 1]
        k_disp = self.depth_offsets.reshape(1, K, 1, 1, 1) + scaled_disp.unsqueeze(1)  # [B, K, 1, H, W]
        hypo_disp = k_disp * e_dir_k  # [B, K, 2, H, W]

        # Final 2D candidate coordinates p_k in [-1, 1]
        sample_coords = torch.clamp(p_base_k + hypo_disp + delta_p, min=-1.0, max=1.0)  # [B, K, 2, H, W]

        # 7. Linear O(NK) bilinear sampling of Key and Value features
        # Permute coordinates to [B*K, H, W, 2] for grid_sample
        coords_for_grid = sample_coords.permute(0, 1, 3, 4, 2).reshape(B * K, H, W, 2)

        # Linear projections
        Q = self.q_proj(feat_src)  # [B, C, H, W]
        K_feat = self.k_proj(feat_tgt)  # [B, C, H, W]
        V_feat = self.v_proj(feat_tgt)  # [B, C, H, W]

        # Expand Key and Value: [B*K, C, H, W]
        K_expanded = K_feat.repeat_interleave(K, dim=0)
        V_expanded = V_feat.repeat_interleave(K, dim=0)

        # Sample at all K hypotheses simultaneously in parallel
        K_sampled = F.grid_sample(
            K_expanded,
            coords_for_grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=False,
        ).reshape(B, K, C, H, W)  # [B, K, C, H, W]

        V_sampled = F.grid_sample(
            V_expanded,
            coords_for_grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=False,
        ).reshape(B, K, C, H, W)  # [B, K, C, H, W]

        # 8. Attention computation
        # Q: [B, 1, C, H, W]
        Q_exp = Q.unsqueeze(1)
        scale = 1.0 / math.sqrt(C)
        # Dot product across channel dimension: [B, K, 1, H, W]
        attn_logits = (Q_exp * K_sampled).sum(dim=2, keepdim=True) * scale

        # Optional soft Gaussian geometric prior centered near current hypothesis
        attn_weights = F.softmax(attn_logits, dim=1)  # [B, K, 1, H, W]

        # 9. Aggregate warped feature and expected 2D forward coordinate grid
        # Expected sampling grid: sum_k A_k * p_k -> [B, 2, H, W]
        # sample_coords: [B, K, 2, H, W], attn_weights: [B, K, 1, H, W]
        warp_grid = (attn_weights * sample_coords).sum(dim=1)  # [B, 2, H, W]

        # Sample target feature directly at the expected epipolar warp_grid as geometric base
        grid_for_tgt = warp_grid.permute(0, 2, 3, 1)  # [B, H, W, 2]
        feat_tgt_aligned = F.grid_sample(
            feat_tgt,
            grid_for_tgt,
            mode='bilinear',
            padding_mode='border',
            align_corners=False,
        )

        # Alignment-gain gate: blend warped feature only where epipolar alignment improves agreement
        with torch.no_grad():
            sim_unaligned = F.cosine_similarity(feat_src.float(), feat_tgt.float(), dim=1, eps=1e-6).unsqueeze(1)
            sim_aligned = F.cosine_similarity(feat_src.float(), feat_tgt_aligned.float(), dim=1, eps=1e-6).unsqueeze(1)
            align_gain = torch.sigmoid((sim_aligned - sim_unaligned) * 5.0).to(dtype=orig_dtype)  # [B, 1, H, W]
        feat_tgt_base = (1.0 - align_gain) * feat_tgt + align_gain * feat_tgt_aligned

        # Feature: sum_k A_k * V_k -> [B, C, H, W]
        feat_warped_agg = (attn_weights * V_sampled).sum(dim=1)  # [B, C, H, W]
        feat_warped = self.out_proj(feat_warped_agg)
        gamma = self.attn_gamma.to(dtype=orig_dtype)
        feat_warped = self.norm(feat_tgt_base + gamma * align_gain * feat_warped)

        return feat_warped, warp_grid, attn_weights.squeeze(2)

    def forward(
        self,
        feat1: torch.Tensor,
        feat2: torch.Tensor,
        F_12: torch.Tensor,
        depth1: torch.Tensor,
        depth2: Optional[torch.Tensor] = None,
        scale_shift: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        H_gnd: Optional[torch.Tensor] = None,
        bidirectional: bool = True,
    ) -> ECDCAOutput:
        """Forward pass for epipolar-constrained deformable cross-attention.

        Args:
            feat1: Source feature map at time 1 [B, C, H_p, W_p]
            feat2: Target feature map at time 2 [B, C, H_p, W_p]
            F_12: Fundamental matrix from view 1 to view 2 [B, 3, 3] (x2^T * F_12 * x1 = 0)
            depth1: Relative depth at time 1 [B, 1, H_p, W_p]
            depth2: Relative depth at time 2 [B, 1, H_p, W_p] (optional, used for backward warp)
            scale_shift: Optional tuple (alpha, beta) for depth calibration
            H_gnd: Optional ground homography [B, 3, 3] in SL(3)
            bidirectional: Whether to also compute backward warp grid W_{2->1} (default True)

        Returns:
            ECDCAOutput dict containing:
                'feat2_warped': Warped feature aligned to view 1 [B, C, H_p, W_p]
                'warp_grid_12': Dense forward sampling grid W_{1->2} [B, 2, H_p, W_p]
                'warp_grid_21': Dense backward sampling grid W_{2->1} [B, 2, H_p, W_p]
                'attn_weights': Forward attention weights [B, K, H_p, W_p]
        """
        B, C, H, W = feat1.shape

        # Calibrate depth1 if scale_shift is provided
        if scale_shift is not None:
            alpha, beta = scale_shift
            if not isinstance(alpha, torch.Tensor):
                alpha = torch.tensor(alpha, dtype=feat1.dtype, device=feat1.device)
            if not isinstance(beta, torch.Tensor):
                beta = torch.tensor(beta, dtype=feat1.dtype, device=feat1.device)
            alpha_val = alpha.reshape(alpha.numel(), 1, 1, 1)
            beta_val = beta.reshape(beta.numel(), 1, 1, 1)
            z1_calib = alpha_val * depth1 + beta_val
        else:
            z1_calib = depth1

        # 1. Forward warp: Align feat2 to feat1
        feat2_warped, warp_grid_12, attn_12 = self._sample_epipolar_warp(
            feat_src=feat1,
            feat_tgt=feat2,
            F_mat=F_12,
            depth_src=z1_calib,
            H_gnd=H_gnd,
        )

        # 2. Backward warp: Compute W_{2->1} for cycle error
        if bidirectional:
            F_21 = F_12.transpose(-1, -2)  # x1^T * F_21 * x2 = 0
            if H_gnd is not None:
                # Approximate inverse ground homography via batch matrix inverse in FP32
                H_inv = torch.linalg.inv(H_gnd.float()).to(dtype=feat1.dtype)
            else:
                H_inv = None

            z2_calib = depth2 if depth2 is not None else depth1

            _, warp_grid_21, _ = self._sample_epipolar_warp(
                feat_src=feat2,
                feat_tgt=feat1,
                F_mat=F_21,
                depth_src=z2_calib,
                H_gnd=H_inv,
            )
        else:
            warp_grid_21 = warp_grid_12

        return ECDCAOutput({
            'feat2_warped': feat2_warped,
            'warp_grid_12': warp_grid_12,
            'warp_grid_21': warp_grid_21,
            'attn_weights': attn_12,
        })


# Alias
EpipolarDeformableCrossAttention = ECDCA
