"""Perspective Differential Disocclusion & Parallax Gate (P-DAG).

Decouples genuine semantic changes from 3D parallax displacements and perspective disocclusions.
Integrates:
1. Bidirectional cycle error: E_cycle(p_1) = ||p_1 - W_{2->1}(W_{1->2}(p_1))||_2
2. 3D surface normal rotational consistency: Delta_N = 1.0 - <n_1, R_12^T n_2(W_{1->2})>
3. Projective depth discrepancy: Delta_Z = (alpha * zeta_1 + beta) - zeta_2(W_{1->2})
4. Continuous disocclusion gate G_occ in [0, 1]
5. Decoupled pure difference: D_pure = |F_1 - F_{2->1}| * (1 - G_occ) * Sigmoid(W_n * Delta_N + b_n)
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class PDAGOutput(dict):
    """Output container supporting both dictionary key access and tuple unpacking.

    Unpacking order matches Interface Contract 6:
        fused_feat, g_occ, delta_normal, cycle_error = p_dag(...)
    """

    def __iter__(self):
        yield self['fused_feat']
        yield self['g_occ']
        yield self['delta_normal']
        yield self['cycle_error']


class PDAG(nn.Module):
    """Perspective Differential Disocclusion & Parallax Gate (P-DAG).

    Args:
        in_channels: Input feature channels C (default 256).
        out_channels: Output fused representation channels (default 256).
        tau_z: Initial depth threshold for occlusion gating (default 0.25).
        gamma_z: Temperature scale for depth gating (default 0.15).
        tau_c: Initial cycle error threshold for disocclusion gating (default 0.15).
        gamma_c: Temperature scale for cycle error gating (default 0.08).
    """

    def __init__(
        self,
        in_channels: int = 256,
        out_channels: int = 256,
        tau_z: float = 0.25,
        gamma_z: float = 0.15,
        tau_c: float = 0.15,
        gamma_c: float = 0.08,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Learnable gating thresholds and temperature parameters
        self.tau_z = nn.Parameter(torch.tensor(tau_z, dtype=torch.float32))
        self.gamma_z = nn.Parameter(torch.tensor(gamma_z, dtype=torch.float32))
        self.tau_c = nn.Parameter(torch.tensor(tau_c, dtype=torch.float32))
        self.gamma_c = nn.Parameter(torch.tensor(gamma_c, dtype=torch.float32))

        # Surface normal angular sensitivity modulator: Sigmoid(W_n * Delta_N + b_n)
        self.normal_gate_proj = nn.Conv2d(1, 1, kernel_size=1)
        with torch.no_grad():
            self.normal_gate_proj.weight.fill_(2.0)
            self.normal_gate_proj.bias.fill_(0.0)

        # Multi-channel fusion subnet
        # Input channels: F1 (C) + F_{2->1} (C) + D_raw (C) + D_pure (C) + G_occ (1) + Delta_N (1) + E_cycle (1) = 4C + 3
        total_in_channels = in_channels * 4 + 3
        self.fuse_proj = nn.Sequential(
            nn.Conv2d(total_in_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
        )

        self.res_block = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
        )
        self.final_act = nn.GELU()

    @staticmethod
    def _create_base_grid(
        batch_size: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Creates base normalized coordinate grid [B, 2, H, W] in [-1, 1]."""
        y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        base_grid = torch.stack([grid_x, grid_y], dim=0)  # [2, H, W]
        return base_grid.unsqueeze(0).repeat(batch_size, 1, 1, 1)  # [B, 2, H, W]

    def forward(
        self,
        feat1: torch.Tensor,
        feat2_warped: torch.Tensor,
        depth1: torch.Tensor,
        depth2: torch.Tensor,
        normal1: torch.Tensor,
        normal2: torch.Tensor,
        R_12: torch.Tensor,
        warp_grid_12: torch.Tensor,
        warp_grid_21: Optional[torch.Tensor] = None,
        scale_shift: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        feat2_raw: Optional[torch.Tensor] = None,
    ) -> PDAGOutput:
        """Forward pass for perspective differential disocclusion & parallax gating.

        Args:
            feat1: Reference feature map [B, C, H_p, W_p]
            feat2_warped: Epipolar-aligned warped feature [B, C, H_p, W_p]
            depth1: Scalar relative depth at view 1 [B, 1, H_p, W_p]
            depth2: Scalar relative depth at view 2 [B, 1, H_p, W_p]
            normal1: Unit surface normal field at view 1 [B, 3, H_p, W_p] in S^2
            normal2: Unit surface normal field at view 2 [B, 3, H_p, W_p] in S^2
            R_12: Relative rotation matrix in SO(3) [B, 3, 3]
            warp_grid_12: Dense forward sampling coordinates [B, 2, H_p, W_p] in [-1, 1]
            warp_grid_21: Dense backward sampling coordinates [B, 2, H_p, W_p] in [-1, 1] (optional)
            scale_shift: Optional tuple (alpha, beta) for depth calibration

        Returns:
            PDAGOutput dict containing:
                'fused_feat': Refined multi-channel change representation [B, out_channels, H_p, W_p]
                'g_occ': Continuous disocclusion gate [B, 1, H_p, W_p] in [0, 1]
                'delta_normal': Surface normal rotational discrepancy [B, 1, H_p, W_p]
                'cycle_error': Bidirectional loop closure error [B, 1, H_p, W_p]
                'd_pure': Decoupled pure semantic difference [B, C, H_p, W_p]
                'd_raw': Raw difference magnitude |feat1 - feat2_warped| [B, C, H_p, W_p]
        """
        B, C, H, W = feat1.shape
        orig_dtype = feat1.dtype

        # 1. Base 2D coordinate grid in [-1, 1]
        base_grid = self._create_base_grid(B, H, W, feat1.device, orig_dtype)  # [B, 2, H, W]

        # 2. Bidirectional Cycle Error: E_cycle = ||p_1 - W_{2->1}(W_{1->2}(p_1))||_2
        # grid_12_sample: [B, H, W, 2] for grid_sample
        grid_12_sample = warp_grid_12.permute(0, 2, 3, 1)

        if warp_grid_21 is not None:
            # Re-project coordinate p_1 through W_{2->1}(W_{1->2}(p_1))
            p1_reproj = F.grid_sample(
                warp_grid_21,
                grid_12_sample,
                mode='bilinear',
                padding_mode='border',
                align_corners=False,
            )  # [B, 2, H, W]
            cycle_diff = base_grid.float() - p1_reproj.float()  # [B, 2, H, W] in FP32
            # Safe Euclidean norm avoiding derivative singularity at 0
            cycle_error_f = torch.sqrt((cycle_diff * cycle_diff).sum(dim=1, keepdim=True).clamp(min=1e-6))
            cycle_error = cycle_error_f.to(dtype=orig_dtype)  # [B, 1, H, W]
        else:
            # Fallback when backward grid is absent
            cycle_error = torch.zeros((B, 1, H, W), dtype=orig_dtype, device=feat1.device)
            cycle_error_f = cycle_error.float()

        # 3. 3D Surface Normal Rotational Invariance: Delta_N = 1.0 - <n_1, R_12^T * n_2(W_{1->2})>
        normal2_warped = F.grid_sample(
            normal2,
            grid_12_sample,
            mode='bilinear',
            padding_mode='border',
            align_corners=False,
        )  # [B, 3, H, W]
        normal2_warped_f = F.normalize(normal2_warped.float(), p=2, dim=1, eps=1e-5)

        # Rotate normal2_warped by R_12^T in FP32
        R_12_T = R_12.transpose(-1, -2).float()  # [B, 3, 3]
        # Matrix-vector multiplication for each spatial pixel
        n2_rot_f = torch.einsum('bij, bjhw -> bihw', R_12_T, normal2_warped_f)
        n2_rot_f = F.normalize(n2_rot_f, p=2, dim=1, eps=1e-5)
        n2_rot = n2_rot_f.to(dtype=orig_dtype)

        # Dot product with normal1: <n_1, R^T n_2>
        cos_normal_f = (normal1.float() * n2_rot_f).sum(dim=1, keepdim=True)  # [B, 1, H, W]
        # Rotational discrepancy Delta_N in [0, 2]
        delta_normal_f = torch.clamp(1.0 - cos_normal_f, min=0.0, max=2.0)
        delta_normal = delta_normal_f.to(dtype=orig_dtype)  # [B, 1, H, W]

        # 4. Projective Depth Difference: Delta_Z = (alpha * zeta_1 + beta) - zeta_2(W_{1->2})
        if scale_shift is not None:
            alpha, beta = scale_shift
            if not isinstance(alpha, torch.Tensor):
                alpha = torch.tensor(alpha, dtype=feat1.dtype, device=feat1.device)
            if not isinstance(beta, torch.Tensor):
                beta = torch.tensor(beta, dtype=feat1.dtype, device=feat1.device)
            alpha_val = alpha.reshape(alpha.numel(), 1, 1, 1)
            beta_val = beta.reshape(beta.numel(), 1, 1, 1)
            depth1_calib = alpha_val * depth1 + beta_val
        else:
            depth1_calib = depth1

        depth2_warped = F.grid_sample(
            depth2,
            grid_12_sample,
            mode='bilinear',
            padding_mode='border',
            align_corners=False,
        )  # [B, 1, H, W]

        delta_z_f = depth1_calib.float() - depth2_warped.float()  # [B, 1, H, W]
        delta_z = delta_z_f.to(dtype=orig_dtype)

        # 5. Continuous Disocclusion Gate G_occ in [0, 1]
        # Safe positive scale temperatures
        gamma_z_safe = self.gamma_z.abs().clamp(min=1e-4).float()
        gamma_c_safe = self.gamma_c.abs().clamp(min=1e-4).float()

        # Raw aligned difference
        d_raw = torch.abs(feat1 - feat2_warped)  # [B, C, H, W]
        if feat2_raw is not None:
            d_unaligned = torch.abs(feat1 - feat2_raw)
            # Parallax relief: positive when epipolar alignment reduces feature difference
            parallax_relief = (d_unaligned.float().mean(dim=1, keepdim=True) - d_raw.float().mean(dim=1, keepdim=True))
            parallax_weight = torch.sigmoid(parallax_relief * 4.0)
            eff_cycle_f = cycle_error_f * (0.35 + 0.65 * parallax_weight)
            d_persist = 0.5 * d_raw + 0.5 * torch.minimum(d_unaligned, d_raw)
        else:
            eff_cycle_f = cycle_error_f
            d_persist = d_raw

        gate_depth = torch.sigmoid((delta_z_f - self.tau_z.float()) / gamma_z_safe)
        gate_cycle = torch.sigmoid((eff_cycle_f - self.tau_c.float()) / gamma_c_safe)

        # Soft union: G_occ = 1 - (1 - G_depth) * (1 - G_cycle)
        g_occ_f = torch.clamp(1.0 - (1.0 - gate_depth) * (1.0 - gate_cycle), min=0.0, max=1.0)
        g_occ = g_occ_f.to(dtype=orig_dtype)

        # 6. Decoupled Pure Difference D_pure
        # Normal consistency gate: higher normal change increases change probability
        normal_modulator = torch.sigmoid(self.normal_gate_proj(delta_normal))  # [B, 1, H, W]

        # G_occ close to 1 (occlusion) suppresses difference; normal discrepancy enhances it
        d_pure = d_raw * (1.0 - g_occ) * normal_modulator  # [B, C, H, W]

        # 7. Multi-Channel Fusion Representation
        # Concat: [feat1, feat2_warped, d_persist, d_pure, g_occ, delta_normal, cycle_error]
        x_concat = torch.cat([
            feat1,
            feat2_warped,
            d_persist,
            d_pure,
            g_occ,
            delta_normal,
            cycle_error,
        ], dim=1)  # [B, 4C+3, H, W]

        fused_init = self.fuse_proj(x_concat)  # [B, out_channels, H, W]
        fused_feat = self.final_act(fused_init + self.res_block(fused_init))  # [B, out_channels, H, W]

        return PDAGOutput({
            'fused_feat': fused_feat,
            'g_occ': g_occ,
            'delta_normal': delta_normal,
            'cycle_error': cycle_error,
            'd_pure': d_pure,
            'd_raw': d_raw,
            'delta_z': delta_z,
        })


# Alias
DisocclusionParallaxGate = PDAG
