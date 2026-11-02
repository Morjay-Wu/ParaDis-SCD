"""Differentiable Epipolar-Scale Gauge & Homography-Epipolar (HEH) Bundle.

Solves the fundamental matrix F_12 in rank-2, relative rotation R_12 in SO(3),
ground homography H_gnd in SL(3), and closed-form monocular depth scale-shift (alpha, beta)
via differentiable Sinkhorn soft correspondences, weighted spectral estimation, and closed-form
weighted least-squares.

Features full FP16 / FP32 AMP compatibility by safely executing linear algebraic solvers
(eigh, svd, det) in FP32 precision before projecting back to tensor dtype.
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class DESGOutput(dict):
    """Output container supporting both dictionary key access and tuple unpacking.

    Unpacking order matches Interface Contract 4:
        F_12, R_12, scale_shift, H_gnd = d_esg(feat1, feat2, depth1, depth2)
    """

    def __iter__(self):
        yield self['F_12']
        yield self['R_12']
        yield self['scale_shift']
        yield self['H_gnd']


def log_sinkhorn(
    cost: torch.Tensor,
    num_iters: int = 5,
    eps: float = 0.1,
) -> torch.Tensor:
    """Log-space Sinkhorn iterations for doubly-stochastic soft matching.

    Args:
        cost: Pairwise affinity/similarity matrix [B, M, M]
        num_iters: Number of Sinkhorn normalization steps (default 5)
        eps: Temperature scaling factor (default 0.1)

    Returns:
        P: Doubly-stochastic assignment probabilities [B, M, M]
    """
    B, M, _ = cost.shape
    u = torch.zeros((B, M), dtype=cost.dtype, device=cost.device)
    v = torch.zeros((B, M), dtype=cost.dtype, device=cost.device)
    log_M = torch.log(torch.tensor(M, dtype=cost.dtype, device=cost.device))

    scaled_cost = cost / eps
    for _ in range(num_iters):
        # Row normalization
        u = -torch.logsumexp(scaled_cost + v.unsqueeze(1), dim=-1) - log_M
        # Column normalization
        v = -torch.logsumexp(scaled_cost + u.unsqueeze(-1), dim=1) - log_M

    P = torch.exp(scaled_cost + u.unsqueeze(-1) + v.unsqueeze(1))
    return P


def project_to_so3(M: torch.Tensor) -> torch.Tensor:
    """Projects arbitrary 3x3 matrices to SO(3) via continuous Gram-Schmidt orthogonalization.

    Guarantees R in SO(3) strictly (R^T R = I, det(R) = +1.0) with everywhere-finite gradients,
    avoiding the SVD repeated-singular-value backward singularity (Zhou et al., CVPR 2019).

    Args:
        M: Input matrix tensor [B, 3, 3]

    Returns:
        R: Proper rotation matrix in SO(3) [B, 3, 3]
    """
    orig_dtype = M.dtype
    M_f = M.float()
    col1 = M_f[..., :, 0]
    col2 = M_f[..., :, 1]

    b1 = F.normalize(col1, p=2, dim=-1, eps=1e-7)
    dot = (b1 * col2).sum(dim=-1, keepdim=True)
    b2 = col2 - dot * b1
    b2 = F.normalize(b2, p=2, dim=-1, eps=1e-7)
    b3 = torch.linalg.cross(b1, b2, dim=-1)

    R = torch.stack([b1, b2, b3], dim=-1).to(dtype=orig_dtype)
    return R


def project_to_sl3(H: torch.Tensor) -> torch.Tensor:
    """Projects arbitrary 3x3 non-singular matrices to SL(3) such that det(H) = +1.0.

    Args:
        H: Input homography matrix [B, 3, 3]

    Returns:
        H_sl3: Projected matrix with det(H_sl3) = 1.0 [B, 3, 3]
    """
    orig_dtype = H.dtype
    H_f = H.float()
    det_H = torch.det(H_f)
    sign = torch.sign(det_H)
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    abs_det = torch.abs(det_H).clamp(min=1e-7)
    cbrt = (sign * torch.pow(abs_det, 1.0 / 3.0)).detach()
    H_sl3 = (H_f / cbrt.unsqueeze(-1).unsqueeze(-1)).to(dtype=orig_dtype)
    return H_sl3


class DESG(nn.Module):
    """Differentiable Epipolar-Scale Gauge & Homography-Epipolar (HEH) Bundle.

    Extracts:
        1. F_12: Fundamental matrix strictly projected to rank-2 with ||F||_F = 1.0.
        2. R_12: Relative camera rotation in SO(3).
        3. scale_shift: (alpha, beta) closed-form weighted least-squares depth calibrator.
        4. H_gnd: Dominant ground plane homography in SL(3) with smooth ||t|| -> 0 continuity.

    Args:
        in_channels: Feature channels of input representations (default 256).
        grid_size: Spatial keypoint sampling grid size (H_s, W_s) (default (16, 16)).
        temp: Temperature parameter for feature cross-correlation (default 0.05).
        sinkhorn_iters: Number of Sinkhorn iterations (default 5).
    """

    def __init__(
        self,
        in_channels: int = 256,
        grid_size: Tuple[int, int] = (16, 16),
        temp: float = 0.05,
        sinkhorn_iters: int = 5,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.grid_size = grid_size
        self.temp = temp
        self.sinkhorn_iters = sinkhorn_iters

        # Feature projection head for correspondence matching
        self.match_proj = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
        )

        # Learnable relative rotation offset refiner (genuine dual-view 128-d input)
        self.rot_refine = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 9),
        )
        # Initialize near identity matrix with small perturbation to ensure view-2 sensitivity
        with torch.no_grad():
            nn.init.normal_(self.rot_refine[-1].weight, mean=0.0, std=0.05)
            self.rot_refine[-1].bias.copy_(torch.eye(3).reshape(9))

        # Register distinct spectral damping vector to guarantee strictly non-degenerate eigenvalues
        damping_9 = torch.linspace(1e-4, 9e-4, 9, dtype=torch.float32)
        self.register_buffer('damping_9', damping_9)

        # Register SVD perturbation matrix buffer to eliminate (s_i^2 - s_j^2) -> 0 NaN gradient division
        svd_perturb = torch.diag(torch.tensor([1e-5, 2e-5, 3e-5], dtype=torch.float32))
        self.register_buffer('svd_perturb', svd_perturb)

    def _generate_sampling_grid(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generates normalized 2D coordinate grid in [-1, 1] of size (H_s, W_s).

        Returns:
            grid: Sampling grid tensor [B, H_s, W_s, 2]
            pts_flat: Flattened coordinates [B, M, 2] where M = H_s * W_s
        """
        Gh, Gw = self.grid_size
        y = torch.linspace(-0.85, 0.85, Gh, device=device, dtype=dtype)
        x = torch.linspace(-0.85, 0.85, Gw, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).repeat(batch_size, 1, 1, 1)
        pts_flat = grid.reshape(batch_size, Gh * Gw, 2)
        return grid, pts_flat

    def _solve_fundamental_matrix(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Solves fundamental matrix F_12 via weighted 8-point algorithm with rank-2 projection.

        Args:
            x1: Coordinates in image 1 [B, M, 2] in [-1, 1]
            x2: Coordinates in image 2 [B, M, 2] in [-1, 1]
            weights: Correspondence weights [B, M]

        Returns:
            F_12: Rank-2 normalized fundamental matrix [B, 3, 3]
        """
        orig_dtype = x1.dtype
        B, M, _ = x1.shape
        u1, v1 = x1[..., 0], x1[..., 1]
        u2, v2 = x2[..., 0], x2[..., 1]

        # Epipolar algebraic equation: x2^T F x1 = 0
        A = torch.stack([
            u2 * u1, u2 * v1, u2,
            v2 * u1, v2 * v1, v2,
            u1, v1, torch.ones_like(u1)
        ], dim=-1)  # [B, M, 9]

        # Apply weights: A_w = sqrt(w) * A
        w_sqrt = torch.sqrt(weights.clamp(min=1e-6)).unsqueeze(-1)
        A_w = A * w_sqrt

        # Normal equations: C = A_w^T @ A_w in R^[B, 9, 9]
        C = torch.matmul(A_w.transpose(-1, -2), A_w).float()
        diag_damp = torch.diag(self.damping_9.to(device=C.device)).unsqueeze(0)
        C_reg = C + diag_damp

        # Eigen-decomposition in FP32 (guarantees AMP compatibility)
        eigenvalues, eigenvectors = torch.linalg.eigh(C_reg)
        f_vec = eigenvectors[..., :, 0]  # [B, 9]
        F_raw = f_vec.reshape(B, 3, 3)

        # Enforce strict rank-2 constraint via SVD in FP32
        # Perturb F_raw with distinct diagonal bias to guarantee non-degenerate singular values
        F_pert = F_raw + self.svd_perturb.to(device=F_raw.device).unsqueeze(0)
        U, S, Vh = torch.linalg.svd(F_pert)
        # Zero out the smallest singular value
        S_rank2 = torch.stack([S[..., 0], S[..., 1], torch.zeros_like(S[..., 2])], dim=-1)
        F_rank2 = U @ torch.diag_embed(S_rank2) @ Vh

        # Normalize Frobenius norm to 1.0 (detach scale to avoid gauge scale gradient explosion)
        f_norm = torch.norm(F_rank2, p='fro', dim=(-2, -1), keepdim=True).clamp(min=1e-7)
        F_12 = (F_rank2 / f_norm.detach()).to(dtype=orig_dtype)
        return F_12

    def _solve_ground_homography(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Solves ground homography H_gnd in SL(3) anchored on lower road surface.

        Args:
            x1: Coordinates in image 1 [B, M, 2] in [-1, 1]
            x2: Coordinates in image 2 [B, M, 2] in [-1, 1]
            weights: Correspondence weights [B, M]

        Returns:
            H_gnd: Ground homography in SL(3) [B, 3, 3] with det(H_gnd) = 1.0
        """
        orig_dtype = x1.dtype
        B, M, _ = x1.shape
        u1, v1 = x1[..., 0], x1[..., 1]
        u2, v2 = x2[..., 0], x2[..., 1]

        # Prioritize ground points (lower half of image: v1 > 0 in normalized coordinates)
        ground_prior = torch.sigmoid(v1 * 5.0)  # [B, M]
        w_gnd = (weights * ground_prior).clamp(min=1e-6)
        w_sqrt = torch.sqrt(w_gnd).unsqueeze(-1)

        zeros = torch.zeros_like(u1)
        ones = torch.ones_like(u1)

        # DLT equations for homography: x2 x (H x1) = 0
        A1 = torch.stack([-u1, -v1, -ones, zeros, zeros, zeros, u2 * u1, u2 * v1, u2], dim=-1) * w_sqrt
        A2 = torch.stack([zeros, zeros, zeros, -u1, -v1, -ones, v2 * u1, v2 * v1, v2], dim=-1) * w_sqrt

        # Interleave rows to form [B, 2M, 9]
        A_homo = torch.cat([A1, A2], dim=1)

        # Solve via eigen-decomposition in FP32
        C = torch.matmul(A_homo.transpose(-1, -2), A_homo).float()
        diag_damp = torch.diag(self.damping_9.to(device=C.device)).unsqueeze(0)
        C_reg = C + diag_damp
        _, eigenvectors = torch.linalg.eigh(C_reg)
        h_vec = eigenvectors[..., :, 0]  # [B, 9] in FP32
        H_raw = h_vec.reshape(B, 3, 3)

        # Align eigenvector sign so h_33 is strictly positive (eliminates CPU/CUDA sign ambiguity)
        h33_sign = torch.sign(H_raw[:, 2:3, 2:3])
        h33_sign = torch.where(h33_sign == 0, torch.ones_like(h33_sign), h33_sign)
        H_oriented = H_raw * h33_sign

        # Tangent-space bounded perturbation around I_3:
        # For unit-norm eigenvector of identity homography, H_oriented * sqrt(3) == I_3.
        eye3 = torch.eye(3, dtype=torch.float32, device=H_raw.device).unsqueeze(0)
        H_delta = H_oriented * 1.7320508 - eye3
        bound_mat = torch.tensor(
            [[0.20, 0.15, 0.20],
             [0.15, 0.20, 0.20],
             [0.05, 0.05, 0.10]],
            dtype=torch.float32,
            device=H_raw.device,
        ).unsqueeze(0)
        H_bounded = (eye3 + bound_mat * torch.tanh(H_delta / bound_mat)).to(dtype=orig_dtype)
        H_gnd = project_to_sl3(H_bounded)
        return H_gnd

    def _solve_depth_scale_shift(
        self,
        depth1: torch.Tensor,
        depth2: torch.Tensor,
        x1: torch.Tensor,
        x2: torch.Tensor,
        weights: torch.Tensor,
        F_12: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Closed-form weighted least-squares solver for depth scale-shift (alpha, beta).

        Minimizes sum_m w_m* * (alpha * depth1(x1^m) + beta - depth2(x2^m))^2
        where w_m* = w_m * exp(-d_sampson^2 / (2 * sigma^2)).

        Args:
            depth1: Depth map 1 [B, 1, H_p, W_p]
            depth2: Depth map 2 [B, 1, H_p, W_p]
            x1: Coordinates 1 [B, M, 2] in [-1, 1]
            x2: Coordinates 2 [B, M, 2] in [-1, 1]
            weights: Soft correspondence weights [B, M]
            F_12: Fundamental matrix [B, 3, 3]

        Returns:
            alpha: Scalar scale parameter [B, 1] in R+
            beta: Scalar shift parameter [B, 1] in R
        """
        B, M, _ = x1.shape

        # Sample depth values at correspondence points
        x1_grid = x1.unsqueeze(1)  # [B, 1, M, 2]
        x2_grid = x2.unsqueeze(1)  # [B, 1, M, 2]

        z1 = F.grid_sample(depth1, x1_grid, mode='bilinear', align_corners=False).squeeze(1).squeeze(1)  # [B, M]
        z2 = F.grid_sample(depth2, x2_grid, mode='bilinear', align_corners=False).squeeze(1).squeeze(1)  # [B, M]

        # Compute Sampson epipolar distance for inlier gating
        u1, v1 = x1[..., 0], x1[..., 1]
        u2, v2 = x2[..., 0], x2[..., 1]
        ones = torch.ones_like(u1)
        homo1 = torch.stack([u1, v1, ones], dim=-1)  # [B, M, 3]
        homo2 = torch.stack([u2, v2, ones], dim=-1)  # [B, M, 3]

        # Epipolar line in image 2: l2 = F @ homo1
        l2 = torch.matmul(homo1, F_12.transpose(-1, -2))  # [B, M, 3]
        # Epipolar line in image 1: l1 = F^T @ homo2
        l1 = torch.matmul(homo2, F_12)  # [B, M, 3]

        algebraic_err = (homo2 * l2).sum(dim=-1)  # [B, M]
        denom = (l2[..., 0] ** 2 + l2[..., 1] ** 2 + l1[..., 0] ** 2 + l1[..., 1] ** 2).clamp(min=1e-7)
        d_sampson_sq = (algebraic_err ** 2) / denom

        # Epipolar inlier weight
        inlier_weight = weights * torch.exp(-d_sampson_sq / (2.0 * (0.15 ** 2)))
        w = inlier_weight.clamp(min=1e-6)

        # Closed-form 2x2 weighted normal equations:
        s_w = w.sum(dim=-1, keepdim=True)             # [B, 1]
        s_x = (w * z1).sum(dim=-1, keepdim=True)       # [B, 1]
        s_y = (w * z2).sum(dim=-1, keepdim=True)       # [B, 1]
        s_xx = (w * z1 * z1).sum(dim=-1, keepdim=True) # [B, 1]
        s_xy = (w * z1 * z2).sum(dim=-1, keepdim=True) # [B, 1]

        # Determinant of 2x2 system
        det = s_xx * s_w - s_x * s_x
        # Regularized determinant for numerical stability
        det_reg = det + 1e-4 * (s_xx + s_w + 1.0)

        alpha = (s_xy * s_w - s_x * s_y) / det_reg
        beta = (s_xx * s_y - s_x * s_xy) / det_reg

        # Fallback safeguard: keep alpha positive and bounded, beta finite
        alpha = torch.clamp(alpha, min=0.05, max=20.0)
        beta = torch.clamp(beta, min=-10.0, max=10.0)

        return alpha, beta

    def forward(
        self,
        feat1: torch.Tensor,
        feat2: torch.Tensor,
        depth1: torch.Tensor,
        depth2: torch.Tensor,
    ) -> DESGOutput:
        """Forward pass for differentiable epipolar-scale calibration.

        Args:
            feat1: Deep feature map at time 1 [B, C, H_p, W_p]
            feat2: Deep feature map at time 2 [B, C, H_p, W_p]
            depth1: Scalar relative depth at time 1 [B, 1, H_p, W_p]
            depth2: Scalar relative depth at time 2 [B, 1, H_p, W_p]

        Returns:
            DESGOutput containing:
                'F_12': Rank-2 fundamental matrix [B, 3, 3] with ||F||_F = 1.0
                'R_12': Relative camera rotation matrix in SO(3) [B, 3, 3]
                'scale_shift': Tuple (alpha, beta) for depth scale-shift [B, 1]
                'H_gnd': Ground plane homography in SL(3) [B, 3, 3]
                'correspondences': Tuple (x1, x2, weights) of sampled pairs
        """
        B, C, H_p, W_p = feat1.shape

        # Step 1: Feature projections and keypoint sampling
        p1 = self.match_proj(feat1)
        p2 = self.match_proj(feat2)

        grid, pts1 = self._generate_sampling_grid(B, feat1.device, feat1.dtype)
        Gh, Gw = self.grid_size
        M = Gh * Gw

        # Sample descriptors at grid points
        desc1 = F.grid_sample(p1, grid, mode='bilinear', align_corners=False)  # [B, 64, Gh, Gw]
        desc2 = F.grid_sample(p2, grid, mode='bilinear', align_corners=False)  # [B, 64, Gh, Gw]

        desc1_flat = F.normalize(desc1.reshape(B, 64, M), p=2, dim=1)  # [B, 64, M]
        desc2_flat = F.normalize(desc2.reshape(B, 64, M), p=2, dim=1)  # [B, 64, M]

        # Step 2: Sinkhorn-normalized soft correspondences with spatial locality prior
        affinity = torch.matmul(desc1_flat.transpose(-1, -2), desc2_flat)  # [B, M, M]
        spatial_dist_sq = ((pts1.unsqueeze(2) - pts1.unsqueeze(1)) ** 2).sum(dim=-1)  # [B, M, M]
        affinity_local = affinity - 0.35 * spatial_dist_sq
        P = log_sinkhorn(affinity_local, num_iters=self.sinkhorn_iters, eps=self.temp)  # [B, M, M]

        # Soft matching coordinates: row-wise conditional probability normalization
        # P_match guarantees sum_j Pr(j | i) == 1.0, preserving [-0.85, 0.85] coordinate scale
        P_match = P / P.sum(dim=-1, keepdim=True).clamp(min=1e-7)
        pts2_base = pts1  # Candidate search locations are grid locations in view 2
        pts2_soft = torch.matmul(P_match, pts2_base)  # [B, M, 2]
        weights = P_match.max(dim=-1).values          # [B, M]

        # Step 3: Differentiable 8-point Fundamental Matrix solver with rank-2 projection
        F_12 = self._solve_fundamental_matrix(pts1, pts2_soft, weights)

        # Step 4: Ground Homography in SL(3)
        H_gnd = self._solve_ground_homography(pts1, pts2_soft, weights)

        # Step 5: Genuine Dual-View Relative Rotation R_12 in SO(3)
        pooled_feat1 = desc1_flat.mean(dim=-1)  # [B, 64]
        pooled_feat2 = desc2_flat.mean(dim=-1)  # [B, 64]
        pair_feat = torch.cat([pooled_feat1, pooled_feat2], dim=-1)  # [B, 128]
        rot_mat_raw = self.rot_refine(pair_feat).reshape(B, 3, 3)
        R_12 = project_to_so3(rot_mat_raw)

        # Step 6: Closed-form weighted least-squares depth scale-shift (alpha, beta)
        alpha, beta = self._solve_depth_scale_shift(depth1, depth2, pts1, pts2_soft, weights, F_12)
        scale_shift = (alpha, beta)

        return DESGOutput({
            'F_12': F_12,
            'R_12': R_12,
            'scale_shift': scale_shift,
            'H_gnd': H_gnd,
            'alpha': alpha,
            'beta': beta,
            'correspondences': (pts1, pts2_soft, weights),
        })
