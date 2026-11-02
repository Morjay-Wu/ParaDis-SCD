"""ParaDis-SCD Multi-Task Physics-Informed Loss Formulation.

CVPR Main-Conference Loss Suite:
1. FocalLoss: Asymmetric binary focal loss focusing on hard change boundaries.
2. SoftDiceLoss: Differentiable soft Dice loss addressing extreme change/static class imbalance.
3. SampsonEpipolarLoss: First-order geometric Sampson epipolar distance on predicted fundamental matrix F_12.
4. CycleConsistencyLoss: Bidirectional warp loop-closure consistency on static background regions.
5. SurfaceNormalRotLoss: SO(3) rotational normal invariance loss on static scenes (1 - <N_1, R_12^T N_2_warp>).
6. OrthogonalNullSpaceLoss: Null-space orthogonality penalty ||D_pure * G_occ||^2 enforcing mutual exclusivity.
7. ParaDisLoss: Top-level unified loss module combining task losses with physics constraints.
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Binary Focal Loss for Change Detection.

    FL(p_t) = - alpha_t * (1 - p_t)^gamma * log(p_t)
    where p_t = p if y=1 else (1 - p), and alpha_t = alpha if y=1 else (1 - alpha).

    Args:
        alpha: Positive class weight factor in [0, 1] (default 0.25).
        gamma: Focusing parameter for modulating hard examples (default 2.0).
        reduction: Reduction method ('mean', 'sum', 'none').
        eps: Small epsilon for numerical stability.
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean',
        eps: float = 1e-7,
    ):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.reduction = reduction
        self.eps = float(eps)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Forward computation for binary focal loss.

        Args:
            logits: Predicted raw change logits [B, 1, H, W] or [B, H, W]
            targets: Binary ground truth change masks [B, 1, H, W] or [B, H, W] in {0, 1}

        Returns:
            Scalar loss tensor if reduction is 'mean' or 'sum', else elementwise tensor.
        """
        if logits.dim() == 3 and targets.dim() == 4:
            logits = logits.unsqueeze(1)
        elif logits.dim() == 4 and targets.dim() == 3:
            targets = targets.unsqueeze(1)

        targets = targets.to(dtype=logits.dtype)

        # Numerically stable BCE with logits
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        # Probability of positive class
        p = torch.sigmoid(logits)
        p_t = targets * p + (1.0 - targets) * (1.0 - p)
        p_t = torch.clamp(p_t, min=self.eps, max=1.0 - self.eps)

        # Alpha weighting
        alpha_t = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)

        # Focal modulating factor
        focal_factor = alpha_t * torch.pow(1.0 - p_t, self.gamma)
        loss = focal_factor * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class SoftDiceLoss(nn.Module):
    r"""Differentiable Soft Dice Loss for Extreme Class Imbalance.

    Dice = (2 * |P \cap Y| + smooth) / (|P| + |Y| + smooth)
    Loss = 1.0 - Dice

    Args:
        smooth: Additive smoothing factor to prevent division by zero (default 1.0).
        eps: Small epsilon for numerical stability (default 1e-7).
        reduction: Reduction across batch ('mean', 'sum', 'none').
    """

    def __init__(
        self,
        smooth: float = 1.0,
        eps: float = 1e-7,
        reduction: str = 'mean',
    ):
        super().__init__()
        self.smooth = float(smooth)
        self.eps = float(eps)
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Forward computation for Soft Dice loss.

        Args:
            logits: Predicted raw change logits [B, 1, H, W] or [B, H, W]
            targets: Binary ground truth change masks [B, 1, H, W] or [B, H, W] in {0, 1}

        Returns:
            Scalar loss tensor if reduction is 'mean' or 'sum'.
        """
        if logits.dim() == 3:
            logits = logits.unsqueeze(1)
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)

        targets = targets.to(dtype=logits.dtype)
        probs = torch.sigmoid(logits)

        # Flatten spatial dimensions per batch sample: [B, H*W]
        B = logits.shape[0]
        probs_flat = probs.view(B, -1)
        targets_flat = targets.view(B, -1)

        intersection = torch.sum(probs_flat * targets_flat, dim=-1)
        cardinality = torch.sum(probs_flat, dim=-1) + torch.sum(targets_flat, dim=-1)

        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth + self.eps)
        loss = 1.0 - dice

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class SampsonEpipolarLoss(nn.Module):
    """Multi-Point Sampson Epipolar Distance Loss.

    Computes the first-order geometric Sampson error:
        d_S(x_1, x_2; F) = (x_2^T F x_1)^2 / ((F x_1)_1^2 + (F x_1)_2^2 + (F^T x_2)_1^2 + (F^T x_2)_2^2 + eps)
    for corresponding point pairs (x_1, x_2) in static background regions.

    Args:
        eps: Small epsilon for denominator stability (default 1e-7).
        max_error_clamp: Upper bound clamp on squared residual to prevent gradient spikes (default 50.0).
        num_samples: Number of spatial samples when sampling from dense grids (default 128).
    """

    def __init__(
        self,
        eps: float = 1e-7,
        max_error_clamp: float = 50.0,
        num_samples: int = 128,
    ):
        super().__init__()
        self.eps = float(eps)
        self.max_error_clamp = float(max_error_clamp)
        self.num_samples = int(num_samples)

    def forward(
        self,
        F_12: torch.Tensor,
        pts1: Optional[torch.Tensor] = None,
        pts2: Optional[torch.Tensor] = None,
        warp_grid_12: Optional[torch.Tensor] = None,
        static_mask: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute Sampson epipolar loss.

        Args:
            F_12: Predicted fundamental matrix [B, 3, 3]
            pts1: Optional 2D/3D points in image 1 [B, N, 2] or [B, N, 3] in [-1, 1]
            pts2: Optional 2D/3D points in image 2 [B, N, 2] or [B, N, 3] in [-1, 1]
            warp_grid_12: Optional dense sampling grid [B, 2, H, W] from EC-DCA
            static_mask: Optional binary or continuous static weight map [B, 1, H, W]
            weights: Optional correspondence weights [B, N]

        Returns:
            Scalar Sampson epipolar loss tensor with backpropagation to F_12.
        """
        B = F_12.shape[0]
        device = F_12.device
        dtype = F_12.dtype

        # 1. Derive point correspondences if not provided explicitly
        if pts1 is None or pts2 is None:
            if warp_grid_12 is not None:
                _, _, H_w, W_w = warp_grid_12.shape
                # Sample grid points in [-0.8, 0.8]
                num_s = int(self.num_samples ** 0.5)
                y_s = torch.linspace(-0.8, 0.8, num_s, device=device, dtype=dtype)
                x_s = torch.linspace(-0.8, 0.8, num_s, device=device, dtype=dtype)
                grid_y, grid_x = torch.meshgrid(y_s, x_s, indexing='ij')
                # pts1 in [-1, 1]: [B, N, 2]
                sample_pts1 = torch.stack([grid_x, grid_y], dim=-1).reshape(1, -1, 2).repeat(B, 1, 1)
                pts1 = sample_pts1  # [B, N, 2]

                # Sample pts2 from warp_grid_12
                # warp_grid_12 is [B, 2, H_w, W_w] -> sample with grid_sample
                sample_grid = sample_pts1.unsqueeze(2)  # [B, N, 1, 2]
                pts2_sample = F.grid_sample(
                    warp_grid_12,
                    sample_grid,
                    mode='bilinear',
                    padding_mode='border',
                    align_corners=False,
                )  # [B, 2, N, 1]
                pts2 = pts2_sample.squeeze(-1).permute(0, 2, 1)  # [B, N, 2]

                if static_mask is not None and weights is None:
                    # Sample static mask weights
                    static_sample = F.grid_sample(
                        static_mask.to(dtype=dtype),
                        sample_grid,
                        mode='bilinear',
                        padding_mode='zeros',
                        align_corners=False,
                    )  # [B, 1, N, 1]
                    weights = static_sample.squeeze(1).squeeze(-1)  # [B, N]
            else:
                # Fallback: regular coordinate identity grid
                num_s = int(self.num_samples ** 0.5)
                y_s = torch.linspace(-0.8, 0.8, num_s, device=device, dtype=dtype)
                x_s = torch.linspace(-0.8, 0.8, num_s, device=device, dtype=dtype)
                grid_y, grid_x = torch.meshgrid(y_s, x_s, indexing='ij')
                pts = torch.stack([grid_x, grid_y], dim=-1).reshape(1, -1, 2).repeat(B, 1, 1)
                pts1 = pts
                pts2 = pts

        # Convert to homogeneous coordinates [B, N, 3]
        if pts1.shape[-1] == 2:
            ones = torch.ones((*pts1.shape[:-1], 1), dtype=dtype, device=device)
            p1_homo = torch.cat([pts1, ones], dim=-1)
        else:
            p1_homo = pts1

        if pts2.shape[-1] == 2:
            ones = torch.ones((*pts2.shape[:-1], 1), dtype=dtype, device=device)
            p2_homo = torch.cat([pts2, ones], dim=-1)
        else:
            p2_homo = pts2

        # 2. Sampson Distance Computation
        # Epipolar line in image 2: l2 = F_12 @ x1 -> shape [B, N, 3]
        # p1_homo: [B, N, 3], F_12: [B, 3, 3] -> (F_12 @ p1.T).T = p1 @ F_12^T
        l2 = torch.matmul(p1_homo, F_12.transpose(-1, -2))  # [B, N, 3]

        # Epipolar line in image 1: l1 = F_12^T @ x2 -> shape [B, N, 3]
        l1 = torch.matmul(p2_homo, F_12)  # [B, N, 3]

        # Algebraic residual: x2^T @ F_12 @ x1 = (p2_homo * l2).sum(dim=-1)
        epi_res = torch.sum(p2_homo * l2, dim=-1)  # [B, N]
        epi_res_sq = epi_res * epi_res  # [B, N]

        # Geometric denominator: l2_x^2 + l2_y^2 + l1_x^2 + l1_y^2 + eps
        denom = (
            l2[..., 0] ** 2 + l2[..., 1] ** 2 +
            l1[..., 0] ** 2 + l1[..., 1] ** 2 +
            self.eps
        )  # [B, N]

        sampson_dist = epi_res_sq / denom  # [B, N]
        sampson_dist = torch.clamp(sampson_dist, min=0.0, max=self.max_error_clamp)

        if weights is not None:
            weights = weights.clamp(min=0.0)
            loss = torch.sum(sampson_dist * weights) / (torch.sum(weights) + self.eps)
        else:
            loss = torch.mean(sampson_dist)

        return loss


class CycleConsistencyLoss(nn.Module):
    """Bidirectional Warp Cycle Consistency Loss.

    Penalizes geometric loop-closure tracking errors:
        E_cycle(p) = ||p - W_{2->1}(W_{1->2}(p))||_2
    predominantly on static background regions (1 - Y).

    Args:
        eps: Small epsilon for numerical stability (default 1e-6).
        loss_type: 'l1' or 'l2' (default 'l1').
    """

    def __init__(
        self,
        eps: float = 1e-6,
        loss_type: str = 'l1',
    ):
        super().__init__()
        self.eps = float(eps)
        self.loss_type = loss_type

    def forward(
        self,
        cycle_error: Optional[torch.Tensor] = None,
        warp_grid_12: Optional[torch.Tensor] = None,
        warp_grid_21: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute cycle consistency loss.

        Args:
            cycle_error: Precomputed cycle error map [B, 1, H, W]
            warp_grid_12: Dense forward sampling coordinates [B, 2, H, W] in [-1, 1]
            warp_grid_21: Dense backward sampling coordinates [B, 2, H, W] in [-1, 1]
            mask: Spatial weighting mask [B, 1, H, W], e.g., static mask (1 - Y)

        Returns:
            Scalar cycle consistency loss tensor.
        """
        if cycle_error is None:
            if warp_grid_12 is not None and warp_grid_21 is not None:
                B, _, H, W = warp_grid_12.shape
                # Base normalized coordinate grid
                y = torch.linspace(-1.0, 1.0, H, device=warp_grid_12.device, dtype=warp_grid_12.dtype)
                x = torch.linspace(-1.0, 1.0, W, device=warp_grid_12.device, dtype=warp_grid_12.dtype)
                gy, gx = torch.meshgrid(y, x, indexing='ij')
                base_grid = torch.stack([gx, gy], dim=0).unsqueeze(0).repeat(B, 1, 1, 1)  # [B, 2, H, W]

                grid_12_sample = warp_grid_12.permute(0, 2, 3, 1)
                p1_reproj = F.grid_sample(
                    warp_grid_21,
                    grid_12_sample,
                    mode='bilinear',
                    padding_mode='border',
                    align_corners=False,
                )
                cycle_diff = base_grid - p1_reproj
                cycle_error = torch.sqrt((cycle_diff * cycle_diff).sum(dim=1, keepdim=True) + self.eps)
            else:
                raise ValueError("Either cycle_error or both (warp_grid_12, warp_grid_21) must be provided.")

        if self.loss_type == 'l2':
            err_map = cycle_error ** 2
        else:
            err_map = cycle_error

        if mask is not None:
            mask = mask.to(dtype=err_map.dtype)
            if mask.shape[-2:] != err_map.shape[-2:]:
                mask = F.interpolate(mask, size=err_map.shape[-2:], mode='nearest')
            loss = torch.sum(err_map * mask) / (torch.sum(mask) + self.eps)
        else:
            loss = torch.mean(err_map)

        return loss


class SurfaceNormalRotLoss(nn.Module):
    """SO(3) Rotational Normal Invariance Loss on Static Regions.

    Penalizes surface normal angular discrepancy after 3D camera rotation R_12:
        Delta_N(p) = 1.0 - <N_1(p), R_12^T * N_2(W_{1->2}(p))>
    enforcing that rigid 3D physical surfaces satisfy normal rotational invariance on static scenes.

    Args:
        eps: Small epsilon for numerical stability (default 1e-6).
    """

    def __init__(
        self,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.eps = float(eps)

    def forward(
        self,
        delta_normal: Optional[torch.Tensor] = None,
        normal1: Optional[torch.Tensor] = None,
        normal2: Optional[torch.Tensor] = None,
        R_12: Optional[torch.Tensor] = None,
        warp_grid_12: Optional[torch.Tensor] = None,
        static_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute surface normal rotational invariance loss.

        Args:
            delta_normal: Precomputed normal angular discrepancy map [B, 1, H, W] in [0, 2]
            normal1: Unit surface normal field at view 1 [B, 3, H, W]
            normal2: Unit surface normal field at view 2 [B, 3, H, W]
            R_12: Relative rotation matrix in SO(3) [B, 3, 3]
            warp_grid_12: Dense forward sampling coordinates [B, 2, H, W]
            static_mask: Optional static weight mask [B, 1, H, W] (1 for static, 0 for change)

        Returns:
            Scalar surface normal rotational invariance loss tensor.
        """
        if delta_normal is None:
            if normal1 is not None and normal2 is not None and R_12 is not None and warp_grid_12 is not None:
                orig_dtype = normal1.dtype
                grid_12_sample = warp_grid_12.permute(0, 2, 3, 1)
                normal2_warped = F.grid_sample(
                    normal2,
                    grid_12_sample,
                    mode='bilinear',
                    padding_mode='border',
                    align_corners=False,
                )
                normal2_warped_f = F.normalize(normal2_warped.float(), p=2, dim=1, eps=1e-5)
                R_12_T = R_12.transpose(-1, -2).float()
                n2_rot_f = torch.einsum('bij, bjhw -> bihw', R_12_T, normal2_warped_f)
                n2_rot_f = F.normalize(n2_rot_f, p=2, dim=1, eps=1e-5)
                cos_normal = (normal1.float() * n2_rot_f).sum(dim=1, keepdim=True)
                delta_normal = torch.clamp(1.0 - cos_normal, min=0.0, max=2.0).to(dtype=orig_dtype)
            else:
                raise ValueError("Either delta_normal or all of (normal1, normal2, R_12, warp_grid_12) must be provided.")

        if static_mask is not None:
            static_mask = static_mask.to(dtype=delta_normal.dtype)
            if static_mask.shape[-2:] != delta_normal.shape[-2:]:
                static_mask = F.interpolate(static_mask, size=delta_normal.shape[-2:], mode='nearest')
            loss = torch.sum(delta_normal * static_mask) / (torch.sum(static_mask) + self.eps)
        else:
            loss = torch.mean(delta_normal)

        return loss


class OrthogonalNullSpaceLoss(nn.Module):
    """Orthogonal Null-Space Mutual Exclusivity Loss.

    Enforces that genuine physical semantic changes (D_pure or P_change) and
    viewpoint disocclusions/parallax artifacts (G_occ) occupy mutually orthogonal subspaces:
        L_ortho = || P_change * G_occ ||_2^2
    Guarantees that a pixel cannot simultaneously be classified as genuine scene change and disocclusion.

    Args:
        eps: Small epsilon for numerical stability (default 1e-7).
    """

    def __init__(
        self,
        eps: float = 1e-7,
    ):
        super().__init__()
        self.eps = float(eps)

    def forward(
        self,
        d_pure_or_prob: torch.Tensor,
        g_occ: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute orthogonal null-space loss.

        Args:
            d_pure_or_prob: Either pure change difference [B, C, H, W] or change prob [B, 1, H, W]
            g_occ: Continuous disocclusion gate [B, 1, H, W] in [0, 1]
            mask: Optional region mask [B, 1, H, W]

        Returns:
            Scalar null-space orthogonality loss tensor.
        """
        # Align spatial dimensions if necessary
        if d_pure_or_prob.shape[-2:] != g_occ.shape[-2:]:
            g_occ = F.interpolate(g_occ, size=d_pure_or_prob.shape[-2:], mode='bilinear', align_corners=False)

        overlap = d_pure_or_prob * g_occ
        overlap_sq = overlap * overlap

        if mask is not None:
            mask = mask.to(dtype=overlap_sq.dtype)
            if mask.shape[-2:] != overlap_sq.shape[-2:]:
                mask = F.interpolate(mask, size=overlap_sq.shape[-2:], mode='nearest')
            loss = torch.sum(overlap_sq * mask) / (torch.sum(mask) + self.eps)
        else:
            loss = torch.mean(overlap_sq)

        return loss


class ParaDisLoss(nn.Module):
    """Top-Level Unified Multi-Task Physics-Informed Loss for ParaDis-SCD.

    Combines:
    1. L_focal: Binary Focal Loss for change boundary refinement
    2. L_dice: Soft Dice Loss for extreme class imbalance
    3. L_sampson: Multi-Point Sampson Epipolar Distance on static regions
    4. L_cycle: Bidirectional Warp Cycle Consistency on static regions
    5. L_normal: SO(3) Surface Normal Rotational Invariance
    6. L_ortho: Orthogonal Null-Space Penalty between Change and Disocclusion

    Total Loss:
        L_total = lambda_focal * L_focal + lambda_dice * L_dice
                + lambda_sampson * L_sampson + lambda_cycle * L_cycle
                + lambda_normal * L_normal + lambda_ortho * L_ortho

    Args:
        lambda_focal: Weight for Focal Loss (default 1.0).
        lambda_dice: Weight for Soft Dice Loss (default 1.0).
        lambda_sampson: Weight for Sampson Epipolar Loss (default 0.1).
        lambda_cycle: Weight for Cycle Consistency Loss (default 0.05).
        lambda_normal: Weight for Surface Normal Invariance Loss (default 0.1).
        lambda_ortho: Weight for Orthogonal Null-Space Loss (default 0.05).
    """

    def __init__(
        self,
        lambda_focal: float = 1.0,
        lambda_dice: float = 1.0,
        lambda_sampson: float = 0.02,
        lambda_cycle: float = 0.02,
        lambda_normal: float = 0.02,
        lambda_ortho: float = 0.02,
        focal_alpha: float = 0.65,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.lambda_focal = float(lambda_focal)
        self.lambda_dice = float(lambda_dice)
        self.lambda_sampson = float(lambda_sampson)
        self.lambda_cycle = float(lambda_cycle)
        self.lambda_normal = float(lambda_normal)
        self.lambda_ortho = float(lambda_ortho)

        # Component Loss Modules
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dice_loss = SoftDiceLoss(smooth=1.0)
        self.sampson_loss = SampsonEpipolarLoss(max_error_clamp=50.0)
        self.cycle_loss = CycleConsistencyLoss(loss_type='l1')
        self.normal_loss = SurfaceNormalRotLoss()
        self.ortho_loss = OrthogonalNullSpaceLoss()

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute all loss components and combined total loss.

        Args:
            outputs: Dictionary returned by ParaDisSCDNet containing:
                'logits': Raw change logits [B, 1, H, W]
                'prob': Sigmoid change probability [B, 1, H, W]
                'F_12': Fundamental matrix [B, 3, 3]
                'warp_grid_12': Forward warp grid [B, 2, H_p, W_p]
                'warp_grid_21': Optional backward warp grid [B, 2, H_p, W_p]
                'cycle_error': Bidirectional cycle error map [B, 1, H, W]
                'delta_normal': Surface normal discrepancy map [B, 1, H, W]
                'g_occ': Disocclusion gate [B, 1, H, W]
            targets: Binary ground truth change masks [B, 1, H, W] or [B, H, W] in {0, 1}

        Returns:
            Dict containing:
                'loss': Total backpropagatable scalar loss
                'loss_focal': Focal loss scalar
                'loss_dice': Soft dice loss scalar
                'loss_sampson': Sampson epipolar loss scalar
                'loss_cycle': Cycle consistency loss scalar
                'loss_normal': Surface normal loss scalar
                'loss_ortho': Orthogonal null-space loss scalar
        """
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)
        targets = targets.to(dtype=outputs['logits'].dtype, device=outputs['logits'].device)

        # Static mask: 1 on static background, 0 on genuine changed regions
        static_mask = 1.0 - targets

        # 1. Primary Change Detection Losses
        loss_focal_raw = self.focal_loss(outputs['logits'], targets)
        loss_bce = F.binary_cross_entropy_with_logits(outputs['logits'], targets)
        loss_focal = 0.5 * loss_focal_raw + 0.5 * loss_bce
        loss_dice = self.dice_loss(outputs['logits'], targets)

        # Deep coarse change prior auxiliary supervision
        aux_logits = outputs.get('aux_logits')
        if aux_logits is not None:
            loss_aux = 0.4 * (
                0.5 * self.focal_loss(aux_logits, targets)
                + 0.5 * F.binary_cross_entropy_with_logits(aux_logits, targets)
                + self.dice_loss(aux_logits, targets)
            )
        else:
            loss_aux = outputs['logits'].new_tensor(0.0)

        # 2. Sampson Epipolar Geometry Loss
        # Epipolar constraints hold strictly on static scenes
        loss_sampson = self.sampson_loss(
            F_12=outputs['F_12'],
            warp_grid_12=outputs.get('warp_grid_12'),
            static_mask=static_mask,
        )

        # 3. Cycle Consistency Loss on Static Regions
        cycle_error = outputs.get('cycle_error')
        if cycle_error is not None:
            loss_cycle = self.cycle_loss(cycle_error=cycle_error, mask=static_mask)
        else:
            loss_cycle = outputs['logits'].new_tensor(0.0)

        # 4. Surface Normal Rotational Invariance Loss
        delta_normal = outputs.get('delta_normal')
        if delta_normal is not None:
            loss_normal = self.normal_loss(delta_normal=delta_normal, static_mask=static_mask)
        else:
            loss_normal = outputs['logits'].new_tensor(0.0)

        # 5. Orthogonal Null-Space Loss (strictly masked on static background so true changes are not suppressed)
        # Change probability vs Disocclusion gate
        prob = outputs.get('prob')
        if prob is None:
            prob = torch.sigmoid(outputs['logits'])
        g_occ = outputs.get('g_occ')
        if g_occ is not None:
            loss_ortho = self.ortho_loss(prob, g_occ, mask=static_mask)
        else:
            loss_ortho = outputs['logits'].new_tensor(0.0)

        # 6. Static-region illumination & homography gauge regularization
        img1_norm = outputs.get('img1_norm')
        img2_norm = outputs.get('img2_norm')
        if img1_norm is not None and img2_norm is not None:
            illum_diff = torch.abs(img1_norm - img2_norm).mean(dim=1, keepdim=True)
            loss_illum = 0.05 * (illum_diff * static_mask).sum() / (static_mask.sum() + 1e-6)
        else:
            loss_illum = outputs['logits'].new_tensor(0.0)

        H_gnd = outputs.get('H_gnd')
        if H_gnd is not None:
            eye3 = torch.eye(3, dtype=H_gnd.dtype, device=H_gnd.device).unsqueeze(0)
            loss_hgnd = 0.01 * ((H_gnd - eye3) ** 2).mean()
        else:
            loss_hgnd = outputs['logits'].new_tensor(0.0)

        # Total Weighted Multi-Task Loss
        total_loss = (
            self.lambda_focal * loss_focal +
            self.lambda_dice * loss_dice +
            loss_aux +
            self.lambda_sampson * loss_sampson +
            self.lambda_cycle * loss_cycle +
            self.lambda_normal * loss_normal +
            self.lambda_ortho * loss_ortho +
            loss_illum +
            loss_hgnd
        )

        return {
            'loss': total_loss,
            'loss_focal': loss_focal,
            'loss_dice': loss_dice,
            'loss_sampson': loss_sampson,
            'loss_cycle': loss_cycle,
            'loss_normal': loss_normal,
            'loss_ortho': loss_ortho,
        }
