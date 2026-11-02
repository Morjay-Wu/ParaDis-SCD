"""ParaDis-SCD: Parallax-Disocclusion Decoupled 3D Geometric Scene Change Detection.

Top-level end-to-end model for CVPR main conference.
Unifies:
1. DynamicBackbone (Rectangular dynamic multi-scale feature extractor)
2. DBAIA (Dual-Stream Background-Anchored Illumination Alignment)
3. MGPriors (Zero-Sensor Monocular Depth & Unit Surface Normal Priors)
4. DESG (Differentiable Epipolar-Scale Gauge & HEH-Bundle)
5. ECDCA (Epipolar-Constrained Deformable Cross-Attention)
6. PDAG (Perspective Differential Disocclusion & Parallax Gate)
7. ProgressiveDecoder (Parallax-Suppressed Multi-Scale Progressive Decoder)
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import DynamicBackbone
from .retinex_align import DBAIA
from .mg_priors import MGPriors
from .d_esg import DESG
from .ec_deform_attn import ECDCA
from .p_dag import PDAG
from .decoder import ProgressiveDecoder


class ParaDisSCDNet(nn.Module):
    """ParaDis-SCD Top-Level Network.

    Args:
        backbone_name: Backbone architecture name ('native_convnext', 'convnext_tiny', 'resnet34', etc.).
        pretrained: Whether to load pretrained weights for backbone (default False).
        feature_dim: Primary semantic feature channel dimension C (default 256).
        num_hypotheses: Number of depth hypotheses K in EC-DCA (default 16).
        freeze_backbone: Whether to freeze backbone weights (default False).
        freeze_priors: Whether to freeze monocular geometry priors weights (default True).
    """

    def __init__(
        self,
        backbone_name: str = 'native_convnext',
        pretrained: bool = False,
        feature_dim: int = 256,
        num_hypotheses: int = 16,
        freeze_backbone: bool = False,
        freeze_priors: bool = False,
        ablation: str = 'full',
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_hypotheses = num_hypotheses
        self.freeze_priors = freeze_priors
        self.ablation = ablation.lower()

        # Module 1: Illumination Alignment (D-BAIA)
        self.illum_align = DBAIA(in_channels=3)

        # Module 2: Monocular Geometry Priors (MG-Priors)
        self.mg_priors = MGPriors(in_channels=3, base_channels=32)
        if freeze_priors:
            for p in self.mg_priors.parameters():
                p.requires_grad = False
            self.mg_priors.eval()

        # Module 3: Dynamic Rectangular Feature Backbone
        self.backbone = DynamicBackbone(
            name=backbone_name,
            pretrained=pretrained,
            out_channels=feature_dim,
            freeze=freeze_backbone,
        )

        # Module 4: Differentiable Epipolar-Scale Gauge (D-ESG)
        self.d_esg = DESG(
            in_channels=feature_dim,
            grid_size=(16, 16),
            temp=0.05,
            sinkhorn_iters=5,
        )

        # Module 5: Epipolar-Constrained Deformable Cross-Attention (EC-DCA)
        self.ec_dca = ECDCA(
            in_channels=feature_dim,
            num_hypotheses=num_hypotheses,
            num_heads=4,
            max_offset=0.05,
            lambda_geom=2.0,
        )

        # Module 6: Perspective Differential Disocclusion & Parallax Gate (P-DAG)
        self.p_dag = PDAG(
            in_channels=feature_dim,
            out_channels=feature_dim,
            tau_z=0.25,
            gamma_z=0.15,
            tau_c=0.15,
            gamma_c=0.08,
        )

        # Module 7: Parallax-Suppressed Multi-Scale Progressive Decoder
        self.decoder = ProgressiveDecoder(
            in_channels=feature_dim,
            c3_channels=128,
            c2_channels=64,
            out_channels=1,
            c1_channels=32,
        )

    def forward(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass supporting arbitrary rectangular input image pairs.

        Args:
            img1: Image at time 1 [B, 3, H, W] in [0, 1]
            img2: Image at time 2 [B, 3, H, W] in [0, 1]

        Returns:
            Dict containing:
                'logits': Raw change logits [B, 1, H, W]
                'prob': Sigmoid change probability [B, 1, H, W]
                'g_occ': Continuous disocclusion gate at full resolution [B, 1, H, W]
                'delta_normal': Surface normal discrepancy at full resolution [B, 1, H, W]
                'cycle_error': Bidirectional loop closure error at full resolution [B, 1, H, W]
                'g_occ_p': Disocclusion gate at feature scale [B, 1, H_p, W_p]
                'delta_normal_p': Normal discrepancy at feature scale [B, 1, H_p, W_p]
                'cycle_error_p': Cycle error at feature scale [B, 1, H_p, W_p]
                'F_12': Rank-2 fundamental matrix [B, 3, 3] with ||F||_F = 1.0 (with full grad graph)
                'R_12': Relative camera rotation matrix in SO(3) [B, 3, 3] (with full grad graph)
                'alpha': Depth scale parameter [B, 1] (with full grad graph)
                'beta': Depth shift parameter [B, 1] (with full grad graph)
                'H_gnd': Ground plane homography in SL(3) [B, 3, 3] (with full grad graph)
                'feat2_warped': Epipolar-aligned warped feature [B, C, H_p, W_p]
                'warp_grid_12': Dense forward sampling grid [B, 2, H_p, W_p]
                'warp_grid_21': Dense backward sampling grid [B, 2, H_p, W_p]
                'depth1': Monocular relative depth for image 1 [B, 1, H, W]
                'depth2': Monocular relative depth for image 2 [B, 1, H, W]
                'normal1': Unit surface normal field for image 1 [B, 3, H, W] in S^2
                'normal2': Unit surface normal field for image 2 [B, 3, H, W] in S^2
                'img1_norm': Illumination-aligned image 1 [B, 3, H, W]
                'img2_norm': Normalized image 2 [B, 3, H, W]
        """
        B, C_in, H, W = img1.shape
        device_type = 'cuda' if img1.is_cuda else 'cpu'

        # Step 1: Illumination Alignment (D-BAIA)
        illum_out = self.illum_align(img1, img2)
        if self.ablation == 'no_dbaia':
            img1_norm = torch.clamp(img1, min=0.0, max=1.0)
            img2_norm = torch.clamp(img2, min=0.0, max=1.0)
        else:
            img1_norm = illum_out['img1_norm']
            img2_norm = illum_out['img2_norm']

        # Step 2: Monocular Geometry Priors (MG-Priors)
        # Execute in FP32 precision to guarantee numerical stability across all hardware architectures
        if self.freeze_priors:
            with torch.no_grad():
                with torch.amp.autocast(device_type=device_type, enabled=False):
                    mg1 = self.mg_priors(img1_norm.float())
                    mg2 = self.mg_priors(img2_norm.float())
        else:
            with torch.amp.autocast(device_type=device_type, enabled=False):
                mg1 = self.mg_priors(img1_norm.float())
                mg2 = self.mg_priors(img2_norm.float())

        depth1 = mg1['depth'].to(dtype=img1.dtype)
        normal1 = mg1['normal'].to(dtype=img1.dtype)
        depth2 = mg2['depth'].to(dtype=img2.dtype)
        normal2 = mg2['normal'].to(dtype=img2.dtype)

        # Step 3: Dynamic Rectangular Feature Backbone
        feats1 = self.backbone(img1_norm)
        feats2 = self.backbone(img2_norm)

        c1_1 = feats1.get('c1')
        c1_2 = feats2.get('c1')
        c2_1, c3_1, c4_1 = feats1['c2'], feats1['c3'], feats1['c4']
        c2_2, c3_2, c4_2 = feats2['c2'], feats2['c3'], feats2['c4']
        pad = feats1['pad']
        H_p, W_p = c4_1.shape[-2:]

        # Downsample depth and unit normal to feature resolution (H_p, W_p) with S^2 re-normalization
        depth1_p, normal1_p = MGPriors.downsample_priors(depth1, normal1, size=(H_p, W_p))
        depth2_p, normal2_p = MGPriors.downsample_priors(depth2, normal2, size=(H_p, W_p))

        # Step 4: Differentiable Epipolar-Scale Gauge (D-ESG)
        esg_out = self.d_esg(c4_1, c4_2, depth1_p, depth2_p)
        F_12 = esg_out['F_12']
        R_12 = esg_out['R_12']
        scale_shift = esg_out['scale_shift']
        H_gnd = esg_out['H_gnd']
        alpha = esg_out['alpha']
        beta = esg_out['beta']

        # Gauge detachment for cross-attention sampling coordinates to guarantee
        # everywhere-finite gradient flow in FP16/AMP without second-order SVD singularities,
        # while preserving full gradients on returned F_12, R_12, H_gnd, alpha, beta for M3 multi-task losses.
        if self.ablation == 'no_desg':
            eye3 = torch.eye(3, device=img1.device, dtype=img1.dtype).unsqueeze(0).repeat(B, 1, 1)
            F_12_gauge = F_12.detach() * 0.0
            F_12_gauge[:, 1, 2] = -0.7071
            F_12_gauge[:, 2, 1] = 0.7071
            H_gnd_gauge = eye3
            R_12_gauge = eye3
            scale_shift_gauge = (torch.ones_like(alpha.detach()), torch.zeros_like(beta.detach()))
        else:
            F_12_gauge = F_12.detach()
            H_gnd_gauge = H_gnd.detach()
            R_12_gauge = R_12
            scale_shift_gauge = (alpha.detach(), beta.detach())

        # Step 5: Epipolar-Constrained Deformable Cross-Attention (EC-DCA)
        dca_out = self.ec_dca(
            feat1=c4_1,
            feat2=c4_2,
            F_12=F_12_gauge,
            depth1=depth1_p,
            depth2=depth2_p,
            scale_shift=scale_shift_gauge,
            H_gnd=H_gnd_gauge,
            bidirectional=True,
        )
        if self.ablation == 'no_ecdca':
            c4_2_warped = c4_2
            y_g = torch.linspace(-1.0, 1.0, H_p, device=img1.device, dtype=img1.dtype)
            x_g = torch.linspace(-1.0, 1.0, W_p, device=img1.device, dtype=img1.dtype)
            gy, gx = torch.meshgrid(y_g, x_g, indexing='ij')
            base_g = torch.stack([gx, gy], dim=0).unsqueeze(0).repeat(B, 1, 1, 1)
            warp_grid_12 = base_g
            warp_grid_21 = base_g
        else:
            c4_2_warped = dca_out['feat2_warped']
            warp_grid_12 = dca_out['warp_grid_12']
            warp_grid_21 = dca_out['warp_grid_21']

        # Step 6: Perspective Differential Disocclusion & Parallax Gate (P-DAG)
        dag_out = self.p_dag(
            feat1=c4_1,
            feat2_warped=c4_2_warped,
            depth1=depth1_p,
            depth2=depth2_p,
            normal1=normal1_p,
            normal2=normal2_p,
            R_12=R_12_gauge,
            warp_grid_12=warp_grid_12,
            warp_grid_21=warp_grid_21,
            scale_shift=scale_shift_gauge,
            feat2_raw=c4_2,
        )
        fused_feat = dag_out['fused_feat']
        g_occ_p = dag_out['g_occ']
        delta_normal_p = dag_out['delta_normal']
        cycle_error_p = dag_out['cycle_error']
        if self.ablation == 'no_pdag':
            d_raw_ab = dag_out['d_raw']
            zero_map = torch.zeros_like(g_occ_p)
            x_cat_ab = torch.cat([c4_1, c4_2_warped, d_raw_ab, d_raw_ab, zero_map, zero_map, zero_map], dim=1)
            fused_init_ab = self.p_dag.fuse_proj(x_cat_ab)
            fused_feat = self.p_dag.final_act(fused_init_ab + self.p_dag.res_block(fused_init_ab))
            g_occ_p = zero_map
            g_occ_for_dec = zero_map
        else:
            g_occ_for_dec = g_occ_p

        # Step 7: Parallax-Suppressed Multi-Scale Progressive Decoder
        dec_out = self.decoder(
            fused_feat=fused_feat,
            skip_c3=c3_1,
            skip_c2=c2_1,
            target_size=(H, W),
            pad=pad,
            skip_c3_2=c3_2,
            skip_c2_2=c2_2,
            skip_c1=c1_1,
            skip_c1_2=c1_2,
            warp_grid_12=warp_grid_12,
            g_occ=g_occ_for_dec,
        )
        logits = dec_out['logits']
        prob = dec_out['prob']
        aux_logits = dec_out.get('aux_logits')

        # Upsample diagnostic maps to full resolution for visualization and losses
        g_occ = F.interpolate(g_occ_p, size=(H, W), mode='bilinear', align_corners=False)
        delta_normal = F.interpolate(delta_normal_p, size=(H, W), mode='bilinear', align_corners=False)
        cycle_error = F.interpolate(cycle_error_p, size=(H, W), mode='bilinear', align_corners=False)

        return {
            'logits': logits,
            'prob': prob,
            'aux_logits': aux_logits,
            'g_occ': g_occ,
            'delta_normal': delta_normal,
            'cycle_error': cycle_error,
            'g_occ_p': g_occ_p,
            'delta_normal_p': delta_normal_p,
            'cycle_error_p': cycle_error_p,
            'F_12': F_12,
            'R_12': R_12,
            'alpha': alpha,
            'beta': beta,
            'H_gnd': H_gnd,
            'feat2_warped': c4_2_warped,
            'warp_grid_12': warp_grid_12,
            'warp_grid_21': warp_grid_21,
            'depth1': depth1,
            'depth2': depth2,
            'normal1': normal1,
            'normal2': normal2,
            'img1_norm': img1_norm,
            'img2_norm': img2_norm,
        }


# Aliases
ParaDisSCD = ParaDisSCDNet
ParaDis = ParaDisSCDNet
