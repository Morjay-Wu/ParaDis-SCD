"""ChangeFormer: Transformer-Based Siamese Network for Scene Change Detection.

Paper: "A Transformer-Based Siamese Network for Change Detection"
IEEE JSTARS 2022 / CVPRW.

Key Architecture Components:
1. Hierarchical Mix Transformer (MiT / SegFormer) Siamese encoder.
2. Dynamic 2D Overlapped Patch Embedding tracking explicit (H_i, W_i) grid dimensions.
   CRITICAL: Strict avoidance of square-grid sqrt(N) assumptions, enabling native
   rectangular resolutions (480x640, 224x1024, etc.).
3. Spatial Reduction Attention (SRA) with dynamic key-value downsampling.
4. Mix-FFN with depthwise 3x3 convolution providing dynamic 2D positional encodings.
5. Multi-scale difference module and All-MLP progressive change decoder.
6. Returns unified baseline contract: {'logits': [B, 1, H, W], 'prob': [B, 1, H, W]}.
"""

from typing import Dict, List, Optional, Tuple, Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicPatchEmbed(nn.Module):
    """Dynamic Overlapped Patch Embedding with explicit 2D spatial tracking."""

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 64,
        patch_size: int = 7,
        stride: int = 4,
        padding: int = 3,
    ):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=padding,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """Projects image tensor into token sequence while recording (H_p, W_p).
        
        Args:
            x: Input feature tensor [B, C, H, W]
            
        Returns:
            x_tokens: Token sequence [B, H_p * W_p, embed_dim]
            H_p, W_p: Precise rectangular spatial dimensions
        """
        x = self.proj(x)
        B, C, H_p, W_p = x.shape
        # Permute to [B, H_p, W_p, C] and flatten spatial tokens
        x = x.permute(0, 2, 3, 1).contiguous().view(B, H_p * W_p, C)
        x = self.norm(x)
        return x, H_p, W_p


class SpatialReductionAttention(nn.Module):
    """Multi-Head Self-Attention with Spatial Reduction (SRA).
    
    Dynamically downsamples Key and Value spatial dimensions by factor sr_ratio,
    reducing O(N^2) complexity to O(N^2 / R^2).
    Uses explicit (H, W) rectangular spatial shapes (NO sqrt assumptions!).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        sr_ratio: int = 1,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.sr_ratio = sr_ratio

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        # Verify shape consistency
        assert N == H * W, f"Token count N={N} does not match explicit H*W={H*W}"

        # Query projection: [B, num_heads, N, head_dim]
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            # Reshape using EXPLICIT (H, W) - strictly NO math.sqrt(N)
            x_2d = x.view(B, H, W, C).permute(0, 3, 1, 2)
            x_reduced = self.sr(x_2d)
            B_r, C_r, H_r, W_r = x_reduced.shape
            x_reduced = x_reduced.permute(0, 2, 3, 1).contiguous().view(B, H_r * W_r, C)
            x_kv = self.norm(x_reduced)
        else:
            x_kv = x

        # Key and Value projections: [B, num_heads, N_kv, head_dim]
        N_kv = x_kv.shape[1]
        kv = self.kv(x_kv).reshape(B, N_kv, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        # Scaled Dot-Product Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        return out


class MixFFN(nn.Module):
    """Mix Feed-Forward Network with 3x3 Depthwise Convolution.
    
    Acts as a dynamic 2D positional encoding without fixed positional lookup tables,
    supporting arbitrary rectangular resolutions natively.
    """

    def __init__(self, in_features: int, hidden_features: Optional[int] = None):
        super().__init__()
        hidden_features = hidden_features or in_features * 4
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = nn.Conv2d(
            hidden_features,
            hidden_features,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features,
            bias=True,
        )
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        x = self.fc1(x)
        # Reshape strictly using dynamic (H, W)
        x_2d = x.view(B, H, W, -1).permute(0, 3, 1, 2)
        x_2d = self.dwconv(x_2d)
        x_2d = self.act(x_2d)
        x = x_2d.permute(0, 2, 3, 1).contiguous().view(B, N, -1)
        x = self.fc2(x)
        return x


class MiTBlock(nn.Module):
    """Mix Transformer (MiT) Block."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        sr_ratio: int = 1,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SpatialReductionAttention(
            dim=dim,
            num_heads=num_heads,
            sr_ratio=sr_ratio,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MixFFN(in_features=dim, hidden_features=int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.mlp(self.norm2(x), H, W)
        return x


class HierarchicalMiTEncoder(nn.Module):
    """Hierarchical SegFormer / MiT Encoder for Siamese Feature Extraction."""

    def __init__(
        self,
        in_channels: int = 3,
        embed_dims: List[int] = [32, 64, 160, 256],
        num_heads: List[int] = [1, 2, 5, 8],
        depths: List[int] = [2, 2, 2, 2],
        sr_ratios: List[int] = [8, 4, 2, 1],
    ):
        super().__init__()
        self.embed_dims = embed_dims

        # Patch Embeddings for 4 stages
        self.patch_embed1 = DynamicPatchEmbed(
            in_channels=in_channels,
            embed_dim=embed_dims[0],
            patch_size=7,
            stride=4,
            padding=3,
        )
        self.patch_embed2 = DynamicPatchEmbed(
            in_channels=embed_dims[0],
            embed_dim=embed_dims[1],
            patch_size=3,
            stride=2,
            padding=1,
        )
        self.patch_embed3 = DynamicPatchEmbed(
            in_channels=embed_dims[1],
            embed_dim=embed_dims[2],
            patch_size=3,
            stride=2,
            padding=1,
        )
        self.patch_embed4 = DynamicPatchEmbed(
            in_channels=embed_dims[2],
            embed_dim=embed_dims[3],
            patch_size=3,
            stride=2,
            padding=1,
        )

        # Transformer Blocks for 4 stages
        self.block1 = nn.ModuleList([
            MiTBlock(embed_dims[0], num_heads[0], sr_ratio=sr_ratios[0]) for _ in range(depths[0])
        ])
        self.block2 = nn.ModuleList([
            MiTBlock(embed_dims[1], num_heads[1], sr_ratio=sr_ratios[1]) for _ in range(depths[1])
        ])
        self.block3 = nn.ModuleList([
            MiTBlock(embed_dims[2], num_heads[2], sr_ratio=sr_ratios[2]) for _ in range(depths[2])
        ])
        self.block4 = nn.ModuleList([
            MiTBlock(embed_dims[3], num_heads[3], sr_ratio=sr_ratios[3]) for _ in range(depths[3])
        ])

        # Normalization layers
        self.norm1 = nn.LayerNorm(embed_dims[0])
        self.norm2 = nn.LayerNorm(embed_dims[1])
        self.norm3 = nn.LayerNorm(embed_dims[2])
        self.norm4 = nn.LayerNorm(embed_dims[3])

    def forward_single(self, x: torch.Tensor) -> List[torch.Tensor]:
        B = x.shape[0]

        # Stage 1: [B, C1, H/4, W/4]
        x1, H1, W1 = self.patch_embed1(x)
        for blk in self.block1:
            x1 = blk(x1, H1, W1)
        x1 = self.norm1(x1)
        f1 = x1.view(B, H1, W1, self.embed_dims[0]).permute(0, 3, 1, 2).contiguous()

        # Stage 2: [B, C2, H/8, W/8]
        x2, H2, W2 = self.patch_embed2(f1)
        for blk in self.block2:
            x2 = blk(x2, H2, W2)
        x2 = self.norm2(x2)
        f2 = x2.view(B, H2, W2, self.embed_dims[1]).permute(0, 3, 1, 2).contiguous()

        # Stage 3: [B, C3, H/16, W/16]
        x3, H3, W3 = self.patch_embed3(f2)
        for blk in self.block3:
            x3 = blk(x3, H3, W3)
        x3 = self.norm3(x3)
        f3 = x3.view(B, H3, W3, self.embed_dims[2]).permute(0, 3, 1, 2).contiguous()

        # Stage 4: [B, C4, H/32, W/32]
        x4, H4, W4 = self.patch_embed4(f3)
        for blk in self.block4:
            x4 = blk(x4, H4, W4)
        x4 = self.norm4(x4)
        f4 = x4.view(B, H4, W4, self.embed_dims[3]).permute(0, 3, 1, 2).contiguous()

        return [f1, f2, f3, f4]

    def forward(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Shared Siamese encoder forward pass."""
        return self.forward_single(img1), self.forward_single(img2)


class ChangeFormerDecoder(nn.Module):
    """All-MLP Multi-Scale Progressive Change Decoder."""

    def __init__(
        self,
        in_channels: List[int] = [32, 64, 160, 256],
        embed_dim: int = 128,
    ):
        super().__init__()
        c1, c2, c3, c4 = in_channels
        self.linear_c4 = nn.Sequential(
            nn.Conv2d(c4, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )
        self.linear_c3 = nn.Sequential(
            nn.Conv2d(c3, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )
        self.linear_c2 = nn.Sequential(
            nn.Conv2d(c2, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )
        self.linear_c1 = nn.Sequential(
            nn.Conv2d(c1, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )

        # Difference modules at each stage: diff = |f1 - f2|
        # Fusion layer across all 4 scales concatenated
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(embed_dim * 4, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )

        # Final projection to binary change logits
        self.classifier = nn.Conv2d(embed_dim, 1, kernel_size=1)

    def forward(
        self,
        feats1: List[torch.Tensor],
        feats2: List[torch.Tensor],
        target_size: Tuple[int, int],
    ) -> torch.Tensor:
        f1_1, f2_1, f3_1, f4_1 = feats1
        f1_2, f2_2, f3_2, f4_2 = feats2

        # 1. Multi-scale feature difference
        d1 = torch.abs(f1_1 - f1_2)  # [B, C1, H/4, W/4]
        d2 = torch.abs(f2_1 - f2_2)  # [B, C2, H/8, W/8]
        d3 = torch.abs(f3_1 - f3_2)  # [B, C3, H/16, W/16]
        d4 = torch.abs(f4_1 - f4_2)  # [B, C4, H/32, W/32]

        # 2. Linear projection to common embed_dim
        _d1 = self.linear_c1(d1)
        _d2 = self.linear_c2(d2)
        _d3 = self.linear_c3(d3)
        _d4 = self.linear_c4(d4)

        # 3. Upsample all features to stage 1 spatial size (H/4, W/4)
        size1 = _d1.shape[-2:]
        _d2 = F.interpolate(_d2, size=size1, mode='bilinear', align_corners=False)
        _d3 = F.interpolate(_d3, size=size1, mode='bilinear', align_corners=False)
        _d4 = F.interpolate(_d4, size=size1, mode='bilinear', align_corners=False)

        # 4. Concatenate and fuse
        fused = self.linear_fuse(torch.cat([_d1, _d2, _d3, _d4], dim=1))

        # 5. Classify and upsample to full target resolution (H, W)
        logits = self.classifier(fused)
        if logits.shape[-2:] != target_size:
            logits = F.interpolate(logits, size=target_size, mode='bilinear', align_corners=False)

        return logits


class ChangeFormer(nn.Module):
    """ChangeFormer Baseline Model for Scene Change Detection.
    
    Args:
        in_channels: Input image channel count (default 3).
        embed_dims: Stage channel dimensions (default [32, 64, 160, 256]).
        decoder_dim: Channel dimension for fused decoder (default 128).
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dims: List[int] = [32, 64, 160, 256],
        decoder_dim: int = 128,
    ):
        super().__init__()
        self.encoder = HierarchicalMiTEncoder(
            in_channels=in_channels,
            embed_dims=embed_dims,
            num_heads=[1, 2, 5, 8],
            depths=[2, 2, 2, 2],
            sr_ratios=[8, 4, 2, 1],
        )
        self.decoder = ChangeFormerDecoder(
            in_channels=embed_dims,
            embed_dim=decoder_dim,
        )

    def forward(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for ChangeFormer.
        
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
