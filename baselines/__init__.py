"""External SOTA Baselines for Scene Change Detection (SCD).

Includes:
1. CSNet: ChangeSim Official Baseline (ICRA 2021).
2. ChangeFormer: Hierarchical SegFormer / MiT Transformer Baseline (IEEE JSTARS 2022).
3. DR-TANet: Dual-Resolution Spatio-Temporal Attention Network (IEEE TIP 2020).
4. C-3PO (C3PONet): Combine, Compare and Compress for Change Detection (CVPR 2023 / PR).
5. CGNet: Change Guiding Network (IEEE J-STARS 2023).
6. USSFC-Net (USSFCNet): Ultralightweight Spatial-Spectral Feature Cooperation Net (IEEE TGRS 2023).
7. Changer: Feature Interaction is What You Need for Change Detection (ICCV 2023).

All baseline models strictly adhere to the unified forward contract:
    inputs:  img1 [B, 3, H, W], img2 [B, 3, H, W]
    outputs: {'logits': [B, 1, H, W], 'prob': [B, 1, H, W]}
"""

from typing import Any, Dict, List, Optional
import torch.nn as nn

from .csnet import CSNet
from .changeformer import ChangeFormer
from .dr_tanet import DRTANet
from .c3po import C3PONet
from .cgnet import CGNet
from .ussfcnet import USSFCNet
from .changer import Changer

# Aliases
DR_TANet = DRTANet
C3PO = C3PONet
USSFC_Net = USSFCNet


def get_baseline_model(name: str, **kwargs: Any) -> nn.Module:
    """Factory function for instantiating SOTA SCD baseline models.

    Args:
        name: Baseline model identifier string ('csnet', 'changeformer', 'dr_tanet',
              'c3po', 'cgnet', 'ussfcnet', 'changer').
        **kwargs: Optional model architecture parameters.

    Returns:
        Instantiated baseline nn.Module.
    """
    key = name.lower().replace("-", "_").replace(" ", "")

    if key in ("csnet", "cs_net"):
        return CSNet(**kwargs)
    elif key in ("changeformer", "change_former"):
        return ChangeFormer(**kwargs)
    elif key in ("drtanet", "dr_tanet"):
        return DRTANet(**kwargs)
    elif key in ("c3po", "c_3po", "c3ponet"):
        return C3PONet(**kwargs)
    elif key in ("cgnet", "cg_net"):
        return CGNet(**kwargs)
    elif key in ("ussfcnet", "ussfc_net"):
        return USSFCNet(**kwargs)
    elif key in ("changer", "changer_ex"):
        return Changer(**kwargs)
    else:
        raise ValueError(
            f"Unknown baseline model '{name}'. "
            f"Supported baselines: ['csnet', 'changeformer', 'dr_tanet', "
            f"'c3po', 'cgnet', 'ussfcnet', 'changer']."
        )


__all__ = [
    "CSNet",
    "ChangeFormer",
    "DRTANet",
    "DR_TANet",
    "C3PONet",
    "C3PO",
    "CGNet",
    "USSFCNet",
    "USSFC_Net",
    "Changer",
    "get_baseline_model",
]

