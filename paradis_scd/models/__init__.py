"""Core Geometry & Feature Alignment Models for ParaDis-SCD.
"""

from .backbone import DynamicBackbone, NativeConvNeXtBackbone
from .retinex_align import DBAIA, RetinexAlign, DBAIAOutput, FastGuidedFilter
from .mg_priors import MGPriors, MGPriorsOutput
from .d_esg import DESG, DESGOutput, project_to_so3, project_to_sl3, log_sinkhorn
from .ec_deform_attn import ECDCA, EpipolarDeformableCrossAttention, ECDCAOutput
from .p_dag import PDAG, DisocclusionParallaxGate, PDAGOutput
from .decoder import ProgressiveDecoder, ChangeDecoder, DecoderBlock, DecoderOutput
from .paradis_scd import ParaDisSCDNet, ParaDisSCD, ParaDis

# Aliases
RectangularBackbone = DynamicBackbone
EC_DCA = ECDCA
P_DAG = PDAG
Decoder = ProgressiveDecoder

__all__ = [
    # M1 modules
    "DynamicBackbone",
    "RectangularBackbone",
    "NativeConvNeXtBackbone",
    "DBAIA",
    "RetinexAlign",
    "DBAIAOutput",
    "FastGuidedFilter",
    "MGPriors",
    "MGPriorsOutput",
    "DESG",
    "DESGOutput",
    "project_to_so3",
    "project_to_sl3",
    "log_sinkhorn",
    # M2 modules
    "ECDCA",
    "EC_DCA",
    "EpipolarDeformableCrossAttention",
    "ECDCAOutput",
    "PDAG",
    "P_DAG",
    "DisocclusionParallaxGate",
    "PDAGOutput",
    "ProgressiveDecoder",
    "Decoder",
    "ChangeDecoder",
    "DecoderBlock",
    "DecoderOutput",
    "ParaDisSCDNet",
    "ParaDisSCD",
    "ParaDis",
]
