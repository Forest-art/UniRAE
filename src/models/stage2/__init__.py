"""
Stage 2 models: DiT and DDT for latent diffusion.
"""

from typing import Any, Callable, Dict, Optional, Type, Union
from typing import Protocol, runtime_checkable

from .ddt import DiTwDDTHead, DDTBlock, DDTFinalLayer, DDTModulate, DDTGate
from .lightning_dit import LightningDiT, LightningDiTBlock, LightningFinalLayer
from .model_utils import (
    GaussianFourierEmbedding,
    LabelEmbedder,
    NormAttention,
    RMSNorm,
    SwiGLUFFN,
    VisionRotaryEmbedding,
    VisionRotaryEmbeddingFast,
    broadcat,
    get_1d_sincos_pos_embed_from_grid,
    get_2d_sincos_pos_embed,
    get_2d_sincos_pos_embed_from_grid,
    modulate,
    rotate_half,
)


@runtime_checkable
class Stage2ModelProtocol(Protocol):
    """Protocol for Stage 2 diffusion models."""

    def forward(self, *args: Any, **kwargs: Any) -> Any: ...
    def forward_with_cfg(self, *args: Any, **kwargs: Any) -> Any: ...
    def forward_with_autoguidance(self, *args: Any, **kwargs: Any) -> Any: ...

__all__ = [
    # Protocol
    "Stage2ModelProtocol",
    # DiT
    "LightningDiT",
    "LightningDiTBlock",
    "LightningFinalLayer",
    # DDT
    "DiTwDDTHead",
    "DDTBlock",
    "DDTFinalLayer",
    "DDTModulate",
    "DDTGate",
    # Utils
    "GaussianFourierEmbedding",
    "LabelEmbedder",
    "NormAttention",
    "RMSNorm",
    "SwiGLUFFN",
    "VisionRotaryEmbedding",
    "VisionRotaryEmbeddingFast",
    "broadcat",
    "get_1d_sincos_pos_embed_from_grid",
    "get_2d_sincos_pos_embed",
    "get_2d_sincos_pos_embed_from_grid",
    "modulate",
    "rotate_half",
    # Transport
    "Transport",
    "ModelType",
    "WeightType",
    "PathType",
    "Sampler",
    "create_transport",
    "sde",
    "ode",
    "ICPlan",
    "VPCPlan",
    "GVPCPlan",
    "EasyDict",
]

# Import transport components for easy access
from .transport import (
    Transport,
    ModelType,
    WeightType,
    PathType,
    Sampler,
    create_transport,
    sde,
    ode,
    ICPlan,
    VPCPlan,
    GVPCPlan,
    EasyDict,
)
