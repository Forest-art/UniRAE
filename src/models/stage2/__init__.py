"""
Stage 2 models: DiT and DDT for latent diffusion.
"""

from .lightning_dit import LightningDiT, LightningDiTBlock, LightningFinalLayer
from .ddt import DiTwDDTHead, DDTBlock, DDTFinalLayer, DDTModulate, DDTGate
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

__all__ = [
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
]