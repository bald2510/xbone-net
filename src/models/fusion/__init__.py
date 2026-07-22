"""
Multimodal Fusion Module Registry & Factory for XBone-Net.
===============================================================================
Defines the registry and factory function for instantiating multimodal feature fusion
strategies:
  - IdentityFusion: Pass-through image features (Phase 1 or single-modality)
  - CrossAttentionFusion: Multi-head cross-attention mixing image and text embeddings
  - ConcatFusion: Feature concatenation with linear projection bottleneck
"""

from .identity import IdentityFusion
from .cross_attention import CrossAttentionFusion
from .gated_cross_attention import GatedCrossAttentionFusion
from .concat import ConcatFusion

# ============================================================
# Fusion Registry Mapping
# ============================================================

FUSION_REGISTRY = {
    'none': IdentityFusion,
    'cross_attention': CrossAttentionFusion,
    'gated_cross_attention': GatedCrossAttentionFusion,
    'concat': ConcatFusion,
}


# ============================================================
# Fusion Module Factory
# ============================================================

def build_fusion_module(cfg: dict):
    """Build multimodal fusion module from configuration dictionary.

    Args:
        cfg (dict): Fusion configuration containing 'type' and optional 'params'.

    Returns:
        nn.Module: Instantiated fusion module matching specified strategy.

    Raises:
        ValueError: If specified fusion type is not supported in FUSION_REGISTRY.

    Example:
        >>> fusion = build_fusion_module({'type': 'cross_attention', 'params': {'img_dim': 512}})
    """
    fusion_type = cfg.get('type', 'none')
    if fusion_type not in FUSION_REGISTRY:
        raise ValueError(f"Fusion '{fusion_type}' not supported. Choose from {list(FUSION_REGISTRY.keys())}")

    return FUSION_REGISTRY[fusion_type](**cfg.get('params', {}))

