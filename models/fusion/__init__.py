import torch.nn as nn
from .identity import IdentityFusion
from .cross_attention import CrossAttentionFusion
from .concat import ConcatFusion

FUSION_REGISTRY = {
    'none': IdentityFusion,
    'cross_attention': CrossAttentionFusion,
    'concat': ConcatFusion,
}

def build_fusion_module(cfg):
    fusion_type = cfg.get('type', 'none')
    if fusion_type not in FUSION_REGISTRY:
        raise ValueError(f"Fusion '{fusion_type}' not supported. Choose from {list(FUSION_REGISTRY.keys())}")
        
    return FUSION_REGISTRY[fusion_type](**cfg.get('params', {}))