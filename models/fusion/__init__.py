import torch.nn as nn
from .identity import IdentityFusion
from .film import FiLMFusion
from .cross_attention import CrossAttentionFusion

FUSION_REGISTRY = {
    'none': IdentityFusion,
    'film': FiLMFusion,
    'cross_attention': CrossAttentionFusion
}

def build_fusion_module(cfg):
    fusion_type = cfg.get('type', 'none')
    if fusion_type not in FUSION_REGISTRY:
        raise ValueError(f"Fusion '{fusion_type}' chưa được hỗ trợ.")
        
    return FUSION_REGISTRY[fusion_type](**cfg.get('params', {}))