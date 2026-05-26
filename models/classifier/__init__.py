import torch.nn as nn
from .identity import IdentityHead
from .linear import LinearHead
from .prototypical import PrototypicalHead

HEAD_REGISTRY = {
    'none': IdentityHead,
    'linear': LinearHead,
    'prototypical': PrototypicalHead
}

def build_head_module(cfg):
    head_type = cfg.get('type', 'none')
    if head_type not in HEAD_REGISTRY:
        raise ValueError(f"Head '{head_type}' chưa được hỗ trợ.")
        
    return HEAD_REGISTRY[head_type](**cfg.get('params', {}))