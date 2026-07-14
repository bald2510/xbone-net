"""
Classifier Head Module Registry & Factory for XBone-Net.
===============================================================================
Defines the registry and factory function for instantiating classification heads:
  - IdentityHead: Pass-through features (Phase 1 contrastive pre-training)
  - LinearHead: Standard linear layer mapping features to class logits
  - EmpiricalCentroidHead: Non-parametric cosine classifier from train-set means
  - PrototypicalHead: Legacy learnable-prototype head retained for ablations
"""

from .identity import IdentityHead
from .linear import LinearHead
from .prototypical import PrototypicalHead
from .empirical_centroid import EmpiricalCentroidHead

# ============================================================
# Classifier Head Registry Mapping
# ============================================================

HEAD_REGISTRY = {
    'none': IdentityHead,
    'linear': LinearHead,
    'prototypical': PrototypicalHead,
    'learnable_prototype': PrototypicalHead,
    'empirical_centroid': EmpiricalCentroidHead,
}


# ============================================================
# Classifier Head Factory
# ============================================================

def build_head_module(cfg: dict):
    """Build classifier head module from configuration dictionary.

    Args:
        cfg (dict): Classifier head configuration containing 'type' and optional 'params'.

    Returns:
        nn.Module: Instantiated classifier head module.

    Raises:
        ValueError: If specified classifier head type is not supported in HEAD_REGISTRY.

    Example:
        >>> head = build_head_module({'type': 'empirical_centroid', 'params': {'feature_dim': 512, 'num_classes': 4}})
    """
    head_type = cfg.get('type', 'none')
    if head_type not in HEAD_REGISTRY:
        raise ValueError(f"Classifier head '{head_type}' not supported.")

    return HEAD_REGISTRY[head_type](**cfg.get('params', {}))

