"""
Parameter-Efficient Fine-Tuning (PEFT) Registry & Factory for XBone-Net.
===============================================================================
Defines the registry and factory functions for applying parameter-efficient fine-tuning:
  - none: Pass-through without modification
  - full_ft: Unfreezes all module parameters for full fine-tuning
  - lora: Low-Rank Adaptation (LoRA) via HuggingFace peft library
  - qlora: Quantized 4-bit Low-Rank Adaptation (QLoRA) via bitsandbytes
"""

import torch.nn as nn
from .qlora import inject_qlora
from .lora import inject_lora


# ============================================================
# Basic PEFT Handlers
# ============================================================

def apply_none(module: nn.Module, **kwargs) -> nn.Module:
    """Pass-through helper leaving module parameters unmodified.

    Args:
        module (nn.Module): Target PyTorch module.
        **kwargs: Unused extra keyword arguments.

    Returns:
        nn.Module: Unmodified target module.
    """
    return module


def apply_full_ft(module: nn.Module, **kwargs) -> nn.Module:
    """Unfreeze all module parameters for full fine-tuning.

    Args:
        module (nn.Module): Target PyTorch module.
        **kwargs: Unused extra keyword arguments.

    Returns:
        nn.Module: Target module with all requires_grad set to True.
    """
    for param in module.parameters():
        param.requires_grad = True
    return module


# ============================================================
# PEFT Registry Mapping
# ============================================================

PEFT_REGISTRY = {
    'none': apply_none,
    'full_ft': apply_full_ft,
    'lora': inject_lora,
    'qlora': inject_qlora,
}


# ============================================================
# PEFT Injection Factory
# ============================================================

def apply_peft(module: nn.Module, cfg: dict) -> nn.Module:
    """Apply specified PEFT strategy to a target PyTorch module.

    Args:
        module (nn.Module): Target module (e.g., visual or text encoder).
        cfg (dict): PEFT configuration dictionary containing 'type' and 'params'.

    Returns:
        nn.Module: Modified PyTorch module with PEFT adapters injected.

    Raises:
        ValueError: If specified peft_type is not supported in PEFT_REGISTRY.

    Example:
        >>> module = apply_peft(backbone.model.visual, {'type': 'lora', 'params': {'r': 16}})
    """
    peft_type = cfg.get('type', 'none')
    if peft_type not in PEFT_REGISTRY:
        raise ValueError(f"PEFT '{peft_type}' not supported. Choose from {list(PEFT_REGISTRY.keys())}")
        
    return PEFT_REGISTRY[peft_type](module=module, **cfg.get('params', {}))
