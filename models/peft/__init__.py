import torch.nn as nn
from .QLoRA import inject_qlora, inject_lora


def apply_none(module: nn.Module, **kwargs) -> nn.Module:
    """No PEFT injection. Returns the model unchanged (for frozen baselines)."""
    return module


def apply_full_ft(module: nn.Module, **kwargs) -> nn.Module:
    """Full fine-tuning: unfreezes ALL parameters. No adapters injected."""
    for param in module.parameters():
        param.requires_grad = True
    return module


PEFT_REGISTRY = {
    'none': apply_none,
    'full_ft': apply_full_ft,
    'lora': inject_lora,
    'qlora': inject_qlora,
}


def apply_peft(module: nn.Module, cfg: dict) -> nn.Module:
    """Factory: apply PEFT technique to a module based on config."""
    peft_type = cfg.get('type', 'none')
    
    if peft_type not in PEFT_REGISTRY:
        raise ValueError(f"PEFT '{peft_type}' not supported. Choose from {list(PEFT_REGISTRY.keys())}")
        
    inject_fn = PEFT_REGISTRY[peft_type]
    return inject_fn(module=module, **cfg.get('params', {}))