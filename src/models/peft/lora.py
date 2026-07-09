"""
Parameter-Efficient Fine-Tuning Adapters (LoRA) for XBone-Net.
===============================================================================
Implements LoRA (Low-Rank Adaptation; Hu et al., 2021) adapter injection for 
VLM vision and text backbones.
"""

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

def resolve_target_modules(module: nn.Module, requested_targets: list | None = None) -> list:
    """Resolve valid target module names for PEFT injection on the base module.

    If requested_targets are provided and at least one target exists inside module,
    returns the matching targets. Otherwise, automatically inspects module structure
    and selects valid linear layer suffixes compatible with the model architecture.
    """
    all_module_names = [name for name, _ in module.named_modules()]

    if requested_targets:
        matching = []
        for target in requested_targets:
            if any(name.endswith(target) or f".{target}." in name or name == target for name in all_module_names):
                matching.append(target)
        if matching:
            return matching

    candidate_suffixes = [
        "c_fc", "c_proj", "in_proj", "out_proj",
        "q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2",
        "attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2",
        "query", "key", "value", "attention.output.dense", "intermediate.dense", "output.dense",
        "qkv", "proj",
    ]

    found = []
    for suffix in candidate_suffixes:
        if any(name.endswith(suffix) or f".{suffix}." in name or name == suffix for name in all_module_names):
            if suffix not in found:
                found.append(suffix)

    if found:
        return found

    for name, m in module.named_modules():
        cls_name = type(m).__name__
        if cls_name in ["Linear", "Linear4bit"]:
            part = name.split(".")[-1]
            if part not in found:
                found.append(part)

    return found if found else ["qkv", "proj"]


def inject_lora(
    module: nn.Module, 
    r: int = 16, 
    alpha: int = 32,
    dropout: float = 0.1, 
    target_modules: list = None, 
    **kwargs
) -> nn.Module:
    """Apply standard unquantized LoRA adapters to an encoder module.

    Freezes base parameters of module and attaches trainable low-rank adapter matrices
    to specified target layers.

    Args:
        module (nn.Module): PyTorch module to adapt.
        r (int): Rank r of LoRA decomposition. Defaults to 16.
        alpha (int): Scaling factor alpha for LoRA updates. Defaults to 32.
        dropout (float): Dropout probability applied to LoRA inputs. Defaults to 0.1.
        target_modules (list, optional): Target sub-module names. Defaults to ["qkv", "proj"].
        **kwargs: Unused extra keyword arguments.

    Returns:
        nn.Module: Wrapped HuggingFace PeftModel ready for fine-tuning.
    """
    target_modules = resolve_target_modules(module, target_modules)

    # Freeze base parameters
    for param in module.parameters():
        param.requires_grad = False

    print(f"[LoRA] Injecting LoRA adapters (r={r}, alpha={alpha}) on targets: {target_modules}")

    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
        modules_to_save=[],
    )

    peft_module = get_peft_model(module, lora_config)
    print("[LoRA] LoRA injection complete.")

    return peft_module
