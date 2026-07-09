"""
Parameter-Efficient Fine-Tuning Adapters (LoRA & QLoRA) for XBone-Net.
===============================================================================
Implements LoRA (Low-Rank Adaptation; Hu et al., 2021) and QLoRA (Quantized LoRA;
Dettmers et al., 2023) adapter injection for VLM vision and text backbones.

Mathematical Formulation:
  Given base weight matrix W_0 in R^{d x k}, LoRA introduces low-rank update delta W:
    W' = W_0 + (alpha / r) * (B * A)
  where B in R^{d x r} and A in R^{r x k} with rank r << min(d, k).
"""

import torch
import torch.nn as nn
import bitsandbytes as bnb
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from .lora import resolve_target_modules


# ============================================================
# Quantization Helper
# ============================================================

def _quantize_linear_layers(
    module: nn.Module, 
    quant_type: str = "nf4",
    compute_dtype: torch.dtype = torch.bfloat16,
    double_quant: bool = True
) -> nn.Module:
    """Replace standard Linear layers in a module with bitsandbytes 4-bit layers.

    Traverses module hierarchy recursively and substitutes nn.Linear instances with
    bitsandbytes.nn.Linear4bit layers.

    Args:
        module (nn.Module): PyTorch module whose linear layers will be quantized.
        quant_type (str): Quantization data type ('nf4' or 'fp4'). Defaults to 'nf4'.
        compute_dtype (torch.dtype): PyTorch data type used for activation computations. Defaults to bfloat16.
        double_quant (bool): If True, compress quantization statistics. Defaults to True.

    Returns:
        nn.Module: Input module modified in-place with 4-bit linear layers.
    """
    replacements = []
    # --- Find all linear layers to replace ---
    for name, child in module.named_modules():
        if isinstance(child, nn.Linear):
            replacements.append((name, child))

    # --- Substitute linear layers with 4-bit layers ---
    for name, old_linear in replacements:
        parts = name.split(".")
        parent = module
        for part in parts[:-1]:
            parent = getattr(parent, part)
        attr_name = parts[-1]

        has_bias = old_linear.bias is not None
        new_linear = bnb.nn.Linear4bit(
            input_features=old_linear.in_features,
            output_features=old_linear.out_features,
            bias=has_bias,
            compute_dtype=compute_dtype,
            quant_type=quant_type,
            compress_statistics=double_quant,
        )

        new_linear.weight = bnb.nn.Params4bit(
            old_linear.weight.data,
            requires_grad=False,
            quant_type=quant_type,
            compress_statistics=double_quant,
        )
        if has_bias:
            new_linear.bias = nn.Parameter(old_linear.bias.data, requires_grad=False)

        setattr(parent, attr_name, new_linear)

    print(f"[QLoRA] Quantized {len(replacements)} Linear layers to 4-bit {quant_type.upper()}.")
    return module


# ============================================================
# QLoRA Injection
# ============================================================

def inject_qlora(
    module: nn.Module, 
    r: int = 16, 
    alpha: int = 32,
    dropout: float = 0.1, 
    target_modules: list = None,
    quant_type: str = "nf4", 
    double_quant: bool = True
) -> nn.Module:
    """Apply QLoRA (4-bit quantization + LoRA adapters) to an encoder module.

    Quantizes base weights to 4-bit NF4 format, prepares module for k-bit training,
    and attaches trainable low-rank adapter matrices to specified target layers.

    Args:
        module (nn.Module): PyTorch module (visual or text encoder).
        r (int): Rank r of LoRA decomposition. Defaults to 16.
        alpha (int): Scaling factor alpha for LoRA updates. Defaults to 32.
        dropout (float): Dropout probability applied to LoRA input features. Defaults to 0.1.
        target_modules (list, optional): List of target sub-module names. Defaults to ["qkv", "proj"].
        quant_type (str): Quantization format. Defaults to 'nf4'.
        double_quant (bool): Whether to use double quantization. Defaults to True.

    Returns:
        nn.Module: Wrapped HuggingFace PeftModel ready for training.

    Example:
        >>> visual_encoder = inject_qlora(model.visual, r=16, alpha=32, target_modules=["qkv", "proj"])
    """
def inject_qlora(
    module: nn.Module, 
    r: int = 16, 
    alpha: int = 32,
    dropout: float = 0.1, 
    target_modules: list = None,
    quant_type: str = 'nf4',
    double_quant: bool = True,
    **kwargs
) -> nn.Module:
    """Apply 4-bit NF4 quantization and inject QLoRA adapters into an encoder module.

    Quantizes linear layers of the module to 4-bit NormalFloat format and wraps it
    with trainable low-rank adaptation matrices on specified target layers.

    Args:
        module (nn.Module): PyTorch module (visual or text encoder).
        r (int): Rank r of LoRA decomposition. Defaults to 16.
        alpha (int): Scaling factor alpha for LoRA updates. Defaults to 32.
        dropout (float): Dropout probability applied to LoRA input features. Defaults to 0.1.
        target_modules (list, optional): List of target sub-module names. Defaults to ["qkv", "proj"].
        quant_type (str): Quantization format. Defaults to 'nf4'.
        double_quant (bool): Whether to use double quantization. Defaults to True.

    Returns:
        nn.Module: Wrapped HuggingFace PeftModel ready for training.

    Example:
        >>> visual_encoder = inject_qlora(model.visual, r=16, alpha=32, target_modules=["qkv", "proj"])
    """
    target_modules = resolve_target_modules(module, target_modules)
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    # Step 1: Quantize linear layers to 4-bit
    print(f"[QLoRA] Quantizing base weights to 4-bit {quant_type.upper()}")
    module = _quantize_linear_layers(
        module,
        quant_type=quant_type,
        compute_dtype=compute_dtype,
        double_quant=double_quant,
    )

    # Step 2: Prepare for k-bit training
    module = prepare_model_for_kbit_training(module, use_gradient_checkpointing=False)

    # Step 3: Inject LoRA adapters
    print(f"[QLoRA] Injecting LoRA adapters (r={r}, alpha={alpha}) on targets: {target_modules}")

    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
        modules_to_save=[],
    )

    peft_module = get_peft_model(module, lora_config)
    print("[QLoRA] QLoRA injection complete.")

    return peft_module
