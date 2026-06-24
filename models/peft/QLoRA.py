"""
QLoRA: Quantized Low-Rank Adaptation for BiomedCLIP encoders.

Quantizes frozen base model weights to 4-bit NF4 precision using bitsandbytes,
then injects full-precision LoRA adapters via PEFT. This achieves near-full-
finetuning accuracy with ~75% less GPU memory than standard LoRA.
"""

import torch
import torch.nn as nn
import bitsandbytes as bnb
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def _quantize_linear_layers(module: nn.Module, quant_type: str = "nf4",
                            compute_dtype: torch.dtype = torch.bfloat16,
                            double_quant: bool = True) -> nn.Module:
    """
    Replace all nn.Linear layers in `module` with bitsandbytes Linear4bit layers.
    Weights are quantized in-place; the module structure is preserved.

    Args:
        module: PyTorch module to quantize.
        quant_type: Quantization type — "nf4" (Normal Float) or "fp4".
        compute_dtype: Dtype for dequantized activations during forward pass.
        double_quant: If True, quantize the quantization constants too (~0.4 bits/param saved).
    """
    replacements = []
    for name, child in module.named_modules():
        if isinstance(child, nn.Linear):
            replacements.append((name, child))

    for name, old_linear in replacements:
        # Navigate to the parent module
        parts = name.split(".")
        parent = module
        for part in parts[:-1]:
            parent = getattr(parent, part)
        attr_name = parts[-1]

        # Create quantized replacement
        has_bias = old_linear.bias is not None
        new_linear = bnb.nn.Linear4bit(
            input_features=old_linear.in_features,
            output_features=old_linear.out_features,
            bias=has_bias,
            compute_dtype=compute_dtype,
            quant_type=quant_type,
            compress_statistics=double_quant,
        )

        # Copy weights (they will be quantized when moved to CUDA)
        new_linear.weight = bnb.nn.Params4bit(
            old_linear.weight.data,
            requires_grad=False,
            quant_type=quant_type,
            compress_statistics=double_quant,
        )
        if has_bias:
            new_linear.bias = nn.Parameter(old_linear.bias.data, requires_grad=False)

        setattr(parent, attr_name, new_linear)

    n_quantized = len(replacements)
    print(f"[QLoRA] Quantized {n_quantized} nn.Linear layers to 4-bit {quant_type.upper()}.")
    return module


def inject_qlora(module: nn.Module, r: int = 16, alpha: int = 32,
                 dropout: float = 0.1, target_modules: list = None,
                 quant_type: str = "nf4", double_quant: bool = True) -> nn.Module:
    """
    Apply QLoRA to a module:
      1. Quantize all nn.Linear layers to 4-bit NF4
      2. Inject LoRA adapters on the specified target modules

    Args:
        module: The encoder module (e.g. backbone.model.visual or backbone.model.text.transformer).
        r: LoRA rank.
        alpha: LoRA scaling factor (recommended: alpha = 2 * r).
        dropout: LoRA dropout probability.
        target_modules: List of layer name patterns to attach LoRA adapters to.
        quant_type: Quantization type — "nf4" or "fp4".
        double_quant: Whether to apply double quantization on quantization constants.

    Returns:
        PEFT-wrapped module with quantized base weights and LoRA adapters.
    """
    if target_modules is None:
        target_modules = ["qkv", "proj"]

    # Detect compute dtype from GPU capability
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    # Step 1: Quantize all Linear layers to 4-bit
    print(f"[QLoRA] Step 1/3: Quantizing base weights to 4-bit {quant_type.upper()} "
          f"(double_quant={double_quant}, compute_dtype={compute_dtype})")
    module = _quantize_linear_layers(
        module,
        quant_type=quant_type,
        compute_dtype=compute_dtype,
        double_quant=double_quant,
    )

    # Step 2: Prepare the quantized model for k-bit training
    # This sets requires_grad=False on all params and handles gradient checkpointing compatibility
    print("[QLoRA] Step 2/3: Preparing quantized model for k-bit training...")
    module = prepare_model_for_kbit_training(module, use_gradient_checkpointing=False)

    # Step 3: Inject LoRA adapters on specified target modules
    print(f"[QLoRA] Step 3/3: Injecting LoRA adapters (r={r}, alpha={alpha}) on targets: {target_modules}")

    # Debug: show which layers will be targeted
    for name, sub_module in module.named_modules():
        if any(target in name for target in target_modules):
            print(f"  [QLoRA Target] {name}")

    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
        modules_to_save=[],
    )

    peft_module = get_peft_model(module, lora_config)
    print("[QLoRA] Tích hợp QLoRA thành công!")

    return peft_module


def inject_lora(module: nn.Module, r: int = 16, alpha: int = 32,
                dropout: float = 0.1, target_modules: list = None, **kwargs) -> nn.Module:
    """
    Apply standard LoRA (no quantization) to a module.
    """
    if target_modules is None:
        target_modules = ["qkv", "proj"]

    print(f"[LoRA] Step 1/2: Preparing base model parameters for LoRA adapter injection...")
    for param in module.parameters():
        param.requires_grad = False

    print(f"[LoRA] Step 2/2: Injecting LoRA adapters (r={r}, alpha={alpha}) on targets: {target_modules}")
    for name, sub_module in module.named_modules():
        if any(target in name for target in target_modules):
            print(f"  [LoRA Target] {name}")

    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
        modules_to_save=[],
    )

    peft_module = get_peft_model(module, lora_config)
    print("[LoRA] Tích hợp LoRA thành công!")

    return peft_module
