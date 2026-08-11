"""Cung cấp cơ chế tinh chỉnh hiệu quả tham số lora cho XBone-Net.

Notes
-----
Mô-đun này thuộc cơ sở mã nguồn nghiên cứu XBone-Net và giữ các quy ước dùng chung của dự án.
"""

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

def resolve_target_modules(module: nn.Module, requested_targets: list | None = None) -> list:
    """Xác định target modules cho bước xử lý hiện tại.

    Parameters
    ----------
    module : nn.Module
        Giá trị ``module`` được sử dụng trong phép xử lý.
    requested_targets : list | None, optional
        Giá trị ``requested_targets`` được sử dụng trong phép xử lý.

    Returns
    -------
    list
        Kết quả được tạo bởi bước xử lý của hàm.
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
    """Thực hiện bước inject lora trong quy trình hiện tại.

    Parameters
    ----------
    module : nn.Module
        Giá trị ``module`` được sử dụng trong phép xử lý.
    r : int, optional
        Giá trị ``r`` được sử dụng trong phép xử lý.
    alpha : int, optional
        Giá trị ``alpha`` được sử dụng trong phép xử lý.
    dropout : float, optional
        Giá trị ``dropout`` được sử dụng trong phép xử lý.
    target_modules : list, optional
        Nhãn hoặc chỉ số lớp liên quan.
    **kwargs : dict
        Các đối số từ khóa bổ sung.

    Returns
    -------
    nn.Module
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    target_modules = resolve_target_modules(module, target_modules)

    # Thiết lập trạng thái và thống kê các tham số mô hình.
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
