"""Cung cấp cơ chế tinh chỉnh hiệu quả tham số   init   cho XBone-Net.

Notes
-----
Mô-đun này thuộc cơ sở mã nguồn nghiên cứu XBone-Net và giữ các quy ước dùng chung của dự án.
"""

import torch.nn as nn
from .lora import inject_lora


# ============================================================
# Thiết lập thành phần dùng chung cho quy trình xử lý của mô-đun.
# ============================================================

def apply_none(module: nn.Module, **kwargs) -> nn.Module:
    """Thực hiện bước apply none trong quy trình hiện tại.

    Parameters
    ----------
    module : nn.Module
        Giá trị ``module`` được sử dụng trong phép xử lý.
    **kwargs : dict
        Các đối số từ khóa bổ sung.

    Returns
    -------
    nn.Module
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    return module


def apply_full_ft(module: nn.Module, **kwargs) -> nn.Module:
    """Thực hiện bước apply full ft trong quy trình hiện tại.

    Parameters
    ----------
    module : nn.Module
        Giá trị ``module`` được sử dụng trong phép xử lý.
    **kwargs : dict
        Các đối số từ khóa bổ sung.

    Returns
    -------
    nn.Module
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    for param in module.parameters():
        param.requires_grad = True
    return module


# ============================================================
# Thiết lập thành phần dùng chung cho quy trình xử lý của mô-đun.
# ============================================================

PEFT_REGISTRY = {
    'none': apply_none,
    'full_ft': apply_full_ft,
    'lora': inject_lora,
}


# ============================================================
# Thiết lập thành phần dùng chung cho quy trình xử lý của mô-đun.
# ============================================================

def apply_peft(module: nn.Module, cfg: dict) -> nn.Module:
    """Thực hiện bước apply peft trong quy trình hiện tại.

    Parameters
    ----------
    module : nn.Module
        Giá trị ``module`` được sử dụng trong phép xử lý.
    cfg : dict
        Cấu hình điều khiển bước xử lý.

    Returns
    -------
    nn.Module
        Kết quả được tạo bởi bước xử lý của hàm.

    Raises
    ------
    ValueError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    """
    peft_type = cfg.get('type', 'none')
    if peft_type not in PEFT_REGISTRY:
        raise ValueError(f"PEFT '{peft_type}' not supported. Choose from {list(PEFT_REGISTRY.keys())}")

    return PEFT_REGISTRY[peft_type](module=module, **(cfg.get('params', {}) or {}))

