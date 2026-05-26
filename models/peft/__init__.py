import torch.nn as nn
from .LoRA import inject_lora
# from .kfac_utils import inject_kfac  # Bổ sung module K-FAC sau

def apply_none(module: nn.Module, **kwargs) -> nn.Module:
    """
    Dummy PEFT injection.
    Không làm gì cả, trả về đúng mô hình gốc (thích hợp cho Zero-shot baseline).
    """
    return module

# ==========================================
# PEFT REGISTRY
# Lưu trữ các hàm biến đổi mô hình
# ==========================================
PEFT_REGISTRY = {
    'none': apply_none,
    'lora': inject_lora,
    # 'kfac': inject_kfac
}

def apply_peft(module: nn.Module, cfg: dict) -> nn.Module:
    """
    Hàm Factory nhận vào một module (thường là image encoder) 
    và cấu hình để áp dụng kỹ thuật PEFT tương ứng.
    """
    peft_type = cfg.get('type', 'none')
    
    if peft_type not in PEFT_REGISTRY:
        raise ValueError(f"Kỹ thuật PEFT '{peft_type}' chưa được hỗ trợ. Hãy thêm vào PEFT_REGISTRY.")
        
    # Lấy hàm tiêm PEFT từ Registry
    inject_fn = PEFT_REGISTRY[peft_type]
    
    # Thực thi hàm tiêm với các tham số từ config (r, alpha, dropout...)
    modified_module = inject_fn(module=module, **cfg.get('params', {}))
    
    return modified_module