import torch.nn as nn
from peft import LoraConfig, get_peft_model

def inject_lora(module: nn.Module, r: int = 8, alpha: int = 16, dropout: float = 0.1, target_modules: list = None) -> nn.Module:
    """
    Tiêm module LoRA vào các lớp attention và projection được cấu hình của mô hình.
    """
    if target_modules is None:
        target_modules = ["qkv", "proj"]
        
    lora_config = LoraConfig(
        r=r, 
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
        modules_to_save=[]
    )

    # In ra để debug xem đã bắt đúng các layer cần thiết chưa
    for name, sub_module in module.named_modules():
        if any(target in name for target in target_modules):
            print(f"[LoRA Debug] Target layer found: {name}")

    # Áp dụng LoRA vào module
    peft_module = get_peft_model(module, lora_config)
    print("Đã tích hợp LoRA thành công!")
    
    return peft_module