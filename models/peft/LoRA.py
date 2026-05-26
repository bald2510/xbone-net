import torch.nn as nn
from peft import LoraConfig, get_peft_model

def inject_lora(module: nn.Module, r: int = 8, alpha: int = 16, dropout: float = 0.1) -> nn.Module:
    """
    Tiêm module LoRA vào các lớp attention (qkv) và projection (proj) của mô hình.
    """
    lora_config = LoraConfig(
        r=r, 
        lora_alpha=alpha,
        target_modules=["qkv", "proj"],
        lora_dropout=dropout,
        bias="none",
        modules_to_save=[]
    )

    # In ra để debug xem đã bắt đúng các layer cần thiết chưa
    for name, sub_module in module.named_modules():
        if "proj" in name or "qkv" in name:
            print(f"[LoRA Debug] Target layer found: {name}")

    # Áp dụng LoRA vào module
    peft_module = get_peft_model(module, lora_config)
    print("Đã tích hợp LoRA thành công!")
    
    return peft_module