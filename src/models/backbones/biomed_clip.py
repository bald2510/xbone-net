import torch
import torch.nn as nn
from open_clip import create_model_and_transforms, get_tokenizer
from peft import LoraConfig, get_peft_model

class BiomedCLIPBackbone(nn.Module):
    def __init__(self, use_lora=True):
        super().__init__()
        
        # Tên chuẩn của mô hình trên HuggingFace
        model_name = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        
        # OpenCLIP sẽ tự động lo phần config và weights
        self.model, _, self.preprocess = create_model_and_transforms(model_name)
        self.tokenizer = get_tokenizer(model_name)

        # Đóng băng trọng số gốc
        for param in self.model.parameters():
            param.requires_grad = False

        # Tích hợp LoRA cho Image Encoder (ViT)
        if use_lora:
            lora_config = LoraConfig(
                r=8, 
                lora_alpha=16,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.1,
                bias="none",
                modules_to_save=[]
            )
            self.model.visual = get_peft_model(self.model.visual, lora_config)
            print("Đã tích hợp LoRA vào Image Encoder thành công!")

    def forward(self, images, input_ids):
        image_features = self.model.encode_image(images)
        text_features = self.model.encode_text(input_ids)
        
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return image_features, text_features