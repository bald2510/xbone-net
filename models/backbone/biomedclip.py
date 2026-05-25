import torch
import torch.nn as nn
from open_clip import create_model_and_transforms, get_tokenizer

class BiomedCLIPFoundation(nn.Module):
    def __init__(self, freeze_base=True):
        super().__init__()
        
        # Tên chuẩn của mô hình trên HuggingFace
        model_name = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        
        # OpenCLIP sẽ tự động lo phần config và weights
        self.model, _, self.preprocess = create_model_and_transforms(model_name)
        self.tokenizer = get_tokenizer(model_name)

        # Đóng băng trọng số gốc
        if freeze_base:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, images, input_ids):
        image_features = self.model.encode_image(images)
        text_features = self.model.encode_text(input_ids)
        
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return image_features, text_features