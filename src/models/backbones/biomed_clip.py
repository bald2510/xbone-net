import json
import torch
import torch.nn as nn
from open_clip import create_model_and_transforms, get_tokenizer
from open_clip.factory import HF_HUB_PREFIX, _MODEL_CONFIGS

class BiomedCLIPBackbone(nn.Module):
    def __init__(self, config_path, checkpoint_path, model_name="biomedclip_local"):
        super().__init__()
        
        # Load config từ file json local
        with open(config_path, "r") as f:
            config = json.load(f)
            model_cfg = config["model_cfg"]
            preprocess_cfg = config["preprocess_cfg"]

        # Đăng ký cấu hình vào open_clip registry
        if (not model_name.startswith(HF_HUB_PREFIX)
            and model_name not in _MODEL_CONFIGS):
            _MODEL_CONFIGS[model_name] = model_cfg

        # Khởi tạo model và prepocess
        self.model, _, self.preprocess = create_model_and_transforms(
            model_name=model_name,
            pretrained=checkpoint_path,
            **{f"image_{k}": v for k, v in preprocess_cfg.items()},
        )
        self.tokenizer = get_tokenizer(model_name)

    def forward(self, images, input_ids):
        # Trích xuất feature
        image_features = self.model.encode_image(images)
        text_features = self.model.encode_text(input_ids)
        
        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return image_features, text_features