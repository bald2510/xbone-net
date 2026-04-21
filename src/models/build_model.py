import torch.nn as nn
from .backbones.biomed_clip import BiomedCLIPBackbone
from .components.fusion import ConcatFusion
from .components.classifier import MedicalHead # Import file mới tách

class MyMultiModalModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # 1. Backbone (Load từ json/bin)
        self.backbone = BiomedCLIPBackbone(
            config_path=config['model']['config_path'], 
            checkpoint_path=config['model']['checkpoint_path']
        )
        
        # 2. Fusion (Concat)
        self.fusion_layer = ConcatFusion()
        
        # 3. Classifier (Tách riêng)
        self.classifier = MedicalHead(
            input_dim=1024, # 512 (img) + 512 (text)
            num_classes=config['model']['num_classes'],
            hidden_dim=config['model']['hidden_dim'],
            dropout_rate=config['model']['dropout']
        )

    def forward(self, images, input_ids):
        # Feature extraction
        img_feats, txt_feats = self.backbone(images, input_ids)
        
        # Fusion
        combined = self.fusion_layer(img_feats, txt_feats)
        
        # Classification
        logits = self.classifier(combined)
        return logits