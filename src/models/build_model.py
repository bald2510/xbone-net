import torch.nn as nn
from .backbones.biomed_clip import BiomedCLIPBackbone
from .components.fusion import FiLMFusion
from .components.classifier import PrototypicalHead
from .components.losses import SoftTargetSemanticMatchingLoss

class XBoneMultiModalModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # 1. Khởi tạo Backbone với LoRA
        self.backbone = BiomedCLIPBackbone(
            use_lora=True
        )
        
        # 2. Khởi tạo FiLM Fusion Module
        self.fusion_layer = FiLMFusion(text_dim=512, img_dim=512)
        
        # 3. Khởi tạo Prototypical Head
        self.classifier = PrototypicalHead(
            feature_dim=512, # Đầu ra của FiLM có số chiều bằng với img_dim
            num_classes=config['model']['num_classes']
        )
        
        # 4. Khởi tạo Loss cho Giai đoạn 1
        self.alignment_loss_fn = SoftTargetSemanticMatchingLoss(temperature=0.07)

    def forward(self, images, input_ids, labels=None, phase="tuning"):
        """
        phase='alignment': Giai đoạn 1 - Train LoRA bằng Contrastive Loss
        phase='tuning': Giai đoạn 2 - Train FiLM và Prototypes
        phase='online': Suy diễn thực tế để bắt OOD
        """
        # Trích xuất đặc trưng
        img_feats, txt_feats = self.backbone(images, input_ids)
        
        if phase == "alignment":
            # Đảm bảo bạn đang gọi hàm SoftTargetSemanticMatchingLoss
            # Truyền thêm labels vào hàm loss
            if labels is None:
                raise ValueError("Giai đoạn Alignment với Soft-target yêu cầu phải có 'labels'.")
                
            loss = self.alignment_loss_fn(img_feats, txt_feats, labels)
            return loss
            
        elif phase == "tuning":
            # Giai đoạn 2: Bơm text để điều chế ảnh
            combined = self.fusion_layer(img_feats, txt_feats)
            logits, _ = self.classifier(combined)
            return logits
            
        elif phase == "online":
            # Giai đoạn Online: Trả về Cosine Similarity để làm OOD Detection
            combined = self.fusion_layer(img_feats, txt_feats)
            _, cosine_sim = self.classifier(combined)
            return cosine_sim