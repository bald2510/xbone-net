import torch
import torch.nn as nn
import torch.nn.functional as F

class SemanticMatchingLoss(nn.Module):
    """
    Hàm mất mát Contrastive (InfoNCE) để kéo vector Ảnh về sát với vector Text tương ứng
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * (1 / temperature))

    def forward(self, image_features, text_features):
        # Tính ma trận tương đồng (Cosine similarity)
        logits_per_image = self.logit_scale * image_features @ text_features.T
        logits_per_text = logits_per_image.T
        
        # Nhãn mục tiêu nằm trên đường chéo (positive pairs)
        labels = torch.arange(logits_per_image.shape[0], device=image_features.device)
        
        loss_i = F.cross_entropy(logits_per_image, labels)
        loss_t = F.cross_entropy(logits_per_text, labels)
        
        return (loss_i + loss_t) / 2