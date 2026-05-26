import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftTargetSemanticMatchingLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / temperature))

    def forward(self, image_features, text_features, disease_labels):
        # 1. Chuẩn hóa L2
        image_features = F.normalize(image_features, dim=-1, p=2)
        text_features = F.normalize(text_features, dim=-1, p=2)
        
        logit_scale = torch.clamp(self.logit_scale.exp(), max=100.0)
        
        # 2. Tính ma trận Logits (S_ij trong bài báo)
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logits_per_image.T
        
        # 3. TẠO MA TRẬN SOFT TARGET (T_ij)
        batch_size = image_features.shape[0]
        targets = torch.zeros((batch_size, batch_size), device=image_features.device)
        
        for i in range(batch_size):
            for j in range(batch_size):
                if i == j:
                    targets[i, j] = 1.0  # Paired image-text
                elif torch.equal(disease_labels[i], disease_labels[j]):
                    targets[i, j] = 0.95 # Unpaired but same-class
                else:
                    targets[i, j] = 0.0  # Different class
                    
        # 4. Chuẩn hóa ma trận T thành phân phối mục tiêu Y (Công thức 4)
        Y_v2t = targets / targets.sum(dim=1, keepdim=True)
        Y_t2v = targets / targets.sum(dim=0, keepdim=True).T
        
        # 5. Tính Cross Entropy Soft-target bằng KLDivLoss hoặc tính tay (Công thức 5)
        # Sử dụng log_softmax cho logits
        log_preds_v2t = F.log_softmax(logits_per_image, dim=1)
        log_preds_t2v = F.log_softmax(logits_per_text, dim=1)
        
        loss_i = -(Y_v2t * log_preds_v2t).sum(dim=1).mean()
        loss_t = -(Y_t2v * log_preds_t2v).sum(dim=1).mean()
        
        return (loss_i + loss_t) / 2