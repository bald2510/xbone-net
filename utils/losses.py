import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftTargetSemanticMatchingLoss(nn.Module):
    """
    Semantic Matching Loss that creates a soft target matrix based on sharing classes.
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / temperature))

    def forward(self, image_features, text_features, disease_labels):
        # 1. L2 Normalize
        image_features = F.normalize(image_features, dim=-1, p=2)
        text_features = F.normalize(text_features, dim=-1, p=2)
        
        logit_scale = torch.clamp(self.logit_scale.exp(), max=100.0)
        
        # 2. Compute Logits
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logits_per_image.T
        
        # 3. Create Soft Target Matrix (T_ij)
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
                    
        # 4. Normalize T to target distribution Y
        Y_v2t = targets / targets.sum(dim=1, keepdim=True)
        Y_t2v = targets / targets.sum(dim=0, keepdim=True).T
        
        # 5. Compute Soft-target Cross Entropy
        log_preds_v2t = F.log_softmax(logits_per_image, dim=1)
        log_preds_t2v = F.log_softmax(logits_per_text, dim=1)
        
        loss_i = -(Y_v2t * log_preds_v2t).sum(dim=1).mean()
        loss_t = -(Y_t2v * log_preds_t2v).sum(dim=1).mean()
        
        return (loss_i + loss_t) / 2


class InfoNCELoss(nn.Module):
    """
    Standard contrastive InfoNCE loss (like standard CLIP).
    Only diagonal elements are treated as positive pairs.
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / temperature))

    def forward(self, image_features, text_features, disease_labels=None):
        # 1. L2 Normalize
        image_features = F.normalize(image_features, dim=-1, p=2)
        text_features = F.normalize(text_features, dim=-1, p=2)
        
        logit_scale = torch.clamp(self.logit_scale.exp(), max=100.0)
        
        # 2. Calculate Similarity Logits
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logits_per_image.T
        
        # 3. Targets: diagonal indices (0, 1, ..., batch_size-1)
        batch_size = image_features.shape[0]
        labels = torch.arange(batch_size, device=image_features.device)
        
        # 4. Cross Entropy loss for both directions
        loss_i = F.cross_entropy(logits_per_image, labels)
        loss_t = F.cross_entropy(logits_per_text, labels)
        
        return (loss_i + loss_t) / 2


class SigCLIPLoss(nn.Module):
    """
    Sigmoid Loss for Language-Image Pre-training (SigLIP / SigCLIP).
    Optimizes a pairwise binary sigmoid classification loss.
    """
    def __init__(self, temperature=0.07, initial_bias=-10.0):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / temperature))
        self.bias = nn.Parameter(torch.ones([]) * initial_bias)

    def forward(self, image_features, text_features, disease_labels=None):
        # 1. L2 Normalize
        image_features = F.normalize(image_features, dim=-1, p=2)
        text_features = F.normalize(text_features, dim=-1, p=2)
        
        logit_scale = torch.clamp(self.logit_scale.exp(), max=100.0)
        
        # 2. Pairwise logits: t * (x_i^T y_j) + b
        logits = logit_scale * image_features @ text_features.T + self.bias
        
        # 3. Binary labels: 1 on diagonal (positives), -1 off-diagonal (negatives)
        batch_size = image_features.shape[0]
        labels = 2 * torch.eye(batch_size, device=image_features.device) - 1.0
        
        # 4. Sigmoid loss using stable softplus
        loss = F.softplus(-labels * logits).mean()
        
        return loss


# Registry of loss modules
LOSS_REGISTRY = {
    "semantic_matching": SoftTargetSemanticMatchingLoss,
    "infonce": InfoNCELoss,
    "sigclip": SigCLIPLoss
}

def build_loss(loss_type, **kwargs):
    """
    Factory function to instantiate loss modules.
    """
    if loss_type not in LOSS_REGISTRY:
        raise ValueError(f"Loss type '{loss_type}' not supported. Choose from {list(LOSS_REGISTRY.keys())}")
    return LOSS_REGISTRY[loss_type](**kwargs)