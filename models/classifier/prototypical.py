import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypicalHead(nn.Module):
    """
    Prototypical classifier head with learnable class prototypes.

    Returns logits based on cosine similarity between features and prototypes.
    When return_features=True, also returns (logits, features) for prototype/OOD losses.
    """

    def __init__(self, feature_dim: int = 512, num_classes: int = 14, scale: float = 10.0):
        super().__init__()
        self.num_classes = num_classes
        self.scale = scale

        # Learnable class prototypes [K, D]
        self.prototypes = nn.Parameter(torch.randn(num_classes, feature_dim))
        nn.init.xavier_uniform_(self.prototypes)

    def forward(self, features, return_features: bool = False):
        """
        Args:
            features: Fused features [B, D]
            return_features: If True, returns (logits, features) tuple for loss computation
        Returns:
            logits [B, K] or (logits, features) if return_features=True
        """
        # L2 normalize for cosine similarity
        normed_features = F.normalize(features, dim=-1)
        normed_prototypes = F.normalize(self.prototypes, dim=-1)

        # Cosine similarity [B, K], scaled
        cosine_sim = normed_features @ normed_prototypes.T
        logits = cosine_sim * self.scale

        if return_features:
            return logits, features
        return logits