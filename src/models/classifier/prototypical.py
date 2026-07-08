"""
Prototypical Classification Head Module for XBone-Net.
===============================================================================
Implements prototypical networks with learnable class prototypes (Snell et al., 2017):
  - Class Prototypes: Learnable centroid vectors p_c in R^D for each class c in {1, ..., K}
  - Cosine Metric: L2-normalized cosine similarity on unit hypersphere
  - Temperature Scaling: Temperature scaling factor s to control logit sharpness
  - OOD Support: Enables feature and prototype extraction for Mahalanobis distance OOD detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Prototypical Classifier Head
# ============================================================

class PrototypicalHead(nn.Module):
    """Prototypical classifier head with learnable class centroids.

    Computes class probabilities via cosine similarity to learnable prototype
    vectors on the unit hypersphere.

    Mathematical Formulation:
        logits_c = s * (f . p_c) / (||f||_2 * ||p_c||_2)
        where s is a temperature scaling factor.

    Attributes:
        num_classes (int): Number of target classification categories K.
        scale (float): Cosine similarity temperature multiplier s.
        prototypes (nn.Parameter): Learnable prototype tensor of shape [K, D].

    Example:
        >>> head = PrototypicalHead(feature_dim=512, num_classes=4, scale=10.0)
        >>> feats = torch.randn(16, 512)
        >>> logits = head(feats)
    """

    def __init__(self, feature_dim: int = 512, num_classes: int = 14, scale: float = 5.0):
        """Initialize prototype vectors and scaling factor.

        Args:
            feature_dim (int): Dimensionality D of input feature representations. Defaults to 512.
            num_classes (int): Number of target classification categories K. Defaults to 14.
            scale (float): Temperature scaling factor s applied to cosine logits. Defaults to 5.0.
        """
        super().__init__()
        self.num_classes = num_classes
        self.scale = scale

        # --- Learnable class prototypes [K, D] ---
        self.prototypes = nn.Parameter(torch.randn(num_classes, feature_dim))
        nn.init.xavier_uniform_(self.prototypes)

    def forward(self, features: torch.Tensor, return_features: bool = False):
        """Compute logits from feature embeddings via scaled cosine similarity.

        Args:
            features (torch.Tensor): Fused feature embeddings tensor of shape [B, D].
            return_features (bool): If True, return tuple (logits, features) for prototype
                compactness and OOD margin loss computation. Defaults to False.

        Returns:
            torch.Tensor or tuple:
                - If return_features is False: logits tensor of shape [B, K].
                - If return_features is True: tuple (logits, features).
        """
        # Step 1: L2-normalize features and prototypes onto unit hypersphere
        normed_features = F.normalize(features, dim=-1)
        normed_prototypes = F.normalize(self.prototypes, dim=-1)

        # Step 2: Cosine similarity matrix [B, K] scaled by temperature factor
        cosine_sim = normed_features @ normed_prototypes.T
        logits = cosine_sim * self.scale

        if return_features:
            return logits, features
        return logits