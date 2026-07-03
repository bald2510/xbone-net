"""
Linear Classification Head Module for XBone-Net.
===============================================================================
Maps input feature representations to class logits using a standard linear projection:
  - Input: fused_feats [B, D]
  - Transformation: Linear(D -> K)
  - Output: logits [B, K]
"""

import torch.nn as nn


# ============================================================
# Linear Classifier Head
# ============================================================

class LinearHead(nn.Module):
    """Linear classification head mapping fused feature vectors to class logits.

    Attributes:
        classifier (nn.Linear): Fully connected linear layer projecting from feature_dim to num_classes.

    Example:
        >>> head = LinearHead(feature_dim=512, num_classes=5)
        >>> logits = head(fused_features)
    """

    def __init__(self, feature_dim: int = 512, num_classes: int = 10):
        """Initialize linear classifier head.

        Args:
            feature_dim (int): Dimensionality of input feature vectors. Defaults to 512.
            num_classes (int): Number of target output classes. Defaults to 10.
        """
        super().__init__()
        # --- Linear transformation layer: y = W * x + b ---
        self.classifier = nn.Linear(feature_dim, num_classes)
        
    def forward(self, x):
        """Map feature embeddings to class logits.

        Mathematical Formulation:
            logits = W * x + b

        Args:
            x (torch.Tensor): Fused feature tensor of shape [B, D].

        Returns:
            torch.Tensor: Unnormalized class logits tensor of shape [B, K].
        """
        return self.classifier(x)
