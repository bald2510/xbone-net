"""
Identity Pass-Through Classifier Head for XBone-Net.
===============================================================================
Returns input features directly without applying classification projection.
Used during Phase 1 contrastive pre-training to yield raw feature vectors.
"""

import torch.nn as nn


# ============================================================
# Identity Classifier Head
# ============================================================

class IdentityHead(nn.Module):
    """Pass-through classifier head returning input features unmodified.

    Used when raw embedding representations are required (e.g., Phase 1 contrastive
    learning or feature extraction pipelines).

    Example:
        >>> head = IdentityHead()
        >>> feats = head(features)  # Returns features unchanged
    """

    def __init__(self, **kwargs):
        """Initialize IdentityHead instance.

        Args:
            **kwargs: Unused extra keyword arguments.
        """
        super().__init__()
        
    def forward(self, features):
        """Pass through input feature embeddings.

        Args:
            features (torch.Tensor): Feature embedding tensor, shape [B, D].

        Returns:
            torch.Tensor: Unmodified input feature tensor, shape [B, D].
        """
        return features
