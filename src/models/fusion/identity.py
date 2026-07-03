"""
Identity Pass-Through Fusion Module for XBone-Net.
===============================================================================
Passes image features directly to downstream stages without multimodal fusion.
Used in Phase 1 contrastive pre-training or single-modality baselines.
"""

import torch.nn as nn


# ============================================================
# Identity Fusion Strategy
# ============================================================

class IdentityFusion(nn.Module):
    """Pass-through fusion module returning image features unmodified.

    Used when no feature fusion is required (e.g., during Phase 1 contrastive training
    or when evaluating image-only backbones).

    Example:
        >>> fusion = IdentityFusion()
        >>> fused_feats = fusion(img_feats, txt_feats)  # Returns img_feats
    """

    def __init__(self, **kwargs):
        """Initialize IdentityFusion module.

        Args:
            **kwargs: Unused extra keyword arguments.
        """
        super().__init__()
        
    def forward(self, img_feats, txt_feats=None):
        """Pass through image features.

        Args:
            img_feats (torch.Tensor): Image feature embeddings, shape [B, D].
            txt_feats (torch.Tensor, optional): Text feature embeddings (ignored).

        Returns:
            torch.Tensor: Unmodified image feature tensor, shape [B, D].
        """
        return img_feats
