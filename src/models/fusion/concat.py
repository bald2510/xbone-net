"""
Concatenation-Based Fusion Module for XBone-Net.
===============================================================================
Concatenates visual and textual feature representations along the channel dimension,
followed by a non-linear bottleneck projection layer to map back to target dimension:
  - Input: img_feats [B, D], txt_feats [B, D]
  - Bottleneck Layer: Linear(2D -> D) + LayerNorm + GELU
  - Output: fused_feats [B, D]
"""

import torch
import torch.nn as nn


# ============================================================
# Concatenation Fusion Strategy
# ============================================================

class ConcatFusion(nn.Module):
    """Concatenation-based multimodal feature fusion module with projection bottleneck.

    Combines image and text feature vectors by concatenating them along feature space
    and projecting back to original dimension via Linear + LayerNorm + GELU layers.

    Attributes:
        projection (nn.Sequential): Sequential bottleneck projection network.

    Example:
        >>> fusion = ConcatFusion(text_dim=512, img_dim=512)
        >>> fused = fusion(img_feats, txt_feats)
    """

    def __init__(self, text_dim: int = 512, img_dim: int = 512, **kwargs):
        """Initialize concatenation fusion bottleneck layers.

        Args:
            text_dim (int): Dimension of input text features. Defaults to 512.
            img_dim (int): Dimension of input image features. Defaults to 512.
            **kwargs: Unused extra keyword arguments.
        """
        super().__init__()
        # --- Bottleneck projection: Linear(2D -> D) + LayerNorm + GELU ---
        self.projection = nn.Sequential(
            nn.Linear(img_dim + text_dim, img_dim),
            nn.LayerNorm(img_dim),
            nn.GELU(),
        )

    def forward(self, img_feats: torch.Tensor, txt_feats: torch.Tensor) -> torch.Tensor:
        """Forward pass merging image and text features via concatenation and projection.

        Mathematical Formulation:
            f_concat = [x_img ; x_txt]
            f_fused = GELU(LayerNorm(W * f_concat + b))

        Args:
            img_feats (torch.Tensor): Image feature embeddings, shape [B, D].
            txt_feats (torch.Tensor): Text feature embeddings, shape [B, D].

        Returns:
            torch.Tensor: Merged fused feature representation, shape [B, D].
        """
        # Step 1: Concatenate along feature dimension [B, 2D]
        combined = torch.cat([img_feats, txt_feats], dim=-1)

        # Step 2: Pass through bottleneck projection [B, D]
        return self.projection(combined)

