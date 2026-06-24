import torch
import torch.nn as nn


class ConcatFusion(nn.Module):
    """
    Concatenation-based fusion: concatenates image and text features,
    then projects back to the original feature dimension via a linear layer.

    Input:  img_feats [B, D], txt_feats [B, D]
    Output: fused [B, D]
    """

    def __init__(self, text_dim=512, img_dim=512, **kwargs):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(img_dim + text_dim, img_dim),
            nn.LayerNorm(img_dim),
            nn.GELU(),
        )

    def forward(self, img_feats, txt_feats):
        combined = torch.cat([img_feats, txt_feats], dim=-1)  # [B, 2D]
        return self.projection(combined)  # [B, D]
