"""
Bi-Directional Cross-Attention Fusion Module for XBone-Net.
===============================================================================
Implements a dual-stream cross-attention mechanism for fine-grained multimodal interaction:
  - Top Branch (Image -> Text): Image queries text to identify relevant tokens
  - Bottom Branch (Text -> Image): Text queries image to highlight relevant visual regions
  - Transformer Block: MLP Fusion with LayerNorm and Residuals
"""

import torch
import torch.nn as nn

# ============================================================
# Bi-Directional Cross-Attention Fusion
# ============================================================

class CrossAttentionFusion(nn.Module):
    """Bi-directional cross-attention fusion module with post-LN residuals and MLP.

    Assembles dual cross-attention branches (Image->Text and Text->Image) and concatenates 
    the context vectors with the original global representations. Fuses the output 
    through a non-linear MLP.

    Attributes:
        img_proj (nn.Linear): Linear projection for image features.
        txt_proj (nn.Linear): Linear projection for text features.
        img_to_txt_attn (nn.MultiheadAttention): MHA module for Image->Text.
        txt_to_img_attn (nn.MultiheadAttention): MHA module for Text->Image.
        fusion_mlp (nn.Sequential): Non-linear MLP mixer for fusion.
    """
    def __init__(
        self,
        img_dim: int = 512,
        text_dim: int = 512,
        embed_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()

        # If img_dim is not explicitly provided in kwargs but via kwargs, handle it.
        # Ensure it works gracefully with kwargs.
        img_dim = kwargs.get('img_dim', img_dim)
        text_dim = kwargs.get('text_dim', text_dim)

        self.img_proj = nn.Linear(img_dim, embed_dim)
        self.txt_proj = nn.Linear(text_dim, embed_dim)

        self.img_to_txt_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.txt_to_img_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm_img = nn.LayerNorm(embed_dim)
        self.norm_txt = nn.LayerNorm(embed_dim)
        self.norm_fuse = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(dropout)

        self.fusion_mlp = nn.Sequential(
            nn.Linear(embed_dim * 4, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(
        self,
        img_feats,
        txt_feats,
        img_key_padding_mask=None,
        txt_key_padding_mask=None,
        return_attn=False,
    ):
        # [B, D] -> [B, 1, D]
        if img_feats.dim() == 2:
            img_feats = img_feats.unsqueeze(1)
        if txt_feats.dim() == 2:
            txt_feats = txt_feats.unsqueeze(1)

        img_feats = self.img_proj(img_feats)
        txt_feats = self.txt_proj(txt_feats)

        img_global = img_feats[:, 0:1, :]
        txt_global = txt_feats[:, 0:1, :]

        img_local = img_feats[:, 1:, :]
        txt_local = txt_feats[:, 1:, :]

        # fallback nếu encoder chỉ trả global embedding
        if img_local.size(1) == 0:
            img_local = img_global
        if txt_local.size(1) == 0:
            txt_local = txt_global

        # Image global attends to local text tokens
        txt_ctx, attn_i2t = self.img_to_txt_attn(
            query=img_global,
            key=txt_local,
            value=txt_local,
            key_padding_mask=txt_key_padding_mask,
            need_weights=return_attn,
            average_attn_weights=False,
        )

        # Text global attends to local image patches
        img_ctx, attn_t2i = self.txt_to_img_attn(
            query=txt_global,
            key=img_local,
            value=img_local,
            key_padding_mask=img_key_padding_mask,
            need_weights=return_attn,
            average_attn_weights=False,
        )

        # Residual + LayerNorm, không phải gated
        txt_ctx = self.norm_img(img_global + self.dropout(txt_ctx))
        img_ctx = self.norm_txt(txt_global + self.dropout(img_ctx))

        fused = torch.cat(
            [
                img_global.squeeze(1),
                txt_global.squeeze(1),
                txt_ctx.squeeze(1),
                img_ctx.squeeze(1),
            ],
            dim=-1,
        )

        fused = self.norm_fuse(self.fusion_mlp(fused))

        if return_attn:
            return fused, {
                "attn_img_to_txt": attn_i2t,
                "attn_txt_to_img": attn_t2i,
                "img_global": img_global.squeeze(1),
                "txt_global": txt_global.squeeze(1),
                "txt_context_for_image": txt_ctx.squeeze(1),
                "img_context_for_text": img_ctx.squeeze(1),
            }

        return fused
