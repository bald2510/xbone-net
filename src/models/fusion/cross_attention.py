"""Bi-directional cross-attention fusion for XBone-Net."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class CrossAttentionFusion(nn.Module):
    """Fuse global image/text tokens through two cross-attention directions."""

    supports_padding_mask = True

    def __init__(
        self,
        img_dim: int = 512,
        text_dim: int = 512,
        embed_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        img_dim = kwargs.get("img_dim", img_dim)
        text_dim = kwargs.get("text_dim", text_dim)

        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")

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

    @staticmethod
    def _validate_mask(
        mask: Optional[torch.Tensor],
        batch_size: int,
        seq_len: int,
        name: str,
    ) -> Optional[torch.Tensor]:
        if mask is None:
            return None
        mask = mask.to(dtype=torch.bool)
        if mask.shape != (batch_size, seq_len):
            raise ValueError(
                f"{name} must have shape {(batch_size, seq_len)}, got {tuple(mask.shape)}"
            )
        if torch.any(mask.all(dim=1)):
            raise ValueError(f"{name} masks every key token for at least one sample.")
        return mask

    def forward(
        self,
        img_feats: torch.Tensor,
        txt_feats: torch.Tensor,
        img_key_padding_mask: Optional[torch.Tensor] = None,
        txt_key_padding_mask: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ):
        if img_feats.ndim == 2:
            img_feats = img_feats.unsqueeze(1)
        if txt_feats.ndim == 2:
            txt_feats = txt_feats.unsqueeze(1)
        if img_feats.ndim != 3 or txt_feats.ndim != 3:
            raise ValueError("Cross-attention expects [B,T,D] or [B,D] inputs.")
        if img_feats.size(0) != txt_feats.size(0):
            raise ValueError("Image and text batch sizes must match.")

        img_feats = self.img_proj(img_feats)
        txt_feats = self.txt_proj(txt_feats)

        img_global = img_feats[:, 0:1, :]
        txt_global = txt_feats[:, 0:1, :]
        img_local = img_feats[:, 1:, :]
        txt_local = txt_feats[:, 1:, :]

        # Global-only backbones use their global token as the key/value sequence.
        if img_local.size(1) == 0:
            img_local = img_global
            img_key_padding_mask = None
        if txt_local.size(1) == 0:
            txt_local = txt_global
            txt_key_padding_mask = None

        batch_size = img_feats.size(0)
        img_key_padding_mask = self._validate_mask(
            img_key_padding_mask,
            batch_size,
            img_local.size(1),
            "img_key_padding_mask",
        )
        txt_key_padding_mask = self._validate_mask(
            txt_key_padding_mask,
            batch_size,
            txt_local.size(1),
            "txt_key_padding_mask",
        )

        txt_ctx, attn_i2t = self.img_to_txt_attn(
            query=img_global,
            key=txt_local,
            value=txt_local,
            key_padding_mask=txt_key_padding_mask,
            need_weights=return_attn,
            average_attn_weights=False,
        )
        img_ctx, attn_t2i = self.txt_to_img_attn(
            query=txt_global,
            key=img_local,
            value=img_local,
            key_padding_mask=img_key_padding_mask,
            need_weights=return_attn,
            average_attn_weights=False,
        )

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