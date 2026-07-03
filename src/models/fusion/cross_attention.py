"""
Bi-Directional Cross-Attention Fusion Module for XBone-Net.
===============================================================================
Implements a dual-stream cross-attention mechanism for fine-grained multimodal interaction:
  - Top Branch (Image -> Text): Image queries text to identify relevant tokens
  - Bottom Branch (Text -> Image): Text queries image to highlight relevant visual regions
  - Transformer Block: Single-layer transformer encoder for joint sequence modeling
  - Pooling: Mean pooling over concatenated sequence to yield global multimodal embedding [B, D]
"""

import torch
import torch.nn as nn


# ============================================================
# Bi-Directional Cross-Attention Fusion
# ============================================================

class CrossAttentionFusion(nn.Module):
    """Bi-directional cross-attention fusion module with joint transformer encoder.

    Assembles dual cross-attention branches (Image->Text and Text->Image) followed by a
    single-layer Transformer encoder block to model non-linear joint interactions.

    Mathematical Formulation:
        Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
        Top Branch: Q = Image, K,V = Text
        Bottom Branch: Q = Text, K,V = Image
        fused_global = mean(TransformerEncoder([Top_Out ; Bottom_Out]))

    Attributes:
        embed_dim (int): Unified feature dimensionality (e.g., 512).
        top_cross_attn (nn.MultiheadAttention): MHA module for Image->Text.
        top_linear (nn.Linear): Linear projection layer for top branch.
        bottom_cross_attn (nn.MultiheadAttention): MHA module for Text->Image.
        bottom_linear (nn.Linear): Linear projection layer for bottom branch.
        transformer_block (nn.TransformerEncoder): Transformer block for joint sequence processing.

    Example:
        >>> fusion = CrossAttentionFusion(img_dim=512, text_dim=512, num_heads=8)
        >>> fused_feats = fusion(img_feats, txt_feats)
    """

    def __init__(self, text_dim: int = 512, img_dim: int = 512, num_heads: int = 8, **kwargs):
        """Initialize cross-attention fusion parameters.

        Args:
            text_dim (int): Input dimensionality of text features. Defaults to 512.
            img_dim (int): Input dimensionality of image features. Defaults to 512.
            num_heads (int): Number of attention heads for multi-head attention. Defaults to 8.
            **kwargs: Unused extra keyword arguments.
        """
        super().__init__()
        self.embed_dim = img_dim 

        # --- Top branch: Query=Image, Key/Value=Text (Image attends to Text) ---
        self.top_cross_attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim, num_heads=num_heads, batch_first=True
        )
        self.top_linear = nn.Linear(self.embed_dim, self.embed_dim)

        # --- Bottom branch: Query=Text, Key/Value=Image (Text attends to Image) ---
        self.bottom_cross_attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim, num_heads=num_heads, batch_first=True
        )
        self.bottom_linear = nn.Linear(self.embed_dim, self.embed_dim)

        # --- Transformer Encoder block for joint sequence interaction ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim, 
            nhead=num_heads, 
            dim_feedforward=self.embed_dim * 4,
            batch_first=True
        )
        self.transformer_block = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def forward(self, img_feats: torch.Tensor, txt_feats: torch.Tensor, return_attn: bool = False):
        """Fuse image and text features via bi-directional cross-attention.

        Args:
            img_feats (torch.Tensor): Image feature tensor, shape [B, D] or [B, N_img, D].
            txt_feats (torch.Tensor): Text feature tensor, shape [B, D] or [B, N_txt, D].
            return_attn (bool): If True, return attention maps dict alongside fused tensor.

        Returns:
            torch.Tensor or tuple:
                - If return_attn is False: fused_global tensor of shape [B, D].
                - If return_attn is True: tuple (fused_global, attn_info) where attn_info is dict.
        """
        # Ensure 3D sequence shape [B, L, D] for multi-head attention
        if img_feats.dim() == 2:
            img_feats = img_feats.unsqueeze(1)
        if txt_feats.dim() == 2:
            txt_feats = txt_feats.unsqueeze(1)

        # --- Top branch: Image attends to Text ---
        top_out, top_attn_weights = self.top_cross_attn(
            query=img_feats, key=txt_feats, value=txt_feats,
            need_weights=True, average_attn_weights=False
        )
        top_out = self.top_linear(top_out)

        # --- Bottom branch: Text attends to Image ---
        bottom_out, bottom_attn_weights = self.bottom_cross_attn(
            query=txt_feats, key=img_feats, value=img_feats,
            need_weights=True, average_attn_weights=False
        )
        bottom_out = self.bottom_linear(bottom_out)

        # --- Joint interaction through Transformer encoder ---
        combined_seq = torch.cat([top_out, bottom_out], dim=1)
        fused_seq = self.transformer_block(combined_seq)

        # --- Mean pooling over sequence length ---
        fused_global = fused_seq.mean(dim=1)
        
        if return_attn:
            attn_info = {
                "top_attn_weights": top_attn_weights,
                "bottom_attn_weights": bottom_attn_weights,
                "top_out": top_out.squeeze(1),
                "bottom_out": bottom_out.squeeze(1),
                "combined_seq": combined_seq,
                "fused_seq": fused_seq,
            }
            return fused_global, attn_info

        return fused_global

