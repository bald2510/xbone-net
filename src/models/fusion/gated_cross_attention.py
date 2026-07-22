"""Sample-adaptive gated cross-attention fusion for XBone-Net."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .cross_attention import CrossAttentionFusion


class GatedCrossAttentionFusion(CrossAttentionFusion):
    """Interpolate a direct concat path with cross-attention per sample.

    The direct path provides a conservative multimodal representation, while
    the cross-attention path is allowed to modify it only to the extent selected
    by a learned scalar gate:

        z = z_concat + sigmoid(g(x)) * (z_cross - z_concat)

    The final gate layer is initialized with zero weights and a negative bias,
    so Phase 2 starts close to concat instead of immediately depending on a
    randomly initialized cross-attention branch.
    """

    supports_padding_mask = True

    def __init__(
        self,
        img_dim: int = 512,
        text_dim: int = 512,
        embed_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        attention_direction: str = "bidirectional",
        gate_hidden_dim: int = 128,
        gate_bias_init: float = -2.0,
        **kwargs,
    ) -> None:
        if gate_hidden_dim < 1:
            raise ValueError("gate_hidden_dim must be positive.")
        super().__init__(
            img_dim=img_dim,
            text_dim=text_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            attention_direction=attention_direction,
            **kwargs,
        )
        self.gate_bias_init = float(gate_bias_init)

        # Match the existing concat baseline: pool the original visual token
        # sequence, retain the global text token, then project [image; text].
        self.concat_projection = nn.Sequential(
            nn.Linear(img_dim + text_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )
        self.gate_network = nn.Sequential(
            nn.Linear(embed_dim * 2, gate_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden_dim, 1),
        )
        nn.init.zeros_(self.gate_network[-1].weight)
        nn.init.constant_(self.gate_network[-1].bias, self.gate_bias_init)

    @staticmethod
    def _pool_image_tokens(
        image_tokens: torch.Tensor,
        local_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Pool global and valid local image tokens for the concat branch."""
        if image_tokens.ndim == 2:
            return image_tokens
        if image_tokens.ndim != 3:
            raise ValueError(
                "Gated fusion expects image features with shape [B,T,D] or "
                f"[B,D], got {tuple(image_tokens.shape)}."
            )
        if local_padding_mask is None:
            return image_tokens.mean(dim=1)
        expected_shape = (image_tokens.size(0), image_tokens.size(1) - 1)
        local_padding_mask = local_padding_mask.to(
            device=image_tokens.device,
            dtype=torch.bool,
        )
        if tuple(local_padding_mask.shape) != expected_shape:
            raise ValueError(
                "img_key_padding_mask must describe local image tokens with "
                f"shape {expected_shape}, got {tuple(local_padding_mask.shape)}."
            )
        global_valid = torch.ones(
            (image_tokens.size(0), 1),
            dtype=torch.bool,
            device=image_tokens.device,
        )
        valid = torch.cat([global_valid, ~local_padding_mask], dim=1)
        weights = valid.unsqueeze(-1).to(image_tokens.dtype)
        return (image_tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def forward(
        self,
        img_feats: torch.Tensor,
        txt_feats: torch.Tensor,
        img_key_padding_mask: Optional[torch.Tensor] = None,
        txt_key_padding_mask: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ):
        image_vector = self._pool_image_tokens(
            img_feats,
            img_key_padding_mask,
        )
        if txt_feats.ndim == 3:
            text_vector = txt_feats[:, 0, :]
        elif txt_feats.ndim == 2:
            text_vector = txt_feats
        else:
            raise ValueError(
                "Gated fusion expects text features with shape [B,T,D] or "
                f"[B,D], got {tuple(txt_feats.shape)}."
            )
        if image_vector.size(0) != text_vector.size(0):
            raise ValueError("Image and text batch sizes must match.")

        concat_fused = self.concat_projection(
            torch.cat([image_vector, text_vector], dim=-1)
        )
        cross_output = super().forward(
            img_feats,
            txt_feats,
            img_key_padding_mask=img_key_padding_mask,
            txt_key_padding_mask=txt_key_padding_mask,
            return_attn=return_attn,
        )
        if return_attn:
            cross_fused, attention = cross_output
        else:
            cross_fused = cross_output
        gate = torch.sigmoid(
            self.gate_network(torch.cat([concat_fused, cross_fused], dim=-1))
        )
        # Non-persistent diagnostic used by evaluation/inference. Keeping only
        # the latest detached batch does not alter checkpoints or autograd.
        self.last_gate = gate.detach()
        fused = concat_fused + gate * (cross_fused - concat_fused)

        if return_attn:
            attention.update(
                {
                    "fusion_gate": gate,
                    "concat_branch": concat_fused,
                    "cross_attention_branch": cross_fused,
                }
            )
            return fused, attention
        return fused
