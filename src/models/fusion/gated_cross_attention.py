"""Cung cấp cơ chế dung hợp đa phương thức gated cross attention cho XBone-Net.

Notes
-----
Mô-đun này thuộc cơ sở mã nguồn nghiên cứu XBone-Net và giữ các quy ước dùng chung của dự án.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .cross_attention import CrossAttentionFusion


class GatedCrossAttentionFusion(CrossAttentionFusion):
    """Dung hợp biểu diễn đa phương thức bằng lớp ``GatedCrossAttentionFusion``.

    Notes
    -----
    Lớp này đóng gói trạng thái và hành vi để các thành phần khác có thể tái sử dụng nhất quán.
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
        """Thực hiện bước init trong quy trình hiện tại.

        Parameters
        ----------
        img_dim : int, optional
            Số lượng, kích thước hoặc tỷ lệ được sử dụng.
        text_dim : int, optional
            Số lượng, kích thước hoặc tỷ lệ được sử dụng.
        embed_dim : int, optional
            Số lượng, kích thước hoặc tỷ lệ được sử dụng.
        num_heads : int, optional
            Số lượng, kích thước hoặc tỷ lệ được sử dụng.
        dropout : float, optional
            Giá trị ``dropout`` được sử dụng trong phép xử lý.
        attention_direction : str, optional
            Giá trị ``attention_direction`` được sử dụng trong phép xử lý.
        gate_hidden_dim : int, optional
            Số lượng, kích thước hoặc tỷ lệ được sử dụng.
        gate_bias_init : float, optional
            Giá trị ``gate_bias_init`` được sử dụng trong phép xử lý.
        **kwargs : dict
            Các đối số từ khóa bổ sung.

        Raises
        ------
        ValueError
            Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
        """
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

        # Chuẩn hóa chuỗi token và mặt nạ đệm cho batch.
        # Chuẩn hóa chuỗi token và mặt nạ đệm cho batch.
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
        """Thực hiện bước pool ảnh các token trong quy trình hiện tại.

        Parameters
        ----------
        image_tokens : torch.Tensor
            Ảnh hoặc biểu diễn ảnh đầu vào.
        local_padding_mask : Optional[torch.Tensor]
            Giá trị ``local_padding_mask`` được sử dụng trong phép xử lý.

        Returns
        -------
        torch.Tensor
            Kết quả được tạo bởi bước xử lý của hàm.

        Raises
        ------
        ValueError
            Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
        """
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
        """Thực hiện lượt lan truyền xuôi của mô hình.

        Parameters
        ----------
        img_feats : torch.Tensor
            Ảnh hoặc biểu diễn ảnh đầu vào.
        txt_feats : torch.Tensor
            Giá trị ``txt_feats`` được sử dụng trong phép xử lý.
        img_key_padding_mask : Optional[torch.Tensor]
            Ảnh hoặc biểu diễn ảnh đầu vào.
        txt_key_padding_mask : Optional[torch.Tensor]
            Tên hoặc khóa định danh của giá trị.
        return_attn : bool, optional
            Giá trị ``return_attn`` được sử dụng trong phép xử lý.

        Returns
        -------
        object
            Kết quả được tạo bởi bước xử lý của hàm.

        Raises
        ------
        ValueError
            Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
        """
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
        # Bước hỗ trợ để thực hiện lượt lan truyền xuôi của mô hình.
        # Kiểm tra và xử lý checkpoint tương ứng của mô hình.
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
