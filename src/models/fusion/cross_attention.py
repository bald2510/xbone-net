"""Cung cấp cơ chế dung hợp đa phương thức cross attention cho XBone-Net.

Notes
-----
Mô-đun này thuộc cơ sở mã nguồn nghiên cứu XBone-Net và giữ các quy ước dùng chung của dự án.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def reduce_attention_to_keys(
    attention: torch.Tensor,
    key_padding_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Thực hiện bước reduce attention to keys trong quy trình hiện tại.

    Parameters
    ----------
    attention : torch.Tensor
        Giá trị ``attention`` được sử dụng trong phép xử lý.
    key_padding_mask : Optional[torch.Tensor]
        Tên hoặc khóa định danh của giá trị.

    Returns
    -------
    torch.Tensor
        Kết quả được tạo bởi bước xử lý của hàm.

    Raises
    ------
    ValueError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    """
    if attention.ndim != 4:
        raise ValueError(
            "attention must have shape [B,H,Q,K], "
            f"got {tuple(attention.shape)}."
        )
    key_scores = attention.mean(dim=(1, 2))
    if key_padding_mask is not None:
        key_padding_mask = key_padding_mask.to(
            device=key_scores.device,
            dtype=torch.bool,
        )
        if key_padding_mask.shape != key_scores.shape:
            raise ValueError(
                "key_padding_mask must match reduced attention shape "
                f"{tuple(key_scores.shape)}, got {tuple(key_padding_mask.shape)}."
            )
        key_scores = key_scores.masked_fill(key_padding_mask, 0.0)
    return key_scores / key_scores.sum(dim=-1, keepdim=True).clamp_min(1e-8)


class CrossAttentionFusion(nn.Module):
    """Dung hợp biểu diễn đa phương thức bằng lớp ``CrossAttentionFusion``.

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
        **kwargs,
    ):
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
        **kwargs : dict
            Các đối số từ khóa bổ sung.

        Raises
        ------
        ValueError
            Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
        """
        super().__init__()
        img_dim = kwargs.get("img_dim", img_dim)
        text_dim = kwargs.get("text_dim", text_dim)
        attention_direction = kwargs.get(
            "attention_direction", attention_direction
        )

        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")
        valid_directions = {
            "bidirectional",
            "image_to_text",
            "text_to_image",
        }
        if attention_direction not in valid_directions:
            raise ValueError(
                "attention_direction must be one of "
                f"{sorted(valid_directions)}, got '{attention_direction}'."
            )
        self.attention_direction = attention_direction

        self.img_input_norm = nn.LayerNorm(img_dim)
        self.txt_input_norm = nn.LayerNorm(text_dim)
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
        self.norm_img_global = nn.LayerNorm(embed_dim)
        self.norm_txt_global = nn.LayerNorm(embed_dim)
        self.norm_fuse = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )

    @staticmethod
    def _validate_mask(
        mask: Optional[torch.Tensor],
        batch_size: int,
        seq_len: int,
        name: str,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """Kiểm tra tính hợp lệ của mask cho bước xử lý hiện tại.

        Parameters
        ----------
        mask : Optional[torch.Tensor]
            Giá trị ``mask`` được sử dụng trong phép xử lý.
        batch_size : int
            Số lượng, kích thước hoặc tỷ lệ được sử dụng.
        seq_len : int
            Giá trị ``seq_len`` được sử dụng trong phép xử lý.
        name : str
            Tên hoặc khóa định danh của giá trị.
        device : torch.device
            Thiết bị thực thi phép tính.

        Returns
        -------
        Optional[torch.Tensor]
            Kết quả được tạo bởi bước xử lý của hàm.

        Raises
        ------
        ValueError
            Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
        """
        if mask is None:
            return None
        mask = mask.to(device=device, dtype=torch.bool)
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
        if img_feats.ndim == 2:
            img_feats = img_feats.unsqueeze(1)
        if txt_feats.ndim == 2:
            txt_feats = txt_feats.unsqueeze(1)
        if img_feats.ndim != 3 or txt_feats.ndim != 3:
            raise ValueError("Cross-attention expects [B,T,D] or [B,D] inputs.")
        if img_feats.size(0) != txt_feats.size(0):
            raise ValueError("Image and text batch sizes must match.")
        if img_feats.device != txt_feats.device:
            raise ValueError(
                "Image and text features must be on the same device: "
                f"{img_feats.device} vs {txt_feats.device}."
            )

        img_feats = self.img_proj(self.img_input_norm(img_feats))
        txt_feats = self.txt_proj(self.txt_input_norm(txt_feats))

        img_global = self.norm_img_global(img_feats[:, 0:1, :])
        txt_global = self.norm_txt_global(txt_feats[:, 0:1, :])
        img_local = img_feats[:, 1:, :]
        txt_local = txt_feats[:, 1:, :]

        # Xử lý bộ mã hóa nền tảng theo giao diện tương ứng.
        if img_local.size(1) == 0:
            img_local = img_global
            img_key_padding_mask = None
        if txt_local.size(1) == 0:
            txt_local = txt_global
            txt_key_padding_mask = None

        batch_size = img_feats.size(0)
        uses_i2t = self.attention_direction in {
            "bidirectional",
            "image_to_text",
        }
        uses_t2i = self.attention_direction in {
            "bidirectional",
            "text_to_image",
        }
        if uses_t2i:
            img_key_padding_mask = self._validate_mask(
                img_key_padding_mask,
                batch_size,
                img_local.size(1),
                "img_key_padding_mask",
                img_feats.device,
            )
        if uses_i2t:
            txt_key_padding_mask = self._validate_mask(
                txt_key_padding_mask,
                batch_size,
                txt_local.size(1),
                "txt_key_padding_mask",
                txt_feats.device,
            )

        attn_i2t = None
        image_enhanced = self.norm_img(img_global)
        if uses_i2t:
            image_from_text, attn_i2t = self.img_to_txt_attn(
                query=img_global,
                key=txt_local,
                value=txt_local,
                key_padding_mask=txt_key_padding_mask,
                need_weights=return_attn,
                average_attn_weights=False,
            )
            image_enhanced = self.norm_img(
                img_global + self.dropout(image_from_text)
            )

        attn_t2i = None
        text_enhanced = self.norm_txt(txt_global)
        if uses_t2i:
            text_from_image, attn_t2i = self.txt_to_img_attn(
                query=txt_global,
                key=img_local,
                value=img_local,
                key_padding_mask=img_key_padding_mask,
                need_weights=return_attn,
                average_attn_weights=False,
            )
            text_enhanced = self.norm_txt(
                txt_global + self.dropout(text_from_image)
            )

        # Thu thập và xử lý biểu diễn đặc trưng của mô hình.
        # Thiết lập mô-đun dung hợp và đầu phân lớp theo cấu hình.
        joint = torch.cat(
            [
                image_enhanced.squeeze(1),
                text_enhanced.squeeze(1),
            ],
            dim=-1,
        )
        fused = self.norm_fuse(self.fusion_mlp(joint))

        if return_attn:
            return fused, {
                "attn_img_to_txt": attn_i2t,
                "attn_txt_to_img": attn_t2i,
                "attention_direction": self.attention_direction,
                "img_global": img_global.squeeze(1),
                "txt_global": txt_global.squeeze(1),
                "txt_context_for_image": image_enhanced.squeeze(1),
                "img_context_for_text": text_enhanced.squeeze(1),
            }
        return fused
