"""Backbone BiomedCLIP dùng một ảnh letterbox và văn bản lâm sàng.

Backbone giữ nguyên bộ mã hóa ảnh và văn bản của BiomedCLIP. Trong pha căn
chỉnh, mô hình trả về embedding toàn cục; trong pha phân loại có cross-attention,
mô hình trả về chuỗi patch token ảnh và token văn bản gốc. Không có nhánh ảnh
nhánh ảnh bổ sung hoặc phép gom token trung gian.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from open_clip import create_model_and_transforms, get_tokenizer


class BiomedCLIPFoundation(nn.Module):
    """Bao bọc BiomedCLIP với giao diện thống nhất cho XBone-Net.

    Parameters
    ----------
    freeze_base : bool, default=True
        Đóng băng trọng số foundation khi khởi tạo. LoRA hoặc full fine-tuning
        được cấu hình sau đó trong builder.
    """

    def __init__(self, freeze_base: bool = True) -> None:
        """Khởi tạo model, transform ảnh và tokenizer BiomedCLIP."""
        super().__init__()
        model_name = (
            "hf-hub:microsoft/"
            "BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        )
        self.model, _, self.preprocess = create_model_and_transforms(model_name)
        self.tokenizer = get_tokenizer(model_name)
        self.return_tokens = False
        self.explain_mode = False
        self.last_image_key_padding_mask: Optional[torch.Tensor] = None
        self.last_global_feature: Optional[torch.Tensor] = None

        if freeze_base:
            for parameter in self.model.parameters():
                parameter.requires_grad = False

    @property
    def tokenizer_obj(self):
        """Trả về tokenizer OpenCLIP dùng cho dữ liệu và suy luận."""
        return self.tokenizer

    def _encode_text_tokens(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Chiếu toàn bộ hidden state văn bản sang không gian 512 chiều.

        Parameters
        ----------
        input_ids : torch.Tensor
            Token IDs có dạng ``[B, L]``.
        attention_mask : torch.Tensor or None
            Mặt nạ token hợp lệ có cùng hai chiều đầu.

        Returns
        -------
        torch.Tensor
            Chuỗi token văn bản đã chiếu, dạng ``[B, L, 512]``.

        Raises
        ------
        RuntimeError
            Nếu model BiomedCLIP không cung cấp text transformer hoặc projection.
        """
        text_module = getattr(
            self.model,
            "text",
            getattr(self.model, "text_model", None),
        )
        if text_module is None:
            raise RuntimeError("BiomedCLIP text module was not found.")
        transformer = getattr(text_module, "transformer", text_module)
        output = (
            transformer(input_ids, attention_mask=attention_mask)
            if attention_mask is not None
            else transformer(input_ids)
        )
        hidden = (
            output[0]
            if isinstance(output, (tuple, list))
            else output.last_hidden_state
        )
        projection = getattr(text_module, "proj", None)
        if projection is None:
            raise RuntimeError("BiomedCLIP text projection was not found.")
        return projection(hidden)

    def _encode_visual_tokens(self, images: torch.Tensor) -> torch.Tensor:
        """Mã hóa ảnh 224x224 thành CLS token và patch token BiomedCLIP."""
        patch_features = self.model.visual.trunk.forward_features(images)
        return self.model.visual.head(patch_features)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Trả về embedding ảnh toàn cục đã chuẩn hóa L2."""
        return F.normalize(self.model.encode_image(images), dim=-1)

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Trả về embedding văn bản toàn cục đã chuẩn hóa L2.

        ``attention_mask`` được nhận để thống nhất giao diện; OpenCLIP tự suy ra
        vị trí padding từ ``input_ids`` trong phép pooling gốc.
        """
        del attention_mask
        return F.normalize(self.model.encode_text(input_ids), dim=-1)

    def pool_contrastive_image_features(
        self,
        image_features: torch.Tensor,
    ) -> torch.Tensor:
        """Chuyển đầu ra ảnh thành embedding dùng cho loss tương phản.

        Nếu nhận chuỗi token, CLS token gốc của BiomedCLIP được dùng; nếu nhận
        embedding toàn cục, hàm chỉ chuẩn hóa L2.
        """
        if image_features.ndim == 3:
            image_features = image_features[:, 0]
        if image_features.ndim != 2:
            raise ValueError(
                "image_features must have shape [B,D] or [B,T,D]."
            )
        return F.normalize(image_features, dim=-1)

    def forward(
        self,
        images: Optional[torch.Tensor],
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Mã hóa các modality được cung cấp.

        Parameters
        ----------
        images : torch.Tensor or None
            Batch ảnh letterbox đã qua transform của BiomedCLIP.
        input_ids : torch.Tensor or None
            Batch token văn bản lâm sàng.
        attention_mask : torch.Tensor or None
            Mặt nạ padding văn bản.

        Returns
        -------
        tuple
            ``(image_features, text_features)``. Khi ``return_tokens=True``, mỗi
            phần tử là chuỗi token; ngược lại là embedding toàn cục.
        """
        return_tokens = bool(self.return_tokens)
        self.last_image_key_padding_mask = None

        if images is None:
            image_features = None
            self.last_global_feature = None
        elif return_tokens:
            image_features = self._encode_visual_tokens(images)
            self.last_global_feature = F.normalize(
                image_features[:, 0], dim=-1
            )
            self.last_image_key_padding_mask = torch.zeros(
                image_features.shape[:2],
                dtype=torch.bool,
                device=image_features.device,
            )
        else:
            image_features = self.encode_image(images)
            self.last_global_feature = image_features

        if input_ids is None:
            text_features = None
        elif return_tokens:
            text_features = self._encode_text_tokens(input_ids, attention_mask)
        else:
            text_features = self.encode_text(input_ids, attention_mask)

        return image_features, text_features
