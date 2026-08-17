"""Ghép backbone BiomedCLIP, fusion và linear head thành XBone-Net."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class XBoneMultiModalModel(nn.Module):
    """Ghép backbone ảnh-văn bản, mô-đun dung hợp và đầu phân lớp."""

    def __init__(self, backbone: nn.Module, fusion_module: nn.Module, head_module: nn.Module):
        """Khởi tạo mô hình đa phương thức.

        Parameters
        ----------
        backbone : nn.Module
            Backbone cung cấp chuỗi token ảnh và văn bản.
        fusion_module : nn.Module
            Mô-đun kết hợp hai biểu diễn modality.
        head_module : nn.Module
            Đầu ánh xạ embedding dung hợp sang logits lớp.
        """
        super().__init__()
        self.backbone = backbone
        self.fusion = fusion_module
        self.head = head_module
        self.drl_auxiliary = None
        self.phase3_mode = False
        self.use_image_in_fusion = True

    def train(self, mode: bool = True):
        """Đặt chế độ train/eval và giữ ba mô-đun chính frozen ở pha DRL.

        Parameters
        ----------
        mode : bool, optional
            Phương pháp hoặc chế độ xử lý được chọn.

        Returns
        -------
        XBoneMultiModalModel
            Chính instance hiện tại.
        """
        super().train(mode)
        if mode and self.phase3_mode:
            self.backbone.eval()
            self.fusion.eval()
            self.head.eval()
            if self.drl_auxiliary is not None:
                self.drl_auxiliary.train(True)
        return self

    def _encode_modalities(
        self,
        images,
        input_ids=None,
        attention_mask=None,
    ):
        """Mã hóa ảnh letterbox và chuỗi token lâm sàng.

        Parameters
        ----------
        images : object
            Giá trị ``images`` được sử dụng trong phép xử lý.
        input_ids : object, optional
            Dữ liệu nguồn của phép xử lý.
        attention_mask : object, optional
            Giá trị ``attention_mask`` được sử dụng trong phép xử lý.
        Returns
        -------
        tuple
            Token ảnh, token văn bản và các padding mask tương ứng.

        Raises
        ------
        ValueError
            Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
        """
        use_image = bool(getattr(self, "use_image_in_fusion", True))
        img_feats, txt_feats = self.backbone(
            images if use_image else None,
            input_ids,
            attention_mask=attention_mask,
        )
        full_img_padding_mask = getattr(
            self.backbone,
            "last_image_key_padding_mask",
            None,
        )
        txt_key_padding_mask = None
        if attention_mask is not None and txt_feats is not None and txt_feats.ndim == 3 and txt_feats.size(1) > 1:
            txt_key_padding_mask = attention_mask[:, 1:] == 0
            expected_len = txt_feats.size(1) - 1
            if txt_key_padding_mask.size(1) != expected_len:
                raise ValueError(
            "Text attention mask length does not match text tokens: "
                    f"{txt_key_padding_mask.size(1)} vs {expected_len}."
                )
        img_key_padding_mask = None
        if img_feats is not None and img_feats.ndim == 3 and img_feats.size(1) > 1 and full_img_padding_mask is not None:
            img_key_padding_mask = full_img_padding_mask[:, 1:]
        return (
            img_feats,
            txt_feats,
            full_img_padding_mask,
            img_key_padding_mask,
            txt_key_padding_mask,
        )

    def _fuse_modalities(
        self,
        img_feats,
        txt_feats,
        full_img_padding_mask=None,
        img_key_padding_mask=None,
        txt_key_padding_mask=None,
    ) -> torch.Tensor:
        """Dung hợp modality có mặt thành một embedding mỗi mẫu.

        Parameters
        ----------
        img_feats : torch.Tensor or None
            Embedding ảnh toàn cục hoặc chuỗi CLS + patch token.
        txt_feats : torch.Tensor or None
            Embedding văn bản toàn cục hoặc chuỗi token.
        full_img_padding_mask : torch.Tensor, optional
            Mask cho toàn bộ chuỗi token ảnh.
        img_key_padding_mask : torch.Tensor, optional
            Mask patch token truyền vào cross-attention.
        txt_key_padding_mask : torch.Tensor, optional
            Mask token văn bản truyền vào cross-attention.

        Returns
        -------
        torch.Tensor
            Embedding dung hợp có dạng ``[B, D]``.

        Raises
        ------
        ValueError
            Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
        """
        if img_feats is None:
            if txt_feats is None:
                raise ValueError("Text-only mode requires text input features.")
            fused_feats = txt_feats[:, 0, :] if txt_feats.ndim == 3 else txt_feats
        elif txt_feats is None:
            if img_feats.ndim == 3:
                fused_feats = self._masked_token_mean(img_feats, full_img_padding_mask)
            elif img_feats.ndim == 2:
                fused_feats = img_feats
            else:
                raise ValueError(
                    "Unexpected image feature shape in image-only mode: "
                    f"{tuple(img_feats.shape)}"
                )
        elif getattr(self.fusion, "supports_padding_mask", False):
            fused_feats = self.fusion(
                img_feats,
                txt_feats,
                img_key_padding_mask=img_key_padding_mask,
                txt_key_padding_mask=txt_key_padding_mask,
            )
        else:
            img_vector = (
                self._masked_token_mean(img_feats, full_img_padding_mask)
                if img_feats.ndim == 3
                else img_feats
            )
            txt_vector = txt_feats[:, 0, :] if txt_feats.ndim == 3 else txt_feats
            fused_feats = self.fusion(img_vector, txt_vector)
        if fused_feats.ndim != 2:
            raise ValueError(
                "Fusion module must return one [D] embedding per sample, "
                f"got {tuple(fused_feats.shape)}."
            )
        return fused_feats

    @staticmethod
    def _masked_token_mean(
        features: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Lấy trung bình các token không bị đánh dấu padding.

        Parameters
        ----------
        features : torch.Tensor
            Chuỗi đặc trưng dạng ``[B, T, D]``.
        key_padding_mask : Optional[torch.Tensor]
            Mask dạng ``[B, T]``; ``True`` biểu thị padding.

        Returns
        -------
        torch.Tensor
            Tensor dạng ``[B, D]``.

        Raises
        ------
        ValueError
            Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
        """
        if features.ndim != 3:
            raise ValueError(f"Expected [B,T,D] features, got {tuple(features.shape)}")
        if key_padding_mask is None:
            return features.mean(dim=1)
        if key_padding_mask.shape != features.shape[:2]:
            raise ValueError(
                "Image padding mask and feature sequence have incompatible shapes: "
                f"{tuple(key_padding_mask.shape)} vs {tuple(features.shape)}"
            )
        valid = (~key_padding_mask).unsqueeze(-1).to(features.dtype)
        denom = valid.sum(dim=1).clamp_min(1.0)
        return (features * valid).sum(dim=1) / denom

    def encode_fused(
        self,
        images,
        input_ids=None,
        attention_mask=None,
    ) -> torch.Tensor:
        """Trả về một embedding dung hợp cho mỗi mẫu trong batch.

        Parameters
        ----------
        images : object
            Giá trị ``images`` được sử dụng trong phép xử lý.
        input_ids : object, optional
            Dữ liệu nguồn của phép xử lý.
        attention_mask : object, optional
            Giá trị ``attention_mask`` được sử dụng trong phép xử lý.
        Returns
        -------
        torch.Tensor
            Kết quả được tạo bởi bước xử lý của hàm.
        """
        encoded = self._encode_modalities(
            images,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return self._fuse_modalities(*encoded)

    def forward_drl(
        self,
        images,
        input_ids=None,
        attention_mask=None,
        return_details: bool = False,
    ):
        """Thực hiện bước forward drl trong quy trình hiện tại.

        Parameters
        ----------
        images : object
            Giá trị ``images`` được sử dụng trong phép xử lý.
        input_ids : object, optional
            Dữ liệu nguồn của phép xử lý.
        attention_mask : object, optional
            Giá trị ``attention_mask`` được sử dụng trong phép xử lý.
        return_details : bool, optional
            Giá trị ``return_details`` được sử dụng trong phép xử lý.

        Returns
        -------
        object
            Kết quả được tạo bởi bước xử lý của hàm.

        Raises
        ------
        RuntimeError
            Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
        """
        if self.drl_auxiliary is None:
            raise RuntimeError("DRL auxiliary branch is not configured.")
        encoded = self._encode_modalities(
            images,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        (
            img_feats,
            txt_feats,
            full_img_padding_mask,
            img_key_padding_mask,
            txt_key_padding_mask,
        ) = encoded
        label_features = self._fuse_modalities(
            img_feats,
            txt_feats,
            full_img_padding_mask,
            img_key_padding_mask,
            txt_key_padding_mask,
        )
        primary_logits = self.head(label_features)
        auxiliary_output = self.drl_auxiliary(
            img_feats,
            txt_feats,
            label_features,
            image_local_padding_mask=img_key_padding_mask,
            return_details=return_details,
        )
        if return_details:
            auxiliary_logits, distribution_features, details = auxiliary_output
        else:
            auxiliary_logits, distribution_features = auxiliary_output
            details = None
        from .drl import drl_ood_score

        result = {
            "primary_logits": primary_logits,
            "auxiliary_logits": auxiliary_logits,
            "label_features": label_features,
            "distribution_features": distribution_features,
            "drl_ood_score": drl_ood_score(primary_logits, auxiliary_logits),
        }
        if details is not None:
            result["auxiliary_details"] = details
        return result

    def forward(
        self,
        images,
        input_ids=None,
        attention_mask=None,
        return_features: bool = False,
    ):
        """Thực hiện lượt lan truyền xuôi của mô hình.

        Parameters
        ----------
        images : object
            Giá trị ``images`` được sử dụng trong phép xử lý.
        input_ids : object, optional
            Dữ liệu nguồn của phép xử lý.
        attention_mask : object, optional
            Giá trị ``attention_mask`` được sử dụng trong phép xử lý.
        return_features : bool, optional
            Giá trị ``return_features`` được sử dụng trong phép xử lý.

        Returns
        -------
        object
            Kết quả được tạo bởi bước xử lý của hàm.
        """
        fused_feats = self.encode_fused(
            images,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        logits = self.head(fused_feats)
        if return_features:
            prototypes = getattr(self.head, "prototypes", None)
            return logits, fused_feats, prototypes
        return logits

    def print_parameter_summary(self):
        """In parameter summary cho bước xử lý hiện tại."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params

        print("\n" + "=" * 50)
        print("MODEL PARAMETER SUMMARY")
        print("=" * 50)
        print(f"Total Parameters     : {total_params:,}")
        print(f"Frozen Parameters    : {frozen_params:,}")
        print(f"Trainable Parameters : {trainable_params:,}")
        ratio = 100.0 * trainable_params / max(total_params, 1)
        print(f"Trainable Ratio      : {ratio:.4f}%\n")

        print("--- Module Breakdown ---")
        for name, module in self.named_children():
            mod_total = sum(p.numel() for p in module.parameters())
            mod_train = sum(p.numel() for p in module.parameters() if p.requires_grad)
            if mod_total > 0:
                print(f"  {name.upper()}: Total={mod_total:,}, Trainable={mod_train:,}")
        print("=" * 50 + "\n")

    def print_encoder_layers(self):
        """In encoder layers cho bước xử lý hiện tại."""
        print("\n" + "=" * 50)
        print("IMAGE ENCODER STRUCTURE")
        print("=" * 50)
        if hasattr(self.backbone, "model") and hasattr(self.backbone.model, "visual"):
            print(self.backbone.model.visual)
        else:
            print("No visual encoder found in backbone.model.visual")

        print("\n" + "=" * 50)
        print("TEXT ENCODER STRUCTURE")
        print("=" * 50)
        if hasattr(self.backbone, "model") and hasattr(self.backbone.model, "text"):
            print(self.backbone.model.text)
        else:
            print("No text encoder found in backbone.model.text")
        print("=" * 50 + "\n")

    def print_architecture(self, verbose: bool = False):
        """In architecture cho bước xử lý hiện tại.

        Parameters
        ----------
        verbose : bool, optional
            Giá trị ``verbose`` được sử dụng trong phép xử lý.
        """
        self.print_parameter_summary()
        if verbose:
            print("[Debug Mode] Printing verbose model encoder architecture...")
            self.print_encoder_layers()

    def gradient_checkpointing_enable(self, **kwargs):
        """Thực hiện bước gradient checkpointing enable trong quy trình hiện tại.

        Parameters
        ----------
        **kwargs : dict
            Các đối số từ khóa bổ sung.
        """
        backbone_model = getattr(self.backbone, "model", None)
        if backbone_model is None:
            print("[XBone Model] Gradient checkpointing is not supported by this backbone.")
            return

        print("[XBone Model] Enabling gradient checkpointing.")
        if hasattr(backbone_model, "set_grad_checkpointing"):
            backbone_model.set_grad_checkpointing(True)
        elif hasattr(backbone_model, "visual") and hasattr(
            backbone_model.visual, "set_grad_checkpointing"
        ):
            backbone_model.visual.set_grad_checkpointing(True)

        text_module = getattr(backbone_model, "text", None)
        text_transformer = getattr(text_module, "transformer", None) if text_module is not None else None
        if text_transformer is not None and hasattr(text_transformer, "gradient_checkpointing_enable"):
            text_transformer.gradient_checkpointing_enable(**kwargs)

        visual = getattr(backbone_model, "visual", None)
        if visual is not None and hasattr(visual, "enable_input_require_grads"):
            visual.enable_input_require_grads()
        if text_transformer is not None and hasattr(text_transformer, "enable_input_require_grads"):
            text_transformer.enable_input_require_grads()
