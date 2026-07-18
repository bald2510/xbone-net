"""Composite multi-modal architecture for XBone-Net."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class XBoneMultiModalModel(nn.Module):
    """Compose backbone, fusion module, and classification head."""

    def __init__(self, backbone: nn.Module, fusion_module: nn.Module, head_module: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.fusion = fusion_module
        self.head = head_module
        self.use_image_in_fusion = True

    @staticmethod
    def _masked_token_mean(
        features: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
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
        tile_values=None,
        tile_mask=None,
        tile_boxes=None,
    ) -> torch.Tensor:
        """Encode a batch into the fused representation before classification."""
        use_image = bool(getattr(self, "use_image_in_fusion", True))
        img_feats, txt_feats = self.backbone(
            images if use_image else None,
            input_ids,
            attention_mask=attention_mask,
            tile_values=tile_values if use_image else None,
            tile_mask=tile_mask if use_image else None,
            tile_boxes=tile_boxes if use_image else None,
        )

        full_img_padding_mask = getattr(
            self.backbone,
            "last_image_key_padding_mask",
            None,
        )

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
        else:
            txt_key_padding_mask = None
            if attention_mask is not None and txt_feats.ndim == 3 and txt_feats.size(1) > 1:
                txt_key_padding_mask = attention_mask[:, 1:] == 0
                expected_len = txt_feats.size(1) - 1
                if txt_key_padding_mask.size(1) != expected_len:
                    raise ValueError(
                        "Text attention mask length does not match local text tokens: "
                        f"{txt_key_padding_mask.size(1)} vs {expected_len}."
                    )

            img_key_padding_mask = None
            if img_feats.ndim == 3 and img_feats.size(1) > 1 and full_img_padding_mask is not None:
                img_key_padding_mask = full_img_padding_mask[:, 1:]

            if getattr(self.fusion, "supports_padding_mask", False):
                fused_feats = self.fusion(
                    img_feats,
                    txt_feats,
                    img_key_padding_mask=img_key_padding_mask,
                    txt_key_padding_mask=txt_key_padding_mask,
                )
            else:
                # Concat/identity fusion operates on one vector per modality.
                # Pool high-resolution visual tokens and use the text global token.
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

    def forward(
        self,
        images,
        input_ids=None,
        attention_mask=None,
        tile_values=None,
        tile_mask=None,
        tile_boxes=None,
        return_features: bool = False,
    ):
        fused_feats = self.encode_fused(
            images,
            input_ids=input_ids,
            attention_mask=attention_mask,
            tile_values=tile_values,
            tile_mask=tile_mask,
            tile_boxes=tile_boxes,
        )

        logits = self.head(fused_feats)
        if return_features:
            prototypes = getattr(self.head, "prototypes", None)
            return logits, fused_feats, prototypes
        return logits

    def print_parameter_summary(self):
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
        self.print_parameter_summary()
        if verbose:
            print("[Debug Mode] Printing verbose model encoder architecture...")
            self.print_encoder_layers()

    def gradient_checkpointing_enable(self, **kwargs):
        """Enable checkpointing when the underlying backbone supports it."""
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
