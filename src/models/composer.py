"""
Composite Multi-Modal Architecture for XBone-Net.
===============================================================================
Defines XBoneMultiModalModel, the unified composite wrapper that assembles the
three core modular stages of the XBone-Net architecture:
  - Stage 1 (Backbone): Foundation vision-language or vision encoder
  - Stage 2 (Fusion): Multimodal feature mixing (Cross-Attention, Concat, Identity)
  - Stage 3 (Classifier Head): Classification head (Prototypical, Linear, Identity)

Supports Phase 1 contrastive feature extraction and Phase 2 classification, as well
as PEFT fine-tuning and gradient checkpointing across diverse backbones.
"""

import torch.nn as nn


# ============================================================
# XBone Composite Multi-Modal Model
# ============================================================

class XBoneMultiModalModel(nn.Module):
    """Unified XBone-Net multi-modal model composing backbone, fusion, and head.

    This class wires the three pluggable components of XBone-Net into a single
    forward pass. During Phase 1 (contrastive learning), the fusion and head
    are typically IdentityFusion and IdentityHead pass-throughs. During
    Phase 2 (classification), they are replaced with a cross-attention fusion
    module and a prototypical or linear classifier head.

    Attributes:
        backbone (nn.Module): Foundation model that encodes images and (optionally)
            text into fixed-dimensional embeddings.
        fusion (nn.Module): Module that merges image and text feature vectors into a
            single fused representation.
        head (nn.Module): Classification head that maps fused features to class logits.

    Example:
        >>> backbone = BiomedCLIPFoundation(...)
        >>> fusion = CrossAttentionFusion(embed_dim=512)
        >>> head = PrototypicalHead(in_features=512, num_classes=5)
        >>> model = XBoneMultiModalModel(backbone, fusion, head)
        >>> logits = model(images, input_ids)
    """

    def __init__(self, backbone: nn.Module, fusion_module: nn.Module, head_module: nn.Module):
        """Initialize the composite model with specified sub-modules.

        Args:
            backbone (nn.Module): Foundation encoder whose forward(images, input_ids)
                returns (image_features, text_features).
            fusion_module (nn.Module): Multimodal fusion module (e.g., CrossAttentionFusion,
                ConcatFusion, or IdentityFusion).
            head_module (nn.Module): Classifier head (e.g., PrototypicalHead,
                LinearHead, or IdentityHead).
        """
        super().__init__()
        self.backbone = backbone
        self.fusion = fusion_module
        self.head = head_module

    def forward(self, images, input_ids, return_features: bool = False):
        """Run the full backbone -> fusion -> head pipeline.

        Data flow algorithm:
            1. Extract embeddings via backbone(images, input_ids) -> (img_feats, txt_feats).
            2. If txt_feats is None (image-only backbone), skip fusion step.
               Otherwise pass through fusion(img_feats, txt_feats) -> fused_feats.
            3. Map fused_feats through head(fused_feats) -> logits.

        Args:
            images (torch.Tensor): Batch of preprocessed images, shape [B, C, H, W].
            input_ids (torch.Tensor): Tokenized text input IDs, shape [B, L]. For
                image-only backbones, this argument is ignored.
            return_features (bool): If True and the classifier head exposes learnable
                prototypes (PrototypicalHead), returns intermediate features and
                prototypes alongside logits. Defaults to False.

        Returns:
            torch.Tensor or tuple:
                - If return_features is False (default): class logits tensor of shape [B, K].
                - If return_features is True and head has prototypes: tuple (logits, features, prototypes)
                  where logits has shape [B, K], features has shape [B, D], and prototypes has shape [K, D].
        """
        # Step 1: Feature extraction from foundation backbone
        img_feats, txt_feats = self.backbone(images, input_ids)

        # Step 2: Feature fusion (image-only backbones return None for text features)
        if txt_feats is None:
            fused_feats = img_feats
        else:
            fused_feats = self.fusion(img_feats, txt_feats)

        # Step 3: Classification head evaluation
        # In Phase 2 with a prototypical head, raw features and prototypes can be returned
        if return_features and hasattr(self.head, 'prototypes'):
            logits, features = self.head(fused_feats, return_features=True)
            return logits, features, self.head.prototypes

        logits = self.head(fused_feats)
        return logits
    
    def print_parameter_summary(self):
        """Print a human-readable summary of model parameters.

        Outputs total, frozen, and trainable parameter counts for the entire model
        and breaks them down per sub-module (backbone, fusion, head). Useful for
        verifying PEFT configurations and freezing settings before training.
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params

        print("\n" + "=" * 50)
        print("MODEL PARAMETER SUMMARY")
        print("=" * 50)
        print(f"Total Parameters     : {total_params:,}")
        print(f"Frozen Parameters    : {frozen_params:,}")
        print(f"Trainable Parameters : {trainable_params:,}")
        print(f"Trainable Ratio      : {(trainable_params / total_params) * 100:.4f}%\n")

        print("--- Module Breakdown ---")
        for name, module in self.named_children():
            mod_total = sum(p.numel() for p in module.parameters())
            mod_train = sum(p.numel() for p in module.parameters() if p.requires_grad)
            if mod_total > 0:
                print(f"  {name.upper()}: Total={mod_total:,}, Trainable={mod_train:,}")
        print("=" * 50 + "\n")

    def print_encoder_layers(self):
        """Print structural details of image and text encoders."""
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
        """Print parameter summary and optionally detailed layer-by-layer structure.

        Args:
            verbose (bool): If True, also outputs the full PyTorch module layer
                structure for visual and text encoders. Defaults to False.
        """
        self.print_parameter_summary()
        if verbose:
            print("[Debug Mode] Printing verbose model encoder architecture...")
            self.print_encoder_layers()

    def gradient_checkpointing_enable(self, **kwargs):
        """Enable gradient checkpointing on both vision and text encoders.

        Vision and text encoders are handled separately due to framework differences:
          - Visual encoder (OpenCLIP ViT): uses set_grad_checkpointing(True)
          - Text encoder (HuggingFace Transformers): uses gradient_checkpointing_enable()

        Args:
            **kwargs: Forwarded to the HuggingFace gradient_checkpointing_enable call.
        """
        print("[XBone Model] Enabling gradient checkpointing.")
        # --- Visual encoder (OpenCLIP API) ---
        if hasattr(self.backbone.model, "set_grad_checkpointing"):
            self.backbone.model.set_grad_checkpointing(True)
        elif hasattr(self.backbone.model, "visual") and hasattr(self.backbone.model.visual, "set_grad_checkpointing"):
            self.backbone.model.visual.set_grad_checkpointing(True)

        # --- Text encoder (HuggingFace Transformers API) ---
        text_module = getattr(self.backbone.model, "text", None)
        if text_module is not None:
            text_transformer = getattr(text_module, "transformer", None)
            if text_transformer is not None and hasattr(text_transformer, "gradient_checkpointing_enable"):
                text_transformer.gradient_checkpointing_enable(**kwargs)

        # --- Enable input gradient requirements ---
        if hasattr(self.backbone.model, "visual"):
            visual = self.backbone.model.visual
            if hasattr(visual, "enable_input_require_grads"):
                visual.enable_input_require_grads()
        if text_module is not None:
            text_transformer = getattr(text_module, "transformer", None)
            if text_transformer is not None and hasattr(text_transformer, "enable_input_require_grads"):
                text_transformer.enable_input_require_grads()


