"""
BiomedCLIP Foundation Backbone Module for XBone-Net.
===============================================================================
Wraps the BiomedCLIP foundation model (ViT-B/16 image encoder + PubMedBERT text
encoder) pre-trained contrastively on 15 million biomedical figure-caption pairs
(PMC-15M).

Provides normalized 512-dimensional multimodal feature embeddings for contrastive
alignment (Phase 1) and downstream classification (Phase 2).
"""

import torch.nn as nn
from open_clip import create_model_and_transforms, get_tokenizer


# ============================================================
# BiomedCLIP Foundation Backbone
# ============================================================

class BiomedCLIPFoundation(nn.Module):
    """BiomedCLIP foundation backbone combining ViT-B/16 and PubMedBERT.

    Loads pre-trained BiomedCLIP weights from HuggingFace Hub via OpenCLIP.
    Produces L2-normalized 512-dimensional feature embeddings for both visual
    and textual inputs.

    Attributes:
        model (nn.Module): Underlying OpenCLIP CLIP model instance.
        preprocess (callable): Torchvision transformation pipeline for image preprocessing.
        tokenizer (callable): Tokenizer for converting text strings into token IDs.

    Example:
        >>> backbone = BiomedCLIPFoundation(freeze_base=True)
        >>> img_feats, txt_feats = backbone(images, input_ids)
    """

    def __init__(self, freeze_base: bool = True):
        """Initialize BiomedCLIP model and optionally freeze parameters.

        Args:
            freeze_base (bool): If True, freezes pre-trained backbone parameters
                to prevent catastrophic forgetting during adapter fine-tuning.
                Defaults to True.
        """
        super().__init__()
        model_name = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

        # --- Load model checkpoint and transforms ---
        self.model, _, self.preprocess = create_model_and_transforms(model_name)
        self.tokenizer = get_tokenizer(model_name)
        
        # Attribute to control local vs global feature extraction dynamically
        self.return_local = False

        # --- Freeze backbone parameters if requested ---
        if freeze_base:
            for param in self.model.parameters():
                param.requires_grad = False

    @property
    def tokenizer_obj(self):
        """Tokenizer callable alias for interface consistency."""
        return self.tokenizer

    def forward(self, images, input_ids, attention_mask=None):
        """Extract and L2-normalize image and text feature embeddings.

        Supports dynamic local feature extraction if self.return_local is True.

        Args:
            images (torch.Tensor): Preprocessed image batch, shape [B, 3, 224, 224].
            input_ids (torch.Tensor): Tokenized text IDs batch, shape [B, L].
            attention_mask (torch.Tensor, optional): Text attention mask (1 for real, 0 for pad).

        Returns:
            tuple: (image_features, text_features) where:
                - If self.return_local is False: shapes [B, 512] and [B, 512].
                - If self.return_local is True: shapes [B, 197, 512] and [B, L, 512].
        """
        if getattr(self, "return_local", False):
            # --- Local Feature Extraction for Cross-Attention ---
            # 1. Image visual patch embeddings: trunk.forward_features yields [B, 197, 768]
            patch_feats_768 = self.model.visual.trunk.forward_features(images)
            # Project using visual head (Dropout + Linear) to [B, 197, 512]
            image_features = self.model.visual.head(patch_feats_768)
            
            # 2. Text token embeddings: transformer yields last_hidden_state [B, L, 768]
            text_module = getattr(self.model, 'text', getattr(self.model, 'text_model', None))
            
            # Pass attention mask down to PubMedBERT transformer if available
            if attention_mask is not None:
                transformer_out = text_module.transformer(input_ids, attention_mask=attention_mask)
            else:
                transformer_out = text_module.transformer(input_ids)
                
            text_feats_768 = transformer_out[0]
            # Project using text proj (Linear + GELU + Linear) to [B, L, 512]
            text_features = text_module.proj(text_feats_768)
        else:
            # --- Global Feature Extraction ---
            image_features = self.model.encode_image(images)
            text_features = self.model.encode_text(input_ids)

        # --- L2 normalization (along embedding dimension) ---
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return image_features, text_features
