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

        # --- Freeze backbone parameters if requested ---
        if freeze_base:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, images, input_ids):
        """Extract and L2-normalize image and text feature embeddings.

        Mathematical Formulation:
            Features are L2-normalized onto the unit hypersphere:
            v_norm = v / ||v||_2
            where ||v||_2 = sqrt(sum(v_i^2)). This ensures dot product equals cosine similarity.

        Args:
            images (torch.Tensor): Preprocessed image batch, shape [B, 3, 224, 224].
            input_ids (torch.Tensor): Tokenized text IDs batch, shape [B, L].

        Returns:
            tuple: (image_features, text_features) where each tensor is L2-normalized
                with shape [B, 512].
        """
        # --- Feature extraction ---
        image_features = self.model.encode_image(images)
        text_features = self.model.encode_text(input_ids)

        # --- L2 normalization (cosine similarity matching) ---
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return image_features, text_features
