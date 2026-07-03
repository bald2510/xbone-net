"""
DenseNet-121 Backbone Foundation Module for XBone-Net.
===============================================================================
Provides a DenseNet-121 image-only baseline architecture for bone X-ray classification.
Extracted visual features are projected into a 512-dimensional embedding space.

Exposes a unified interface matching BiomedCLIP, returning (image_features, None)
to indicate image-only modality.
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms
from .resnet50 import DummyTokenizer, DummyTextModule


# ============================================================
# DenseNet-121 Image Backbone
# ============================================================

class DenseNetBackbone(nn.Module):
    """DenseNet-121 image-only feature extractor with linear projection layer.

    Extracts high-level visual features from DenseNet-121 bottleneck layers and projects
    them to a normalized embedding space of dimension embed_dim (default 512).

    Attributes:
        EMBED_DIM (int): Default feature embedding dimension (512).
        pretrained_type (str): Pre-training weight initialization type ('imagenet', 'medical').
        densenet (nn.Module): Truncated DenseNet-121 feature extractor.
        preprocess (callable): Input image preprocessing transform pipeline.
        projection (nn.Module): Sequential linear and LayerNorm projection layers.
        tokenizer (DummyTokenizer): Pass-through tokenizer for architecture compatibility.

    Example:
        >>> backbone = DenseNetBackbone(pretrained="imagenet", embed_dim=512)
        >>> img_feats, _ = backbone(images)
    """

    EMBED_DIM = 512

    def __init__(
        self,
        pretrained: str = "imagenet",
        embed_dim: int = 512,
        freeze_base: bool = False,
    ):
        """Initialize DenseNet-121 feature extractor.

        Args:
            pretrained (str): Weight initialization ('imagenet' or 'medical'). Defaults to 'imagenet'.
            embed_dim (int): Output feature projection dimension. Defaults to 512.
            freeze_base (bool): If True, freezes DenseNet backbone parameters. Defaults to False.
        """
        super().__init__()
        self.pretrained_type = pretrained

        # --- Load DenseNet backbone weights and transforms ---
        if pretrained == "imagenet" or pretrained == "medical":
            weights = models.DenseNet121_Weights.IMAGENET1K_V1
            self.densenet = models.densenet121(weights=weights)
            self.preprocess = weights.transforms()
            if pretrained == "medical":
                print("[DenseNet121] Using ImageNet weights as medical baseline.")
        else:
            self.densenet = models.densenet121(weights=None)
            self.preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])

        # --- Replace classification head with projection layer ---
        densenet_feature_dim = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Identity()

        self.projection = nn.Sequential(
            nn.Linear(densenet_feature_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        # --- Freeze backbone weights if requested ---
        if freeze_base:
            for param in self.densenet.parameters():
                param.requires_grad = False

        self.tokenizer = DummyTokenizer()

    def forward(self, images, input_ids=None):
        """Extract and L2-normalize image embeddings.

        Args:
            images (torch.Tensor): Preprocessed image batch, shape [B, 3, 224, 224].
            input_ids (torch.Tensor, optional): Tokenized text input (ignored). Defaults to None.

        Returns:
            tuple: (image_features, None) where image_features is L2-normalized [B, D].
        """
        feats = self.densenet(images)
        projected = self.projection(feats)
        image_features = projected / projected.norm(dim=-1, keepdim=True)

        return image_features, None


# ============================================================
# DenseNet Foundation Wrapper
# ============================================================

class DenseNetFoundation(nn.Module):
    """Foundation model wrapper matching BiomedCLIPFoundation interface.

    Wraps DenseNetBackbone to expose .model, .preprocess, and .tokenizer attributes
    compatible with the XBone composite builder.

    Attributes:
        model (DenseNetModelWrapper): Sub-module wrapper exposing .visual and .text attributes.
        preprocess (callable): Preprocessing transforms for input images.
        tokenizer (DummyTokenizer): Dummy tokenizer returning pass-through tokens.
    """

    def __init__(
        self,
        pretrained: str = "imagenet",
        embed_dim: int = 512,
        freeze_base: bool = False,
    ):
        """Initialize DenseNet foundation wrapper.

        Args:
            pretrained (str): Weight initialization type. Defaults to 'imagenet'.
            embed_dim (int): Projection embedding dimension. Defaults to 512.
            freeze_base (bool): If True, freeze backbone weights. Defaults to False.
        """
        super().__init__()
        self._backbone = DenseNetBackbone(
            pretrained=pretrained,
            embed_dim=embed_dim,
            freeze_base=freeze_base,
        )

        self.model = DenseNetModelWrapper(self._backbone)
        self.preprocess = self._backbone.preprocess
        self.tokenizer = self._backbone.tokenizer

    def forward(self, images, input_ids=None):
        """Forward pass forwarding to underlying DenseNetBackbone.

        Args:
            images (torch.Tensor): Input image tensor batch, shape [B, 3, 224, 224].
            input_ids (torch.Tensor, optional): Input text IDs batch (ignored).

        Returns:
            tuple: (image_features, None)
        """
        return self._backbone(images, input_ids)


# ============================================================
# DenseNet Sub-module API Compatibility Wrapper
# ============================================================

class DenseNetModelWrapper:
    """Wrapper providing .visual and .text attributes for builder compatibility."""

    def __init__(self, backbone: DenseNetBackbone):
        """Initialize model wrapper with underlying backbone.

        Args:
            backbone (DenseNetBackbone): Instantiated DenseNet feature extractor.
        """
        self.visual = nn.Sequential(
            backbone.densenet,
            backbone.projection,
        )
        self.text = DummyTextModule()

    def encode_image(self, images):
        """Encode images through visual pipeline and apply L2 normalization.

        Args:
            images (torch.Tensor): Preprocessed input image batch, shape [B, 3, 224, 224].

        Returns:
            torch.Tensor: L2-normalized visual feature embeddings, shape [B, 512].
        """
        feats = self.visual(images)
        return feats / feats.norm(dim=-1, keepdim=True)

    def encode_text(self, input_ids):
        """Dummy text encoding returning zero feature vectors.

        Args:
            input_ids (torch.Tensor): Token ID tensor (used only for batch dimension and device).

        Returns:
            torch.Tensor: Zero tensor of shape [B, 512].
        """
        batch_size = input_ids.shape[0] if isinstance(input_ids, torch.Tensor) else 1
        return torch.zeros(batch_size, 512, device=input_ids.device if isinstance(input_ids, torch.Tensor) else "cpu")


