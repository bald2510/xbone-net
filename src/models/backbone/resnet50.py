"""
ResNet-50 Backbone Foundation Module for XBone-Net.
===============================================================================
Provides a ResNet-50 image-only baseline architecture for bone X-ray classification.
Extracted visual features are projected into a 512-dimensional embedding space.

Exposes a unified interface matching BiomedCLIP, returning (image_features, None)
to indicate image-only modality, alongside dummy text/tokenizer modules.
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms


# ============================================================
# ResNet-50 Image Backbone
# ============================================================

class ResNet50Backbone(nn.Module):
    """ResNet-50 image-only feature extractor with linear projection layer.

    Extracts visual features from ResNet-50 pool layers and projects them to a
    normalized embedding space of dimension embed_dim (default 512).

    Attributes:
        EMBED_DIM (int): Default feature embedding dimension (512).
        pretrained_type (str): Pre-training weight initialization type ('imagenet', 'medical').
        resnet (nn.Module): Truncated ResNet-50 feature extractor.
        preprocess (callable): Input image preprocessing transform pipeline.
        projection (nn.Module): Sequential linear and LayerNorm projection layers.
        tokenizer (DummyTokenizer): Pass-through tokenizer for architecture compatibility.

    Example:
        >>> backbone = ResNet50Backbone(pretrained="imagenet", embed_dim=512)
        >>> img_feats, _ = backbone(images)
    """

    EMBED_DIM = 512

    def __init__(
        self,
        pretrained: str = "imagenet",
        embed_dim: int = 512,
        freeze_base: bool = False,
    ):
        """Initialize ResNet-50 feature extractor.

        Args:
            pretrained (str): Weight initialization ('imagenet' or 'medical'). Defaults to 'imagenet'.
            embed_dim (int): Output feature projection dimension. Defaults to 512.
            freeze_base (bool): If True, freezes ResNet backbone parameters. Defaults to False.
        """
        super().__init__()
        self.pretrained_type = pretrained

        # --- Load ResNet-50 weights and transforms ---
        if pretrained == "imagenet" or pretrained == "medical":
            weights = models.ResNet50_Weights.IMAGENET1K_V2
            self.resnet = models.resnet50(weights=weights)
            self.preprocess = weights.transforms()
            if pretrained == "medical":
                print("[ResNet50] Using ImageNet weights as medical baseline.")
        else:
            self.resnet = models.resnet50(weights=None)
            self.preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])

        # --- Replace classification head with projection layer ---
        resnet_feature_dim = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()

        self.projection = nn.Sequential(
            nn.Linear(resnet_feature_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        # --- Freeze backbone weights if requested ---
        if freeze_base:
            for param in self.resnet.parameters():
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
        feats = self.resnet(images)
        projected = self.projection(feats)
        image_features = projected / projected.norm(dim=-1, keepdim=True)

        return image_features, None


# ============================================================
# Dummy Tokenizer & Text Support Modules
# ============================================================

class DummyTokenizer:
    """Dummy tokenizer returning zero tensors for image-only compatibility."""

    def __call__(self, texts, **kwargs):
        """Tokenize string input into dummy zero tensor.

        Args:
            texts (str or list of str): Input text string or list of text strings.

        Returns:
            torch.Tensor: Long tensor of shape [B, 1] containing zeros.
        """
        if isinstance(texts, str):
            texts = [texts]
        return torch.zeros(len(texts), 1, dtype=torch.long)


class DummyTextModule(nn.Module):
    """Dummy text module with .transformer attribute for builder compatibility."""

    def __init__(self):
        """Initialize dummy text module."""
        super().__init__()
        self.transformer = nn.Identity()

    def forward(self, x):
        """Pass-through forward method.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Unmodified input tensor.
        """
        return x


# ============================================================
# ResNet-50 Foundation Wrapper
# ============================================================

class ResNet50Foundation(nn.Module):
    """Foundation model wrapper matching BiomedCLIPFoundation interface.

    Attributes:
        model (ResNet50ModelWrapper): Sub-module wrapper exposing .visual and .text attributes.
        preprocess (callable): Preprocessing transforms for input images.
        tokenizer (DummyTokenizer): Dummy tokenizer returning pass-through tokens.
    """

    def __init__(
        self,
        pretrained: str = "imagenet",
        embed_dim: int = 512,
        freeze_base: bool = False,
    ):
        """Initialize ResNet-50 foundation wrapper.

        Args:
            pretrained (str): Weight initialization type. Defaults to 'imagenet'.
            embed_dim (int): Projection embedding dimension. Defaults to 512.
            freeze_base (bool): If True, freeze backbone weights. Defaults to False.
        """
        super().__init__()
        self._backbone = ResNet50Backbone(
            pretrained=pretrained,
            embed_dim=embed_dim,
            freeze_base=freeze_base,
        )

        self.model = ResNet50ModelWrapper(self._backbone)
        self.preprocess = self._backbone.preprocess
        self.tokenizer = self._backbone.tokenizer

    def forward(self, images, input_ids=None):
        """Forward pass forwarding to underlying ResNet50Backbone.

        Args:
            images (torch.Tensor): Input image tensor batch, shape [B, 3, 224, 224].
            input_ids (torch.Tensor, optional): Input text IDs batch (ignored).

        Returns:
            tuple: (image_features, None)
        """
        return self._backbone(images, input_ids)


# ============================================================
# ResNet-50 Sub-module API Compatibility Wrapper
# ============================================================

class ResNet50ModelWrapper:
    """Wrapper providing .visual and .text attributes for builder compatibility."""

    def __init__(self, backbone: ResNet50Backbone):
        """Initialize model wrapper with underlying backbone.

        Args:
            backbone (ResNet50Backbone): Instantiated ResNet-50 feature extractor.
        """
        self.visual = nn.Sequential(
            backbone.resnet,
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


