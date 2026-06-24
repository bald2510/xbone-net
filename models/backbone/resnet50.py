"""
ResNet-50 Backbone for image-only baselines.
============================================
Two variants:
  - ImageNet pretrained (general-purpose)
  - Medical pretrained (if available via BiomedCLIP weights or similar)

Interface matches BiomedCLIPFoundation: forward(images, input_ids) -> (image_features, text_features)
Text features are always None for this backbone.
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms


class ResNet50Backbone(nn.Module):
    """
    ResNet-50 backbone for image-only experiments.

    Extracts features from the penultimate layer (2048-d)
    and projects to the same embedding dimension as BiomedCLIP (512-d)
    for fair comparison.

    Args:
        pretrained: "imagenet" for ImageNet pretrained, "medical" for medical pretrained
        embed_dim: Output embedding dimension (default 512 to match BiomedCLIP)
        freeze_base: Whether to freeze ResNet layers (for linear probe experiments)
    """

    EMBED_DIM = 512  # Match BiomedCLIP's embedding dimension

    def __init__(
        self,
        pretrained: str = "imagenet",
        embed_dim: int = 512,
        freeze_base: bool = False,
    ):
        super().__init__()
        self.pretrained_type = pretrained

        # Load ResNet-50
        if pretrained == "imagenet":
            weights = models.ResNet50_Weights.IMAGENET1K_V2
            self.resnet = models.resnet50(weights=weights)
            self.preprocess = weights.transforms()
        elif pretrained == "medical":
            # Start from ImageNet and note this can be replaced with medical weights
            # e.g., from MedCLIP, CheXpert pretrained, etc.
            weights = models.ResNet50_Weights.IMAGENET1K_V2
            self.resnet = models.resnet50(weights=weights)
            self.preprocess = weights.transforms()
            print("[ResNet50] Using ImageNet weights as medical baseline. "
                  "Replace with domain-specific weights when available.")
        else:
            # Random init
            self.resnet = models.resnet50(weights=None)
            self.preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])

        # Remove the final classification layer — keep features only
        resnet_feature_dim = self.resnet.fc.in_features  # 2048
        self.resnet.fc = nn.Identity()

        # Project to BiomedCLIP-compatible embedding dimension
        self.projection = nn.Sequential(
            nn.Linear(resnet_feature_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        # Freeze base if requested (for linear probe experiments)
        if freeze_base:
            for name, param in self.resnet.named_parameters():
                param.requires_grad = False

        # Dummy tokenizer for compatibility
        self.tokenizer = DummyTokenizer()

    def forward(self, images, input_ids=None):
        """
        Forward pass. Returns (image_features, text_features).
        text_features is always None for image-only backbone.

        Args:
            images: (B, C, H, W) tensor
            input_ids: ignored (compatibility with VLM interface)

        Returns:
            image_features: (B, embed_dim) L2-normalized
            text_features: None
        """
        # Extract features
        feats = self.resnet(images)  # (B, 2048) -> (B, 2048) after fc=Identity
        projected = self.projection(feats)  # (B, embed_dim)

        # L2 normalize to match BiomedCLIP output
        image_features = projected / projected.norm(dim=-1, keepdim=True)

        return image_features, None


class DummyTokenizer:
    """Dummy tokenizer that returns zero tensors for compatibility."""

    def __call__(self, texts, **kwargs):
        # Return a tensor of zeros — will be ignored by image-only models
        if isinstance(texts, str):
            texts = [texts]
        return torch.zeros(len(texts), 1, dtype=torch.long)


class ResNet50Foundation(nn.Module):
    """
    Wrapper matching BiomedCLIPFoundation's interface exactly.

    This is the model that gets stored as `backbone` in XBoneMultiModalModel.
    Has .model, .preprocess, .tokenizer attributes.
    """

    def __init__(
        self,
        pretrained: str = "imagenet",
        embed_dim: int = 512,
        freeze_base: bool = False,
    ):
        super().__init__()
        self._backbone = ResNet50Backbone(
            pretrained=pretrained,
            embed_dim=embed_dim,
            freeze_base=freeze_base,
        )

        # Expose model attribute for compatibility with builder.py
        # builder.py accesses backbone.model.visual and backbone.model.text
        self.model = ResNet50ModelWrapper(self._backbone)
        self.preprocess = self._backbone.preprocess
        self.tokenizer = self._backbone.tokenizer

    def forward(self, images, input_ids=None):
        return self._backbone(images, input_ids)


class ResNet50ModelWrapper:
    """
    Provides .visual and .text attributes for compatibility with builder.py.
    builder.py expects backbone.model.visual and backbone.model.text.transformer.
    """

    def __init__(self, backbone: ResNet50Backbone):
        # The visual encoder is the entire resnet + projection
        self.visual = nn.Sequential(
            backbone.resnet,
            backbone.projection,
        )
        # Dummy text "encoder" — just an Identity module
        self.text = DummyTextModule()

    def encode_image(self, images):
        """Encode images through visual pipeline."""
        feats = self.visual(images)
        return feats / feats.norm(dim=-1, keepdim=True)

    def encode_text(self, input_ids):
        """Dummy text encoding — returns zeros."""
        batch_size = input_ids.shape[0] if isinstance(input_ids, torch.Tensor) else 1
        return torch.zeros(batch_size, 512, device=input_ids.device if isinstance(input_ids, torch.Tensor) else "cpu")


class DummyTextModule(nn.Module):
    """Dummy text module with .transformer attribute for builder.py compatibility."""

    def __init__(self):
        super().__init__()
        self.transformer = nn.Identity()

    def forward(self, x):
        return x
