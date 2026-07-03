"""
OpenCLIP Foundation Backbone Module for XBone-Net.
===============================================================================
Provides a unified interface wrapping pre-trained OpenCLIP and HuggingFace vision-language
foundation models (CLIP ViT-B/16, PubMedCLIP, BiomedCLIP).

Handles tokenizer pickling compatibility for multiprocessing, attribute mapping between
HuggingFace and OpenCLIP formats, and pre-trained parameter freezing.
"""

import torch.nn as nn
from open_clip import create_model_and_transforms, get_tokenizer
from transformers import CLIPModel


# ============================================================
# Auxiliary Tokenizer & Model Adaptors
# ============================================================

class PubMedCLIPTokenizerWrapper:
    """Picklable callable wrapper around HuggingFace CLIPTokenizer.

    Ensures compatibility with PyTorch DataLoader multiprocessing on Windows by wrapping
    the tokenizer instance in a picklable structure that returns padded/truncated token IDs.

    Attributes:
        tokenizer_obj: Loaded HuggingFace tokenizer instance.
    """

    def __init__(self, tokenizer_obj):
        """Initialize wrapper with HuggingFace tokenizer instance.

        Args:
            tokenizer_obj: Loaded HuggingFace CLIPTokenizer instance.
        """
        self.tokenizer_obj = tokenizer_obj

    def __call__(self, texts):
        """Tokenize a sequence of text strings into PyTorch token tensors.

        Args:
            texts (str or list of str): Input text string or list of text strings.

        Returns:
            torch.Tensor: Encoded token IDs tensor of shape [B, 77].
        """
        return self.tokenizer_obj(
            texts, padding="max_length", max_length=77, truncation=True, return_tensors="pt"
        )["input_ids"]


class PubMedCLIPModel(CLIPModel):
    """Subclass of HuggingFace CLIPModel projecting .visual property to .vision_model.

    Bridges the interface difference between OpenCLIP (which names its vision encoder .visual)
    and HuggingFace Transformers (which names it .vision_model), allowing PEFT/LoRA modules
    to target vision components seamlessly across architectures.
    """

    @property
    def visual(self):
        """Alias for vision_model property.

        Returns:
            nn.Module: Vision encoder model.
        """
        return self.vision_model
    
    @visual.setter
    def visual(self, value):
        """Setter for visual property alias.

        Args:
            value (nn.Module): New vision encoder module.
        """
        self.vision_model = value

    def __setattr__(self, name, value):
        """Intercept attribute assignments for 'visual'.

        Args:
            name (str): Attribute name.
            value: Attribute value.
        """
        if name == 'visual':
            super().__setattr__('vision_model', value)
        else:
            super().__setattr__(name, value)


# ============================================================
# Model Registry Mapping
# ============================================================

# Model registry mapping shorthand config keys to HuggingFace / OpenCLIP hub names.
OPENCLIP_MODEL_REGISTRY = {
    "clip": {
        "model_name": "ViT-B-16",
        "pretrained": "openai",
        "description": "CLIP ViT-B/16 (Radford et al., 2021)",
    },
    "pubmedclip": {
        "model_name": "hf-hub:flaviagiammarino/pubmed-clip-vit-base-patch32",
        "pretrained": None,
        "description": "PubMedCLIP ViT-B/32 (Eslami et al., 2023)",
    },
    "biomedclip": {
        "model_name": "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        "pretrained": None,
        "description": "BiomedCLIP ViT-B/16 (Zhang et al., 2023)",
    },
}


# ============================================================
# OpenCLIP Foundation Backbone
# ============================================================

class OpenCLIPFoundation(nn.Module):
    """Generic backbone wrapper for OpenCLIP-compatible vision-language models.

    Attributes:
        model (nn.Module): Underlying OpenCLIP or PubMedCLIP model instance.
        preprocess (callable): Vision preprocessing pipeline.
        tokenizer (callable): Tokenizer converting text strings to token ID tensors.

    Example:
        >>> backbone = OpenCLIPFoundation(model_key="clip", freeze_base=True)
        >>> img_feats, txt_feats = backbone(images, input_ids)
    """

    def __init__(self, model_key: str = "clip", freeze_base: bool = True):
        """Initialize OpenCLIP foundation model based on registry key.

        Args:
            model_key (str): Model key in OPENCLIP_MODEL_REGISTRY ('clip', 'pubmedclip', 'biomedclip').
            freeze_base (bool): If True, freeze base model parameters. Defaults to True.

        Raises:
            ValueError: If model_key is not recognized in OPENCLIP_MODEL_REGISTRY.
        """
        super().__init__()

        if model_key not in OPENCLIP_MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model_key '{model_key}'. "
                f"Choose from: {list(OPENCLIP_MODEL_REGISTRY.keys())}"
            )

        entry = OPENCLIP_MODEL_REGISTRY[model_key]
        model_name = entry["model_name"]
        pretrained = entry["pretrained"]

        # --- Load PubMedCLIP model and custom wrappers ---
        if model_key == "pubmedclip":
            from transformers import CLIPTokenizer
            from torchvision import transforms
            
            print("[OpenCLIP] Loading PubMedCLIP via HuggingFace Transformers")
            self.model = PubMedCLIPModel.from_pretrained("flaviagiammarino/pubmed-clip-vit-base-patch32")
            
            # Make model parameters contiguous
            for name, module in self.model.named_modules():
                for param_name, param in list(module.named_parameters(recurse=False)):
                    if not param.is_contiguous():
                        new_param = nn.Parameter(param.contiguous(), requires_grad=param.requires_grad)
                        setattr(module, param_name, new_param)
            
            tokenizer_obj = CLIPTokenizer.from_pretrained("flaviagiammarino/pubmed-clip-vit-base-patch32")
            self.tokenizer = PubMedCLIPTokenizerWrapper(tokenizer_obj)
            
            self.preprocess = transforms.Compose([
                transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711)
                ),
            ])
            
            self.model.encode_image = lambda images: self.model.get_image_features(images).pooler_output
            self.model.encode_text = lambda input_ids: self.model.get_text_features(input_ids).pooler_output
        else:
            print(f"[OpenCLIP] Loading {entry['description']}")

            if pretrained:
                self.model, _, self.preprocess = create_model_and_transforms(
                    model_name, pretrained=pretrained
                )
            else:
                self.model, _, self.preprocess = create_model_and_transforms(model_name)

            self.tokenizer = get_tokenizer(model_name)

        # --- Freeze backbone weights if requested ---
        if freeze_base:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, images, input_ids):
        """Extract and L2-normalize image and text embeddings.

        Args:
            images (torch.Tensor): Preprocessed image batch, shape [B, 3, 224, 224].
            input_ids (torch.Tensor): Tokenized text IDs batch, shape [B, L].

        Returns:
            tuple: (image_features, text_features) with shape [B, 512], L2-normalized.
        """
        # --- Feature extraction ---
        image_features = self.model.encode_image(images)
        text_features = self.model.encode_text(input_ids)

        # --- L2 normalization ---
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return image_features, text_features


