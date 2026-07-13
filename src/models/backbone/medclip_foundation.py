"""
MedCLIP Foundation Backbone Module for XBone-Net.
===============================================================================
Wraps the MedCLIP foundation model (Swin-Tiny image encoder + BioClinicalBERT text
encoder) pre-trained on medical image-text pairs.

Provides normalized 512-dimensional multimodal feature embeddings and compatibility wrappers
for open_clip tokenization and builder PEFT integration.
"""

import sys
import torch
import torch.nn as nn
from torchvision import transforms
import transformers
from transformers import CLIPImageProcessor
import transformers.processing_utils

# --- Patch HuggingFace transformers compatibility for MedCLIP ---
if 'feature_extractor' in transformers.processing_utils.MODALITY_TO_BASE_CLASS_MAPPING:
    transformers.processing_utils.MODALITY_TO_BASE_CLASS_MAPPING['feature_extractor'] = (
        'FeatureExtractionMixin', 'ImageProcessingMixin'
    )

original_clip_init = CLIPImageProcessor.__init__
def wrapped_clip_init(self, *args, **kwargs):
    arg_names = [
        "do_resize", "size", "resample", "do_center_crop", 
        "crop_size", "do_normalize", "image_mean", "image_std", 
        "do_convert_rgb"
    ]
    new_kwargs = dict(kwargs)
    for name, val in zip(arg_names, args):
        new_kwargs[name] = val
    return original_clip_init(self, **new_kwargs)

CLIPImageProcessor.__init__ = wrapped_clip_init
sys.modules['transformers'].CLIPFeatureExtractor = CLIPImageProcessor
transformers.CLIPFeatureExtractor = CLIPImageProcessor

try:
    from medclip import MedCLIPModel, MedCLIPVisionModel, MedCLIPProcessor
except ImportError:
    raise ImportError("MedCLIP is not installed. Install via `pip install medclip`.")


# ============================================================
# MedCLIP Auxiliary Wrappers
# ============================================================

class MedCLIPTokenizerWrapper:
    """Tokenizer wrapper matching open_clip tokenizer interface.

    Attributes:
        processor (MedCLIPProcessor): Underlying HuggingFace processor.
    """

    def __init__(self, processor: MedCLIPProcessor):
        """Initialize MedCLIP tokenizer wrapper.

        Args:
            processor (MedCLIPProcessor): MedCLIP processor instance.
        """
        self.processor = processor

    def __call__(self, texts, **kwargs):
        """Tokenize text input strings into PyTorch token ID tensors.

        Args:
            texts (str or list of str): Input text string or list of strings.

        Returns:
            torch.Tensor: Encoded token IDs tensor of shape [B, 256].
        """
        if isinstance(texts, str):
            texts = [texts]
        encoded = self.processor(
            text=texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=256,
        )
        return encoded["input_ids"]


class MedCLIPTextModule(nn.Module):
    """Wrapper exposing MedCLIP text encoder transformer for builder compatibility.

    Attributes:
        transformer (nn.Module): Underlying BERT transformer module.
    """

    def __init__(self, medclip_model: "MedCLIPModel"):
        """Initialize text module wrapper.

        Args:
            medclip_model (MedCLIPModel): Instantiated MedCLIP model.
        """
        super().__init__()
        self.transformer = getattr(medclip_model, "bert_model", nn.Identity())

    def forward(self, x):
        """Pass-through forward method.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Unmodified input tensor.
        """
        return x


class MedCLIPModelWrapper:
    """Wrapper providing .visual and .text attributes for builder compatibility.

    Attributes:
        visual (nn.Module): Visual encoder module (Swin Transformer).
        text (MedCLIPTextModule): Text encoder wrapper module.
    """

    def __init__(self, medclip_model: "MedCLIPModel"):
        """Initialize model wrapper.

        Args:
            medclip_model (MedCLIPModel): Pre-trained MedCLIP model instance.
        """
        self._medclip = medclip_model
        self.visual: nn.Module = medclip_model.vision_model
        self.text = MedCLIPTextModule(medclip_model)

    def set_grad_checkpointing(self, enable: bool = True):
        """Enable or disable gradient checkpointing on vision encoder.

        Args:
            enable (bool): Whether to enable gradient checkpointing. Defaults to True.
        """
        if hasattr(self.visual, "set_grad_checkpointing"):
            self.visual.set_grad_checkpointing(enable)
        elif hasattr(self.visual, "gradient_checkpointing_enable"):
            if enable:
                self.visual.gradient_checkpointing_enable()
            else:
                self.visual.gradient_checkpointing_disable()

    def encode_image(self, pixel_values):
        """Encode image batch through MedCLIP vision encoder.

        Args:
            pixel_values (torch.Tensor): Preprocessed image batch, shape [B, 3, 224, 224].

        Returns:
            torch.Tensor: Visual feature embeddings, shape [B, 512].
        """
        return self._medclip.encode_image(pixel_values)

    def encode_text(self, input_ids, attention_mask=None):
        """Encode text batch through MedCLIP text encoder.

        Args:
            input_ids (torch.Tensor): Tokenized text IDs batch, shape [B, L].
            attention_mask (torch.Tensor, optional): Binary attention mask. Auto-constructed if None.

        Returns:
            torch.Tensor: Text feature embeddings, shape [B, 512].
        """
        if attention_mask is None:
            attention_mask = (input_ids != 0).long()
        return self._medclip.encode_text(input_ids, attention_mask)

    def parameters(self):
        """Get model parameters generator.

        Returns:
            generator: Underlying MedCLIP model parameters.
        """
        return self._medclip.parameters()


# ============================================================
# MedCLIP Foundation Backbone
# ============================================================

class MedCLIPFoundation(nn.Module):
    """MedCLIP foundation model combining Swin-Tiny and BioClinicalBERT.

    Attributes:
        EMBED_DIM (int): Default feature dimension (512).
        model (MedCLIPModelWrapper): Model wrapper exposing .visual and .text attributes.
        preprocess (callable): torchvision image preprocessing transform pipeline.
        tokenizer (MedCLIPTokenizerWrapper): Tokenizer wrapper for string tokenization.

    Example:
        >>> backbone = MedCLIPFoundation(freeze_base=True)
        >>> img_feats, txt_feats = backbone(images, input_ids)
    """

    EMBED_DIM = 512

    def __init__(self, freeze_base: bool = True):
        """Initialize MedCLIP model, patch state dict loading, and set freeze settings.

        Args:
            freeze_base (bool): If True, freeze base model parameters. Defaults to True.
        """
        super().__init__()

        medclip_model = MedCLIPModel(vision_cls=MedCLIPVisionModel)
        
        # --- Custom state dict load patch to strip non-persistent position_ids ---
        original_load_state_dict = medclip_model.load_state_dict
        def custom_load_state_dict(state_dict, strict=True):
            keys_to_remove = ["text_model.model.embeddings.position_ids"]
            for key in keys_to_remove:
                if key in state_dict:
                    del state_dict[key]
            return original_load_state_dict(state_dict, strict=False)
            
        medclip_model.load_state_dict = custom_load_state_dict
        medclip_model.from_pretrained()

        # Ensure contiguity of model parameters
        for param in medclip_model.parameters():
            if param.data is not None:
                param.data = param.data.contiguous()

        self._medclip = medclip_model
        self.model = MedCLIPModelWrapper(medclip_model)

        # --- Preprocessing and tokenizer ---
        self.preprocess = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        processor = MedCLIPProcessor()
        self.tokenizer = MedCLIPTokenizerWrapper(processor)

        # --- Freeze base model parameters if requested ---
        if freeze_base:
            for param in self._medclip.parameters():
                param.requires_grad = False

    def forward(self, images, input_ids, attention_mask=None, **kwargs):
        """Extract and L2-normalize image and text embeddings.

        Supports dynamic local feature extraction if self.return_local is True.

        Args:
            images (torch.Tensor): Preprocessed image batch, shape [B, 3, 224, 224].
            input_ids (torch.Tensor): Tokenized text IDs batch, shape [B, L].
            attention_mask (torch.Tensor, optional): Text attention mask (1 for real, 0 for pad).

        Returns:
            tuple: (image_features, text_features) with shape [B, 512], L2-normalized.
        """
        # --- Feature extraction ---
        image_features = self._medclip.encode_image(images)
        if attention_mask is None:
            attention_mask = (input_ids != 0).long()
        text_features = self._medclip.encode_text(input_ids, attention_mask)

        # --- L2 normalization ---
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return image_features, text_features


