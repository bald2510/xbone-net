"""
MedCLIP Foundation Backbone.
============================
Wraps the ``medclip`` package (Swin-Tiny vision encoder) so it exposes the
same interface as ``BiomedCLIPFoundation``:

    backbone.model          – has .visual, .text, .set_grad_checkpointing()
    backbone.preprocess     – torchvision transform
    backbone.tokenizer      – callable(texts) → input_ids tensor
    backbone.forward(images, input_ids) → (image_features, text_features)

Both feature tensors are L2-normalised with dim = 512.
"""

import torch
import torch.nn as nn
from torchvision import transforms

# Monkeypatch CLIPFeatureExtractor and CLIPImageProcessor.__init__ for newer Hugging Face transformers versions
# where CLIPFeatureExtractor has been renamed/deprecated/removed, and CLIPImageProcessor.__init__ has changed.
import sys
import transformers
from transformers import CLIPImageProcessor
import transformers.processing_utils

# Allow ImageProcessingMixin for feature_extractor key in MODALITY_TO_BASE_CLASS_MAPPING
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
    raise ImportError(
        "MedCLIP is not installed. Install it with:\n"
        "  pip install medclip\n"
        "or clone from https://github.com/RyanWangZf/MedCLIP"
    )


# ---------------------------------------------------------------------------
# Tokenizer wrapper – adapts MedCLIPProcessor to the open_clip tokenizer API
# ---------------------------------------------------------------------------

class MedCLIPTokenizerWrapper:
    """Wraps ``MedCLIPProcessor`` so that ``tokenizer(texts)`` returns a plain
    ``input_ids`` tensor of shape ``(B, max_len)``, matching the open_clip
    tokenizer contract used elsewhere in the codebase."""

    def __init__(self, processor: MedCLIPProcessor):
        self.processor = processor

    def __call__(self, texts, **kwargs):
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


# ---------------------------------------------------------------------------
# Dummy text module – gives builder.py a .text.transformer to probe
# ---------------------------------------------------------------------------

class MedCLIPTextModule(nn.Module):
    """Thin wrapper around MedCLIP's internal text encoder so that
    ``backbone.model.text.transformer`` resolves to an ``nn.Module``.

    builder.py uses ``getattr(backbone.model, 'text', None)`` and then
    accesses ``.transformer`` for optional PEFT injection.  We point
    ``.transformer`` at the actual Hugging-Face text encoder inside MedCLIP.
    """

    def __init__(self, medclip_model: "MedCLIPModel"):
        super().__init__()
        # MedCLIPModel stores its text encoder as .bert_model (a HF AutoModel)
        self.transformer = getattr(medclip_model, "bert_model", nn.Identity())

    def forward(self, x):
        return x


# ---------------------------------------------------------------------------
# Model wrapper – provides .visual / .text / .set_grad_checkpointing()
# ---------------------------------------------------------------------------

class MedCLIPModelWrapper:
    """Provides the ``.visual``, ``.text``, and ``set_grad_checkpointing``
    attributes that ``builder.py`` and ``composer.py`` expect on
    ``backbone.model``.

    ``.visual`` is set to the Swin-Tiny ``nn.Module`` so that PEFT adapters
    can be injected directly onto it.
    """

    def __init__(self, medclip_model: "MedCLIPModel"):
        self._medclip = medclip_model
        # Expose the Swin-Tiny vision encoder as an nn.Module
        # MedCLIPModel stores it as .vision_model (a MedCLIPVisionModel)
        self.visual: nn.Module = medclip_model.vision_model
        self.text = MedCLIPTextModule(medclip_model)

    # ------------------------------------------------------------------
    # Gradient checkpointing (best-effort)
    # ------------------------------------------------------------------
    def set_grad_checkpointing(self, enable: bool = True):
        """Enable gradient checkpointing on the vision encoder if supported."""
        if hasattr(self.visual, "set_grad_checkpointing"):
            self.visual.set_grad_checkpointing(enable)
        elif hasattr(self.visual, "gradient_checkpointing_enable"):
            # HuggingFace-style API
            if enable:
                self.visual.gradient_checkpointing_enable()
            else:
                self.visual.gradient_checkpointing_disable()
        # Silently ignore if not supported – matches composer.py fallback

    # Convenience delegates so the wrapper can also be used to encode
    def encode_image(self, pixel_values):
        return self._medclip.encode_image(pixel_values)

    def encode_text(self, input_ids, attention_mask=None):
        if attention_mask is None:
            attention_mask = (input_ids != 0).long()
        return self._medclip.encode_text(input_ids, attention_mask)

    def parameters(self):
        """Yield all parameters – used by freeze loop in __init__."""
        return self._medclip.parameters()


# ---------------------------------------------------------------------------
# Foundation class – the public API
# ---------------------------------------------------------------------------

class MedCLIPFoundation(nn.Module):
    """MedCLIP backbone with the same interface as ``BiomedCLIPFoundation``.

    Attributes:
        model      : ``MedCLIPModelWrapper`` with ``.visual`` / ``.text``
        preprocess : ``torchvision.transforms.Compose`` for images
        tokenizer  : ``MedCLIPTokenizerWrapper`` (callable → input_ids)
    """

    EMBED_DIM = 512  # MedCLIP projection dimension

    def __init__(self, freeze_base: bool = True):
        super().__init__()

        # ---- Load MedCLIP (Swin-Tiny) ----
        medclip_model = MedCLIPModel(vision_cls=MedCLIPVisionModel)
        
        # Monkeypatch load_state_dict to handle strict loading issues with newer transformers versions
        original_load_state_dict = medclip_model.load_state_dict
        def custom_load_state_dict(state_dict, strict=True):
            # Remove keys that cause problems in newer transformers versions
            keys_to_remove = ["text_model.model.embeddings.position_ids"]
            for key in keys_to_remove:
                if key in state_dict:
                    del state_dict[key]
            # Use strict=False to bypass other non-critical matching conflicts
            return original_load_state_dict(state_dict, strict=False)
            
        medclip_model.load_state_dict = custom_load_state_dict
        medclip_model.from_pretrained()

        # Ensure all parameters are contiguous to prevent safetensors saving issues
        for param in medclip_model.parameters():
            if param.data is not None:
                param.data = param.data.contiguous()

        # Store the raw medclip model as a sub-module so its parameters are
        # registered in this nn.Module (important for .to(device), state_dict, etc.)
        self._medclip = medclip_model

        # ---- Model wrapper (builder.py / composer.py compatibility) ----
        self.model = MedCLIPModelWrapper(medclip_model)

        # ---- Image preprocessing (ImageNet stats, 224×224) ----
        self.preprocess = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        # ---- Tokenizer ----
        processor = MedCLIPProcessor()
        self.tokenizer = MedCLIPTokenizerWrapper(processor)

        # ---- Freeze base weights ----
        if freeze_base:
            for param in self._medclip.parameters():
                param.requires_grad = False

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, images, input_ids):
        """
        Args:
            images:    ``(B, C, H, W)`` pixel tensor.
            input_ids: ``(B, seq_len)`` token-id tensor.

        Returns:
            image_features: ``(B, 512)`` L2-normalised.
            text_features:  ``(B, 512)`` L2-normalised.
        """
        # MedCLIP's encode_image internally projects + L2-norms
        image_features = self._medclip.encode_image(images)

        # Build attention_mask from input_ids (non-padding positions)
        attention_mask = (input_ids != 0).long()
        text_features = self._medclip.encode_text(input_ids, attention_mask)

        # Ensure L2 normalisation (defensive – MedCLIP should already do it)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return image_features, text_features
