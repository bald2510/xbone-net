"""
OpenCLIP-compatible backbones for baseline comparisons.
=======================================================
Supports any model loadable via open_clip.create_model_and_transforms():
  - CLIP (OpenAI ViT-B/16)
  - PubMedCLIP (ViT-B/32, fine-tuned on ROCO)
  - BiomedCLIP (ViT-B/16, fine-tuned on PMC-15M)

Interface matches BiomedCLIPFoundation exactly.
"""

import torch.nn as nn
from open_clip import create_model_and_transforms, get_tokenizer
from transformers import CLIPModel


class PubMedCLIPTokenizerWrapper:
    """Picklable wrapper for Hugging Face CLIPTokenizer to be used in PyTorch dataloaders."""
    def __init__(self, tokenizer_obj):
        self.tokenizer_obj = tokenizer_obj

    def __call__(self, texts):
        return self.tokenizer_obj(
            texts, padding="max_length", max_length=77, truncation=True, return_tensors="pt"
        )["input_ids"]


class PubMedCLIPModel(CLIPModel):
    """CLIPModel subclass that redirects 'visual' property and __setattr__ 
    to 'vision_model' to support standard open_clip visual attribute access 
    without duplicating child submodules in PyTorch."""
    @property
    def visual(self):
        return self.vision_model
    
    @visual.setter
    def visual(self, value):
        self.vision_model = value

    def __setattr__(self, name, value):
        if name == 'visual':
            super().__setattr__('vision_model', value)
        else:
            super().__setattr__(name, value)


# Registry of supported open_clip models
OPENCLIP_MODEL_REGISTRY = {
    "clip": {
        "model_name": "ViT-B-16",
        "pretrained": "openai",
        "description": "CLIP ViT-B/16 (Radford et al., 2021)",
    },
    "pubmedclip": {
        "model_name": "hf-hub:flaviagiammarino/pubmed-clip-vit-base-patch32",
        "pretrained": None,  # Weights bundled in HF hub
        "description": "PubMedCLIP ViT-B/32 (Eslami et al., 2023)",
    },
    "biomedclip": {
        "model_name": "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        "pretrained": None,
        "description": "BiomedCLIP ViT-B/16 (Zhang et al., 2023)",
    },
}


class OpenCLIPFoundation(nn.Module):
    """
    Generic backbone for any open_clip-compatible vision-language model.

    Provides the same interface as BiomedCLIPFoundation:
      - forward(images, input_ids) -> (image_features, text_features)
      - .model, .preprocess, .tokenizer attributes
    """

    def __init__(self, model_key: str = "clip", freeze_base: bool = True):
        super().__init__()

        if model_key not in OPENCLIP_MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model_key '{model_key}'. "
                f"Choose from: {list(OPENCLIP_MODEL_REGISTRY.keys())}"
            )

        entry = OPENCLIP_MODEL_REGISTRY[model_key]
        model_name = entry["model_name"]
        pretrained = entry["pretrained"]

        if model_key == "pubmedclip":
            from transformers import CLIPTokenizer
            from torchvision import transforms
            
            print(f"[OpenCLIP] Loading PubMedCLIP via HuggingFace Transformers")
            self.model = PubMedCLIPModel.from_pretrained("flaviagiammarino/pubmed-clip-vit-base-patch32")
            
            # Fix any non-contiguous parameters to avoid Safetensors serialization error
            for name, module in self.model.named_modules():
                for param_name, param in list(module.named_parameters(recurse=False)):
                    if not param.is_contiguous():
                        new_param = nn.Parameter(param.contiguous(), requires_grad=param.requires_grad)
                        setattr(module, param_name, new_param)
            
            # Tokenizer wrapper to match open_clip interface
            tokenizer_obj = CLIPTokenizer.from_pretrained("flaviagiammarino/pubmed-clip-vit-base-patch32")
            self.tokenizer = PubMedCLIPTokenizerWrapper(tokenizer_obj)
            
            # Standard CLIP visual preprocessing
            self.preprocess = transforms.Compose([
                transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711)
                ),
            ])
            
            # Expose standard open_clip feature encoding methods and attributes
            self.model.encode_image = lambda images: self.model.get_image_features(images).pooler_output
            self.model.encode_text = lambda input_ids: self.model.get_text_features(input_ids).pooler_output
        else:
            print(f"[OpenCLIP] Loading {entry['description']}")
            print(f"  model_name: {model_name}, pretrained: {pretrained}")

            # Load model via open_clip
            if pretrained:
                self.model, _, self.preprocess = create_model_and_transforms(
                    model_name, pretrained=pretrained
                )
            else:
                # HuggingFace hub models include weights in the model name
                self.model, _, self.preprocess = create_model_and_transforms(model_name)

            self.tokenizer = get_tokenizer(model_name)

        # Freeze all backbone parameters
        if freeze_base:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, images, input_ids):
        image_features = self.model.encode_image(images)
        text_features = self.model.encode_text(input_ids)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return image_features, text_features
