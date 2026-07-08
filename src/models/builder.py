"""
Model Builder Factory for XBone-Net Architecture.
===============================================================================
Implements the builder pattern for assembling the complete XBone-Net pipeline:
  - Backbone selection (BiomedCLIP, OpenCLIP, ResNet-50, DenseNet-121, MedCLIP)
  - Parameter-Efficient Fine-Tuning (PEFT): LoRA, QLoRA, Full FT
  - Multimodal fusion module wiring (Cross-Attention, Concat, Identity)
  - Classifier head instantiation (Prototypical, Linear, Identity)
  - Phase 2 module hot-swapping (transitioning from Phase 1 to Phase 2)
"""

from .backbone.biomedclip import BiomedCLIPFoundation
from .backbone.openclip_foundation import OpenCLIPFoundation
from .backbone.resnet50 import ResNet50Foundation
from .backbone.densenet import DenseNetFoundation
from .fusion import build_fusion_module
from .classifier import build_head_module
from .peft import apply_peft
from .composer import XBoneMultiModalModel

# Backbone keys that are loaded via the unified OpenCLIPFoundation wrapper
OPENCLIP_BACKBONE_TYPES = {"clip", "pubmedclip"}


# ============================================================
# Model Builder Interface
# ============================================================

def build_model(cfg: dict) -> XBoneMultiModalModel:
    """Build the complete XBone-Net multi-modal model.

    Assembly order algorithm:
        1. Instantiate the backbone encoder based on backbone_type.
        2. Apply PEFT (LoRA, QLoRA, or full fine-tuning) to visual and text encoders.
        3. Build the fusion module (identity, cross-attention, or concat).
        4. Build the classifier head (identity, linear, or prototypical).
        5. Wrap sub-modules inside XBoneMultiModalModel composite.

    Args:
        cfg (dict): Model configuration dictionary with keys:
            - backbone_type (str): Backbone key (e.g., 'biomedclip', 'clip', 'resnet50').
            - freeze_backbone (bool): If True, freeze base backbone weights.
            - peft (dict): PEFT settings dictionary ('type', 'params').
            - fusion (dict): Fusion module settings dictionary.
            - classifier (dict): Classifier head settings dictionary.

    Returns:
        XBoneMultiModalModel: Fully assembled composite model instance ready for training.

    Raises:
        ValueError: If an invalid backbone type or PEFT type is specified.

    Example:
        >>> cfg = {
        ...     'backbone_type': 'biomedclip',
        ...     'freeze_backbone': True,
        ...     'peft': {'type': 'lora', 'params': {'r': 16, 'alpha': 32}},
        ...     'fusion': {'type': 'cross_attention', 'params': {'img_dim': 512}},
        ...     'classifier': {'type': 'prototypical', 'params': {'num_classes': 4}},
        ... }
        >>> model = build_model(cfg)
    """
    backbone_type = cfg.get('backbone_type', 'biomedclip')
    freeze_base = cfg.get('freeze_backbone', True)
    
    # --- Instantiate backbone ---
    if backbone_type.startswith('resnet50'):
        pretrained = "medical" if "medical" in backbone_type else "imagenet"
        backbone = ResNet50Foundation(
            pretrained=pretrained,
            embed_dim=512,
            freeze_base=freeze_base,
        )
        print(f"[Builder] ResNet-50 backbone (pretrained={pretrained}, frozen={freeze_base})")
    elif backbone_type.startswith('densenet121'):
        pretrained = "medical" if "medical" in backbone_type else "imagenet"
        backbone = DenseNetFoundation(
            pretrained=pretrained,
            embed_dim=512,
            freeze_base=freeze_base,
        )
        print(f"[Builder] DenseNet-121 backbone (pretrained={pretrained}, frozen={freeze_base})")
    elif backbone_type == "medclip":
        from .backbone.medclip_foundation import MedCLIPFoundation
        backbone = MedCLIPFoundation(freeze_base=freeze_base)
        print(f"[Builder] MedCLIP backbone (Swin-Tiny, frozen={freeze_base})")
    elif backbone_type in OPENCLIP_BACKBONE_TYPES:
        backbone = OpenCLIPFoundation(model_key=backbone_type, freeze_base=freeze_base)
        print(f"[Builder] {backbone_type} backbone (frozen={freeze_base})")
    elif backbone_type == "biomedclip":
        backbone = BiomedCLIPFoundation(freeze_base=freeze_base)
        print(f"[Builder] BiomedCLIP backbone (frozen={freeze_base})")
    else:
        raise ValueError(
            f"Unknown backbone_type '{backbone_type}'. "
            f"Supported options: ['biomedclip', 'pubmedclip', 'clip', 'medclip', 'resnet50_imagenet', 'densenet121_imagenet']"
        )
    
    # --- Parameter-Efficient Fine-Tuning (PEFT) injection ---
    peft_cfg = cfg.get('peft', {'type': 'none', 'params': {}})
    peft_type = peft_cfg.get('type', 'none')
    
    # Image-only and MedCLIP backbones lack OpenCLIP .visual/.text interfaces
    is_non_adapter_backbone = backbone_type.startswith('resnet50') or backbone_type.startswith('densenet121') or backbone_type == 'medclip'
    if is_non_adapter_backbone:
        if peft_type not in ('none', 'full_ft'):
            print(f"[Builder] Warning: PEFT type '{peft_type}' not supported for {backbone_type}. Skipping.")
        elif peft_type == 'full_ft':
            for param in backbone.parameters():
                param.requires_grad = True
    if backbone_type == 'clip' and peft_type == 'qlora':
        print("[Builder] Warning: OpenAI CLIP uses PyTorch native MultiheadAttention which is incompatible with 4-bit qlora. Automatically falling back to standard lora.")
        peft_type = 'lora'

    elif peft_type in ('lora', 'qlora'):
        params = peft_cfg.get('params', {})
        # Separate shared hyper-parameters from encoder-specific target lists
        common_params = {k: v for k, v in params.items() if k not in ['visual_target_modules', 'text_target_modules']}

        # Visual encoder PEFT adapter
        visual_params = common_params.copy()
        visual_params['target_modules'] = params.get('visual_target_modules', ["qkv", "proj"])
        backbone.model.visual = apply_peft(
            module=backbone.model.visual,
            cfg={'type': peft_type, 'params': visual_params}
        )
        
        # Text encoder PEFT adapter
        text_module = getattr(backbone.model, 'text', getattr(backbone.model, 'text_model', None))
        if text_module is not None:
            text_params = common_params.copy()
            if 'text_target_modules' in params:
                text_params['target_modules'] = params['text_target_modules']
            
            target_text_submodule = getattr(text_module, 'transformer', text_module)
            adapted_text_submodule = apply_peft(
                module=target_text_submodule,
                cfg={'type': peft_type, 'params': text_params}
            )
            if hasattr(text_module, 'transformer'):
                text_module.transformer = adapted_text_submodule
            elif hasattr(backbone.model, 'text_model'):
                backbone.model.text_model = adapted_text_submodule
    elif peft_type == 'full_ft':
        backbone.model.visual = apply_peft(
            module=backbone.model.visual,
            cfg=peft_cfg
        )
        text_module = getattr(backbone.model, 'text', getattr(backbone.model, 'text_model', None))
        if text_module is not None:
            target_text_submodule = getattr(text_module, 'transformer', text_module)
            adapted_text_submodule = apply_peft(
                module=target_text_submodule,
                cfg=peft_cfg
            )
            if hasattr(text_module, 'transformer'):
                text_module.transformer = adapted_text_submodule
            elif hasattr(backbone.model, 'text_model'):
                backbone.model.text_model = adapted_text_submodule
    else:
        backbone.model.visual = apply_peft(
            module=backbone.model.visual, 
            cfg=peft_cfg
        )
        
    # --- Instantiate fusion and classifier head modules ---
    fusion_cfg = cfg.get('fusion', {'type': 'none', 'params': {}})
    classifier_cfg = cfg.get('classifier', {'type': 'none', 'params': {}})
    
    fusion_module = build_fusion_module(fusion_cfg)
    classifier_module = build_head_module(classifier_cfg)
    
    return XBoneMultiModalModel(
        backbone=backbone,
        fusion_module=fusion_module,
        head_module=classifier_module
    )


# ============================================================
# Phase 2 Module Setup Helper
# ============================================================

def setup_phase2_modules(model, cfg: dict, device):
    """Hot-swap fusion and classifier modules for Phase 2 classification training.

    Phase 1 contrastive learning uses identity pass-through modules. When transitioning
    to Phase 2, this function swaps uninitialized placeholder modules with configured
    fusion (e.g., cross-attention) and head (e.g., prototypical) modules in-place.

    Args:
        model (XBoneMultiModalModel): Pre-constructed composite model (modified in-place).
        cfg (dict): Full experiment configuration containing model and dataset specs.
        device (torch.device): Computation device to place new modules on.

    Returns:
        tuple: (model, classifier_type, fusion_type, num_classes) where model is updated.
    """
    feature_dim = 512
    if hasattr(model.backbone, 'model') and hasattr(model.backbone.model, 'visual'):
        feature_dim = getattr(model.backbone.model.visual, 'output_dim', 512)

    dataset_params = cfg.dataset.get('params', {})
    classes = dataset_params.get('classes', dataset_params.get('pathologies', []))
    num_classes = dataset_params.get('num_classes', len(classes))

    params_cfg = cfg.get("params", {}) or {}
    p2_phase_cfg = params_cfg.get("phase2", {}) or {}

    backbone_type = cfg.model.get("backbone_type", "biomedclip")
    is_image_only = backbone_type.startswith("resnet50") or backbone_type.startswith("densenet121")

    # --- Resolve fusion and head types from Hydra model defaults or Phase 2 params ---
    model_fusion_cfg = cfg.model.get("fusion", {}) or {}
    model_head_cfg = cfg.model.get("classifier", {}) or {}

    cfg_fusion_type = model_fusion_cfg.get("type", "none")
    cfg_classifier_type = model_head_cfg.get("type", "none")

    p2_fusion_type = p2_phase_cfg.get("fusion_type", "none")
    p2_classifier_type = p2_phase_cfg.get("classifier_type", "none")

    run_p2 = params_cfg.get("run_phase2", True)
    exp_name = str(cfg.get("experiment_name", "")).lower()
    is_zero_shot = (not run_p2) or ("zeroshot" in exp_name)

    fusion_type = cfg_fusion_type if cfg_fusion_type != "none" else p2_fusion_type
    classifier_type = cfg_classifier_type if cfg_classifier_type != "none" else p2_classifier_type

    if is_zero_shot:
        fusion_type = "none"
        classifier_type = "none"
    elif is_image_only:
        fusion_type = "none"
        if classifier_type == "none":
            classifier_type = "linear"
        print(f"[Builder] Image-only backbone: skipping fusion, using {classifier_type} head")
    else:
        if fusion_type == "none":
            fusion_type = "cross_attention"
        if classifier_type == "none":
            classifier_type = "prototypical"

    # --- Hot-swap fusion module if uninitialized (0 params) ---
    fusion_params = sum(p.numel() for p in model.fusion.parameters()) if model.fusion is not None else 0
    if fusion_params == 0 and fusion_type != "none":
        fusion_params_dict = dict(model_fusion_cfg.get("params", {}))
        fusion_params_dict.update({'text_dim': feature_dim, 'img_dim': feature_dim})
        model.fusion = build_fusion_module({
            'type': fusion_type,
            'params': fusion_params_dict
        }).to(device)
        print(f"[Builder] Created {fusion_type} fusion ({feature_dim}d)")

    # --- Hot-swap classifier head if uninitialized (0 params) ---
    head_params = sum(p.numel() for p in model.head.parameters()) if model.head is not None else 0
    if head_params == 0 and classifier_type != "none":
        head_params_dict = dict(model_head_cfg.get("params", {}))
        head_params_dict.update({'feature_dim': feature_dim, 'num_classes': num_classes})
        model.head = build_head_module({
            'type': classifier_type,
            'params': head_params_dict
        }).to(device)
        print(f"[Builder] Created {classifier_type} head ({feature_dim} -> {num_classes} classes)")

    # --- Enable local feature extraction for bi-directional cross-attention ---
    if hasattr(model.backbone, 'return_local'):
        model.backbone.return_local = (fusion_type == "cross_attention")
        print(f"[Builder] Set backbone return_local = {model.backbone.return_local}")

    return model, classifier_type, fusion_type, num_classes
