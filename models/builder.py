from .backbone.biomedclip import BiomedCLIPFoundation
from .backbone.openclip_foundation import OpenCLIPFoundation
from .backbone.resnet50 import ResNet50Foundation
from .fusion import build_fusion_module
from .classifier import build_head_module
from .peft import apply_peft
from .composer import XBoneMultiModalModel

# Backbone types that use the generic OpenCLIP loader
OPENCLIP_BACKBONE_TYPES = {"clip", "pubmedclip"}

def build_model(cfg):
    """
    Builds the complete multi-modal model including backbone, PEFT modifications,
    fusion module, and classifier head.
    
    cfg: equivalent to cfg.model (contains: peft, fusion, classifier, backbone_type)
    
    Supported backbone_type values:
        - "biomedclip" (default): BiomedCLIP ViT-B/16 + PubMedBERT
        - "clip": OpenAI CLIP ViT-B/16
        - "pubmedclip": PubMedCLIP ViT-B/32 (fine-tuned on ROCO)
        - "medclip": MedCLIP Swin-Tiny + BioClinicalBERT
        - "resnet50_imagenet": ResNet-50 with ImageNet pretrained weights
        - "resnet50_medical": ResNet-50 with medical pretrained weights
    """
    # 1. Load Backbone based on config
    backbone_type = cfg.get('backbone_type', 'biomedclip')
    freeze_base = cfg.get('freeze_backbone', True)
    
    if backbone_type.startswith('resnet50'):
        # ResNet-50 image-only backbone
        pretrained = "medical" if "medical" in backbone_type else "imagenet"
        backbone = ResNet50Foundation(
            pretrained=pretrained,
            embed_dim=512,
            freeze_base=freeze_base,
        )
        print(f"[Builder] Using ResNet-50 backbone (pretrained={pretrained}, frozen={freeze_base})")
    elif backbone_type == "medclip":
        # MedCLIP: Swin-Tiny + BioClinicalBERT (Full FT, no QLoRA support)
        from .backbone.medclip_foundation import MedCLIPFoundation
        backbone = MedCLIPFoundation(freeze_base=freeze_base)
        print(f"[Builder] Using MedCLIP backbone (Swin-Tiny, frozen={freeze_base})")
    elif backbone_type in OPENCLIP_BACKBONE_TYPES:
        # CLIP, PubMedCLIP via generic OpenCLIP loader
        backbone = OpenCLIPFoundation(model_key=backbone_type, freeze_base=freeze_base)
        print(f"[Builder] Using {backbone_type} backbone (frozen={freeze_base})")
    else:
        backbone = BiomedCLIPFoundation(freeze_base=freeze_base)
        print(f"[Builder] Using BiomedCLIP backbone (frozen={freeze_base})")
    
    # 2. Extract PEFT Config (skip for ResNet-50 / MedCLIP — no LoRA adapter support)
    peft_cfg = cfg.get('peft', {'type': 'none', 'params': {}})
    peft_type = peft_cfg.get('type', 'none')
    
    # Image-only or non-open_clip backbones: only support none / full_ft
    is_non_adapter_backbone = backbone_type.startswith('resnet50') or backbone_type == 'medclip'
    if is_non_adapter_backbone:
        if peft_type not in ('none', 'full_ft'):
            print(f"[Builder] Warning: PEFT type '{peft_type}' not supported for {backbone_type}. Skipping.")
        elif peft_type == 'full_ft':
            for param in backbone.parameters():
                param.requires_grad = True
    elif peft_type in ('lora', 'qlora'):
        params = peft_cfg.get('params', {})
        # Filter out target module overrides to avoid unexpected keyword arguments in apply_peft
        common_params = {k: v for k, v in params.items() if k not in ['visual_target_modules', 'text_target_modules']}
        
        # Inject LoRA/QLoRA into visual encoder (ViT)
        visual_params = common_params.copy()
        visual_params['target_modules'] = params.get('visual_target_modules', ["qkv", "proj"])
        backbone.model.visual = apply_peft(
            module=backbone.model.visual,
            cfg={'type': peft_type, 'params': visual_params}
        )
        
        # Inject LoRA/QLoRA into text encoder (PubMedBERT / Transformer)
        # Guard: some models (e.g., OpenCLIP CLIP) don't have .text attribute
        text_module = getattr(backbone.model, 'text', None)
        if text_module is not None and 'text_target_modules' in params:
            text_params = common_params.copy()
            text_params['target_modules'] = params['text_target_modules']
            text_module.transformer = apply_peft(
                module=text_module.transformer,
                cfg={'type': peft_type, 'params': text_params}
            )
    elif peft_type == 'full_ft':
        # Full fine-tuning: unfreeze ALL parameters in both encoders
        backbone.model.visual = apply_peft(
            module=backbone.model.visual,
            cfg=peft_cfg
        )
        text_module = getattr(backbone.model, 'text', None)
        if text_module is not None:
            text_module.transformer = apply_peft(
                module=text_module.transformer,
                cfg=peft_cfg
            )
    else:
        # Fallback for baseline or other PEFT methods
        backbone.model.visual = apply_peft(
            module=backbone.model.visual, 
            cfg=peft_cfg
        )
        
    # 3. Initialize Fusion and Head
    fusion_cfg = cfg.get('fusion', {'type': 'none', 'params': {}})
    classifier_cfg = cfg.get('classifier', {'type': 'none', 'params': {}})
    
    fusion_module = build_fusion_module(fusion_cfg)
    classifier_module = build_head_module(classifier_cfg)
    
    # 4. Assemble complete model wrapper
    model = XBoneMultiModalModel(
        backbone=backbone,
        fusion_module=fusion_module,
        head_module=classifier_module
    )
    
    return model


def setup_phase2_modules(model, cfg, device):
    """
    Dynamically configures and attaches fusion and classifier modules to the model
    based on the phase2 configuration inside the hydra cfg.
    
    Returns:
        model, classifier_type, fusion_type, num_classes
    """
    # 1. Determine feature dimensions from backbone
    feature_dim = 512  # Default for both BiomedCLIP and ResNet50
    if hasattr(model.backbone, 'model') and hasattr(model.backbone.model, 'visual'):
        feature_dim = getattr(model.backbone.model.visual, 'output_dim', 512)

    # 2. Get classification classes
    dataset_params = cfg.dataset.get('params', {})
    classes = dataset_params.get('classes', dataset_params.get('pathologies', []))
    num_classes = dataset_params.get('num_classes', len(classes))

    # 3. Retrieve Phase 2 config
    params_cfg = cfg.get("params", {}) or {}
    p2_phase_cfg = params_cfg.get("phase2", {}) or {}

    backbone_type = cfg.model.get("backbone_type", "biomedclip")
    is_image_only = backbone_type.startswith("resnet50")

    # Determine types
    if is_image_only:
        fusion_type = "none"
        classifier_type = p2_phase_cfg.get("classifier_type", "linear")
        print(f"[Builder] Image-only backbone: skipping fusion, using {classifier_type} head")
    else:
        fusion_type = p2_phase_cfg.get("fusion_type", "cross_attention")
        classifier_type = p2_phase_cfg.get("classifier_type", "prototypical")

    # 4. Attach Fusion module if not already set or if it's IdentityFusion (parameters == 0)
    fusion_params = sum(p.numel() for p in model.fusion.parameters()) if model.fusion is not None else 0
    if fusion_params == 0 and fusion_type != "none":
        model.fusion = build_fusion_module({
            'type': fusion_type,
            'params': {'text_dim': feature_dim, 'img_dim': feature_dim}
        }).to(device)
        print(f"[Builder] Created {fusion_type} fusion ({feature_dim}d)")

    # 5. Attach Classifier Head if not already set or if it's IdentityHead (parameters == 0)
    head_params = sum(p.numel() for p in model.head.parameters()) if model.head is not None else 0
    if head_params == 0 and classifier_type != "none":
        model.head = build_head_module({
            'type': classifier_type,
            'params': {'feature_dim': feature_dim, 'num_classes': num_classes}
        }).to(device)
        print(f"[Builder] Created {classifier_type} head ({feature_dim} -> {num_classes} classes)")

    return model, classifier_type, fusion_type, num_classes