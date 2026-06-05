from .backbone.biomedclip import BiomedCLIPFoundation
from .fusion import build_fusion_module
from .classifier import build_head_module
from .peft import apply_peft
from .composer import XBoneMultiModalModel

def build_model(cfg):
    """
    Builds the complete multi-modal model including backbone, PEFT modifications,
    fusion module, and classifier head.
    
    cfg: equivalent to cfg.model (contains: peft, fusion, classifier)
    """
    # 1. Load Backbone
    backbone = BiomedCLIPFoundation(freeze_base=True)
    
    # 2. Extract PEFT Config
    peft_cfg = cfg.get('peft', {'type': 'none', 'params': {}})
    peft_type = peft_cfg.get('type', 'none')
    
    if peft_type == 'lora':
        params = peft_cfg.get('params', {})
        # Filter out target module overrides to avoid unexpected keyword arguments in apply_peft
        common_params = {k: v for k, v in params.items() if k not in ['visual_target_modules', 'text_target_modules']}
        
        # Inject LoRA into visual encoder (ViT)
        visual_params = common_params.copy()
        visual_params['target_modules'] = params.get('visual_target_modules', ["qkv", "proj"])
        backbone.model.visual = apply_peft(
            module=backbone.model.visual,
            cfg={'type': 'lora', 'params': visual_params}
        )
        
        # Inject LoRA into text encoder (PubMedBERT / Transformer)
        if 'text_target_modules' in params:
            text_params = common_params.copy()
            text_params['target_modules'] = params['text_target_modules']
            backbone.model.text.transformer = apply_peft(
                module=backbone.model.text.transformer,
                cfg={'type': 'lora', 'params': text_params}
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