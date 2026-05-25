from .backbone.biomedclip import BiomedCLIPFoundation
from .fusion import build_fusion_module
from .classifier import build_head_module
from .peft import apply_peft
from .composer import XBoneMultiModalModel

def build_model(cfg):
    """
    cfg lúc này tương đương với cfg.model (có chứa các key: peft, fusion, classifier)
    """
    # 1. Load Backbone
    backbone = BiomedCLIPFoundation(freeze_base=True)
    
    # 2. Truy cập trực tiếp vào key 'peft'. 
    # Dùng .get() với default value để phòng hờ file config bị thiếu
    peft_cfg = cfg.get('peft', {'type': 'none', 'params': {}})
    backbone.model.visual = apply_peft(
        module=backbone.model.visual, 
        cfg=peft_cfg
    )
        
    # 3. Khởi tạo Fusion và Head
    fusion_cfg = cfg.get('fusion', {'type': 'none', 'params': {}})
    classifier_cfg = cfg.get('classifier', {'type': 'none', 'params': {}})
    
    fusion_module = build_fusion_module(fusion_cfg)
    classifier_module = build_head_module(classifier_cfg)
    
    # 4. Lắp ráp
    model = XBoneMultiModalModel(
        backbone=backbone,
        fusion_module=fusion_module,
        head_module=classifier_module
    )
    
    return model