"""Cung cấp thành phần mô hình builder trong kiến trúc XBone-Net.

Notes
-----
Mô-đun này thuộc cơ sở mã nguồn nghiên cứu XBone-Net và giữ các quy ước dùng chung của dự án.
"""

from copy import deepcopy
import torch.nn as nn
from omegaconf import OmegaConf

from .backbone.biomedclip import BiomedCLIPFoundation
from .backbone.openclip_foundation import OpenCLIPFoundation
from .backbone.resnet50 import ResNet50Foundation
from .backbone.densenet import DenseNetFoundation
from .fusion import build_fusion_module
from .classifier import build_head_module
from .peft import apply_peft
from .composer import XBoneMultiModalModel
from .drl import DRLAuxiliaryBranch

# Xử lý bộ mã hóa nền tảng theo giao diện tương ứng.
OPENCLIP_BACKBONE_TYPES = {"clip", "pubmedclip"}


def resolve_phase_enabled(
    params_cfg: dict,
    phase_name: str,
    default: bool = True,
) -> bool:
    """Xác định phase enabled cho bước xử lý hiện tại.

    Parameters
    ----------
    params_cfg : dict
        Cấu hình điều khiển bước xử lý.
    phase_name : str
        Tên hoặc khóa định danh của giá trị.
    default : bool, optional
        Giá trị ``default`` được sử dụng trong phép xử lý.

    Returns
    -------
    bool
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    params_cfg = params_cfg or {}
    phase_cfg = params_cfg.get(phase_name, {}) or {}
    run_enabled = bool(params_cfg.get(f"run_{phase_name}", default))
    nested_enabled = bool(phase_cfg.get("enabled", True))
    return run_enabled and nested_enabled


def merge_peft_adapters(module: nn.Module, verbose: bool = True) -> nn.Module:
    """Hợp nhất tất cả các PEFT/LoRA adapters vào trọng số gốc của bộ mã hóa để tối ưu suy luận."""
    try:
        from peft import PeftModel
    except ImportError:
        return module

    for name, child in list(module.named_children()):
        if isinstance(child, PeftModel):
            if verbose:
                print(f"[*] Hợp nhất (Merging) PEFT adapter tại: {name}...")
            merged_child = child.merge_and_unload()
            setattr(module, name, merged_child)
        else:
            merge_peft_adapters(child, verbose=verbose)
    return module


def uses_merged_phase1_checkpoint(cfg: dict) -> bool:
    """Thực hiện bước uses merged phase1 checkpoint trong quy trình hiện tại.

    Parameters
    ----------
    cfg : dict
        Cấu hình điều khiển bước xử lý.

    Returns
    -------
    bool
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    params_cfg = cfg.get("params", {}) or {}
    phase1_cfg = params_cfg.get("phase1", {}) or {}
    phase2_cfg = params_cfg.get("phase2", {}) or {}
    return bool(
        phase1_cfg.get("merge_lora_after_training", False)
        or phase2_cfg.get("init_from_phase1_merged", False)
    )


def checkpoint_model_config(cfg: dict) -> dict:
    """Thực hiện bước checkpoint mô hình cấu hình trong quy trình hiện tại.

    Parameters
    ----------
    cfg : dict
        Cấu hình điều khiển bước xử lý.

    Returns
    -------
    dict
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    model_cfg = cfg.get("model", {}) or {}
    if OmegaConf.is_config(model_cfg):
        resolved = OmegaConf.to_container(model_cfg, resolve=True)
    else:
        resolved = deepcopy(dict(model_cfg))
    if uses_merged_phase1_checkpoint(cfg):
        resolved["peft"] = {"type": "none", "params": {}}
    return resolved


# ============================================================
# Giao diện xây dựng mô hình
# ============================================================

def build_model(cfg: dict) -> XBoneMultiModalModel:
    """Xây dựng mô hình cho bước xử lý hiện tại.

    Parameters
    ----------
    cfg : dict
        Cấu hình điều khiển bước xử lý.

    Returns
    -------
    XBoneMultiModalModel
        Kết quả được tạo bởi bước xử lý của hàm.

    Raises
    ------
    ValueError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    """
    backbone_type = cfg.get('backbone_type', 'biomedclip')
    freeze_base = cfg.get('freeze_backbone', True)
    
    # --- Khởi tạo bộ mã hóa nền tảng ---
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
        num_visual_tokens = cfg.get('num_visual_tokens', 0)
        backbone = BiomedCLIPFoundation(
            freeze_base=freeze_base,
            num_visual_tokens=num_visual_tokens,
            resampler_cfg=cfg.get('visual_resampler', {}),
            local_pool_grid=cfg.get('local_pool_grid', 2),
            single_view_pool_shape=cfg.get('single_view_pool_shape', None),
            include_local_cls_token=cfg.get('include_local_cls_token', False),
            contrastive_local_weight=cfg.get('contrastive_local_weight', 0.25),
            contrastive_pooling=cfg.get('contrastive_pooling', 'mean'),
            contrastive_attention_hidden_dim=cfg.get(
                'contrastive_attention_hidden_dim', 128
            ),
            tile_encode_chunk_size=cfg.get('tile_encode_chunk_size', 32),
            local_tile_grad_enabled=cfg.get('local_tile_grad_enabled', True),
            local_tile_gradient_checkpointing=cfg.get(
                'local_tile_gradient_checkpointing', True
            ),
        )
        print(f"[Builder] BiomedCLIP backbone (frozen={freeze_base})")
    else:
        raise ValueError(
            f"Unknown backbone_type '{backbone_type}'. "
            f"Supported options: ['biomedclip', 'pubmedclip', 'clip', 'medclip', 'resnet50_imagenet', 'densenet121_imagenet']"
        )
    
    # --- Chèn cơ chế tinh chỉnh hiệu quả tham số PEFT ---
    peft_cfg = cfg.get("peft", {"type": "none", "params": {}})
    peft_type = peft_cfg.get("type", "none")

    is_non_adapter_backbone = (
        backbone_type.startswith("resnet50")
        or backbone_type.startswith("densenet121")
        or backbone_type == "medclip"
    )

    if is_non_adapter_backbone:
        # Xử lý bộ mã hóa nền tảng theo giao diện tương ứng.
        if peft_type == "full_ft":
            for param in backbone.parameters():
                param.requires_grad = True
            print(f"[Builder] Full fine-tuning enabled for {backbone_type}")

        elif peft_type == "none":
            # Bước hỗ trợ để xây dựng model cho bước xử lý hiện tại.
            print(
                f"[Builder] No PEFT applied to {backbone_type}; "
                f"freeze_base={freeze_base}"
            )

        else:
            raise ValueError(
                f"PEFT type '{peft_type}' is not supported for "
                f"backbone '{backbone_type}'."
            )

    else:
        # Xử lý bộ mã hóa nền tảng theo giao diện tương ứng.
        if peft_type == "lora":
            params = peft_cfg.get("params", {})

            common_params = {
                k: v for k, v in params.items()
                if k not in ("visual_target_modules", "text_target_modules")
            }

            visual_params = common_params.copy()
            visual_params["target_modules"] = params.get(
                "visual_target_modules",
                ["qkv", "proj"],
            )

            backbone.model.visual = apply_peft(
                module=backbone.model.visual,
                cfg={
                    "type": peft_type,
                    "params": visual_params,
                },
            )

            text_module = getattr(
                backbone.model,
                "text",
                getattr(backbone.model, "text_model", None),
            )

            if text_module is not None:
                text_params = common_params.copy()

                if "text_target_modules" in params:
                    text_params["target_modules"] = params[
                        "text_target_modules"
                    ]

                target_text_submodule = getattr(
                    text_module,
                    "transformer",
                    text_module,
                )

                adapted_text_submodule = apply_peft(
                    module=target_text_submodule,
                    cfg={
                        "type": peft_type,
                        "params": text_params,
                    },
                )

                if hasattr(text_module, "transformer"):
                    text_module.transformer = adapted_text_submodule
                elif hasattr(backbone.model, "text_model"):
                    backbone.model.text_model = adapted_text_submodule

        elif peft_type == "full_ft":
            backbone.model.visual = apply_peft(
                module=backbone.model.visual,
                cfg=peft_cfg,
            )

            text_module = getattr(
                backbone.model,
                "text",
                getattr(backbone.model, "text_model", None),
            )

            if text_module is not None:
                target_text_submodule = getattr(
                    text_module,
                    "transformer",
                    text_module,
                )

                adapted_text_submodule = apply_peft(
                    module=target_text_submodule,
                    cfg=peft_cfg,
                )

                if hasattr(text_module, "transformer"):
                    text_module.transformer = adapted_text_submodule
                elif hasattr(backbone.model, "text_model"):
                    backbone.model.text_model = adapted_text_submodule

        elif peft_type == "none":
            # Bước hỗ trợ để xây dựng model cho bước xử lý hiện tại.
            pass

        else:
            raise ValueError(f"Unknown PEFT type: {peft_type}")

    if backbone_type == "biomedclip" and hasattr(backbone, "visual_resampler"):
        trainable_visual = sum(
            parameter.numel()
            for parameter in backbone.model.visual.parameters()
            if parameter.requires_grad
        )
        local_grad_active = (
            backbone.local_tile_grad_enabled and trainable_visual > 0
        )
        print(
            "[Builder] High-res local tile gradients: "
            f"{'enabled' if local_grad_active else 'disabled'} "
            f"(checkpointing={backbone.local_tile_gradient_checkpointing}, "
            f"trainable_visual={trainable_visual:,})"
        )
        
    # --- Khởi tạo mô-đun dung hợp và đầu phân lớp ---
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
# Tiện ích thiết lập mô-đun cho pha 2
# ============================================================

def setup_phase2_modules(model, cfg: dict, device):
    """Thực hiện bước setup phase2 modules trong quy trình hiện tại.

    Parameters
    ----------
    model : object
        Mô hình hoặc thành phần mô hình cần xử lý.
    cfg : dict
        Cấu hình điều khiển bước xử lý.
    device : object
        Thiết bị thực thi phép tính.

    Returns
    -------
    object
        Kết quả được tạo bởi bước xử lý của hàm.
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
    use_text_in_p2 = bool(p2_phase_cfg.get("use_text", True))
    use_image_in_p2 = bool(p2_phase_cfg.get("use_image", True))
    is_image_only = (
        backbone_type.startswith("resnet50")
        or backbone_type.startswith("densenet121")
        or not use_text_in_p2
    )
    is_text_only = use_text_in_p2 and not use_image_in_p2

    # Thiết lập giá trị trung gian cho bước xử lý tiếp theo.
    model_fusion_cfg = cfg.model.get("fusion", {}) or {}
    model_head_cfg = cfg.model.get("classifier", {}) or {}

    cfg_fusion_type = model_fusion_cfg.get("type", "none")
    cfg_classifier_type = model_head_cfg.get("type", "none")
    p2_fusion_type = p2_phase_cfg.get("fusion_type", "none")
    p2_classifier_type = p2_phase_cfg.get("classifier_type", "none")

    run_p2 = resolve_phase_enabled(params_cfg, "phase2", default=True)
    run_p3 = resolve_phase_enabled(params_cfg, "phase3", default=False)
    exp_name = str(cfg.get("experiment_name", "")).lower()
    is_zero_shot = (not run_p2 and not run_p3) or ("zeroshot" in exp_name)

    fusion_type = (
        p2_fusion_type if p2_fusion_type != "none" else cfg_fusion_type
    )
    classifier_type = (
        p2_classifier_type
        if p2_classifier_type != "none"
        else cfg_classifier_type
    )

    if is_zero_shot:
        fusion_type = "none"
        classifier_type = "none"
    elif is_image_only:
        fusion_type = "none"
        if classifier_type == "none":
            classifier_type = "linear"
        print(
            f"[Builder] Image-only Phase 2: fusion disabled, "
            f"using {classifier_type} head"
        )
    elif is_text_only:
        fusion_type = "none"
        if classifier_type == "none":
            classifier_type = "linear"
        print(
            f"[Builder] Text-only Phase 2: visual path disabled, "
            f"using {classifier_type} head"
        )
    else:
        if fusion_type == "none":
            fusion_type = "cross_attention"
        if classifier_type == "none":
            classifier_type = "empirical_centroid"

    # Thiết lập mô-đun dung hợp và đầu phân lớp theo cấu hình.
    # Thiết lập giá trị trung gian cho bước xử lý tiếp theo.
    fusion_params_dict = {}
    if fusion_type != "none":
        fusion_params_dict = (
            dict(model_fusion_cfg.get("params", {}))
            if fusion_type == cfg_fusion_type
            else {}
        )
        fusion_params_dict.update(dict(p2_phase_cfg.get("fusion_params", {}) or {}))
        fusion_params_dict.update({"text_dim": feature_dim, "img_dim": feature_dim})
    model.fusion = build_fusion_module({
        "type": fusion_type,
        "params": fusion_params_dict,
    }).to(device)
    print(f"[Builder] Phase-2 fusion: {fusion_type} ({feature_dim}d)")

    head_params_dict = {}
    if classifier_type != "none":
        head_params_dict = (
            dict(model_head_cfg.get("params", {}))
            if classifier_type == cfg_classifier_type
            else {}
        )
        head_params_dict.update(dict(p2_phase_cfg.get("classifier_params", {}) or {}))
        head_params_dict.update({
            "feature_dim": feature_dim,
            "num_classes": num_classes,
        })
    model.head = build_head_module({
        "type": classifier_type,
        "params": head_params_dict,
    }).to(device)
    print(
        f"[Builder] Phase-2 head: {classifier_type} "
        f"({feature_dim} -> {num_classes} classes)"
    )

    # Thiết lập mô-đun dung hợp và đầu phân lớp theo cấu hình.
    if hasattr(model.backbone, 'return_local'):
        model.backbone.return_local = fusion_type in {
            "cross_attention",
            "gated_cross_attention",
        }
        print(f"[Builder] Set backbone return_local = {model.backbone.return_local}")

    model.use_image_in_fusion = use_image_in_p2

    return model, classifier_type, fusion_type, num_classes


def setup_phase3_modules(model, cfg: dict, device):
    """Thực hiện bước setup phase3 modules trong quy trình hiện tại.

    Parameters
    ----------
    model : object
        Mô hình hoặc thành phần mô hình cần xử lý.
    cfg : dict
        Cấu hình điều khiển bước xử lý.
    device : object
        Thiết bị thực thi phép tính.

    Returns
    -------
    object
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    params_cfg = cfg.get("params", {}) or {}
    phase3_cfg = params_cfg.get("phase3", {}) or {}
    model_drl_cfg = cfg.model.get("drl", {}) or {}
    enabled = bool(model_drl_cfg.get("enabled", False)) and bool(
        phase3_cfg.get("enabled", True)
    )
    if not enabled:
        model.drl_auxiliary = None
        return model, False

    dataset_params = cfg.dataset.get("params", {}) or {}
    class_names = dataset_params.get(
        "classes", dataset_params.get("pathologies", [])
    )
    num_classes = int(dataset_params.get("num_classes", len(class_names)))
    feature_dim = int(model_drl_cfg.get("feature_dim", 512))
    module_params = dict(model_drl_cfg.get("params", {}) or {})
    module_params.update(
        {"feature_dim": feature_dim, "num_classes": num_classes}
    )
    model.drl_auxiliary = DRLAuxiliaryBranch(**module_params).to(device)
    print(
        "[Builder] Phase-3 DRL auxiliary: "
        f"{feature_dim}d -> {num_classes} classes"
    )
    return model, True
