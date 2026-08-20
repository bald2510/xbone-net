"""Thực hiện suy luận XBone-Net cho một mẫu ảnh và văn bản lâm sàng.

Notes
-----
Mô-đun này thuộc cơ sở mã nguồn nghiên cứu XBone-Net và giữ các quy ước dùng chung của dự án.
"""

import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import hydra
from omegaconf import DictConfig

from src.models.builder import (
    build_model,
    checkpoint_model_config,
    merge_peft_adapters,
    setup_phase2_modules,
    setup_phase3_modules,
)
from src.models.fusion.cross_attention import reduce_attention_to_keys
from src.datasets.preprocessing import prepare_image
from src.utils.ood import MultimodalEnsembleOODDetector, calibrate_ood_threshold
from src.utils.trainer import resolve_pad_token_id


# ============================================================
# Thiết lập thành phần dùng chung cho quy trình xử lý của mô-đun.
# ============================================================


# ============================================================
# Kiểm tra và xử lý checkpoint tương ứng của mô hình.
# ============================================================

def adapt_state_dict_keys(
    state_dict: dict[str, torch.Tensor],
    model_keys: list[str],
) -> dict[str, torch.Tensor]:
    """Thực hiện bước adapt state dict keys trong quy trình hiện tại.

    Parameters
    ----------
    state_dict : dict[str, torch.Tensor]
        Giá trị ``state_dict`` được sử dụng trong phép xử lý.
    model_keys : list[str]
        Mô hình hoặc thành phần mô hình cần xử lý.

    Returns
    -------
    dict[str, torch.Tensor]
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    model_has_backbone_model = any(k.startswith("backbone.model.") for k in model_keys)
    checkpoint_keys = list(state_dict.keys())
    if not checkpoint_keys:
        return state_dict

    first_ckpt_key = checkpoint_keys[0]

    if model_has_backbone_model and not first_ckpt_key.startswith("backbone.model."):
        if first_ckpt_key.startswith("model."):
            print("Detected 'model.' prefix in checkpoint keys. Converting to 'backbone.model.'...")
            return {k.replace("model.", "backbone.model.", 1): v for k, v in state_dict.items()}
        elif first_ckpt_key.startswith(("visual.", "transformer.", "text.")):
            print("Detected raw submodule prefix. Prepending 'backbone.model.'...")
            return {"backbone.model." + k: v for k, v in state_dict.items()}

    return state_dict


def load_state_dict_checked(model, state_dict, context: str):
    """Tải trọng số mô hình và kiểm tra mức độ tương thích.

    Parameters
    ----------
    model : object
        Mô hình hoặc thành phần mô hình cần xử lý.
    state_dict : object
        Giá trị ``state_dict`` được sử dụng trong phép xử lý.
    context : str
        Giá trị ``context`` được sử dụng trong phép xử lý.

    Returns
    -------
    object
        Kết quả được tạo bởi bước xử lý của hàm.

    Raises
    ------
    RuntimeError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    """
    result = model.load_state_dict(state_dict, strict=False)
    missing = list(result.missing_keys)
    unexpected = list(result.unexpected_keys)

    has_fusion_in_ckpt = any(k.startswith("fusion.") for k in state_dict)
    has_head_in_ckpt = any(k.startswith("head.") for k in state_dict)
    has_lora_in_ckpt = any("lora_" in k for k in state_dict)

    critical_missing = [
        key for key in missing
        if (has_fusion_in_ckpt and key.startswith("fusion."))
        or (has_head_in_ckpt and key.startswith("head."))
        or (has_lora_in_ckpt and any(t in key for t in ("lora_A", "lora_B")))
    ]
    print(
        f" -> [{context}] checkpoint load: missing={len(missing)}, "
        f"unexpected={len(unexpected)}"
    )
    if unexpected:
        print("    Unexpected examples:", unexpected[:10])
    if critical_missing:
        raise RuntimeError(
            "Critical checkpoint weights were not loaded:\n"
            + "\n".join(critical_missing[:30])
        )
    return result


def load_checkpoint_from_dir(
    model: torch.nn.Module,
    model_dir: str,
    device: torch.device,
) -> str | None:
    """Tải checkpoint from dir cho bước xử lý hiện tại.

    Parameters
    ----------
    model : torch.nn.Module
        Mô hình hoặc thành phần mô hình cần xử lý.
    model_dir : str
        Đường dẫn tài nguyên được sử dụng.
    device : torch.device
        Thiết bị thực thi phép tính.

    Returns
    -------
    str | None
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    if not os.path.isdir(model_dir):
        return None

    checkpoint_filenames = [
        "best_phase3.pth",
        "best_phase2.pth",
    ]

    checkpoint_path = None
    for filename in checkpoint_filenames:
        path = os.path.join(model_dir, filename)
        if os.path.exists(path):
            checkpoint_path = path
            break

    if checkpoint_path:
        print(f"Loading weights from checkpoint folder: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        state_dict = adapt_state_dict_keys(state_dict, list(model.state_dict().keys()))
        load_state_dict_checked(model, state_dict, "inference")
        print("-> Checkpoint loaded successfully from folder!\n")
        return checkpoint_path
    return None


def load_model_checkpoint(
    model: torch.nn.Module,
    cfg: DictConfig,
    device: torch.device,
    custom_checkpoint_path: str = None,
) -> str:
    """Tải mô hình checkpoint cho bước xử lý hiện tại.

    Parameters
    ----------
    model : torch.nn.Module
        Mô hình hoặc thành phần mô hình cần xử lý.
    cfg : DictConfig
        Cấu hình điều khiển bước xử lý.
    device : torch.device
        Thiết bị thực thi phép tính.
    custom_checkpoint_path : str, optional
        Đường dẫn tài nguyên được sử dụng.

    Returns
    -------
    str
        Kết quả được tạo bởi bước xử lý của hàm.

    Raises
    ------
    FileNotFoundError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    """
    if custom_checkpoint_path:
        if not os.path.isfile(custom_checkpoint_path):
            raise FileNotFoundError(
                f"Requested checkpoint does not exist: {custom_checkpoint_path}"
            )
        print(f"Loading weights from custom path: {custom_checkpoint_path}")
        checkpoint = torch.load(custom_checkpoint_path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        state_dict = adapt_state_dict_keys(state_dict, list(model.state_dict().keys()))
        load_state_dict_checked(model, state_dict, "inference")
        print("-> Checkpoint loaded successfully from custom path!\n")
        return custom_checkpoint_path

    loaded_path = None
    params_cfg = cfg.get("params", {}) or {}
    model_dir = params_cfg.get("model_dir", None)

    if model_dir:
        loaded_path = load_checkpoint_from_dir(model, model_dir, device)

    if not loaded_path:
        checkpoint_path = cfg.get("checkpoint_path", None)
        if not checkpoint_path:
            checkpoint_path = params_cfg.get("checkpoint_path", None)
            if not checkpoint_path:
                phase3_cfg = params_cfg.get("phase3", {}) or {}
                checkpoint_path = phase3_cfg.get("checkpoint_path", None)
                if not checkpoint_path:
                    phase2_cfg = params_cfg.get("phase2", {}) or {}
                    checkpoint_path = phase2_cfg.get("checkpoint_path", None)

        if checkpoint_path and os.path.isfile(checkpoint_path):
            print(f"Loading weights from config checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            state_dict = adapt_state_dict_keys(state_dict, list(model.state_dict().keys()))
            load_state_dict_checked(model, state_dict, "inference")
            print("-> Checkpoint loaded successfully!\n")
            loaded_path = checkpoint_path

    if not loaded_path:
        raise FileNotFoundError(
            "No trained classifier checkpoint was found. Provide --checkpoint or "
            "set params.model_dir/phase2/phase3 checkpoint_path to a valid "
            "best_phase2.pth or best_phase3.pth file."
        )
    return loaded_path


# ============================================================
# Điểm vào chính và phân tích tham số dòng lệnh
# ============================================================

def parse_args():
    """Phân tích các tham số dòng lệnh.

    Returns
    -------
    object
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    parser = argparse.ArgumentParser(description="Multi-Modal Inference & OOD Detection")
    parser.add_argument("--image", required=True, type=str, help="Path to test image file")
    parser.add_argument("--clinical-text", type=str, default="", help="Clinical text of the patient (if applicable)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Custom checkpoint path (.pth)")
    parser.add_argument("--id-embeddings", type=str, default=None, 
                        help="Path to In-Distribution reference embeddings (.npz)")
    parser.add_argument("--ood-params", type=str, default=None,
                        help="Path to pre-computed multimodal-ensemble OOD parameters (.npz)")
    parser.add_argument("--fpr-threshold", type=float, default=0.05, 
                        help="Target False Positive Rate to calculate OOD threshold (default: 0.05)")
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda")
    parser.add_argument("--debug", action="store_true", default=False,
                        help="Print detailed model architecture layers for debugging")

    extra_args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return extra_args


args = None


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Thực thi điểm vào chính của mô-đun.

    Parameters
    ----------
    cfg : DictConfig
        Cấu hình điều khiển bước xử lý.

    Raises
    ------
    KeyError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    ValueError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    """
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Building model architecture...")
    model = build_model(checkpoint_model_config(cfg)).to(device)

    _, classifier_type, fusion_type, _ = setup_phase2_modules(model, cfg, device)
    model, _ = setup_phase3_modules(model, cfg, device)

    params_cfg = cfg.get("params", {}) or {}
    debug_mode = getattr(args, "debug", False) or params_cfg.get("debug", cfg.get("debug", False))
    model.print_architecture(verbose=debug_mode)

    model.eval()
    checkpoint_path = load_model_checkpoint(
        model, cfg, device, custom_checkpoint_path=args.checkpoint
    )
    print(f"Using trained checkpoint: {checkpoint_path}")
    merge_peft_adapters(model)


    dataset_params = cfg.dataset.get('params', {}) or {}
    classes = dataset_params.get('classes', dataset_params.get('pathologies', []))
    num_classes = dataset_params.get('num_classes', len(classes))
    is_multilabel = (dataset_params.get('task_type', 'multiclass') == 'multilabel')

    print(f"Dataset classes: {list(classes)}")
    print(f"Task type: {'Multi-label' if is_multilabel else 'Multi-class'}")

    ood_params_path = args.ood_params
    if not ood_params_path and args.checkpoint:
        checkpoint_dir = os.path.dirname(args.checkpoint)
        default_params_path = os.path.join(checkpoint_dir, "ood_parameters.npz")
        if os.path.exists(default_params_path):
            ood_params_path = default_params_path
            print(f"Auto-resolved OOD parameters to: {ood_params_path}")

    detector = None
    threshold = None

    if ood_params_path and os.path.exists(ood_params_path):
        print(f"\nLoading pre-computed OOD calibration parameters from: {ood_params_path}...")
        try:
            ood_data = np.load(ood_params_path, allow_pickle=True)
            method = str(ood_data["method"].item())
            if method != "multimodal_ensemble":
                raise ValueError(
                    f"Unsupported OOD parameter method: {method!r}."
                )
            references = np.asarray(
                ood_data["reference_visual_embeddings"],
                dtype=np.float64,
            )
            text_references = (
                np.asarray(
                    ood_data["reference_text_embeddings"],
                    dtype=np.float64,
                )
                if "reference_text_embeddings" in ood_data.files
                else None
            )
            detector = MultimodalEnsembleOODDetector(
                knn_k=int(ood_data["knn_k"].item()),
                knn_reduction=str(ood_data["knn_reduction"].item()),
                knn_metric=str(ood_data["knn_metric"].item()),
            ).fit(
                visual_embeddings=references,
                text_embeddings=text_references,
                val_visual_embeddings=references,
                val_text_embeddings=text_references,
            )
            detector.scalers["vis"] = (
                float(ood_data["visual_score_mean"].item()),
                float(ood_data["visual_score_std"].item()),
            )
            if text_references is not None:
                detector.scalers["txt"] = (
                    float(ood_data["text_score_mean"].item()),
                    float(ood_data["text_score_std"].item()),
                )
            if "calibrated_threshold" in ood_data.files:
                threshold = float(ood_data["calibrated_threshold"])
            elif "threshold" in ood_data.files:
                threshold = float(ood_data["threshold"])
            else:
                raise KeyError("OOD parameter file contains no calibrated threshold.")

            print(
                "Multimodal ensemble OOD detector initialized from "
                f"pre-computed parameters (Threshold={threshold:.4f})!"
            )
        except Exception as e:
            print(f"[Warning] Failed to load pre-computed OOD parameters: {e}. Falling back to fitting from ID embeddings.")
            detector = None

    if detector is None:
        id_embed_path = args.id_embeddings
        if not id_embed_path:
            exp_name = cfg.get("experiment_name", "default")
            seed_val = cfg.get("seed", 42)
            default_path = os.path.join("results", str(exp_name), f"seed_{seed_val}", "embeddings.npz")
            if os.path.exists(default_path):
                id_embed_path = default_path
                print(f"Auto-resolved ID reference embeddings to: {id_embed_path}")
            else:
                print(f"\n[Warning] No In-Distribution reference embeddings (.npz) found at default path '{default_path}'")
                print("OOD Detection CANNOT be run without reference embeddings or pre-computed OOD parameters. Skipping OOD checks.")

        if id_embed_path and os.path.exists(id_embed_path):
            print(f"\nFitting OOD Detector using reference: {id_embed_path}...")
            id_data = np.load(id_embed_path, allow_pickle=True)
            if "visual_global_embeddings" in id_data.files:
                reference_key = "visual_global_embeddings"
            elif "image_embeddings" in id_data.files:
                reference_key = "image_embeddings"
            elif "fused_embeddings" in id_data.files:
                reference_key = "fused_embeddings"
            else:
                raise KeyError(
                    "Reference archive has no visual_global_embeddings, "
                    "image_embeddings, or fused_embeddings array."
                )
            id_embeddings = np.asarray(id_data[reference_key], dtype=np.float64)
            id_text = (
                np.asarray(id_data["text_global_embeddings"], dtype=np.float64)
                if "text_global_embeddings" in id_data.files
                else None
            )
            detector = MultimodalEnsembleOODDetector().fit(
                visual_embeddings=id_embeddings,
                text_embeddings=id_text,
                val_visual_embeddings=id_embeddings,
                val_text_embeddings=id_text,
            )
            id_scores = detector.score(id_embeddings, text_embeddings=id_text)
            threshold = calibrate_ood_threshold(
                id_scores,
                target_id_fpr=args.fpr_threshold,
            )
            print(
                "Multimodal ensemble OOD detector fitted successfully! "
                f"Threshold (ID FPR={args.fpr_threshold:.2f}): {threshold:.4f}"
            )

    print(f"\nProcessing input image: {args.image}...")
    if not os.path.exists(args.image):
        print(f"[Error] Image file not found: {args.image}")
        return

    image = Image.open(args.image).convert("RGB")
    preprocess_cfg = cfg.dataset.get("params", {}).get("preprocess", {})
    image_tensor = prepare_image(
        image,
        model.backbone.preprocess,
        preprocess_cfg,
    ).unsqueeze(0).to(device)

    backbone_type = cfg.model.get("backbone_type", "biomedclip")
    phase2_cfg = cfg.get("params", {}).get("phase2", {}) or {}
    use_text = bool(phase2_cfg.get("use_text", True)) and not (
        backbone_type.startswith("resnet50")
        or backbone_type.startswith("densenet121")
    )

    text_tokens = None
    text_attention_mask = None
    if use_text:
        if not args.clinical_text.strip():
            raise ValueError(
                "This checkpoint requires text input. Provide --clinical-text, "
                "or evaluate an image-only configuration with phase2.use_text=false."
            )
        clinical_text = args.clinical_text.strip()
        tokenizer = model.backbone.tokenizer
        text_tokens = tokenizer([clinical_text])
        if isinstance(text_tokens, torch.Tensor):
            text_tokens = text_tokens.to(device)
            pad_id = resolve_pad_token_id(tokenizer)
            text_attention_mask = (text_tokens != pad_id).long()

    print("Running forward pass...")
    attn_info = None
    txt_mask = None
    img_mask = None

    with torch.no_grad():
        drl_score = None
        if getattr(model, "drl_auxiliary", None) is not None:
            drl_output = model.forward_drl(
                images=image_tensor,
                input_ids=text_tokens,
                attention_mask=text_attention_mask,
            )
            logits = drl_output["primary_logits"]
            fused_feats = drl_output["label_features"]
            drl_score = float(drl_output["drl_ood_score"].item())
        else:
            logits, fused_feats, _ = model(
                images=image_tensor,
                input_ids=text_tokens,
                attention_mask=text_attention_mask,
                return_features=True,
            )
        test_embed = F.normalize(fused_feats, dim=-1).cpu().numpy()

        if is_multilabel:
            probs = torch.sigmoid(logits).cpu().numpy()[0]
        else:
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        ood_visual_embedding = test_embed
        ood_text_embedding = None
        if detector is not None:
            image_features, text_features = model.backbone(
                image_tensor,
                text_tokens,
                attention_mask=text_attention_mask,
            )
            image_global = (
                image_features[:, 0]
                if image_features.ndim == 3
                else image_features
            )
            ood_visual_embedding = F.normalize(
                image_global,
                dim=-1,
            ).cpu().numpy()
            if text_features is not None:
                text_global = (
                    text_features[:, 0]
                    if text_features.ndim == 3
                    else text_features
                )
                ood_text_embedding = F.normalize(
                    text_global,
                    dim=-1,
                ).cpu().numpy()

        # Thiết lập mô-đun dung hợp và đầu phân lớp theo cấu hình.
        if use_text and getattr(model.fusion, "supports_padding_mask", False):
            img_feats, txt_feats = model.backbone(
                image_tensor,
                text_tokens,
                attention_mask=text_attention_mask,
            )
            txt_mask = None
            if text_attention_mask is not None and txt_feats.ndim == 3 and txt_feats.size(1) > 1:
                txt_mask = text_attention_mask[:, 1:] == 0
            full_img_mask = getattr(model.backbone, "last_image_key_padding_mask", None)
            img_mask = full_img_mask[:, 1:] if full_img_mask is not None else None
            _, attn_info = model.fusion(
                img_feats,
                txt_feats,
                img_key_padding_mask=img_mask,
                txt_key_padding_mask=txt_mask,
                return_attn=True,
            )

    is_ood = False
    test_score = None
    if detector is not None:
        test_score = detector.score(
            ood_visual_embedding,
            text_embeddings=ood_text_embedding,
        )[0]
        is_ood = test_score > threshold

    if attn_info is not None:
        if "fusion_gate" in attn_info:
            print(
                "Gated fusion cross-attention contribution: "
                f"{100.0 * float(attn_info['fusion_gate'].mean()):.2f}%"
            )
        img_to_txt = attn_info["attn_img_to_txt"]
        txt_to_img = attn_info["attn_txt_to_img"]
        text_attention = (
            reduce_attention_to_keys(img_to_txt, txt_mask).squeeze(0)
            if img_to_txt is not None
            else None
        )
        image_attention = (
            reduce_attention_to_keys(txt_to_img, img_mask).squeeze(0)
            if txt_to_img is not None
            else None
        )

        mutual_affinity = None
        if img_to_txt is not None and txt_to_img is not None:
            top_context = attn_info["txt_context_for_image"]
            bottom_context = attn_info["img_context_for_text"]
            mutual_affinity = F.cosine_similarity(
                top_context, bottom_context, dim=-1
            ).mean().item()

        print("\n" + "=" * 60)
        print(
            f"CROSS-ATTENTION MAP ({attn_info['attention_direction'].upper()})"
        )
        print("=" * 60)
        if text_attention is not None:
            text_top_k = min(10, text_attention.numel())
            text_weights, text_indices = torch.topk(text_attention, text_top_k)
            print("1. Image query -> most attended text-token positions:")
            print("   " + " | ".join(
                f"T{int(index) + 1}: {float(weight):.4f}"
                for weight, index in zip(text_weights, text_indices)
            ))
        if image_attention is not None:
            image_top_k = min(10, image_attention.numel())
            image_weights, image_indices = torch.topk(image_attention, image_top_k)
            print("2. Text query -> most attended visual-token positions:")
            print("   " + " | ".join(
                f"V{int(index) + 1}: {float(weight):.4f}"
                for weight, index in zip(image_weights, image_indices)
            ))
        if mutual_affinity is not None:
            print(f"3. Cross-modal context cosine affinity: {mutual_affinity:.4f}")
        print("=" * 60)

    print("\n" + "=" * 60)
    print("PREDICTION RESULTS & OOD DETECTION")
    print("=" * 60)
    if drl_score is not None:
        print(f"DRL OOD Score        : {drl_score:10.4f} (larger means more OOD)")

    if detector is not None:
        print("OOD Method          : MULTIMODAL ENSEMBLE")
        print(
            f"OOD Score           : {test_score:10.4f}  "
            f"(Threshold: {threshold:.4f})"
        )

        print("-" * 60)

        if is_ood:
            print("STATUS       : OUT-OF-DISTRIBUTION (OOD)")
            print("WARNING      : Sample is out-of-distribution. Skipping disease prediction.")
            print("=" * 60 + "\n")
            return
        else:
            print("STATUS       : IN-DISTRIBUTION (ID)")
            print("               Valid sample, proceeding with classification.")
    else:
        print("STATUS       : UNKNOWN OOD (No reference loaded)")

    print("-" * 60)
    print("Classification Details:")
    if is_multilabel:
        predicted_classes = []
        for idx, prob in enumerate(probs):
            class_name = classes[idx]
            status_flag = "[PREDICTED]" if prob >= 0.5 else "           "
            print(f"{status_flag} {class_name:25s} : Prob {prob*100:6.2f}%")
            if prob >= 0.5:
                predicted_classes.append((class_name, prob))

        print("-" * 60)
        if predicted_classes:
            predicted_str = ", ".join([f"{name} ({prob*100:.1f}%)" for name, prob in predicted_classes])
            print(f"Clinical Diagnosis: Detected {predicted_str}")
        else:
            print("Clinical Diagnosis: No abnormalities detected (Normal)")
    else:
        pred_idx = int(np.argmax(probs))
        for idx, prob in enumerate(probs):
            class_name = classes[idx]
            status_flag = "[PREDICTED]" if idx == pred_idx else "           "
            print(f"{status_flag} {class_name:25s} : Prob {prob*100:6.2f}%")

        print("-" * 60)
        print(f"Clinical Diagnosis: '{classes[pred_idx]}' ({probs[pred_idx]*100:.2f}%)")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    args = parse_args()
    main()
