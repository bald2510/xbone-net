"""Đánh giá mô hình phân lớp và xuất các độ đo thực nghiệm.

Notes
-----
Mô-đun này thuộc cơ sở mã nguồn nghiên cứu XBone-Net và giữ các quy ước dùng chung của dự án.
"""

import os
import sys
import json
import argparse
import datetime
from typing import Optional, Tuple, Dict, Any

# Kiểm tra điều kiện trước khi thực hiện nhánh xử lý tương ứng.
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

import torch
import torch.nn as nn
import numpy as np
import hydra
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf

from src.models.builder import (
    build_model,
    checkpoint_model_config,
    merge_peft_adapters,
    setup_phase2_modules,
    setup_phase3_modules,
)
from src.datasets.builder import build_dataloader
from src.utils.prompts import (
    ORIGINAL_CLIP_PROMPT_TEMPLATE,
    generate_clip_class_prompts,
)
from src.utils.metrics import (
    compute_metrics,
    compute_metrics_multiclass,
    bootstrap_confidence_intervals,
)


# ============================================================
# Kiểm tra và xử lý checkpoint tương ứng của mô hình.
# ============================================================

def seed_everything(seed: int = 42) -> None:
    """Cố định các nguồn ngẫu nhiên để bảo đảm khả năng tái lập.

    Parameters
    ----------
    seed : int, optional
        Hạt giống phục vụ khả năng tái lập.
    """
    import random
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def adapt_state_dict_keys(
    state_dict: Dict[str, torch.Tensor],
    model_keys: list[str],
) -> Dict[str, torch.Tensor]:
    """Thực hiện bước adapt state dict keys trong quy trình hiện tại.

    Parameters
    ----------
    state_dict : Dict[str, torch.Tensor]
        Giá trị ``state_dict`` được sử dụng trong phép xử lý.
    model_keys : list[str]
        Mô hình hoặc thành phần mô hình cần xử lý.

    Returns
    -------
    Dict[str, torch.Tensor]
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    model_has_backbone = any(k.startswith("backbone.model.") for k in model_keys)
    checkpoint_keys = list(state_dict.keys())
    if not checkpoint_keys:
        return state_dict

    first_key = checkpoint_keys[0]

    if model_has_backbone and not first_key.startswith("backbone.model."):
        if first_key.startswith("model."):
            print(" -> Converting 'model.' prefix in checkpoint keys to 'backbone.model.'")
            return {k.replace("model.", "backbone.model.", 1): v for k, v in state_dict.items()}
        elif first_key.startswith(("visual.", "transformer.", "text.")):
            print(" -> Prepending 'backbone.model.' prefix to raw OpenCLIP checkpoint keys.")
            return {"backbone.model." + k: v for k, v in state_dict.items()}

    return state_dict


def load_state_dict_checked(model: nn.Module, state_dict: Dict[str, torch.Tensor], context: str):
    """Tải trọng số mô hình và kiểm tra mức độ tương thích.

    Parameters
    ----------
    model : nn.Module
        Mô hình hoặc thành phần mô hình cần xử lý.
    state_dict : Dict[str, torch.Tensor]
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

    model_keys = set(model.state_dict().keys())
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


def load_model_checkpoint(
    model: nn.Module,
    cfg: DictConfig,
    device: torch.device,
    is_zero_shot: bool = False,
) -> bool:
    """Tải mô hình checkpoint cho bước xử lý hiện tại.

    Parameters
    ----------
    model : nn.Module
        Mô hình hoặc thành phần mô hình cần xử lý.
    cfg : DictConfig
        Cấu hình điều khiển bước xử lý.
    device : torch.device
        Thiết bị thực thi phép tính.
    is_zero_shot : bool, optional
        Giá trị ``is_zero_shot`` được sử dụng trong phép xử lý.

    Returns
    -------
    bool
        Kết quả được tạo bởi bước xử lý của hàm.

    Raises
    ------
    ValueError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    """
    if is_zero_shot:
        print(" -> [Zero-Shot Baseline] Using official pre-trained foundation weights loaded at model build time.\n")
        return True

    params_cfg = cfg.get("params", {}) or {}
    model_dir = params_cfg.get("model_dir", None)
    experiment_name = str(cfg.get("experiment_name", "default_experiment"))
    seed_val = params_cfg.get("seed", cfg.get("seed", 42))

    default_dir = os.path.join("checkpoints", experiment_name, f"seed_{seed_val}")
    if not model_dir or "//" in model_dir or "seed_/" in str(model_dir):
        model_dir = default_dir

    checkpoint_path = None
    if model_dir and os.path.isdir(model_dir):
        for candidate in ["best_phase3.pth", "best_phase2.pth", "best_phase1.pth", "checkpoint_epoch_10.pth"]:
            p = os.path.join(model_dir, candidate)
            if os.path.exists(p):
                checkpoint_path = p
                break

    if not checkpoint_path:
        checkpoint_path = cfg.get("checkpoint_path", None) or params_cfg.get("checkpoint_path", None)
        if not checkpoint_path:
            p3_cfg = params_cfg.get("phase3", {}) or {}
            checkpoint_path = p3_cfg.get("checkpoint_path", None)
            if not checkpoint_path:
                p2_cfg = params_cfg.get("phase2", {}) or {}
                checkpoint_path = p2_cfg.get("checkpoint_path", None)
                if not checkpoint_path:
                    p1_cfg = params_cfg.get("phase1", {}) or {}
                    checkpoint_path = p1_cfg.get("checkpoint_path", None)

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading weights from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        state_dict = adapt_state_dict_keys(state_dict, list(model.state_dict().keys()))

        dataset_params = cfg.dataset.get("params", {})
        classes = dataset_params.get("classes", dataset_params.get("pathologies", []))
        expected_num_classes = int(dataset_params.get("num_classes", len(classes)))
        for key, value in state_dict.items():
            if key.startswith("head.") and value.ndim > 0 and value.shape[0] != expected_num_classes:
                raise ValueError(
                    f"Checkpoint head '{key}' has shape {tuple(value.shape)} but "
                    f"the config expects {expected_num_classes} classes."
                )

        load_state_dict_checked(model, state_dict, context="evaluation")
        print(" -> Checkpoint loaded successfully!\n")
        return True
    elif is_zero_shot:
        print(" -> [Zero-Shot Baseline] Using official pre-trained foundation weights loaded at model build time.\n")
        return True

    print(f"[Warning] No checkpoint file (.pth) found. Using initial weights.\n")
    return False


# ============================================================
# Thiết lập thành phần dùng chung cho quy trình xử lý của mô-đun.
# ============================================================

def run_evaluation(
    model: nn.Module,
    test_loader,
    pathologies: list[str],
    is_classifier: bool,
    device: torch.device,
    temperature: float = 0.07,
    p2_report_type: str = "clinical",
    use_text_in_p2: bool = True,
    is_multilabel: bool = False,
    save_embeddings_flag: bool = False,
) -> Dict[str, Any]:
    """Thực hiện evaluation cho bước xử lý hiện tại.

    Parameters
    ----------
    model : nn.Module
        Mô hình hoặc thành phần mô hình cần xử lý.
    test_loader : object
        Bộ nạp dữ liệu cung cấp các batch đầu vào.
    pathologies : list[str]
        Danh sách tên bệnh lý hoặc lớp đích.
    is_classifier : bool
        Giá trị ``is_classifier`` được sử dụng trong phép xử lý.
    device : torch.device
        Thiết bị thực thi phép tính.
    temperature : float, optional
        Giá trị ``temperature`` được sử dụng trong phép xử lý.
    p2_report_type : str, optional
        Văn bản hoặc biểu diễn văn bản đầu vào.
    use_text_in_p2 : bool, optional
        Văn bản hoặc biểu diễn văn bản đầu vào.
    is_multilabel : bool, optional
        Giá trị ``is_multilabel`` được sử dụng trong phép xử lý.
    save_embeddings_flag : bool, optional
        Giá trị ``save_embeddings_flag`` được sử dụng trong phép xử lý.

    Returns
    -------
    Dict[str, Any]
        Kết quả được tạo bởi bước xử lý của hàm.

    Raises
    ------
    AttributeError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    NotImplementedError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    TypeError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    ValueError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    """
    tokenizer = getattr(model.backbone, "tokenizer", None)
    text_features_dict: Dict[str, torch.Tensor] = {}
    text_embeddings_np: Optional[Dict[str, np.ndarray]] = None

    if p2_report_type in ("both", "xray_clinical"):
        raise NotImplementedError(
            "Simultaneous token-level dual-report evaluation is not implemented. "
            "Use p2_report_type='xray' or 'clinical'."
        )

    if not is_classifier:
        if tokenizer is None:
            raise ValueError("Zero-shot evaluation requires a text tokenizer.")
        print("\nZero-shot mode: pre-extracting one text prompt per class...")
        print(f'Prompt template: "{ORIGINAL_CLIP_PROMPT_TEMPLATE}"')
        prompt_dict = generate_clip_class_prompts(pathologies)
        with torch.no_grad():
            prompt_tokens = tokenizer(list(prompt_dict.values()))
            attention_mask = None
            if isinstance(prompt_tokens, dict):
                attention_mask = prompt_tokens.get("attention_mask")
                prompt_tokens = prompt_tokens["input_ids"]
            prompt_tokens = prompt_tokens.to(device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            encoder = getattr(model.backbone, "encode_text", None)
            if encoder is None:
                encoder = getattr(getattr(model.backbone, "model", None), "encode_text", None)
            if encoder is None:
                raise AttributeError("The backbone does not expose encode_text().")
            try:
                text_matrix = encoder(prompt_tokens, attention_mask=attention_mask)
            except TypeError:
                text_matrix = encoder(prompt_tokens)
            text_matrix = torch.nn.functional.normalize(text_matrix, dim=-1)
            text_features_dict = {
                pathology: text_matrix[index : index + 1]
                for index, pathology in enumerate(pathologies)
            }

        if save_embeddings_flag:
            text_embeddings_np = {
                name: tensor.cpu().numpy() for name, tensor in text_features_dict.items()
            }

    all_probs = []
    all_ground_truths = []
    all_fused_embeddings = [] if save_embeddings_flag else None
    all_logits = [] if is_classifier else None
    has_drl = bool(is_classifier and getattr(model, "drl_auxiliary", None) is not None)
    all_distribution_embeddings = [] if save_embeddings_flag and has_drl else None
    all_auxiliary_logits = [] if has_drl else None
    all_drl_ood_scores = [] if has_drl else None
    all_fusion_gates = []

    print("\nScanning test dataset...")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            if not isinstance(batch, dict):
                raise TypeError(
                    "Evaluation expects dictionary batches from BioMedCLIPDataCollator."
                )

            images = batch["pixel_values"].to(device)
            labels = batch["labels"]
            input_ids = None
            attention_mask = None
            if is_classifier and use_text_in_p2:
                if p2_report_type == "xray":
                    input_ids = batch["xray_input_ids"].to(device)
                    attention_mask = batch.get("xray_attention_mask")
                else:
                    input_ids = batch["clinical_input_ids"].to(device)
                    attention_mask = batch.get("clinical_attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)

            if is_classifier:
                if has_drl:
                    drl_output = model.forward_drl(
                        images=images,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
                    # Thiết lập giá trị trung gian cho bước xử lý tiếp theo.
                    logits = drl_output["primary_logits"]
                    fused_features = drl_output["label_features"]
                    all_auxiliary_logits.append(drl_output["auxiliary_logits"].cpu())
                    all_drl_ood_scores.append(drl_output["drl_ood_score"].cpu())
                    if save_embeddings_flag:
                        all_distribution_embeddings.append(
                            torch.nn.functional.normalize(
                                drl_output["distribution_features"], dim=-1
                            ).cpu().numpy()
                        )
                else:
                    logits, fused_features, _ = model(
                        images=images,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        return_features=True,
                    )
                batch_probs = (
                    torch.sigmoid(logits)
                    if is_multilabel
                    else torch.softmax(logits, dim=-1)
                )
                all_logits.append(logits.cpu())
                fusion_gate = getattr(model.fusion, "last_gate", None)
                if fusion_gate is not None:
                    all_fusion_gates.append(fusion_gate.cpu().numpy())
                if save_embeddings_flag:
                    all_fused_embeddings.append(
                        torch.nn.functional.normalize(fused_features, dim=-1).cpu().numpy()
                    )
            else:
                image_encoder = getattr(getattr(model.backbone, "model", None), "encode_image", None)
                if image_encoder is None:
                    raise AttributeError("Zero-shot evaluation requires encode_image().")
                img_feat = torch.nn.functional.normalize(image_encoder(images), dim=-1)
                if save_embeddings_flag:
                    all_fused_embeddings.append(img_feat.cpu().numpy())

                class_text_features = torch.cat(
                    [text_features_dict[pathology] for pathology in pathologies],
                    dim=0,
                )
                class_logits = img_feat @ class_text_features.T
                batch_probs = torch.softmax(class_logits / temperature, dim=-1)

            all_probs.append(batch_probs.cpu())
            all_ground_truths.append(labels.cpu())

    probs_np = torch.cat(all_probs, dim=0).numpy()
    ground_truth_np = torch.cat(all_ground_truths, dim=0).numpy()
    embeddings_np = (
        np.concatenate(all_fused_embeddings, axis=0)
        if save_embeddings_flag and all_fused_embeddings
        else None
    )
    logits_np = torch.cat(all_logits, dim=0).numpy() if all_logits else None
    auxiliary_logits_np = (
        torch.cat(all_auxiliary_logits, dim=0).numpy()
        if all_auxiliary_logits
        else None
    )
    drl_ood_scores_np = (
        torch.cat(all_drl_ood_scores, dim=0).numpy()
        if all_drl_ood_scores
        else None
    )
    distribution_embeddings_np = (
        np.concatenate(all_distribution_embeddings, axis=0)
        if save_embeddings_flag and all_distribution_embeddings
        else None
    )
    fusion_gates_np = (
        np.concatenate(all_fusion_gates, axis=0)
        if all_fusion_gates
        else None
    )

    return {
        "all_probs": probs_np,
        "all_ground_truths": ground_truth_np,
        "image_embeddings": embeddings_np,  # Thu thập và xử lý biểu diễn đặc trưng của mô hình.
        "fused_embeddings": embeddings_np,
        "label_discriminative_embeddings": embeddings_np,
        "distribution_discriminative_embeddings": distribution_embeddings_np,
        "logits": logits_np,
        "drl_auxiliary_logits": auxiliary_logits_np,
        "drl_ood_scores": drl_ood_scores_np,
        "text_embeddings": text_embeddings_np,
        "fusion_gates": fusion_gates_np,
    }


# ============================================================
# Thiết lập thành phần dùng chung cho quy trình xử lý của mô-đun.
# ============================================================

def save_results_json(
    metrics: Dict[str, Any],
    cfg: DictConfig,
    output_dir: str,
    ci_95: Optional[Dict[str, Any]] = None,
) -> str:
    """Lưu các kết quả json cho bước xử lý hiện tại.

    Parameters
    ----------
    metrics : Dict[str, Any]
        Giá trị ``metrics`` được sử dụng trong phép xử lý.
    cfg : DictConfig
        Cấu hình điều khiển bước xử lý.
    output_dir : str
        Đường dẫn tài nguyên được sử dụng.
    ci_95 : Optional[Dict[str, Any]]
        Giá trị ``ci_95`` được sử dụng trong phép xử lý.

    Returns
    -------
    str
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    os.makedirs(output_dir, exist_ok=True)
    params_cfg = cfg.get("params", {}) or {}
    experiment_name = params_cfg.get("experiment_name", cfg.get("experiment_name", "unknown"))
    dataset_name = cfg.dataset.get("name", "unknown")
    seed = params_cfg.get("seed", cfg.get("seed", -1))

    result = {
        "experiment_name": str(experiment_name),
        "dataset": str(dataset_name),
        "seed": int(seed),
        "timestamp": datetime.datetime.now().isoformat(),
        "metrics": metrics,
        "config": OmegaConf.to_container(cfg, resolve=True),
    }

    if ci_95 is not None:
        result["ci_95"] = ci_95

    json_path = os.path.join(output_dir, "metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nResults saved to: {json_path}")
    return json_path


def export_embeddings(
    image_embeddings: Optional[np.ndarray],
    text_embeddings: Optional[Dict[str, np.ndarray]],
    labels: np.ndarray,
    output_dir: str,
    logits: Optional[np.ndarray] = None,
    probabilities: Optional[np.ndarray] = None,
    distribution_embeddings: Optional[np.ndarray] = None,
    auxiliary_logits: Optional[np.ndarray] = None,
    drl_ood_scores: Optional[np.ndarray] = None,
) -> Optional[str]:
    """Xuất các biểu diễn cho bước xử lý hiện tại.

    Parameters
    ----------
    image_embeddings : Optional[np.ndarray]
        Ảnh hoặc biểu diễn ảnh đầu vào.
    text_embeddings : Optional[Dict[str, np.ndarray]]
        Văn bản hoặc biểu diễn văn bản đầu vào.
    labels : np.ndarray
        Giá trị ``labels`` được sử dụng trong phép xử lý.
    output_dir : str
        Đường dẫn tài nguyên được sử dụng.
    logits : Optional[np.ndarray]
        Giá trị ``logits`` được sử dụng trong phép xử lý.
    probabilities : Optional[np.ndarray]
        Giá trị ``probabilities`` được sử dụng trong phép xử lý.
    distribution_embeddings : Optional[np.ndarray]
        Giá trị ``distribution_embeddings`` được sử dụng trong phép xử lý.
    auxiliary_logits : Optional[np.ndarray]
        Giá trị ``auxiliary_logits`` được sử dụng trong phép xử lý.
    drl_ood_scores : Optional[np.ndarray]
        Giá trị ``drl_ood_scores`` được sử dụng trong phép xử lý.

    Returns
    -------
    Optional[str]
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    if image_embeddings is None:
        print("[Warning] No embeddings to save.")
        return None

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "embeddings.npz")
    save_dict = {
        "image_embeddings": image_embeddings,
        "fused_embeddings": image_embeddings,
        "label_discriminative_embeddings": image_embeddings,
        "labels": labels,
    }
    if distribution_embeddings is not None:
        save_dict["distribution_discriminative_embeddings"] = distribution_embeddings
    if auxiliary_logits is not None:
        save_dict["drl_auxiliary_logits"] = auxiliary_logits
    if drl_ood_scores is not None:
        save_dict["drl_ood_scores"] = drl_ood_scores
    if logits is not None:
        save_dict["logits"] = logits
    if probabilities is not None:
        save_dict["probabilities"] = probabilities
    if text_embeddings is not None:
        for path_name, embedding in text_embeddings.items():
            safe_key = f"text_embeddings_{path_name.replace(' ', '_')}"
            save_dict[safe_key] = embedding

    np.savez_compressed(save_path, **save_dict)
    print(f"Embeddings saved to: {save_path}")
    print(f"  Embeddings shape: {image_embeddings.shape}")
    print(f"  Labels shape:     {labels.shape}")
    return save_path


# ============================================================
# Điểm vào chính và phân tích tham số dòng lệnh
# ============================================================

def parse_extra_args() -> argparse.Namespace:
    """Phân tích extra tham số cho bước xử lý hiện tại.

    Returns
    -------
    argparse.Namespace
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--bootstrap", action="store_true", default=False, help="Compute bootstrap 95% confidence intervals")
    parser.add_argument("--n-bootstrap", type=int, default=10_000, help="Number of bootstrap resamples (default: 10000)")
    parser.add_argument("--save-embeddings", action="store_true", default=False, help="Save image/text embeddings to .npz")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save JSON results and embeddings")
    parser.add_argument("--debug", action="store_true", default=False, help="Print detailed model architecture layers for debugging")

    extra_args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return extra_args


extra_args = parse_extra_args()


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Thực thi điểm vào chính của mô-đun.

    Parameters
    ----------
    cfg : DictConfig
        Cấu hình điều khiển bước xử lý.

    Raises
    ------
    ValueError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    """
    params_cfg = cfg.get("params", {}) or {}
    seed_val = params_cfg.get("seed", cfg.get("seed", 42))
    seed_everything(seed_val)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting evaluation on device: {device} (seed: {seed_val})")

    print("Building model architecture...")
    model = build_model(checkpoint_model_config(cfg)).to(device)

    p2_phase_cfg = params_cfg.get("phase2", {}) or {}
    model, classifier_type, fusion_type, _ = setup_phase2_modules(model, cfg, device)
    model, _ = setup_phase3_modules(model, cfg, device)

    debug_mode = extra_args.debug or params_cfg.get("debug", cfg.get("debug", False))
    model.print_architecture(verbose=debug_mode)

    pathologies = list(cfg.dataset.params.get("classes", cfg.dataset.params.get("pathologies", [])))
    is_classifier = classifier_type != "none"
    exp_name = str(params_cfg.get("experiment_name", cfg.get("experiment_name", ""))).lower()
    is_zero_shot = (not is_classifier) or ("zeroshot" in exp_name)

    # Chuẩn bị dữ liệu và chiến lược lấy mẫu tương ứng.
    # Chuẩn bị và ghi tài nguyên đầu ra theo định dạng yêu cầu.
    # Thiết lập giá trị trung gian cho bước xử lý tiếp theo.
    prompt_pathologies = pathologies
    if is_zero_shot:
        prompt_pathologies = list(cfg.dataset.params.get("prompt_classes", pathologies))
        if len(prompt_pathologies) != len(pathologies):
            raise ValueError(
                "dataset.params.prompt_classes must contain exactly one English "
                f"prompt label per dataset class ({len(pathologies)} expected, "
                f"got {len(prompt_pathologies)})."
            )

    if is_zero_shot:
        backbone_type = cfg.model.get("backbone_type", "biomedclip")
        print(f"[Zero-Shot Baseline] Using official pre-trained foundation weights for '{backbone_type}'.")

    model.eval()

    # Thiết lập trạng thái và thống kê các tham số mô hình.
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    print(f"\n  [Params] Total: {total_params:,} | Trainable: {trainable_params:,} ({100*trainable_params/total_params:.2f}%) | Frozen: {frozen_params:,}")

    ckpt_loaded = load_model_checkpoint(model, cfg, device, is_zero_shot=is_zero_shot)
    if not ckpt_loaded and not is_zero_shot:
        print("[ERROR] Cannot run evaluation for fine-tuned experiment because no trained checkpoint (.pth) was found!")
        print("        Please ensure train.py completes successfully before running evaluate.py.\n")
        sys.exit(1)

    if not is_zero_shot:
        merge_peft_adapters(model)


    print(f"Loading test dataset: {cfg.dataset.name}...")
    tokenizer_func = getattr(model.backbone, "tokenizer_obj", getattr(model.backbone, "tokenizer", None))
    test_loader = build_dataloader(
        cfg=cfg.dataset,
        split="test",
        transform=model.backbone.preprocess,
        tokenizer=tokenizer_func,
    )
    from src.utils.trainer import resolve_pad_token_id, BioMedCLIPDataCollator
    pad_id = resolve_pad_token_id(tokenizer_func) if tokenizer_func is not None else 0
    test_loader.collate_fn = BioMedCLIPDataCollator(pad_token_id=pad_id)

    task_type = cfg.dataset.params.get("task_type", "multiclass")
    is_multilabel = (task_type == "multilabel")

    eval_output = run_evaluation(
        model=model,
        test_loader=test_loader,
        pathologies=prompt_pathologies,
        is_classifier=is_classifier,
        device=device,
        temperature=params_cfg.get("temperature", 0.07),
        p2_report_type=p2_phase_cfg.get("p2_report_type", "clinical"),
        use_text_in_p2=p2_phase_cfg.get("use_text", True),
        is_multilabel=is_multilabel,
        save_embeddings_flag=extra_args.save_embeddings,
    )

    all_probs = eval_output["all_probs"]
    all_gt = eval_output["all_ground_truths"]

    if is_multilabel:
        metrics = compute_metrics(all_probs, all_gt, pathologies, is_multilabel)
    else:
        metrics = compute_metrics_multiclass(all_probs, all_gt, pathologies)

    ci_95 = None
    if extra_args.bootstrap:
        ci_95 = bootstrap_confidence_intervals(
            all_probs,
            all_gt,
            pathologies=pathologies,
            is_multilabel=is_multilabel,
            n_bootstrap=extra_args.n_bootstrap,
            seed=seed_val,
        )

    if extra_args.output_dir:
        output_dir = extra_args.output_dir
    else:
        exp_name_str = params_cfg.get("experiment_name", cfg.get("experiment_name", "default"))
        output_dir = os.path.join("results", str(exp_name_str), f"seed_{seed_val}")

    # Thiết lập trạng thái và thống kê các tham số mô hình.
    metrics["param_total"] = total_params
    metrics["param_trainable"] = trainable_params
    metrics["param_trainable_pct"] = round(100 * trainable_params / total_params, 2) if total_params > 0 else 0

    fusion_gates = eval_output.get("fusion_gates")
    if fusion_gates is not None:
        fusion_gates = np.asarray(fusion_gates, dtype=np.float64).reshape(-1)
        metrics.update({
            "fusion_gate_mean": float(fusion_gates.mean()),
            "fusion_gate_std": float(fusion_gates.std()),
            "fusion_gate_p05": float(np.quantile(fusion_gates, 0.05)),
            "fusion_gate_median": float(np.median(fusion_gates)),
            "fusion_gate_p95": float(np.quantile(fusion_gates, 0.95)),
        })

    save_results_json(metrics, cfg, output_dir, ci_95=ci_95)

    if extra_args.save_embeddings:
        export_embeddings(
            image_embeddings=eval_output["image_embeddings"],
            text_embeddings=eval_output["text_embeddings"],
            labels=all_gt,
            output_dir=output_dir,
            logits=eval_output.get("logits"),
            probabilities=all_probs,
            distribution_embeddings=eval_output.get("distribution_discriminative_embeddings"),
            auxiliary_logits=eval_output.get("drl_auxiliary_logits"),
            drl_ood_scores=eval_output.get("drl_ood_scores"),
        )

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
