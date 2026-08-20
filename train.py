"""Huấn luyện XBone-Net theo quy trình thích nghi và phân lớp nhiều giai đoạn.

Notes
-----
Mô-đun này thuộc cơ sở mã nguồn nghiên cứu XBone-Net và giữ các quy ước dùng chung của dự án.
"""

import os
import sys
import math
import json
import time
import warnings
from typing import Optional

warnings.filterwarnings("ignore", category=FutureWarning, module="timm.*")
warnings.filterwarnings("ignore", message="triton not found")

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
from omegaconf import DictConfig, OmegaConf
from torch.optim import AdamW, SGD
import torch.nn.functional as F
from transformers import EarlyStoppingCallback, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score

from src.models.builder import (
    build_model,
    resolve_phase_enabled,
    setup_phase2_modules,
    setup_phase3_modules,
)
from src.datasets.builder import build_dataloader
from src.utils.losses import (
    build_loss,
    build_phase2_loss,
    resolve_phase2_loss_type,
)
from src.utils.trainer import BioMedCLIPDataCollator, SFTrainer, resolve_pad_token_id
from src.utils.logging import TrainingLogger, XBoneTrainerCallback
from src.utils.ood import MultimodalEnsembleOODDetector, calibrate_ood_threshold
from src.utils.centroids import (
    EmpiricalCentroidUpdateCallback,
    compute_empirical_centroids,
)


# ============================================================
# Thiết lập khả năng tái lập
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


def load_state_dict_checked(
    model: nn.Module,
    state_dict: dict,
    context: str,
    critical_substrings=("lora_A", "lora_B"),
):
    """Tải trọng số mô hình và kiểm tra mức độ tương thích.

    Parameters
    ----------
    model : nn.Module
        Mô hình hoặc thành phần mô hình cần xử lý.
    state_dict : dict
        Giá trị ``state_dict`` được sử dụng trong phép xử lý.
    context : str
        Giá trị ``context`` được sử dụng trong phép xử lý.
    critical_substrings : object, optional
        Giá trị ``critical_substrings`` được sử dụng trong phép xử lý.

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

    critical_missing = [
        key for key in missing
        if any(token in key for token in critical_substrings)
    ]

    print(
        f" -> [{context}] checkpoint load: "
        f"missing={len(missing)}, unexpected={len(unexpected)}"
    )
    if unexpected:
        print("    Unexpected examples:", unexpected[:10])
    if critical_missing:
        raise RuntimeError(
            f"Critical checkpoint weights were not loaded in {context}:\n"
            + "\n".join(critical_missing[:30])
        )
    return result


def extract_class_ids(dataset) -> np.ndarray:
    """Thực hiện bước extract class ids trong quy trình hiện tại.

    Parameters
    ----------
    dataset : object
        Dữ liệu đầu vào của bước xử lý.

    Returns
    -------
    np.ndarray
        Kết quả được tạo bởi bước xử lý của hàm.

    Raises
    ------
    AttributeError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    """
    if hasattr(dataset, "df") and "class_id" in dataset.df.columns:
        return np.asarray(dataset.df["class_id"].values, dtype=np.int64)

    if hasattr(dataset, "indices") and hasattr(dataset, "dataset"):
        base_ids = extract_class_ids(dataset.dataset)
        return base_ids[np.asarray(dataset.indices, dtype=np.int64)]

    if hasattr(dataset, "labels"):
        labels = np.asarray(dataset.labels)
        if labels.ndim > 1:
            labels = labels.argmax(axis=-1)
        return labels.astype(np.int64)

    if hasattr(dataset, "targets"):
        labels = np.asarray(dataset.targets)
        if labels.ndim > 1:
            labels = labels.argmax(axis=-1)
        return labels.astype(np.int64)

    raise AttributeError(
        "Cannot extract class IDs. Expected df['class_id'], labels, targets, "
        "or a Subset wrapper around one of these datasets."
    )


def compute_class_weights(
    class_ids: np.ndarray,
    num_classes: int,
    weight_type: str = "effective_num",
    effective_num_beta: float = 0.999,
    max_class_weight: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Tính class weights cho bước xử lý hiện tại.

    Parameters
    ----------
    class_ids : np.ndarray
        Nhãn hoặc chỉ số lớp liên quan.
    num_classes : int
        Số lượng, kích thước hoặc tỷ lệ được sử dụng.
    weight_type : str, optional
        Phương pháp hoặc chế độ xử lý được chọn.
    effective_num_beta : float, optional
        Số lượng, kích thước hoặc tỷ lệ được sử dụng.
    max_class_weight : float, optional
        Nhãn hoặc chỉ số lớp liên quan.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, list[int]]
        Kết quả được tạo bởi bước xử lý của hàm.

    Raises
    ------
    ValueError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    """
    class_ids = np.asarray(class_ids, dtype=np.int64).reshape(-1)
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2.")
    if class_ids.size == 0:
        raise ValueError(
            "Cannot compute class weights from an empty training subset."
        )
    if not np.isfinite(max_class_weight) or max_class_weight <= 0:
        raise ValueError("max_class_weight must be finite and positive.")
    if np.any(class_ids < 0) or np.any(class_ids >= num_classes):
        invalid = np.unique(
            class_ids[(class_ids < 0) | (class_ids >= num_classes)]
        ).tolist()
        raise ValueError(
            f"Training labels outside configured range [0, {num_classes - 1}]: "
            f"{invalid}"
        )

    class_counts = np.bincount(class_ids, minlength=num_classes).astype(float)
    present_mask = class_counts > 0
    missing_classes = np.flatnonzero(~present_mask).tolist()
    raw_weights = np.zeros(num_classes, dtype=np.float64)

    if weight_type == "effective_num":
        beta = float(effective_num_beta)
        if not 0.0 <= beta < 1.0:
            raise ValueError("effective_num_beta must be in [0, 1).")
        present_counts = class_counts[present_mask]
        effective_num = (1.0 - np.power(beta, present_counts)) / max(
            1.0 - beta, 1e-12
        )
        raw_weights[present_mask] = 1.0 / np.maximum(effective_num, 1e-8)
    elif weight_type in ("inverse", "inverse_frequency", "inv_freq"):
        present_count = int(present_mask.sum())
        raw_weights[present_mask] = class_counts.sum() / (
            present_count * class_counts[present_mask]
        )
    else:
        raise ValueError(
            f"Unknown class weight_type='{weight_type}'. "
            "Use effective_num or inverse_frequency."
        )

    present_weight_sum = raw_weights[present_mask].sum()
    if not np.isfinite(present_weight_sum) or present_weight_sum <= 0:
        raise ValueError(
            "Class-weight computation produced invalid present-class weights."
        )
    raw_weights[present_mask] *= present_mask.sum() / present_weight_sum
    # Bước hỗ trợ để tính class weights cho bước xử lý hiện tại.
    # Tính và giới hạn trọng số lớp cho hàm mất mát có trọng số.
    # Bước hỗ trợ để tính class weights cho bước xử lý hiện tại.
    raw_weights[present_mask] = np.minimum(
        raw_weights[present_mask], float(max_class_weight)
    )
    return raw_weights, class_counts, missing_classes


def phase_training_stats(
    trainer_metrics: dict,
    wall_clock_seconds: float,
    device: torch.device,
) -> dict:
    """Thực hiện bước phase training stats trong quy trình hiện tại.

    Parameters
    ----------
    trainer_metrics : dict
        Giá trị ``trainer_metrics`` được sử dụng trong phép xử lý.
    wall_clock_seconds : float
        Giá trị ``wall_clock_seconds`` được sử dụng trong phép xử lý.
    device : torch.device
        Thiết bị thực thi phép tính.

    Returns
    -------
    dict
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    stats = {
        key: float(value) if isinstance(value, (int, float, np.number)) else value
        for key, value in trainer_metrics.items()
    }
    stats["wall_clock_seconds"] = float(wall_clock_seconds)
    stats["gpu_hours"] = float(wall_clock_seconds / 3600.0) if device.type == "cuda" else 0.0
    if device.type == "cuda":
        divisor = 1024.0 ** 2
        stats["peak_allocated_mb"] = float(
            torch.cuda.max_memory_allocated(device) / divisor
        )
        stats["peak_reserved_mb"] = float(
            torch.cuda.max_memory_reserved(device) / divisor
        )
    return stats


# ============================================================
# Thiết lập và thực thi pha 1 căn chỉnh ảnh-văn bản.
# ============================================================

def run_phase1(
    cfg: DictConfig,
    model: nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    cp_p1: str,
    log_dir: str,
    experiment_name: str,
    use_bf16: bool,
    use_fp16: bool,
) -> tuple[nn.Module, dict]:
    """Thực hiện phase1 cho bước xử lý hiện tại.

    Parameters
    ----------
    cfg : DictConfig
        Cấu hình điều khiển bước xử lý.
    model : nn.Module
        Mô hình hoặc thành phần mô hình cần xử lý.
    train_loader : object
        Bộ nạp dữ liệu cung cấp các batch đầu vào.
    val_loader : object
        Bộ nạp dữ liệu cung cấp các batch đầu vào.
    device : torch.device
        Thiết bị thực thi phép tính.
    cp_p1 : str
        Giá trị ``cp_p1`` được sử dụng trong phép xử lý.
    log_dir : str
        Đường dẫn tài nguyên được sử dụng.
    experiment_name : str
        Tên hoặc khóa định danh của giá trị.
    use_bf16 : bool
        Giá trị ``use_bf16`` được sử dụng trong phép xử lý.
    use_fp16 : bool
        Giá trị ``use_fp16`` được sử dụng trong phép xử lý.

    Returns
    -------
    tuple[nn.Module, dict]
        Kết quả được tạo bởi bước xử lý của hàm.

    Raises
    ------
    RuntimeError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    ValueError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    """
    p1_cfg = cfg.params.phase1
    epochs_p1 = p1_cfg.get("epochs", 50)
    lr_p1 = p1_cfg.get("lr", 2e-4)
    wd_p1 = p1_cfg.get("weight_decay", 1e-2)
    patience_p1 = int(p1_cfg.get("early_stopping_patience", 5))
    gradient_accumulation_p1 = int(
        p1_cfg.get(
            "gradient_accumulation_steps",
            cfg.params.get("gradient_accumulation_steps", 1),
        )
    )
    if gradient_accumulation_p1 < 1:
        raise ValueError("Phase-1 gradient_accumulation_steps must be positive.")
    if gradient_accumulation_p1 > 1:
        print(
            "[Protocol warning] Phase-1 gradient accumulation preserves the "
            "optimizer effective batch, but the semantic-matching loss still "
            "sees only the physical micro-batch as its in-batch comparison "
            "set. Keep the physical batch size identical across runs whose "
            "Phase-1 performance is compared."
        )
    loss_type_p1 = p1_cfg.get("loss_type", "semantic_matching")
    p1_report_type = p1_cfg.get("p1_report_type", "clinical")
    class_aware_sampling = dict(p1_cfg.get("class_aware_sampling", {}) or {})

    print("\n" + "=" * 60)
    print("PHASE 1: MULTIMODAL CONTRASTIVE ALIGNMENT (backbone only)")
    print("=" * 60)

    # Thiết lập và thực thi pha 1 căn chỉnh ảnh-văn bản.
    n_backbone = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)
    n_fusion = sum(p.numel() for p in model.fusion.parameters() if p.requires_grad)
    n_head = sum(p.numel() for p in model.head.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable Parameters:")
    print(f"    Backbone  : {n_backbone:>12,}  <- contrastive alignment")
    print(f"    Fusion    : {n_fusion:>12,}  (frozen, will be reinitialized in Phase 2)")
    print(f"    Head      : {n_head:>12,}  (frozen, will be reinitialized in Phase 2)")
    print(f"    ----------------------")
    print(f"    Total     : {n_backbone:>12,} / {n_total:,} ({n_backbone/n_total*100:.2f}%)")

    logger_p1 = TrainingLogger(log_dir=log_dir, experiment_name=experiment_name, phase="phase1")
    logger_p1.log_hyperparams({
        "experiment": experiment_name,
        "phase": "phase1",
        "epochs": epochs_p1,
        "learning_rate": lr_p1,
        "weight_decay": wd_p1,
        "loss_type": loss_type_p1,
        "batch_size": cfg.dataset.batch_size,
        "micro_batch_size": cfg.dataset.batch_size,
        "gradient_accumulation_steps": gradient_accumulation_p1,
        "effective_batch_size": (
            int(cfg.dataset.batch_size) * gradient_accumulation_p1
        ),
        "optimizer_effective_batch_size": (
            int(cfg.dataset.batch_size) * gradient_accumulation_p1
        ),
        "contrastive_in_batch_size": int(cfg.dataset.batch_size),
        "class_aware_sampling": class_aware_sampling,
        "precision": "bf16" if use_bf16 else ("fp16" if use_fp16 else "fp32"),
    })
    logger_p1.log_model_summary(model)

    # Thiết lập hàm mất mát cho bước tối ưu hiện tại.
    # Thiết lập hàm mất mát cho bước tối ưu hiện tại.
    loss_fn_p1 = build_loss(
        loss_type_p1,
        clip_model=model.backbone.model,
        temperature=p1_cfg.get("temperature", 0.07),
        target_similarity=p1_cfg.get("target_similarity", 0.95),
    )

    trainable_params_p1 = [
        parameter for parameter in model.backbone.parameters()
        if parameter.requires_grad
    ]
    if not trainable_params_p1:
        raise RuntimeError("Phase 1 has no trainable backbone parameters.")
    optimizer_type_p1 = str(p1_cfg.get("optimizer", "adamw")).lower()
    if optimizer_type_p1 == "sgd":
        optimizer_p1 = SGD(
            trainable_params_p1,
            lr=lr_p1,
            weight_decay=wd_p1,
            momentum=0.9,
        )
    elif optimizer_type_p1 == "adamw":
        optimizer_p1 = AdamW(
            trainable_params_p1, lr=lr_p1, weight_decay=wd_p1
        )
    else:
        raise ValueError(f"Unsupported Phase-1 optimizer: {optimizer_type_p1}")

    batches_per_epoch_p1 = math.ceil(
        len(train_loader.dataset) / cfg.dataset.batch_size
    )
    steps_per_epoch_p1 = math.ceil(
        batches_per_epoch_p1 / gradient_accumulation_p1
    )
    total_steps_p1 = steps_per_epoch_p1 * epochs_p1
    warmup_steps_p1 = int(p1_cfg.get("warmup_ratio", 0.1) * total_steps_p1)
    tokenizer_p1 = getattr(model.backbone, "tokenizer_obj", getattr(model.backbone, "tokenizer", None))
    pad_id = resolve_pad_token_id(tokenizer_p1) if tokenizer_p1 is not None else 0

    p1_args = TrainingArguments(
        output_dir=os.path.dirname(cp_p1) if os.path.dirname(cp_p1) else "./checkpoints",
        num_train_epochs=epochs_p1,
        learning_rate=lr_p1,
        weight_decay=wd_p1,
        per_device_train_batch_size=cfg.dataset.batch_size,
        per_device_eval_batch_size=cfg.dataset.batch_size,
        gradient_accumulation_steps=gradient_accumulation_p1,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=10,
        lr_scheduler_type=p1_cfg.get("scheduler", "cosine"),
        warmup_steps=warmup_steps_p1,
        max_steps=total_steps_p1,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        save_total_limit=1,
        bf16=use_bf16,
        fp16=use_fp16,
        max_grad_norm=1.0,
        dataloader_num_workers=cfg.dataset.num_workers,
        seed=int(cfg.get("seed", 42)),
        data_seed=int(cfg.get("seed", 42)),
        report_to="none",
        remove_unused_columns=False,
    )

    trainer_p1 = SFTrainer(
        model=model,
        args=p1_args,
        train_dataset=train_loader.dataset,
        eval_dataset=val_loader.dataset,
        data_collator=BioMedCLIPDataCollator(pad_token_id=pad_id),
        callbacks=[XBoneTrainerCallback(logger_p1), EarlyStoppingCallback(early_stopping_patience=patience_p1)],
        loss_fn=loss_fn_p1,
        optimizers=(optimizer_p1, None),
        phase="phase1",
        p1_report_type=p1_report_type,
        class_aware_sampling=class_aware_sampling,
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    phase_started = time.perf_counter()
    train_output = trainer_p1.train()
    phase_stats = phase_training_stats(
        train_output.metrics,
        time.perf_counter() - phase_started,
        device,
    )
    phase_stats["gradient_accumulation_steps"] = gradient_accumulation_p1
    phase_stats["micro_batch_size"] = int(cfg.dataset.batch_size)
    phase_stats["effective_batch_size"] = (
        int(cfg.dataset.batch_size) * gradient_accumulation_p1
    )
    phase_stats["optimizer_effective_batch_size"] = (
        int(cfg.dataset.batch_size) * gradient_accumulation_p1
    )
    phase_stats["contrastive_in_batch_size"] = int(cfg.dataset.batch_size)
    phase_stats["batch_protocol_note"] = (
        "Gradient accumulation does not enlarge the Phase-1 in-batch "
        "contrastive comparison set."
    )

    os.makedirs(os.path.dirname(cp_p1) or ".", exist_ok=True)
    best_model_p1 = trainer_p1.model
    torch.save(best_model_p1.state_dict(), cp_p1)
    print(f" [*] Saved best Phase 1 checkpoint: {cp_p1} (eval_loss={trainer_p1.state.best_metric})")

    logger_p1.close()
    print("Phase 1 complete!\n")
    return model, phase_stats


# ============================================================
# Thiết lập và thực thi pha 2 phân lớp đa phương thức.
# ============================================================

def run_phase2(
    cfg: DictConfig,
    model: nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    cp_p2: str,
    log_dir: str,
    experiment_name: str,
    use_bf16: bool,
    use_fp16: bool,
    classifier_type: str = "empirical_centroid",
) -> tuple[nn.Module, dict]:
    """Thực hiện phase2 cho bước xử lý hiện tại.

    Parameters
    ----------
    cfg : DictConfig
        Cấu hình điều khiển bước xử lý.
    model : nn.Module
        Mô hình hoặc thành phần mô hình cần xử lý.
    train_loader : object
        Bộ nạp dữ liệu cung cấp các batch đầu vào.
    val_loader : object
        Bộ nạp dữ liệu cung cấp các batch đầu vào.
    device : torch.device
        Thiết bị thực thi phép tính.
    cp_p2 : str
        Giá trị ``cp_p2`` được sử dụng trong phép xử lý.
    log_dir : str
        Đường dẫn tài nguyên được sử dụng.
    experiment_name : str
        Tên hoặc khóa định danh của giá trị.
    use_bf16 : bool
        Giá trị ``use_bf16`` được sử dụng trong phép xử lý.
    use_fp16 : bool
        Giá trị ``use_fp16`` được sử dụng trong phép xử lý.
    classifier_type : str, optional
        Phương pháp hoặc chế độ xử lý được chọn.

    Returns
    -------
    tuple[nn.Module, dict]
        Kết quả được tạo bởi bước xử lý của hàm.

    Raises
    ------
    RuntimeError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    ValueError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    """
    p2_cfg = cfg.params.phase2
    epochs_p2 = p2_cfg.get("epochs", 50)
    lr_p2 = p2_cfg.get("lr", 1e-3)
    wd_p2 = p2_cfg.get("weight_decay", 1e-4)
    patience_p2 = int(p2_cfg.get("early_stopping_patience", 5))
    gradient_accumulation_p2 = int(
        p2_cfg.get(
            "gradient_accumulation_steps",
            cfg.params.get("gradient_accumulation_steps", 1),
        )
    )
    if gradient_accumulation_p2 < 1:
        raise ValueError("Phase-2 gradient_accumulation_steps must be positive.")
    use_text_p2 = bool(p2_cfg.get("use_text", True))
    report_type_p2 = str(p2_cfg.get("p2_report_type", "clinical"))
    uses_empirical_centroids = classifier_type == "empirical_centroid"
    loss_type_p2 = resolve_phase2_loss_type(
        p2_cfg.get("loss_type", None), classifier_type
    )

    print("\n" + "=" * 60)
    print("PHASE 2: CLASSIFIER AND FUSION TRAINING")
    print("=" * 60)

    # Thiết lập và thực thi pha 2 phân lớp đa phương thức.
    n_backbone = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)
    n_fusion = sum(p.numel() for p in model.fusion.parameters() if p.requires_grad)
    n_head = sum(p.numel() for p in model.head.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = n_backbone + n_fusion + n_head
    peft_type = cfg.model.peft.get("type", "none")

    if n_backbone == 0:
        backbone_status = "FROZEN"
    elif peft_type == "lora":
        backbone_status = "BASE FROZEN, ADAPTERS TRAINABLE"
    elif peft_type == "full_ft":
        backbone_status = "TRAINABLE (FULL FINE-TUNING)"
    else:
        backbone_status = "FOUNDATION FROZEN, AUXILIARY MODULES TRAINABLE"
    print(f"  Backbone    : {backbone_status}")
    print(f"  Trainable Parameters:")
    print(f"    Backbone  : {n_backbone:>12,}")
    print(f"    Fusion    : {n_fusion:>12,}")
    print(f"    Head      : {n_head:>12,}")
    print(f"    ----------------------")
    print(f"    Total     : {n_trainable:>12,} / {n_total:,} ({n_trainable/n_total*100:.2f}%)")

    logger_p2 = TrainingLogger(log_dir=log_dir, experiment_name=experiment_name, phase="phase2")
    logger_p2.log_hyperparams({
        "experiment": experiment_name,
        "phase": "phase2",
        "epochs": epochs_p2,
        "learning_rate": lr_p2,
        "weight_decay": wd_p2,
        "batch_size": cfg.dataset.batch_size,
        "micro_batch_size": cfg.dataset.batch_size,
        "gradient_accumulation_steps": gradient_accumulation_p2,
        "effective_batch_size": (
            int(cfg.dataset.batch_size) * gradient_accumulation_p2
        ),
        "optimizer_effective_batch_size": (
            int(cfg.dataset.batch_size) * gradient_accumulation_p2
        ),
        "precision": "bf16" if use_bf16 else ("fp16" if use_fp16 else "fp32"),
        "loss_type": loss_type_p2,
    })
    logger_p2.log_model_summary(model)

    # Thiết lập trạng thái và thống kê các tham số mô hình.
    # Kiểm tra điều kiện trước khi thực hiện nhánh xử lý tương ứng.
    if uses_empirical_centroids:
        initial_counts = compute_empirical_centroids(
            model,
            train_loader,
            device,
            use_text=use_text_p2,
            report_type=report_type_p2,
        )
        print(
            "  [Centroids] Initialized from the training subset; "
            f"class counts={initial_counts.tolist()}"
        )

    # --- Hàm mất mát ---
    loss_cfg = p2_cfg.get("loss", {}) or {}
    use_class_weights = loss_cfg.get("use_class_weights", False)

    # Tính và giới hạn trọng số lớp cho hàm mất mát có trọng số.
    class_weights_tensor = None
    if use_class_weights:
        class_ids = extract_class_ids(train_loader.dataset)
        dataset_params = cfg.dataset.get("params", {})
        class_names = dataset_params.get(
            "classes", dataset_params.get("pathologies", [])
        )
        configured_num_classes = int(
            dataset_params.get(
                "num_classes",
                len(class_names) if class_names else int(class_ids.max() + 1),
            )
        )
        num_classes = configured_num_classes
        weight_type = loss_cfg.get("weight_type", "effective_num")
        beta = float(loss_cfg.get("effective_num_beta", 0.999))
        max_weight = float(loss_cfg.get("max_class_weight", 10.0))
        raw_weights, class_counts, missing_classes = compute_class_weights(
            class_ids=class_ids,
            num_classes=num_classes,
            weight_type=weight_type,
            effective_num_beta=beta,
            max_class_weight=max_weight,
        )

        missing_policy = str(
            dataset_params.get("missing_train_class_policy", "error")
        ).lower()
        if missing_classes and missing_policy == "error":
            raise ValueError(
                "Class-weighted training requires every configured class to be "
                f"present in the training subset. Missing classes: {missing_classes}. "
                "Set dataset.params.missing_train_class_policy=zero_weight only "
                "when unseen classes are intentional and will be reported."
            )
        if missing_classes and missing_policy == "zero_weight":
            print(
                "  [Loss][WARNING] Configured classes absent from the training "
                f"subset: {missing_classes}. Their target weights are zero; "
                "evaluation must report them as unseen classes."
            )
        elif missing_classes:
            raise ValueError(
                f"Unknown missing_train_class_policy='{missing_policy}'. "
                "Use error or zero_weight."
            )

        if weight_type == "effective_num":
            print(
                "  [Loss] Effective-number class weights "
                f"(beta={beta}, cap={max_weight}):"
            )
        else:
            print(f"  [Loss] Inverse-frequency class weights (cap={max_weight}):")

        class_weights_tensor = torch.tensor(
            raw_weights, dtype=torch.float32, device=device
        )
        for class_id, weight in enumerate(raw_weights):
            name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
            print(f"    {name:30s}: {weight:.3f} (n={int(class_counts[class_id])})")

    criterion_p2 = build_phase2_loss(
        loss_type=loss_type_p2,
        classifier_type=classifier_type,
        class_weights=class_weights_tensor,
        label_smoothing=loss_cfg.get("label_smoothing", 0.0),
    )
    print(
        f"  [Loss] Phase-2 objective: {loss_type_p2} "
        f"({criterion_p2.__class__.__name__})"
    )

    trainable_params_p2 = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params_p2:
        raise RuntimeError("Phase 2 has no trainable parameters.")
    optimizer_type_p2 = p2_cfg.get("optimizer", "adamw")
    if optimizer_type_p2 == "sgd":
        optimizer_p2 = SGD(trainable_params_p2, lr=lr_p2, weight_decay=wd_p2, momentum=0.9)
    else:
        optimizer_p2 = AdamW(trainable_params_p2, lr=lr_p2, weight_decay=wd_p2)

    batches_per_epoch_p2 = math.ceil(
        len(train_loader.dataset) / cfg.dataset.batch_size
    )
    steps_per_epoch_p2 = math.ceil(
        batches_per_epoch_p2 / gradient_accumulation_p2
    )
    total_steps_p2 = steps_per_epoch_p2 * epochs_p2
    warmup_steps_p2 = int(p2_cfg.get("warmup_ratio", 0.1) * total_steps_p2)

    tokenizer = getattr(model.backbone, "tokenizer_obj", None)
    pad_id = resolve_pad_token_id(tokenizer) if tokenizer is not None else 0
    os.environ["TENSORBOARD_LOGGING_DIR"] = logger_p2.tb_dir

    def compute_metrics_eval(eval_pred):
        """Tính các độ đo cho bước đánh giá mô hình.

        Parameters
        ----------
        eval_pred : object
            Giá trị ``eval_pred`` được sử dụng trong phép xử lý.

        Returns
        -------
        object
            Kết quả được tạo bởi bước xử lý của hàm.
        """
        preds, labels = eval_pred

        if isinstance(preds, tuple):
            preds = preds[0]

        if labels.ndim > 1:
            labels = np.argmax(labels, axis=-1)

        preds_cls = (
            np.argmax(preds, axis=1)
            if preds.ndim > 1
            else (preds > 0).astype(int)
        )

        return {
            "accuracy": accuracy_score(labels, preds_cls),
            "f1_macro": f1_score(
                labels,
                preds_cls,
                average="macro",
                zero_division=0,
            ),
        }

    p2_args = TrainingArguments(
        output_dir=os.path.dirname(cp_p2) if os.path.dirname(cp_p2) else "./checkpoints",
        num_train_epochs=epochs_p2,
        learning_rate=lr_p2,
        weight_decay=wd_p2,
        per_device_train_batch_size=cfg.dataset.batch_size,
        per_device_eval_batch_size=cfg.dataset.batch_size,
        gradient_accumulation_steps=gradient_accumulation_p2,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=10,
        lr_scheduler_type=p2_cfg.get("scheduler", "cosine"),
        warmup_steps=warmup_steps_p2,
        max_steps=total_steps_p2,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=1,
        bf16=use_bf16,
        fp16=use_fp16,
        max_grad_norm=1.0,
        dataloader_num_workers=cfg.dataset.num_workers,
        seed=int(cfg.get("seed", 42)),
        data_seed=int(cfg.get("seed", 42)),
        report_to="none",
        remove_unused_columns=False,
    )

    callbacks = [
        XBoneTrainerCallback(logger_p2),
        EarlyStoppingCallback(
            early_stopping_patience=patience_p2
        ),
    ]
    if uses_empirical_centroids:
        centroid_cfg = p2_cfg.get("centroid", {}) or {}
        callbacks.append(
            EmpiricalCentroidUpdateCallback(
                model=model,
                data_loader=train_loader,
                device=device,
                use_text=use_text_p2,
                report_type=report_type_p2,
                interval_epochs=int(
                    centroid_cfg.get("update_interval_epochs", 1)
                ),
            )
        )

    trainer_p2 = SFTrainer(
        model=model,
        args=p2_args,
        train_dataset=train_loader.dataset,
        eval_dataset=val_loader.dataset,
        data_collator=BioMedCLIPDataCollator(pad_token_id=pad_id),
        compute_metrics=compute_metrics_eval,
        callbacks=callbacks,
        loss_fn=criterion_p2,
        optimizers=(optimizer_p2, None),
        phase="phase2",
        use_text_in_p2=use_text_p2,
        p2_report_type=report_type_p2,
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    phase_started = time.perf_counter()
    train_output = trainer_p2.train()
    phase_stats = phase_training_stats(
        train_output.metrics,
        time.perf_counter() - phase_started,
        device,
    )
    phase_stats["gradient_accumulation_steps"] = gradient_accumulation_p2
    phase_stats["micro_batch_size"] = int(cfg.dataset.batch_size)
    phase_stats["effective_batch_size"] = (
        int(cfg.dataset.batch_size) * gradient_accumulation_p2
    )
    phase_stats["optimizer_effective_batch_size"] = (
        int(cfg.dataset.batch_size) * gradient_accumulation_p2
    )

    os.makedirs(os.path.dirname(cp_p2) or ".", exist_ok=True)
    best_model_p2 = trainer_p2.model
    if uses_empirical_centroids:
        # Kiểm tra và xử lý checkpoint tương ứng của mô hình.
        # Kiểm tra và xử lý checkpoint tương ứng của mô hình.
        final_counts = compute_empirical_centroids(
            best_model_p2,
            train_loader,
            device,
            use_text=use_text_p2,
            report_type=report_type_p2,
        )
        print(
            "  [Centroids] Recomputed for the best Phase-2 encoder; "
            f"class counts={final_counts.tolist()}"
        )
    torch.save(best_model_p2.state_dict(), cp_p2)
    print(f" [*] Saved best Phase 2 checkpoint: {cp_p2} (eval_f1={trainer_p2.state.best_metric})")

    logger_p2.close()
    print("Phase 2 complete!\n")
    return best_model_p2, phase_stats


# ============================================================
# Thiết lập và thực thi pha 3 học biểu diễn bổ sung.
# ============================================================

def _phase_text_inputs(batch: dict, report_type: str, device: torch.device):
    """Thực hiện bước phase văn bản inputs trong quy trình hiện tại.

    Parameters
    ----------
    batch : dict
        Batch dữ liệu đầu vào.
    report_type : str
        Văn bản hoặc biểu diễn văn bản đầu vào.
    device : torch.device
        Thiết bị thực thi phép tính.

    Returns
    -------
    object
        Kết quả được tạo bởi bước xử lý của hàm.

    Raises
    ------
    ValueError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    """
    if report_type not in {"xray", "clinical"}:
        raise ValueError("Phase 3 supports report_type='xray' or 'clinical'.")
    prefix = "xray" if report_type == "xray" else "clinical"
    input_ids = batch[f"{prefix}_input_ids"].to(device)
    attention_mask = batch.get(f"{prefix}_attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    return input_ids, attention_mask


@torch.no_grad()
def estimate_drl_label_covariance(
    model: nn.Module,
    train_loader,
    device: torch.device,
    report_type: str,
) -> dict:
    """Ước lượng drl nhãn covariance cho bước xử lý hiện tại.

    Parameters
    ----------
    model : nn.Module
        Mô hình hoặc thành phần mô hình cần xử lý.
    train_loader : object
        Bộ nạp dữ liệu cung cấp các batch đầu vào.
    device : torch.device
        Thiết bị thực thi phép tính.
    report_type : str
        Văn bản hoặc biểu diễn văn bản đầu vào.

    Returns
    -------
    dict
        Kết quả được tạo bởi bước xử lý của hàm.

    Raises
    ------
    RuntimeError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    ValueError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    """
    if model.drl_auxiliary is None:
        raise RuntimeError("Cannot estimate DRL covariance without an auxiliary branch.")
    model.eval()
    label_features = []
    for batch in train_loader:
        images = batch["pixel_values"].to(device)
        input_ids, attention_mask = _phase_text_inputs(
            batch, report_type, device
        )
        features = model.encode_fused(
            images,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        label_features.append(features.detach().float().cpu())
    if not label_features:
        raise ValueError("Cannot estimate DRL covariance from an empty loader.")
    matrix = torch.cat(label_features, dim=0).to(torch.float64)
    if matrix.size(0) < 2:
        raise ValueError("DRL covariance requires at least two training samples.")
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / float(matrix.size(0) - 1)
    model.drl_auxiliary.set_label_covariance(covariance.float().to(device))
    return {
        "samples": int(matrix.size(0)),
        "dimension": int(matrix.size(1)),
        "trace": float(torch.trace(covariance).item()),
    }


def run_phase3(
    cfg: DictConfig,
    model: nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    cp_p3: str,
    log_dir: str,
    experiment_name: str,
    use_bf16: bool,
    use_fp16: bool,
) -> tuple[nn.Module, dict]:
    """Thực hiện phase3 cho bước xử lý hiện tại.

    Parameters
    ----------
    cfg : DictConfig
        Cấu hình điều khiển bước xử lý.
    model : nn.Module
        Mô hình hoặc thành phần mô hình cần xử lý.
    train_loader : object
        Bộ nạp dữ liệu cung cấp các batch đầu vào.
    val_loader : object
        Bộ nạp dữ liệu cung cấp các batch đầu vào.
    device : torch.device
        Thiết bị thực thi phép tính.
    cp_p3 : str
        Giá trị ``cp_p3`` được sử dụng trong phép xử lý.
    log_dir : str
        Đường dẫn tài nguyên được sử dụng.
    experiment_name : str
        Tên hoặc khóa định danh của giá trị.
    use_bf16 : bool
        Giá trị ``use_bf16`` được sử dụng trong phép xử lý.
    use_fp16 : bool
        Giá trị ``use_fp16`` được sử dụng trong phép xử lý.

    Returns
    -------
    tuple[nn.Module, dict]
        Kết quả được tạo bởi bước xử lý của hàm.

    Raises
    ------
    RuntimeError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    ValueError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    """
    p3_cfg = cfg.params.phase3
    epochs = int(p3_cfg.get("epochs", 20))
    learning_rate = float(p3_cfg.get("lr", 1e-4))
    weight_decay = float(p3_cfg.get("weight_decay", 1e-4))
    patience = int(p3_cfg.get("early_stopping_patience", 5))
    gradient_accumulation = int(
        p3_cfg.get(
            "gradient_accumulation_steps",
            cfg.params.get("gradient_accumulation_steps", 1),
        )
    )
    if gradient_accumulation < 1:
        raise ValueError("Phase-3 gradient_accumulation_steps must be positive.")
    report_type = str(p3_cfg.get("report_type", "clinical"))

    print("\n" + "=" * 60)
    print("PHASE 3: DUAL REPRESENTATION AUXILIARY TRAINING")
    print("=" * 60)

    for parameter in model.parameters():
        parameter.requires_grad = False
    if model.drl_auxiliary is None:
        raise RuntimeError("Phase 3 requested but no DRL auxiliary branch is attached.")
    for parameter in model.drl_auxiliary.parameters():
        parameter.requires_grad = True
    model.phase3_mode = True

    covariance_stats = estimate_drl_label_covariance(
        model,
        train_loader,
        device,
        report_type,
    )
    print(
        "  [DRL] Label covariance estimated from "
        f"{covariance_stats['samples']} samples "
        f"({covariance_stats['dimension']}d, trace={covariance_stats['trace']:.4f})."
    )

    loss_cfg = p3_cfg.get("loss", {}) or {}
    class_weights_tensor = None
    if bool(loss_cfg.get("use_class_weights", True)):
        class_ids = extract_class_ids(train_loader.dataset)
        dataset_params = cfg.dataset.get("params", {}) or {}
        class_names = dataset_params.get(
            "classes", dataset_params.get("pathologies", [])
        )
        num_classes = int(dataset_params.get("num_classes", len(class_names)))
        raw_weights, _, missing_classes = compute_class_weights(
            class_ids=class_ids,
            num_classes=num_classes,
            weight_type=str(loss_cfg.get("weight_type", "effective_num")),
            effective_num_beta=float(
                loss_cfg.get("effective_num_beta", 0.999)
            ),
            max_class_weight=float(loss_cfg.get("max_class_weight", 5.0)),
        )
        if missing_classes:
            raise ValueError(
                "DRL auxiliary training requires all configured classes in the "
                f"training split; missing {missing_classes}."
            )
        class_weights_tensor = torch.tensor(
            raw_weights, dtype=torch.float32, device=device
        )
    criterion = nn.CrossEntropyLoss(
        weight=class_weights_tensor,
        label_smoothing=float(loss_cfg.get("label_smoothing", 0.05)),
    )

    trainable = [
        parameter
        for parameter in model.drl_auxiliary.parameters()
        if parameter.requires_grad
    ]
    optimizer_name = str(p3_cfg.get("optimizer", "adamw")).lower()
    if optimizer_name == "sgd":
        optimizer = SGD(
            trainable,
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=0.9,
        )
    elif optimizer_name == "adamw":
        optimizer = AdamW(
            trainable,
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"Unsupported Phase-3 optimizer: {optimizer_name}")

    total_steps = math.ceil(
        math.ceil(len(train_loader.dataset) / cfg.dataset.batch_size)
        / gradient_accumulation
    ) * epochs
    warmup_steps = int(float(p3_cfg.get("warmup_ratio", 0.1)) * total_steps)
    tokenizer = getattr(
        model.backbone,
        "tokenizer_obj",
        getattr(model.backbone, "tokenizer", None),
    )
    pad_id = resolve_pad_token_id(tokenizer) if tokenizer is not None else 0

    logger = TrainingLogger(
        log_dir=log_dir,
        experiment_name=experiment_name,
        phase="phase3",
    )
    logger.log_hyperparams({
        "experiment": experiment_name,
        "phase": "phase3",
        "epochs": epochs,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "batch_size": cfg.dataset.batch_size,
        "gradient_accumulation_steps": gradient_accumulation,
        "report_type": report_type,
        "epsilon": float(model.drl_auxiliary.epsilon),
        "covariance_trace": covariance_stats["trace"],
    })
    logger.log_model_summary(model)

    def compute_metrics_eval(eval_pred):
        """Tính các độ đo cho bước đánh giá mô hình.

        Parameters
        ----------
        eval_pred : object
            Giá trị ``eval_pred`` được sử dụng trong phép xử lý.

        Returns
        -------
        object
            Kết quả được tạo bởi bước xử lý của hàm.
        """
        predictions, labels = eval_pred
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        if labels.ndim > 1:
            labels = np.argmax(labels, axis=-1)
        predicted_classes = np.argmax(predictions, axis=1)
        return {
            "accuracy": accuracy_score(labels, predicted_classes),
            "f1_macro": f1_score(
                labels, predicted_classes, average="macro", zero_division=0
            ),
        }

    args = TrainingArguments(
        output_dir=os.path.dirname(cp_p3) if os.path.dirname(cp_p3) else "./checkpoints",
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        per_device_train_batch_size=cfg.dataset.batch_size,
        per_device_eval_batch_size=cfg.dataset.batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=10,
        lr_scheduler_type=str(p3_cfg.get("scheduler", "cosine")),
        warmup_steps=warmup_steps,
        max_steps=total_steps,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=1,
        bf16=use_bf16,
        fp16=use_fp16,
        max_grad_norm=1.0,
        dataloader_num_workers=cfg.dataset.num_workers,
        seed=int(cfg.get("seed", 42)),
        data_seed=int(cfg.get("seed", 42)),
        report_to="none",
        remove_unused_columns=False,
    )
    trainer = SFTrainer(
        model=model,
        args=args,
        train_dataset=train_loader.dataset,
        eval_dataset=val_loader.dataset,
        data_collator=BioMedCLIPDataCollator(pad_token_id=pad_id),
        compute_metrics=compute_metrics_eval,
        callbacks=[
            XBoneTrainerCallback(logger),
            EarlyStoppingCallback(early_stopping_patience=patience),
        ],
        loss_fn=criterion,
        optimizers=(optimizer, None),
        phase="phase3",
        use_text_in_p2=True,
        p2_report_type=report_type,
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    train_output = trainer.train()
    phase_stats = phase_training_stats(
        train_output.metrics,
        time.perf_counter() - started,
        device,
    )
    phase_stats.update({
        "gradient_accumulation_steps": gradient_accumulation,
        "micro_batch_size": int(cfg.dataset.batch_size),
        "effective_batch_size": int(cfg.dataset.batch_size) * gradient_accumulation,
        "covariance": covariance_stats,
    })

    os.makedirs(os.path.dirname(cp_p3) or ".", exist_ok=True)
    best_model = trainer.model
    torch.save(best_model.state_dict(), cp_p3)
    print(
        f" [*] Saved best Phase 3 checkpoint: {cp_p3} "
        f"(eval_f1={trainer.state.best_metric})"
    )
    logger.close()
    print("Phase 3 complete!\n")
    return best_model, phase_stats


# ============================================================
# Hiệu chỉnh OOD trên tập xác thực
# ============================================================

def run_ood_calibration(
    model: nn.Module,
    val_loader,
    device: torch.device,
    output_dir: str,
    p2_report_type: str = "clinical",
    fpr_threshold: float = 0.05,
) -> None:
    """Thực hiện ood calibration cho bước xử lý hiện tại.

    Parameters
    ----------
    model : nn.Module
        Mô hình hoặc thành phần mô hình cần xử lý.
    val_loader : object
        Bộ nạp dữ liệu cung cấp các batch đầu vào.
    device : torch.device
        Thiết bị thực thi phép tính.
    output_dir : str
        Đường dẫn tài nguyên được sử dụng.
    p2_report_type : str, optional
        Văn bản hoặc biểu diễn văn bản đầu vào.
    fpr_threshold : float, optional
        Ngưỡng quyết định của phép đánh giá.

    Raises
    ------
    RuntimeError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    TypeError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    ValueError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    """
    import torch.nn.functional as F

    print("\n" + "=" * 50)
    print("RUNNING OOD CALIBRATION ON VALIDATION SPLIT")
    print("=" * 50)

    if p2_report_type not in ("xray", "clinical"):
        raise ValueError(
            "OOD calibration currently supports only "
            "p2_report_type='xray' or 'clinical'."
        )

    model.eval()

    all_visual_features = []
    all_text_features = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            if not isinstance(batch, dict):
                raise TypeError("OOD calibration requires dictionary-format batches.")

            images = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            if labels.ndim > 1:
                labels = labels.argmax(dim=-1)

            if p2_report_type == "xray":
                input_ids = batch["xray_input_ids"].to(device)
                attention_mask = batch.get(
                    "xray_attention_mask"
                )
            else:
                input_ids = batch["clinical_input_ids"].to(device)
                attention_mask = batch.get(
                    "clinical_attention_mask"
                )

            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            image_features, text_features, *_ = model._encode_modalities(
                images,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            image_global = (
                image_features[:, 0]
                if image_features.ndim == 3
                else image_features
            )
            text_global = (
                text_features[:, 0]
                if text_features.ndim == 3
                else text_features
            )

            all_visual_features.append(
                F.normalize(image_global, dim=-1).cpu().numpy()
            )
            all_text_features.append(
                F.normalize(text_global, dim=-1).cpu().numpy()
            )
            all_labels.append(
                labels.cpu().numpy()
            )

    all_labels = np.concatenate(
        all_labels,
        axis=0,
    )
    all_visual_features = np.concatenate(all_visual_features, axis=0)
    all_text_features = np.concatenate(all_text_features, axis=0)

    detector = MultimodalEnsembleOODDetector().fit(
        visual_embeddings=all_visual_features,
        text_embeddings=all_text_features,
        val_visual_embeddings=all_visual_features,
        val_text_embeddings=all_text_features,
    )
    id_scores = detector.score(
        all_visual_features,
        text_embeddings=all_text_features,
    )
    calibrated_threshold = calibrate_ood_threshold(
        id_scores,
        target_id_fpr=fpr_threshold,
    )

    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(
        output_dir,
        "ood_parameters.npz",
    )

    np.savez(
        output_path,
        method="multimodal_ensemble",
        reference_visual_embeddings=all_visual_features,
        reference_text_embeddings=all_text_features,
        reference_labels=all_labels,
        visual_score_mean=detector.scalers["vis"][0],
        visual_score_std=detector.scalers["vis"][1],
        text_score_mean=detector.scalers["txt"][0],
        text_score_std=detector.scalers["txt"][1],
        knn_k=detector.knn_k,
        knn_reduction=detector.knn_reduction,
        knn_metric=detector.knn_metric,
        calibrated_threshold=calibrated_threshold,
        fpr_threshold=float(fpr_threshold),
        id_scores_mean=float(id_scores.mean()),
        id_scores_std=float(id_scores.std()),
    )

    print("Validation OOD calibration complete!")
    print(f"  Saved parameters to: {output_path}")
    print(
        f"  Threshold at ID FPR={fpr_threshold:.2%}: "
        f"{calibrated_threshold:.4f}"
    )
    print(
        "  ID score statistics — "
        f"mean={id_scores.mean():.4f}, "
        f"std={id_scores.std():.4f}"
    )

# ============================================================
# Điểm vào chính của bộ điều phối
# ============================================================

@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Thực thi điểm vào chính của mô-đun.

    Parameters
    ----------
    cfg : DictConfig
        Cấu hình điều khiển bước xử lý.

    Raises
    ------
    FileNotFoundError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    RuntimeError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    """
    seed_everything(int(cfg.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Thiết lập giá trị trung gian cho bước xử lý tiếp theo.
    do_phase1 = resolve_phase_enabled(cfg.params, "phase1", default=True)
    do_phase2 = resolve_phase_enabled(cfg.params, "phase2", default=True)
    do_phase3 = resolve_phase_enabled(cfg.params, "phase3", default=False)
    init_from_merged = cfg.params.get("phase2", {}).get("init_from_phase1_merged", False)
    init_from_phase1 = cfg.params.get("phase2", {}).get(
        "init_from_phase1_checkpoint", False
    )

    if not do_phase1 and do_phase2 and init_from_merged:
        print(" -> [Config] Khởi tạo từ checkpoint đã gộp (Merged Phase 1). Tắt khởi tạo PEFT.")
        cfg.model.peft.type = 'none'

    # --- Khởi tạo mô hình ---
    print("Building model...")
    model = build_model(cfg.model).to(device)

    # Chọn thiết bị và độ chính xác tính toán phù hợp.
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16

    if use_bf16:
        print(" -> [Config] Enabling BF16 training via PyTorch AMP.")
    elif use_fp16:
        print(" -> [Config] Enabling FP16 training via PyTorch AMP.")

    enable_gradient_checkpointing = cfg.params.get(
        "gradient_checkpointing",
        True,
    )

    supports_gradient_checkpointing = (
        hasattr(model, "gradient_checkpointing_enable")
        and hasattr(model.backbone, "model")
    )

    if enable_gradient_checkpointing and supports_gradient_checkpointing:
        print(" -> [Config] Gradient Checkpointing: True")
        model.gradient_checkpointing_enable()

    elif enable_gradient_checkpointing:
        print(
            " -> [Config] Gradient Checkpointing skipped: "
            f"{cfg.model.backbone_type} does not expose a compatible model interface."
        )

    # --- Khởi tạo các bộ nạp dữ liệu ---
    print("\nInitializing DataLoaders...")
    tokenizer_func = getattr(model.backbone, "tokenizer_obj", getattr(model.backbone, "tokenizer", None))
    preprocess_func = getattr(model.backbone, "preprocess", None)
    train_loader = build_dataloader(cfg.dataset, split="train", transform=preprocess_func, tokenizer=tokenizer_func)
    val_loader = build_dataloader(cfg.dataset, split="val", transform=preprocess_func, tokenizer=tokenizer_func)
    from src.utils.trainer import resolve_pad_token_id, BioMedCLIPDataCollator
    pad_id = resolve_pad_token_id(tokenizer_func) if tokenizer_func is not None else 0
    collator = BioMedCLIPDataCollator(pad_token_id=pad_id)
    train_loader.collate_fn = collator
    val_loader.collate_fn = collator

    # --- Xác định đường dẫn checkpoint ---
    log_dir = os.path.join(hydra.utils.get_original_cwd(), cfg.params.model_dir)
    experiment_name = cfg.get("experiment_name", "default_experiment")
    seed_val = cfg.get("seed", 42)

    # Duy trì khả năng tương thích với cấu hình hoặc dữ liệu phiên bản cũ.
    p1_cfg = cfg.params.get("phase1", {})
    p2_cfg = cfg.params.get("phase2", {})
    p3_cfg = cfg.params.get("phase3", {})
    
    cp_p1 = p1_cfg.get("checkpoint_path", "")
    cp_merged = p1_cfg.get("merged_checkpoint_path", "")
    cp_p2 = p2_cfg.get("checkpoint_path", "")
    cp_p3 = p3_cfg.get("checkpoint_path", "")

    merge_after_p1 = p1_cfg.get("merge_lora_after_training", False)
    phase_runtime: dict[str, dict] = {}
    training_window_started = time.perf_counter()

    # Thiết lập và thực thi pha 1 căn chỉnh ảnh-văn bản.
    if do_phase1:
        # Thiết lập và thực thi pha 1 căn chỉnh ảnh-văn bản.
        # Thiết lập và thực thi pha 2 phân lớp đa phương thức.
        for param in model.fusion.parameters():
            param.requires_grad = False
        for param in model.head.parameters():
            param.requires_grad = False

        model, phase_runtime["phase1"] = run_phase1(
            cfg=cfg,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            cp_p1=cp_p1,
            log_dir=log_dir,
            experiment_name=experiment_name,
            use_bf16=use_bf16,
            use_fp16=use_fp16,
        )

        if merge_after_p1:
            from peft import PeftModel
            def merge_peft_adapters(module):
                """Kết hợp peft adapters cho bước xử lý hiện tại.

                Parameters
                ----------
                module : object
                    Giá trị ``module`` được sử dụng trong phép xử lý.
                """
                for name, child in list(module.named_children()):
                    if isinstance(child, PeftModel):
                        print(f"[*] Hợp nhất (Merging) PEFT adapters tại: {name}...")
                        merged_child = child.merge_and_unload()
                        setattr(module, name, merged_child)
                    else:
                        merge_peft_adapters(child)
            print("\n[Phase 1] Hợp nhất trọng số LoRA vào Base Model...")
            merge_peft_adapters(model)
            if cp_merged:
                os.makedirs(os.path.dirname(cp_merged) or ".", exist_ok=True)
                torch.save(model.state_dict(), cp_merged)
                print(f" [*] Đã lưu Merged Phase 1 Checkpoint tại: {cp_merged}")

    # Thiết lập và thực thi pha 1 căn chỉnh ảnh-văn bản.
    if not do_phase1 and do_phase2:
        if init_from_merged and cp_merged and os.path.exists(cp_merged):
            print(f"Loading merged Phase 1 checkpoint from: {cp_merged}")
            checkpoint = torch.load(cp_merged, map_location=device)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            load_state_dict_checked(
                model, state_dict, context="merged Phase 1"
            )
            print(" -> Merged Phase 1 weights loaded successfully!")
        elif init_from_phase1 and cp_p1 and os.path.exists(cp_p1):
            print(f"Loading best Phase 1 checkpoint from: {cp_p1}")
            checkpoint = torch.load(cp_p1, map_location=device)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            load_state_dict_checked(
                model, state_dict, context="Phase 1"
            )
            print(" -> Phase 1 weights loaded successfully!")
            
            # Kiểm tra điều kiện trước khi thực hiện nhánh xử lý tương ứng.
            if merge_after_p1:
                from peft import PeftModel
                def merge_peft_adapters(module):
                    """Kết hợp peft adapters cho bước xử lý hiện tại.

                    Parameters
                    ----------
                    module : object
                        Giá trị ``module`` được sử dụng trong phép xử lý.
                    """
                    for name, child in list(module.named_children()):
                        if isinstance(child, PeftModel):
                            print(f"[*] Hợp nhất (Merging) PEFT adapters tại: {name}...")
                            merged_child = child.merge_and_unload()
                            setattr(module, name, merged_child)
                        else:
                            merge_peft_adapters(child)
                print("\n[Chuyển đổi] Hợp nhất trọng số LoRA vào Base Model...")
                merge_peft_adapters(model)
            else:
                print("\n[Chuyển đổi] Giữ nguyên LoRA adapters (KHÔNG hợp nhất) cho Phase 2.")
        elif init_from_merged:
            raise FileNotFoundError(
                "Phase 2 requested a merged Phase-1 checkpoint, but it was not "
                f"found at: {cp_merged!r}"
            )
        elif init_from_phase1:
            raise FileNotFoundError(
                "Phase 2 requested a Phase-1 checkpoint, but it was not found "
                f"at: {cp_p1!r}"
            )
        else:
            print(
                "\n[Phase 2 Setup] Phase 1 was skipped and no Phase-1 "
                "initialization was requested; using configured foundation weights."
            )

    # Thiết lập và thực thi pha 2 phân lớp đa phương thức.
    # hoặc khởi tạo pha 3 từ checkpoint của mô hình đã huấn luyện.
    classifier_type = "none"
    if do_phase2 or do_phase3:
        model, classifier_type, _, _ = setup_phase2_modules(model, cfg, device)

    # Thiết lập và thực thi pha 2 phân lớp đa phương thức.
    if do_phase2:

        # Thiết lập và thực thi pha 2 phân lớp đa phương thức.
        is_merged = (do_phase1 and merge_after_p1) or init_from_merged
        peft_type = str(cfg.model.get("peft", {}).get("type", "none")).lower()
        has_adapters = peft_type == "lora" and not is_merged

        if is_merged:
            for param in model.backbone.parameters():
                param.requires_grad = False
            for param in model.fusion.parameters():
                param.requires_grad = True
            for param in model.head.parameters():
                param.requires_grad = True
            print("\n[Phase 2 Setup] Backbone FROZEN (Merged) | Fusion + Head UNFROZEN")
        elif has_adapters:
            # Thiết lập trạng thái và thống kê các tham số mô hình.
            # Thiết lập trạng thái và thống kê các tham số mô hình.
            for param in model.fusion.parameters():
                param.requires_grad = True
            for param in model.head.parameters():
                param.requires_grad = True
            print("\n[Phase 2 Setup] Backbone Base FROZEN, LoRA TRAINABLE | Fusion + Head UNFROZEN")
        else:
            trainable_backbone = sum(
                parameter.numel()
                for parameter in model.backbone.parameters()
                if parameter.requires_grad
            )
            if peft_type == "full_ft":
                status = "Backbone TRAINABLE (full fine-tuning)"
            elif trainable_backbone:
                status = (
                    "Foundation encoders FROZEN; auxiliary backbone modules "
                    f"TRAINABLE ({trainable_backbone:,} parameters)"
                )
            else:
                status = "Backbone FROZEN"
            print(f"\n[Phase 2 Setup] {status}")

        if not bool(p2_cfg.get("use_image", True)):
            visual = getattr(getattr(model.backbone, "model", None), "visual", None)
            if visual is not None:
                for parameter in visual.parameters():
                    parameter.requires_grad = False
            print("[Phase 2 Setup] Visual encoder disabled for text-only mode")

        model = model.to(device)

        model, phase_runtime["phase2"] = run_phase2(
            cfg=cfg,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            cp_p2=cp_p2,
            log_dir=log_dir,
            experiment_name=experiment_name,
            use_bf16=use_bf16,
            use_fp16=use_fp16,
            classifier_type=classifier_type,
        )

    # Thiết lập và thực thi pha 3 học biểu diễn bổ sung.
    if do_phase3:
        if not do_phase2:
            init_checkpoint = str(
                p3_cfg.get("init_checkpoint_path", cp_p2) or cp_p2
            )
            if not init_checkpoint or not os.path.exists(init_checkpoint):
                raise FileNotFoundError(
                    "Phase 3 requires a trained Phase-2 checkpoint, but it was "
                    f"not found at: {init_checkpoint!r}"
                )
            print(f"Loading frozen primary checkpoint for Phase 3: {init_checkpoint}")
            checkpoint = torch.load(init_checkpoint, map_location=device)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            result = load_state_dict_checked(
                model,
                state_dict,
                context="Phase 3 primary initialization",
            )
            critical_primary_missing = [
                key for key in result.missing_keys
                if key.startswith(("fusion.", "head."))
            ]
            if critical_primary_missing:
                raise RuntimeError(
                    "Phase-3 initialization did not restore the primary "
                    f"classification path: {critical_primary_missing[:20]}"
                )

        model, drl_enabled = setup_phase3_modules(model, cfg, device)
        if not drl_enabled:
            raise RuntimeError(
                "Phase 3 is enabled but model.drl.enabled is false or missing."
            )
        model, phase_runtime["phase3"] = run_phase3(
            cfg=cfg,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            cp_p3=cp_p3,
            log_dir=log_dir,
            experiment_name=experiment_name,
            use_bf16=use_bf16,
            use_fp16=use_fp16,
        )

    do_ood = cfg.params.get("run_ood", False)
    if do_ood and (do_phase2 or do_phase3):
        calibration_checkpoint = cp_p3 if do_phase3 else cp_p2
        run_ood_calibration(
            model=model,
            val_loader=val_loader,
            device=device,
            output_dir=os.path.dirname(calibration_checkpoint),
            p2_report_type=cfg.params.phase2.get("p2_report_type", "clinical"),
        )

    total_wall_seconds = time.perf_counter() - training_window_started
    phase_seconds = sum(
        float(stats.get("wall_clock_seconds", 0.0))
        for stats in phase_runtime.values()
    )
    runtime_summary = {
        "experiment_name": str(experiment_name),
        "seed": int(seed_val),
        "scope": (
            "Training phases, validation, checkpointing, centroid updates and "
            "phase transition; excludes model download/build and dataloader construction."
        ),
        "dataset": str(cfg.dataset.name),
        "train_samples": int(len(train_loader.dataset)),
        "validation_samples": int(len(val_loader.dataset)),
        "batch_size_per_device": int(cfg.dataset.batch_size),
        "precision": "bf16" if use_bf16 else ("fp16" if use_fp16 else "fp32"),
        "phase_enabled": {
            "phase1": bool(do_phase1),
            "phase2": bool(do_phase2),
            "phase3": bool(do_phase3),
        },
        "phases": phase_runtime,
        "phase_runtime_seconds": float(phase_seconds),
        "orchestration_wall_clock_seconds": float(total_wall_seconds),
        "gpu_hours": float(phase_seconds / 3600.0) if device.type == "cuda" else 0.0,
        "hardware": {
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
            ),
            "cuda_version": torch.version.cuda,
            "torch_version": torch.__version__,
            "visible_cuda_devices": int(torch.cuda.device_count()),
        },
        "comparison_note": (
            "Compare wall-clock training time only between runs produced on the "
            "same GPU model, software environment, precision, physical "
            "micro-batch and gradient-accumulation protocol. For Phase 1, "
            "gradient accumulation does not reproduce a larger in-batch "
            "contrastive comparison set."
        ),
    }
    os.makedirs(str(cfg.params.model_dir), exist_ok=True)
    runtime_path = os.path.join(str(cfg.params.model_dir), "training_summary.json")
    with open(runtime_path, "w", encoding="utf-8") as handle:
        json.dump(runtime_summary, handle, indent=2, ensure_ascii=False)
    print(f"Training runtime summary saved to: {runtime_path}")
    print("Training process completed successfully!")


if __name__ == "__main__":
    main()
