"""
XBone-Net Training Pipeline.
===============================================================================
Executes the streamlined two-phase training workflow for bone X-ray classification:

  - Phase 1 (Contrastive Alignment): Fine-tunes VLM backbones (e.g., BiomedCLIP)
    using soft-target semantic matching loss to align image and text embeddings.
  - Phase 2 (Classification): Attaches cross-attention fusion and an empirical
    centroid head trained with class-weighted cross-entropy.
  - OOD Calibration: Fits a Mahalanobis-based OOD detector on validation set
    embeddings to calibrate the decision threshold at a target FPR (e.g., 5%).

Configuration is managed via Hydra (configs in configs/). Logging to TensorBoard and CSV.
"""

import os
import sys
import math
import warnings
from typing import Optional

warnings.filterwarnings("ignore", category=FutureWarning, module="timm.*")
warnings.filterwarnings("ignore", message="triton not found")

# Safely force UTF-8 stdout/stderr on Windows environments
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

from src.models.builder import build_model, setup_phase2_modules
from src.datasets.builder import build_dataloader
from src.utils.losses import (
    build_loss,
    build_phase2_loss,
    resolve_phase2_loss_type,
)
from src.utils.trainer import BioMedCLIPDataCollator, SFTrainer, resolve_pad_token_id
from src.utils.logging import TrainingLogger, XBoneTrainerCallback
from src.utils.ood import OODDetector
from src.utils.centroids import (
    EmpiricalCentroidUpdateCallback,
    compute_empirical_centroids,
)


# ============================================================
# Reproducibility Setup
# ============================================================

def seed_everything(seed: int = 42) -> None:
    """Set random seeds across Python, NumPy, and PyTorch for deterministic training.

    Args:
        seed (int): Integer seed value (default: 42).
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
    critical_substrings=("lora_A", "lora_B", "visual_resampler"),
):
    """Load a checkpoint and fail when critical trainable modules are missing."""
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
    """Extract integer class IDs from datasets and torch Subset wrappers."""
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
    """Compute finite weights while preserving absent configured classes.

    Absent classes receive weight zero. Present-class weights are normalized to
    mean one; weighted cross-entropy only indexes the target class, so zero
    weights for classes with no training targets are safe and explicit.
    """
    class_ids = np.asarray(class_ids, dtype=np.int64).reshape(-1)
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2.")
    if class_ids.size == 0:
        raise ValueError(
            "Cannot compute class weights from an empty training subset."
        )
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
        raw_weights[present_mask] = np.minimum(
            raw_weights[present_mask], float(max_class_weight)
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
    return raw_weights, class_counts, missing_classes


# ============================================================
# Phase 1: Multimodal Contrastive Alignment
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
) -> nn.Module:
    """Execute Phase 1 contrastive image-text alignment training.

    Args:
        cfg (DictConfig): Complete Hydra configuration.
        model (nn.Module): Assembled XBone-Net foundation model.
        train_loader: Training DataLoader instance.
        val_loader: Validation DataLoader instance.
        device (torch.device): Computing device.
        cp_p1 (str): Filepath to save Phase 1 best model checkpoint.
        log_dir (str): Logging directory.
        experiment_name (str): Experiment identifier.
        use_bf16 (bool): Flag to enable BF16 precision.
        use_fp16 (bool): Flag to enable FP16 precision.

    Returns:
        nn.Module: Model updated with Phase 1 fine-tuned weights.
    """
    p1_cfg = cfg.params.phase1
    epochs_p1 = p1_cfg.get("epochs", 50)
    lr_p1 = p1_cfg.get("lr", 2e-4)
    wd_p1 = p1_cfg.get("weight_decay", 1e-2)
    patience_p1 = int(p1_cfg.get("early_stopping_patience", 5))
    loss_type_p1 = p1_cfg.get("loss_type", "semantic_matching")
    p1_report_type = p1_cfg.get("p1_report_type", "clinical")

    print("\n" + "=" * 60)
    print("PHASE 1: MULTIMODAL CONTRASTIVE ALIGNMENT (backbone only)")
    print("=" * 60)

    # --- Phase 1 parameter breakdown ---
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
        "precision": "bf16" if use_bf16 else ("fp16" if use_fp16 else "fp32"),
    })
    logger_p1.log_model_summary(model)

    # Build the loss before collecting optimizer parameters. Some non-OpenCLIP
    # backbones receive a logit_scale parameter during loss construction.
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
    optimizer_p1 = AdamW(trainable_params_p1, lr=lr_p1, weight_decay=wd_p1)

    steps_per_epoch_p1 = math.ceil(len(train_loader.dataset) / cfg.dataset.batch_size)
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
    )

    trainer_p1.train()

    os.makedirs(os.path.dirname(cp_p1) or ".", exist_ok=True)
    best_model_p1 = trainer_p1.model
    torch.save(best_model_p1.state_dict(), cp_p1)
    print(f" [*] Saved best Phase 1 checkpoint: {cp_p1} (eval_loss={trainer_p1.state.best_metric})")

    logger_p1.close()
    print("Phase 1 complete!\n")
    return model


# ============================================================
# Phase 2: Classification & Empirical Centroid Learning
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
) -> nn.Module:
    """Execute Phase 2 supervised classification training.

    Args:
        cfg (DictConfig): Complete Hydra configuration.
        model (nn.Module): XBone-Net model with Phase 2 fusion and head attached.
        train_loader: Training DataLoader instance.
        val_loader: Validation DataLoader instance.
        device (torch.device): Computing device.
        cp_p2 (str): Filepath to save Phase 2 best model checkpoint.
        log_dir (str): Logging directory.
        experiment_name (str): Experiment identifier.
        use_bf16 (bool): Flag to enable BF16 precision.
        use_fp16 (bool): Flag to enable FP16 precision.
        classifier_type (str): Resolved classifier head type.

    Returns:
        nn.Module: Model updated with Phase 2 fine-tuned weights.
    """
    p2_cfg = cfg.params.phase2
    epochs_p2 = p2_cfg.get("epochs", 50)
    lr_p2 = p2_cfg.get("lr", 1e-3)
    wd_p2 = p2_cfg.get("weight_decay", 1e-4)
    patience_p2 = int(p2_cfg.get("early_stopping_patience", 5))
    use_text_p2 = bool(p2_cfg.get("use_text", True))
    report_type_p2 = str(p2_cfg.get("p2_report_type", "clinical"))
    uses_empirical_centroids = classifier_type == "empirical_centroid"
    loss_type_p2 = resolve_phase2_loss_type(
        p2_cfg.get("loss_type", None), classifier_type
    )

    print("\n" + "=" * 60)
    print("PHASE 2: CLASSIFIER AND FUSION TRAINING")
    print("=" * 60)

    # --- Phase 2 parameter breakdown ---
    n_backbone = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)
    n_fusion = sum(p.numel() for p in model.fusion.parameters() if p.requires_grad)
    n_head = sum(p.numel() for p in model.head.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = n_backbone + n_fusion + n_head
    peft_type = cfg.model.peft.get("type", "none")

    if n_backbone == 0:
        backbone_status = "FROZEN"
    elif peft_type in ("lora", "qlora"):
        backbone_status = "BASE FROZEN, ADAPTERS TRAINABLE"
    else:
        backbone_status = "TRAINABLE (FULL FINE-TUNING)"
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
        "precision": "bf16" if use_bf16 else ("fp16" if use_fp16 else "fp32"),
        "loss_type": loss_type_p2,
    })
    logger_p2.log_model_summary(model)

    # The head has no trainable class vectors. Initialize it from embeddings of
    # the exact training subset before the first optimization/evaluation step.
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

    # --- Loss Criterion ---
    loss_cfg = p2_cfg.get("loss", {}) or {}
    use_class_weights = loss_cfg.get("use_class_weights", False)

    # Compute class weights from the exact training subset.
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
            print(f"  [Loss] Effective-number class weights (beta={beta}):")
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

    steps_per_epoch_p2 = math.ceil(len(train_loader.dataset) / cfg.dataset.batch_size)
    total_steps_p2 = steps_per_epoch_p2 * epochs_p2
    warmup_steps_p2 = int(p2_cfg.get("warmup_ratio", 0.1) * total_steps_p2)

    tokenizer = getattr(model.backbone, "tokenizer_obj", None)
    pad_id = resolve_pad_token_id(tokenizer) if tokenizer is not None else 0
    os.environ["TENSORBOARD_LOGGING_DIR"] = logger_p2.tb_dir

    # def compute_metrics_eval(eval_pred):
    #     preds, labels = eval_pred
    #     if isinstance(preds, tuple):
    #         preds = preds[0]
    #     preds_cls = np.argmax(preds, axis=1) if preds.ndim > 1 else (preds > 0).astype(int)
    #     acc = accuracy_score(labels, preds_cls)
    #     f1_mac = f1_score(labels, preds_cls, average="macro", zero_division=0)
    #     return {"accuracy": acc, "f1_macro": f1_mac}

    def compute_metrics_eval(eval_pred):
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

    trainer_p2.train()

    os.makedirs(os.path.dirname(cp_p2) or ".", exist_ok=True)
    best_model_p2 = trainer_p2.model
    if uses_empirical_centroids:
        # load_best_model_at_end restores the best encoder checkpoint. Recompute
        # its centroids so the exported checkpoint is internally consistent.
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
    return best_model_p2


# ============================================================
# Validation OOD Calibration
# ============================================================

def run_ood_calibration(
    model: nn.Module,
    val_loader,
    device: torch.device,
    output_dir: str,
    p2_report_type: str = "clinical",
    fpr_threshold: float = 0.05,
) -> None:
    """Fit Mahalanobis OOD statistics using the trained model path.

    Uses the same backbone, padding masks, high-resolution tiles,
    cross-attention fusion, and fused representation as Phase 2.
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

    if not hasattr(model.head, "prototypes"):
        raise ValueError(
            "OOD calibration requires a classifier head that exposes "
            "class centers."
        )

    model.eval()

    all_features = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            if not isinstance(batch, dict):
                raise TypeError(
                    "High-resolution OOD calibration requires "
                    "dictionary-format batches."
                )

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

            tile_values = batch.get("tile_values")
            tile_mask = batch.get("tile_mask")
            tile_boxes = batch.get("tile_boxes")

            if tile_values is not None:
                tile_values = tile_values.to(device)

            if tile_mask is not None:
                tile_mask = tile_mask.to(device)
            if tile_boxes is not None:
                tile_boxes = tile_boxes.to(device)

            outputs = model(
                images=images,
                input_ids=input_ids,
                attention_mask=attention_mask,
                tile_values=tile_values,
                tile_mask=tile_mask,
                tile_boxes=tile_boxes,
                return_features=True,
            )

            if not (
                isinstance(outputs, tuple)
                and len(outputs) == 3
            ):
                raise RuntimeError(
                    "Expected model to return "
                    "(logits, fused_features, class_centers)."
                )

            _, fused_features, _ = outputs

            fused_features = F.normalize(
                fused_features,
                dim=-1,
            )

            all_features.append(
                fused_features.cpu().numpy()
            )
            all_labels.append(
                labels.cpu().numpy()
            )

    all_features = np.concatenate(
        all_features,
        axis=0,
    )
    all_labels = np.concatenate(
        all_labels,
        axis=0,
    )

    learned_prototypes = F.normalize(
        model.head.prototypes.detach(),
        dim=-1,
    ).cpu().numpy()

    detector = OODDetector()

    # OODDetector đã hỗ trợ optional learned prototypes.
    detector.fit(
        embeddings=all_features,
        labels=all_labels,
        prototypes=learned_prototypes,
    )

    id_scores = detector.score(
        all_features,
        method="mahalanobis",
    )

    calibrated_threshold = float(
        np.percentile(
            id_scores,
            (1.0 - fpr_threshold) * 100.0,
        )
    )

    os.makedirs(output_dir, exist_ok=True)

    class_ids = np.asarray(
        sorted(detector.class_means.keys()),
        dtype=np.int64,
    )

    class_means = np.stack(
        [
            detector.class_means[class_id]
            for class_id in class_ids
        ],
        axis=0,
    )

    output_path = os.path.join(
        output_dir,
        "ood_parameters.npz",
    )

    np.savez(
        output_path,
        class_ids=class_ids,
        class_means=class_means,
        shared_cov_inv=detector.shared_cov_inv,
        learned_prototypes=learned_prototypes,
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
# Main Orchestrator Entry Point
# ============================================================

@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Main execution orchestrator for XBone-Net training pipeline.

    Args:
        cfg (DictConfig): Complete Hydra configuration object.
    """
    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Resolve phase control flags
    do_phase1 = cfg.params.get("phase1", {}).get("enabled", cfg.params.get("run_phase1", True))
    do_phase2 = cfg.params.get("phase2", {}).get("enabled", cfg.params.get("run_phase2", True))
    init_from_merged = cfg.params.get("phase2", {}).get("init_from_phase1_merged", False)

    if not do_phase1 and do_phase2 and init_from_merged:
        print(" -> [Config] Khởi tạo từ checkpoint đã gộp (Merged Phase 1). Tắt khởi tạo PEFT.")
        cfg.model.peft.type = 'none'

    # --- Initialize Model ---
    print("Building model...")
    model = build_model(cfg.model).to(device)

    # Enable Mixed Precision flags
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

    # --- Initialize DataLoaders ---
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

    # --- Resolve Checkpoint Paths ---
    log_dir = os.path.join(hydra.utils.get_original_cwd(), cfg.params.model_dir)
    experiment_name = cfg.get("experiment_name", "default_experiment")
    seed_val = cfg.get("seed", 42)

    # Use dict.get to support old format gracefully but prioritize new format
    p1_cfg = cfg.params.get("phase1", {})
    p2_cfg = cfg.params.get("phase2", {})
    
    cp_p1 = p1_cfg.get("checkpoint_path", "")
    cp_merged = p1_cfg.get("merged_checkpoint_path", "")
    cp_p2 = p2_cfg.get("checkpoint_path", "")

    merge_after_p1 = p1_cfg.get("merge_lora_after_training", False)

    # --- Phase 1 Execution ---
    if do_phase1:
        # Freeze fusion and head — Phase 1 only trains backbone
        # (fusion/head will be reinitialized by setup_phase2_modules before Phase 2)
        for param in model.fusion.parameters():
            param.requires_grad = False
        for param in model.head.parameters():
            param.requires_grad = False

        model = run_phase1(
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

    # Checkpoint loading for Phase 2 if Phase 1 was skipped
    if not do_phase1 and do_phase2:
        if init_from_merged and cp_merged and os.path.exists(cp_merged):
            print(f"Loading merged Phase 1 checkpoint from: {cp_merged}")
            checkpoint = torch.load(cp_merged, map_location=device)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            load_state_dict_checked(
                model, state_dict, context="merged Phase 1"
            )
            print(" -> Merged Phase 1 weights loaded successfully!")
        elif cp_p1 and os.path.exists(cp_p1):
            print(f"Loading best Phase 1 checkpoint from: {cp_p1}")
            checkpoint = torch.load(cp_p1, map_location=device)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            load_state_dict_checked(
                model, state_dict, context="Phase 1"
            )
            print(" -> Phase 1 weights loaded successfully!")
            
            # Since we loaded unmerged LoRA weights, merge them now if requested
            if merge_after_p1:
                from peft import PeftModel
                def merge_peft_adapters(module):
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
        else:
            print(f"\n[Warning] No Phase 1 checkpoint found. Using initial weights.")

    # --- Phase 2 Execution ---
    if do_phase2:
        model, classifier_type, _, _ = setup_phase2_modules(model, cfg, device)

        # --- Configure parameter trainability for Phase 2 ---
        is_merged = (do_phase1 and merge_after_p1) or init_from_merged
        peft_type = str(cfg.model.get("peft", {}).get("type", "none")).lower()
        has_adapters = peft_type in ("lora", "qlora") and not is_merged

        if is_merged:
            for param in model.backbone.parameters():
                param.requires_grad = False
            for param in model.fusion.parameters():
                param.requires_grad = True
            for param in model.head.parameters():
                param.requires_grad = True
            print("\n[Phase 2 Setup] Backbone FROZEN (Merged) | Fusion + Head UNFROZEN")
        elif has_adapters:
            # Do NOT freeze model.backbone.parameters() because that would freeze the LoRA adapters.
            # PEFT already froze the base weights and kept LoRA trainable during initialization.
            for param in model.fusion.parameters():
                param.requires_grad = True
            for param in model.head.parameters():
                param.requires_grad = True
            print("\n[Phase 2 Setup] Backbone Base FROZEN, LoRA TRAINABLE | Fusion + Head UNFROZEN")
        else:
            print("\n[Phase 2 Setup] Backbone TRAINABLE (full fine-tuning)")

        model = model.to(device)

        model = run_phase2(
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

        do_ood = cfg.params.get("run_ood", False)
        if do_ood:
            run_ood_calibration(
                model=model,
                val_loader=val_loader,
                device=device,
                output_dir=os.path.dirname(cp_p2),
                p2_report_type=cfg.params.phase2.get("p2_report_type", "clinical"),
            )

    print("Training process completed successfully!")


if __name__ == "__main__":
    main()
