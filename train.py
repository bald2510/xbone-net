"""
XBone-Net Training Pipeline.
===============================================================================
Executes the streamlined two-phase training workflow for bone X-ray classification:

  - Phase 1 (Contrastive Alignment): Fine-tunes VLM backbones (e.g., BiomedCLIP)
    using soft-target semantic matching loss to align image and text embeddings.
  - Phase 2 (Classification & Prototype Learning): Attaches cross-attention fusion
    and prototypical head to train with Unweighted Cross-Entropy + Prototype Loss.
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
from trl import SFTConfig
from transformers import EarlyStoppingCallback
from sklearn.metrics import accuracy_score, f1_score

from src.models.builder import build_model, setup_phase2_modules
from src.datasets.builder import build_dataloader
from src.utils.losses import build_loss, CombinedPhase2LossMulticlass
from src.utils.trainer import BioMedCLIPDataCollator, SFTrainer, resolve_pad_token_id
from src.utils.logging import TrainingLogger, XBoneTrainerCallback
from src.utils.ood import OODDetector


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

    # Only train backbone parameters in Phase 1 (contrastive alignment)
    trainable_params_p1 = [p for p in model.backbone.parameters() if p.requires_grad]
    optimizer_p1 = AdamW(trainable_params_p1, lr=lr_p1, weight_decay=wd_p1)

    steps_per_epoch_p1 = math.ceil(len(train_loader.dataset) / cfg.dataset.batch_size)
    total_steps_p1 = steps_per_epoch_p1 * epochs_p1
    warmup_steps_p1 = int(p1_cfg.get("warmup_ratio", 0.1) * total_steps_p1)

    loss_fn_p1 = build_loss(
        loss_type_p1, 
        clip_model=model.backbone.model, 
        temperature=p1_cfg.get("temperature", 0.07),
        target_similarity=p1_cfg.get("target_similarity", 0.7)
    )
    tokenizer_p1 = getattr(model.backbone, "tokenizer_obj", getattr(model.backbone, "tokenizer", None))
    pad_id = resolve_pad_token_id(tokenizer_p1) if tokenizer_p1 is not None else 0

    p1_args = SFTConfig(
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

    os.makedirs(os.path.dirname(cp_p1), exist_ok=True)
    best_model_p1 = trainer_p1.model
    torch.save(best_model_p1.state_dict(), cp_p1)
    print(f" [*] Saved best Phase 1 checkpoint: {cp_p1} (eval_loss={trainer_p1.state.best_metric})")

    logger_p1.close()
    print("Phase 1 complete!\n")
    return model


# ============================================================
# Phase 2: Classification & Prototype Learning
# ============================================================

from transformers import TrainerCallback

class DynamicProtoLossCallback(TrainerCallback):
    """Dynamically schedules the lambda_proto weight based on the epoch."""
    def __init__(self, criterion):
        self.criterion = criterion

    def on_epoch_begin(self, args, state, control, **kwargs):
        # state.epoch is a float indicating the number of epochs completed
        # e.g., at the start of the first epoch, state.epoch = 0.0
        epoch = int(state.epoch) + 1  
        
        if epoch <= 10:
            new_lambda = 0.0
        elif epoch <= 30:
            new_lambda = 0.05
        else:
            new_lambda = 0.1
            
        if hasattr(self.criterion, 'lambda_proto'):
            old_lambda = self.criterion.lambda_proto
            if old_lambda != new_lambda:
                self.criterion.lambda_proto = new_lambda
                print(f"\n[DynamicProtoLossCallback] Epoch {epoch}: Updated lambda_proto from {old_lambda} to {new_lambda}")

                if old_lambda == 0.0 and new_lambda > 0.0:
                    model = kwargs.get('model')
                    train_dataloader = kwargs.get('train_dataloader')
                    if model is not None and train_dataloader is not None:
                        actual_model = model.module if hasattr(model, 'module') else model
                        if hasattr(actual_model, 'head') and hasattr(actual_model.head, 'prototypes'):
                            self._warm_start_prototypes(actual_model, train_dataloader)

    def _warm_start_prototypes(self, model, dataloader):
        import torch
        import torch.nn.functional as F
        from tqdm import tqdm
        
        print("\n[DynamicProtoLossCallback] Calculating data centroids to warm-start prototypes...")
        model.eval()
        device = next(model.parameters()).device
        
        all_features = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting Train Features"):
                pixel_values = batch["pixel_values"].to(device)
                input_ids = batch.get("xray_input_ids", batch.get("clinical_input_ids", batch.get("input_ids", None)))
                attention_mask = batch.get("xray_attention_mask", batch.get("clinical_attention_mask", None))
                labels = batch["labels"].to(device)
                
                if input_ids is not None:
                    input_ids = input_ids.to(device)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                
                _, features, _ = model(images=pixel_values, input_ids=input_ids, attention_mask=attention_mask, return_features=True)
                
                all_features.append(features.cpu())
                if labels.ndim > 1:
                    labels = labels.argmax(dim=-1)
                all_labels.append(labels.cpu())
                
        all_features = torch.cat(all_features, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        num_classes = model.head.prototypes.size(0)
        new_prototypes = torch.zeros_like(model.head.prototypes.data)
        
        for c in range(num_classes):
            mask = (all_labels == c)
            if mask.sum() > 0:
                class_feats = all_features[mask]
                mean_feat = class_feats.mean(dim=0)
                mean_feat = F.normalize(mean_feat, dim=-1)
                new_prototypes[c] = mean_feat.to(device)
            else:
                new_prototypes[c] = model.head.prototypes.data[c]
                
        model.head.prototypes.data.copy_(new_prototypes)
        print("[DynamicProtoLossCallback] Successfully updated prototypes with Data Centroids!\n")
        model.train()


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
) -> nn.Module:
    """Execute Phase 2 supervised classification and prototype training.

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

    Returns:
        nn.Module: Model updated with Phase 2 fine-tuned weights.
    """
    p2_cfg = cfg.params.phase2
    epochs_p2 = p2_cfg.get("epochs", 50)
    lr_p2 = p2_cfg.get("lr", 1e-3)
    wd_p2 = p2_cfg.get("weight_decay", 1e-4)
    patience_p2 = int(p2_cfg.get("early_stopping_patience", 5))
    classifier_type = p2_cfg.get("classifier_type", "prototypical")

    print("\n" + "=" * 60)
    print("PHASE 2: CLASSIFIER AND FUSION TRAINING")
    print("=" * 60)

    # --- Phase 2 parameter breakdown ---
    n_backbone = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)
    n_fusion = sum(p.numel() for p in model.fusion.parameters() if p.requires_grad)
    n_head = sum(p.numel() for p in model.head.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = n_backbone + n_fusion + n_head
    backbone_status = "TRAINABLE (full fine-tuning)" if n_backbone > 0 else "FROZEN (after Phase 1)"
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
    })
    logger_p2.log_model_summary(model)

    # --- Loss Criterion ---
    loss_cfg = p2_cfg.get("loss", {}) or {}
    use_class_weights = loss_cfg.get("use_class_weights", False)

    # Compute class weights from training set distribution
    class_weights_tensor = None
    if use_class_weights:
        try:
            train_dataset = train_loader.dataset
            class_ids = train_dataset.df["class_id"].values
            num_classes = len(train_dataset.classes) if train_dataset.classes else int(class_ids.max() + 1)
            class_counts = np.bincount(class_ids, minlength=num_classes).astype(float)
            
            weight_type = loss_cfg.get("weight_type", "effective_num")
            if weight_type == "effective_num":
                beta = loss_cfg.get("effective_num_beta", 0.999)
                # w_i = (1 - beta) / (1 - beta^N_i)
                effective_num = (1.0 - np.power(beta, class_counts)) / (1.0 - beta)
                raw_weights = 1.0 / np.maximum(effective_num, 1e-8)
                # Normalize weights so they sum to num_classes
                raw_weights = raw_weights * (num_classes / np.sum(raw_weights))
                print(f"  [Loss] Class weights (Effective Number of Samples, beta={beta}):")
            else:
                # Inverse frequency weighting: w_i = N_total / (num_classes * N_i)
                total = class_counts.sum()
                raw_weights = total / (num_classes * np.maximum(class_counts, 1.0))
                # Clamp to prevent extreme weights for very rare classes
                max_weight = loss_cfg.get("max_class_weight", 10.0)
                raw_weights = np.minimum(raw_weights, max_weight)
                print(f"  [Loss] Class weights (inv-freq, clamped at {max_weight}):")
                
            class_weights_tensor = torch.tensor(raw_weights, dtype=torch.float32).to(device)
            dataset_params = cfg.dataset.get('params', {})
            class_names = dataset_params.get('classes', dataset_params.get('pathologies', []))
            for i, w in enumerate(raw_weights):
                name = class_names[i] if i < len(class_names) else f"class_{i}"
                print(f"    {name:30s}: {w:.3f} (n={int(class_counts[i])})")
        except Exception as e:
            print(f"  [Loss] Could not compute class weights: {e}. Using uniform weights.")

    if classifier_type == "prototypical":
        criterion_p2 = CombinedPhase2LossMulticlass(
            class_weights=class_weights_tensor,
            proto_margin=loss_cfg.get("proto_margin", 0.5),
            lambda_proto=loss_cfg.get("lambda_proto", 0.3),
            label_smoothing=loss_cfg.get("label_smoothing", 0.1),
        )
    else:
        criterion_p2 = nn.CrossEntropyLoss(
            weight=class_weights_tensor, label_smoothing=0.1
        )

    trainable_params_p2 = [p for p in model.parameters() if p.requires_grad]
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

    def compute_metrics_eval(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]
        preds_cls = np.argmax(preds, axis=1) if preds.ndim > 1 else (preds > 0).astype(int)
        acc = accuracy_score(labels, preds_cls)
        f1_mac = f1_score(labels, preds_cls, average="macro", zero_division=0)
        return {"accuracy": acc, "f1_macro": f1_mac}

    p2_args = SFTConfig(
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
    )

    trainer_p2 = SFTrainer(
        model=model,
        args=p2_args,
        train_dataset=train_loader.dataset,
        eval_dataset=val_loader.dataset,
        data_collator=BioMedCLIPDataCollator(pad_token_id=pad_id),
        compute_metrics=compute_metrics_eval,
        callbacks=[XBoneTrainerCallback(logger_p2), EarlyStoppingCallback(early_stopping_patience=patience_p2), DynamicProtoLossCallback(criterion_p2)],
        loss_fn=criterion_p2,
        optimizers=(optimizer_p2, None),
        phase="phase2",
        use_text_in_p2=p2_cfg.get("use_text", True),
        p2_report_type=p2_cfg.get("p2_report_type", "clinical"),
    )

    trainer_p2.train()

    os.makedirs(os.path.dirname(cp_p2), exist_ok=True)
    best_model_p2 = trainer_p2.model
    torch.save(best_model_p2.state_dict(), cp_p2)
    print(f" [*] Saved best Phase 2 checkpoint: {cp_p2} (eval_f1={trainer_p2.state.best_metric})")

    logger_p2.close()
    print("Phase 2 complete!\n")
    return model


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
    """Calibrate Mahalanobis-based OOD detector on validation set embeddings.

    Extracts fused embeddings from validation samples, fits class-conditional
    Gaussian means and shared precision matrix, and saves calibrated parameters.

    Args:
        model (nn.Module): Trained XBone-Net model in evaluation mode.
        val_loader: DataLoader for validation split.
        device (torch.device): Computing device.
        output_dir (str): Directory where ood_parameters.npz will be stored.
        p2_report_type (str): Report type for Phase 2 ('xray', 'clinical', 'both').
        fpr_threshold (float): FPR target for decision threshold calibration.
    """
    print("\n" + "=" * 50)
    print("RUNNING OOD CALIBRATION ON VALIDATION SPLIT")
    print("=" * 50)

    model.eval()
    all_image_embeds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            if len(batch) == 4:
                images, xray_ids, clinical_ids, labels = batch
                if p2_report_type in ("both", "xray_clinical"):
                    img_feat, xray_feat = model.backbone(images.to(device), xray_ids.to(device))
                    _, clinical_feat = model.backbone(images.to(device), clinical_ids.to(device))
                    text_feat = (xray_feat + clinical_feat) / 2.0
                elif p2_report_type == "xray":
                    img_feat, text_feat = model.backbone(images.to(device), xray_ids.to(device))
                else:
                    img_feat, text_feat = model.backbone(images.to(device), clinical_ids.to(device))

                if model.fusion is not None:
                    fused = model.fusion(img_feat, text_feat)
                else:
                    fused = img_feat
            else:
                images, text_ids, labels = batch
                img_feat, text_feat = model.backbone(images.to(device), text_ids.to(device))
                if model.fusion is not None:
                    fused = model.fusion(img_feat, text_feat)
                else:
                    fused = img_feat

            all_image_embeds.append(fused.cpu().numpy())
            all_labels.append(labels.numpy())

    all_image_embeds = np.concatenate(all_image_embeds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    detector = OODDetector()
    detector.fit(all_image_embeds, all_labels)

    id_scores = detector.score(all_image_embeds, method="mahalanobis")
    calibrated_threshold = float(np.percentile(id_scores, (1.0 - fpr_threshold) * 100.0))

    os.makedirs(output_dir, exist_ok=True)
    np.savez(
        os.path.join(output_dir, "ood_parameters.npz"),
        class_means=detector.class_means,
        shared_cov_inv=detector.shared_cov_inv,
        calibrated_threshold=calibrated_threshold,
    )

    print("Validation OOD Calibration complete!")
    print(f"  Saved parameters to: {os.path.join(output_dir, 'ood_parameters.npz')}")
    print(f"  Calibrated OOD Threshold (FPR={fpr_threshold}): {calibrated_threshold:.4f}")
    print(f"  ID Validation Score Stats — Mean: {id_scores.mean():.4f}, Std: {id_scores.std():.4f}\n")


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

    if cfg.params.get("gradient_checkpointing", True) and hasattr(model, "gradient_checkpointing_enable"):
        print(" -> [Config] Gradient Checkpointing: True")
        model.gradient_checkpointing_enable()

    # --- Initialize DataLoaders ---
    print("\nInitializing DataLoaders...")
    tokenizer_func = getattr(model.backbone, "tokenizer_obj", getattr(model.backbone, "tokenizer", None))
    preprocess_func = getattr(model.backbone, "preprocess", None)
    train_loader = build_dataloader(cfg.dataset, split="train", transform=preprocess_func, tokenizer=tokenizer_func)
    val_loader = build_dataloader(cfg.dataset, split="val", transform=preprocess_func, tokenizer=tokenizer_func)

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
                os.makedirs(os.path.dirname(cp_merged), exist_ok=True)
                torch.save(model.state_dict(), cp_merged)
                print(f" [*] Đã lưu Merged Phase 1 Checkpoint tại: {cp_merged}")

    # Checkpoint loading for Phase 2 if Phase 1 was skipped
    if not do_phase1 and do_phase2:
        if init_from_merged and cp_merged and os.path.exists(cp_merged):
            print(f"Loading merged Phase 1 checkpoint from: {cp_merged}")
            checkpoint = torch.load(cp_merged, map_location=device)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            model.load_state_dict(state_dict, strict=False)
            print(" -> Merged Phase 1 weights loaded successfully!")
        elif cp_p1 and os.path.exists(cp_p1):
            print(f"Loading best Phase 1 checkpoint from: {cp_p1}")
            checkpoint = torch.load(cp_p1, map_location=device)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            model.load_state_dict(state_dict, strict=False)
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
        model, _, _, _ = setup_phase2_modules(model, cfg, device)

        # --- Configure parameter trainability for Phase 2 ---
        is_merged = (do_phase1 and merge_after_p1) or init_from_merged
        has_adapters = (do_phase1 and not merge_after_p1) or (
            not do_phase1 and cp_p1 and not init_from_merged
        )

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
