"""
XBone-Net Training Pipeline.
===============================================================================
Executes the full two-stage training workflow for bone X-ray classification:
  - Phase 1 (Contrastive Alignment): Fine-tunes VLM backbones (e.g., BiomedCLIP)
    using semantic matching loss to align image and text embeddings.
  - Phase 2 (Classification): Attaches cross-attention fusion and prototypical
    classification head to train with combined ASL/CE and prototype compactness.
  - OOD Calibration: Fits Mahalanobis-based OOD detector on validation set
    embeddings to calibrate threshold at target false-positive rate.

Configuration is managed via Hydra (configs in configs/). Logging to TensorBoard.
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="timm.*")
warnings.filterwarnings("ignore", message="triton not found")

# Force UTF-8 stdout/stderr on Windows safely
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

import torch
import torch.nn as nn
import math
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.optim import AdamW, SGD
from trl import SFTConfig
from transformers import EarlyStoppingCallback
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

from src.models.builder import build_model, setup_phase2_modules
from src.datasets.builder import build_dataloader
from src.utils.losses import build_loss
from src.utils.trainer import BioMedCLIPDataCollator, SFTrainer, resolve_pad_token_id
from src.utils.logging import TrainingLogger, XBoneTrainerCallback


# ============================================================
# Helper Functions & Calibration
# ============================================================

def calibrate_ood_on_validation(
    model,
    val_loader,
    device,
    output_dir,
    p2_report_type="clinical",
    fpr_threshold=0.05,
):
    """Calibrate a Mahalanobis-based OOD detector on validation embeddings.

    Extracts fused image-text embeddings from the validation split, fits a
    class-conditional Gaussian model with a shared covariance matrix, and
    selects an OOD decision threshold at a target false-positive rate (FPR).

    The Mahalanobis distance D_c(x) for sample x to class c is computed as:
        D_c(x) = sqrt((x - mu_c)^T * Sigma_inv * (x - mu_c))
    where mu_c is the class mean and Sigma_inv is the shared precision matrix.
    The minimum distance across all classes serves as the OOD score.

    Args:
        model: Trained XBone-Net model in evaluation mode.
        val_loader: DataLoader for the validation split.
        device: Target torch.device for model execution.
        output_dir: Directory path where ood_parameters.npz will be stored.
        p2_report_type: Report branch for Phase 2 inference ('xray', 'clinical', or 'both').
        fpr_threshold: Target false-positive rate for threshold selection (default: 0.05).

    Returns:
        None
    """
    print("\n" + "=" * 50)
    print("RUNNING OOD CALIBRATION ON VALIDATION SPLIT")
    print("=" * 50)

    from src.utils.ood import OODDetector

    model.eval()
    all_image_embeds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            if len(batch) == 4:
                images, xray_ids, clinical_ids, labels = batch
                if p2_report_type == "both":
                    xray_ids = xray_ids.to(device)
                    clinical_ids = clinical_ids.to(device)
                    img_feat, xray_feat = model.backbone(images.to(device), xray_ids)
                    _, clinical_feat = model.backbone(images.to(device), clinical_ids)
                    text_feat = (xray_feat + clinical_feat) / 2.0
                    fused = model.fusion(img_feat, text_feat) if model.fusion is not None else img_feat
                else:
                    input_ids = xray_ids if p2_report_type == "xray" else clinical_ids
                    img_feats, txt_feats = model.backbone(
                        images.to(device),
                        input_ids.to(device) if input_ids is not None else None
                    )
                    fused = model.fusion(img_feats, txt_feats) if model.fusion is not None else img_feats
            elif len(batch) == 3:
                images, input_ids, labels = batch
                img_feats, txt_feats = model.backbone(
                    images.to(device),
                    input_ids.to(device) if input_ids is not None else None
                )
                fused = model.fusion(img_feats, txt_feats) if model.fusion is not None else img_feats
            else:
                continue

            all_image_embeds.append(fused.cpu().numpy())
            all_labels.append(labels.numpy())

    if not all_image_embeds:
        print("[Warning] No validation embeddings extracted. Skipping OOD Calibration.")
        return

    image_embeddings = np.vstack(all_image_embeds)
    labels = np.concatenate(all_labels)

    norms = np.linalg.norm(image_embeddings, axis=1, keepdims=True)
    normed_embeddings = image_embeddings / (norms + 1e-8)

    prototypes = None
    if hasattr(model, "head") and hasattr(model.head, "prototypes"):
        prototypes = model.head.prototypes.detach().cpu().numpy()

    detector = OODDetector()
    detector.fit(normed_embeddings, labels, prototypes=prototypes)

    id_scores = detector.score_mahalanobis(normed_embeddings)
    threshold = np.percentile(id_scores, 100 * (1.0 - fpr_threshold))
    id_scores_mean = float(np.mean(id_scores))
    id_scores_std = float(np.std(id_scores))

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "ood_parameters.npz")
    np.savez_compressed(
        save_path,
        shared_cov_inv=detector.shared_cov_inv,
        threshold=threshold,
        id_scores_mean=id_scores_mean,
        id_scores_std=id_scores_std,
        method="mahalanobis"
    )
    print("Validation OOD Calibration complete!")
    print(f"  Saved parameters to: {save_path}")
    print(f"  Calibrated OOD Threshold (FPR={fpr_threshold}): {threshold:.4f}")
    print(f"  ID Validation Score Statistics — Mean: {id_scores_mean:.4f}, Std: {id_scores_std:.4f}\n")


def seed_everything(seed=42):
    """Set random seeds across all libraries for deterministic execution.

    Seeds Python built-in random, NumPy, PyTorch CPU, and PyTorch CUDA. Also
    enables CuDNN deterministic mode to guarantee bitwise reproducibility.

    Args:
        seed: Integer seed value (default: 42).

    Returns:
        None
    """
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def compute_class_weights(dataset, num_classes, device, weight_type="cb", beta=0.99):
    """Compute per-class loss weights to mitigate class imbalance.

    Supports three weighting strategies:
      - inverse: w_c = N / (C * n_c)
      - sqrt: w_c = 1 / sqrt(n_c), rescaled so sum(w) = C
      - cb (class-balanced): w_c = (1 - beta) / (1 - beta^n_c), rescaled so sum(w) = C

    Args:
        dataset: PyTorch Dataset or wrapper containing target class labels.
        num_classes: Total number of target classes C.
        device: Target torch.device for output weight tensor.
        weight_type: Weighting scheme ('inverse', 'sqrt', 'cb', or 'none').
        beta: Effective number hyperparameter for class-balanced loss (default: 0.99).

    Returns:
        torch.Tensor: Tensor of shape (num_classes,) containing class weights.
    """
    if weight_type == "none":
        return torch.ones(num_classes, device=device)

    raw_dataset = getattr(dataset, "dataset", dataset)
    if hasattr(raw_dataset, "df") and "class_id" in raw_dataset.df.columns:
        counts = torch.zeros(num_classes)
        for val in raw_dataset.df["class_id"].values:
            counts[int(val)] += 1
    else:
        counts = torch.zeros(num_classes)
        for i in range(len(dataset)):
            item = dataset[i]
            label = item["labels"] if isinstance(item, dict) else item[-1]
            if isinstance(label, torch.Tensor):
                counts[label.item()] += 1
            else:
                counts[int(label)] += 1

    counts = counts.clamp(min=1)
    if weight_type == "inverse":
        weights = len(dataset) / (num_classes * counts)
    elif weight_type == "sqrt":
        weights = 1.0 / torch.sqrt(counts)
        weights = weights * (num_classes / weights.sum())
    elif weight_type == "cb":
        cb_weights = torch.zeros(num_classes)
        for idx in range(num_classes):
            n = counts[idx].item()
            cb_weights[idx] = (1.0 - beta) / (1.0 - math.pow(beta, n))
        weights = cb_weights * (num_classes / cb_weights.sum())
    else:
        weights = torch.ones(num_classes)

    return weights.to(device)


# ============================================================
# Dataset Adapters
# ============================================================

class HuggingFaceDatasetWrapper(torch.utils.data.Dataset):
    """Adapt tuple-returning PyTorch Datasets to dictionary format for HuggingFace SFTrainer.

    Converts dataset tuples (image, xray_ids, clinical_ids, labels) or
    (image, input_ids, labels) into dictionary format expected by SFTrainer with
    keys pixel_values, xray_input_ids, clinical_input_ids, and labels.

    Attributes:
        dataset: The wrapped PyTorch Dataset.
    """

    def __init__(self, dataset):
        """Initialize the dataset wrapper.

        Args:
            dataset: PyTorch Dataset returning tuples.
        """
        self.dataset = dataset

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx):
        """Fetch sample by index and convert to dict format.

        Args:
            idx: Integer sample index.

        Returns:
            dict: Sample dictionary formatted for SFTrainer.
        """
        result = self.dataset[idx]
        if len(result) == 4:
            image, xray_ids, clinical_ids, labels = result
            return {
                "pixel_values": image,
                "xray_input_ids": xray_ids,
                "clinical_input_ids": clinical_ids,
                "labels": labels
            }
        else:
            image, input_ids, labels = result
            return {
                "pixel_values": image,
                "xray_input_ids": input_ids,
                "clinical_input_ids": input_ids,
                "labels": labels
            }


# ============================================================
# Main Training Entry Point
# ============================================================

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """Execute full XBone-Net training and calibration pipeline.

    Orchestrates configuration loading, model building, dataset creation, Phase 1
    contrastive pre-training, Phase 2 classification fine-tuning, and post-hoc
    validation OOD detector calibration.

    Args:
        cfg: Hydra DictConfig containing experiment configuration.

    Returns:
        None
    """
    print("=== EXPERIMENT CONFIGURATION ===")
    print(OmegaConf.to_yaml(cfg))
    print("================================\n")

    seed_everything(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    params_cfg = cfg.get("params", {}) or {}
    debug_mode = params_cfg.get("debug", cfg.get("debug", False)) or params_cfg.get("debug_model", False)

    print("Building model...")
    model = build_model(cfg.model).to(device)
    print("Model built successfully!\n")

    model.print_architecture(verbose=debug_mode)

    print("Initializing DataLoaders...")
    preprocess = model.backbone.preprocess
    tokenizer = model.backbone.tokenizer

    train_loader = build_dataloader(cfg=cfg.dataset, split="train", transform=preprocess, tokenizer=tokenizer)
    val_loader = build_dataloader(cfg=cfg.dataset, split="val", transform=preprocess, tokenizer=tokenizer)
    print("DataLoader initialization complete!\n")

    print("=== DATALOADER SANITY CHECK ===")
    try:
        sample_batch = next(iter(train_loader))
        if len(sample_batch) == 4:
            images_check, xray_check, clinical_check, labels_check = sample_batch
            print(f" -> Batch: Images {images_check.shape} | Xray IDs {xray_check.shape} | Clinical IDs {clinical_check.shape} | Labels {labels_check.shape}")
        else:
            images_check, input_ids_check, labels_check = sample_batch
            print(f" -> Batch: Images {images_check.shape} | Text IDs {input_ids_check.shape} | Labels {labels_check.shape}")
        print(f" -> Label sample: {labels_check[0].tolist()}")
    except Exception as e:
        print(f" -> [ERROR] Could not read sample from dataloader: {e}")
    print("=" * 50 + "\n")

    # --- Hyperparameter extraction ---
    params_cfg = cfg.get("params", {}) or {}
    run_phase1 = params_cfg.get("run_phase1", True)
    run_phase2 = params_cfg.get("run_phase2", True)

    p1_cfg = params_cfg.get("phase1", {}) or {}
    epochs_p1 = p1_cfg.get("epochs", params_cfg.get("epochs", 100))
    lr_p1 = p1_cfg.get("lr", params_cfg.get("lr", 1e-4))
    wd_p1 = p1_cfg.get("weight_decay", params_cfg.get("weight_decay", 1e-2))
    cp_p1 = p1_cfg.get("checkpoint_path", "checkpoints/best_phase1.pth")
    temp_p1 = p1_cfg.get("temperature", params_cfg.get("temperature", 0.07))
    patience_p1 = p1_cfg.get("early_stopping_patience", params_cfg.get("early_stopping_patience", 3))

    p2_cfg = params_cfg.get("phase2", {}) or {}
    epochs_p2 = p2_cfg.get("epochs", params_cfg.get("epochs", 10))
    lr_p2 = p2_cfg.get("lr", params_cfg.get("lr", 1e-4))
    wd_p2 = p2_cfg.get("weight_decay", params_cfg.get("weight_decay", 1e-2))
    cp_p2 = p2_cfg.get("checkpoint_path", params_cfg.get("checkpoint_path", "checkpoints/best_phase2.pth"))
    patience_p2 = p2_cfg.get("early_stopping_patience", params_cfg.get("early_stopping_patience", 3))

    os.makedirs(os.path.dirname(cp_p1) if os.path.dirname(cp_p1) else ".", exist_ok=True)
    os.makedirs(os.path.dirname(cp_p2) if os.path.dirname(cp_p2) else ".", exist_ok=True)

    use_bf16 = False
    use_fp16 = False
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            use_bf16 = True
            print(" -> [Config] Enabling BF16 training.")
        else:
            use_fp16 = True
            print(" -> [Config] Enabling FP16 training.")
    else:
        print(" -> [Config] Using FP32 on CPU.")

    gradient_checkpointing = params_cfg.get("gradient_checkpointing", True)
    print(f" -> [Config] Gradient Checkpointing: {gradient_checkpointing}\n")

    experiment_name = cfg.get("experiment_name", "default_experiment")
    log_dir = os.path.join(hydra.utils.get_original_cwd(), "runs")
    os.makedirs(log_dir, exist_ok=True)

    train_dataset = HuggingFaceDatasetWrapper(train_loader.dataset)
    val_dataset = HuggingFaceDatasetWrapper(val_loader.dataset)
    data_collator = BioMedCLIPDataCollator(pad_token_id=resolve_pad_token_id(tokenizer))

    backbone_type = cfg.model.get('backbone_type', 'biomedclip')
    is_image_only = backbone_type.startswith('resnet50')

    if run_phase1 and is_image_only:
        print("\n[Skip] Phase 1 (contrastive) not supported for image-only backbone. Skipping.")
        run_phase1 = False

    # --- Phase 1: Contrastive pre-training ---
    if run_phase1:
        print("\n" + "=" * 50)
        print("STARTING PHASE 1: SEMANTIC MATCHING CONTRASTIVE TRAINING")
        print("=" * 50)

        peft_type = cfg.model.peft.get('type', 'none')

        for name, param in model.named_parameters():
            param.requires_grad = False

        if peft_type in ('lora', 'qlora'):
            for name, param in model.named_parameters():
                if "lora" in name or "logit_scale" in name or "logit_bias" in name:
                    param.requires_grad = True
        elif peft_type == 'full_ft':
            for param in model.backbone.parameters():
                param.requires_grad = True
        elif peft_type == 'none':
            for name, param in model.named_parameters():
                if "logit_scale" in name or "logit_bias" in name:
                    param.requires_grad = True

        print("Phase 1 Parameter Summary:")
        model.print_parameter_summary()

        logger_p1 = TrainingLogger(log_dir=log_dir, experiment_name=experiment_name, phase="phase1")
        warmup_ratio_p1 = p1_cfg.get("warmup_ratio", 0.1)
        logger_p1.log_hyperparams({
            "experiment": experiment_name,
            "phase": "phase1",
            "peft_type": cfg.model.peft.get("type", "none"),
            "epochs": epochs_p1,
            "learning_rate": lr_p1,
            "weight_decay": wd_p1,
            "batch_size": cfg.dataset.batch_size,
            "loss_type": p1_cfg.get("loss_type", "semantic_matching"),
            "precision": "bf16" if use_bf16 else ("fp16" if use_fp16 else "fp32"),
        })
        logger_p1.log_model_summary(model)
        callback_p1 = XBoneTrainerCallback(logger_p1)

        loss_type = p1_cfg.get("loss_type", "semantic_matching")
        criterion_p1 = build_loss(
            loss_type,
            clip_model=model.backbone.model,
            temperature=temp_p1,
        )

        trainable_params_p1 = [p for p in model.parameters() if p.requires_grad]

        optimizer_type_p1 = p1_cfg.get("optimizer", "adamw")
        if optimizer_type_p1 == "sgd":
            optimizer_p1 = SGD(trainable_params_p1, lr=lr_p1, weight_decay=wd_p1, momentum=0.9)
        else:
            optimizer_p1 = AdamW(trainable_params_p1, lr=lr_p1, weight_decay=wd_p1)

        steps_per_epoch_p1 = math.ceil(len(train_dataset) / cfg.dataset.batch_size)
        total_steps_p1 = steps_per_epoch_p1 * epochs_p1
        warmup_steps_p1 = int(warmup_ratio_p1 * total_steps_p1)

        os.environ["TENSORBOARD_LOGGING_DIR"] = logger_p1.tb_dir

        p1_args = SFTConfig(
            output_dir=os.path.dirname(cp_p1) if os.path.dirname(cp_p1) else "./checkpoints",
            num_train_epochs=epochs_p1,
            learning_rate=lr_p1,
            weight_decay=wd_p1,
            per_device_train_batch_size=cfg.dataset.batch_size,
            per_device_eval_batch_size=cfg.dataset.batch_size,
            eval_strategy="epoch" if epochs_p1 > 0 else "no",
            save_strategy="epoch" if epochs_p1 > 0 else "no",
            save_total_limit=2,
            logging_strategy="steps",
            logging_steps=10,
            load_best_model_at_end=True if epochs_p1 > 0 else False,
            metric_for_best_model="loss",
            greater_is_better=False,
            remove_unused_columns=False,
            seed=cfg.get("seed", 42),
            dataloader_num_workers=cfg.dataset.num_workers,
            lr_scheduler_type=p1_cfg.get("scheduler", "cosine"),
            warmup_steps=warmup_steps_p1,
            bf16=use_bf16,
            fp16=use_fp16,
            gradient_checkpointing=gradient_checkpointing,
            dataset_kwargs={"skip_prepare_dataset": True},
            report_to="tensorboard",
        )

        p1_report_type = p1_cfg.get("p1_report_type", "xray")

        trainer_p1 = SFTrainer(
            phase="phase1",
            loss_fn=criterion_p1,
            p1_report_type=p1_report_type,
            model=model,
            args=p1_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            optimizers=(optimizer_p1, None),
            callbacks=[callback_p1, EarlyStoppingCallback(early_stopping_patience=patience_p1)],
        )

        if epochs_p1 > 0:
            trainer_p1.train()

        model_dir_p1 = os.path.dirname(cp_p1) if os.path.dirname(cp_p1) else "./checkpoints"

        best_eval_loss_p1 = None
        if hasattr(trainer_p1, 'state') and trainer_p1.state.best_metric is not None:
            best_eval_loss_p1 = trainer_p1.state.best_metric
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': OmegaConf.to_container(cfg, resolve=True),
            'phase': 'phase1',
            'type': 'best',
            'best_eval_loss': best_eval_loss_p1,
            'epoch': trainer_p1.state.epoch if hasattr(trainer_p1, 'state') else epochs_p1,
        }, cp_p1)
        print(f" [*] Saved best Phase 1 checkpoint: {cp_p1} (eval_loss={best_eval_loss_p1})")

        last_cp_p1 = os.path.join(model_dir_p1, "last_phase1.pth")
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': OmegaConf.to_container(cfg, resolve=True),
            'phase': 'phase1',
            'type': 'last',
            'epoch': trainer_p1.state.epoch if hasattr(trainer_p1, 'state') else epochs_p1,
        }, last_cp_p1)

        logger_p1.close()
        print("Phase 1 complete!\n")

    if os.path.exists(cp_p1):
        print(f"Loading best Phase 1 checkpoint from: {cp_p1}")
        checkpoint = torch.load(cp_p1, map_location=device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        print(" -> Phase 1 weights loaded successfully!")
    else:
        print(f"\n[Warning] No Phase 1 checkpoint found at: {cp_p1}. Using initial weights.")

    # --- Phase 2: Classification and fusion training ---
    if run_phase2:
        print("\n" + "=" * 50)
        print("STARTING PHASE 2: CLASSIFIER AND FUSION TRAINING")
        print("=" * 50)

        model, classifier_type, fusion_type, num_classes = setup_phase2_modules(model, cfg, device)

        freeze_backbone = cfg.model.get("freeze_backbone", True)
        peft_type = cfg.model.peft.get('type', 'none')

        if freeze_backbone:
            for param in model.backbone.parameters():
                param.requires_grad = False
            print("  [Phase 2] Backbone is frozen.")
        else:
            if peft_type in ('lora', 'qlora'):
                for name, param in model.backbone.named_parameters():
                    if "lora" in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
                print(f"  [Phase 2] Backbone is trainable via PEFT ({peft_type}).")
            else:
                for param in model.backbone.parameters():
                    param.requires_grad = True
                print("  [Phase 2] Backbone is trainable (Full Fine-Tuning).")

        if model.fusion is not None:
            for param in model.fusion.parameters():
                param.requires_grad = True

        if model.head is not None:
            for param in model.head.parameters():
                param.requires_grad = True

        print("Phase 2 Parameter Summary:")
        model.print_parameter_summary()

        logger_p2 = TrainingLogger(log_dir=log_dir, experiment_name=experiment_name, phase="phase2")
        warmup_ratio_p2 = p2_cfg.get("warmup_ratio", 0.1)
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
        callback_p2 = XBoneTrainerCallback(logger_p2)

        phase2_loss_type = p2_cfg.get("phase2_loss_type", "asl")
        task_type = cfg.dataset.params.get("task_type", "multilabel")

        if task_type == "multiclass":
            weight_type = p2_cfg.get("class_weight_type", "cb")
            cb_beta = p2_cfg.get("cb_beta", 0.99)
            class_weights = compute_class_weights(train_dataset, num_classes, device, weight_type, cb_beta)
            loss_cfg = p2_cfg.get("loss", {}) or {}
            if classifier_type == "prototypical":
                from src.utils.losses import CombinedPhase2LossMulticlass
                criterion_p2 = CombinedPhase2LossMulticlass(
                    class_weights=class_weights,
                    proto_margin=loss_cfg.get("proto_margin", 0.5),
                    lambda_proto=loss_cfg.get("lambda_proto", 0.3),
                    label_smoothing=loss_cfg.get("label_smoothing", 0.1),
                )
            else:
                criterion_p2 = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
        else:
            if classifier_type == "prototypical" and phase2_loss_type != "bce":
                from src.utils.losses import CombinedPhase2Loss
                loss_cfg = p2_cfg.get("loss", {}) or {}
                criterion_p2 = CombinedPhase2Loss(
                    gamma_pos=loss_cfg.get("gamma_pos", 0.0),
                    gamma_neg=loss_cfg.get("gamma_neg", 4.0),
                    asl_clip=loss_cfg.get("asl_clip", 0.05),
                    proto_margin=loss_cfg.get("proto_margin", 0.5),
                    ood_margin=loss_cfg.get("ood_margin", 1.0),
                    lambda_proto=loss_cfg.get("lambda_proto", 0.5),
                    lambda_ood=loss_cfg.get("lambda_ood", 0.0),
                )
            else:
                criterion_p2 = nn.BCEWithLogitsLoss()

        trainable_params_p2 = [p for p in model.parameters() if p.requires_grad]
        optimizer_type_p2 = p2_cfg.get("optimizer", "adamw")
        if optimizer_type_p2 == "sgd":
            optimizer_p2 = SGD(trainable_params_p2, lr=lr_p2, weight_decay=wd_p2, momentum=0.9)
        else:
            optimizer_p2 = AdamW(trainable_params_p2, lr=lr_p2, weight_decay=wd_p2)

        steps_per_epoch_p2 = math.ceil(len(train_dataset) / cfg.dataset.batch_size)
        total_steps_p2 = steps_per_epoch_p2 * epochs_p2
        warmup_steps_p2 = int(warmup_ratio_p2 * total_steps_p2)

        os.environ["TENSORBOARD_LOGGING_DIR"] = logger_p2.tb_dir

        metric_for_best_p2 = p2_cfg.get("metric_for_best_model", "f1_macro" if task_type == "multiclass" else "loss")
        greater_is_better_p2 = True if metric_for_best_p2 in ["f1_macro", "accuracy"] else False

        p2_args = SFTConfig(
            output_dir=os.path.dirname(cp_p2) if os.path.dirname(cp_p2) else "./checkpoints",
            num_train_epochs=epochs_p2,
            learning_rate=lr_p2,
            weight_decay=wd_p2,
            per_device_train_batch_size=cfg.dataset.batch_size,
            per_device_eval_batch_size=cfg.dataset.batch_size,
            eval_strategy="epoch" if epochs_p2 > 0 else "no",
            save_strategy="epoch" if epochs_p2 > 0 else "no",
            save_total_limit=2,
            logging_strategy="steps",
            logging_steps=10,
            load_best_model_at_end=True if epochs_p2 > 0 else False,
            metric_for_best_model=metric_for_best_p2,
            greater_is_better=greater_is_better_p2,
            remove_unused_columns=False,
            seed=cfg.get("seed", 42),
            dataloader_num_workers=cfg.dataset.num_workers,
            lr_scheduler_type=p2_cfg.get("scheduler", "cosine"),
            warmup_steps=warmup_steps_p2,
            bf16=use_bf16,
            fp16=use_fp16,
            gradient_checkpointing=gradient_checkpointing,
            dataset_kwargs={"skip_prepare_dataset": True},
            report_to="tensorboard",
        )

        use_text_in_p2 = p2_cfg.get("use_text", fusion_type != "none")
        p2_report_type = p2_cfg.get("p2_report_type", "clinical")

        compute_metrics_fn = None
        if task_type == "multiclass":
            def compute_metrics_fn(eval_pred):
                logits, labels = eval_pred
                if isinstance(logits, tuple):
                    logits = logits[0]
                preds = np.argmax(logits, axis=-1)
                acc = accuracy_score(labels, preds)
                f1 = f1_score(labels, preds, average="macro", zero_division=0)
                return {"f1_macro": f1, "accuracy": acc}

        trainer_p2 = SFTrainer(
            phase="phase2",
            loss_fn=criterion_p2,
            use_text_in_p2=use_text_in_p2,
            p2_report_type=p2_report_type,
            model=model,
            args=p2_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            optimizers=(optimizer_p2, None),
            callbacks=[callback_p2, EarlyStoppingCallback(early_stopping_patience=patience_p2)],
            compute_metrics=compute_metrics_fn,
        )

        if epochs_p2 > 0:
            trainer_p2.train()

        model_dir_p2 = os.path.dirname(cp_p2) if os.path.dirname(cp_p2) else "./checkpoints"

        best_eval_loss_p2 = None
        if hasattr(trainer_p2, 'state') and trainer_p2.state.best_metric is not None:
            best_eval_loss_p2 = trainer_p2.state.best_metric
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': OmegaConf.to_container(cfg, resolve=True),
            'phase': 'phase2',
            'type': 'best',
            'best_eval_loss': best_eval_loss_p2,
            'epoch': trainer_p2.state.epoch if hasattr(trainer_p2, 'state') else epochs_p2,
        }, cp_p2)
        print(f" [*] Saved best Phase 2 checkpoint: {cp_p2} (eval_loss={best_eval_loss_p2})")

        last_cp_p2 = os.path.join(model_dir_p2, "last_phase2.pth")
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': OmegaConf.to_container(cfg, resolve=True),
            'phase': 'phase2',
            'type': 'last',
            'epoch': trainer_p2.state.epoch if hasattr(trainer_p2, 'state') else epochs_p2,
        }, last_cp_p2)

        # --- Post-training OOD calibration ---
        p2_params = cfg.model.get("params", {}).get("phase2", {}) or {}
        p2_report_type = p2_params.get("report_type", "clinical")
        calibrate_ood_on_validation(
            model=model,
            val_loader=val_loader,
            device=device,
            output_dir=model_dir_p2,
            p2_report_type=p2_report_type,
            fpr_threshold=0.05,
        )

        logger_p2.close()
        print("Phase 2 complete!\n")

    print("Training process completed successfully!")


if __name__ == "__main__":
    main()

