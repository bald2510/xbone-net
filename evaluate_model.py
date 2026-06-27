"""
Evaluation Framework for XBone-Net
====================================
Refactored from inference.py with structured JSON output, embedding
extraction, and bootstrap confidence intervals.

Usage (Hydra config override):
    # Basic evaluation
    python evaluate_model.py

    # With bootstrap CIs
    python evaluate_model.py --bootstrap

    # Save embeddings for OOD detection
    python evaluate_model.py --save-embeddings

    # Cross-dataset evaluation (train on fracatlas, evaluate on mura)
    python evaluate_model.py dataset=mura

    # Custom output directory
    python evaluate_model.py --output-dir results/my_experiment/seed_0

    # Full pipeline
    python evaluate_model.py --bootstrap --save-embeddings --output-dir results/exp1/seed_42
"""

import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import os
import json
import argparse
import datetime
from typing import Optional

import torch
import hydra
import numpy as np
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import (confusion_matrix, hamming_loss, f1_score,
                             accuracy_score, precision_score, recall_score,
                             roc_auc_score)
import evaluate

from models.builder import build_model, setup_phase2_modules
from local_datasets.builder import build_dataloader


# =========================================================================
# Prompt generation (identical to inference.py)
# =========================================================================

def generate_custom_prompts(
    pathologies: list[str],
    image_context: str = "a bone x-ray",
) -> dict[str, dict[str, str]]:
    """
    Generate automatic prompts based on the list of pathologies.
    Structure: 'this is an image of {image_context}; {pathology} presented in image'
    """
    return {
        path: {
            "positive": f"this is an image of {image_context}; {path.lower()} presented in image",
            "negative": f"this is an image of {image_context}; no {path.lower()} presented in image",
        }
        for path in pathologies
    }


# =========================================================================
# Checkpoint loading (identical to inference.py)
# =========================================================================

def adapt_state_dict_keys(
    state_dict: dict[str, torch.Tensor],
    model_keys: list[str],
) -> dict[str, torch.Tensor]:
    """
    Automatically adjusts mismatched key prefixes between standard OpenCLIP/BiomedCLIP
    and the wrapper XBone multimodal model.
    e.g., converts 'model.visual...' or 'visual...' to 'backbone.model.visual...'.
    """
    model_has_backbone_model = any(k.startswith("backbone.model.") for k in model_keys)
    checkpoint_keys = list(state_dict.keys())
    if not checkpoint_keys:
        return state_dict

    first_ckpt_key = checkpoint_keys[0]

    # Check if model has wrapper backbone structure but checkpoint doesn't
    if model_has_backbone_model and not first_ckpt_key.startswith("backbone.model."):
        # Case A: checkpoint keys start with 'model.' (e.g. standard open_clip)
        if first_ckpt_key.startswith("model."):
            print("Detected 'model.' prefix in checkpoint keys. Converting to 'backbone.model.' for compatibility.")
            return {k.replace("model.", "backbone.model.", 1): v for k, v in state_dict.items()}

        # Case B: checkpoint keys start with 'visual.', 'transformer.', or 'text.'
        elif first_ckpt_key.startswith(("visual.", "transformer.", "text.")):
            print("Detected raw OpenCLIP submodule prefix in checkpoint keys. Prepending 'backbone.model.' for compatibility.")
            return {"backbone.model." + k: v for k, v in state_dict.items()}

    return state_dict


def load_checkpoint_from_dir(
    model: torch.nn.Module,
    model_dir: str,
    device: torch.device,
) -> bool:
    """
    Loads model weights from a directory containing checkpoints.
    Searches for common checkpoint filenames (best_phase2.pth, best_phase1.pth, etc.).
    """
    if not os.path.isdir(model_dir):
        print(f"[Warning] model_dir '{model_dir}' does not exist or is not a directory.")
        return False

    # Search filenames in order of preference
    checkpoint_filenames = [
        "best_phase2.pth",
        "best_phase1.pth",
        "best_semantic_lora.pth",
        "open_clip_pytorch_model.bin",
        "checkpoint_epoch_10.pth",
    ]

    checkpoint_path = None
    for filename in checkpoint_filenames:
        path = os.path.join(model_dir, filename)
        if os.path.exists(path):
            checkpoint_path = path
            break

    # Fallback to any .pth or .bin file in the directory
    if not checkpoint_path:
        for file in os.listdir(model_dir):
            if file.endswith((".pth", ".bin")):
                checkpoint_path = os.path.join(model_dir, file)
                break

    if checkpoint_path:
        print(f"Loading weights from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        state_dict = adapt_state_dict_keys(state_dict, list(model.state_dict().keys()))
        model.load_state_dict(state_dict, strict=False)
        print("-> Checkpoint loaded successfully from directory!\n")
        return True
    else:
        print(f"[Warning] No checkpoint file (.pth or .bin) found in directory '{model_dir}'")
        return False


def load_model_checkpoint(
    model: torch.nn.Module,
    cfg: DictConfig,
    device: torch.device,
) -> None:
    """Unified checkpoint loading from dir or file path, mirroring inference.py logic."""
    loaded = False
    params_cfg = cfg.get("params", {}) or {}
    model_dir = params_cfg.get("model_dir", None)

    # Try loading from specified directory first
    if model_dir:
        print(f"Searching for checkpoints in folder: {model_dir}")
        loaded = load_checkpoint_from_dir(model, model_dir, device)

    if not loaded:
        # Fallback to traditional checkpoint paths
        checkpoint_path = cfg.get("checkpoint_path", None)
        if not checkpoint_path:
            checkpoint_path = params_cfg.get("checkpoint_path", None)
            if not checkpoint_path:
                phase2_cfg = params_cfg.get("phase2", {}) or {}
                checkpoint_path = phase2_cfg.get("checkpoint_path", None)
                if not checkpoint_path:
                    phase1_cfg = params_cfg.get("phase1", {}) or {}
                    checkpoint_path = phase1_cfg.get("checkpoint_path", None)

        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Loading weights from file: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            state_dict = adapt_state_dict_keys(state_dict, list(model.state_dict().keys()))
            model.load_state_dict(state_dict, strict=False)
            print("-> Checkpoint loaded successfully from file!\n")
        else:
            print("-> [Info] No checkpoint found or defined. Running model with initial/random weights.\n")


# =========================================================================
# Core evaluation loop
# =========================================================================

def run_evaluation(
    model: torch.nn.Module,
    test_loader,
    pathologies: list[str],
    is_classifier: bool,
    device: torch.device,
    temperature: float = 0.07,
    image_context: str = "a bone x-ray",
    p2_report_type: str = "clinical",
    save_embeddings: bool = False,
) -> dict:
    """
    Run forward pass on test set.

    Returns dict with keys:
        - all_probs: np.ndarray  (N, C)
        - all_ground_truths: np.ndarray  (N, C)
        - image_embeddings: np.ndarray or None  (N, D)
        - text_embeddings: dict[str, np.ndarray] or None
    """
    tokenizer = model.backbone.tokenizer

    # ------------------------------------------------------------------
    # Pre-extract text prompt features for zero-shot mode
    # ------------------------------------------------------------------
    text_features_dict: dict[str, torch.Tensor] = {}
    text_embeddings_np: Optional[dict[str, np.ndarray]] = None

    if not is_classifier:
        print("\nZero-shot mode: Pre-extracting text prompt features...")
        prompt_dict = generate_custom_prompts(pathologies, image_context)
        print(f"Generated prompts for pathologies: {list(prompt_dict.values())}")

        with torch.no_grad():
            for path, pair in prompt_dict.items():
                pos_tokens = tokenizer([pair["positive"]]).to(device)
                neg_tokens = tokenizer([pair["negative"]]).to(device)

                pos_feat = model.backbone.model.encode_text(pos_tokens)
                neg_feat = model.backbone.model.encode_text(neg_tokens)

                pos_feat /= pos_feat.norm(dim=-1, keepdim=True)
                neg_feat /= neg_feat.norm(dim=-1, keepdim=True)

                text_features_dict[path] = torch.cat([pos_feat, neg_feat], dim=0)

        if save_embeddings:
            text_embeddings_np = {
                path: text_features_dict[path].cpu().numpy()
                for path in pathologies
            }

    # ------------------------------------------------------------------
    # Forward pass over test set
    # ------------------------------------------------------------------
    all_probs = []
    all_ground_truths = []
    all_image_embeds = [] if save_embeddings else None

    print("\nScanning test dataset...")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            if len(batch) == 4:
                images, xray_ids, clinical_ids, labels = batch
                # Select text based on p2_report_type
                if p2_report_type == "both":
                    input_ids = None  # handled below
                elif p2_report_type == "xray":
                    input_ids = xray_ids
                else:  # "clinical" (default)
                    input_ids = clinical_ids
            elif len(batch) == 3:
                images, input_ids, labels = batch
                xray_ids = None
                clinical_ids = None
            else:
                raise ValueError(f"Unexpected batch format with length {len(batch)}")

            images = images.to(device)
            if input_ids is not None and isinstance(input_ids, torch.Tensor):
                input_ids = input_ids.to(device)

            if is_classifier:
                if p2_report_type == "both" and len(batch) == 4:
                    # "both" mode: encode xray + clinical separately, average
                    xray_ids = xray_ids.to(device)
                    clinical_ids = clinical_ids.to(device)
                    img_feat, xray_feat = model.backbone(images, xray_ids)
                    _, clinical_feat = model.backbone(images, clinical_ids)
                    text_feat = (xray_feat + clinical_feat) / 2.0
                    if model.fusion is not None:
                        fused = model.fusion(img_feat, text_feat)
                    else:
                        fused = img_feat
                    if save_embeddings:
                        all_image_embeds.append(fused.cpu().numpy())
                    logits = model.head(fused)
                elif save_embeddings:
                    # Capture image embeddings before the head
                    img_feats, txt_feats = model.backbone(images, input_ids)
                    if txt_feats is None:
                        fused = img_feats
                    else:
                        fused = model.fusion(img_feats, txt_feats)
                    all_image_embeds.append(fused.cpu().numpy())
                    logits = model.head(fused)
                else:
                    outputs = model(images, input_ids)
                    if isinstance(outputs, tuple):
                        logits = outputs[0]
                    else:
                        logits = outputs
                batch_probs = torch.softmax(logits, dim=-1)
            else:
                # Cosine similarity matching (zero-shot)
                img_feat = model.backbone.model.encode_image(images)
                img_feat /= img_feat.norm(dim=-1, keepdim=True)

                if save_embeddings:
                    all_image_embeds.append(img_feat.cpu().numpy())

                batch_probs_list = []
                for path in pathologies:
                    text_weights = text_features_dict[path]
                    logits = img_feat @ text_weights.T
                    probs = torch.softmax(logits / temperature, dim=-1)[:, 0].unsqueeze(1)
                    batch_probs_list.append(probs)
                batch_probs = torch.cat(batch_probs_list, dim=1)

            all_probs.append(batch_probs.cpu())
            all_ground_truths.append(labels.cpu())

    all_probs_np = torch.cat(all_probs, dim=0).numpy()
    all_gt_np = torch.cat(all_ground_truths, dim=0).numpy()

    image_embeddings_np = None
    if save_embeddings and all_image_embeds:
        image_embeddings_np = np.concatenate(all_image_embeds, axis=0)

    return {
        "all_probs": all_probs_np,
        "all_ground_truths": all_gt_np,
        "image_embeddings": image_embeddings_np,
        "text_embeddings": text_embeddings_np,
    }


# =========================================================================
# Metric computation
# =========================================================================

def compute_metrics_multiclass(
    all_probs: np.ndarray,
    all_gt: np.ndarray,
    class_names: list[str],
) -> dict:
    """
    Compute evaluation metrics for multi-class classification.

    Args:
        all_probs: [N, C] softmax probabilities
        all_gt: [N] integer class indices
        class_names: list of class name strings

    Returns a dict with keys: auroc_macro, f1_macro, f1_per_class,
        accuracy, sensitivity, specificity, precision, confusion_matrix
    """
    num_classes = len(class_names)
    all_preds = np.argmax(all_probs, axis=1)

    # Overall accuracy
    acc = float(accuracy_score(all_gt, all_preds))

    # Macro-averaged metrics
    macro_f1 = float(f1_score(all_gt, all_preds, average="macro", zero_division=0))
    macro_prec = float(precision_score(all_gt, all_preds, average="macro", zero_division=0))
    macro_rec = float(recall_score(all_gt, all_preds, average="macro", zero_division=0))

    # Per-class F1
    per_class_f1_arr = f1_score(all_gt, all_preds, average=None, zero_division=0)
    per_class_prec_arr = precision_score(all_gt, all_preds, average=None, zero_division=0)
    per_class_rec_arr = recall_score(all_gt, all_preds, average=None, zero_division=0)

    f1_per_class = {}
    for i, name in enumerate(class_names):
        if i < len(per_class_f1_arr):
            f1_per_class[name] = float(per_class_f1_arr[i])

    # AUROC (one-vs-rest)
    try:
        # Check if probabilities sum to 1.0; if not, L1-normalize them for OvR AUROC
        probs_sum = all_probs.sum(axis=1, keepdims=True)
        if not np.allclose(probs_sum, 1.0, atol=1e-5):
            probs_for_auc = all_probs / (probs_sum + 1e-8)
        else:
            probs_for_auc = all_probs
        macro_auroc = float(roc_auc_score(
            all_gt, probs_for_auc, average="macro", multi_class="ovr",
        ))
    except ValueError:
        macro_auroc = float("nan")

    # Confusion matrix
    cm = confusion_matrix(all_gt, all_preds)

    # Per-class specificity
    specificities = []
    for i in range(num_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificities.append(spec)
    macro_spec = float(np.mean(specificities))

    # Print per-class metrics
    print("\n=== MULTI-CLASS DETAILED METRICS ===")
    for i, name in enumerate(class_names):
        prec_i = float(per_class_prec_arr[i]) if i < len(per_class_prec_arr) else 0.0
        rec_i = float(per_class_rec_arr[i]) if i < len(per_class_rec_arr) else 0.0
        f1_i = float(per_class_f1_arr[i]) if i < len(per_class_f1_arr) else 0.0
        spec_i = specificities[i] if i < len(specificities) else 0.0
        print(
            f"{name:25s} | F1: {f1_i:.3f} | Prec: {prec_i:.3f} | "
            f"Rec: {rec_i:.3f} | Spec: {spec_i:.3f}"
        )

    print("\n=== OVERALL MULTI-CLASS METRICS ===")
    print(f"Accuracy                      : {acc:.4f}")
    print(f"Macro AUROC                   : {macro_auroc:.4f}")
    print(f"Macro F1-Score                : {macro_f1:.4f}")
    print(f"Macro Precision               : {macro_prec:.4f}")
    print(f"Macro Recall (Sensitivity)    : {macro_rec:.4f}")
    print(f"Macro Specificity             : {macro_spec:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    metrics = {
        "auroc_macro": macro_auroc,
        "f1_macro": macro_f1,
        "f1_per_class": f1_per_class,
        "accuracy": acc,
        "sensitivity": macro_rec,
        "specificity": macro_spec,
        "precision": macro_prec,
        "confusion_matrix": cm.tolist(),
    }

    return metrics


def compute_metrics(
    all_probs: np.ndarray,
    all_ground_truths: np.ndarray,
    pathologies: list[str],
    is_multilabel: bool,
) -> dict:
    """
    Compute evaluation metrics using HuggingFace evaluate library.

    Returns a structured dict matching the JSON output schema:
        auroc_macro, auroc_per_class, f1_macro, f1_per_class,
        accuracy, sensitivity, specificity, precision, hamming_loss (multi-label only)
    """
    # Load HuggingFace evaluate metrics
    accuracy_metric = evaluate.load("accuracy")
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    f1_metric = evaluate.load("f1")
    roc_auc_metric = evaluate.load("roc_auc")

    auroc_per_class: dict[str, float] = {}
    f1_per_class: dict[str, float] = {}
    per_class_results: list[dict] = []

    print("\n=== DETAILED METRICS EVALUATION ===")
    for idx, path in enumerate(pathologies):
        gt_labels = all_ground_truths[:, idx]
        probs = all_probs[:, idx]

        if len(np.unique(gt_labels)) < 2:
            print(f"[Warning] Pathology '{path}' skipped: only 1 class present in test set.")
            continue

        preds = (probs >= 0.5).astype(int)

        # Specificity via confusion matrix
        tn, fp, fn, tp = confusion_matrix(gt_labels, preds).ravel()
        print(f"\nConfusion Matrix for {path}:")
        print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        # Compute metrics using HuggingFace evaluate
        acc = accuracy_metric.compute(predictions=preds, references=gt_labels)["accuracy"]
        prec = precision_metric.compute(predictions=preds, references=gt_labels)["precision"]
        rec = recall_metric.compute(predictions=preds, references=gt_labels)["recall"]
        f1 = f1_metric.compute(predictions=preds, references=gt_labels)["f1"]
        auroc = roc_auc_metric.compute(prediction_scores=probs, references=gt_labels)["roc_auc"]

        auroc_per_class[path] = float(auroc)
        f1_per_class[path] = float(f1)

        per_class_results.append({
            "Pathology": path,
            "AUROC": auroc,
            "Accuracy": acc,
            "F1_Score": f1,
            "Precision": prec,
            "Recall_Sens": rec,
            "Specificity": specificity,
        })

        print(
            f"{path:25s} | AUC: {auroc:.3f} | Acc: {acc:.3f} | "
            f"F1: {f1:.3f} | Prec: {prec:.3f} | Rec: {rec:.3f} | Spec: {specificity:.3f}"
        )

    # ----- Aggregate metrics -----
    metrics: dict = {}

    if is_multilabel:
        print("\n=== OVERALL MULTI-LABEL METRICS ===")
        all_preds = (all_probs >= 0.5).astype(int)

        h_loss = float(hamming_loss(all_ground_truths, all_preds))
        macro_f1 = float(f1_score(
            all_ground_truths.astype(int), all_preds,
            average="macro", zero_division=0,
        ))

        try:
            from sklearn.metrics import roc_auc_score
            macro_auroc = float(roc_auc_score(
                all_ground_truths.astype(int), all_probs,
                average="macro", multi_class="ovr",
            ))
        except ValueError:
            macro_auroc = float("nan")

        # Overall accuracy, sensitivity, specificity, precision (macro-averaged across classes)
        overall_acc = float(np.mean([r["Accuracy"] for r in per_class_results])) if per_class_results else 0.0
        overall_sens = float(np.mean([r["Recall_Sens"] for r in per_class_results])) if per_class_results else 0.0
        overall_spec = float(np.mean([r["Specificity"] for r in per_class_results])) if per_class_results else 0.0
        overall_prec = float(np.mean([r["Precision"] for r in per_class_results])) if per_class_results else 0.0

        metrics = {
            "auroc_macro": float(macro_auroc),
            "auroc_per_class": auroc_per_class,
            "f1_macro": float(macro_f1),
            "f1_per_class": f1_per_class,
            "accuracy": overall_acc,
            "sensitivity": overall_sens,
            "specificity": overall_spec,
            "precision": overall_prec,
            "hamming_loss": h_loss,
        }

        print(f"Macro AUROC                   : {macro_auroc:.4f}")
        print(f"Macro F1-Score                : {macro_f1:.4f}")
        print(f"Hamming Loss                  : {h_loss:.4f}")
        print(f"Mean Accuracy                 : {overall_acc:.4f}")
        print(f"Mean Sensitivity              : {overall_sens:.4f}")
        print(f"Mean Specificity              : {overall_spec:.4f}")
        print(f"Mean Precision                : {overall_prec:.4f}")
    else:
        print("\n=== SINGLE-LABEL METRICS SUMMARY ===")
        if per_class_results:
            res = per_class_results[0]
            metrics = {
                "auroc_macro": float(res["AUROC"]),
                "auroc_per_class": auroc_per_class,
                "f1_macro": float(res["F1_Score"]),
                "f1_per_class": f1_per_class,
                "accuracy": float(res["Accuracy"]),
                "sensitivity": float(res["Recall_Sens"]),
                "specificity": float(res["Specificity"]),
                "precision": float(res["Precision"]),
            }
            print(f"Accuracy            : {res['Accuracy']:.4f}")
            print(f"Precision           : {res['Precision']:.4f}")
            print(f"Recall (Sensitivity): {res['Recall_Sens']:.4f}")
            print(f"Specificity         : {res['Specificity']:.4f}")
            print(f"F1-Score            : {res['F1_Score']:.4f}")
            print(f"AUROC               : {res['AUROC']:.4f}")
        else:
            print("No evaluation results computed.")
            metrics = {}

    return metrics


# =========================================================================
# Bootstrap confidence intervals
# =========================================================================

def _safe_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """AUROC that returns NaN for degenerate samples."""
    from sklearn.metrics import roc_auc_score
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def _safe_f1(labels: np.ndarray, scores: np.ndarray) -> float:
    """F1 from predicted probabilities (threshold=0.5)."""
    from sklearn.metrics import f1_score
    preds = (scores >= 0.5).astype(int)
    return float(f1_score(labels, preds, zero_division=0))


def bootstrap_confidence_intervals(
    all_probs: np.ndarray,
    all_ground_truths: np.ndarray,
    pathologies: list[str],
    is_multilabel: bool,
    n_bootstrap: int = 10_000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict:
    """
    Paired bootstrap resampling for 95% CIs on AUROC and F1.

    Returns:
        {
            "auroc_macro": [lower, upper],
            "f1_macro": [lower, upper],
            "auroc_per_class": {"pathology": [lower, upper], ...},
            "f1_per_class": {"pathology": [lower, upper], ...},
        }
    """
    rng = np.random.RandomState(seed)
    n = len(all_ground_truths)

    auroc_boot: dict[str, list[float]] = {p: [] for p in pathologies}
    f1_boot: dict[str, list[float]] = {p: [] for p in pathologies}
    auroc_macro_boot: list[float] = []
    f1_macro_boot: list[float] = []

    print(f"\nRunning {n_bootstrap:,} bootstrap resamples for 95% CIs...")
    for _ in tqdm(range(n_bootstrap), desc="Bootstrap"):
        idx = rng.randint(0, n, size=n)
        boot_gt = all_ground_truths[idx]
        boot_probs = all_probs[idx]

        per_class_aurocs = []
        per_class_f1s = []

        for ci, path in enumerate(pathologies):
            gt_col = boot_gt[:, ci]
            prob_col = boot_probs[:, ci]

            auc_val = _safe_auroc(gt_col, prob_col)
            f1_val = _safe_f1(gt_col, prob_col)

            auroc_boot[path].append(auc_val)
            f1_boot[path].append(f1_val)

            if not np.isnan(auc_val):
                per_class_aurocs.append(auc_val)
            per_class_f1s.append(f1_val)

        # Macro average across classes for this bootstrap sample
        auroc_macro_boot.append(float(np.mean(per_class_aurocs)) if per_class_aurocs else float("nan"))
        f1_macro_boot.append(float(np.mean(per_class_f1s)))

    lo = (alpha / 2) * 100
    hi = (1 - alpha / 2) * 100

    def _ci(values: list[float]) -> list[float]:
        arr = np.array([v for v in values if not np.isnan(v)])
        if len(arr) == 0:
            return [float("nan"), float("nan")]
        return [float(np.percentile(arr, lo)), float(np.percentile(arr, hi))]

    ci_result: dict = {
        "auroc_macro": _ci(auroc_macro_boot),
        "f1_macro": _ci(f1_macro_boot),
        "auroc_per_class": {p: _ci(auroc_boot[p]) for p in pathologies},
        "f1_per_class": {p: _ci(f1_boot[p]) for p in pathologies},
    }

    print(f"  AUROC macro 95% CI: [{ci_result['auroc_macro'][0]:.4f}, {ci_result['auroc_macro'][1]:.4f}]")
    print(f"  F1 macro 95% CI:    [{ci_result['f1_macro'][0]:.4f}, {ci_result['f1_macro'][1]:.4f}]")

    return ci_result


# =========================================================================
# Output helpers
# =========================================================================

def save_results_json(
    metrics: dict,
    cfg: DictConfig,
    output_dir: str,
    ci_95: Optional[dict] = None,
) -> str:
    """Save structured evaluation results to JSON."""
    os.makedirs(output_dir, exist_ok=True)

    # Resolve experiment name and seed from config
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


def save_embeddings(
    image_embeddings: Optional[np.ndarray],
    text_embeddings: Optional[dict[str, np.ndarray]],
    labels: np.ndarray,
    output_dir: str,
) -> Optional[str]:
    """Save embeddings to .npz for downstream OOD detection."""
    if image_embeddings is None:
        print("[Warning] No image embeddings to save.")
        return None

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "embeddings.npz")

    save_dict = {
        "image_embeddings": image_embeddings,
        "labels": labels,
    }

    if text_embeddings is not None:
        # Flatten per-pathology text embeddings into single arrays
        for path_name, emb in text_embeddings.items():
            safe_key = f"text_embeddings_{path_name.replace(' ', '_')}"
            save_dict[safe_key] = emb

    np.savez_compressed(save_path, **save_dict)
    print(f"Embeddings saved to: {save_path}")
    print(f"  Image embeddings shape: {image_embeddings.shape}")
    print(f"  Labels shape:           {labels.shape}")
    return save_path


# =========================================================================
# CLI argument parsing (non-Hydra flags)
# =========================================================================

def parse_extra_args() -> argparse.Namespace:
    """
    Parse non-Hydra CLI arguments.

    Hydra consumes its own args (overrides like dataset=mura), so we extract
    our custom flags manually from sys.argv before Hydra processes it.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--bootstrap", action="store_true", default=False,
                        help="Compute bootstrap 95%% confidence intervals")
    parser.add_argument("--n-bootstrap", type=int, default=10_000,
                        help="Number of bootstrap resamples (default: 10000)")
    parser.add_argument("--save-embeddings", action="store_true", default=False,
                        help="Save image/text embeddings to .npz")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save JSON results and embeddings")

    # Parse only known args — Hydra overrides are left untouched
    extra_args, remaining = parser.parse_known_args()

    # Restore sys.argv so Hydra only sees its own overrides
    sys.argv = [sys.argv[0]] + remaining

    return extra_args


# =========================================================================
# Main entry point
# =========================================================================

@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting evaluation on device: {device}")

    # ---- 1. Build model & load checkpoint ----
    print("Building model architecture...")
    model = build_model(cfg.model).to(device)

    # Mimic train_2stage.py: dynamically attach fusion/head based on phase2 config
    params_cfg = cfg.get("params", {}) or {}
    p2_phase_cfg = params_cfg.get("phase2", {}) or {}
    _, classifier_type, fusion_type, _ = setup_phase2_modules(model, cfg, device)

    model.eval()
    load_model_checkpoint(model, cfg, device)

    preprocess = model.backbone.preprocess
    tokenizer = model.backbone.tokenizer

    # ---- 2. Build test dataloader ----
    print(f"Loading test dataset: {cfg.dataset.name}...")
    test_loader = build_dataloader(
        cfg=cfg.dataset,
        split="test",
        transform=preprocess,
        tokenizer=tokenizer,
    )

    pathologies = cfg.dataset.params.get('classes', cfg.dataset.params.get('pathologies', []))
    is_classifier = classifier_type != "none"
    task_type = cfg.dataset.params.get('task_type', 'multiclass')
    is_multilabel = (task_type == 'multilabel')

    temperature = params_cfg.get("temperature", 0.07)
    image_context = params_cfg.get("image_context", "a bone x-ray")
    p2_report_type = p2_phase_cfg.get("p2_report_type", "clinical")

    # ---- 3. Run evaluation ----
    eval_output = run_evaluation(
        model=model,
        test_loader=test_loader,
        pathologies=list(pathologies),
        is_classifier=is_classifier,
        device=device,
        temperature=temperature,
        image_context=image_context,
        p2_report_type=p2_report_type,
        save_embeddings=extra_args.save_embeddings,
    )

    all_probs = eval_output["all_probs"]
    all_gt = eval_output["all_ground_truths"]

    # ---- 4. Compute metrics ----
    if is_multilabel:
        metrics = compute_metrics(all_probs, all_gt, list(pathologies), is_multilabel)
    else:
        metrics = compute_metrics_multiclass(all_probs, all_gt, list(pathologies))

    # ---- 5. Bootstrap CIs (optional) ----
    ci_95 = None
    if extra_args.bootstrap:
        ci_95 = bootstrap_confidence_intervals(
            all_probs, all_gt,
            pathologies=list(pathologies),
            is_multilabel=is_multilabel,
            n_bootstrap=extra_args.n_bootstrap,
        )

    # ---- 6. Determine output directory ----
    if extra_args.output_dir:
        output_dir = extra_args.output_dir
    else:
        # Default: results/{experiment_name}/seed_{N}/
        exp_name = params_cfg.get("experiment_name", cfg.get("experiment_name", "default"))
        seed_val = params_cfg.get("seed", cfg.get("seed", 0))
        output_dir = os.path.join("results", str(exp_name), f"seed_{seed_val}")

    # ---- 7. Save structured JSON results ----
    save_results_json(metrics, cfg, output_dir, ci_95=ci_95)

    # ---- 8. Save embeddings (optional) ----
    if extra_args.save_embeddings:
        save_embeddings(
            image_embeddings=eval_output["image_embeddings"],
            text_embeddings=eval_output["text_embeddings"],
            labels=all_gt,
            output_dir=output_dir,
        )

    print("\nEvaluation complete!")


# Parse extra CLI args *before* Hydra takes over
extra_args = parse_extra_args()

if __name__ == "__main__":
    main()
