"""
Evaluation Metrics & Bootstrap Resampling for XBone-Net.
===============================================================================
Computes standard classification metrics for XBone-Net evaluation:
  - Multi-class (single-label) diagnosis: AUROC, F1, precision, recall, specificity
  - Multi-label pathology detection: Macro AUROC, F1, Hamming loss, per-class metrics
  - Paired bootstrap resampling: 95% confidence interval estimation (10,000 resamples)

Per-class metrics use HuggingFace evaluate and scikit-learn for aggregation.
"""

import sys
import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    confusion_matrix, f1_score, accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, roc_auc_score, average_precision_score,
    hamming_loss,
)

# ============================================================
# HuggingFace Evaluate Package Import Isolation
# ============================================================

# Temporarily remove project root from sys.path so that ``import evaluate``
# resolves to the HuggingFace ``evaluate`` package rather than a local
# module with the same name.
_sys_path = list(sys.path)
try:
    if "" in sys.path:
        sys.path.remove("")
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if root_dir in sys.path:
        sys.path.remove(root_dir)
    import evaluate
finally:
    sys.path = _sys_path


# ============================================================
# Robust Safe Helper Functions
# ============================================================

def _safe_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute AUROC safely, returning NaN when only one class is present.

    Args:
        labels: Binary ground-truth labels of shape (N,).
        scores: Predicted probabilities of shape (N,).

    Returns:
        AUROC value in [0, 1], or NaN if degenerate single-class batch.
    """
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def _safe_f1(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute binary F1 from probabilities with a 0.5 decision threshold.

    Formula:
        F1 = 2 * (Precision * Recall) / (Precision + Recall)

    Args:
        labels: Binary ground-truth labels of shape (N,).
        scores: Predicted probabilities of shape (N,).

    Returns:
        F1 score in [0, 1]. Returns 0.0 when denominator is zero.
    """
    preds = (scores >= 0.5).astype(int)
    return float(f1_score(labels, preds, zero_division=0))


# ============================================================
# Classification Evaluation Functions
# ============================================================

def compute_metrics_multiclass(
    all_probs: np.ndarray,
    all_gt: np.ndarray,
    class_names: list[str],
) -> dict:
    """Compute robust single-label multiclass metrics with fixed class order."""
    all_probs = np.asarray(all_probs, dtype=np.float64)
    all_gt = np.asarray(all_gt)
    if all_gt.ndim > 1:
        all_gt = np.argmax(all_gt, axis=-1)
    all_gt = all_gt.astype(np.int64).reshape(-1)

    num_classes = len(class_names)
    labels_order = np.arange(num_classes, dtype=np.int64)
    if all_probs.ndim != 2 or all_probs.shape[1] != num_classes:
        raise ValueError(
            f"Expected probabilities [N,{num_classes}], got {all_probs.shape}."
        )
    if all_probs.shape[0] != all_gt.shape[0]:
        raise ValueError("Probability and ground-truth sample counts differ.")
    if np.any(all_gt < 0) or np.any(all_gt >= num_classes):
        raise ValueError("Ground-truth labels are outside the configured class range.")

    row_sums = all_probs.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("Every probability row must have a positive sum.")
    probs = all_probs / row_sums
    all_preds = np.argmax(probs, axis=1)

    acc = float(accuracy_score(all_gt, all_preds))
    balanced_acc = float(balanced_accuracy_score(all_gt, all_preds))
    macro_f1 = float(f1_score(
        all_gt, all_preds, labels=labels_order, average="macro", zero_division=0
    ))
    macro_prec = float(precision_score(
        all_gt, all_preds, labels=labels_order, average="macro", zero_division=0
    ))
    macro_rec = float(recall_score(
        all_gt, all_preds, labels=labels_order, average="macro", zero_division=0
    ))

    per_class_f1_arr = f1_score(
        all_gt, all_preds, labels=labels_order, average=None, zero_division=0
    )
    per_class_prec_arr = precision_score(
        all_gt, all_preds, labels=labels_order, average=None, zero_division=0
    )
    per_class_rec_arr = recall_score(
        all_gt, all_preds, labels=labels_order, average=None, zero_division=0
    )

    auroc_per_class = {}
    auprc_per_class = {}
    valid_aurocs = []
    valid_auprcs = []
    for class_id, name in enumerate(class_names):
        binary_gt = (all_gt == class_id).astype(np.int64)
        if np.unique(binary_gt).size < 2:
            auroc = float("nan")
            auprc = float("nan")
        else:
            auroc = float(roc_auc_score(binary_gt, probs[:, class_id]))
            auprc = float(average_precision_score(binary_gt, probs[:, class_id]))
            valid_aurocs.append(auroc)
            valid_auprcs.append(auprc)
        auroc_per_class[name] = auroc
        auprc_per_class[name] = auprc

    macro_auroc = float(np.mean(valid_aurocs)) if valid_aurocs else float("nan")
    macro_auprc = float(np.mean(valid_auprcs)) if valid_auprcs else float("nan")

    cm = confusion_matrix(all_gt, all_preds, labels=labels_order)
    specificities = []
    for class_id in labels_order:
        tp = cm[class_id, class_id]
        fn = cm[class_id, :].sum() - tp
        fp = cm[:, class_id].sum() - tp
        tn = cm.sum() - tp - fn - fp
        specificities.append(float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0)
    macro_spec = float(np.mean(specificities))

    f1_per_class = {
        name: float(per_class_f1_arr[i]) for i, name in enumerate(class_names)
    }
    precision_per_class = {
        name: float(per_class_prec_arr[i]) for i, name in enumerate(class_names)
    }
    recall_per_class = {
        name: float(per_class_rec_arr[i]) for i, name in enumerate(class_names)
    }
    specificity_per_class = {
        name: float(specificities[i]) for i, name in enumerate(class_names)
    }

    print("\n=== MULTI-CLASS DETAILED METRICS ===")
    for i, name in enumerate(class_names):
        auc_text = (
            f"{auroc_per_class[name]:.3f}"
            if not np.isnan(auroc_per_class[name]) else "N/A"
        )
        print(
            f"{name:25s} | F1: {per_class_f1_arr[i]:.3f} | "
            f"Prec: {per_class_prec_arr[i]:.3f} | Rec: {per_class_rec_arr[i]:.3f} | "
            f"Spec: {specificities[i]:.3f} | AUROC: {auc_text}"
        )

    print("\n=== OVERALL MULTI-CLASS METRICS ===")
    print(f"Accuracy                      : {acc:.4f}")
    print(f"Balanced Accuracy             : {balanced_acc:.4f}")
    print(f"Macro AUROC                   : {macro_auroc:.4f}")
    print(f"Macro AUPRC                   : {macro_auprc:.4f}")
    print(f"Macro F1-Score                : {macro_f1:.4f}")
    print(f"Macro Precision               : {macro_prec:.4f}")
    print(f"Macro Recall (Sensitivity)    : {macro_rec:.4f}")
    print(f"Macro Specificity             : {macro_spec:.4f}")
    print("\nConfusion Matrix:")
    print(cm)

    return {
        "auroc_macro": macro_auroc,
        "auroc_per_class": auroc_per_class,
        "auprc_macro": macro_auprc,
        "auprc_per_class": auprc_per_class,
        "f1_macro": macro_f1,
        "f1_per_class": f1_per_class,
        "accuracy": acc,
        "balanced_accuracy": balanced_acc,
        "sensitivity": macro_rec,
        "specificity": macro_spec,
        "precision": macro_prec,
        "precision_per_class": precision_per_class,
        "recall_per_class": recall_per_class,
        "specificity_per_class": specificity_per_class,
        "confusion_matrix": cm.tolist(),
    }


def compute_metrics(
    all_probs: np.ndarray,
    all_ground_truths: np.ndarray,
    pathologies: list[str],
    is_multilabel: bool,
) -> dict:
    """Compute evaluation metrics for multi-label or binary classification.

    Args:
        all_probs: Predicted probabilities of shape (N, C).
        all_ground_truths: Ground-truth labels of shape (N, C) for multi-label
            or (N, 1) for binary.
        pathologies: Ordered list of pathology names.
        is_multilabel: If True, aggregate using multi-label macro averaging;
            otherwise report single-label metrics.

    Returns:
        Dictionary of aggregated metrics including auroc_macro, f1_macro,
        accuracy, sensitivity, specificity, and precision.
    """
    # --- Load HuggingFace evaluation metrics ---
    accuracy_metric = evaluate.load("accuracy")
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    f1_metric = evaluate.load("f1")
    roc_auc_metric = evaluate.load("roc_auc")

    auroc_per_class: dict[str, float] = {}
    f1_per_class: dict[str, float] = {}
    per_class_results: list[dict] = []

    # --- Per-pathology evaluation loop ---
    print("\n=== DETAILED METRICS EVALUATION ===")
    for idx, path in enumerate(pathologies):
        gt_labels = all_ground_truths[:, idx]
        probs = all_probs[:, idx]

        if len(np.unique(gt_labels)) < 2:
            print(f"[Warning] Pathology '{path}' skipped: only 1 class present in test set.")
            continue

        preds = (probs >= 0.5).astype(int)

        tn, fp, fn, tp = confusion_matrix(gt_labels, preds).ravel()
        print(f"\nConfusion Matrix for {path}:")
        print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

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

    metrics: dict = {}

    # --- Aggregation logic ---
    if is_multilabel:
        print("\n=== OVERALL MULTI-LABEL METRICS ===")
        all_preds = (all_probs >= 0.5).astype(int)

        h_loss = float(hamming_loss(all_ground_truths, all_preds))
        macro_f1 = float(f1_score(
            all_ground_truths.astype(int), all_preds,
            average="macro", zero_division=0,
        ))

        try:
            macro_auroc = float(roc_auc_score(
                all_ground_truths.astype(int), all_probs,
                average="macro", multi_class="ovr",
            ))
        except ValueError:
            macro_auroc = float("nan")

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
                "f1_macro": float(res["F1_Score"]),
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


# ============================================================
# Bootstrap Resampling & Confidence Intervals
# ============================================================

def bootstrap_confidence_intervals(
    all_probs: np.ndarray,
    all_ground_truths: np.ndarray,
    pathologies: list[str],
    is_multilabel: bool,
    n_bootstrap: int = 10_000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict:
    """Compute paired percentile bootstrap confidence intervals.

    For multiclass tasks, predictions are always obtained with ``argmax`` so the
    bootstrap statistic matches the reported point estimate.
    """
    all_probs = np.asarray(all_probs)
    all_ground_truths = np.asarray(all_ground_truths)
    if not is_multilabel and all_ground_truths.ndim > 1:
        all_ground_truths = np.argmax(all_ground_truths, axis=-1)

    n = len(all_ground_truths)
    if n == 0:
        raise ValueError("Cannot bootstrap an empty dataset.")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be >= 1.")

    rng = np.random.RandomState(seed)
    num_classes = len(pathologies)
    class_order = np.arange(num_classes)

    auroc_boot = {name: [] for name in pathologies}
    f1_boot = {name: [] for name in pathologies}
    auroc_macro_boot = []
    f1_macro_boot = []
    accuracy_boot = []

    print(f"\nRunning {n_bootstrap:,} bootstrap resamples for 95% CIs...")
    for _ in tqdm(range(n_bootstrap), desc="Bootstrap"):
        indices = rng.randint(0, n, size=n)
        boot_gt = all_ground_truths[indices]
        boot_probs = all_probs[indices]

        per_class_aurocs = []
        if is_multilabel:
            boot_preds = (boot_probs >= 0.5).astype(np.int64)
            macro_f1 = f1_score(
                boot_gt.astype(np.int64), boot_preds, average="macro", zero_division=0
            )
            accuracy_value = float(np.mean(np.all(boot_preds == boot_gt, axis=1)))
            for class_id, name in enumerate(pathologies):
                binary_gt = boot_gt[:, class_id].astype(np.int64)
                auc_value = _safe_auroc(binary_gt, boot_probs[:, class_id])
                f1_value = float(f1_score(
                    binary_gt, boot_preds[:, class_id], zero_division=0
                ))
                auroc_boot[name].append(auc_value)
                f1_boot[name].append(f1_value)
                if not np.isnan(auc_value):
                    per_class_aurocs.append(auc_value)
        else:
            boot_gt = boot_gt.astype(np.int64).reshape(-1)
            boot_preds = np.argmax(boot_probs, axis=1)
            macro_f1 = f1_score(
                boot_gt, boot_preds, labels=class_order,
                average="macro", zero_division=0,
            )
            accuracy_value = accuracy_score(boot_gt, boot_preds)
            per_class_f1 = f1_score(
                boot_gt, boot_preds, labels=class_order,
                average=None, zero_division=0,
            )
            for class_id, name in enumerate(pathologies):
                binary_gt = (boot_gt == class_id).astype(np.int64)
                auc_value = _safe_auroc(binary_gt, boot_probs[:, class_id])
                f1_value = float(per_class_f1[class_id])
                auroc_boot[name].append(auc_value)
                f1_boot[name].append(f1_value)
                if not np.isnan(auc_value):
                    per_class_aurocs.append(auc_value)

        auroc_macro_boot.append(
            float(np.mean(per_class_aurocs)) if per_class_aurocs else float("nan")
        )
        f1_macro_boot.append(float(macro_f1))
        accuracy_boot.append(float(accuracy_value))

    lo = (alpha / 2.0) * 100.0
    hi = (1.0 - alpha / 2.0) * 100.0

    def _ci(values):
        array = np.asarray(values, dtype=float)
        array = array[~np.isnan(array)]
        if array.size == 0:
            return [float("nan"), float("nan")]
        return [float(np.percentile(array, lo)), float(np.percentile(array, hi))]

    result = {
        "auroc_macro": _ci(auroc_macro_boot),
        "f1_macro": _ci(f1_macro_boot),
        "accuracy": _ci(accuracy_boot),
        "auroc_per_class": {name: _ci(auroc_boot[name]) for name in pathologies},
        "f1_per_class": {name: _ci(f1_boot[name]) for name in pathologies},
    }
    print(
        f"  AUROC macro 95% CI: [{result['auroc_macro'][0]:.4f}, "
        f"{result['auroc_macro'][1]:.4f}]"
    )
    print(
        f"  F1 macro 95% CI:    [{result['f1_macro'][0]:.4f}, "
        f"{result['f1_macro'][1]:.4f}]"
    )
    return result