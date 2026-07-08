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
from sklearn.metrics import (confusion_matrix, f1_score, accuracy_score,
                             precision_score, recall_score, roc_auc_score,
                             hamming_loss)

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
    """Compute evaluation metrics for multi-class (single-label) classification.

    Formula:
        Specificity_i = TN_i / (TN_i + FP_i) (via one-vs-rest confusion matrix)

    Args:
        all_probs: Predicted class probabilities of shape (N, C).
        all_gt: Integer ground-truth class indices of shape (N,).
        class_names: Ordered list of human-readable class names.

    Returns:
        Dictionary containing auroc_macro, f1_macro, f1_per_class, accuracy,
        sensitivity, specificity, precision, and confusion_matrix.
    """
    # --- Compute predicted class indices and overall accuracy ---
    num_classes = len(class_names)
    all_preds = np.argmax(all_probs, axis=1)

    acc = float(accuracy_score(all_gt, all_preds))

    # --- Compute macro and per-class metrics ---
    macro_f1 = float(f1_score(all_gt, all_preds, average="macro", zero_division=0))
    macro_prec = float(precision_score(all_gt, all_preds, average="macro", zero_division=0))
    macro_rec = float(recall_score(all_gt, all_preds, average="macro", zero_division=0))

    per_class_f1_arr = f1_score(all_gt, all_preds, average=None, zero_division=0)
    per_class_prec_arr = precision_score(all_gt, all_preds, average=None, zero_division=0)
    per_class_rec_arr = recall_score(all_gt, all_preds, average=None, zero_division=0)

    f1_per_class = {}
    for i, name in enumerate(class_names):
        if i < len(per_class_f1_arr):
            f1_per_class[name] = float(per_class_f1_arr[i])

    # --- Compute macro AUROC ---
    try:
        # Ensure probs columns match the actual number of classes in gt
        n_unique_classes = len(np.unique(all_gt))
        probs_for_auc = all_probs
        if probs_for_auc.shape[1] > n_unique_classes:
            print(f"  [AUROC Warning] Probs has {probs_for_auc.shape[1]} columns but gt has "
                  f"{n_unique_classes} unique classes. Trimming to {n_unique_classes} columns.")
            probs_for_auc = probs_for_auc[:, :n_unique_classes]
        # Re-normalize so rows sum to 1.0 (required by roc_auc_score)
        row_sums = probs_for_auc.sum(axis=1, keepdims=True)
        probs_for_auc = probs_for_auc / (row_sums + 1e-8)
        macro_auroc = float(roc_auc_score(
            all_gt, probs_for_auc, average="macro", multi_class="ovr",
        ))
    except ValueError as e:
        print(f"  [AUROC Error] {e} — probs shape: {all_probs.shape}, "
              f"gt unique: {np.unique(all_gt)}, gt shape: {all_gt.shape}")
        macro_auroc = float("nan")

    # --- Compute confusion matrix & per-class specificity ---
    cm = confusion_matrix(all_gt, all_preds)

    specificities = []
    for i in range(num_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificities.append(spec)
    macro_spec = float(np.mean(specificities))

    # --- Print reports ---
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
    """Compute bootstrap confidence intervals for AUROC and F1 metrics.

    Uses paired bootstrap resampling to extract 95% confidence intervals:
        CI = [percentile(alpha / 2), percentile(1 - alpha / 2)]

    Args:
        all_probs: Predicted probabilities of shape (N, C).
        all_ground_truths: Ground-truth labels of shape (N, C) or (N,).
        pathologies: Ordered list of pathology / class names.
        is_multilabel: Whether the classification task is multi-label.
        n_bootstrap: Number of bootstrap iterations (default: 10,000).
        seed: Random seed for reproducible sampling.
        alpha: Significance level for CI percentile bounds (default: 0.05).

    Returns:
        Dictionary mapping metric names ('auroc_macro', 'f1_macro', etc.)
        to [lower, upper] percentile confidence interval lists.
    """
    rng = np.random.RandomState(seed)
    n = len(all_ground_truths)

    auroc_boot: dict[str, list[float]] = {p: [] for p in pathologies}
    f1_boot: dict[str, list[float]] = {p: [] for p in pathologies}
    auroc_macro_boot: list[float] = []
    f1_macro_boot: list[float] = []

    # --- Bootstrap resampling loop ---
    print(f"\nRunning {n_bootstrap:,} bootstrap resamples for 95% CIs...")
    for _ in tqdm(range(n_bootstrap), desc="Bootstrap"):
        idx = rng.randint(0, n, size=n)
        boot_gt = all_ground_truths[idx]
        boot_probs = all_probs[idx]

        per_class_aurocs = []
        per_class_f1s = []

        for ci, path in enumerate(pathologies):
            if is_multilabel:
                gt_col = boot_gt[:, ci]
            else:
                gt_col = (boot_gt == ci).astype(int)
            prob_col = boot_probs[:, ci]

            auc_val = _safe_auroc(gt_col, prob_col)
            f1_val = _safe_f1(gt_col, prob_col)

            auroc_boot[path].append(auc_val)
            f1_boot[path].append(f1_val)

            if not np.isnan(auc_val):
                per_class_aurocs.append(auc_val)
            per_class_f1s.append(f1_val)

        auroc_macro_boot.append(float(np.mean(per_class_aurocs)) if per_class_aurocs else float("nan"))
        f1_macro_boot.append(float(np.mean(per_class_f1s)))

    # --- Percentile bound calculation ---
    lo = (alpha / 2) * 100
    hi = (1 - alpha / 2) * 100

    def _ci(values: list[float]) -> list[float]:
        """Extract [lower, upper] percentile CI, ignoring NaN entries."""
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


