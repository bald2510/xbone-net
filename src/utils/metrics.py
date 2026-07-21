"""
Evaluation Metrics & Bootstrap Resampling for XBone-Net.
===============================================================================
Computes standard classification metrics for XBone-Net evaluation:
  - Multi-class (single-label) diagnosis: AUROC, F1, precision, recall, specificity
  - Multi-label pathology detection: Macro AUROC, F1, Hamming loss, per-class metrics
  - Paired bootstrap resampling: 95% confidence interval estimation (10,000 resamples)

Per-class metrics use scikit-learn for both per-class and aggregate values.
"""

import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    confusion_matrix, f1_score, accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, roc_auc_score, average_precision_score,
    hamming_loss,
)

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


def _safe_auprc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute binary AUPRC safely for bootstrap samples."""
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(average_precision_score(labels, scores))


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


def multiclass_calibration_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> dict[str, float]:
    """Compute top-label ECE, adaptive ECE, NLL, and multiclass Brier score."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if probabilities.ndim != 2 or len(probabilities) != len(labels):
        raise ValueError("Calibration inputs must have shapes [N,C] and [N].")
    if len(labels) == 0 or n_bins < 1:
        raise ValueError("Calibration metrics require samples and positive n_bins.")
    row_sums = probabilities.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("Every probability row must have a positive sum.")
    probabilities = probabilities / row_sums
    if np.any(labels < 0) or np.any(labels >= probabilities.shape[1]):
        raise ValueError("Calibration labels are outside the probability columns.")

    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = (predictions == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for index in range(n_bins):
        lower, upper = edges[index], edges[index + 1]
        mask = (confidence > lower) & (confidence <= upper)
        if index == 0:
            mask |= confidence == 0.0
        if np.any(mask):
            ece += float(mask.mean()) * abs(
                float(correct[mask].mean()) - float(confidence[mask].mean())
            )

    adaptive_ece = 0.0
    for indices in np.array_split(np.argsort(confidence), min(n_bins, len(labels))):
        if len(indices):
            adaptive_ece += (len(indices) / len(labels)) * abs(
                float(correct[indices].mean())
                - float(confidence[indices].mean())
            )

    clipped_true = np.clip(
        probabilities[np.arange(len(labels)), labels], 1e-12, 1.0
    )
    one_hot = np.eye(probabilities.shape[1], dtype=np.float64)[labels]
    return {
        "ece_15": float(ece),
        "adaptive_ece_15": float(adaptive_ece),
        "nll": float(-np.log(clipped_true).mean()),
        "brier_score": float(np.square(probabilities - one_hot).sum(axis=1).mean()),
    }


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
    calibration = multiclass_calibration_metrics(probs, all_gt, n_bins=15)

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
    print(f"ECE (15 equal-width bins)     : {calibration['ece_15']:.4f}")
    print(f"Adaptive ECE (15 bins)        : {calibration['adaptive_ece_15']:.4f}")
    print(f"Negative Log-Likelihood       : {calibration['nll']:.4f}")
    print(f"Multiclass Brier Score        : {calibration['brier_score']:.4f}")
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
        **calibration,
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

        acc = accuracy_score(gt_labels, preds)
        prec = precision_score(gt_labels, preds, zero_division=0)
        rec = recall_score(gt_labels, preds, zero_division=0)
        f1 = f1_score(gt_labels, preds, zero_division=0)
        auroc = roc_auc_score(gt_labels, probs)

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
    auprc_macro_boot = []
    f1_macro_boot = []
    accuracy_boot = []
    balanced_accuracy_boot = []
    ece_boot = []
    adaptive_ece_boot = []
    nll_boot = []
    brier_boot = []

    print(f"\nRunning {n_bootstrap:,} bootstrap resamples for 95% CIs...")
    for _ in tqdm(range(n_bootstrap), desc="Bootstrap"):
        indices = rng.randint(0, n, size=n)
        boot_gt = all_ground_truths[indices]
        boot_probs = all_probs[indices]

        per_class_aurocs = []
        per_class_auprcs = []
        if is_multilabel:
            boot_preds = (boot_probs >= 0.5).astype(np.int64)
            macro_f1 = f1_score(
                boot_gt.astype(np.int64), boot_preds, average="macro", zero_division=0
            )
            accuracy_value = float(np.mean(np.all(boot_preds == boot_gt, axis=1)))
            for class_id, name in enumerate(pathologies):
                binary_gt = boot_gt[:, class_id].astype(np.int64)
                auc_value = _safe_auroc(binary_gt, boot_probs[:, class_id])
                auprc_value = _safe_auprc(binary_gt, boot_probs[:, class_id])
                f1_value = float(f1_score(
                    binary_gt, boot_preds[:, class_id], zero_division=0
                ))
                auroc_boot[name].append(auc_value)
                f1_boot[name].append(f1_value)
                if not np.isnan(auc_value):
                    per_class_aurocs.append(auc_value)
                if not np.isnan(auprc_value):
                    per_class_auprcs.append(auprc_value)
        else:
            boot_gt = boot_gt.astype(np.int64).reshape(-1)
            row_sums = boot_probs.sum(axis=1, keepdims=True)
            boot_probs = boot_probs / np.clip(row_sums, 1e-12, None)
            boot_preds = np.argmax(boot_probs, axis=1)
            macro_f1 = f1_score(
                boot_gt, boot_preds, labels=class_order,
                average="macro", zero_division=0,
            )
            accuracy_value = accuracy_score(boot_gt, boot_preds)
            balanced_accuracy_boot.append(
                float(balanced_accuracy_score(boot_gt, boot_preds))
            )
            per_class_f1 = f1_score(
                boot_gt, boot_preds, labels=class_order,
                average=None, zero_division=0,
            )
            for class_id, name in enumerate(pathologies):
                binary_gt = (boot_gt == class_id).astype(np.int64)
                auc_value = _safe_auroc(binary_gt, boot_probs[:, class_id])
                auprc_value = _safe_auprc(binary_gt, boot_probs[:, class_id])
                f1_value = float(per_class_f1[class_id])
                auroc_boot[name].append(auc_value)
                f1_boot[name].append(f1_value)
                if not np.isnan(auc_value):
                    per_class_aurocs.append(auc_value)
                if not np.isnan(auprc_value):
                    per_class_auprcs.append(auprc_value)
            calibration = multiclass_calibration_metrics(
                boot_probs, boot_gt, n_bins=15
            )
            ece_boot.append(calibration["ece_15"])
            adaptive_ece_boot.append(calibration["adaptive_ece_15"])
            nll_boot.append(calibration["nll"])
            brier_boot.append(calibration["brier_score"])

        auroc_macro_boot.append(
            float(np.mean(per_class_aurocs)) if per_class_aurocs else float("nan")
        )
        auprc_macro_boot.append(
            float(np.mean(per_class_auprcs)) if per_class_auprcs else float("nan")
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
        "auprc_macro": _ci(auprc_macro_boot),
        "f1_macro": _ci(f1_macro_boot),
        "accuracy": _ci(accuracy_boot),
        "auroc_per_class": {name: _ci(auroc_boot[name]) for name in pathologies},
        "f1_per_class": {name: _ci(f1_boot[name]) for name in pathologies},
    }
    if not is_multilabel:
        result.update({
            "balanced_accuracy": _ci(balanced_accuracy_boot),
            "ece_15": _ci(ece_boot),
            "adaptive_ece_15": _ci(adaptive_ece_boot),
            "nll": _ci(nll_boot),
            "brier_score": _ci(brier_boot),
        })
    print(
        f"  AUROC macro 95% CI: [{result['auroc_macro'][0]:.4f}, "
        f"{result['auroc_macro'][1]:.4f}]"
    )
    print(
        f"  F1 macro 95% CI:    [{result['f1_macro'][0]:.4f}, "
        f"{result['f1_macro'][1]:.4f}]"
    )
    return result
