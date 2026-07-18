"""Visualize score archives from the locked CTCH OOD protocol.

This script is intentionally read-only with respect to the protocol: it never
fits a detector, splits ID data, or chooses a threshold. All score arrays and
ID-validation thresholds must already have been produced by ``evaluate_ood.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument(
        "--metrics",
        type=Path,
        default=None,
        help="Defaults to ood_metrics.json beside --scores.",
    )
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    metrics_path = args.metrics or args.scores.with_name("ood_metrics.json")
    if not args.scores.is_file():
        raise FileNotFoundError(args.scores)
    if not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if metrics.get("type") != "locked_ctch_ood_evaluation":
        raise ValueError("Metrics were not produced by the locked CTCH OOD evaluator.")

    with np.load(args.scores, allow_pickle=False) as archive:
        available = sorted(
            key.removesuffix("_id")
            for key in archive.files
            if key.endswith("_id") and f"{key.removesuffix('_id')}_ood" in archive.files
        )
        methods = args.methods or available
        if not methods:
            raise ValueError("Score archive contains no plottable OOD methods.")
        unknown = sorted(set(methods) - set(available))
        if unknown:
            raise ValueError(f"Score archive has no methods {unknown}; available={available}")
        score_sets = {
            method: {
                "id": archive[f"{method}_id"],
                "ood": archive[f"{method}_ood"],
                "calibration": archive[f"{method}_calibration"],
                "threshold": float(archive[f"{method}_threshold"].item()),
            }
            for method in methods
        }

    columns = min(3, len(methods))
    rows = int(np.ceil(len(methods) / columns))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(6.0 * columns, 4.7 * rows),
        squeeze=False,
        facecolor="white",
    )
    scenario = str(metrics.get("scenario", "unknown"))
    for axis, method in zip(axes.flat, methods):
        values = score_sets[method]
        method_metrics = metrics["results"][method]
        axis.hist(
            values["id"],
            bins=35,
            alpha=0.67,
            label="CTCH ID test",
            density=True,
            edgecolor="black",
            linewidth=0.35,
        )
        axis.hist(
            values["ood"],
            bins=35,
            alpha=0.60,
            label=scenario,
            density=True,
            edgecolor="black",
            linewidth=0.35,
        )
        axis.axvline(
            values["threshold"],
            color="#111111",
            linestyle="--",
            linewidth=1.4,
            label="CTCH-val ID threshold",
        )
        role = "primary" if method_metrics.get("primary_analysis", True) else "secondary"
        axis.set_title(
            f"{method} ({role})\n"
            f"AUROC={method_metrics['auroc']:.3f}, "
            f"FPR95={method_metrics['fpr_at_95tpr']:.3f}"
        )
        axis.set_xlabel("OOD score (higher = more OOD)")
        axis.set_ylabel("Density")
        axis.grid(alpha=0.25, linestyle="--")
        axis.legend(fontsize=8)
    for axis in axes.flat[len(methods):]:
        axis.axis("off")

    figure.suptitle(
        f"CTCH proposed OOD separation: {scenario}\n"
        "fit=CTCH train, threshold=CTCH validation ID, evaluation=CTCH test/OOD",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=220, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
