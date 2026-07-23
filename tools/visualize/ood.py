"""Explain OOD decisions from locked CTCH score archives.

This script is intentionally read-only with respect to the protocol: it never
fits a detector, splits ID data, or chooses a threshold. All score arrays and
ID-validation thresholds must already have been produced by ``evaluate_ood.py``.
"""

from __future__ import annotations

import argparse
import csv
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
    parser.add_argument(
        "--max-heatmap-samples",
        type=int,
        default=60,
        help="Maximum number of near-threshold ID/OOD cases in the evidence heatmap.",
    )
    return parser.parse_args()


def _robust_margin(
    scores: np.ndarray,
    threshold: float,
    calibration: np.ndarray,
) -> np.ndarray:
    """Return threshold-relative margins on a robust, comparable scale."""
    calibration = np.asarray(calibration, dtype=float)
    median = float(np.median(calibration))
    mad = float(np.median(np.abs(calibration - median)))
    scale = max(1.4826 * mad, float(np.std(calibration)), 1e-8)
    return (np.asarray(scores, dtype=float) - threshold) / scale


def _render_decision_heatmap(
    score_sets: dict[str, dict[str, np.ndarray | float]],
    methods: list[str],
    id_image_ids: np.ndarray,
    ood_image_ids: np.ndarray,
    output: Path,
    max_samples: int,
) -> Path:
    """Visualize each detector's signed evidence relative to its threshold."""
    if max_samples < 2:
        raise ValueError("--max-heatmap-samples must be at least 2.")
    margin_rows = []
    for method in methods:
        values = score_sets[method]
        margin_rows.append(
            np.concatenate(
                [
                    _robust_margin(
                        values["id"], values["threshold"], values["calibration"]
                    ),
                    _robust_margin(
                        values["ood"], values["threshold"], values["calibration"]
                    ),
                ]
            )
        )
    margins = np.vstack(margin_rows)
    sample_types = np.asarray(
        ["ID"] * len(id_image_ids) + ["OOD"] * len(ood_image_ids)
    )
    image_ids = np.concatenate([id_image_ids, ood_image_ids]).astype(str)
    uncertainty = np.min(np.abs(margins), axis=0)
    per_group = max(1, max_samples // 2)
    selected = np.concatenate(
        [
            np.flatnonzero(sample_types == group)[
                np.argsort(uncertainty[sample_types == group])[:per_group]
            ]
            for group in ("ID", "OOD")
        ]
    )
    selected = selected[np.argsort(sample_types[selected], kind="stable")]
    selected_margins = margins[:, selected]
    limit = max(1.0, float(np.nanpercentile(np.abs(selected_margins), 95)))

    width = max(12.0, min(24.0, 0.28 * len(selected) + 5.0))
    figure, axis = plt.subplots(figsize=(width, 2.0 + 0.75 * len(methods)))
    image = axis.imshow(
        selected_margins,
        aspect="auto",
        cmap="coolwarm",
        vmin=-limit,
        vmax=limit,
        interpolation="nearest",
    )
    labels = [
        f"{sample_types[index]}:{Path(image_ids[index]).stem[:12]}"
        for index in selected
    ]
    axis.set_xticks(np.arange(len(selected)), labels=labels, rotation=90)
    axis.set_yticks(np.arange(len(methods)), labels=methods)
    axis.set_xlabel("Near-threshold cases (ground-truth protocol role : image id)")
    axis.set_ylabel("OOD detector")
    axis.set_title(
        "Threshold-relative OOD evidence\n"
        "blue = accepted as ID, red = flagged as OOD, zero = calibrated threshold"
    )
    colorbar = figure.colorbar(image, ax=axis, fraction=0.025, pad=0.02)
    colorbar.set_label("Robust signed distance from threshold")
    figure.tight_layout()
    heatmap_path = output.with_name(f"{output.stem}_decision_heatmap{output.suffix}")
    figure.savefig(heatmap_path, dpi=220, bbox_inches="tight")
    plt.close(figure)

    csv_path = output.with_name(f"{output.stem}_decision_evidence.csv")
    fields = ["sample_type", "image_id", "consensus_ood_fraction"]
    for method in methods:
        fields.extend([f"{method}_score", f"{method}_threshold", f"{method}_margin"])
    rows = []
    for index in selected:
        row: dict[str, object] = {
            "sample_type": sample_types[index],
            "image_id": image_ids[index],
            "consensus_ood_fraction": float(np.mean(margins[:, index] >= 0.0)),
        }
        local_index = index if sample_types[index] == "ID" else index - len(id_image_ids)
        for method_index, method in enumerate(methods):
            values = score_sets[method]
            score = values["id"][local_index] if sample_types[index] == "ID" else values["ood"][local_index]
            row[f"{method}_score"] = float(score)
            row[f"{method}_threshold"] = float(values["threshold"])
            row[f"{method}_margin"] = float(margins[method_index, index])
        rows.append(row)
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return heatmap_path


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
        id_image_ids = np.asarray(
            archive.get("id_image_id", np.arange(len(score_sets[methods[0]]["id"]))),
            dtype=str,
        )
        ood_image_ids = np.asarray(
            archive.get("ood_image_id", np.arange(len(score_sets[methods[0]]["ood"]))),
            dtype=str,
        )

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
    heatmap_path = _render_decision_heatmap(
        score_sets,
        methods,
        id_image_ids,
        ood_image_ids,
        args.output,
        args.max_heatmap_samples,
    )
    print(f"Saved: {heatmap_path}")


if __name__ == "__main__":
    main()
