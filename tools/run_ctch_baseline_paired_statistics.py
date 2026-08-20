"""Run paired CTCH/BTXRD statistics for XBone-Net against baselines.

The analysis reuses the patient/seed crossed bootstrap and paired permutation
implementation in :mod:`tools.visualize.results`.  All archives are aligned by
``image_id`` and checked for identical labels and patient identifiers before a
comparison is run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.visualize import results as results_module


DEFAULT_RESULTS_ROOT = ROOT / "results"
BASELINE_NAMES = (
    ("fft_resnet50", "ResNet-50"),
    ("fft_densenet", "DenseNet-121"),
    ("fft_clip", "CLIP"),
    ("fft_medclip", "MedCLIP"),
    ("fft_pubmedclip", "PubMedCLIP"),
    ("fft_biomedclip", "BiomedCLIP"),
)
DEFAULT_METRICS = (
    "accuracy",
    "balanced_accuracy",
    "f1_macro",
    "auroc_macro",
    "auprc_macro",
)


def _experiment_specifications(
    dataset: str,
) -> tuple[tuple[str, str], tuple[tuple[str, str], ...]]:
    """Return the reference and baseline experiment paths for a dataset."""
    reference = (f"{dataset}/proposed/ours_xbone_net", "XBone-Net")
    baselines = tuple(
        (f"{dataset}/baselines/full_finetuned/{experiment}", label)
        for experiment, label in BASELINE_NAMES
    )
    return reference, baselines


def _default_output_dir(results_root: Path, dataset: str) -> Path:
    """Return the conventional output directory for a dataset."""
    return (
        results_root
        / "summary"
        / "classification"
        / f"statistics_{dataset}_xbone_vs_baselines"
    )


def _holm_adjust(p_values: np.ndarray) -> np.ndarray:
    """Return Holm-adjusted p-values in their original order."""
    values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(values)
    adjusted_sorted = np.empty_like(values)
    running_max = 0.0
    count = len(values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * float(values[index]))
        running_max = max(running_max, candidate)
        adjusted_sorted[rank] = running_max
    adjusted = np.empty_like(values)
    adjusted[order] = adjusted_sorted
    return adjusted


def _load_predictions(
    results_root: Path,
    seeds: list[int],
    *,
    dataset: str,
    specifications: tuple[tuple[str, str], ...],
) -> tuple[
    dict[str, dict[int, np.ndarray]],
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Load and align all model prediction archives."""
    probabilities: dict[str, dict[int, np.ndarray]] = {}
    common_image_ids: np.ndarray | None = None
    common_labels: np.ndarray | None = None
    common_patient_ids: np.ndarray | None = None

    for experiment, label in specifications:
        probabilities[experiment] = {}
        for seed in seeds:
            source = (
                results_root
                / experiment
                / f"seed_{seed}"
                / "analysis"
                / "features"
                / f"{dataset}_test.npz"
            )
            archive = results_module._load_aligned_prediction_archive(
                source,
                expected_image_ids=common_image_ids,
                expected_labels=common_labels,
            )
            if common_image_ids is None:
                common_image_ids = archive["image_id"]
                common_labels = archive["labels"]
                common_patient_ids = archive["patient_id"]
            elif not np.array_equal(
                archive["patient_id"].astype(str),
                np.asarray(common_patient_ids).astype(str),
            ):
                raise ValueError(f"Patient IDs differ in paired archive: {source}")

            current = archive["probabilities"]
            if current.ndim != 2 or current.shape[0] != len(archive["labels"]):
                raise ValueError(f"Invalid probability shape in {source}: {current.shape}")
            if not np.isfinite(current).all():
                raise ValueError(f"Non-finite probabilities in {source}")
            if np.any(current.sum(axis=1) <= 0.0):
                raise ValueError(f"Non-positive probability row sum in {source}")
            probabilities[experiment][seed] = current
        print(f"Aligned {label}: {len(seeds)} seeds")

    assert common_image_ids is not None
    assert common_labels is not None
    assert common_patient_ids is not None
    return probabilities, common_labels, common_image_ids, common_patient_ids


def run_analysis(
    results_root: Path,
    output_dir: Path,
    *,
    dataset: str,
    seeds: list[int],
    metrics: list[str],
    n_bootstrap: int,
    n_permutations: int,
    test_method: str,
    alpha: float,
    random_seed: int,
) -> dict[str, Path]:
    """Run paired bootstrap, permutation tests, and Holm correction."""
    reference, baselines = _experiment_specifications(dataset)
    probabilities, labels, image_ids, patient_ids = _load_predictions(
        results_root,
        seeds,
        dataset=dataset,
        specifications=(reference, *baselines),
    )
    reference_experiment, reference_label = reference
    rows: list[dict[str, object]] = []

    for index, (experiment, label) in enumerate(baselines, start=1):
        print(f"[{index}/{len(baselines)}] Paired analysis: {reference_label} vs {label}")
        comparison_rows = results_module._paired_variant_statistics(
            probabilities[reference_experiment],
            probabilities[experiment],
            labels=labels,
            patient_ids=patient_ids,
            seeds=seeds,
            metrics=metrics,
            n_bootstrap=n_bootstrap,
            n_permutations=n_permutations,
            alpha=alpha,
            random_seed=random_seed,
            test_method=test_method,
        )
        for row in comparison_rows:
            # The shared helper returns variant-reference; report the requested
            # and easier-to-read direction XBone-Net minus baseline instead.
            delta_low = -float(row["delta_ci_high"])
            delta_high = -float(row["delta_ci_low"])
            rows.append(
                {
                    "metric": row["metric"],
                    "reference": reference_label,
                    "baseline": label,
                    "reference_mean": row["reference_mean"],
                    "reference_ci_low": row["reference_ci_low"],
                    "reference_ci_high": row["reference_ci_high"],
                    "baseline_mean": row["variant_mean"],
                    "baseline_ci_low": row["variant_ci_low"],
                    "baseline_ci_high": row["variant_ci_high"],
                    "delta_xbone_minus_baseline": -float(row["delta_mean"]),
                    "delta_ci_low": delta_low,
                    "delta_ci_high": delta_high,
                    "p_raw": row["p_raw"],
                    "ci_excludes_zero": delta_low > 0.0 or delta_high < 0.0,
                    "reference_experiment": reference_experiment,
                    "baseline_experiment": experiment,
                }
            )

    frame = pd.DataFrame(rows)
    frame["p_holm"] = np.nan
    for metric in metrics:
        mask = frame["metric"] == metric
        frame.loc[mask, "p_holm"] = _holm_adjust(
            frame.loc[mask, "p_raw"].to_numpy(dtype=float)
        )
    frame["significant_holm"] = frame["p_holm"] < alpha
    higher_is_better = frame["metric"].map(
        lambda metric: bool(
            results_module.PAIRED_STATISTIC_METRICS[str(metric)]["higher_is_better"]
        )
    )
    frame["xbone_direction_favorable"] = np.where(
        higher_is_better,
        frame["delta_xbone_minus_baseline"] > 0.0,
        frame["delta_xbone_minus_baseline"] < 0.0,
    )
    frame["evidence_xbone_favorable"] = (
        frame["ci_excludes_zero"]
        & frame["significant_holm"]
        & frame["xbone_direction_favorable"]
    )
    frame["evidence_baseline_favorable"] = (
        frame["ci_excludes_zero"]
        & frame["significant_holm"]
        & ~frame["xbone_direction_favorable"]
    )
    frame["n_cases"] = len(labels)
    frame["n_patients"] = len(np.unique(patient_ids.astype(str)))
    frame["n_seeds"] = len(seeds)
    frame["seeds"] = "|".join(map(str, seeds))
    frame["n_bootstrap"] = n_bootstrap
    frame["n_permutations"] = n_permutations if test_method == "permutation" else 0

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_output = output_dir / "paired_baseline_statistics.csv"
    frame.to_csv(csv_output, index=False, encoding="utf-8-sig")

    compact_columns = [
        "metric",
        "baseline",
        "reference_mean",
        "baseline_mean",
        "delta_xbone_minus_baseline",
        "delta_ci_low",
        "delta_ci_high",
        "p_raw",
        "p_holm",
        "evidence_xbone_favorable",
        "evidence_baseline_favorable",
    ]
    compact_output = output_dir / "paired_baseline_statistics_compact.csv"
    frame[compact_columns].to_csv(
        compact_output,
        index=False,
        encoding="utf-8-sig",
    )

    protocol = {
        "analysis": f"paired_{dataset}_xbone_net_vs_full_finetuned_baselines",
        "dataset": dataset,
        "scenario": f"{dataset}_test",
        "reference": reference_experiment,
        "baselines": [experiment for experiment, _ in baselines],
        "metrics": metrics,
        "primary_metric": "f1_macro",
        "seeds": seeds,
        "n_cases": int(len(image_ids)),
        "n_patients": int(len(np.unique(patient_ids.astype(str)))),
        "bootstrap": {
            "type": "paired crossed percentile bootstrap",
            "patient_sampling": "class-stratified cluster resampling",
            "seed_sampling": "training seeds resampled with replacement",
            "same_resample_for_models": True,
            "n_resamples": n_bootstrap,
            "confidence_level": 1.0 - alpha,
        },
        "test": {
            "type": (
                "two-sided crossed seed/patient permutation"
                if test_method == "permutation"
                else "two-sided centered paired bootstrap"
            ),
            "same_patient_swap_across_seeds": test_method == "permutation",
            "seed_sampling": "training seeds resampled with replacement",
            "n_permutations": n_permutations if test_method == "permutation" else 0,
            "multiplicity_correction": "Holm within each metric across baselines",
            "alpha": alpha,
        },
        "delta_definition": "XBone-Net minus baseline",
        "random_seed": random_seed,
    }
    protocol_output = output_dir / "statistical_protocol.json"
    protocol_output.write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "csv": csv_output,
        "compact_csv": compact_output,
        "protocol": protocol_output,
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("ctch", "btxrd"), required=True)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: results/summary/classification/statistics_<dataset>_xbone_vs_baselines",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=tuple(results_module.PAIRED_STATISTIC_METRICS),
        default=list(DEFAULT_METRICS),
    )
    parser.add_argument("--n-bootstrap", type=int, default=10_000)
    parser.add_argument("--n-permutations", type=int, default=10_000)
    parser.add_argument(
        "--test-method",
        choices=("permutation", "bootstrap"),
        default="permutation",
        help="Paired significance test; permutation is exact to the current protocol but slower.",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--random-seed", type=int, default=2026)
    return parser


def main() -> None:
    """Execute the paired baseline analysis."""
    args = build_parser().parse_args()
    results_root = args.results_root.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else _default_output_dir(results_root, args.dataset)
    )
    outputs = run_analysis(
        results_root,
        output_dir,
        dataset=args.dataset,
        seeds=list(args.seeds),
        metrics=list(args.metrics),
        n_bootstrap=args.n_bootstrap,
        n_permutations=args.n_permutations,
        test_method=args.test_method,
        alpha=args.alpha,
        random_seed=args.random_seed,
    )
    for label, path in outputs.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
