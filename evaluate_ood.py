"""Leakage-free OOD evaluation for CTCH proposed-model feature archives.

Protocol:
  * fit density scores on CTCH train only;
  * calibrate deployment thresholds on CTCH validation ID only;
  * evaluate once on CTCH test versus the requested OOD scenario.

Every feature archive is provenance-checked and must come from the same locked
``ctch/proposed/ours_xbone_net`` checkpoint seed.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.datasets.fracatlas import FRACATLAS_IMAGE_RESOLVER_VERSION
from src.utils.analysis import (
    SOURCE_EXPERIMENT,
    load_feature_archive,
    sha256_file,
)
from src.utils.ood import (
    MAHALANOBIS_SCORE_DEFINITION,
    OOD_PROTOCOL_VERSION,
    OODDetector,
    bootstrap_ood_metrics,
    calibrate_ood_threshold,
    evaluate_ood,
)


SCENARIO_FEATURES = {
    "semantic_ood": "fused_embeddings",
    "domain_ood": "fused_embeddings",
    "domain_ood_btxrd": "fused_embeddings",
    "report_mismatch_cross_class": "fused_embeddings",
    "report_mismatch_same_class": "fused_embeddings",
}
SUPPORTED_METHODS = (
    "mahalanobis",
    "knn",
    "msp",
    "entropy",
    "energy",
    "max_logit",
)


def _labels(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim == 2:
        values = values.argmax(axis=1)
    return values.astype(np.int64).reshape(-1)


def _validate_archive(
    arrays: dict[str, np.ndarray],
    provenance: dict[str, Any],
    expected_seed: int,
    expected_scenario: str,
) -> None:
    if provenance.get("source_experiment") != SOURCE_EXPERIMENT:
        raise ValueError("OOD archives must come from the locked CTCH proposed model.")
    if int(provenance.get("seed", -1)) != int(expected_seed):
        raise ValueError(
            f"Archive seed {provenance.get('seed')} does not match --seed {expected_seed}."
        )
    if provenance.get("scenario") != expected_scenario:
        raise ValueError(
            f"Expected scenario {expected_scenario!r}, got "
            f"{provenance.get('scenario')!r}."
        )
    required = {"labels", "logits", "image_id"}
    missing = required - arrays.keys()
    if missing:
        raise ValueError(f"Feature archive is missing {sorted(missing)}.")
    count = len(arrays["labels"])
    for key, value in arrays.items():
        if value.ndim > 0 and len(value) != count:
            raise ValueError(
                f"Archive field {key!r} has {len(value)} rows, expected {count}."
            )

    if expected_scenario == "fracatlas_test":
        coverage = provenance.get("coverage", {})
        if (
            int(provenance.get("fracatlas_image_resolver_version", -1))
            != FRACATLAS_IMAGE_RESOLVER_VERSION
            or int(coverage.get("rows", -1)) != count
            or int(coverage.get("resolved_images", -1)) != count
            or int(coverage.get("missing_images", -1)) != 0
            or int(coverage.get("missing_reports", -1)) != 0
            or not bool(coverage.get("decode_validation", False))
            or int(coverage.get("decode_failure_count", -1)) != 0
        ):
            raise ValueError(
                "FracAtlas domain-OOD archive lacks complete fail-closed image/report "
                "coverage. Re-export fracatlas_test features with the current loader."
            )
    if expected_scenario == "btxrd_test":
        coverage = provenance.get("coverage", {})
        if (
            provenance.get("ood_dataset") != "BTXRD"
            or provenance.get("ood_split") != "test"
            or int(coverage.get("rows", -1)) != count
            or int(coverage.get("resolved_images", -1)) != count
            or int(coverage.get("missing_images", -1)) != 0
            or int(coverage.get("missing_xray_reports", -1)) != 0
            or int(coverage.get("missing_clinical_reports", -1)) != 0
        ):
            raise ValueError(
                "BTXRD domain-OOD archive lacks complete fail-closed image/report "
                "coverage. Re-export btxrd_test features with the current loader."
            )

    if expected_scenario in {"fracatlas_test", "btxrd_test"}:
        visual = arrays.get("visual_global_embeddings")
        if visual is None:
            raise ValueError(
                f"{expected_scenario} domain-OOD archive has no "
                "visual_global_embeddings."
            )
        max_pairwise_from_first = float(
            np.linalg.norm(
                np.asarray(visual, dtype=np.float64) - visual[:1], axis=1
            ).max()
        )
        if max_pairwise_from_first <= 1e-6:
            raise ValueError(
                f"{expected_scenario} visual-global features are effectively constant. "
                "This usually indicates repeated fallback images; refusing to "
                "report domain-OOD metrics."
            )


def _check_shared_checkpoint(provenances: list[dict[str, Any]]) -> str:
    checksums = {item.get("checkpoint_sha256") for item in provenances}
    if None in checksums or len(checksums) != 1:
        raise ValueError("All feature archives must share one checkpoint SHA-256.")
    config_checksums = {item.get("config_sha256") for item in provenances}
    if None in config_checksums or len(config_checksums) != 1:
        raise ValueError("All feature archives must share one resolved config SHA-256.")
    return str(next(iter(checksums)))


def _subset_rows(arrays: dict[str, np.ndarray], indices: np.ndarray) -> dict[str, np.ndarray]:
    count = len(arrays["labels"])
    return {
        key: value[indices] if value.ndim > 0 and len(value) == count else value
        for key, value in arrays.items()
    }


def align_paired_id(
    id_arrays: dict[str, np.ndarray],
    ood_arrays: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Align native ID rows to report-mismatch recipient IDs."""
    native_ids = id_arrays["image_id"].astype(str)
    if len(np.unique(native_ids)) != len(native_ids):
        raise ValueError("CTCH ID test archive contains duplicate image_id values.")
    lookup = {image_id: index for index, image_id in enumerate(native_ids)}
    missing = [
        image_id
        for image_id in ood_arrays["image_id"].astype(str)
        if image_id not in lookup
    ]
    if missing:
        raise ValueError(f"Mismatch recipients are absent from ID test: {missing[:5]}")
    indices = np.asarray(
        [lookup[image_id] for image_id in ood_arrays["image_id"].astype(str)],
        dtype=np.int64,
    )
    aligned = _subset_rows(id_arrays, indices)
    if not np.array_equal(
        aligned["image_id"].astype(str), ood_arrays["image_id"].astype(str)
    ):
        raise RuntimeError("Failed to align paired report-mismatch samples.")
    return aligned


def _score_method(
    method: str,
    detector: OODDetector,
    arrays: dict[str, np.ndarray],
    feature_key: str,
    knn_k: int,
    temperature: float,
) -> np.ndarray:
    if method == "mahalanobis":
        return detector.score_mahalanobis(arrays[feature_key])
    if method == "knn":
        return detector.score_knn(arrays[feature_key], k=knn_k)
    logits = arrays["logits"]
    if method == "msp":
        return detector.score_msp(logits)
    if method == "entropy":
        return detector.score_entropy(logits)
    if method == "energy":
        return detector.score_energy(logits, temperature=temperature)
    if method == "max_logit":
        return detector.score_max_logit(logits)
    raise ValueError(f"Unsupported OOD method: {method}")


def run_protocol(
    train: dict[str, np.ndarray],
    calibration: dict[str, np.ndarray],
    id_test: dict[str, np.ndarray],
    ood: dict[str, np.ndarray],
    methods: list[str],
    feature_key: str,
    target_id_fpr: float,
    knn_k: int,
    temperature: float,
    n_bootstrap: int,
    bootstrap_seed: int,
    paired: bool,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    for name, arrays in {
        "train": train,
        "calibration": calibration,
        "id_test": id_test,
        "ood": ood,
    }.items():
        if feature_key not in arrays:
            raise ValueError(f"{name} archive has no feature {feature_key!r}.")
    train_labels = _labels(train["labels"])
    if np.any(train_labels < 0):
        raise ValueError("CTCH train labels must be valid ID class indices.")
    detector = OODDetector().fit(train[feature_key], train_labels)

    id_labels = _labels(id_test["labels"])
    id_correct = id_test["logits"].argmax(axis=1) == id_labels
    results: dict[str, Any] = {}
    score_archive: dict[str, np.ndarray] = {
        "id_image_id": id_test["image_id"].astype(str),
        "ood_image_id": ood["image_id"].astype(str),
        "id_correct": id_correct.astype(np.int8),
    }
    id_groups = id_test.get("patient_id")
    if id_groups is not None and np.all(id_groups.astype(str) == ""):
        id_groups = None
    ood_groups = ood.get("patient_id")
    if ood_groups is not None and np.all(ood_groups.astype(str) == ""):
        ood_groups = None

    for method_index, method in enumerate(methods):
        calibration_scores = _score_method(
            method, detector, calibration, feature_key, knn_k, temperature
        )
        id_scores = _score_method(
            method, detector, id_test, feature_key, knn_k, temperature
        )
        ood_scores = _score_method(
            method, detector, ood, feature_key, knn_k, temperature
        )
        threshold = calibrate_ood_threshold(
            calibration_scores, target_id_fpr=target_id_fpr
        )
        metrics = evaluate_ood(
            id_scores,
            ood_scores,
            calibrated_threshold=threshold,
            id_correct=id_correct,
        )
        metrics["ci_95"] = bootstrap_ood_metrics(
            id_scores,
            ood_scores,
            calibrated_threshold=threshold,
            id_correct=id_correct,
            n_bootstrap=n_bootstrap,
            seed=bootstrap_seed + method_index,
            paired=paired,
            id_groups=None if paired else id_groups,
            ood_groups=None if paired else ood_groups,
        )
        metrics["bootstrap_units"] = (
            {"id": "paired_image", "ood": "paired_image"}
            if paired
            else {
                "id": "patient_cluster" if id_groups is not None else "sample",
                "ood": "patient_cluster" if ood_groups is not None else "sample",
            }
        )
        metrics["calibration_id_count"] = int(len(calibration_scores))
        metrics["target_calibration_id_fpr"] = float(target_id_fpr)
        metrics["score_direction"] = "higher_is_more_ood"
        metrics["score_input"] = (
            feature_key
            if method in {"mahalanobis", "knn"}
            else "fused_classifier_logits"
        )
        if method == "mahalanobis":
            metrics["score_definition"] = MAHALANOBIS_SCORE_DEFINITION
        results[method] = metrics
        score_archive[f"{method}_calibration"] = calibration_scores
        score_archive[f"{method}_id"] = id_scores
        score_archive[f"{method}_ood"] = ood_scores
        score_archive[f"{method}_threshold"] = np.asarray(threshold)

    groups = ood.get("group")
    if groups is not None and len(np.unique(groups.astype(str))) > 1:
        subgroup_results: dict[str, Any] = {}
        for group in sorted(np.unique(groups.astype(str))):
            mask = groups.astype(str) == group
            if int(mask.sum()) < 2:
                continue
            subgroup_results[group] = {
                method: evaluate_ood(
                    score_archive[f"{method}_id"],
                    score_archive[f"{method}_ood"][mask],
                    calibrated_threshold=float(
                        score_archive[f"{method}_threshold"].item()
                    ),
                    id_correct=id_correct,
                )
                for method in methods
            }
        if subgroup_results:
            results["subgroups"] = subgroup_results
    return results, score_archive


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--scenario", choices=tuple(SCENARIO_FEATURES), required=True)
    parser.add_argument("--train-embeddings", type=Path, required=True)
    parser.add_argument("--calibration-embeddings", type=Path, required=True)
    parser.add_argument("--id-test-embeddings", type=Path, required=True)
    parser.add_argument("--ood-embeddings", type=Path, required=True)
    parser.add_argument("--feature-key", default=None)
    parser.add_argument(
        "--methods", nargs="+", choices=SUPPORTED_METHODS, default=list(SUPPORTED_METHODS)
    )
    parser.add_argument(
        "--primary-methods",
        nargs="+",
        choices=SUPPORTED_METHODS,
        default=None,
        help="Methods designated primary by the locked scenario config.",
    )
    parser.add_argument(
        "--scenario-role",
        choices=("primary", "secondary"),
        default="primary",
    )
    parser.add_argument(
        "--paired-by-image-id",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--target-id-fpr", type=float, default=0.05)
    parser.add_argument("--knn-k", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=3107)
    parser.add_argument(
        "--allow-incomplete-ood",
        action="store_true",
        help="Accept an explicitly exploratory CTCH-OOD archive with coverage below 100 percent.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    expected_scenarios = (
        "ctch_train",
        "ctch_val",
        "ctch_test",
        {
            "semantic_ood": "ctch_ood",
            "domain_ood": "fracatlas_test",
            "domain_ood_btxrd": "btxrd_test",
            "report_mismatch_cross_class": "report_mismatch_cross_class",
            "report_mismatch_same_class": "report_mismatch_same_class",
        }[args.scenario],
    )
    archive_paths = {
        "fit": args.train_embeddings,
        "calibration": args.calibration_embeddings,
        "id_test": args.id_test_embeddings,
        "ood_test": args.ood_embeddings,
    }
    loaded = [load_feature_archive(path) for path in archive_paths.values()]
    arrays = [item[0] for item in loaded]
    provenances = [item[1] for item in loaded]
    for archive_arrays, provenance, expected in zip(
        arrays, provenances, expected_scenarios
    ):
        _validate_archive(archive_arrays, provenance, args.seed, expected)
    checkpoint_sha256 = _check_shared_checkpoint(provenances)

    if args.scenario == "semantic_ood":
        coverage = provenances[-1].get("coverage", {})
        incomplete = int(coverage.get("missing_rows", 0)) > 0
        if incomplete and not args.allow_incomplete_ood:
            raise RuntimeError(
                "Refusing to report semantic-OOD metrics from an incomplete archive. "
                "Use --allow-incomplete-ood only for exploratory analysis."
            )

    train, calibration, id_test, ood = arrays
    paired = (
        args.scenario.startswith("report_mismatch_")
        if args.paired_by_image_id is None
        else bool(args.paired_by_image_id)
    )
    if paired:
        id_test = align_paired_id(id_test, ood)
        if np.any(ood["image_id"].astype(str) == ood["report_source_id"].astype(str)):
            raise ValueError("Report-mismatch archive contains fixed report assignments.")

    feature_key = args.feature_key or SCENARIO_FEATURES[args.scenario]
    results, score_archive = run_protocol(
        train=train,
        calibration=calibration,
        id_test=id_test,
        ood=ood,
        methods=list(args.methods),
        feature_key=feature_key,
        target_id_fpr=args.target_id_fpr,
        knn_k=args.knn_k,
        temperature=args.temperature,
        n_bootstrap=args.n_bootstrap,
        bootstrap_seed=args.bootstrap_seed,
        paired=paired,
    )
    primary_methods = (
        list(args.primary_methods)
        if args.primary_methods is not None
        else (
            ["mahalanobis", "knn"]
            if args.scenario.startswith("domain_ood")
            else list(args.methods)
        )
    )
    unknown_primary = sorted(set(primary_methods) - set(args.methods))
    if unknown_primary:
        raise ValueError(
            f"Primary methods were not evaluated: {unknown_primary}."
        )
    scenario_is_primary = args.scenario_role == "primary"
    for method in args.methods:
        results[method]["primary_analysis"] = (
            scenario_is_primary and method in primary_methods
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir / "ood_scores.npz", **score_archive)
    output = {
        "type": "locked_ctch_ood_evaluation",
        "ood_protocol_version": OOD_PROTOCOL_VERSION,
        "mahalanobis_score_definition": MAHALANOBIS_SCORE_DEFINITION,
        "source_experiment": SOURCE_EXPERIMENT,
        "seed": int(args.seed),
        "checkpoint_sha256": checkpoint_sha256,
        "feature_archive_sha256": {
            role: sha256_file(path)
            for role, path in archive_paths.items()
        },
        "scenario": args.scenario,
        "feature_key": feature_key,
        "paired_by_image_id": paired,
        "protocol": {
            "fit": "ctch_train",
            "threshold_calibration": "ctch_val_id_only",
            "id_test": "ctch_test",
            "ood_test": expected_scenarios[-1],
            "target_id_fpr": args.target_id_fpr,
            "knn_k": args.knn_k,
            "temperature": args.temperature,
            "n_bootstrap": args.n_bootstrap,
            "primary_methods": primary_methods,
            "scenario_role": args.scenario_role,
        },
        "counts": {
            "train": int(len(train["labels"])),
            "calibration": int(len(calibration["labels"])),
            "id_test": int(len(id_test["labels"])),
            "ood_test": int(len(ood["labels"])),
        },
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "results": results,
    }
    destination = args.output_dir / "ood_metrics.json"
    destination.write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    for method in args.methods:
        metrics = results[method]
        print(
            f"{method:12s} AUROC={metrics['auroc']:.4f} "
            f"AUPR-Out={metrics['aupr_out']:.4f} "
            f"FPR@95={metrics['fpr_at_95tpr']:.4f}"
        )
    print(f"Saved: {destination}")


if __name__ == "__main__":
    main()
