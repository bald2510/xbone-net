"""Leakage-free OOD evaluation for CTCH feature archives.

Protocol:
  * fit density scores on CTCH train only;
  * calibrate deployment thresholds on CTCH validation ID only;
  * evaluate once on CTCH test versus the requested OOD scenario.

Every feature archive is provenance-checked and must come from the same source
experiment, checkpoint, resolved configuration, and seed.  The canonical
proposed model remains the default source; ablation runners must pass their
source explicitly.
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

from src.utils.analysis import (
    SOURCE_EXPERIMENT,
    ZEROSHOT_BIOMEDCLIP_EXPERIMENT,
    load_feature_archive,
    sha256_file,
)
from src.utils.ood import (
    COSINE_CENTROIDS_SCORE_DEFINITION,
    MAHALANOBIS_SCORE_DEFINITION,
    OOD_PROTOCOL_VERSION,
    OODDetector,
    bootstrap_ood_metrics,
    calibrate_ood_threshold,
    evaluate_ood,
)


FUSED_FEATURE_KEY = "fused_embeddings"
RAW_FUSED_FEATURE_KEY = "fused_embeddings_raw"
METHOD_FEATURE_KEYS = {
    "cosine_centroids": FUSED_FEATURE_KEY,
    "mahalanobis_centroid": RAW_FUSED_FEATURE_KEY,
    "knn": FUSED_FEATURE_KEY,
    "entropy": "logits",
}
SCENARIO_FEATURES = {
    "semantic_ood": "fused_embeddings",
    "domain_ood_btxrd": "fused_embeddings",
    "report_mismatch_cross_class": "fused_embeddings",
    "report_mismatch_same_class": "fused_embeddings",
}
SUPPORTED_METHODS = (
    "cosine_centroids",
    "mahalanobis_centroid",
    "knn",
    "entropy",
)
METHOD_CHOICES = SUPPORTED_METHODS


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
    expected_source_experiment: str = SOURCE_EXPERIMENT,
) -> None:
    if provenance.get("source_experiment") != expected_source_experiment:
        raise ValueError(
            "OOD archive source mismatch: expected "
            f"{expected_source_experiment!r}, got "
            f"{provenance.get('source_experiment')!r}."
        )
    if int(provenance.get("seed", -1)) != int(expected_seed):
        raise ValueError(
            f"Archive seed {provenance.get('seed')} does not match --seed {expected_seed}."
        )
    if provenance.get("scenario") != expected_scenario:
        raise ValueError(
            f"Expected scenario {expected_scenario!r}, got "
            f"{provenance.get('scenario')!r}."
        )
    required = {"labels", "logits", "image_id", FUSED_FEATURE_KEY}
    missing = required - arrays.keys()
    if missing:
        raise ValueError(f"Feature archive is missing {sorted(missing)}.")
    count = len(arrays["labels"])
    for key, value in arrays.items():
        if value.ndim > 0 and len(value) != count:
            raise ValueError(
                f"Archive field {key!r} has {len(value)} rows, expected {count}."
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

    if expected_scenario == "btxrd_test":
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
    normalized_detector: OODDetector,
    raw_detector: OODDetector,
    arrays: dict[str, np.ndarray],
    knn_k: int,
) -> np.ndarray:
    if method == "cosine_centroids":
        return normalized_detector.score_cosine_centroids(
            arrays[FUSED_FEATURE_KEY]
        )
    if method == "mahalanobis_centroid":
        return raw_detector.score_mahalanobis_centroid(
            arrays[RAW_FUSED_FEATURE_KEY]
        )
    if method == "knn":
        return normalized_detector.score_knn(
            arrays[FUSED_FEATURE_KEY], k=knn_k
        )
    if method == "entropy":
        return normalized_detector.score_entropy(arrays["logits"])
    raise ValueError(f"Unsupported OOD method: {method}")


def run_protocol(
    train: dict[str, np.ndarray],
    calibration: dict[str, np.ndarray],
    id_test: dict[str, np.ndarray],
    ood: dict[str, np.ndarray],
    methods: list[str],
    target_id_fpr: float,
    knn_k: int,
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
        missing = [
            key
            for key in (FUSED_FEATURE_KEY, RAW_FUSED_FEATURE_KEY)
            if key not in arrays
        ]
        if missing:
            raise ValueError(f"{name} archive is missing required features: {missing}.")
    train_labels = _labels(train["labels"])
    if np.any(train_labels < 0):
        raise ValueError("CTCH train labels must be valid ID class indices.")
    normalized_detector = OODDetector().fit(
        train[FUSED_FEATURE_KEY], train_labels
    )
    raw_detector = OODDetector().fit(
        train[RAW_FUSED_FEATURE_KEY], train_labels
    )

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
            method, normalized_detector, raw_detector, calibration, knn_k
        )
        id_scores = _score_method(
            method, normalized_detector, raw_detector, id_test, knn_k
        )
        ood_scores = _score_method(
            method, normalized_detector, raw_detector, ood, knn_k
        )
        threshold = calibrate_ood_threshold(
            calibration_scores, target_id_fpr=target_id_fpr
        )
        metrics = evaluate_ood(
            id_scores,
            ood_scores,
        )
        metrics["ci_95"] = bootstrap_ood_metrics(
            id_scores,
            ood_scores,
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
        metrics["score_input"] = METHOD_FEATURE_KEYS[method]
        if method == "cosine_centroids":
            metrics["score_definition"] = COSINE_CENTROIDS_SCORE_DEFINITION
        if method == "mahalanobis_centroid":
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
                )
                for method in methods
            }
        if subgroup_results:
            results["subgroups"] = subgroup_results
    return results, score_archive


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-experiment",
        default=SOURCE_EXPERIMENT,
        help=(
            "Experiment recorded in every feature archive. The canonical "
            "proposed experiment is used when omitted."
        ),
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--scenario", choices=tuple(SCENARIO_FEATURES), required=True)
    parser.add_argument("--train-embeddings", type=Path, required=True)
    parser.add_argument("--calibration-embeddings", type=Path, required=True)
    parser.add_argument("--id-test-embeddings", type=Path, required=True)
    parser.add_argument("--ood-embeddings", type=Path, required=True)
    parser.add_argument(
        "--methods", nargs="+", choices=METHOD_CHOICES, default=list(SUPPORTED_METHODS)
    )
    parser.add_argument(
        "--primary-methods",
        nargs="+",
        choices=METHOD_CHOICES,
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
    source_experiment = str(args.source_experiment).strip("/")
    if not (
        source_experiment == SOURCE_EXPERIMENT
        or source_experiment == ZEROSHOT_BIOMEDCLIP_EXPERIMENT
        or source_experiment.startswith("ctch/ablation_study/")
    ):
        raise ValueError(
            "--source-experiment must be the canonical CTCH model, the "
            "BioMedCLIP zero-shot baseline, or an experiment below "
            "'ctch/ablation_study/'."
        )
    loaded = [
        load_feature_archive(
            path,
            expected_source_experiment=source_experiment,
        )
        for path in archive_paths.values()
    ]
    arrays = [item[0] for item in loaded]
    provenances = [item[1] for item in loaded]
    for archive_arrays, provenance, expected in zip(
        arrays, provenances, expected_scenarios
    ):
        _validate_archive(
            archive_arrays,
            provenance,
            args.seed,
            expected,
            source_experiment,
        )
    checkpoint_sha256 = _check_shared_checkpoint(provenances)

    semantic_ood_incomplete = False
    semantic_ood_coverage: dict[str, Any] = {}
    if args.scenario == "semantic_ood":
        coverage = provenances[-1].get("coverage", {})
        semantic_ood_coverage = dict(coverage)
        semantic_ood_incomplete = int(coverage.get("missing_rows", 0)) > 0
        if semantic_ood_incomplete and not args.allow_incomplete_ood:
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

    results, score_archive = run_protocol(
        train=train,
        calibration=calibration,
        id_test=id_test,
        ood=ood,
        methods=list(args.methods),
        target_id_fpr=args.target_id_fpr,
        knn_k=args.knn_k,
        n_bootstrap=args.n_bootstrap,
        bootstrap_seed=args.bootstrap_seed,
        paired=paired,
    )
    primary_methods = (
        list(args.primary_methods)
        if args.primary_methods is not None
        else list(args.methods)
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
        "type": (
            "locked_ctch_ood_evaluation"
            if source_experiment == SOURCE_EXPERIMENT
            else "ctch_representation_baseline_ood_evaluation"
        ),
        "ood_protocol_version": OOD_PROTOCOL_VERSION,
        "cosine_centroids_score_definition": (
            COSINE_CENTROIDS_SCORE_DEFINITION
        ),
        "mahalanobis_score_definition": MAHALANOBIS_SCORE_DEFINITION,
        "source_experiment": source_experiment,
        "seed": int(args.seed),
        "analysis_status": (
            "exploratory_incomplete_ood"
            if semantic_ood_incomplete
            else "complete"
        ),
        "ood_coverage": semantic_ood_coverage,
        "checkpoint_sha256": checkpoint_sha256,
        "feature_archive_sha256": {
            role: sha256_file(path)
            for role, path in archive_paths.items()
        },
        "scenario": args.scenario,
        "feature_key": FUSED_FEATURE_KEY,
        "feature_keys_by_method": {
            method: METHOD_FEATURE_KEYS[method]
            for method in args.methods
        },
        "paired_by_image_id": paired,
        "protocol": {
            "fit": "ctch_train",
            "threshold_calibration": "ctch_val_id_only",
            "id_test": "ctch_test",
            "ood_test": expected_scenarios[-1],
            "target_id_fpr": args.target_id_fpr,
            "knn_k": args.knn_k,
            "n_bootstrap": args.n_bootstrap,
            "score_methods": list(args.methods),
            "evaluation_metrics": [
                "auroc_ood",
                "aupr_out",
                "fpr_at_95tpr",
            ],
            "fpr95_definition": (
                "ood_accepted_as_id_at_95_percent_id_tpr"
            ),
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
            f"{method:22s} AUROC-OOD={metrics['auroc_ood']:.4f} "
            f"AUPR-Out={metrics['aupr_out']:.4f} "
            f"FPR@95%TPR={metrics['fpr_at_95tpr']:.4f}"
        )
    print(f"Saved: {destination}")


if __name__ == "__main__":
    main()
