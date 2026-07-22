"""Compare seed-matched OOD, representation, and attention across proposed models.

This diagnostic runner is deliberately separate from the locked canonical
analysis.  Every model is rebuilt from the resolved config stored in its own
``metrics.json`` and every cached feature archive is tied to checkpoint and
config SHA-256 fingerprints.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluate_ood import align_paired_id, run_protocol
from src.utils.analysis import (
    build_analysis_loader,
    collect_feature_batches,
    load_proposed_experiment_model,
)
from src.utils.explainability import representation_quality_metrics
from tools.export_analysis_features import build_scenario_dataset


VERSION_TO_EXPERIMENT = {
    "v2": "ctch/proposed/ours_xbone_net_v2",
    "v3": "ctch/proposed/ours_xbone_net_v3",
    "v4": "ctch/proposed/ours_xbone_net_v4",
    "v5": "ctch/proposed/ours_xbone_net_v5",
    "v6": "ctch/proposed/proposed_v6",
}
FEATURE_SCENARIOS = (
    "ctch_train",
    "ctch_val",
    "ctch_test",
    "ctch_ood",
    "fracatlas_test",
    "btxrd_test",
    "report_mismatch_cross_class",
    "report_mismatch_same_class",
)
OOD_SCENARIOS = {
    "semantic_ood": "ctch_ood",
    "domain_ood_fracatlas": "fracatlas_test",
    "domain_ood_btxrd": "btxrd_test",
    "report_mismatch_cross_class": "report_mismatch_cross_class",
    "report_mismatch_same_class": "report_mismatch_same_class",
}
REPRESENTATIONS = (
    "fused_embeddings",
    "visual_global_embeddings",
    "visual_local_summary_embeddings",
    "text_global_embeddings",
    "image_from_text_embeddings",
    "text_from_image_embeddings",
)
METHODS = ("mahalanobis", "knn", "msp", "entropy", "energy", "max_logit")


def _json_default(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}.")


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as archive:
        return {key: archive[key] for key in archive.files}


def _cache_is_current(path: Path, metadata_path: Path, loaded, scenario: str) -> bool:
    if not path.is_file() or not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        metadata.get("checkpoint_sha256") == loaded.checkpoint_sha256
        and metadata.get("config_sha256") == loaded.provenance["config_sha256"]
        and metadata.get("scenario") == scenario
        and int(metadata.get("seed", -1)) == int(loaded.provenance["seed"])
    )


def _export_or_load(
    loaded,
    version_root: Path,
    scenario: str,
    batch_size: int,
    num_workers: int,
    overwrite: bool,
) -> dict[str, np.ndarray]:
    feature_root = version_root / "features"
    feature_root.mkdir(parents=True, exist_ok=True)
    path = feature_root / f"{scenario}.npz"
    metadata_path = feature_root / f"{scenario}.json"
    if not overwrite and _cache_is_current(path, metadata_path, loaded, scenario):
        print(f"  [resume] {scenario}")
        return _load_npz(path)

    print(f"  [export] {scenario}")
    dataset, scenario_metadata = build_scenario_dataset(
        scenario,
        loaded,
        allow_incomplete_ood=False,
        mismatch_seed=2025,
    )
    loader = build_analysis_loader(
        dataset,
        tokenizer=loaded.model.backbone.tokenizer_obj,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    arrays = collect_feature_batches(
        loaded.model,
        loader,
        device=loaded.device,
        report_type=str(loaded.cfg.params.phase2.p2_report_type),
    )
    np.savez_compressed(path, **arrays)
    metadata = {
        **loaded.provenance,
        "scenario": scenario,
        "sample_count": int(len(arrays["labels"])),
        "feature_dimensions": {
            key: list(value.shape) for key, value in arrays.items()
        },
        **scenario_metadata,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    return arrays


def _distribution_summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {}
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "p05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
    }


def _attention_overview(test: dict[str, np.ndarray]) -> dict[str, Any]:
    keys = [
        key for key in test
        if key.startswith("attention_") or key == "fusion_gate"
    ]
    overview = {key: _distribution_summary(test[key]) for key in keys}
    if "fusion_gate" in test:
        gate = np.asarray(test["fusion_gate"], dtype=np.float64).reshape(-1)
        logits = np.asarray(test["logits"], dtype=np.float64)
        labels = np.asarray(test["labels"], dtype=np.int64).reshape(-1)
        predictions = logits.argmax(axis=1)
        correct = predictions == labels
        probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        confidence = probabilities.max(axis=1)
        overview["fusion_gate_diagnostics"] = {
            "mean_correct": float(gate[correct].mean()) if correct.any() else None,
            "mean_incorrect": float(gate[~correct].mean()) if (~correct).any() else None,
            "correlation_with_confidence": float(np.corrcoef(gate, confidence)[0, 1]),
            "correlation_with_correctness": float(
                np.corrcoef(gate, correct.astype(np.float64))[0, 1]
            ),
        }
    return overview


def _representation_overview(
    train: dict[str, np.ndarray],
    test: dict[str, np.ndarray],
    seed: int,
) -> dict[str, Any]:
    output = {}
    for key in REPRESENTATIONS:
        output[key] = representation_quality_metrics(
            train[key],
            train["labels"],
            test[key],
            test["labels"],
            seed=seed,
        )
    return output


def _ood_overview(
    features: dict[str, dict[str, np.ndarray]],
    n_bootstrap: int,
) -> dict[str, Any]:
    output = {}
    for index, (scenario, archive_name) in enumerate(OOD_SCENARIOS.items()):
        id_test = features["ctch_test"]
        ood = features[archive_name]
        paired = scenario.startswith("report_mismatch_")
        if paired:
            id_test = align_paired_id(id_test, ood)
        print(f"  [ood] {scenario}")
        results, _ = run_protocol(
            train=features["ctch_train"],
            calibration=features["ctch_val"],
            id_test=id_test,
            ood=ood,
            methods=list(METHODS),
            feature_key="fused_embeddings",
            target_id_fpr=0.05,
            knn_k=5,
            temperature=1.0,
            n_bootstrap=n_bootstrap,
            bootstrap_seed=3107 + 100 * index,
            paired=paired,
        )
        output[scenario] = results
    return output


def _write_csv_tables(summary: dict[str, Any], output_root: Path) -> None:
    representation_path = output_root / "representation_overview.csv"
    with representation_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "version", "representation", "silhouette_cosine",
            "nearest_centroid_balanced_accuracy", "knn_balanced_accuracy",
            "positive_centroid_margin_fraction", "true_centroid_margin_mean",
        ])
        for version, payload in summary["versions"].items():
            for name, metrics in payload["representation"].items():
                writer.writerow([
                    version,
                    name,
                    metrics["silhouette_cosine"],
                    metrics["nearest_train_centroid_balanced_accuracy"],
                    metrics["knn_balanced_accuracy"],
                    metrics["positive_centroid_margin_fraction"],
                    metrics["true_centroid_margin_mean"],
                ])

    attention_path = output_root / "attention_overview.csv"
    with attention_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["version", "measure", "mean", "std", "p05", "median", "p95"])
        for version, payload in summary["versions"].items():
            for name, metrics in payload["attention"].items():
                if not isinstance(metrics, dict) or "mean" not in metrics:
                    continue
                writer.writerow([
                    version, name, metrics.get("mean"), metrics.get("std"),
                    metrics.get("p05"), metrics.get("median"), metrics.get("p95"),
                ])

    ood_path = output_root / "ood_overview.csv"
    with ood_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "version", "scenario", "method", "auroc", "aupr_out",
            "fpr_at_95tpr", "id_fpr_at_calibrated_threshold",
            "ood_tpr_at_calibrated_threshold",
        ])
        for version, payload in summary["versions"].items():
            for scenario, methods in payload["ood"].items():
                for method, metrics in methods.items():
                    if method == "subgroups":
                        continue
                    writer.writerow([
                        version, scenario, method, metrics["auroc"],
                        metrics["aupr_out"], metrics["fpr_at_95tpr"],
                        metrics.get("id_fpr_at_calibrated_threshold"),
                        metrics.get("ood_tpr_at_calibrated_threshold"),
                    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--versions", nargs="+", choices=tuple(VERSION_TO_EXPERIMENT),
        default=list(VERSION_TO_EXPERIMENT),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--n-bootstrap", type=int, default=200)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.n_bootstrap < 1:
        raise ValueError("--n-bootstrap must be positive.")
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        device = torch.device(args.device)

    output_root = (
        ROOT / "results" / "ctch" / "proposed" / f"comparison_seed_{args.seed}"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "type": "proposed_cross_version_posthoc_comparison",
        "seed": int(args.seed),
        "versions_requested": list(args.versions),
        "protocol": {
            "feature": "fused_embeddings",
            "ood_fit": "ctch_train",
            "ood_threshold_calibration": "ctch_val_id_only",
            "ood_id_test": "ctch_test",
            "target_id_fpr": 0.05,
            "methods": list(METHODS),
            "n_bootstrap": int(args.n_bootstrap),
            "attention_scope": "complete_ctch_test_split",
            "representation_scope": "ctch_train_to_ctch_test",
        },
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "versions": {},
    }

    for version in args.versions:
        experiment = VERSION_TO_EXPERIMENT[version]
        print(f"\n=== {version}: {experiment} ===")
        loaded = load_proposed_experiment_model(
            experiment, args.seed, device=device
        )
        version_root = output_root / version
        features = {
            scenario: _export_or_load(
                loaded,
                version_root,
                scenario,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                overwrite=args.overwrite,
            )
            for scenario in FEATURE_SCENARIOS
        }
        version_payload = {
            "experiment": experiment,
            "provenance": loaded.provenance,
            "classification_metrics": json.loads(
                Path(loaded.provenance["metrics_path"]).read_text(encoding="utf-8")
            )["metrics"],
            "representation": _representation_overview(
                features["ctch_train"], features["ctch_test"], args.seed
            ),
            "attention": _attention_overview(features["ctch_test"]),
            "ood": _ood_overview(features, args.n_bootstrap),
        }
        summary["versions"][version] = version_payload
        (version_root / "summary.json").write_text(
            json.dumps(
                version_payload, indent=2, ensure_ascii=False, default=_json_default
            ),
            encoding="utf-8",
        )
        del loaded, features
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary_path = output_root / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    _write_csv_tables(summary, output_root)
    print(f"\nComparison summary: {summary_path}")


if __name__ == "__main__":
    main()
