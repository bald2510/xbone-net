"""Run the compact CTCH ablation OOD study without training.

The default preset contains only locally available, high-value ablations:
high-resolution input handling, Phase-1/2 training, fusion, and classifier
design.  For every experiment and seed, the runner exports the same
``fused_embeddings`` archives used by the canonical protocol and evaluates
Semantic OOD, FracAtlas, and BTXRD with cosine-centroid,
Mahalanobis-centroid, kNN, and entropy scores.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

from src.utils.analysis import SOURCE_EXPERIMENT, sha256_file
from src.utils.ood import OOD_PROTOCOL_VERSION


OOD_CONFIG = ROOT / "configs" / "analysis" / "ctch" / "ood.yaml"
SUMMARY_ROOT = ROOT / "results" / "summary" / "ablation_ood"

SMALL_ABLATION_PRESET = OrderedDict(
    [
        (SOURCE_EXPERIMENT, "XBone-Net"),
        (
            "ctch/ablation_study/architecture/preprocess/xbone_nohighres",
            "Không high-resolution",
        ),
        (
            "ctch/ablation_study/architecture/preprocess/xbone_mean_pooling",
            "Mean pooling",
        ),
        (
            "ctch/ablation_study/architecture/phase/phase2_only",
            "Chỉ huấn luyện pha 2",
        ),
        (
            "ctch/ablation_study/architecture/fusion/concat",
            "Nối đặc trưng",
        ),
        (
            "ctch/ablation_study/architecture/fusion/text_to_image",
            "Văn bản truy vấn ảnh",
        ),
        (
            "ctch/ablation_study/architecture/classifier/linear",
            "Đầu tuyến tính",
        ),
        (
            "ctch/ablation_study/architecture/classifier/no_class_weight",
            "Không trọng số lớp",
        ),
    ]
)

SCENARIO_ARCHIVES = OrderedDict(
    [
        ("semantic_ood", "ctch_ood"),
        ("domain_ood", "fracatlas_test"),
        ("domain_ood_btxrd", "btxrd_test"),
    ]
)

SCENARIO_NAMES = {
    "semantic_ood": "Semantic OOD",
    "domain_ood": "FracAtlas",
    "domain_ood_btxrd": "BTXRD",
}

METRIC_KEYS = ("auroc_ood", "aupr_out", "fpr_at_95tpr")


def _run(command: list[str], tag: str, *, dry_run: bool) -> None:
    print(f"\n[{tag}] {' '.join(command)}")
    if dry_run:
        return
    environment = dict(os.environ)
    environment.setdefault("PYTHONUTF8", "1")
    result = subprocess.run(command, cwd=ROOT, env=environment)
    if result.returncode != 0:
        raise RuntimeError(f"{tag} failed with exit code {result.returncode}.")


def _checkpoint_path(experiment: str, seed: int) -> Path:
    return (
        ROOT
        / "checkpoints"
        / experiment
        / f"seed_{int(seed)}"
        / "best_phase2.pth"
    )


def _metrics_path(experiment: str, seed: int) -> Path:
    return ROOT / "results" / experiment / f"seed_{int(seed)}" / "metrics.json"


def _analysis_root(experiment: str, seed: int | None = None) -> Path:
    root = ROOT / "results" / experiment
    if seed is not None:
        root = root / f"seed_{int(seed)}"
    return root / "analysis"


def _feature_path(experiment: str, seed: int, archive: str) -> Path:
    return _analysis_root(experiment, seed) / "features" / f"{archive}.npz"


def _ood_output(experiment: str, seed: int, scenario: str) -> Path:
    return _analysis_root(experiment, seed) / "ood" / scenario


def _validate_experiment_name(experiment: str) -> str:
    experiment = str(experiment).replace("\\", "/").strip("/")
    if not (
        experiment == SOURCE_EXPERIMENT
        or experiment.startswith("ctch/ablation_study/")
    ):
        raise ValueError(
            "Experiments must be the canonical CTCH proposed model or live "
            "below 'ctch/ablation_study/'."
        )
    return experiment


def _preflight(
    experiments: list[str],
    seeds: list[int],
    *,
    skip_missing: bool,
) -> list[str]:
    runnable: list[str] = []
    for experiment in experiments:
        missing = []
        for seed in seeds:
            if not _metrics_path(experiment, seed).is_file():
                missing.append(f"seed {seed} metrics")
            if not _checkpoint_path(experiment, seed).is_file():
                missing.append(f"seed {seed} checkpoint")
        if missing:
            message = f"{experiment}: missing {', '.join(missing)}"
            if skip_missing:
                print(f"[SKIP] {message}")
                continue
            raise FileNotFoundError(message)
        runnable.append(experiment)
    if not runnable:
        raise RuntimeError("No experiment has the required metrics and checkpoints.")
    return runnable


def _ood_result_is_current(
    path: Path,
    *,
    experiment: str,
    seed: int,
    feature_paths: dict[str, Path],
) -> tuple[bool, str]:
    if not path.is_file():
        return False, "missing"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False, "unreadable"
    if payload.get("source_experiment") != experiment:
        return False, "source experiment changed"
    if int(payload.get("seed", -1)) != int(seed):
        return False, "seed changed"
    if int(payload.get("ood_protocol_version", -1)) != OOD_PROTOCOL_VERSION:
        return False, "OOD protocol changed"
    expected_hashes = {
        role: sha256_file(source)
        for role, source in feature_paths.items()
    }
    if payload.get("feature_archive_sha256") != expected_hashes:
        return False, "feature archive changed"
    return True, "current"


def _mean_std(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "n": int(len(array)),
    }


def _classification_f1(experiment: str, seeds: list[int]) -> dict[str, Any]:
    values = []
    for seed in seeds:
        payload = json.loads(
            _metrics_path(experiment, seed).read_text(encoding="utf-8")
        )
        value = payload.get("metrics", {}).get("f1_macro")
        if value is not None:
            values.append(float(value))
    return _mean_std(values) if values else {"mean": None, "std": None, "n": 0}


def aggregate_existing(
    experiments: list[str],
    seeds: list[int],
    scenarios: list[str],
    *,
    output_dir: Path,
) -> dict[str, Any]:
    """Aggregate available ablation OOD results and write long-form CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate: dict[str, Any] = {
        "preset": "small",
        "experiments_requested": experiments,
        "seeds_requested": seeds,
        "scenarios_requested": scenarios,
        "experiments": {},
    }
    csv_rows: list[dict[str, Any]] = []

    for experiment in experiments:
        display_name = SMALL_ABLATION_PRESET.get(
            experiment,
            experiment.split("/")[-1],
        )
        experiment_result = {
            "display_name": display_name,
            "classification_f1_macro": _classification_f1(experiment, seeds),
            "ood": {},
        }
        for scenario in scenarios:
            payloads = []
            for seed in seeds:
                path = _ood_output(
                    experiment,
                    seed,
                    scenario,
                ) / "ood_metrics.json"
                if path.is_file():
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    if (
                        payload.get("source_experiment") == experiment
                        and int(payload.get("seed", -1)) == seed
                    ):
                        payloads.append(payload)
            if not payloads:
                continue

            methods = [
                method
                for method in payloads[0].get("results", {})
                if method != "subgroups"
            ]
            scenario_result: dict[str, Any] = {}
            for method in methods:
                method_result = {}
                for metric in METRIC_KEYS:
                    values = [
                        float(payload["results"][method][metric])
                        for payload in payloads
                        if method in payload.get("results", {})
                        and metric in payload["results"][method]
                    ]
                    if values:
                        method_result[metric] = _mean_std(values)
                scenario_result[method] = method_result
                row = {
                    "experiment": experiment,
                    "model": display_name,
                    "scenario": scenario,
                    "scenario_name": SCENARIO_NAMES[scenario],
                    "method": method,
                    "analysis_status": "|".join(
                        sorted(
                            {
                                str(payload.get("analysis_status", "complete"))
                                for payload in payloads
                            }
                        )
                    ),
                    "seeds": "|".join(
                        str(payload["seed"]) for payload in payloads
                    ),
                    "classification_f1_macro_mean": (
                        experiment_result["classification_f1_macro"]["mean"]
                    ),
                    "classification_f1_macro_std": (
                        experiment_result["classification_f1_macro"]["std"]
                    ),
                }
                for metric in METRIC_KEYS:
                    statistics = method_result.get(metric, {})
                    row[f"{metric}_mean"] = statistics.get("mean")
                    row[f"{metric}_std"] = statistics.get("std")
                csv_rows.append(row)
            experiment_result["ood"][scenario] = {
                "seeds": sorted(int(payload["seed"]) for payload in payloads),
                "methods": scenario_result,
            }
        aggregate["experiments"][experiment] = experiment_result

    json_output = output_dir / "aggregated_results.json"
    json_output.write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    csv_output = output_dir / "ablation_ood_summary.csv"
    if csv_rows:
        with csv_output.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
            writer.writeheader()
            writer.writerows(csv_rows)
    else:
        csv_output.write_text("", encoding="utf-8")

    print("\nABLATION OOD SUMMARY")
    print(
        f"{'Model':<31} {'Scenario':<14} {'Method':<24} "
        f"{'AUROC':>8} {'AUPR-Out':>9} {'FPR95':>8}"
    )
    print("-" * 103)
    for row in csv_rows:
        print(
            f"{row['model']:<31} {row['scenario_name']:<14} "
            f"{row['method']:<24} "
            f"{row['auroc_ood_mean']:>8.4f} "
            f"{row['aupr_out_mean']:>9.4f} "
            f"{row['fpr_at_95tpr_mean']:>8.4f}"
        )
    print(f"Aggregated JSON: {json_output}")
    print(f"Summary CSV   : {csv_output}")
    return aggregate


def main() -> None:
    ood_cfg = OmegaConf.load(OOD_CONFIG)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=list(SMALL_ABLATION_PRESET),
        help="Explicit experiment paths; defaults to the compact preset.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=tuple(SCENARIO_ARCHIVES),
        default=list(SCENARIO_ARCHIVES),
    )
    parser.add_argument("--feature-only", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument(
        "--allow-incomplete-ood",
        action="store_true",
        help=(
            "Exploratory only: omit unreadable/missing CTCH semantic-OOD "
            "samples and record the resulting coverage in each archive."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=int(ood_cfg.n_bootstrap),
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SUMMARY_ROOT,
    )
    parser.add_argument("--list-experiments", action="store_true")
    args = parser.parse_args()

    if args.list_experiments:
        for experiment, label in SMALL_ABLATION_PRESET.items():
            print(f"{experiment:<78} {label}")
        return

    experiments = [
        _validate_experiment_name(experiment)
        for experiment in args.experiments
    ]
    experiments = list(dict.fromkeys(experiments))
    seeds = list(dict.fromkeys(int(seed) for seed in args.seeds))
    experiments = _preflight(
        experiments,
        seeds,
        skip_missing=args.skip_missing,
    )

    if args.aggregate_only:
        aggregate_existing(
            experiments,
            seeds,
            args.scenarios,
            output_dir=args.output_dir.resolve(),
        )
        return

    print("+----------------------------------------------------------------+")
    print("|  Compact CTCH ablation OOD study                              |")
    print("+----------------------------------------------------------------+")
    print(f"  Experiments: {len(experiments)}")
    print(f"  Seeds      : {seeds}")
    print(f"  Scenarios  : {args.scenarios}")
    print("  Training   : disabled")

    failures = []
    for experiment in experiments:
        for seed in seeds:
            try:
                required_archives = {
                    "ctch_train",
                    "ctch_val",
                    "ctch_test",
                    *(
                        SCENARIO_ARCHIVES[scenario]
                        for scenario in args.scenarios
                    ),
                }
                feature_command = [
                    sys.executable,
                    str(ROOT / "tools" / "export_experiment_analysis_features.py"),
                    "--experiment",
                    experiment,
                    "--seed",
                    str(seed),
                    "--scenarios",
                    *sorted(required_archives),
                    "--batch-size",
                    str(args.batch_size),
                    "--num-workers",
                    str(args.num_workers),
                    "--device",
                    args.device,
                ]
                if args.overwrite:
                    feature_command.append("--overwrite")
                if args.allow_incomplete_ood:
                    feature_command.append("--allow-incomplete-ood")
                _run(
                    feature_command,
                    f"FEATURES {experiment} seed={seed}",
                    dry_run=args.dry_run,
                )
                if args.feature_only:
                    continue

                for scenario in args.scenarios:
                    archive = SCENARIO_ARCHIVES[scenario]
                    feature_paths = {
                        "fit": _feature_path(
                            experiment,
                            seed,
                            "ctch_train",
                        ),
                        "calibration": _feature_path(
                            experiment,
                            seed,
                            "ctch_val",
                        ),
                        "id_test": _feature_path(
                            experiment,
                            seed,
                            "ctch_test",
                        ),
                        "ood_test": _feature_path(
                            experiment,
                            seed,
                            archive,
                        ),
                    }
                    output_dir = _ood_output(experiment, seed, scenario)
                    if not args.dry_run and not args.overwrite:
                        current, reason = _ood_result_is_current(
                            output_dir / "ood_metrics.json",
                            experiment=experiment,
                            seed=seed,
                            feature_paths=feature_paths,
                        )
                        if current:
                            print(
                                f"[Resume] {experiment} seed={seed} "
                                f"{scenario}: verified result exists."
                            )
                            continue
                        if reason != "missing":
                            print(
                                f"[Refresh] {experiment} seed={seed} "
                                f"{scenario}: {reason}."
                            )

                    command = [
                        sys.executable,
                        str(ROOT / "evaluate_ood.py"),
                        "--source-experiment",
                        experiment,
                        "--seed",
                        str(seed),
                        "--scenario",
                        scenario,
                        "--train-embeddings",
                        str(feature_paths["fit"]),
                        "--calibration-embeddings",
                        str(feature_paths["calibration"]),
                        "--id-test-embeddings",
                        str(feature_paths["id_test"]),
                        "--ood-embeddings",
                        str(feature_paths["ood_test"]),
                        "--methods",
                        *[str(value) for value in ood_cfg.methods],
                        "--primary-methods",
                        *[str(value) for value in ood_cfg.methods],
                        "--scenario-role",
                        "primary",
                        "--target-id-fpr",
                        str(float(ood_cfg.target_id_fpr)),
                        "--knn-k",
                        str(int(ood_cfg.knn_k)),
                        "--n-bootstrap",
                        str(args.n_bootstrap),
                        "--bootstrap-seed",
                        str(int(ood_cfg.bootstrap_seed) + seed),
                        "--no-paired-by-image-id",
                        "--output-dir",
                        str(output_dir),
                    ]
                    if args.allow_incomplete_ood:
                        command.append("--allow-incomplete-ood")
                    _run(
                        command,
                        f"OOD {experiment} seed={seed} {scenario}",
                        dry_run=args.dry_run,
                    )
            except Exception as error:
                failures.append(
                    {
                        "experiment": experiment,
                        "seed": seed,
                        "error": str(error),
                    }
                )
                print(f"[FAIL] {experiment} seed={seed}: {error}")
                if not args.continue_on_error:
                    raise

    if not args.dry_run and not args.feature_only:
        aggregate_existing(
            experiments,
            seeds,
            args.scenarios,
            output_dir=args.output_dir.resolve(),
        )
    if failures:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        failure_path = args.output_dir / "failures.json"
        failure_path.write_text(
            json.dumps(failures, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        raise SystemExit(
            f"{len(failures)} experiment/seed run(s) failed; see {failure_path}"
        )


if __name__ == "__main__":
    main()
