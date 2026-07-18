"""Run locked multi-seed CTCH OOD and explainability analyses.

This orchestrator mirrors ``tools/run_all.py`` but is intentionally post-hoc:
it never invokes ``train.py`` and accepts no experiment override.  Every child
process is pinned to ``ctch/proposed/ours_xbone_net`` and verifies checkpoint
and feature provenance before writing results.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.analysis import (
    SOURCE_EXPERIMENT,
    SOURCE_SEEDS,
    analysis_root,
    locked_checkpoint_path,
)


OOD_CONFIG = ROOT / "configs" / "analysis" / "ctch" / "ood.yaml"
EXPLAIN_CONFIG = ROOT / "configs" / "analysis" / "ctch" / "explainability.yaml"

LOCKED_REPRESENTATION_SPACES = [
    "fused_embeddings",
    "visual_global_embeddings",
    "visual_local_summary_embeddings",
    "text_global_embeddings",
    "image_from_text_embeddings",
    "text_from_image_embeddings",
]


def _validate_locked_configs(ood_cfg, explain_cfg) -> None:
    """Reject config fields the implementation cannot faithfully execute."""
    expected_ood_splits = {
        "fit_split": "ctch_train",
        "calibration_split": "ctch_val",
        "id_test_split": "ctch_test",
    }
    for key, expected in expected_ood_splits.items():
        if str(ood_cfg.get(key)) != expected:
            raise ValueError(f"{key} is locked to {expected!r}.")

    expected_explain = {
        "target": "predicted_class",
        "sampling": "stratified_by_ground_truth_class",
        "integrated_gradients_baseline": (
            "global_image_embedding_repeated_as_local_tokens"
        ),
        "clinical_token_intervention_baseline": (
            "clinical_cls_embedding_repeated_over_removed_tokens"
        ),
    }
    for key, expected in expected_explain.items():
        if str(explain_cfg.get(key)) != expected:
            raise ValueError(
                f"Unsupported explainability setting {key}={explain_cfg.get(key)!r}; "
                f"the locked implementation requires {expected!r}."
            )
    if list(explain_cfg.representation_spaces) != LOCKED_REPRESENTATION_SPACES:
        raise ValueError(
            "representation_spaces must match the representations exported by "
            "the locked explainability implementation."
        )


def _run(command: list[str], tag: str) -> None:
    print(f"\n[{tag}] {' '.join(command)}")
    environment = dict(os.environ)
    environment.setdefault("PYTHONUTF8", "1")
    result = subprocess.run(command, cwd=ROOT, env=environment)
    if result.returncode != 0:
        raise RuntimeError(f"{tag} failed with exit code {result.returncode}.")


def _verified_json(path: Path, seed: int, expected_type: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Expected analysis output was not created: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("source_experiment") != SOURCE_EXPERIMENT:
        raise ValueError(f"Unexpected source experiment in {path}")
    if int(payload.get("seed", -1)) != int(seed):
        raise ValueError(f"Unexpected seed in {path}")
    if payload.get("type") != expected_type:
        raise ValueError(f"Unexpected result type in {path}")
    return payload


def _flatten_numeric(value: Any, prefix: str = "") -> dict[str, float]:
    output: dict[str, float] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "ci_95" or key == "subgroups":
                continue
            name = f"{prefix}.{key}" if prefix else key
            output.update(_flatten_numeric(child, name))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        if np.isfinite(number):
            output[prefix] = number
    return output


def _aggregate_payloads(payloads: dict[int, dict[str, Any]], field: str) -> dict[str, Any]:
    flattened = {
        seed: _flatten_numeric(payload.get(field, {}))
        for seed, payload in payloads.items()
    }
    keys = sorted(set().union(*(values.keys() for values in flattened.values())))
    aggregated = {}
    for key in keys:
        values = [items[key] for items in flattened.values() if key in items]
        aggregated[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "n": len(values),
            "per_seed": {
                str(seed): items[key]
                for seed, items in flattened.items()
                if key in items
            },
        }
    return aggregated


def _feature_path(seed: int, scenario: str) -> Path:
    return analysis_root(seed) / "features" / f"{scenario}.npz"


def _feature_checkpoint_sha(seed: int, scenario: str) -> str:
    sidecar = _feature_path(seed, scenario).with_suffix(".json")
    if not sidecar.is_file():
        raise FileNotFoundError(f"Missing feature provenance sidecar: {sidecar}")
    return str(json.loads(sidecar.read_text(encoding="utf-8"))["checkpoint_sha256"])


def _ood_output(seed: int, scenario: str) -> Path:
    return analysis_root(seed) / "ood" / scenario


def _print_ood_table(results: dict[str, dict[int, dict[str, Any]]]) -> None:
    print("\nOOD RESULTS (mean +/- std across available seeds)")
    print(
        f"{'Scenario':<34} {'Method':<13} {'Role':<10} "
        f"{'AUROC':>16} {'AUPR-Out':>16} {'FPR@95':>16}"
    )
    print("-" * 112)
    for scenario, seed_payloads in results.items():
        methods = next(iter(seed_payloads.values())).get("results", {}) if seed_payloads else {}
        for method in methods:
            if method == "subgroups":
                continue
            rows = [payload["results"][method] for payload in seed_payloads.values()]
            role = "primary" if rows[0].get("primary_analysis", True) else "secondary"
            values = []
            for key in ("auroc", "aupr_out", "fpr_at_95tpr"):
                samples = [row[key] for row in rows]
                std = np.std(samples, ddof=1) if len(samples) > 1 else 0.0
                values.append(f"{np.mean(samples):.4f} +/- {std:.4f}")
            print(
                f"{scenario:<34} {method:<13} {role:<10} {values[0]:>16} "
                f"{values[1]:>16} {values[2]:>16}"
            )


def aggregate_existing(seeds: list[int], scenarios: list[str]) -> dict[str, Any]:
    ood_results: dict[str, dict[int, dict[str, Any]]] = OrderedDict()
    explain_results: dict[int, dict[str, Any]] = {}
    for scenario in scenarios:
        for seed in seeds:
            path = _ood_output(seed, scenario) / "ood_metrics.json"
            if path.is_file():
                ood_results.setdefault(scenario, {})[seed] = _verified_json(
                    path, seed, "locked_ctch_ood_evaluation"
                )
    for seed in seeds:
        path = analysis_root(seed) / "explainability" / "summary.json"
        if path.is_file():
            explain_results[seed] = _verified_json(
                path, seed, "locked_ctch_explainability_evaluation"
            )

    aggregate = {
        "source_experiment": SOURCE_EXPERIMENT,
        "seeds_requested": seeds,
        "ood": {
            scenario: {
                "seeds": sorted(payloads),
                "aggregated": _aggregate_payloads(payloads, "results"),
            }
            for scenario, payloads in ood_results.items()
        },
        "explainability": {
            "seeds": sorted(explain_results),
            "representation_quality": _aggregate_payloads(
                explain_results, "representation_quality"
            )
            if explain_results
            else {},
            "explanation_aggregate": _aggregate_payloads(
                explain_results, "explanation_aggregate"
            )
            if explain_results
            else {},
        },
    }
    destination = analysis_root() / "aggregated_results.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _print_ood_table(ood_results)
    print(f"Aggregated results: {destination}")
    return aggregate


def main() -> None:
    ood_cfg = OmegaConf.load(OOD_CONFIG)
    explain_cfg = OmegaConf.load(EXPLAIN_CONFIG)
    if (
        str(ood_cfg.source_experiment) != SOURCE_EXPERIMENT
        or str(explain_cfg.source_experiment) != SOURCE_EXPERIMENT
    ):
        raise RuntimeError("Analysis configs are not locked to the canonical source.")
    _validate_locked_configs(ood_cfg, explain_cfg)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[int(value) for value in ood_cfg.seeds]
    )
    parser.add_argument(
        "--analyses",
        nargs="+",
        choices=("ood", "explainability"),
        default=["ood", "explainability"],
    )
    parser.add_argument(
        "--ood-scenarios",
        nargs="+",
        choices=tuple(ood_cfg.scenarios.keys()),
        default=list(ood_cfg.scenarios.keys()),
    )
    parser.add_argument("--allow-incomplete-ood", action="store_true")
    parser.add_argument("--feature-only", action="store_true")
    parser.add_argument("--table", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--n-bootstrap", type=int, default=int(ood_cfg.n_bootstrap))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--no-strict-fingerprint", action="store_true")
    args = parser.parse_args()

    invalid_seeds = sorted(set(args.seeds) - set(SOURCE_SEEDS))
    if invalid_seeds:
        raise ValueError(
            f"Only trained proposed seeds {list(SOURCE_SEEDS)} are allowed; got {invalid_seeds}."
        )
    if args.table:
        aggregate_existing(args.seeds, args.ood_scenarios)
        return

    print("+----------------------------------------------------------------+")
    print("|  CTCH proposed post-hoc OOD + explainability                  |")
    print("+----------------------------------------------------------------+")
    print(f"  Source   : {SOURCE_EXPERIMENT}")
    print(f"  Seeds    : {args.seeds}")
    print(f"  Analyses : {args.analyses}")
    print("  Training : disabled (this runner never calls train.py)")

    failures = []
    for seed in args.seeds:
        try:
            checkpoint = locked_checkpoint_path(seed)
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")

            required_features = {"ctch_train", "ctch_test"}
            if "ood" in args.analyses:
                required_features.add("ctch_val")
                required_features.update(
                    str(ood_cfg.scenarios[scenario].archive)
                    for scenario in args.ood_scenarios
                )
            feature_command = [
                sys.executable,
                str(ROOT / "tools" / "export_analysis_features.py"),
                "--seed",
                str(seed),
                "--scenarios",
                *sorted(required_features),
                "--batch-size",
                str(args.batch_size),
                "--num-workers",
                str(args.num_workers),
                "--device",
                args.device,
            ]
            if args.allow_incomplete_ood:
                feature_command.append("--allow-incomplete-ood")
            if args.overwrite:
                feature_command.append("--overwrite")
            if args.no_strict_fingerprint:
                feature_command.append("--no-strict-fingerprint")
            _run(feature_command, f"FEATURES seed={seed}")
            for scenario in required_features:
                if not _feature_path(seed, scenario).is_file():
                    raise FileNotFoundError(
                        f"Feature export did not create {_feature_path(seed, scenario)}"
                    )
            if args.feature_only:
                continue

            if "ood" in args.analyses:
                for scenario in args.ood_scenarios:
                    scenario_cfg = ood_cfg.scenarios[scenario]
                    output_dir = _ood_output(seed, scenario)
                    existing_metrics = output_dir / "ood_metrics.json"
                    if existing_metrics.is_file() and not args.overwrite:
                        existing = _verified_json(
                            existing_metrics,
                            seed,
                            "locked_ctch_ood_evaluation",
                        )
                        if existing.get("checkpoint_sha256") == _feature_checkpoint_sha(
                            seed, "ctch_train"
                        ):
                            print(
                                f"[Resume] OOD {scenario} seed={seed}: "
                                "verified result exists."
                            )
                            continue
                        print(
                            f"[Refresh] OOD {scenario} seed={seed}: checkpoint "
                            "provenance changed."
                        )
                    command = [
                        sys.executable,
                        str(ROOT / "evaluate_ood.py"),
                        "--seed",
                        str(seed),
                        "--scenario",
                        scenario,
                        "--train-embeddings",
                        str(_feature_path(seed, "ctch_train")),
                        "--calibration-embeddings",
                        str(_feature_path(seed, "ctch_val")),
                        "--id-test-embeddings",
                        str(_feature_path(seed, "ctch_test")),
                        "--ood-embeddings",
                        str(_feature_path(seed, str(scenario_cfg.archive))),
                        "--feature-key",
                        str(scenario_cfg.feature_key),
                        "--methods",
                        *[str(value) for value in ood_cfg.methods],
                        "--primary-methods",
                        *[str(value) for value in scenario_cfg.primary_methods],
                        "--scenario-role",
                        "primary" if bool(scenario_cfg.primary) else "secondary",
                        "--target-id-fpr",
                        str(float(ood_cfg.target_id_fpr)),
                        "--knn-k",
                        str(int(ood_cfg.knn_k)),
                        "--temperature",
                        str(float(ood_cfg.temperature)),
                        "--n-bootstrap",
                        str(args.n_bootstrap),
                        "--bootstrap-seed",
                        str(int(ood_cfg.bootstrap_seed) + seed),
                        "--output-dir",
                        str(output_dir),
                    ]
                    command.append(
                        "--paired-by-image-id"
                        if bool(scenario_cfg.get("paired_by_image_id", False))
                        else "--no-paired-by-image-id"
                    )
                    if args.allow_incomplete_ood:
                        command.append("--allow-incomplete-ood")
                    _run(command, f"OOD {scenario} seed={seed}")
                    _verified_json(
                        output_dir / "ood_metrics.json",
                        seed,
                        "locked_ctch_ood_evaluation",
                    )

            if "explainability" in args.analyses:
                output_dir = analysis_root(seed) / "explainability"
                command = [
                    sys.executable,
                    str(ROOT / "tools" / "evaluate_explainability.py"),
                    "--seed",
                    str(seed),
                    "--train-features",
                    str(_feature_path(seed, str(explain_cfg.train_features))),
                    "--test-features",
                    str(_feature_path(seed, str(explain_cfg.test_features))),
                    "--per-class",
                    str(int(explain_cfg.per_class)),
                    "--max-samples",
                    str(int(explain_cfg.max_samples)),
                    "--selection-seed",
                    str(int(explain_cfg.selection_seed)),
                    "--ig-steps",
                    str(int(explain_cfg.integrated_gradients_steps)),
                    "--random-trials",
                    str(int(explain_cfg.random_deletion_trials)),
                    "--stability-repeats",
                    str(int(explain_cfg.stability_repeats)),
                    "--stability-steps",
                    str(int(explain_cfg.stability_steps)),
                    "--noise-scale",
                    str(float(explain_cfg.stability_noise_scale)),
                    "--sanity-samples",
                    str(int(explain_cfg.classifier_randomization_samples)),
                    "--render-samples",
                    str(int(explain_cfg.render_samples)),
                    "--device",
                    args.device,
                    "--output-dir",
                    str(output_dir),
                ]
                if args.overwrite:
                    command.append("--overwrite")
                if args.no_strict_fingerprint:
                    command.append("--no-strict-fingerprint")
                _run(command, f"EXPLAIN seed={seed}")
                _verified_json(
                    output_dir / "summary.json",
                    seed,
                    "locked_ctch_explainability_evaluation",
                )
        except Exception as error:
            failures.append({"seed": seed, "error": str(error)})
            print(f"[FAIL] seed={seed}: {error}")
            if not args.continue_on_error:
                raise

    aggregate_existing(args.seeds, args.ood_scenarios)
    if failures:
        failure_path = analysis_root() / "failures.json"
        failure_path.write_text(
            json.dumps(failures, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        raise SystemExit(f"{len(failures)} seed(s) failed; see {failure_path}")


if __name__ == "__main__":
    main()
