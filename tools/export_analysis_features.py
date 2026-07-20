"""Export locked CTCH-proposed features for OOD and explainability analysis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets.analysis import CTCHOODDataset, ReportMismatchDataset
from src.datasets.ctch import CTCHDataset
from src.datasets.fracatlas import (
    FRACATLAS_IMAGE_RESOLVER_VERSION,
    FracAtlasDataset,
)
from src.utils.analysis import (
    MetadataDataset,
    SOURCE_EXPERIMENT,
    analysis_root,
    build_analysis_loader,
    collect_feature_batches,
    load_feature_archive,
    load_locked_proposed_model,
    save_feature_archive,
)


SCENARIOS = (
    "ctch_train",
    "ctch_val",
    "ctch_test",
    "ctch_ood",
    "fracatlas_test",
    "report_mismatch_cross_class",
    "report_mismatch_same_class",
)


def _plain(value: Any) -> Any:
    return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value


def _ctch_dataset(loaded, split: str) -> CTCHDataset:
    params = dict(_plain(loaded.cfg.dataset.params))
    return CTCHDataset(
        split=split,
        transform=loaded.model.backbone.preprocess,
        tokenizer=loaded.model.backbone.tokenizer_obj,
        **params,
    )


def build_scenario_dataset(
    scenario: str,
    loaded,
    allow_incomplete_ood: bool,
    mismatch_seed: int,
):
    if scenario in {"ctch_train", "ctch_val", "ctch_test"}:
        split = scenario.removeprefix("ctch_")
        return MetadataDataset(_ctch_dataset(loaded, split), scenario), {}

    high_res = dict(_plain(loaded.cfg.dataset.params.high_res))
    if scenario == "ctch_ood":
        dataset = CTCHOODDataset(
            img_dir=str(ROOT / "data" / "CTCH" / "images"),
            xray_report_dir=str(ROOT / "data" / "CTCH" / "reports" / "xray"),
            clinical_report_dir=str(
                ROOT / "data" / "CTCH" / "reports" / "clinical"
            ),
            csv_manifest_path=str(ROOT / "data" / "CTCH" / "ctch-ood.csv"),
            transform=loaded.model.backbone.preprocess,
            tokenizer=loaded.model.backbone.tokenizer_obj,
            high_res=high_res,
            allow_missing=allow_incomplete_ood,
        )
        return dataset, {"coverage": dataset.coverage}

    if scenario == "fracatlas_test":
        dataset = FracAtlasDataset(
            img_dir=str(ROOT / "data" / "FracAtlas" / "images"),
            report_dir=str(ROOT / "data" / "FracAtlas" / "reports"),
            csv_split_path=str(
                ROOT / "data" / "FracAtlas" / "fracatlas-split.csv"
            ),
            csv_labels_path=str(
                ROOT / "data" / "FracAtlas" / "fracatlas-labels.csv"
            ),
            classes=["non-fractured", "fractured"],
            task_type="multiclass",
            split="test",
            transform=loaded.model.backbone.preprocess,
            tokenizer=loaded.model.backbone.tokenizer_obj,
            high_res=high_res,
            strict_files=True,
            allow_truncated_images=True,
        )
        return MetadataDataset(dataset, scenario), {
            "primary_feature": "visual_global_embeddings",
            "coverage": dataset.coverage,
            "fracatlas_image_resolver_version": (
                FRACATLAS_IMAGE_RESOLVER_VERSION
            ),
            "note": (
                "Domain-OOD inference exports fused values for diagnostics, but the "
                "pre-registered primary analysis uses visual_global_embeddings to "
                "avoid report-schema and label-space confounding."
            ),
        }

    if scenario.startswith("report_mismatch_"):
        mode = scenario.removeprefix("report_mismatch_")
        dataset = ReportMismatchDataset(
            _ctch_dataset(loaded, "test"), mode=mode, seed=mismatch_seed
        )
        labels = dataset.df["class_id"].to_numpy()
        recipient_labels = labels[dataset.recipient_indices]
        donor_labels = labels[dataset.donor_indices]
        return dataset, {
            "mismatch_seed": int(mismatch_seed),
            "recipient_count": len(dataset),
            "fixed_points": int(
                (dataset.recipient_indices == dataset.donor_indices).sum()
            ),
            "same_class_pairs": int((recipient_labels == donor_labels).sum()),
            "cross_class_pairs": int((recipient_labels != donor_labels).sum()),
        }
    raise ValueError(f"Unknown scenario {scenario!r}.")


def _can_resume(path: Path, loaded, scenario: str) -> bool:
    if not path.is_file():
        return False
    try:
        _, provenance = load_feature_archive(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    matches_locked_source = (
        provenance.get("checkpoint_sha256") == loaded.checkpoint_sha256
        and provenance.get("config_sha256")
        == loaded.provenance.get("config_sha256")
        and int(provenance.get("seed", -1)) == int(loaded.provenance["seed"])
        and provenance.get("scenario") == scenario
    )
    if not matches_locked_source:
        return False
    if scenario == "fracatlas_test":
        coverage = provenance.get("coverage", {})
        return (
            int(provenance.get("fracatlas_image_resolver_version", -1))
            == FRACATLAS_IMAGE_RESOLVER_VERSION
            and int(coverage.get("rows", -1))
            == int(provenance.get("sample_count", -2))
            and int(coverage.get("resolved_images", -1))
            == int(provenance.get("sample_count", -2))
            and int(coverage.get("missing_images", -1)) == 0
            and int(coverage.get("missing_reports", -1)) == 0
            and bool(coverage.get("decode_validation", False))
            and int(coverage.get("decode_failure_count", -1)) == 0
        )
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=list(SCENARIOS),
        choices=SCENARIOS,
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--mismatch-seed", type=int, default=2025)
    parser.add_argument("--allow-incomplete-ood", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument(
        "--no-strict-fingerprint",
        action="store_true",
        help="Diagnostic only: record but do not enforce the validated parameter count.",
    )
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable.")
        device = torch.device(args.device)

    loaded = load_locked_proposed_model(
        args.seed,
        device=device,
        strict_fingerprint=not args.no_strict_fingerprint,
    )
    feature_root = analysis_root(args.seed) / "features"
    feature_root.mkdir(parents=True, exist_ok=True)
    print(
        f"[Source] {SOURCE_EXPERIMENT} seed={args.seed} "
        f"sha256={loaded.checkpoint_sha256[:12]} device={device}"
    )

    for scenario in args.scenarios:
        output = feature_root / f"{scenario}.npz"
        if not args.overwrite and _can_resume(output, loaded, scenario):
            print(f"[Resume] {scenario}: verified archive already exists.")
            continue

        print(f"[Export] {scenario}")
        dataset, scenario_metadata = build_scenario_dataset(
            scenario,
            loaded,
            allow_incomplete_ood=args.allow_incomplete_ood,
            mismatch_seed=args.mismatch_seed,
        )
        loader = build_analysis_loader(
            dataset,
            tokenizer=loaded.model.backbone.tokenizer_obj,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        arrays = collect_feature_batches(
            loaded.model,
            loader,
            device=loaded.device,
            report_type=str(loaded.cfg.params.phase2.p2_report_type),
        )
        provenance = {
            **loaded.provenance,
            "scenario": scenario,
            "sample_count": int(len(arrays["labels"])),
            "allow_incomplete_ood": bool(args.allow_incomplete_ood),
            "feature_dimensions": {
                key: list(value.shape)
                for key, value in arrays.items()
                if key.endswith("embeddings") or key in {"logits", "probabilities"}
            },
            **scenario_metadata,
        }
        save_feature_archive(output, arrays, provenance)
        print(f"  -> {output} ({len(dataset)} samples)")


if __name__ == "__main__":
    main()
