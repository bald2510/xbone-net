"""Screen currently excluded CTCH OOD rows without changing locked manifests.

The script compares the official ``ctch-ood.csv`` cohort with workbook rows
that are marked OOD and Deleted but still have complete materialized image and
clinical-report artifacts.  Candidate features are extracted with the same
three XBone-Net checkpoints and scored by the benchmark's multimodal ensemble.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_curve


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.analysis import CTCHOODDataset
from src.utils.analysis import AnalysisDataCollator, load_evaluated_classification_model
from src.utils.ood import MultimodalEnsembleOODDetector, evaluate_ood
from tools.export_analysis_features import collect_fused_feature_batches


EXPERIMENT = "ctch/proposed/ours_xbone_net"
DEFAULT_SEEDS = (42, 123, 456)
DATA_ROOT = ROOT / "data" / "CTCH"
FEATURE_ROOT = ROOT / "results" / EXPERIMENT


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description="Screen excluded CTCH OOD candidates with current checkpoints."
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "summary" / "ood" / "candidate_screening",
    )
    return parser.parse_args()


def _image_id(frame: pd.DataFrame) -> pd.Series:
    """Reproduce the preprocessing image identifier without importing data code."""
    row_id = pd.to_numeric(frame["ID"], errors="coerce").astype("Int64").astype(str)
    selected = frame["SelectedImage"].astype(str).map(lambda value: Path(value).name)
    return row_id + "_" + selected


def _candidate_rows() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return complete, patient-disjoint deleted OOD rows and an audit summary."""
    workbook = pd.read_excel(
        DATA_ROOT / "data_labeled_full_selected_cleaned.xlsx",
        sheet_name=0,
        engine="openpyxl",
    )
    workbook = workbook.copy()
    workbook["image_id"] = _image_id(workbook)

    ood = pd.to_numeric(workbook["OOD"], errors="coerce").fillna(0).astype(int).eq(1)
    deleted = (
        pd.to_numeric(workbook["Deleted"], errors="coerce")
        .fillna(0)
        .astype(int)
        .ne(0)
    )
    implants = (
        pd.to_numeric(workbook["Implants"], errors="coerce")
        .fillna(0)
        .astype(int)
        .ne(0)
    )
    deleted_ood = workbook.loc[ood & deleted].copy()
    excluded = workbook.loc[ood & deleted & ~implants].copy()

    split = pd.read_csv(DATA_ROOT / "ctch-split.csv")
    id_patients = set(split["patient_key"].astype(str))
    excluded["patient_key"] = excluded["Mã bệnh nhân"].map(
        lambda value: str(int(value)) if pd.notna(value) else ""
    )

    def missing_artifacts(image_id: str) -> str:
        stem = Path(image_id).stem
        required = {
            "image": DATA_ROOT / "images" / image_id,
            "clinical_en": DATA_ROOT / "reports" / "clinical" / f"{stem}.txt",
            "clinical_vi": DATA_ROOT / "reports" / "clinical_vi" / f"{stem}.txt",
        }
        return ";".join(name for name, path in required.items() if not path.is_file())

    excluded["missing_artifacts"] = excluded["image_id"].map(missing_artifacts)
    excluded["patient_overlap_id"] = excluded["patient_key"].isin(id_patients)
    eligible = excluded.loc[
        excluded["missing_artifacts"].eq("") & ~excluded["patient_overlap_id"]
    ].copy()

    audit = {
        "workbook_ood_rows": int(ood.sum()),
        "official_manifest_rows": int(len(pd.read_csv(DATA_ROOT / "ctch-ood.csv"))),
        "deleted_ood_rows": int(len(deleted_ood)),
        "deleted_ood_nonimplant_rows": int(len(excluded)),
        "excluded_implant_rows": deleted_ood.loc[
            pd.to_numeric(deleted_ood["Implants"], errors="coerce")
            .fillna(0)
            .astype(int)
            .ne(0),
            "image_id",
        ].astype(str).tolist(),
        "complete_patient_disjoint_candidates": int(len(eligible)),
        "excluded_missing_artifacts": excluded.loc[
            excluded["missing_artifacts"].ne(""), "image_id"
        ].astype(str).tolist(),
        "excluded_patient_overlap": excluded.loc[
            excluded["patient_overlap_id"], "image_id"
        ].astype(str).tolist(),
    }
    return eligible.reset_index(drop=True), audit


def _feature_archive(seed: int, name: str) -> dict[str, np.ndarray]:
    """Load a trusted current feature archive."""
    path = FEATURE_ROOT / f"seed_{seed}" / "analysis" / "features" / f"{name}.npz"
    with np.load(path, allow_pickle=True) as archive:
        return {key: archive[key] for key in archive.files if key != "provenance_json"}


def _extract_candidates(
    candidates: pd.DataFrame,
    seed: int,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    """Extract candidate features with the requested evaluated checkpoint."""
    loaded = load_evaluated_classification_model(EXPERIMENT, seed, device=device)
    params = loaded.cfg.dataset.params
    preprocess = dict(getattr(params, "preprocess", {}) or {})
    manifest_columns = ["image_id", "Mã bệnh nhân", "Mã ca khám", "Label", "Nhóm bệnh", "OOD"]

    with tempfile.TemporaryDirectory(prefix="ctch_ood_candidates_") as temp_dir:
        manifest = Path(temp_dir) / "candidates.csv"
        candidates[manifest_columns].to_csv(manifest, index=False)
        dataset = CTCHOODDataset(
            img_dir=str(params.img_dir),
            xray_report_dir=str(params.xray_report_dir),
            clinical_report_dir=str(params.clinical_report_dir),
            csv_manifest_path=str(manifest),
            transform=loaded.model.backbone.preprocess,
            tokenizer=getattr(
                loaded.model.backbone,
                "tokenizer_obj",
                getattr(loaded.model.backbone, "tokenizer", None),
            ),
            preprocess=preprocess,
            required_report_types=("clinical",),
            allow_missing=False,
        )
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=AnalysisDataCollator(loaded.model.backbone.tokenizer_obj),
        )
        return collect_fused_feature_batches(loaded, loader)


def _detector_parts(
    train: dict[str, np.ndarray],
    val: dict[str, np.ndarray],
    values: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return visual, text, and max-ensemble z scores."""
    detector = MultimodalEnsembleOODDetector(
        knn_k=10, knn_reduction="kth", knn_metric="euclidean"
    ).fit(
        visual_embeddings=train["visual_global_embeddings"],
        text_embeddings=train["text_global_embeddings"],
        val_visual_embeddings=val["visual_global_embeddings"],
        val_text_embeddings=val["text_global_embeddings"],
    )
    components = detector.score_components(
        values["visual_global_embeddings"],
        text_embeddings=values["text_global_embeddings"],
    )
    return components["visual_z"], components["text_z"], components["ensemble"]


def _id95_cutoff(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    """Recover the exact score cutoff used by evaluate_ood's FPR@95%TPR."""
    labels = np.concatenate(
        [np.ones(len(id_scores), dtype=np.int64), np.zeros(len(ood_scores), dtype=np.int64)]
    )
    fpr, tpr, thresholds = roc_curve(
        labels, -np.concatenate([id_scores, ood_scores]), pos_label=1
    )
    eligible = np.flatnonzero(tpr >= 0.95)
    best = eligible[np.argmin(fpr[eligible])]
    return float(-thresholds[best])


def main() -> None:
    """Run candidate screening and write auditable CSV/JSON artifacts."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates, audit = _candidate_rows()
    if candidates.empty:
        raise RuntimeError("No complete, patient-disjoint deleted OOD candidates found.")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu"
    )
    print(f"Screening {len(candidates)} candidates on {device} with seeds {args.seeds}")

    output = candidates.copy()
    summaries: dict[str, Any] = {}
    for seed in args.seeds:
        print(f"Seed {seed}: extracting candidate features")
        candidate_features = _extract_candidates(
            candidates, seed=seed, device=device, batch_size=args.batch_size
        )
        train = _feature_archive(seed, "ctch_train")
        val = _feature_archive(seed, "ctch_val")
        test = _feature_archive(seed, "ctch_test")
        current = _feature_archive(seed, "ctch_ood")

        _, _, test_scores = _detector_parts(train, val, test)
        _, _, current_scores = _detector_parts(train, val, current)
        visual, text, candidate_scores = _detector_parts(train, val, candidate_features)
        cutoff = _id95_cutoff(test_scores, current_scores)

        output[f"score_seed_{seed}"] = candidate_scores
        output[f"visual_z_seed_{seed}"] = visual
        output[f"text_z_seed_{seed}"] = text
        output[f"accepted_as_id_seed_{seed}"] = candidate_scores <= cutoff
        output[f"predicted_class_id_seed_{seed}"] = candidate_features["predictions"]

        combined_scores = np.concatenate([current_scores, candidate_scores])
        summaries[str(seed)] = {
            "id95_score_cutoff": cutoff,
            "official": evaluate_ood(test_scores, current_scores),
            "official_plus_all_complete_deleted_candidates": evaluate_ood(
                test_scores, combined_scores
            ),
            "candidate_count": int(len(candidate_scores)),
            "candidate_rejected_as_ood": int(np.sum(candidate_scores > cutoff)),
        }

    score_columns = [f"score_seed_{seed}" for seed in args.seeds]
    visual_columns = [f"visual_z_seed_{seed}" for seed in args.seeds]
    text_columns = [f"text_z_seed_{seed}" for seed in args.seeds]
    accepted_columns = [f"accepted_as_id_seed_{seed}" for seed in args.seeds]
    output["score_mean"] = output[score_columns].mean(axis=1)
    output["score_min"] = output[score_columns].min(axis=1)
    output["accepted_as_id_seed_count"] = output[accepted_columns].sum(axis=1)
    output["dominant_modality"] = np.where(
        (output[text_columns].to_numpy() >= output[visual_columns].to_numpy()).sum(axis=1)
        >= (len(args.seeds) // 2 + 1),
        "text",
        "image",
    )
    output["improves_fpr95_all_seeds"] = output["accepted_as_id_seed_count"].eq(0)
    output = output.sort_values(
        ["improves_fpr95_all_seeds", "score_mean"], ascending=[False, False]
    ).reset_index(drop=True)
    output.insert(0, "candidate_rank", np.arange(1, len(output) + 1))

    keep = [
        "candidate_rank",
        "image_id",
        "Mã bệnh nhân",
        "Mã ca khám",
        "Nhóm bệnh",
        "LyDoVaoVien_BenhAN",
        "score_mean",
        "score_min",
        "accepted_as_id_seed_count",
        "dominant_modality",
        "improves_fpr95_all_seeds",
        *score_columns,
        *visual_columns,
        *text_columns,
        *accepted_columns,
        *[f"predicted_class_id_seed_{seed}" for seed in args.seeds],
    ]
    csv_path = args.output_dir / "ctch_deleted_ood_candidate_scores.csv"
    output[keep].to_csv(csv_path, index=False, encoding="utf-8-sig")

    summary = {
        "experiment": EXPERIMENT,
        "seeds": args.seeds,
        "device": str(device),
        "selection_note": (
            "Exploratory screening only. Deleted rows must not enter the official "
            "cohort until their exclusion reasons and clinical eligibility are reviewed."
        ),
        "audit": audit,
        "per_seed": summaries,
        "candidates_rejected_as_ood_all_seeds": output.loc[
            output["improves_fpr95_all_seeds"], "image_id"
        ].astype(str).tolist(),
    }
    json_path = args.output_dir / "ctch_deleted_ood_candidate_summary.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
