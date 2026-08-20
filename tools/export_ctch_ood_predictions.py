"""Export current CTCH semantic-OOD predictions to an auditable JSON payload.

The source cohort is the locked ``data/CTCH/ctch-ood.csv`` manifest.  The
script reuses current feature archives for XBone-Net seeds 42, 123 and 456 and
recomputes the multimodal-ensemble OOD score with the benchmark implementation.
No dataset manifest or feature archive is modified.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.ood import MultimodalEnsembleOODDetector


EXPERIMENT = "ctch/proposed/ours_xbone_net"
SEEDS = (42, 123, 456)
FEATURE_ROOT = ROOT / "results" / EXPERIMENT


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Export current CTCH OOD predictions for all proposed-model seeds."
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_archive(seed: int, name: str) -> dict[str, np.ndarray]:
    """Load one current feature archive."""
    path = FEATURE_ROOT / f"seed_{seed}" / "analysis" / "features" / f"{name}.npz"
    with np.load(path, allow_pickle=True) as archive:
        return {key: archive[key] for key in archive.files if key != "provenance_json"}


def score_parts(
    train: dict[str, np.ndarray],
    val: dict[str, np.ndarray],
    values: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return visual, text and fused OOD z-scores."""
    detector = MultimodalEnsembleOODDetector(
        knn_k=10,
        knn_reduction="kth",
        knn_metric="euclidean",
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


def id95_cutoff(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    """Recover the score cutoff used by the benchmark's FPR@95%TPR metric."""
    labels = np.concatenate(
        [
            np.ones(len(id_scores), dtype=np.int64),
            np.zeros(len(ood_scores), dtype=np.int64),
        ]
    )
    fpr, tpr, thresholds = roc_curve(
        labels,
        -np.concatenate([id_scores, ood_scores]),
        pos_label=1,
    )
    eligible = np.flatnonzero(tpr >= 0.95)
    best = eligible[np.argmin(fpr[eligible])]
    return float(-thresholds[best])


def benchmark_record() -> dict[str, Any]:
    """Load current benchmark thresholds and metrics."""
    path = ROOT / "results" / "summary" / "ood" / "ood_benchmark_summary.json"
    records = json.loads(path.read_text(encoding="utf-8"))
    return next(record for record in records if record["experiment"] == EXPERIMENT)


def checkpoint_provenance(seed: int) -> dict[str, Any]:
    """Read the saved feature provenance for one OOD archive."""
    path = FEATURE_ROOT / f"seed_{seed}" / "analysis" / "features" / "ctch_ood.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    return {
        "seed": seed,
        "checkpoint": record.get("checkpoint", ""),
        "checkpoint_sha256": record.get("checkpoint_sha256", ""),
        "git_revision": record.get("git_revision", ""),
        "created_at": record.get("created_at", ""),
        "config_sha256": record.get("config_sha256", ""),
    }


def consensus(labels: list[str]) -> str:
    """Return a unique plurality label or an explicit tie string."""
    counts = Counter(labels)
    best = max(counts.values())
    winners = sorted(label for label, count in counts.items() if count == best)
    return winners[0] if len(winners) == 1 else "Tie: " + " | ".join(winners)


def main() -> None:
    """Build and save the prediction payload."""
    args = parse_args()
    labels = [
        line.strip()
        for line in (ROOT / "data" / "CTCH" / "labels.txt")
        .read_text(encoding="utf-8-sig")
        .splitlines()
        if line.strip()
    ]
    manifest = pd.read_csv(ROOT / "data" / "CTCH" / "ctch-ood.csv")
    benchmark = benchmark_record()
    expected_ids = manifest["image_id"].astype(str).tolist()

    long_rows: list[dict[str, Any]] = []
    by_image: dict[str, list[dict[str, Any]]] = {image_id: [] for image_id in expected_ids}
    seed_summaries: list[dict[str, Any]] = []

    for seed in SEEDS:
        train = load_archive(seed, "ctch_train")
        val = load_archive(seed, "ctch_val")
        test = load_archive(seed, "ctch_test")
        ood = load_archive(seed, "ctch_ood")
        archive_ids = ood["image_id"].astype(str).tolist()
        if archive_ids != expected_ids:
            raise ValueError(f"Seed {seed} OOD archive is not aligned to ctch-ood.csv")
        if ood["probabilities"].shape != (len(expected_ids), len(labels)):
            raise ValueError(f"Seed {seed} probability matrix has an unexpected shape")

        _, _, test_scores = score_parts(train, val, test)
        visual, text, scores = score_parts(train, val, ood)
        fpr95_score_cutoff = id95_cutoff(test_scores, scores)
        seed_benchmark = benchmark["per_seed"][str(seed)]["semantic_ood"][
            "multimodal_ensemble"
        ]
        deployment_threshold = float(seed_benchmark["threshold"])

        flagged = scores > deployment_threshold
        accepted_fpr95 = scores <= fpr95_score_cutoff
        seed_summaries.append(
            {
                "seed": seed,
                "sample_count": len(expected_ids),
                "deployment_threshold": deployment_threshold,
                "flagged_ood_count": int(flagged.sum()),
                "flagged_ood_rate": float(flagged.mean()),
                "fpr95_score_cutoff": fpr95_score_cutoff,
                "accepted_as_id_at_fpr95_count": int(accepted_fpr95.sum()),
                "auroc_ood": float(seed_benchmark["auroc_ood"]),
                "aupr_out": float(seed_benchmark["aupr_out"]),
                "fpr_at_95tpr": float(seed_benchmark["fpr_at_95tpr"]),
            }
        )

        for index, image_id in enumerate(expected_ids):
            probabilities = np.asarray(ood["probabilities"][index], dtype=float)
            predicted_id = int(ood["predictions"][index])
            row = {
                "image_id": image_id,
                "patient_id": str(manifest.iloc[index]["Mã bệnh nhân"]),
                "visit_id": str(manifest.iloc[index]["Mã ca khám"]),
                "group": str(manifest.iloc[index]["Nhóm bệnh"]),
                "seed": seed,
                "predicted_class_id": predicted_id,
                "predicted_label": labels[predicted_id],
                "confidence": float(probabilities[predicted_id]),
                "ood_score": float(scores[index]),
                "deployment_threshold": deployment_threshold,
                "ood_margin": float(scores[index] - deployment_threshold),
                "ood_status": "OOD warning" if flagged[index] else "Accepted as ID",
                "visual_z": float(visual[index]),
                "text_z": float(text[index]),
                "dominant_modality": "text" if text[index] >= visual[index] else "image",
                "fpr95_score_cutoff": fpr95_score_cutoff,
                "accepted_as_id_at_fpr95": bool(accepted_fpr95[index]),
                "probabilities": {
                    labels[class_id]: float(probabilities[class_id])
                    for class_id in range(len(labels))
                },
            }
            long_rows.append(row)
            by_image[image_id].append(row)

    wide_rows = []
    for index, image_id in enumerate(expected_ids):
        rows = by_image[image_id]
        predicted_labels = [row["predicted_label"] for row in rows]
        wide_rows.append(
            {
                "image_id": image_id,
                "patient_id": str(manifest.iloc[index]["Mã bệnh nhân"]),
                "visit_id": str(manifest.iloc[index]["Mã ca khám"]),
                "group": str(manifest.iloc[index]["Nhóm bệnh"]),
                "consensus_prediction": consensus(predicted_labels),
                "prediction_agreement_count": max(Counter(predicted_labels).values()),
                "ood_warning_seed_count": sum(
                    row["ood_status"] == "OOD warning" for row in rows
                ),
                "mean_confidence": float(np.mean([row["confidence"] for row in rows])),
                "mean_ood_score": float(np.mean([row["ood_score"] for row in rows])),
                "seed_results": {str(row["seed"]): row for row in rows},
            }
        )

    payload = {
        "title": "Current CTCH semantic-OOD predictions",
        "experiment": EXPERIMENT,
        "protocol": {
            "cohort_manifest": "data/CTCH/ctch-ood.csv",
            "sample_count": len(expected_ids),
            "seeds": list(SEEDS),
            "class_count": len(labels),
            "ood_method": "multimodal_ensemble",
            "knn_k": 10,
            "score_direction": "higher_is_more_ood",
            "deployment_rule": "OOD warning when ood_score > validation-calibrated threshold",
            "fpr95_note": (
                "accepted_as_id_at_fpr95 is an evaluation-only indicator based on the "
                "test-ID 95% acceptance cutoff; it is not the deployment decision."
            ),
        },
        "class_labels": [
            {"class_id": class_id, "label": label}
            for class_id, label in enumerate(labels)
        ],
        "seed_summaries": seed_summaries,
        "wide_rows": wide_rows,
        "long_rows": long_rows,
        "provenance": [checkpoint_provenance(seed) for seed in SEEDS],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {args.output}")
    print(f"Rows: wide={len(wide_rows)}, long={len(long_rows)}")


if __name__ == "__main__":
    main()
