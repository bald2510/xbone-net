"""Chạy kiểm định ghép cặp giữa mô hình tham chiếu và các mô hình cơ sở.

Notes
-----
Chương trình dùng bootstrap phân tầng theo bệnh nhân kết hợp lấy mẫu lại theo
hạt giống. Giá trị p thô của từng phép so sánh được báo cáo trực tiếp.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.visualize.results import (
    CONFIG_DISPLAY_NAMES,
    _load_aligned_prediction_archive,
    _paired_variant_statistics,
)


SEEDS = (42, 123, 456)
BASELINES = (
    ("baselines/peft_finetuned/lora_biomedclip", "lora_biomedclip"),
    ("baselines/peft_finetuned/lora_pubmedclip", "lora_pubmedclip"),
    ("baselines/full_finetuned/fft_biomedclip", "fft_biomedclip"),
    ("baselines/full_finetuned/fft_pubmedclip", "fft_pubmedclip"),
    ("baselines/full_finetuned/fft_clip", "fft_clip"),
    ("baselines/full_finetuned/fft_medclip", "fft_medclip"),
    ("baselines/full_finetuned/fft_resnet50", "fft_resnet50"),
    ("baselines/full_finetuned/fft_densenet", "fft_densenet"),
)
SUPPORTED_METRICS = (
    "accuracy",
    "balanced_accuracy",
    "f1_macro",
    "auroc_macro",
    "auprc_macro",
    "ece_15",
    "brier_score",
)


def parse_args() -> argparse.Namespace:
    """Đọc tham số dòng lệnh.

    Returns
    -------
    argparse.Namespace
        Các tham số của quy trình kiểm định.
    """
    parser = argparse.ArgumentParser(
        description="Bootstrap ghép cặp và p-value thô so với baseline."
    )
    parser.add_argument("--dataset", choices=("btxrd", "ctch"), required=True)
    parser.add_argument(
        "--reference-experiment",
        default="proposed/ours_xbone_net",
        help="Đường dẫn thí nghiệm tham chiếu bên trong thư mục bộ dữ liệu.",
    )
    parser.add_argument("--reference-label", default="XBone-Net")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=SUPPORTED_METRICS,
        required=True,
    )
    parser.add_argument("--n-bootstrap", type=int, default=10_000)
    parser.add_argument("--n-permutations", type=int, default=10_000)
    parser.add_argument(
        "--test-method",
        choices=("permutation", "bootstrap"),
        default="permutation",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--random-seed", type=int, default=2026)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def archive_path(experiment: str, dataset: str, seed: int) -> Path:
    """Tạo đường dẫn tệp dự đoán theo mẫu.

    Parameters
    ----------
    experiment : str
        Đường dẫn tương đối của thí nghiệm bên trong bộ dữ liệu.
    dataset : str
        Tên bộ dữ liệu.
    seed : int
        Hạt giống huấn luyện.

    Returns
    -------
    pathlib.Path
        Đường dẫn tệp dự đoán NPZ.
    """
    return (
        ROOT
        / "results"
        / dataset
        / experiment
        / f"seed_{seed}"
        / "analysis"
        / "features"
        / f"{dataset}_test.npz"
    )


def load_predictions(
    dataset: str,
    seeds: list[int],
    reference_experiment: str,
) -> tuple[
    dict[int, np.ndarray],
    dict[str, dict[int, np.ndarray]],
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Nạp và căn chỉnh dự đoán của mô hình đề xuất và baseline.

    Parameters
    ----------
    dataset : str
        Tên bộ dữ liệu.
    seeds : list[int]
        Các hạt giống dùng trong kiểm định.
    reference_experiment : str
        Đường dẫn thí nghiệm tham chiếu bên trong bộ dữ liệu.

    Returns
    -------
    tuple
        Dự đoán tham chiếu, dự đoán baseline, nhãn, mã ảnh và mã bệnh nhân.

    Raises
    ------
    ValueError
        Khi mã bệnh nhân không đồng nhất giữa các tệp ghép cặp.
    """
    reference: dict[int, np.ndarray] = {}
    baselines: dict[str, dict[int, np.ndarray]] = {
        key: {} for _, key in BASELINES
    }
    common_image_ids: np.ndarray | None = None
    common_labels: np.ndarray | None = None
    common_patient_ids: np.ndarray | None = None

    experiments = ((reference_experiment, "reference"),) + BASELINES
    for experiment, key in experiments:
        target = reference if key == "reference" else baselines[key]
        for seed in seeds:
            source = archive_path(experiment, dataset, seed)
            archive = _load_aligned_prediction_archive(
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
            target[seed] = archive["probabilities"]

    assert common_image_ids is not None
    assert common_labels is not None
    assert common_patient_ids is not None
    return reference, baselines, common_labels, common_image_ids, common_patient_ids


def run_analysis(args: argparse.Namespace) -> tuple[Path, Path]:
    """Thực hiện kiểm định và ghi kết quả.

    Parameters
    ----------
    args : argparse.Namespace
        Các tham số dòng lệnh đã được kiểm tra.

    Returns
    -------
    tuple[pathlib.Path, pathlib.Path]
        Đường dẫn CSV kết quả và JSON mô tả giao thức.
    """
    reference, baselines, labels, image_ids, patient_ids = load_predictions(
        args.dataset,
        args.seeds,
        args.reference_experiment,
    )
    rows: list[dict[str, Any]] = []
    for index, (_, baseline_key) in enumerate(BASELINES, start=1):
        print(f"[{index}/{len(BASELINES)}] {baseline_key}", flush=True)
        baseline_rows = _paired_variant_statistics(
            reference,
            baselines[baseline_key],
            labels=labels,
            patient_ids=patient_ids,
            seeds=args.seeds,
            metrics=args.metrics,
            n_bootstrap=args.n_bootstrap,
            n_permutations=args.n_permutations,
            alpha=args.alpha,
            random_seed=args.random_seed + index,
            test_method=args.test_method,
        )
        for row in baseline_rows:
            row.update(
                {
                    "dataset": args.dataset,
                    "reference_experiment": (
                        f"{args.dataset}/{args.reference_experiment.strip('/')}"
                    ),
                    "reference": args.reference_label,
                    "baseline_experiment": f"{args.dataset}/{BASELINES[index - 1][0]}",
                    "baseline_config": baseline_key,
                    "baseline": CONFIG_DISPLAY_NAMES.get(baseline_key, baseline_key),
                    "delta_definition": "baseline_minus_reference",
                    "n_cases": len(labels),
                    "n_patients": len(np.unique(patient_ids)),
                    "n_seeds": len(args.seeds),
                    "seeds": "|".join(map(str, args.seeds)),
                    "n_bootstrap": args.n_bootstrap,
                    "n_permutations": args.n_permutations,
                    "alpha": args.alpha,
                }
            )
            rows.append(row)

    frame = pd.DataFrame(rows)
    frame["significant_raw"] = frame["p_raw"] < args.alpha
    frame["ci_excludes_zero"] = (
        (frame["delta_ci_low"] > 0.0) | (frame["delta_ci_high"] < 0.0)
    )

    output_dir = args.output_dir or (
        ROOT
        / "results"
        / "summary"
        / "classification"
        / f"statistics_{args.dataset}_{args.test_method}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "paired_bootstrap_results.csv"
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")

    protocol = {
        "analysis": "paired_full_shot_classification",
        "dataset": args.dataset,
        "reference": f"{args.dataset}/{args.reference_experiment.strip('/')}",
        "reference_label": args.reference_label,
        "baselines": [f"{args.dataset}/{path}" for path, _ in BASELINES],
        "metrics": args.metrics,
        "delta_definition": "baseline_minus_reference",
        "confidence_interval": "paired crossed percentile bootstrap",
        "test_method": args.test_method,
        "multiplicity_adjustment": "none; report pointwise raw p-values",
        "n_bootstrap": args.n_bootstrap,
        "n_permutations": args.n_permutations,
        "alpha": args.alpha,
        "random_seed": args.random_seed,
        "seeds": args.seeds,
        "n_cases": len(labels),
        "n_patients": len(np.unique(patient_ids)),
        "image_alignment_verified": len(image_ids),
    }
    protocol_path = output_dir / "statistical_protocol.json"
    protocol_path.write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return csv_path, protocol_path


def main() -> None:
    """Điều phối quy trình kiểm định ghép cặp."""
    args = parse_args()
    csv_path, protocol_path = run_analysis(args)
    print(f"[KẾT QUẢ] {csv_path}")
    print(f"[GIAO THỨC] {protocol_path}")


if __name__ == "__main__":
    main()
