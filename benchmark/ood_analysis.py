"""Chương trình đánh giá Out-of-Distribution (OOD) toàn diện cho XBone-Net và các mô hình cơ sở / ablation.

Hợp nhất toàn bộ quy trình OOD analysis:
1. Dựa hoàn toàn trên các thuật toán và bộ phát hiện chuẩn từ `src/utils/ood.py`.
2. Nhận trực tiếp một hoặc nhiều thí nghiệm (--experiments) để đánh giá.
3. Đánh giá đa kịch bản (Semantic OOD, Domain OOD BTXRD, Report mismatch).
4. Đa phương pháp phát hiện (Cosine centroids, Mahalanobis centroid, kNN, Entropy).
5. Tự động trích xuất / tái sử dụng bộ đặc trưng (features cache) an toàn.
6. Tính toán thống kê Mean ± Std qua các hạt giống (seeds) kèm khoảng tin cậy Bootstrap 95% CI.
7. Xuất bảng tổng hợp kết quả (Console, JSON, CSV, Markdown).
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

from src.utils.analysis import (
    AnalysisDataCollator,
    SOURCE_EXPERIMENT,
    SOURCE_SEEDS,
    load_feature_archive,
    load_evaluated_classification_model,
    save_feature_archive,
    sha256_file,
)
from src.utils.ood import (
    MultimodalEnsembleOODDetector,
    OODDetector,
    bootstrap_ood_metrics,
    calibrate_ood_threshold,
    evaluate_ood,
)
from tools.export_analysis_features import (
    _collect_fused_feature_batches,
    _collect_zeroshot_feature_batches,
    build_scenario_dataset,
)


SCENARIO_ARCHIVES = OrderedDict(
    [
        ("semantic_ood", "ctch_ood"),
        ("domain_ood_btxrd", "btxrd_test"),
        ("report_mismatch_cross_class", "report_mismatch_cross_class"),
        ("report_mismatch_same_class", "report_mismatch_same_class"),
    ]
)


def _ctch_ood_case_count() -> int | None:
    """Return the number of rows in the current CTCH OOD manifest."""
    manifest_path = ROOT / "data" / "CTCH" / "ctch-ood.csv"
    try:
        return int(len(pd.read_csv(manifest_path)))
    except (OSError, ValueError, pd.errors.ParserError):
        return None


CTCH_OOD_CASE_COUNT = _ctch_ood_case_count()
CTCH_OOD_SCENARIO_LABEL = (
    f"CTCH Semantic OOD ({CTCH_OOD_CASE_COUNT} ca ngoại vi)"
    if CTCH_OOD_CASE_COUNT is not None
    else "CTCH Semantic OOD (manifest hiện hành)"
)

SCENARIO_LABELS = {
    "semantic_ood": CTCH_OOD_SCENARIO_LABEL,
    "domain_ood_btxrd": "BTXRD Dataset (Dịch chuyển miền)",
    "report_mismatch_cross_class": "Mâu thuẫn bệnh sử chéo lớp",
    "report_mismatch_same_class": "Mâu thuẫn bệnh sử cùng lớp",
}

ALL_METHODS = [
    "cosine_centroids",
    "mahalanobis_centroid",
    "knn",
    "entropy",
    "multimodal_ensemble",
]
DEFAULT_METHODS = [
    "cosine_centroids",
    "mahalanobis_centroid",
    "multimodal_ensemble",
]


def parse_args() -> argparse.Namespace:
    """Phân tích các tham số dòng lệnh phục vụ đánh giá OOD.

    Returns
    -------
    argparse.Namespace
        Không gian tên chứa các đối số dòng lệnh đã phân tích.
    """
    parser = argparse.ArgumentParser(description="Unified OOD Analysis Benchmark")
    parser.add_argument(
        "--experiments",
        "--experiment",
        nargs="+",
        default=[SOURCE_EXPERIMENT],
        help="Một hoặc nhiều định danh thí nghiệm cần đánh giá (ví dụ: ctch/proposed/ours_xbone_net ctch/baselines/full_finetuned/fft_biomedclip)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 123, 456],
        help="Danh sách hạt giống ngẫu nhiên (mặc định: 42 123 456)",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=list(SCENARIO_ARCHIVES.keys()),
        default=["semantic_ood", "domain_ood_btxrd"],
        help="Các kịch bản OOD cần đánh giá",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=ALL_METHODS,
        default=DEFAULT_METHODS,
        help="Các phương pháp phát hiện OOD cần đánh giá",
    )
    parser.add_argument("--knn-k", type=int, default=10, help="Tham số k lân cận cho kNN OOD (mặc định: 10 theo Sun et al.)")
    parser.add_argument(
        "--knn-reduction",
        choices=["kth", "mean"],
        default="kth",
        help="Quy tắc điểm kNN OOD: 'kth' (khoảng cách tới lân cận thứ k theo Sun et al.) hoặc 'mean' (trung bình k lân cận)",
    )
    parser.add_argument(
        "--knn-metric",
        choices=["euclidean", "cosine"],
        default="euclidean",
        help="Hàm khoảng cách kNN OOD: 'euclidean' (trên vector chuẩn hóa L2) hoặc 'cosine'",
    )
    parser.add_argument("--target-fpr", type=float, default=0.05, help="Mức FPR mục tiêu trên tập validation (mặc định 0.05 cho 95%% TPR)")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--force-recompute", action="store_true", help="Buộc trích xuất lại đặc trưng kể cả khi đã có cache")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "summary" / "ood",
        help="Thư mục lưu báo cáo JSON và bảng tổng hợp",
    )
    return parser.parse_args()


def extract_or_load_features(
    experiment: str,
    seed: int,
    scenarios: list[str],
    *,
    device: torch.device,
    batch_size: int = 16,
    num_workers: int = 0,
    force_recompute: bool = False,
) -> dict[str, dict[str, np.ndarray]]:
    """Tải checkpoint và trích xuất hoặc nạp đặc trưng cache cho các tập ID và OOD.

    Parameters
    ----------
    experiment : str
        Định danh thí nghiệm (ví dụ: ctch/proposed/ours_xbone_net).
    seed : int
        Hạt giống ngẫu nhiên của mô hình.
    scenarios : list[str]
        Danh sách các kịch bản cần trích xuất đặc trưng.
    device : torch.device
        Thiết bị tính toán (CUDA / CPU).
    batch_size : int, optional
        Kích thước batch khi trích xuất, mặc định 16.
    num_workers : int, optional
        Số luồng nạp dữ liệu, mặc định 0.
    force_recompute : bool, optional
        Buộc tính toán lại đặc trưng kể cả khi đã tồn tại file cache, mặc định False.

    Returns
    -------
    dict[str, dict[str, np.ndarray]]
        Từ điển ánh xạ tên kịch bản sang các mảng đặc trưng tương ứng.
    """
    feature_dir = (
        ROOT / "results" / experiment / f"seed_{seed}" / "analysis" / "features"
    )
    required_archives = {"ctch_train", "ctch_val", "ctch_test"}
    for sc in scenarios:
        if sc in SCENARIO_ARCHIVES:
            required_archives.add(SCENARIO_ARCHIVES[sc])

    feature_dict: dict[str, dict[str, np.ndarray]] = {}
    missing_archives = set()
    checkpoint_path = (
        ROOT
        / "checkpoints"
        / experiment
        / f"seed_{seed}"
        / "best_phase2.pth"
    )
    current_checkpoint_sha = (
        sha256_file(checkpoint_path) if checkpoint_path.is_file() else None
    )

    if not force_recompute and feature_dir.is_dir():
        for archive in required_archives:
            archive_path = feature_dir / f"{archive}.npz"
            if archive_path.is_file():
                try:
                    data, provenance = load_feature_archive(
                        archive_path,
                        expected_source_experiment=experiment,
                    )
                    provenance_matches = (
                        int(provenance.get("seed", -1)) == int(seed)
                        and (
                            current_checkpoint_sha is None
                            or provenance.get("checkpoint_sha256")
                            == current_checkpoint_sha
                        )
                    )
                    if provenance_matches and (
                        "fused_embeddings" in data or "logits" in data
                    ):
                        feature_dict[archive] = data
                    else:
                        missing_archives.add(archive)
                except Exception:
                    missing_archives.add(archive)
            else:
                missing_archives.add(archive)
    else:
        missing_archives = required_archives

    if missing_archives:
        print(f"  [Features] Đang trích xuất đặc trưng ({len(missing_archives)} archives còn thiếu) cho {experiment} seed={seed}...")
        loaded = load_evaluated_classification_model(experiment, seed, device=device)
        is_zeroshot = "zeroshot" in experiment.lower()
        feature_dir.mkdir(parents=True, exist_ok=True)

        for archive in sorted(missing_archives):
            scenario_name = next(
                (key for key, value in SCENARIO_ARCHIVES.items() if value == archive),
                archive,
            )
            dataset, scenario_metadata = build_scenario_dataset(
                archive,
                loaded,
                # Thesis results must use the complete manifest. Abort instead
                # of silently reporting metrics on fewer than the current 26 cases.
                allow_incomplete_ood=False,
            )
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=AnalysisDataCollator(
                    loaded.model.backbone.tokenizer_obj
                ),
            )
            if is_zeroshot:
                arrays = _collect_zeroshot_feature_batches(loaded, loader)
            else:
                arrays = _collect_fused_feature_batches(loaded, loader)

            provenance = dict(loaded.provenance)
            provenance.update(
                {
                    "source_experiment": experiment,
                    "experiment": experiment,
                    "seed": seed,
                    "archive": archive,
                    "scenario": scenario_name,
                    "scenario_metadata": scenario_metadata,
                    "created_at": datetime.datetime.now(
                        datetime.timezone.utc
                    ).isoformat(),
                }
            )
            save_feature_archive(
                feature_dir / f"{archive}.npz",
                arrays,
                provenance=provenance,
            )
            feature_dict[archive] = arrays

    return feature_dict


def evaluate_single_seed_ood(
    feature_dict: dict[str, dict[str, np.ndarray]],
    scenarios: list[str],
    methods: list[str],
    *,
    knn_k: int = 10,
    knn_reduction: str = "kth",
    knn_metric: str = "euclidean",
    target_fpr: float = 0.05,
) -> dict[str, dict[str, Any]]:
    """Đánh giá các độ đo OOD cho 1 seed trên tất cả các kịch bản và phương pháp.

    Parameters
    ----------
    feature_dict : dict[str, dict[str, np.ndarray]]
        Từ điển chứa đặc trưng của các tập ID và OOD.
    scenarios : list[str]
        Danh sách các kịch bản OOD cần đánh giá.
    methods : list[str]
        Danh sách các phương pháp phát hiện OOD.
    knn_k : int, optional
        Số láng giềng k trong kNN, mặc định 5.
    target_fpr : float, optional
        Mức FPR mục tiêu trên validation để xác định ngưỡng, mặc định 0.05.

    Returns
    -------
    dict[str, dict[str, Any]]
        Kết quả đánh giá độ đo OOD chi tiết cho từng kịch bản và phương pháp.
    """
    train_data = feature_dict["ctch_train"]
    val_data = feature_dict["ctch_val"]
    test_id_data = feature_dict["ctch_test"]

    # 1. Huấn luyện các bộ dò OOD trên tập train
    train_embeddings = np.asarray(train_data["fused_embeddings"], dtype=np.float64)
    train_embeddings_raw = np.asarray(train_data.get("fused_embeddings_raw", train_embeddings), dtype=np.float64)
    train_labels = np.asarray(train_data["labels"], dtype=np.int64)

    train_vis = np.asarray(train_data.get("visual_global_embeddings", train_embeddings), dtype=np.float64)
    train_txt = np.asarray(train_data["text_global_embeddings"], dtype=np.float64) if "text_global_embeddings" in train_data else None

    detector_norm = OODDetector().fit(train_embeddings, train_labels)
    detector_raw = OODDetector().fit(train_embeddings_raw, train_labels)

    # 2. Cân chỉnh ngưỡng cảnh báo trên tập validation ID
    val_emb = np.asarray(val_data["fused_embeddings"], dtype=np.float64)
    val_emb_raw = np.asarray(val_data.get("fused_embeddings_raw", val_emb), dtype=np.float64)
    val_logits = np.asarray(val_data["logits"], dtype=np.float64)
    val_vis = np.asarray(val_data.get("visual_global_embeddings", val_emb), dtype=np.float64)
    val_txt = np.asarray(val_data["text_global_embeddings"], dtype=np.float64) if "text_global_embeddings" in val_data else None

    multimodal_detector = None
    if "multimodal_ensemble" in methods:
        multimodal_detector = MultimodalEnsembleOODDetector(
            knn_k=knn_k,
            knn_reduction=knn_reduction,
            knn_metric=knn_metric,
        ).fit(
            visual_embeddings=train_vis,
            labels=train_labels,
            text_embeddings=train_txt,
            logits=train_data.get("logits"),
            val_visual_embeddings=val_vis,
            val_text_embeddings=val_txt,
            val_logits=val_logits,
        )

    val_scores: dict[str, np.ndarray] = {}
    thresholds: dict[str, float] = {}

    for method in methods:
        if method == "mahalanobis_centroid":
            scores = detector_raw.score(val_emb_raw, method=method)
        elif method == "entropy":
            scores = OODDetector.score_entropy(val_logits)
        elif method == "knn":
            scores = detector_norm.score(
                val_emb,
                method="knn",
                k=knn_k,
                reduction=knn_reduction,
                metric=knn_metric,
            )
        elif method == "multimodal_ensemble":
            scores = multimodal_detector.score(
                visual_embeddings=val_vis,
                text_embeddings=val_txt,
                logits=val_logits,
            )
        else:  # cosine_centroids
            scores = detector_norm.score(val_emb, method=method)
        val_scores[method] = scores
        thresholds[method] = calibrate_ood_threshold(scores, target_id_fpr=target_fpr)

    # 3. Tính điểm trên ID test
    test_emb = np.asarray(test_id_data["fused_embeddings"], dtype=np.float64)
    test_emb_raw = np.asarray(test_id_data.get("fused_embeddings_raw", test_emb), dtype=np.float64)
    test_logits = np.asarray(test_id_data["logits"], dtype=np.float64)
    test_vis = np.asarray(test_id_data.get("visual_global_embeddings", test_emb), dtype=np.float64)
    test_txt = np.asarray(test_id_data["text_global_embeddings"], dtype=np.float64) if "text_global_embeddings" in test_id_data else None

    test_id_scores: dict[str, np.ndarray] = {}
    for method in methods:
        if method == "mahalanobis_centroid":
            scores = detector_raw.score(test_emb_raw, method=method)
        elif method == "entropy":
            scores = OODDetector.score_entropy(test_logits)
        elif method == "knn":
            scores = detector_norm.score(
                test_emb,
                method="knn",
                k=knn_k,
                reduction=knn_reduction,
                metric=knn_metric,
            )
        elif method == "multimodal_ensemble":
            scores = multimodal_detector.score(
                visual_embeddings=test_vis,
                text_embeddings=test_txt,
                logits=test_logits,
            )
        else:
            scores = detector_norm.score(test_emb, method=method)
        test_id_scores[method] = scores

    # 4. Đánh giá từng kịch bản OOD
    scenario_results: dict[str, dict[str, Any]] = {}
    for sc in scenarios:
        archive = SCENARIO_ARCHIVES[sc]
        ood_data = feature_dict[archive]
        ood_emb = np.asarray(ood_data["fused_embeddings"], dtype=np.float64)
        ood_emb_raw = np.asarray(ood_data.get("fused_embeddings_raw", ood_emb), dtype=np.float64)
        ood_logits = np.asarray(ood_data["logits"], dtype=np.float64)
        ood_vis = np.asarray(ood_data.get("visual_global_embeddings", ood_emb), dtype=np.float64)
        ood_txt = np.asarray(ood_data["text_global_embeddings"], dtype=np.float64) if "text_global_embeddings" in ood_data else None

        method_metrics: dict[str, Any] = {}
        for method in methods:
            if method == "mahalanobis_centroid":
                ood_scores = detector_raw.score(ood_emb_raw, method=method)
            elif method == "entropy":
                ood_scores = OODDetector.score_entropy(ood_logits)
            elif method == "knn":
                ood_scores = detector_norm.score(
                    ood_emb,
                    method="knn",
                    k=knn_k,
                    reduction=knn_reduction,
                    metric=knn_metric,
                )
            elif method == "multimodal_ensemble":
                ood_scores = multimodal_detector.score(
                    visual_embeddings=ood_vis,
                    text_embeddings=ood_txt,
                    logits=ood_logits,
                )
            else:
                ood_scores = detector_norm.score(ood_emb, method=method)

            in_scores = test_id_scores[method]
            threshold = thresholds[method]

            # AUROC, AUPR-Out và FPR@95%TPR được tính từ toàn bộ đường cong
            # ID/OOD. Ngưỡng hiệu chỉnh trên validation là ngưỡng vận hành
            # riêng và được lưu bên dưới, không phải đối số của evaluate_ood().
            eval_metrics = evaluate_ood(in_scores, ood_scores)
            bootstrap_res = bootstrap_ood_metrics(in_scores, ood_scores, n_bootstrap=500)

            method_metrics[method] = {
                "auroc_ood": float(eval_metrics["auroc_ood"]),
                "aupr_out": float(eval_metrics["aupr_out"]),
                "fpr_at_95tpr": float(eval_metrics["fpr_at_95tpr"]),
                "threshold": float(threshold),
                "bootstrap_ci": bootstrap_res,
            }
        scenario_results[sc] = method_metrics

    return scenario_results


def aggregate_seeds_results(
    seed_results: dict[int, dict[str, dict[str, Any]]],
    scenarios: list[str],
    methods: list[str],
) -> dict[str, dict[str, dict[str, float]]]:
    """Tổng hợp giá trị trung bình và độ lệch chuẩn (Mean ± Std) qua các hạt giống.

    Parameters
    ----------
    seed_results : dict[int, dict[str, dict[str, Any]]]
        Kết quả đánh giá chi tiết theo từng hạt giống.
    scenarios : list[str]
        Danh sách kịch bản OOD.
    methods : list[str]
        Danh sách phương pháp phát hiện OOD.

    Returns
    -------
    dict[str, dict[str, dict[str, float]]]
        Từ điển thống kê tổng hợp chứa mean và std cho từng độ đo.
    """
    aggregated: dict[str, dict[str, dict[str, float]]] = {}
    metric_keys = ["auroc_ood", "aupr_out", "fpr_at_95tpr"]

    for sc in scenarios:
        aggregated[sc] = {}
        for m in methods:
            aggregated[sc][m] = {}
            for metric in metric_keys:
                values = [seed_results[s][sc][m][metric] for s in seed_results]
                aggregated[sc][m][f"{metric}_mean"] = float(np.mean(values))
                aggregated[sc][m][f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0

    return aggregated


def run_experiment_ood_pipeline(
    experiment: str,
    seeds: list[int],
    scenarios: list[str],
    methods: list[str],
    *,
    device: torch.device,
    knn_k: int = 10,
    knn_reduction: str = "kth",
    knn_metric: str = "euclidean",
    target_fpr: float = 0.05,
    batch_size: int = 16,
    num_workers: int = 0,
    force_recompute: bool = False,
) -> dict[str, Any]:
    """Chạy toàn bộ quy trình đánh giá OOD cho một mô hình qua nhiều hạt giống.

    Parameters
    ----------
    experiment : str
        Định danh thí nghiệm mô hình.
    seeds : list[int]
        Danh sách hạt giống ngẫu nhiên.
    scenarios : list[str]
        Danh sách kịch bản OOD.
    methods : list[str]
        Danh sách phương pháp dò OOD.
    device : torch.device
        Thiết bị thực thi tính toán.
    knn_k : int, optional
        Tham số k trong kNN OOD, mặc định 5.
    target_fpr : float, optional
        Mức FPR mục tiêu trên tập validation, mặc định 0.05.
    batch_size : int, optional
        Kích thước batch trích xuất đặc trưng, mặc định 16.
    num_workers : int, optional
        Số luồng nạp dữ liệu, mặc định 0.
    force_recompute : bool, optional
        Buộc tính toán lại đặc trưng, mặc định False.

    Returns
    -------
    dict[str, Any]
        Từ điển chứa toàn bộ kết quả từng hạt giống và kết quả tổng hợp của mô hình.
    """
    seed_results: dict[int, dict[str, dict[str, Any]]] = {}

    for seed in seeds:
        feat_dict = extract_or_load_features(
            experiment,
            seed,
            scenarios,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            force_recompute=force_recompute,
        )
        res = evaluate_single_seed_ood(
            feat_dict,
            scenarios,
            methods,
            knn_k=knn_k,
            knn_reduction=knn_reduction,
            knn_metric=knn_metric,
            target_fpr=target_fpr,
        )
        seed_results[seed] = res

    aggregated = aggregate_seeds_results(seed_results, scenarios, methods)

    return {
        "experiment": experiment,
        "seeds": seeds,
        "scenarios": scenarios,
        "methods": methods,
        "target_fpr": target_fpr,
        "per_seed": seed_results,
        "aggregated": aggregated,
    }


def format_summary_table(
    all_experiment_results: list[dict[str, Any]],
    scenarios: list[str],
    methods: list[str],
) -> str:
    """Tạo bảng kết quả định dạng văn bản tổng hợp trực quan.

    Parameters
    ----------
    all_experiment_results : list[dict[str, Any]]
        Danh sách kết quả của các mô hình đã đánh giá.
    scenarios : list[str]
        Danh sách kịch bản OOD.
    methods : list[str]
        Danh sách phương pháp phát hiện OOD.

    Returns
    -------
    str
        Bảng văn bản đã được căn lề và định dạng.
    """
    rows = []

    for exp_res in all_experiment_results:
        exp_name = exp_res["experiment"]
        agg = exp_res["aggregated"]

        for sc in scenarios:
            sc_label = SCENARIO_LABELS.get(sc, sc)
            for m in methods:
                m_data = agg[sc][m]
                auroc_str = f"{m_data['auroc_ood_mean'] * 100:.2f} ± {m_data['auroc_ood_std'] * 100:.2f}"
                aupr_str = f"{m_data['aupr_out_mean'] * 100:.2f} ± {m_data['aupr_out_std'] * 100:.2f}"
                fpr_str = f"{m_data['fpr_at_95tpr_mean'] * 100:.2f} ± {m_data['fpr_at_95tpr_std'] * 100:.2f}"
                rows.append({
                    "Mô hình": exp_name,
                    "Kịch bản OOD": sc_label,
                    "Phương pháp": m,
                    "AUROC (%)": auroc_str,
                    "AUPR-Out (%)": aupr_str,
                    "FPR@95%TPR (%)": fpr_str,
                })

    df = pd.DataFrame(rows)
    return df.to_string(index=False)


def main() -> None:
    """Điểm vào chính của công cụ đánh giá OOD."""
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")

    experiments: list[str] = [str(e).strip() for e in args.experiments if str(e).strip()]
    if not experiments:
        experiments = [SOURCE_EXPERIMENT]

    print("=" * 80)
    print("  XBONE-NET: CHƯƠNG TRÌNH ĐÁNH GIÁ OUT-OF-DISTRIBUTION (OOD) TOÀN DIỆN")
    print(f"  • Thiết bị thực thi : {device}")
    print(f"  • Danh sách seeds   : {args.seeds}")
    print(f"  • Số lượng mô hình  : {len(experiments)}")
    print(f"  • Kịch bản OOD      : {', '.join(args.scenarios)}")
    print(f"  • Phương pháp dò    : {', '.join(args.methods)}")
    print("=" * 80)

    all_results: list[dict[str, Any]] = []

    for exp_name in experiments:
        print(f"\n>>> Đang đánh giá mô hình: {exp_name}")
        try:
            exp_res = run_experiment_ood_pipeline(
                exp_name,
                args.seeds,
                args.scenarios,
                args.methods,
                device=device,
                knn_k=args.knn_k,
                knn_reduction=args.knn_reduction,
                knn_metric=args.knn_metric,
                target_fpr=args.target_fpr,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                force_recompute=args.force_recompute,
            )
            all_results.append(exp_res)
        except Exception as e:
            print(f"  [LỖI] Không thể đánh giá {exp_name}: {e}")
            import traceback
            traceback.print_exc()

    if not all_results:
        print("\n[Cảnh báo] Không có kết quả nào được tạo thành công.")
        return

    # In bảng tổng hợp
    table_str = format_summary_table(all_results, args.scenarios, args.methods)
    print("\n" + table_str + "\n")

    # Xuất báo cáo ra file
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "ood_benchmark_summary.json"
    csv_path = args.output_dir / "ood_benchmark_summary.csv"
    md_path = args.output_dir / "ood_benchmark_summary.md"

    # 1. JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # 2. Markdown
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# BÁO CÁO ĐÁNH GIÁ OUT-OF-DISTRIBUTION (OOD)\n\n")
        f.write(f"Thời gian xuất: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(table_str)
        f.write("\n")

    # 3. CSV
    flat_rows = []
    for exp_res in all_results:
        exp_name = exp_res["experiment"]
        agg = exp_res["aggregated"]
        for sc in args.scenarios:
            for m in args.methods:
                m_data = agg[sc][m]
                flat_rows.append({
                    "experiment": exp_name,
                    "scenario": sc,
                    "method": m,
                    "auroc_mean": m_data["auroc_ood_mean"],
                    "auroc_std": m_data["auroc_ood_std"],
                    "aupr_out_mean": m_data["aupr_out_mean"],
                    "aupr_out_std": m_data["aupr_out_std"],
                    "fpr_at_95tpr_mean": m_data["fpr_at_95tpr_mean"],
                    "fpr_at_95tpr_std": m_data["fpr_at_95tpr_std"],
                })
    pd.DataFrame(flat_rows).to_csv(csv_path, index=False, encoding="utf-8-sig")

    print(f"[Thành công] Đã xuất báo cáo JSON tại : {json_path}")
    print(f"[Thành công] Đã xuất bảng Markdown tại: {md_path}")
    print(f"[Thành công] Đã xuất bảng CSV tại     : {csv_path}\n")


if __name__ == "__main__":
    main()
