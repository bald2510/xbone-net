"""So sánh embedding CTCH của XBone-Net hai pha và ablation Phase-2-only.

Biểu đồ được tạo từ cache đặc trưng đã trích xuất bằng checkpoint đánh giá. Trước
khi sử dụng, công cụ đối chiếu SHA-256 trong provenance với checkpoint hiện tại,
căn chỉnh chính xác ``image_id``/nhãn giữa hai mô hình, rồi tính t-SNE và các chỉ
số phân tách lớp trên cùng tập test.

Notes
-----
t-SNE chỉ là phép chiếu trực quan. Kết luận định lượng nên dựa thêm trên metric
phân loại và các phép thử thống kê ghép cặp, không dựa riêng vào khoảng cách 2D.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    silhouette_score,
)
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "results" / "summary" / "ablation" / "embedding_comparison"


@dataclass(frozen=True)
class ExperimentSpec:
    """Đường dẫn và nhãn hiển thị của một thí nghiệm."""

    key: str
    display_name: str
    experiment: str

    @property
    def result_root(self) -> Path:
        """Trả về thư mục kết quả của thí nghiệm."""
        return ROOT / "results" / self.experiment

    @property
    def checkpoint_root(self) -> Path:
        """Trả về thư mục checkpoint của thí nghiệm."""
        return ROOT / "checkpoints" / self.experiment


EXPERIMENTS = (
    ExperimentSpec(
        key="ours_xbone_net",
        display_name="Ours_xbone_net (2 pha)",
        experiment="ctch/proposed/ours_xbone_net",
    ),
    ExperimentSpec(
        key="phase2_only",
        display_name="Phase2_only",
        experiment="ctch/ablation_study/architecture/phase/phase2_only",
    ),
)


def _parse_args() -> argparse.Namespace:
    """Đọc tham số dòng lệnh."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--feature-key", default="fused_embeddings")
    parser.add_argument("--projection", choices=("tsne", "pca"), default="tsne")
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--tsne-iterations", type=int, default=1500)
    parser.add_argument("--neighbors", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args()
    if not args.seeds:
        parser.error("--seeds phải chứa ít nhất một seed.")
    if args.perplexity <= 0:
        parser.error("--perplexity phải dương.")
    if args.tsne_iterations < 250:
        parser.error("--tsne-iterations phải >= 250.")
    if args.neighbors < 1:
        parser.error("--neighbors phải dương.")
    if args.dpi < 72:
        parser.error("--dpi phải >= 72.")
    return args


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Tính SHA-256 của một tệp."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _feature_path(spec: ExperimentSpec, seed: int, split: str) -> Path:
    """Trả về cache đặc trưng của split cần so sánh."""
    return (
        spec.result_root
        / f"seed_{seed}"
        / "analysis"
        / "features"
        / f"ctch_{split}.npz"
    )


def _load_archive(
    spec: ExperimentSpec,
    seed: int,
    split: str,
    feature_key: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Nạp cache và xác thực checkpoint/provenance bắt buộc."""
    archive_path = _feature_path(spec, seed, split)
    if not archive_path.is_file():
        raise FileNotFoundError(
            f"Thiếu {archive_path}. Hãy chạy tools/export_analysis_features.py trước."
        )
    with np.load(archive_path, allow_pickle=False) as archive:
        if "provenance_json" not in archive.files:
            raise ValueError(f"Cache không có provenance_json: {archive_path}")
        provenance = json.loads(str(archive["provenance_json"].item()))
        arrays = {
            key: np.asarray(archive[key])
            for key in archive.files
            if key != "provenance_json"
        }

    required = {feature_key, "labels", "predictions", "image_id"}
    missing = required.difference(arrays)
    if missing:
        raise ValueError(f"Cache {archive_path} thiếu: {sorted(missing)}")
    if provenance.get("source_experiment") != spec.experiment:
        raise ValueError(
            f"Sai source_experiment trong {archive_path}: "
            f"{provenance.get('source_experiment')!r}"
        )
    if int(provenance.get("seed", -1)) != seed:
        raise ValueError(f"Sai seed trong provenance của {archive_path}")

    checkpoint_path = spec.checkpoint_root / f"seed_{seed}" / "best_phase2.pth"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Thiếu checkpoint: {checkpoint_path}")
    checkpoint_sha256 = _sha256(checkpoint_path)
    if checkpoint_sha256 != provenance.get("checkpoint_sha256"):
        raise ValueError(
            "Cache không thuộc checkpoint hiện tại: "
            f"{archive_path} (cache={provenance.get('checkpoint_sha256')}, "
            f"checkpoint={checkpoint_sha256})"
        )
    provenance = {
        **provenance,
        "validated_checkpoint": str(checkpoint_path.resolve()),
        "validated_checkpoint_sha256": checkpoint_sha256,
    }
    return arrays, provenance


def _validate_alignment(
    reference: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
    context: str,
) -> None:
    """Bảo đảm hai mô hình dùng đúng cùng mẫu và nhãn theo cùng thứ tự."""
    for key in ("image_id", "labels"):
        if not np.array_equal(reference[key], candidate[key]):
            raise ValueError(f"Không căn chỉnh {key} tại {context}.")


def _metric_file_values(spec: ExperimentSpec, seed: int) -> dict[str, float]:
    """Đọc ba metric chính từ metrics.json để phát hiện artifact lệch nhau."""
    path = spec.result_root / f"seed_{seed}" / "metrics.json"
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload.get("metrics", {})
    return {
        "accuracy": float(metrics["accuracy"]),
        "balanced_accuracy": float(metrics["balanced_accuracy"]),
        "macro_f1": float(metrics["f1_macro"]),
    }


def _classification_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    """Tính metric trực tiếp từ dự đoán đi kèm embedding."""
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        ),
    }


def _embedding_metrics(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    neighbors: int,
) -> dict[str, float]:
    """Định lượng độ gom cụm và khả năng phân loại cục bộ của embedding."""
    k = min(neighbors, len(test_labels) - 1, len(train_labels))
    if k < 1:
        raise ValueError("Không đủ mẫu để tính metric láng giềng.")

    test_neighbor_model = NearestNeighbors(n_neighbors=k + 1, metric="cosine")
    test_neighbor_model.fit(test_features)
    neighbor_indices = test_neighbor_model.kneighbors(
        test_features, return_distance=False
    )[:, 1:]
    neighbor_purity = np.mean(test_labels[neighbor_indices] == test_labels[:, None])

    probe = KNeighborsClassifier(
        n_neighbors=k,
        metric="cosine",
        weights="distance",
        n_jobs=-1,
    )
    probe.fit(train_features, train_labels)
    probe_predictions = probe.predict(test_features)
    return {
        "silhouette_cosine": float(
            silhouette_score(test_features, test_labels, metric="cosine")
        ),
        "neighbor_purity": float(neighbor_purity),
        "knn_probe_accuracy": float(accuracy_score(test_labels, probe_predictions)),
        "knn_probe_balanced_accuracy": float(
            balanced_accuracy_score(test_labels, probe_predictions)
        ),
        "knn_probe_macro_f1": float(
            f1_score(test_labels, probe_predictions, average="macro", zero_division=0)
        ),
    }


def _project(
    features: np.ndarray,
    method: str,
    seed: int,
    perplexity: float,
    iterations: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Chiếu embedding về 2D bằng PCA hoặc PCA-50 + t-SNE cosine."""
    features = np.asarray(features, dtype=np.float64)
    if method == "pca":
        reducer = PCA(n_components=2, svd_solver="full")
        points = reducer.fit_transform(features)
        return points, {
            "method": "PCA",
            "explained_variance_ratio": reducer.explained_variance_ratio_.tolist(),
        }

    pca_components = min(50, features.shape[1], features.shape[0] - 1)
    pre_reducer = PCA(n_components=pca_components, svd_solver="randomized", random_state=seed)
    reduced = pre_reducer.fit_transform(features)
    effective_perplexity = min(perplexity, max(5.0, (len(features) - 1) / 3.0))
    reducer = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        metric="cosine",
        init="pca",
        learning_rate="auto",
        max_iter=iterations,
        random_state=seed,
    )
    points = reducer.fit_transform(reduced)
    return points, {
        "method": "PCA-50 + t-SNE",
        "metric": "cosine",
        "perplexity": effective_perplexity,
        "iterations": iterations,
        "kl_divergence": float(reducer.kl_divergence_),
        "pca_components": pca_components,
        "pca_explained_variance": float(pre_reducer.explained_variance_ratio_.sum()),
    }


def _class_names(provenance: dict[str, Any], labels: np.ndarray) -> list[str]:
    """Lấy tên lớp CTCH từ resolved config trong provenance."""
    names = (
        provenance.get("resolved_config", {})
        .get("dataset", {})
        .get("params", {})
        .get("classes", [])
    )
    num_classes = int(np.max(labels)) + 1
    if len(names) != num_classes:
        return [f"Lớp {index}" for index in range(num_classes)]
    return [str(name) for name in names]


def _plot(
    records: list[dict[str, Any]],
    class_names: list[str],
    output: Path,
    projection_name: str,
    dpi: int,
) -> None:
    """Vẽ small multiples theo seed và mô hình."""
    seeds = sorted({int(record["seed"]) for record in records})
    by_key = {(record["seed"], record["model_key"]): record for record in records}
    fig, axes = plt.subplots(
        len(seeds),
        len(EXPERIMENTS),
        figsize=(16, max(5.0, 4.6 * len(seeds))),
        squeeze=False,
    )
    palette = list(plt.get_cmap("tab20").colors) + list(plt.get_cmap("Set3").colors)

    for row, seed in enumerate(seeds):
        row_records = [by_key[(seed, spec.key)] for spec in EXPERIMENTS]
        x_values = np.concatenate([record["projection"][:, 0] for record in row_records])
        y_values = np.concatenate([record["projection"][:, 1] for record in row_records])
        x_pad = max(1e-6, 0.04 * float(np.ptp(x_values)))
        y_pad = max(1e-6, 0.04 * float(np.ptp(y_values)))

        for column, spec in enumerate(EXPERIMENTS):
            record = by_key[(seed, spec.key)]
            axis = axes[row, column]
            points = record["projection"]
            labels = record["labels"]
            predictions = record["predictions"]
            correct = labels == predictions
            for class_id in range(len(class_names)):
                class_mask = labels == class_id
                color = palette[class_id % len(palette)]
                axis.scatter(
                    points[class_mask & correct, 0],
                    points[class_mask & correct, 1],
                    s=15,
                    color=color,
                    marker="o",
                    alpha=0.72,
                    linewidths=0,
                    rasterized=True,
                )
                axis.scatter(
                    points[class_mask & ~correct, 0],
                    points[class_mask & ~correct, 1],
                    s=26,
                    color=color,
                    marker="x",
                    alpha=0.9,
                    linewidths=0.8,
                    rasterized=True,
                )
            metrics = record["metrics"]
            axis.set_title(
                f"{spec.display_name} — seed {seed}\n"
                f"Acc {metrics['accuracy']:.3f} | BAcc {metrics['balanced_accuracy']:.3f} | "
                f"Macro-F1 {metrics['macro_f1']:.3f}",
                fontsize=11,
            )
            axis.text(
                0.01,
                0.01,
                f"Silhouette {metrics['silhouette_cosine']:.3f} · "
                f"{metrics['neighbors']}-NN probe BAcc "
                f"{metrics['knn_probe_balanced_accuracy']:.3f}",
                transform=axis.transAxes,
                ha="left",
                va="bottom",
                fontsize=8.5,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 2.5},
            )
            axis.set_xlim(float(x_values.min() - x_pad), float(x_values.max() + x_pad))
            axis.set_ylim(float(y_values.min() - y_pad), float(y_values.max() + y_pad))
            axis.set_xlabel(f"{projection_name} 1")
            axis.set_ylabel(f"{projection_name} 2")
            axis.grid(alpha=0.16, linewidth=0.5)
            axis.tick_params(labelsize=8)

    class_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=palette[class_id % len(palette)],
            markeredgecolor="none",
            markersize=6,
            label=f"{class_id}: {name}",
        )
        for class_id, name in enumerate(class_names)
    ]
    status_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="0.45", markersize=6, label="Dự đoán đúng"),
        Line2D([0], [0], marker="x", color="0.35", linestyle="none", markersize=6, label="Dự đoán sai"),
    ]
    fig.legend(
        handles=class_handles + status_handles,
        loc="lower center",
        ncol=4,
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.5, 0.005),
    )
    fig.suptitle(
        "CTCH test: embedding dung hợp từ checkpoint — hai pha so với Phase2_only",
        fontsize=15,
        y=0.995,
    )
    fig.text(
        0.5,
        0.974,
        "Cùng 669 mẫu/22 lớp; dấu × là dự đoán sai. t-SNE dùng để quan sát, không phải kiểm định hiệu năng.",
        ha="center",
        va="top",
        fontsize=9,
    )
    fig.subplots_adjust(top=0.915, bottom=0.165, hspace=0.34, wspace=0.14)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _write_csv(records: list[dict[str, Any]], output: Path) -> None:
    """Ghi bảng metric mỗi seed."""
    metric_names = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "silhouette_cosine",
        "neighbor_purity",
        "knn_probe_accuracy",
        "knn_probe_balanced_accuracy",
        "knn_probe_macro_f1",
        "metrics_json_matches_cache",
    ]
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["model", "seed", "n_samples", "feature_dim", *metric_names],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "model": record["model_key"],
                    "seed": record["seed"],
                    "n_samples": len(record["labels"]),
                    "feature_dim": record["features"].shape[1],
                    **{name: record["metrics"][name] for name in metric_names},
                }
            )


def _aggregate_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Tổng hợp trung bình ba seed và chênh lệch Ours trừ Phase2_only."""
    metric_names = (
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "silhouette_cosine",
        "neighbor_purity",
        "knn_probe_accuracy",
        "knn_probe_balanced_accuracy",
        "knn_probe_macro_f1",
    )
    by_model: dict[str, dict[str, float]] = {}
    for spec in EXPERIMENTS:
        model_records = [record for record in records if record["model_key"] == spec.key]
        by_model[spec.key] = {
            name: float(np.mean([record["metrics"][name] for record in model_records]))
            for name in metric_names
        }
    delta = {
        name: by_model[EXPERIMENTS[0].key][name] - by_model[EXPERIMENTS[1].key][name]
        for name in metric_names
    }
    return {"mean_by_model": by_model, "delta_ours_minus_phase2_only": delta}


def _write_aggregate_csv(aggregate: dict[str, Any], output: Path) -> None:
    """Ghi bảng trung bình và chênh lệch để dùng trực tiếp trong báo cáo."""
    means = aggregate["mean_by_model"]
    delta = aggregate["delta_ours_minus_phase2_only"]
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "metric",
                "ours_xbone_net_mean",
                "phase2_only_mean",
                "delta_ours_minus_phase2_only",
            ),
        )
        writer.writeheader()
        for metric, difference in delta.items():
            writer.writerow(
                {
                    "metric": metric,
                    "ours_xbone_net_mean": means["ours_xbone_net"][metric],
                    "phase2_only_mean": means["phase2_only"][metric],
                    "delta_ours_minus_phase2_only": difference,
                }
            )


def _jsonable_record(record: dict[str, Any]) -> dict[str, Any]:
    """Chuyển record nội bộ thành dữ liệu JSON gọn cho QA/tái sử dụng."""
    return {
        "model_key": record["model_key"],
        "display_name": record["display_name"],
        "experiment": record["experiment"],
        "seed": record["seed"],
        "checkpoint": record["provenance"]["validated_checkpoint"],
        "checkpoint_sha256": record["provenance"]["validated_checkpoint_sha256"],
        "feature_archive": record["feature_archive"],
        "feature_key": record["feature_key"],
        "n_samples": len(record["labels"]),
        "feature_dim": int(record["features"].shape[1]),
        "metrics": record["metrics"],
        "projection_metadata": record["projection_metadata"],
    }


def main() -> None:
    """Chạy toàn bộ quy trình so sánh embedding."""
    args = _parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    class_names: list[str] | None = None
    global_reference: dict[str, np.ndarray] | None = None
    for seed in args.seeds:
        seed_reference: dict[str, np.ndarray] | None = None
        for spec in EXPERIMENTS:
            test_arrays, provenance = _load_archive(
                spec, seed, args.split, args.feature_key
            )
            train_arrays, _ = _load_archive(spec, seed, "train", args.feature_key)
            features = np.asarray(test_arrays[args.feature_key], dtype=np.float64)
            labels = np.asarray(test_arrays["labels"], dtype=np.int64)
            predictions = np.asarray(test_arrays["predictions"], dtype=np.int64)
            if features.ndim != 2 or len(features) != len(labels):
                raise ValueError(f"Sai shape embedding tại {spec.key}, seed={seed}.")
            if seed_reference is None:
                seed_reference = test_arrays
            else:
                _validate_alignment(seed_reference, test_arrays, f"seed={seed}")
            if global_reference is None:
                global_reference = test_arrays
            else:
                _validate_alignment(global_reference, test_arrays, f"{spec.key}, seed={seed}")

            direct_metrics = _classification_metrics(labels, predictions)
            file_metrics = _metric_file_values(spec, seed)
            metrics_match = bool(file_metrics) and all(
                np.isclose(direct_metrics[key], file_metrics[key], atol=1e-10, rtol=0.0)
                for key in direct_metrics
            )
            metrics = {
                **direct_metrics,
                **_embedding_metrics(
                    np.asarray(train_arrays[args.feature_key], dtype=np.float64),
                    np.asarray(train_arrays["labels"], dtype=np.int64),
                    features,
                    labels,
                    args.neighbors,
                ),
                "neighbors": int(args.neighbors),
                "metrics_json_matches_cache": metrics_match,
                "metrics_json": file_metrics,
            }
            projection, projection_metadata = _project(
                features,
                args.projection,
                seed,
                args.perplexity,
                args.tsne_iterations,
            )
            record = {
                "model_key": spec.key,
                "display_name": spec.display_name,
                "experiment": spec.experiment,
                "seed": seed,
                "features": features,
                "labels": labels,
                "predictions": predictions,
                "image_id": np.asarray(test_arrays["image_id"]).astype(str),
                "projection": projection,
                "projection_metadata": projection_metadata,
                "metrics": metrics,
                "provenance": provenance,
                "feature_archive": str(_feature_path(spec, seed, args.split).resolve()),
                "feature_key": args.feature_key,
            }
            records.append(record)
            if class_names is None:
                class_names = _class_names(provenance, labels)

    assert class_names is not None
    stem = f"ctch_{args.split}_{args.feature_key}_{args.projection}"
    figure_path = output_dir / f"{stem}.png"
    projection_label = "t-SNE" if args.projection == "tsne" else "PCA"
    _plot(records, class_names, figure_path, projection_label, args.dpi)
    _write_csv(records, output_dir / f"{stem}_metrics.csv")
    aggregate = _aggregate_metrics(records)
    _write_aggregate_csv(aggregate, output_dir / f"{stem}_aggregate_metrics.csv")

    summary = {
        "dataset": "CTCH",
        "split": args.split,
        "seeds": args.seeds,
        "class_names": class_names,
        "projection": args.projection,
        "feature_key": args.feature_key,
        "caution": (
            "t-SNE/PCA is descriptive only. Use paired classification statistics "
            "for claims about model superiority."
        ),
        "aggregate": aggregate,
        "records": [_jsonable_record(record) for record in records],
    }
    (output_dir / f"{stem}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    projection_payload = {
        "class_names": class_names,
        "records": [
            {
                "model_key": record["model_key"],
                "display_name": record["display_name"],
                "seed": record["seed"],
                "metrics": record["metrics"],
                "points": [
                    {
                        "x": round(float(point[0]), 5),
                        "y": round(float(point[1]), 5),
                        "label": int(label),
                        "prediction": int(prediction),
                        "image_id": image_id,
                    }
                    for point, label, prediction, image_id in zip(
                        record["projection"],
                        record["labels"],
                        record["predictions"],
                        record["image_id"],
                        strict=True,
                    )
                ],
            }
            for record in records
        ],
    }
    (output_dir / f"{stem}_points.json").write_text(
        json.dumps(projection_payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    print(f"Đã lưu hình: {figure_path}")
    print(f"Đã lưu metric: {output_dir / f'{stem}_metrics.csv'}")
    for record in records:
        metrics = record["metrics"]
        print(
            f"{record['model_key']} seed={record['seed']}: "
            f"BAcc={metrics['balanced_accuracy']:.4f}, "
            f"Macro-F1={metrics['macro_f1']:.4f}, "
            f"silhouette={metrics['silhouette_cosine']:.4f}, "
            f"{args.neighbors}-NN probe BAcc={metrics['knn_probe_balanced_accuracy']:.4f}, "
            f"metrics.json match={metrics['metrics_json_matches_cache']}"
        )


if __name__ == "__main__":
    if sys.platform == "win32":
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    main()
