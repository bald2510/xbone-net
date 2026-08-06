"""Visualize Proposed-v3 split embeddings and empirical classifier centroids.

The script rebuilds the evaluated model from ``metrics.json``, loads its exact
Phase-2 checkpoint, extracts fused embeddings from a selected CTCH split, and
projects samples and classifier centroids into the same cosine-normalized space.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

from src.datasets.ctch import CTCHDataset  # noqa: E402
from src.utils.analysis import (  # noqa: E402
    MetadataDataset,
    build_analysis_loader,
    collect_feature_batches,
    load_proposed_experiment_model,
)
from src.utils.centroid_visualization import (  # noqa: E402
    compute_centroid_diagnostics,
    l2_normalize,
    load_centroids_from_checkpoint,
    load_fused_embeddings,
    project_cosine_space,
    stratified_subsample,
)


DEFAULT_EXPERIMENT = "ctch/proposed/ours_xbone_net"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        default=DEFAULT_EXPERIMENT,
        help="Evaluated CTCH proposed experiment (default: proposed_v3).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default="train",
        help="CTCH split whose fused embeddings are projected.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device used to extract embeddings from the selected split.",
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=None,
        help=(
            "Optional existing NPZ with fused_embeddings and labels. When omitted, "
            "the exact selected-split embeddings are extracted and cached automatically."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-points-per-class", type=int, default=1000)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--overwrite-embeddings",
        action="store_true",
        help="Re-extract the default selected-split embedding cache.",
    )
    parser.add_argument(
        "--ood-split",
        type=str,
        default=None,
        help="Optional OOD split name to plot a sample from (e.g., ctch_ood, btxrd_test).",
    )
    parser.add_argument(
        "--ood-index",
        type=int,
        default=0,
        help="Index of the OOD sample to plot from the OOD split.",
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive.")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative.")
    if args.max_points_per_class < 1:
        parser.error("--max-points-per-class must be positive.")
    if args.dpi < 72:
        parser.error("--dpi must be at least 72.")
    return args


def _resolve_path(path: Path) -> Path:
    return (path if path.is_absolute() else ROOT / path).resolve()


def _select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable.")
    return torch.device(name)


def _plain_dict(value: Any) -> dict[str, Any]:
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a configuration mapping, got {type(value).__name__}.")
    return dict(value)


def _display_class_names(cfg: Any, num_classes: int) -> list[str]:
    """Prefer the UTF-8 source labels when old metrics contain mojibake."""
    dataset_cfg_path = ROOT / "configs" / "dataset" / "ctch.yaml"
    candidates: list[Any] = []
    if dataset_cfg_path.is_file():
        source_cfg = OmegaConf.load(dataset_cfg_path)
        candidates.append(source_cfg.get("params", {}).get("classes", []))
    candidates.append(cfg.dataset.params.get("classes", []))
    for values in candidates:
        names = [str(value) for value in values]
        if len(names) == num_classes:
            return names
    return [f"Class {class_id}" for class_id in range(num_classes)]


def _cache_scalar(archive: Any, key: str) -> str | None:
    if key not in archive:
        return None
    return str(np.asarray(archive[key]).reshape(-1)[0])


def _validate_embedding_cache(
    cache_path: Path,
    experiment: str,
    seed: int,
    checkpoint_sha256: str,
    split: str,
) -> None:
    with np.load(cache_path, allow_pickle=False) as archive:
        actual = {
            "source_experiment": _cache_scalar(archive, "source_experiment"),
            "seed": _cache_scalar(archive, "seed"),
            "checkpoint_sha256": _cache_scalar(archive, "checkpoint_sha256"),
            "split": _cache_scalar(archive, "split"),
        }
    expected = {
        "source_experiment": experiment,
        "seed": str(seed),
        "checkpoint_sha256": checkpoint_sha256,
        "split": split,
    }
    mismatches = [
        f"{key}={actual[key]!r} (expected {value!r})"
        for key, value in expected.items()
        if actual[key] != value
    ]
    if mismatches:
        raise ValueError(
            f"Embedding cache does not belong to the requested checkpoint: {cache_path}. "
            + "; ".join(mismatches)
            + ". Use --overwrite-embeddings to rebuild it."
        )


def _extract_split_embeddings(
    loaded: Any,
    cache_path: Path,
    batch_size: int,
    num_workers: int,
    split: str,
) -> tuple[np.ndarray, np.ndarray, str]:
    params = _plain_dict(loaded.cfg.dataset.params)
    dataset = CTCHDataset(
        split=split,
        transform=loaded.model.backbone.preprocess,
        tokenizer=loaded.model.backbone.tokenizer_obj,
        **params,
    )
    wrapped = MetadataDataset(dataset, f"ctch_{split}")
    loader = build_analysis_loader(
        wrapped,
        tokenizer=loaded.model.backbone.tokenizer_obj,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    report_type = str(loaded.cfg.params.phase2.get("p2_report_type", "clinical"))
    print(
        f"Extracting fused embeddings for {len(dataset)} CTCH {split} samples "
        f"on {loaded.device}..."
    )
    arrays = collect_feature_batches(
        loaded.model,
        loader,
        device=loaded.device,
        report_type=report_type,
    )
    embeddings = np.asarray(arrays["fused_embeddings"], dtype=np.float32)
    labels = np.asarray(arrays["labels"])
    if labels.ndim == 2:
        labels = labels.argmax(axis=1)
    labels = labels.reshape(-1).astype(np.int64, copy=False)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        fused_embeddings=embeddings,
        labels=labels,
        predictions=np.asarray(arrays["predictions"], dtype=np.int64),
        image_id=np.asarray(arrays.get("image_id", []), dtype=str),
        source_experiment=np.asarray(loaded.provenance["source_experiment"]),
        seed=np.asarray(int(loaded.provenance["seed"]), dtype=np.int64),
        checkpoint_sha256=np.asarray(loaded.checkpoint_sha256),
        checkpoint=np.asarray(str(loaded.checkpoint)),
        representation=np.asarray("phase2_fused_embedding"),
        split=np.asarray(split),
    )
    print(f"Cached {split} embeddings: {cache_path}")
    return embeddings, labels, "fused_embeddings"


def _load_or_extract_embeddings(
    args: argparse.Namespace,
    loaded: Any,
    output_dir: Path,
) -> tuple[np.ndarray, np.ndarray, str, Path]:
    if args.embeddings is not None:
        path = _resolve_path(args.embeddings)
        embeddings, labels, key = load_fused_embeddings(path)
        print(f"Loaded external embeddings: {path}")
        return embeddings, labels, key, path

    cache_path = output_dir / f"{args.split}_fused_embeddings.npz"
    if cache_path.is_file() and not args.overwrite_embeddings:
        _validate_embedding_cache(
            cache_path,
            experiment=args.experiment.strip("/"),
            seed=args.seed,
            checkpoint_sha256=loaded.checkpoint_sha256,
            split=args.split,
        )
        embeddings, labels, key = load_fused_embeddings(cache_path)
        print(f"Loaded cached {args.split} embeddings: {cache_path}")
        return embeddings, labels, key, cache_path
    return (
        *_extract_split_embeddings(
            loaded,
            cache_path=cache_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            split=args.split,
        ),
        cache_path,
    )


def _class_colors(num_classes: int) -> np.ndarray:
    cmap_name = "tab20" if num_classes <= 20 else "turbo"
    cmap = plt.get_cmap(cmap_name, num_classes)
    return np.asarray([cmap(index) for index in range(num_classes)])


def _centroid_split_alignment(
    centroids: np.ndarray,
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> dict[str, np.ndarray]:
    num_classes = centroids.shape[0]
    normalized_centroids = l2_normalize(centroids)
    normalized_embeddings = l2_normalize(embeddings)
    counts = np.bincount(labels, minlength=num_classes).astype(np.int64)
    split_means = np.zeros_like(normalized_centroids)
    cosine = np.full(num_classes, np.nan, dtype=np.float32)
    for class_id in range(num_classes):
        mask = labels == class_id
        if not np.any(mask):
            continue
        split_means[class_id] = normalized_embeddings[mask].mean(axis=0)
        split_means[class_id] = l2_normalize(split_means[class_id : class_id + 1])[0]
        cosine[class_id] = float(
            normalized_centroids[class_id] @ split_means[class_id]
        )
    return {
        "selected_split_counts": counts,
        "split_mean_directions": split_means,
        "checkpoint_vs_split_mean_cosine": cosine,
    }


def _plot_projection(
    centroid_projection: np.ndarray,
    sample_projection: np.ndarray,
    sample_labels: np.ndarray,
    class_names: list[str],
    counts: np.ndarray,
    explained_variance: np.ndarray,
    output: Path,
    dpi: int,
    split: str,
    ood_projection: np.ndarray | None = None,
) -> None:
    colors = _class_colors(len(class_names))
    fig, axis = plt.subplots(figsize=(16, 10))
    for class_id in range(len(class_names)):
        mask = sample_labels == class_id
        if np.any(mask):
            axis.scatter(
                sample_projection[mask, 0],
                sample_projection[mask, 1],
                s=16,
                alpha=0.25,
                color="0.8" if ood_projection is not None else colors[class_id],
                edgecolors="none",
                rasterized=True,
            )

    marker_sizes = 180.0 + 40.0 * np.log1p(
        counts / max(float(counts.min()), 1.0)
    )
    for class_id, (x_value, y_value) in enumerate(centroid_projection):
        axis.scatter(
            x_value,
            y_value,
            s=float(marker_sizes[class_id]),
            marker="X",
            color=colors[class_id],
            edgecolor="black",
            linewidth=1.0,
            zorder=5,
        )
        axis.annotate(
            str(class_id),
            (x_value, y_value),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            weight="bold",
        )

    split_vn = "huấn luyện" if split == "train" else "kiểm thử"
    variance = 100.0 * explained_variance
    axis.set_xlabel(f"PCA 1 ({variance[0]:.1f}% phương sai)")
    axis.set_ylabel(f"PCA 2 ({variance[1]:.1f}% phương sai)")
    axis.set_title(
        f"XBone-Net: Không gian biểu diễn đặc trưng (tập {split_vn})"
    )
    axis.axhline(0.0, color="0.85", linewidth=0.8)
    axis.axvline(0.0, color="0.85", linewidth=0.8)
    axis.grid(alpha=0.16)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="0.45",
            markeredgecolor="none",
            alpha=0.45,
            label=f"Mẫu dữ liệu",
        ),
        Line2D(
            [0],
            [0],
            marker="X",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=10,
            label="Vector nguyên mẫu",
        ),
    ]

    if ood_projection is not None:
        axis.scatter(
            ood_projection[:, 0],
            ood_projection[:, 1],
            s=400,
            marker="*",
            color="red",
            edgecolor="black",
            linewidth=1.5,
            zorder=10,
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="*",
                linestyle="none",
                markerfacecolor="red",
                markeredgecolor="black",
                markersize=12,
                label="Mẫu OOD",
            )
        )

    type_legend = axis.legend(
        handles=legend_handles,
        loc="upper left",
        frameon=True,
    )
    axis.add_artist(type_legend)
    class_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=colors[class_id],
            markeredgecolor="none",
            label=f"{class_id}: {class_name}",
        )
        for class_id, class_name in enumerate(class_names)
    ]
    axis.legend(
        handles=class_handles,
        title="Các lớp bệnh CTCH",
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=8,
        title_fontsize=9,
        frameon=True,
    )
    fig.tight_layout()
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_similarity_heatmap(
    similarity: np.ndarray,
    class_names: list[str],
    output: Path,
    dpi: int,
) -> None:
    num_classes = len(class_names)
    size = min(18.0, max(8.0, num_classes * 0.42))
    fig, axis = plt.subplots(figsize=(size, size))
    image = axis.imshow(similarity, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    labels = [f"{index}: {name}" for index, name in enumerate(class_names)]
    axis.set_xticks(np.arange(num_classes), labels=labels, rotation=90)
    axis.set_yticks(np.arange(num_classes), labels=labels)
    axis.tick_params(labelsize=7)
    axis.set_xlabel("Centroid class")
    axis.set_ylabel("Centroid class")
    axis.set_title("Pairwise centroid cosine similarity")
    colorbar = fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Cosine similarity")
    fig.tight_layout()
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_diagnostics(
    diagnostics: dict[str, Any],
    alignment: dict[str, np.ndarray],
    class_names: list[str],
    output: Path,
    dpi: int,
    split: str,
) -> None:
    num_classes = len(class_names)
    labels = [f"{index}: {name}" for index, name in enumerate(class_names)]
    y_positions = np.arange(num_classes)
    height = max(7.0, min(18.0, 0.36 * num_classes + 2.0))
    fig, axes = plt.subplots(1, 3, figsize=(20, height), sharey=True)
    per_class = diagnostics["per_class"]
    margins = np.asarray(
        [
            np.nan if item["mean_cosine_margin"] is None else item["mean_cosine_margin"]
            for item in per_class
        ],
        dtype=float,
    )
    accuracies = np.asarray(
        [
            np.nan
            if item["nearest_centroid_accuracy"] is None
            else item["nearest_centroid_accuracy"]
            for item in per_class
        ],
        dtype=float,
    )
    alignment_cosine = alignment["checkpoint_vs_split_mean_cosine"]
    bar_colors = [
        "#b2182b" if value < 0 else "#2166ac"
        for value in np.nan_to_num(margins)
    ]
    axes[0].barh(y_positions, np.nan_to_num(margins), color=bar_colors)
    axes[0].axvline(0.0, color="black", linewidth=0.9)
    axes[0].set_xlabel("Mean own-vs-nearest-other cosine margin")
    axes[0].set_title("Class separation")
    axes[1].barh(y_positions, np.nan_to_num(accuracies), color="#4d9221")
    axes[1].set_xlim(0.0, 1.0)
    axes[1].set_xlabel("Nearest-centroid accuracy")
    axes[1].set_title(f"{split.capitalize()}-set agreement")
    axes[2].barh(y_positions, np.nan_to_num(alignment_cosine), color="#762a83")
    axes[2].set_xlim(0.0, 1.0)
    axes[2].set_xlabel("Cosine similarity")
    axes[2].set_title(f"Checkpoint centroid vs. {split} mean")
    axes[0].set_yticks(y_positions, labels=labels)
    axes[0].invert_yaxis()
    for axis in axes:
        axis.grid(axis="x", alpha=0.2)
        axis.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _write_metrics_csv(
    path: Path,
    class_names: list[str],
    checkpoint_counts: np.ndarray,
    diagnostics: dict[str, Any],
    alignment: dict[str, np.ndarray],
    split: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    split_counts = alignment["selected_split_counts"]
    alignment_cosine = alignment["checkpoint_vs_split_mean_cosine"]
    for class_id, class_name in enumerate(class_names):
        nearest_id = int(diagnostics["nearest_centroid_ids"][class_id])
        per_class = diagnostics["per_class"][class_id]
        row = {
            "class_id": class_id,
            "class_name": class_name,
            "checkpoint_centroid_count": int(checkpoint_counts[class_id]),
            "selected_split": split,
            "selected_split_sample_count": int(split_counts[class_id]),
            "count_matches": (
                bool(checkpoint_counts[class_id] == split_counts[class_id])
                if split == "train"
                else None
            ),
            "checkpoint_vs_split_mean_cosine": float(alignment_cosine[class_id]),
            "nearest_centroid_id": nearest_id,
            "nearest_centroid_name": class_names[nearest_id],
            "nearest_centroid_cosine_similarity": float(
                diagnostics["nearest_centroid_similarity"][class_id]
            ),
            "nearest_centroid_angle_degrees": float(
                diagnostics["nearest_centroid_angular_distance_degrees"][class_id]
            ),
            "centroid_separation": float(
                diagnostics["centroid_separation"][class_id]
            ),
            "mean_own_similarity": per_class["mean_own_similarity"],
            "mean_cosine_margin": per_class["mean_cosine_margin"],
            "std_cosine_margin": per_class["std_cosine_margin"],
            "nearest_centroid_accuracy": per_class["nearest_centroid_accuracy"],
        }
        rows.append(row)

    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    args = _parse_args()
    experiment = args.experiment.strip("/")
    device = _select_device(args.device)
    output_dir = (
        _resolve_path(args.output_dir)
        if args.output_dir is not None
        else (
            ROOT
            / "results"
            / experiment
            / f"seed_{args.seed}"
            / "centroid_visualization"
            / args.split
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading evaluated model: {experiment}, seed={args.seed}")
    loaded = load_proposed_experiment_model(
        experiment_name=experiment,
        seed=args.seed,
        device=device,
    )
    centroids, checkpoint_counts = load_centroids_from_checkpoint(loaded.checkpoint)
    class_names = _display_class_names(loaded.cfg, centroids.shape[0])
    embeddings, labels, embedding_key, embeddings_path = _load_or_extract_embeddings(
        args,
        loaded,
        output_dir,
    )
    if embeddings.shape[1] != centroids.shape[1]:
        raise ValueError(
            f"Embedding dimension {embeddings.shape[1]} does not match centroid "
            f"dimension {centroids.shape[1]}."
        )
    if np.any(labels < 0) or np.any(labels >= centroids.shape[0]):
        raise ValueError("Selected-split labels are outside the checkpoint class range.")

    sampled_embeddings, sampled_labels = stratified_subsample(
        embeddings,
        labels,
        num_classes=len(class_names),
        max_points_per_class=args.max_points_per_class,
        seed=args.seed,
    )
    centroid_projection, sample_projection, explained, pca = project_cosine_space(
        centroids,
        sampled_embeddings,
        seed=args.seed,
    )
    if sample_projection is None:
        raise RuntimeError("Selected-split sample projection was unexpectedly empty.")
    diagnostics = compute_centroid_diagnostics(centroids, embeddings, labels)
    alignment = _centroid_split_alignment(centroids, embeddings, labels)

    projection_path = output_dir / "centroid_projection.png"
    heatmap_path = output_dir / "centroid_similarity_heatmap.png"
    diagnostics_path = output_dir / "centroid_diagnostics.png"
    csv_path = output_dir / "centroid_metrics.csv"
    summary_path = output_dir / "centroid_summary.json"

    ood_projection = None
    if args.ood_split is not None:
        analysis_root = ROOT / "results" / experiment / f"seed_{args.seed}" / "analysis"
        ood_feature_path = analysis_root / "features" / f"{args.ood_split}.npz"
        if not ood_feature_path.is_file():
            raise FileNotFoundError(
                f"OOD features not found at {ood_feature_path}. "
                "Please run `python tools/benchmark/ood_analysis.py --analyses ood` first to extract them."
            )
        ood_embeddings, _, _ = load_fused_embeddings(ood_feature_path)
        ood_sample = ood_embeddings[args.ood_index : args.ood_index + 1]
        normalized_ood = l2_normalize(ood_sample)
        ood_projection = pca.transform(normalized_ood)

    _plot_projection(
        centroid_projection,
        sample_projection,
        sampled_labels,
        class_names,
        checkpoint_counts,
        explained,
        projection_path,
        args.dpi,
        args.split,
        ood_projection=ood_projection,
    )
    _plot_similarity_heatmap(
        diagnostics["cosine_similarity_matrix"],
        class_names,
        heatmap_path,
        args.dpi,
    )
    _plot_diagnostics(
        diagnostics,
        alignment,
        class_names,
        diagnostics_path,
        args.dpi,
        args.split,
    )
    rows = _write_metrics_csv(
        csv_path,
        class_names,
        checkpoint_counts,
        diagnostics,
        alignment,
        args.split,
    )

    count_matches = checkpoint_counts == alignment["selected_split_counts"]
    alignment_cosine = alignment["checkpoint_vs_split_mean_cosine"]
    summary = {
        "experiment_name": experiment,
        "seed": args.seed,
        "checkpoint": str(loaded.checkpoint),
        "checkpoint_sha256": loaded.checkpoint_sha256,
        "embeddings": str(embeddings_path),
        "embedding_key": embedding_key,
        "split": args.split,
        "geometry": (
            "L2-normalized Phase-2 fused feature space used by the cosine "
            "empirical-centroid classifier"
        ),
        "num_selected_split_samples": int(len(embeddings)),
        "embedding_dimension": int(embeddings.shape[1]),
        "checkpoint_counts_match_selected_split": (
            bool(np.all(count_matches)) if args.split == "train" else None
        ),
        "checkpoint_vs_split_mean_cosine": {
            "mean": float(np.nanmean(alignment_cosine)),
            "minimum": float(np.nanmin(alignment_cosine)),
            "maximum": float(np.nanmax(alignment_cosine)),
        },
        "projection": {
            "method": "PCA fitted jointly to plotted normalized samples and centroids",
            "explained_variance_ratio": explained.tolist(),
            "plotted_embedding_count": int(len(sampled_embeddings)),
            "max_points_per_class": args.max_points_per_class,
        },
        "overall_nearest_centroid_accuracy": diagnostics[
            "overall_nearest_centroid_accuracy"
        ],
        "overall_mean_cosine_margin": diagnostics["overall_mean_cosine_margin"],
        "classes": rows,
        "cosine_similarity_matrix": diagnostics[
            "cosine_similarity_matrix"
        ].tolist(),
        "outputs": {
            "projection": str(projection_path),
            "similarity_heatmap": str(heatmap_path),
            "diagnostics": str(diagnostics_path),
            "metrics_csv": str(csv_path),
            "summary_json": str(summary_path),
        },
        "model_provenance": {
            key: value
            for key, value in loaded.provenance.items()
            if key != "resolved_config"
        },
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    normalized_centroids = l2_normalize(centroids)
    off_diagonal = normalized_centroids @ normalized_centroids.T
    np.fill_diagonal(off_diagonal, -np.inf)
    closest_pair = np.unravel_index(np.argmax(off_diagonal), off_diagonal.shape)
    print("\nCentroid visualization complete")
    print(f"  {args.split.capitalize()} samples: {len(embeddings):,}")
    print(f"  Fused feature dimension: {embeddings.shape[1]}")
    print(
        f"  Closest centroids: {class_names[closest_pair[0]]} <-> "
        f"{class_names[closest_pair[1]]} "
        f"(cosine={off_diagonal[closest_pair]:.4f})"
    )
    print(
        f"  Nearest-centroid {args.split} accuracy: "
        f"{diagnostics['overall_nearest_centroid_accuracy']:.2%}"
    )
    print(
        f"  Mean checkpoint/{args.split}-mean cosine: "
        f"{np.nanmean(alignment_cosine):.4f}"
    )
    print(f"  Saved to: {output_dir}")


if __name__ == "__main__":
    main()
