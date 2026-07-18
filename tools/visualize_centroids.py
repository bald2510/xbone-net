"""Visualize empirical centroids in the cosine-normalized Phase-2 feature space."""

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
import numpy as np  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Phase-2 checkpoint; defaults to params.phase2.checkpoint_path.",
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=None,
        help="Optional evaluate.py --save-embeddings NPZ.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-points-per-class", type=int, default=500)
    parser.add_argument("--dpi", type=int, default=180)
    args, hydra_args = parser.parse_known_args()
    if args.max_points_per_class < 1:
        parser.error("--max-points-per-class must be positive.")
    if args.dpi < 72:
        parser.error("--dpi must be at least 72.")
    sys.argv = [sys.argv[0], *hydra_args]
    return args


ARGS = _parse_args()

import hydra  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

from src.utils.centroid_visualization import (  # noqa: E402
    compute_centroid_diagnostics,
    l2_normalize,
    load_centroids_from_checkpoint,
    load_fused_embeddings,
    project_cosine_space,
    stratified_subsample,
)


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _default_paths(cfg: DictConfig) -> tuple[Path, Path, Path]:
    params_cfg = cfg.get("params", {}) or {}
    phase2_cfg = params_cfg.get("phase2", {}) or {}
    experiment = str(cfg.get("experiment_name", "default"))
    seed = int(params_cfg.get("seed", cfg.get("seed", 42)))

    checkpoint = ARGS.checkpoint
    if checkpoint is None:
        checkpoint_value = phase2_cfg.get("checkpoint_path")
        if not checkpoint_value:
            checkpoint_value = (
                ROOT / "checkpoints" / experiment / f"seed_{seed}" / "best_phase2.pth"
            )
        checkpoint = Path(str(checkpoint_value))

    default_results = ROOT / "results" / experiment / f"seed_{seed}"
    embeddings = ARGS.embeddings or default_results / "embeddings.npz"
    output_dir = ARGS.output_dir or default_results / "centroid_visualization"
    return (
        _resolve_path(checkpoint).resolve(),
        _resolve_path(embeddings).resolve(),
        _resolve_path(output_dir).resolve(),
    )


def _class_colors(num_classes: int) -> np.ndarray:
    cmap_name = "tab20" if num_classes <= 20 else "turbo"
    cmap = plt.get_cmap(cmap_name, num_classes)
    return np.asarray([cmap(index) for index in range(num_classes)])


def _plot_projection(
    centroid_projection: np.ndarray,
    sample_projection: np.ndarray | None,
    sample_labels: np.ndarray | None,
    class_names: list[str],
    counts: np.ndarray,
    explained_variance: np.ndarray,
    output: Path,
    dpi: int,
) -> None:
    colors = _class_colors(len(class_names))
    fig, axis = plt.subplots(figsize=(12, 9))

    if sample_projection is not None and sample_labels is not None:
        for class_id in range(len(class_names)):
            mask = sample_labels == class_id
            if np.any(mask):
                axis.scatter(
                    sample_projection[mask, 0],
                    sample_projection[mask, 1],
                    s=14,
                    alpha=0.22,
                    color=colors[class_id],
                    edgecolors="none",
                    rasterized=True,
                )

    marker_sizes = 170.0 + 35.0 * np.log1p(counts / max(float(counts.min()), 1.0))
    for class_id, (x_value, y_value) in enumerate(centroid_projection):
        axis.scatter(
            x_value,
            y_value,
            s=float(marker_sizes[class_id]),
            marker="X",
            color=colors[class_id],
            edgecolor="black",
            linewidth=0.9,
            zorder=5,
        )
        label = (
            f"{class_id}: {class_names[class_id]}"
            if len(class_names) <= 15
            else str(class_id)
        )
        axis.annotate(
            label,
            (x_value, y_value),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            weight="bold",
        )

    variance = 100.0 * explained_variance
    axis.set_xlabel(f"PCA 1 ({variance[0]:.1f}% variance)")
    axis.set_ylabel(f"PCA 2 ({variance[1]:.1f}% variance)")
    axis.set_title("Cosine-normalized fused embeddings and empirical centroids")
    axis.axhline(0.0, color="0.85", linewidth=0.8)
    axis.axvline(0.0, color="0.85", linewidth=0.8)
    axis.grid(alpha=0.16)
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
    size = min(18.0, max(8.0, num_classes * 0.36))
    fig, axis = plt.subplots(figsize=(size, size))
    image = axis.imshow(similarity, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    labels = class_names if num_classes <= 20 else [str(i) for i in range(num_classes)]
    axis.set_xticks(np.arange(num_classes), labels=labels, rotation=90)
    axis.set_yticks(np.arange(num_classes), labels=labels)
    axis.tick_params(labelsize=8 if num_classes <= 20 else 6)
    axis.set_xlabel("Centroid class")
    axis.set_ylabel("Centroid class")
    axis.set_title("Pairwise centroid cosine similarity")

    if num_classes <= 15:
        for row in range(num_classes):
            for column in range(num_classes):
                value = similarity[row, column]
                axis.text(
                    column,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if abs(value) > 0.55 else "black",
                )
    colorbar = fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Cosine similarity")
    fig.tight_layout()
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_diagnostics(
    diagnostics: dict[str, Any],
    counts: np.ndarray,
    class_names: list[str],
    output: Path,
    dpi: int,
) -> None:
    num_classes = len(class_names)
    labels = [f"{index}: {name}" for index, name in enumerate(class_names)]
    y_positions = np.arange(num_classes)
    height = max(6.0, min(18.0, 0.34 * num_classes + 2.0))
    fig, axes = plt.subplots(1, 2, figsize=(15, height), sharey=True)

    if "per_class" in diagnostics:
        margins = np.asarray(
            [
                np.nan if item["mean_cosine_margin"] is None else item["mean_cosine_margin"]
                for item in diagnostics["per_class"]
            ],
            dtype=float,
        )
        accuracies = np.asarray(
            [
                np.nan
                if item["nearest_centroid_accuracy"] is None
                else item["nearest_centroid_accuracy"]
                for item in diagnostics["per_class"]
            ],
            dtype=float,
        )
        bar_colors = ["#b2182b" if value < 0 else "#2166ac" for value in np.nan_to_num(margins)]
        axes[0].barh(y_positions, np.nan_to_num(margins), color=bar_colors)
        axes[0].axvline(0.0, color="black", linewidth=0.9)
        axes[0].set_xlabel("Mean own-vs-nearest-other cosine margin")
        axes[0].set_title("Class separation on exported embeddings")
        axes[1].barh(y_positions, np.nan_to_num(accuracies), color="#4d9221")
        axes[1].set_xlim(0.0, 1.0)
        axes[1].set_xlabel("Nearest-centroid accuracy")
        axes[1].set_title("Agreement with empirical-centroid classifier")
    else:
        separation = diagnostics["centroid_separation"]
        axes[0].barh(y_positions, separation, color="#2166ac")
        axes[0].set_xlabel("1 − nearest-centroid cosine similarity")
        axes[0].set_title("Nearest-centroid separation")
        axes[1].barh(y_positions, counts, color="#762a83")
        axes[1].set_xlabel("Train samples used for centroid")
        axes[1].set_title("Empirical centroid support")

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
    counts: np.ndarray,
    diagnostics: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    per_class = diagnostics.get("per_class")
    for class_id, class_name in enumerate(class_names):
        nearest_id = int(diagnostics["nearest_centroid_ids"][class_id])
        row = {
            "class_id": class_id,
            "class_name": class_name,
            "train_centroid_count": int(counts[class_id]),
            "nearest_centroid_id": nearest_id,
            "nearest_centroid_name": class_names[nearest_id],
            "nearest_centroid_cosine_similarity": float(
                diagnostics["nearest_centroid_similarity"][class_id]
            ),
            "nearest_centroid_angle_degrees": float(
                diagnostics["nearest_centroid_angular_distance_degrees"][class_id]
            ),
            "centroid_separation": float(diagnostics["centroid_separation"][class_id]),
            "embedding_sample_count": None,
            "mean_own_similarity": None,
            "mean_cosine_margin": None,
            "std_cosine_margin": None,
            "nearest_centroid_accuracy": None,
        }
        if per_class is not None:
            row.update(per_class[class_id])
            row["embedding_sample_count"] = row.pop("sample_count")
        rows.append(row)

    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    checkpoint_path, embeddings_path, output_dir = _default_paths(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)

    class_names = list(
        cfg.dataset.params.get("classes", cfg.dataset.params.get("pathologies", []))
    )
    centroids, counts = load_centroids_from_checkpoint(checkpoint_path)
    if len(class_names) != centroids.shape[0]:
        raise ValueError(
            f"Config defines {len(class_names)} classes but checkpoint contains "
            f"{centroids.shape[0]} centroids."
        )

    embeddings = None
    labels = None
    embedding_key = None
    sampled_embeddings = None
    sampled_labels = None
    if embeddings_path.is_file():
        embeddings, labels, embedding_key = load_fused_embeddings(embeddings_path)
        if embeddings.shape[1] != centroids.shape[1]:
            raise ValueError(
                f"Embedding dimension {embeddings.shape[1]} does not match centroid "
                f"dimension {centroids.shape[1]}."
            )
        sampled_embeddings, sampled_labels = stratified_subsample(
            embeddings,
            labels,
            num_classes=len(class_names),
            max_points_per_class=ARGS.max_points_per_class,
            seed=int(cfg.get("seed", 42)),
        )
        print(
            f"Loaded {len(embeddings)} embeddings; plotting "
            f"{len(sampled_embeddings)} stratified points."
        )
    elif ARGS.embeddings is not None:
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
    else:
        print(
            "[Info] No embeddings.npz found; generating centroid-only projection "
            "and diagnostics. Run evaluate.py --save-embeddings to overlay samples."
        )

    centroid_projection, sample_projection, explained = project_cosine_space(
        centroids,
        sampled_embeddings,
        seed=int(cfg.get("seed", 42)),
    )
    diagnostics = compute_centroid_diagnostics(centroids, embeddings, labels)

    projection_path = output_dir / "centroid_projection.png"
    heatmap_path = output_dir / "centroid_similarity_heatmap.png"
    diagnostics_path = output_dir / "centroid_diagnostics.png"
    csv_path = output_dir / "centroid_metrics.csv"
    summary_path = output_dir / "centroid_summary.json"

    _plot_projection(
        centroid_projection,
        sample_projection,
        sampled_labels,
        class_names,
        counts,
        explained,
        projection_path,
        ARGS.dpi,
    )
    _plot_similarity_heatmap(
        diagnostics["cosine_similarity_matrix"],
        class_names,
        heatmap_path,
        ARGS.dpi,
    )
    _plot_diagnostics(
        diagnostics,
        counts,
        class_names,
        diagnostics_path,
        ARGS.dpi,
    )
    rows = _write_metrics_csv(csv_path, class_names, counts, diagnostics)

    summary = {
        "experiment_name": str(cfg.get("experiment_name", "default")),
        "seed": int(cfg.get("seed", 42)),
        "checkpoint": str(checkpoint_path),
        "embeddings": str(embeddings_path) if embeddings is not None else None,
        "embedding_key": embedding_key,
        "geometry": "L2-normalized fused feature space used by cosine classifier",
        "projection": {
            "method": "PCA",
            "explained_variance_ratio": explained.tolist(),
            "plotted_embedding_count": (
                int(len(sampled_embeddings)) if sampled_embeddings is not None else 0
            ),
        },
        "overall_nearest_centroid_accuracy": diagnostics.get(
            "overall_nearest_centroid_accuracy"
        ),
        "overall_mean_cosine_margin": diagnostics.get("overall_mean_cosine_margin"),
        "classes": rows,
        "cosine_similarity_matrix": diagnostics["cosine_similarity_matrix"].tolist(),
        "outputs": {
            "projection": str(projection_path),
            "similarity_heatmap": str(heatmap_path),
            "diagnostics": str(diagnostics_path),
            "metrics_csv": str(csv_path),
        },
        "config": OmegaConf.to_container(cfg, resolve=True),
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
    print(
        f"  Closest pair: {class_names[closest_pair[0]]} <-> "
        f"{class_names[closest_pair[1]]} "
        f"(cosine={off_diagonal[closest_pair]:.4f})"
    )
    if diagnostics.get("overall_nearest_centroid_accuracy") is not None:
        print(
            "  Nearest-centroid accuracy on exported embeddings: "
            f"{diagnostics['overall_nearest_centroid_accuracy']:.2%}"
        )
    print(f"  Saved to: {output_dir}")


if __name__ == "__main__":
    main()
