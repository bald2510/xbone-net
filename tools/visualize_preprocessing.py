"""Visualize foreground-aware fixed-budget sparse focal preprocessing."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.high_resolution import SparseFocalViews, build_sparse_focal_views


ROLE_COLORS = {
    "coverage": "#00c7d9",
    "focal": "#ff9f43",
    "fallback": "#c8c8c8",
}


def load_ground_truth_annotations(path: Path | None):
    if path is None or not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [shape for shape in payload.get("shapes", []) if shape.get("points")]


def annotation_bounds(shapes):
    boxes = []
    for shape in shapes:
        xs = [point[0] for point in shape["points"]]
        ys = [point[1] for point in shape["points"]]
        boxes.append((min(xs), min(ys), max(xs), max(ys), shape.get("label", "GT")))
    return boxes


def draw_ground_truth(axis, annotations):
    for left, top, right, bottom, label in annotation_bounds(annotations):
        axis.add_patch(
            Rectangle(
                (left, top),
                right - left,
                bottom - top,
                fill=False,
                edgecolor="#f600ff",
                linewidth=2.0,
            )
        )
        axis.text(
            left,
            top,
            f"GT: {label}",
            color="white",
            fontsize=8,
            bbox={"facecolor": "#a000a8", "alpha": 0.85, "pad": 2},
        )


def _draw_box(axis, box, color, linewidth=1.0, label=None, alpha=1.0):
    left, top, right, bottom = box
    axis.add_patch(
        Rectangle(
            (left, top),
            right - left,
            bottom - top,
            fill=False,
            edgecolor=color,
            linewidth=linewidth,
            alpha=alpha,
        )
    )
    if label:
        axis.text(
            left,
            top,
            label,
            color="white",
            fontsize=8,
            bbox={"facecolor": color, "alpha": 0.88, "pad": 1.5},
        )


def render_preprocessing(
    source: Image.Image,
    views: SparseFocalViews,
    annotations,
    output: Path,
    dpi: int,
):
    figure = plt.figure(figsize=(18, 10), facecolor="#101418")
    layout = figure.add_gridspec(2, 4, height_ratios=(1.65, 1.0))

    source_axis = figure.add_subplot(layout[0, 0])
    source_axis.imshow(source)
    _draw_box(source_axis, views.foreground_box, "#ffe066", linewidth=2.2, label="ROI")
    for box in views.candidate_boxes:
        _draw_box(source_axis, box, "white", linewidth=0.55, alpha=0.30)
    for tile_index, (candidate_index, role) in enumerate(
        zip(views.selected_candidate_indices, views.tile_roles),
        start=1,
    ):
        _draw_box(
            source_axis,
            views.candidate_boxes[candidate_index],
            ROLE_COLORS[role],
            linewidth=2.0,
            label=f"T{tile_index}",
        )
    draw_ground_truth(source_axis, annotations)
    source_axis.set_title(
        f"1. Source {source.width}x{source.height}\nROI + selected fixed-K views",
        color="white",
    )
    source_axis.axis("off")

    foreground_axis = figure.add_subplot(layout[0, 1])
    foreground_axis.imshow(views.foreground_image)
    foreground_axis.set_title(
        f"2. Foreground ROI\n{views.foreground_image.width}x{views.foreground_image.height}",
        color="white",
    )
    foreground_axis.axis("off")

    candidate_axis = figure.add_subplot(layout[0, 2])
    candidate_axis.imshow(views.canvas)
    selected_lookup = {
        candidate_index: (tile_index, role)
        for tile_index, (candidate_index, role) in enumerate(
            zip(views.selected_candidate_indices, views.tile_roles),
            start=1,
        )
    }
    for index, box in enumerate(views.candidate_canvas_boxes):
        selected = selected_lookup.get(index)
        if selected is None:
            _draw_box(candidate_axis, box, "white", linewidth=0.65, alpha=0.35)
        else:
            tile_index, role = selected
            _draw_box(
                candidate_axis,
                box,
                ROLE_COLORS[role],
                linewidth=2.2,
                label=f"T{tile_index} {role[0].upper()}",
            )
    candidate_axis.set_title(
        f"3. Candidate canvas {views.canvas.width}x{views.canvas.height}\n"
        f"{len(views.candidate_boxes)} candidates -> {len(views.tiles)} encoded",
        color="white",
    )
    candidate_axis.axis("off")

    global_axis = figure.add_subplot(layout[0, 3])
    global_axis.imshow(views.global_image)
    global_axis.set_title(
        "4. Global view\nforeground letterbox -> BiomedCLIP 224",
        color="white",
    )
    global_axis.axis("off")

    for tile_index, (tile, candidate_index, role) in enumerate(
        zip(views.tiles, views.selected_candidate_indices, views.tile_roles)
    ):
        axis = figure.add_subplot(layout[1, tile_index])
        axis.imshow(tile)
        score = views.candidate_scores[candidate_index]
        foreground_ratio = views.candidate_foreground_ratios[candidate_index]
        axis.set_title(
            f"T{tile_index + 1} - {role}\n"
            f"score={score:.3f}, foreground={foreground_ratio:.0%}",
            color=ROLE_COLORS[role],
        )
        axis.axis("off")

    figure.suptitle(
        "Sparse Focal AnyRes: 1 global + 4 local views, unchanged 224x224 BiomedCLIP",
        color="white",
        fontsize=16,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--annotation")
    parser.add_argument(
        "--output",
        default="outputs/visualizations/sparse_focal_preprocessing.png",
    )
    parser.add_argument("--global-size", type=int, default=224)
    parser.add_argument("--tile-size", type=int, default=224)
    parser.add_argument("--canonical-long-side", type=int, default=896)
    parser.add_argument("--candidate-stride", type=int, default=168)
    parser.add_argument("--num-tiles", type=int, default=4)
    parser.add_argument("--coverage-tiles", type=int, default=2)
    parser.add_argument("--min-foreground-ratio", type=float, default=0.60)
    parser.add_argument("--nms-iou", type=float, default=0.30)
    parser.add_argument("--foreground-margin", type=float, default=0.04)
    parser.add_argument("--dpi", type=int, default=140)
    args = parser.parse_args()

    image_path = Path(args.image).expanduser().resolve()
    with Image.open(image_path) as opened:
        source = opened.convert("RGB")
    config = {
        "strategy": "sparse_focal",
        "global_size": args.global_size,
        "tile_size": args.tile_size,
        "canonical_long_side": args.canonical_long_side,
        "candidate_stride": args.candidate_stride,
        "num_tiles": args.num_tiles,
        "coverage_tiles": args.coverage_tiles,
        "min_foreground_ratio": args.min_foreground_ratio,
        "nms_iou": args.nms_iou,
        "allow_upsample": False,
        "pad_value": "border_median",
        "foreground": {"margin": args.foreground_margin},
    }
    views = build_sparse_focal_views(source, config)
    annotation_path = (
        Path(args.annotation).expanduser().resolve()
        if args.annotation
        else PROJECT_ROOT / "data" / "BTXRD" / "Annotations" / f"{image_path.stem}.json"
    )
    annotations = load_ground_truth_annotations(annotation_path)
    output = Path(args.output).expanduser().resolve()
    render_preprocessing(source, views, annotations, output, args.dpi)
    print(f"Saved: {output}")
    print(
        f"Candidates: {len(views.candidate_boxes)}; encoded local views: "
        f"{len(views.tiles)}; visual encoder calls including global: {1 + len(views.tiles)}"
    )


if __name__ == "__main__":
    main()
