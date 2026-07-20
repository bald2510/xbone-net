"""Visualize and audit foreground-aware fixed-budget sparse-focal preprocessing."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.patches import Rectangle
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.high_resolution import SparseFocalViews, build_sparse_focal_views


DEFAULT_EXPERIMENT_CONFIG = (
    PROJECT_ROOT / "configs" / "experiment" / "btxrd" / "proposed" / "ours_xbone_net.yaml"
)

ROLE_COLORS = {
    "coverage": "#00c7d9",
    "focal": "#ff9f43",
    "fallback": "#c8c8c8",
}


def load_high_resolution_config(path: Path) -> dict[str, Any]:
    """Load the canonical high-resolution block from an experiment YAML file."""
    if not path.is_file():
        raise FileNotFoundError(f"Experiment config not found: {path}")

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Experiment config must contain a YAML mapping: {path}")

    high_res: Any = None
    dataset = payload.get("dataset")
    if isinstance(dataset, Mapping):
        params = dataset.get("params")
        if isinstance(params, Mapping):
            high_res = params.get("high_res")
    if high_res is None:
        high_res = payload.get("high_res")
    if high_res is None and "strategy" in payload:
        high_res = payload
    if not isinstance(high_res, Mapping):
        raise ValueError(
            "Could not find a high-resolution config mapping at "
            "dataset.params.high_res, high_res, or the YAML root."
        )
    if not bool(high_res.get("enabled", True)):
        raise ValueError(f"High-resolution preprocessing is disabled in: {path}")

    return copy.deepcopy(dict(high_res))


def apply_config_overrides(
    config: Mapping[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Apply only explicitly supplied CLI values to the loaded config."""
    resolved = copy.deepcopy(dict(config))
    top_level_overrides = {
        "global_size": "global_size",
        "tile_size": "tile_size",
        "canonical_long_side": "canonical_long_side",
        "candidate_stride": "candidate_stride",
        "num_tiles": "num_tiles",
        "coverage_tiles": "coverage_tiles",
        "min_foreground_ratio": "min_foreground_ratio",
        "nms_iou": "nms_iou",
        "focal_diversity_weight": "focal_diversity_weight",
        "border_fraction": "border_fraction",
        "min_contrast": "min_contrast",
        "allow_upsample": "allow_upsample",
    }
    for argument_name, config_name in top_level_overrides.items():
        value = getattr(args, argument_name, None)
        if value is not None:
            resolved[config_name] = value

    foreground_margin = getattr(args, "foreground_margin", None)
    if foreground_margin is not None:
        foreground = copy.deepcopy(dict(resolved.get("foreground", {})))
        foreground["margin"] = foreground_margin
        resolved["foreground"] = foreground
    return resolved


def load_ground_truth_annotations(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [shape for shape in payload.get("shapes", []) if shape.get("points")]


def _shape_points(shape: Mapping[str, Any]) -> list[tuple[float, float]]:
    points = []
    for point in shape.get("points", []):
        if isinstance(point, Sequence) and len(point) >= 2:
            points.append((float(point[0]), float(point[1])))
    return points


def annotation_bounds(shapes: Sequence[Mapping[str, Any]]):
    boxes = []
    for shape in shapes:
        points = _shape_points(shape)
        if not points:
            continue
        if str(shape.get("shape_type", "")).lower() == "circle" and len(points) >= 2:
            center_x, center_y = points[0]
            radius = math.hypot(points[1][0] - center_x, points[1][1] - center_y)
            left, top = center_x - radius, center_y - radius
            right, bottom = center_x + radius, center_y + radius
        else:
            xs = [point[0] for point in points]
            ys = [point[1] for point in points]
            left, top, right, bottom = min(xs), min(ys), max(xs), max(ys)
        boxes.append((left, top, right, bottom, shape.get("label", "GT")))
    return boxes


def draw_ground_truth(axis, annotations, offset: tuple[float, float] = (0.0, 0.0)):
    offset_x, offset_y = offset
    for left, top, right, bottom, label in annotation_bounds(annotations):
        left += offset_x
        right += offset_x
        top += offset_y
        bottom += offset_y
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


def _draw_annotation_mask(draw, shape: Mapping[str, Any], image_size: tuple[int, int]):
    points = _shape_points(shape)
    if not points:
        return
    pixel_points = [(round(x), round(y)) for x, y in points]
    shape_type = str(shape.get("shape_type", "polygon")).lower()
    width = max(1, round(min(image_size) * 0.003))

    if shape_type == "rectangle" and len(pixel_points) >= 2:
        xs = [point[0] for point in pixel_points]
        ys = [point[1] for point in pixel_points]
        draw.rectangle((min(xs), min(ys), max(xs), max(ys)), fill=1)
    elif shape_type == "circle" and len(pixel_points) >= 2:
        center_x, center_y = pixel_points[0]
        radius = round(
            math.hypot(
                pixel_points[1][0] - center_x,
                pixel_points[1][1] - center_y,
            )
        )
        draw.ellipse(
            (
                center_x - radius,
                center_y - radius,
                center_x + radius,
                center_y + radius,
            ),
            fill=1,
        )
    elif shape_type == "point":
        center_x, center_y = pixel_points[0]
        radius = max(1, width)
        draw.ellipse(
            (center_x - radius, center_y - radius, center_x + radius, center_y + radius),
            fill=1,
        )
    elif shape_type in {"line", "linestrip"} or len(pixel_points) == 2:
        draw.line(pixel_points, fill=1, width=width, joint="curve")
    elif len(pixel_points) >= 3:
        draw.polygon(pixel_points, fill=1)


def _annotation_mask(
    annotations: Sequence[Mapping[str, Any]],
    image_size: tuple[int, int],
) -> np.ndarray:
    mask = Image.new("1", image_size, color=0)
    draw = ImageDraw.Draw(mask)
    for shape in annotations:
        _draw_annotation_mask(draw, shape, image_size)
    return np.asarray(mask, dtype=bool)


def _boxes_mask(
    boxes: Sequence[Sequence[int | float]],
    image_size: tuple[int, int],
) -> np.ndarray:
    image_width, image_height = image_size
    mask = Image.new("1", image_size, color=0)
    draw = ImageDraw.Draw(mask)
    for left, top, right, bottom in boxes:
        left = max(0, min(image_width, math.floor(left)))
        top = max(0, min(image_height, math.floor(top)))
        right = max(0, min(image_width, math.ceil(right)))
        bottom = max(0, min(image_height, math.ceil(bottom)))
        if right > left and bottom > top:
            # SparseFocal boxes use half-open XYXY coordinates.
            draw.rectangle((left, top, right - 1, bottom - 1), fill=1)
    return np.asarray(mask, dtype=bool)


def compute_annotation_coverage(
    annotations: Sequence[Mapping[str, Any]],
    image_size: tuple[int, int],
    foreground_box: Sequence[int | float],
    tile_boxes: Sequence[Sequence[int | float]],
) -> dict[str, Any]:
    """Measure GT retention for auditing; annotations never affect selection."""
    metrics: dict[str, Any] = {
        "available": False,
        "annotation_count": len(annotations),
        "gt_pixel_area": 0,
        "foreground_coverage_ratio": None,
        "local_union_coverage_ratio": None,
        "local_hit": False,
        "tile_hit_indices": [],
        "per_tile_coverage_ratios": [0.0 for _ in tile_boxes],
    }
    if not annotations:
        return metrics

    gt_mask = _annotation_mask(annotations, image_size)
    gt_area = int(np.count_nonzero(gt_mask))
    metrics["gt_pixel_area"] = gt_area
    if gt_area == 0:
        return metrics

    foreground_mask = _boxes_mask([foreground_box], image_size)
    local_union_mask = _boxes_mask(tile_boxes, image_size)
    foreground_covered = int(np.count_nonzero(gt_mask & foreground_mask))
    local_covered = int(np.count_nonzero(gt_mask & local_union_mask))

    tile_coverage = []
    tile_hit_indices = []
    for tile_index, tile_box in enumerate(tile_boxes, start=1):
        tile_mask = _boxes_mask([tile_box], image_size)
        covered = int(np.count_nonzero(gt_mask & tile_mask))
        tile_coverage.append(covered / gt_area)
        if covered > 0:
            tile_hit_indices.append(tile_index)

    metrics.update(
        {
            "available": True,
            "foreground_coverage_ratio": foreground_covered / gt_area,
            "local_union_coverage_ratio": local_covered / gt_area,
            "local_hit": local_covered > 0,
            "tile_hit_indices": tile_hit_indices,
            "per_tile_coverage_ratios": tile_coverage,
        }
    )
    return metrics


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


def _selected_candidates(views: SparseFocalViews):
    selected: dict[int, list[tuple[int, str]]] = {}
    for tile_index, (candidate_index, role) in enumerate(
        zip(views.selected_candidate_indices, views.tile_roles),
        start=1,
    ):
        selected.setdefault(candidate_index, []).append((tile_index, role))
    return selected


def _config_label(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def render_preprocessing(
    source: Image.Image,
    views: SparseFocalViews,
    annotations,
    output: Path,
    dpi: int,
    metrics: Mapping[str, Any] | None = None,
    config_label: str | None = None,
):
    metrics = metrics or compute_annotation_coverage(
        annotations,
        source.size,
        views.foreground_box,
        views.tile_boxes,
    )
    tile_count = len(views.tiles)
    maximum_columns = min(4, max(1, tile_count))
    tile_columns = min(
        range(1, maximum_columns + 1),
        key=lambda columns: (
            math.ceil(tile_count / columns),
            math.ceil(tile_count / columns) * columns - tile_count,
        ),
    )
    tile_rows = max(1, math.ceil(tile_count / tile_columns))

    figure = plt.figure(
        figsize=(18, 6.2 + 3.8 * tile_rows),
        facecolor="#101418",
        layout="constrained",
    )
    outer_layout = figure.add_gridspec(
        2,
        1,
        height_ratios=(1.65, float(tile_rows)),
    )
    top_layout = outer_layout[0].subgridspec(1, 4)
    tile_layout = outer_layout[1].subgridspec(tile_rows, tile_columns)
    selected_lookup = _selected_candidates(views)

    source_axis = figure.add_subplot(top_layout[0, 0])
    source_axis.imshow(source)
    _draw_box(
        source_axis,
        views.foreground_box,
        "#ffe066",
        linewidth=2.2,
        label="Foreground crop",
    )
    for box in views.candidate_boxes:
        _draw_box(source_axis, box, "white", linewidth=0.55, alpha=0.30)
    for candidate_index, selections in selected_lookup.items():
        role = selections[0][1]
        _draw_box(
            source_axis,
            views.candidate_boxes[candidate_index],
            ROLE_COLORS.get(role, "#c8c8c8"),
            linewidth=2.0,
            label="/".join(f"T{tile_index}" for tile_index, _ in selections),
        )
    draw_ground_truth(source_axis, annotations)
    local_coverage = metrics.get("local_union_coverage_ratio")
    local_summary = (
        f"GT covered by local union: {local_coverage:.1%}"
        if metrics.get("available") and local_coverage is not None
        else "GT coverage: unavailable"
    )
    source_axis.set_title(
        f"1. Source {source.width}x{source.height}\n"
        f"Foreground crop + selected fixed-K views\n{local_summary}",
        color="white",
    )
    source_axis.axis("off")

    foreground_axis = figure.add_subplot(top_layout[0, 1])
    foreground_axis.imshow(views.foreground_image)
    foreground_left, foreground_top, _, _ = views.foreground_box
    draw_ground_truth(
        foreground_axis,
        annotations,
        offset=(-foreground_left, -foreground_top),
    )
    foreground_coverage = metrics.get("foreground_coverage_ratio")
    foreground_summary = (
        f"GT retained: {foreground_coverage:.1%}"
        if metrics.get("available") and foreground_coverage is not None
        else "GT retained: unavailable"
    )
    foreground_axis.set_title(
        f"2. Foreground crop\n"
        f"{views.foreground_image.width}x{views.foreground_image.height}; "
        f"{foreground_summary}",
        color="white",
    )
    foreground_axis.axis("off")

    candidate_axis = figure.add_subplot(top_layout[0, 2])
    candidate_axis.imshow(views.canvas)
    for index, box in enumerate(views.candidate_canvas_boxes):
        selections = selected_lookup.get(index)
        if selections is None:
            _draw_box(candidate_axis, box, "white", linewidth=0.65, alpha=0.35)
        else:
            role = selections[0][1]
            label = "/".join(
                f"T{tile_index}{selection_role[0].upper()}"
                for tile_index, selection_role in selections
            )
            _draw_box(
                candidate_axis,
                box,
                ROLE_COLORS.get(role, "#c8c8c8"),
                linewidth=2.2,
                label=label,
            )
    candidate_axis.set_title(
        f"3. Candidate canvas {views.canvas.width}x{views.canvas.height}\n"
        f"{len(views.candidate_boxes)} candidates -> {tile_count} encoded",
        color="white",
    )
    candidate_axis.axis("off")

    global_axis = figure.add_subplot(top_layout[0, 3])
    global_axis.imshow(views.global_image)
    global_axis.set_title(
        f"4. Global view\nforeground letterbox -> "
        f"{views.global_image.width}x{views.global_image.height}",
        color="white",
    )
    global_axis.axis("off")

    per_tile_coverage = metrics.get("per_tile_coverage_ratios", [])
    for zero_based_index, (tile, candidate_index, role) in enumerate(
        zip(views.tiles, views.selected_candidate_indices, views.tile_roles)
    ):
        row, column = divmod(zero_based_index, tile_columns)
        axis = figure.add_subplot(tile_layout[row, column])
        axis.imshow(tile)
        score = views.candidate_scores[candidate_index]
        foreground_ratio = views.candidate_foreground_ratios[candidate_index]
        gt_summary = ""
        if metrics.get("available") and zero_based_index < len(per_tile_coverage):
            gt_summary = f"; GT cover={per_tile_coverage[zero_based_index]:.1%}"
        axis.set_title(
            f"T{zero_based_index + 1} - {role} "
            f"({tile.width}x{tile.height})\n"
            f"score={score:.3f}; foreground={foreground_ratio:.0%}{gt_summary}",
            color=ROLE_COLORS.get(role, "#c8c8c8"),
        )
        axis.axis("off")

    title = f"Sparse-focal preprocessing: 1 global + {tile_count} local views"
    audit_note = "GT overlay/coverage is audit-only and is not used for tile selection"
    if config_label:
        audit_note = f"{audit_note} | config: {config_label}"
    figure.suptitle(f"{title}\n{audit_note}", color="white", fontsize=15)

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)


def _default_annotation_path(image_path: Path) -> Path:
    if image_path.parent.name.casefold() == "images":
        return image_path.parent.parent / "Annotations" / f"{image_path.stem}.json"
    return PROJECT_ROOT / "data" / "BTXRD" / "Annotations" / f"{image_path.stem}.json"


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize the exact sparse-focal preprocessing used by an experiment "
            "and optionally audit its overlap with LabelMe ground truth."
        )
    )
    parser.add_argument("--image", required=True)
    parser.add_argument("--annotation")
    parser.add_argument(
        "--output",
        default="results/visualizations/sparse_focal_preprocessing.png",
    )
    parser.add_argument(
        "--experiment-config",
        "--config",
        dest="experiment_config",
        default=str(DEFAULT_EXPERIMENT_CONFIG),
        help="Experiment YAML containing dataset.params.high_res.",
    )
    parser.add_argument("--global-size", type=int)
    parser.add_argument("--tile-size", type=int)
    parser.add_argument("--canonical-long-side", type=int)
    parser.add_argument("--candidate-stride", type=int)
    parser.add_argument("--num-tiles", type=int)
    parser.add_argument("--coverage-tiles", type=int)
    parser.add_argument("--min-foreground-ratio", type=float)
    parser.add_argument("--nms-iou", type=float)
    parser.add_argument("--focal-diversity-weight", type=float)
    parser.add_argument("--border-fraction", type=float)
    parser.add_argument("--min-contrast", type=float)
    parser.add_argument("--foreground-margin", type=float)
    parser.add_argument(
        "--allow-upsample",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--dpi", type=int, default=140)
    return parser


def main():
    args = build_argument_parser().parse_args()

    image_path = Path(args.image).expanduser().resolve()
    with Image.open(image_path) as opened:
        source = opened.convert("RGB")

    config_path = Path(args.experiment_config).expanduser().resolve()
    config = apply_config_overrides(load_high_resolution_config(config_path), args)
    views = build_sparse_focal_views(source, config)

    annotation_path = (
        Path(args.annotation).expanduser().resolve()
        if args.annotation
        else _default_annotation_path(image_path)
    )
    annotation_exists = annotation_path.is_file()
    annotations = load_ground_truth_annotations(annotation_path)
    if not annotation_exists:
        print(f"Warning: GT annotation not found; coverage audit unavailable: {annotation_path}")
    elif not annotations:
        print(f"Warning: GT annotation contains no drawable shapes: {annotation_path}")

    metrics = compute_annotation_coverage(
        annotations,
        source.size,
        views.foreground_box,
        views.tile_boxes,
    )
    output = Path(args.output).expanduser().resolve()
    render_preprocessing(
        source,
        views,
        annotations,
        output,
        args.dpi,
        metrics=metrics,
        config_label=_config_label(config_path),
    )

    print(f"Saved: {output}")
    print(f"Config: {config_path}")
    print(
        f"Candidates: {len(views.candidate_boxes)}; encoded local views: "
        f"{len(views.tiles)}; visual encoder calls including global: {1 + len(views.tiles)}"
    )
    if metrics["available"]:
        hit_tiles = ", ".join(f"T{index}" for index in metrics["tile_hit_indices"])
        print(
            "GT audit only (not used for selection): "
            f"foreground retained={metrics['foreground_coverage_ratio']:.1%}; "
            f"selected-local union coverage={metrics['local_union_coverage_ratio']:.1%}; "
            f"hit tiles={hit_tiles or 'none'}"
        )
    else:
        print("GT audit only: unavailable; tile selection remains image-only.")


if __name__ == "__main__":
    main()
