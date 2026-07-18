"""Foreground-aware fixed-budget high-resolution preprocessing for X-rays.

BiomedCLIP remains a fixed 224x224 visual encoder.  Each sample contains one
aspect-preserving global view and a small, constant number of local views.  The
local views are selected using image-only coverage and texture statistics; no
label, report, or lesion annotation is consulted.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image


Box = tuple[int, int, int, int]


@dataclass(frozen=True)
class SparseFocalSelection:
    """Cacheable image-space selection produced without labels or reports."""

    foreground_box: Box
    tile_boxes: tuple[Box, ...]
    tile_roles: tuple[str, ...]


@dataclass(frozen=True)
class SparseFocalViews:
    """Inspectable output of the sparse focal preprocessing pipeline."""

    foreground_box: Box
    foreground_image: Image.Image
    global_image: Image.Image
    canvas: Image.Image
    candidate_boxes: tuple[Box, ...]
    candidate_canvas_boxes: tuple[Box, ...]
    candidate_scores: tuple[float, ...]
    candidate_foreground_ratios: tuple[float, ...]
    selected_candidate_indices: tuple[int, ...]
    tile_roles: tuple[str, ...]
    tiles: tuple[Image.Image, ...]
    tile_boxes: tuple[Box, ...]

    @property
    def selection(self) -> SparseFocalSelection:
        return SparseFocalSelection(
            foreground_box=self.foreground_box,
            tile_boxes=self.tile_boxes,
            tile_roles=self.tile_roles,
        )


def _border_pixels(image: Image.Image, border_fraction: float) -> np.ndarray:
    array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    height, width = array.shape[:2]
    band = max(1, round(min(height, width) * border_fraction))
    return np.concatenate(
        [
            array[:band].reshape(-1, 3),
            array[-band:].reshape(-1, 3),
            array[:, :band].reshape(-1, 3),
            array[:, -band:].reshape(-1, 3),
        ],
        axis=0,
    )


def estimate_border_color(
    image: Image.Image,
    border_fraction: float = 0.04,
) -> tuple[int, int, int]:
    """Estimate scanner-background color robustly from the image perimeter."""
    if not 0 < border_fraction <= 0.25:
        raise ValueError("border_fraction must be in (0, 0.25].")
    median = np.median(_border_pixels(image, border_fraction), axis=0)
    return tuple(int(round(value)) for value in median)


def _resolve_pad_color(
    image: Image.Image,
    pad_value: str | int | tuple[int, int, int],
    border_fraction: float,
) -> tuple[int, int, int]:
    if isinstance(pad_value, str):
        value = pad_value.lower()
        if value == "border_median":
            return estimate_border_color(image, border_fraction)
        if value == "black":
            return (0, 0, 0)
        raise ValueError("pad_value must be 'border_median', 'black', an int, or RGB.")
    if isinstance(pad_value, int):
        value = min(255, max(0, pad_value))
        return (value, value, value)
    if len(pad_value) != 3:
        raise ValueError("RGB pad_value must contain exactly three values.")
    return tuple(min(255, max(0, int(value))) for value in pad_value)


def letterbox_square(
    image: Image.Image,
    size: int | None = None,
    pad_value: str | int | tuple[int, int, int] = "border_median",
    border_fraction: float = 0.04,
) -> Image.Image:
    """Resize and pad to a square while preserving the source aspect ratio."""
    source = image.convert("RGB")
    side = int(size) if size is not None else max(source.size)
    if side < 1:
        raise ValueError("size must be positive.")

    scale = min(side / source.width, side / source.height)
    resized_size = (
        max(1, min(side, round(source.width * scale))),
        max(1, min(side, round(source.height * scale))),
    )
    # Match the bicubic interpolation used by OpenCLIP/BiomedCLIP transforms.
    resized = source.resize(resized_size, resample=Image.Resampling.BICUBIC)
    canvas = Image.new(
        "RGB",
        (side, side),
        color=_resolve_pad_color(source, pad_value, border_fraction),
    )
    canvas.paste(
        resized,
        ((side - resized.width) // 2, (side - resized.height) // 2),
    )
    return canvas


def _stable_extent(
    active: np.ndarray,
    min_run_ratio: float,
) -> tuple[int, int] | None:
    indices = np.flatnonzero(active)
    if indices.size == 0:
        return None

    breaks = np.flatnonzero(np.diff(indices) > 1) + 1
    runs = np.split(indices, breaks)
    minimum_run = max(1, round(active.size * min_run_ratio))
    retained = [run for run in runs if run.size >= minimum_run]
    if not retained:
        retained = [max(runs, key=lambda run: run.size)]
    return int(min(run[0] for run in retained)), int(max(run[-1] for run in retained) + 1)


def find_foreground_box(
    image: Image.Image,
    margin: float = 0.04,
    border_fraction: float = 0.04,
    min_contrast: float = 0.035,
    min_axis_coverage: float = 0.01,
    min_run_ratio: float = 0.03,
    min_dimension_ratio: float = 0.35,
    min_area_ratio: float = 0.20,
    min_energy_retained: float = 0.95,
    mask_max_side: int = 512,
) -> Box:
    """Find a conservative anatomy foreground box using only image pixels."""
    if not 0 <= margin <= 0.25:
        raise ValueError("margin must be in [0, 0.25].")
    if not 0 < min_contrast < 1:
        raise ValueError("min_contrast must be in (0, 1).")
    if not 0 < min_axis_coverage <= 1:
        raise ValueError("min_axis_coverage must be in (0, 1].")
    if not 0 < min_dimension_ratio <= 1:
        raise ValueError("min_dimension_ratio must be in (0, 1].")
    if not 0 < min_area_ratio <= 1:
        raise ValueError("min_area_ratio must be in (0, 1].")
    if not 0 < min_energy_retained <= 1:
        raise ValueError("min_energy_retained must be in (0, 1].")
    if mask_max_side < 32:
        raise ValueError("mask_max_side must be at least 32.")

    source = image.convert("L")
    scale = min(1.0, mask_max_side / max(source.size))
    if scale < 1.0:
        working = source.resize(
            (max(1, round(source.width * scale)), max(1, round(source.height * scale))),
            resample=Image.Resampling.BILINEAR,
        )
    else:
        working = source

    array = np.asarray(working, dtype=np.float32)
    band = max(1, round(min(array.shape) * border_fraction))
    border = np.concatenate(
        [
            array[:band].ravel(),
            array[-band:].ravel(),
            array[:, :band].ravel(),
            array[:, -band:].ravel(),
        ]
    )
    background = float(np.median(border))
    mad = float(np.median(np.abs(border - background)))
    threshold = max(255.0 * min_contrast, 4.0 * 1.4826 * mad)
    mask = np.abs(array - background) >= threshold

    row_extent = _stable_extent(mask.mean(axis=1) >= min_axis_coverage, min_run_ratio)
    col_extent = _stable_extent(mask.mean(axis=0) >= min_axis_coverage, min_run_ratio)
    if row_extent is None or col_extent is None:
        return (0, 0, image.width, image.height)

    work_height, work_width = array.shape
    x1 = int(np.floor(col_extent[0] * image.width / work_width))
    x2 = int(np.ceil(col_extent[1] * image.width / work_width))
    y1 = int(np.floor(row_extent[0] * image.height / work_height))
    y2 = int(np.ceil(row_extent[1] * image.height / work_height))

    box_width = max(1, x2 - x1)
    box_height = max(1, y2 - y1)
    margin_x = round(box_width * margin)
    margin_y = round(box_height * margin)
    x1 = max(0, x1 - margin_x)
    y1 = max(0, y1 - margin_y)
    x2 = min(image.width, x2 + margin_x)
    y2 = min(image.height, y2 + margin_y)

    # A tiny component is more likely a marker than the anatomy. Fall back to
    # the complete image instead of risking removal of clinically useful pixels.
    if (
        (x2 - x1) < min_dimension_ratio * image.width
        or (y2 - y1) < min_dimension_ratio * image.height
    ):
        return (0, 0, image.width, image.height)
    if (x2 - x1) * (y2 - y1) < min_area_ratio * image.width * image.height:
        return (0, 0, image.width, image.height)

    work_x1 = max(0, int(np.floor(x1 * work_width / image.width)))
    work_x2 = min(work_width, int(np.ceil(x2 * work_width / image.width)))
    work_y1 = max(0, int(np.floor(y1 * work_height / image.height)))
    work_y2 = min(work_height, int(np.ceil(y2 * work_height / image.height)))
    contrast_energy = np.abs(array - background)
    total_energy = float(contrast_energy.sum())
    retained_energy = float(
        contrast_energy[work_y1:work_y2, work_x1:work_x2].sum()
    )
    if total_energy > 1e-8 and retained_energy / total_energy < min_energy_retained:
        return (0, 0, image.width, image.height)
    return (x1, y1, x2, y2)


def crop_foreground(
    image: Image.Image,
    config: Mapping[str, Any] | None = None,
) -> tuple[Image.Image, Box]:
    """Crop scanner padding conservatively and return source-image coordinates."""
    config = config or {}
    box = find_foreground_box(
        image,
        margin=float(config.get("margin", 0.04)),
        border_fraction=float(config.get("border_fraction", 0.04)),
        min_contrast=float(config.get("min_contrast", 0.035)),
        min_axis_coverage=float(config.get("min_axis_coverage", 0.01)),
        min_run_ratio=float(config.get("min_run_ratio", 0.03)),
        min_dimension_ratio=float(config.get("min_dimension_ratio", 0.35)),
        min_area_ratio=float(config.get("min_area_ratio", 0.20)),
        min_energy_retained=float(config.get("min_energy_retained", 0.95)),
        mask_max_side=int(config.get("mask_max_side", 512)),
    )
    return image.convert("RGB").crop(box), box


def prepare_global_image(
    image: Image.Image,
    transform: Callable[[Image.Image], Any] | None,
    config: Mapping[str, Any] | None = None,
) -> Any:
    """Apply configured global preprocessing before the encoder transform."""
    config = config or {}
    strategy = str(config.get("strategy", "direct_resize")).lower()
    target_size = int(config.get("target_size", 224))
    pad_value = config.get("pad_value", "border_median")
    border_fraction = float(config.get("border_fraction", 0.04))

    if strategy == "direct_resize":
        source = image.convert("RGB")
    elif strategy == "letterbox":
        source = letterbox_square(
            image,
            size=target_size,
            pad_value=pad_value,
            border_fraction=border_fraction,
        )
    elif strategy == "foreground_letterbox":
        foreground, _ = crop_foreground(image, config.get("foreground", {}))
        source = letterbox_square(
            foreground,
            size=target_size,
            pad_value=pad_value,
            border_fraction=border_fraction,
        )
    else:
        raise ValueError(
            f"Unsupported global preprocessing strategy: {strategy!r}. "
            "Choose 'direct_resize', 'letterbox', or 'foreground_letterbox'."
        )
    return transform(source) if transform else source


def _axis_positions(length: int, window: int, stride: int) -> list[int]:
    if length <= window:
        return [0]
    positions = list(range(0, length - window + 1, stride))
    last = length - window
    if positions[-1] != last:
        positions.append(last)
    return positions


def _resize_long_side(
    image: Image.Image,
    long_side: int,
    allow_upsample: bool,
) -> Image.Image:
    if long_side < 1:
        raise ValueError("canonical_long_side must be positive.")
    scale = long_side / max(image.size)
    if not allow_upsample:
        scale = min(1.0, scale)
    if abs(scale - 1.0) < 1e-8:
        return image.convert("RGB")
    return image.convert("RGB").resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        resample=Image.Resampling.BICUBIC,
    )


def _minmax(values: np.ndarray) -> np.ndarray:
    lower = float(values.min())
    span = float(values.max() - lower)
    if span < 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - lower) / span).astype(np.float32)


def _candidate_statistics(
    canvas: Image.Image,
    boxes: list[Box],
    border_fraction: float,
    min_contrast: float,
) -> tuple[np.ndarray, np.ndarray]:
    gray = np.asarray(canvas.convert("L"), dtype=np.float32) / 255.0
    band = max(1, round(min(gray.shape) * border_fraction))
    border = np.concatenate(
        [gray[:band].ravel(), gray[-band:].ravel(), gray[:, :band].ravel(), gray[:, -band:].ravel()]
    )
    # The border median is unreliable when anatomy touches several edges.  A
    # histogram mode still recovers the dominant scanner-background intensity
    # in that case and does not let a high border MAD erase the whole mask.
    border_histogram, border_edges = np.histogram(border, bins=64, range=(0.0, 1.0))
    mode_index = int(border_histogram.argmax())
    background = float((border_edges[mode_index] + border_edges[mode_index + 1]) / 2.0)
    foreground = np.abs(gray - background) >= min_contrast

    padded = np.pad(gray, 1, mode="edge")
    laplacian = np.abs(
        4.0 * gray
        - padded[1:-1, :-2]
        - padded[1:-1, 2:]
        - padded[:-2, 1:-1]
        - padded[2:, 1:-1]
    )

    coverages: list[float] = []
    laplacian_energy: list[float] = []
    entropy: list[float] = []
    for left, top, right, bottom in boxes:
        crop = gray[top:bottom, left:right]
        coverages.append(float(foreground[top:bottom, left:right].mean()))
        laplacian_energy.append(float(laplacian[top:bottom, left:right].mean()))
        histogram, _ = np.histogram(crop, bins=32, range=(0.0, 1.0))
        probability = histogram.astype(np.float64)
        probability /= max(1.0, probability.sum())
        probability = probability[probability > 0]
        entropy.append(float(-(probability * np.log2(probability)).sum() / 5.0))

    coverage_array = np.asarray(coverages, dtype=np.float32)
    scores = (
        0.55 * _minmax(np.asarray(laplacian_energy, dtype=np.float32))
        + 0.35 * _minmax(np.asarray(entropy, dtype=np.float32))
        + 0.10 * coverage_array
    )
    return scores, coverage_array


def _box_iou(first: Box, second: Box) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    if intersection == 0:
        return 0.0
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    return intersection / max(1, first_area + second_area - intersection)


def _select_candidates(
    boxes: list[Box],
    scores: np.ndarray,
    coverages: np.ndarray,
    num_tiles: int,
    coverage_tiles: int,
    min_foreground_ratio: float,
    nms_iou: float,
    focal_diversity_weight: float,
    canvas_size: tuple[int, int],
) -> tuple[list[int], list[str]]:
    eligible = [
        index for index, coverage in enumerate(coverages)
        if coverage >= min_foreground_ratio
    ]
    if len(eligible) < min(num_tiles, len(boxes)):
        # An uncertain foreground mask must not restrict selection to the first
        # few row-major windows. Fall back to the complete candidate pool.
        eligible = list(range(len(boxes)))
    if not eligible:
        eligible = list(range(len(boxes)))

    width, height = canvas_size
    centers = np.asarray(
        [
            ((left + right) / (2.0 * width), (top + bottom) / (2.0 * height))
            for left, top, right, bottom in boxes
        ],
        dtype=np.float32,
    )
    selected: list[int] = []
    roles: list[str] = []

    coverage_count = min(coverage_tiles, num_tiles, len(eligible))
    aspect_ratio = width / max(1, height)
    if coverage_count == 1:
        coverage_targets = [(0.5, 0.5)]
    elif aspect_ratio >= 1.2:
        positions = np.linspace(
            1.0 / (coverage_count + 1),
            coverage_count / (coverage_count + 1),
            coverage_count,
        )
        coverage_targets = [
            (position, 0.5)
            for position in positions
        ]
    elif aspect_ratio <= (1.0 / 1.2):
        positions = np.linspace(
            1.0 / (coverage_count + 1),
            coverage_count / (coverage_count + 1),
            coverage_count,
        )
        coverage_targets = [
            (0.5, position)
            for position in positions
        ]
    else:
        square_targets = [(0.25, 0.25), (0.75, 0.75), (0.75, 0.25), (0.25, 0.75)]
        coverage_targets = square_targets[:coverage_count]

    for target in coverage_targets:
        remaining = [index for index in eligible if index not in selected]
        if not remaining:
            break
        target_array = np.asarray(target, dtype=np.float32)
        choice = min(
            remaining,
            key=lambda index: (
                float(np.linalg.norm(centers[index] - target_array))
                / (0.5 + 0.5 * float(coverages[index])),
                index,
            ),
        )
        selected.append(choice)
        roles.append("coverage")

    while len(selected) < num_tiles:
        remaining = [
            index for index in eligible
            if index not in selected
            and all(_box_iou(boxes[index], boxes[other]) <= nms_iou for other in selected)
        ]
        if not remaining:
            break
        choice = max(
            remaining,
            key=lambda index: (
                float(scores[index])
                + focal_diversity_weight
                * (
                    min(
                        float(np.linalg.norm(centers[index] - centers[other]))
                        for other in selected
                    ) if selected else 1.0
                ),
                float(scores[index]),
                -index,
            ),
        )
        selected.append(choice)
        roles.append("focal")

    while len(selected) < min(num_tiles, len(eligible)):
        remaining = [index for index in eligible if index not in selected]
        choice = max(
            remaining,
            key=lambda index: (
                min(
                    float(np.linalg.norm(centers[index] - centers[other]))
                    for other in selected
                ) if selected else float(scores[index]),
                float(scores[index]),
                -index,
            ),
        )
        selected.append(choice)
        roles.append("focal")

    # Very small source images can contain fewer distinct 224px candidates than
    # K. Repeating deterministic views keeps the encoder budget and tensor shape
    # fixed without pretending that upsampling introduced new information.
    unique_count = len(selected)
    while len(selected) < num_tiles:
        selected.append(selected[len(selected) % unique_count])
        roles.append("fallback")

    ordered = sorted(
        zip(selected, roles),
        key=lambda item: (
            boxes[item[0]][1],
            boxes[item[0]][0],
            item[0],
            item[1],
        ),
    )
    return [item[0] for item in ordered], [item[1] for item in ordered]


def _canvas_box_to_source(
    box: Box,
    foreground_box: Box,
    foreground_size: tuple[int, int],
    canvas_size: tuple[int, int],
) -> Box:
    source_left, source_top, source_right, source_bottom = foreground_box
    scale_x = foreground_size[0] / canvas_size[0]
    scale_y = foreground_size[1] / canvas_size[1]
    left, top, right, bottom = box
    return (
        max(source_left, source_left + round(left * scale_x)),
        max(source_top, source_top + round(top * scale_y)),
        min(source_right, source_left + round(right * scale_x)),
        min(source_bottom, source_top + round(bottom * scale_y)),
    )


def build_sparse_focal_views(
    image: Image.Image,
    config: Mapping[str, Any] | None = None,
) -> SparseFocalViews:
    """Create one global view and exactly K coverage/focal local views."""
    config = config or {}
    strategy = str(config.get("strategy", "sparse_focal")).lower()
    if strategy != "sparse_focal":
        raise ValueError(
            f"Unsupported high-resolution strategy: {strategy!r}. "
            "The legacy uniform grid has been removed; use 'sparse_focal'."
        )

    tile_size = int(config.get("tile_size", 224))
    global_size = int(config.get("global_size", 224))
    canonical_long_side = int(config.get("canonical_long_side", 896))
    candidate_stride = int(config.get("candidate_stride", 168))
    num_tiles = int(config.get("num_tiles", 4))
    coverage_tiles = int(config.get("coverage_tiles", 2))
    min_foreground_ratio = float(config.get("min_foreground_ratio", 0.60))
    nms_iou = float(config.get("nms_iou", 0.30))
    focal_diversity_weight = float(config.get("focal_diversity_weight", 0.35))
    border_fraction = float(config.get("border_fraction", 0.04))
    min_contrast = float(config.get("min_contrast", 0.035))
    pad_value = config.get("pad_value", "border_median")

    if tile_size < 1 or global_size < 1 or candidate_stride < 1:
        raise ValueError("tile_size, global_size, and candidate_stride must be positive.")
    if num_tiles < 1:
        raise ValueError("num_tiles must be positive.")
    if not 0 <= coverage_tiles <= num_tiles:
        raise ValueError("coverage_tiles must be between zero and num_tiles.")
    if not 0 <= min_foreground_ratio <= 1:
        raise ValueError("min_foreground_ratio must be in [0, 1].")
    if not 0 <= nms_iou <= 1:
        raise ValueError("nms_iou must be in [0, 1].")
    if focal_diversity_weight < 0:
        raise ValueError("focal_diversity_weight must be non-negative.")

    original = image.convert("RGB")
    foreground, foreground_box = crop_foreground(
        original,
        config.get("foreground", {}),
    )
    global_image = letterbox_square(
        foreground,
        size=global_size,
        pad_value=pad_value,
        border_fraction=border_fraction,
    )
    canvas = _resize_long_side(
        foreground,
        canonical_long_side,
        bool(config.get("allow_upsample", False)),
    )

    window_width = min(tile_size, canvas.width)
    window_height = min(tile_size, canvas.height)
    candidate_canvas_boxes = [
        (left, top, left + window_width, top + window_height)
        for top in _axis_positions(canvas.height, window_height, candidate_stride)
        for left in _axis_positions(canvas.width, window_width, candidate_stride)
    ]
    scores, coverages = _candidate_statistics(
        canvas,
        candidate_canvas_boxes,
        border_fraction,
        min_contrast,
    )
    selected_indices, roles = _select_candidates(
        candidate_canvas_boxes,
        scores,
        coverages,
        num_tiles,
        coverage_tiles,
        min_foreground_ratio,
        nms_iou,
        focal_diversity_weight,
        canvas.size,
    )

    candidate_boxes = [
        _canvas_box_to_source(
            box,
            foreground_box,
            foreground.size,
            canvas.size,
        )
        for box in candidate_canvas_boxes
    ]
    tile_boxes = [candidate_boxes[index] for index in selected_indices]
    tiles = [
        letterbox_square(
            original.crop(box),
            size=tile_size,
            pad_value=pad_value,
            border_fraction=border_fraction,
        )
        for box in tile_boxes
    ]

    return SparseFocalViews(
        foreground_box=foreground_box,
        foreground_image=foreground,
        global_image=global_image,
        canvas=canvas,
        candidate_boxes=tuple(candidate_boxes),
        candidate_canvas_boxes=tuple(candidate_canvas_boxes),
        candidate_scores=tuple(float(value) for value in scores),
        candidate_foreground_ratios=tuple(float(value) for value in coverages),
        selected_candidate_indices=tuple(selected_indices),
        tile_roles=tuple(roles),
        tiles=tuple(tiles),
        tile_boxes=tuple(tile_boxes),
    )


def normalize_tile_boxes(
    boxes: list[Box] | tuple[Box, ...],
    image_size: tuple[int, int],
) -> torch.Tensor:
    """Normalize source-image XYXY boxes to the [0, 1] interval."""
    width, height = image_size
    scale = torch.tensor([width, height, width, height], dtype=torch.float32)
    return torch.tensor(boxes, dtype=torch.float32) / scale.clamp_min(1.0)


def _render_cached_selection(
    image: Image.Image,
    config: Mapping[str, Any],
    selection: SparseFocalSelection,
) -> tuple[Image.Image, tuple[Image.Image, ...]]:
    """Render global/local PIL views from cached coordinates without rescoring."""
    original = image.convert("RGB")
    expected_tiles = int(config.get("num_tiles", 4))
    if len(selection.tile_boxes) != expected_tiles:
        raise ValueError(
            f"Cached selection has {len(selection.tile_boxes)} tiles; "
            f"configuration requires {expected_tiles}."
        )
    for left, top, right, bottom in (
        selection.foreground_box,
        *selection.tile_boxes,
    ):
        if not (
            0 <= left < right <= original.width
            and 0 <= top < bottom <= original.height
        ):
            raise ValueError("Cached preprocessing coordinates exceed the source image.")

    tile_size = int(config.get("tile_size", 224))
    global_size = int(config.get("global_size", 224))
    pad_value = config.get("pad_value", "border_median")
    border_fraction = float(config.get("border_fraction", 0.04))
    foreground = original.crop(selection.foreground_box)
    global_image = letterbox_square(
        foreground,
        size=global_size,
        pad_value=pad_value,
        border_fraction=border_fraction,
    )
    tiles = tuple(
        letterbox_square(
            original.crop(box),
            size=tile_size,
            pad_value=pad_value,
            border_fraction=border_fraction,
        )
        for box in selection.tile_boxes
    )
    return global_image, tiles


def prepare_high_resolution_inputs(
    image: Image.Image,
    transform: Callable[[Image.Image], Any] | None,
    config: Mapping[str, Any],
    selection: SparseFocalSelection | None = None,
    return_selection: bool = False,
) -> dict[str, Any] | tuple[dict[str, Any], SparseFocalSelection]:
    """Build global image, fixed-K local tiles, and normalized spatial boxes."""
    strategy = str(config.get("strategy", "sparse_focal")).lower()
    if strategy != "sparse_focal":
        raise ValueError(
            f"Unsupported high-resolution strategy: {strategy!r}. "
            "The legacy uniform grid has been removed; use 'sparse_focal'."
        )

    if selection is None:
        views = build_sparse_focal_views(image, config)
        selection = views.selection
        global_source = views.global_image
        tile_sources = views.tiles
    else:
        global_source, tile_sources = _render_cached_selection(
            image,
            config,
            selection,
        )

    global_image = transform(global_source) if transform else global_source
    tiles = [transform(tile) if transform else tile for tile in tile_sources]
    tile_values = torch.stack(tiles) if isinstance(tiles[0], torch.Tensor) else tiles
    fields = {
        "pixel_values": global_image,
        "tile_values": tile_values,
        "tile_boxes": normalize_tile_boxes(selection.tile_boxes, image.size),
    }
    return (fields, selection) if return_selection else fields
