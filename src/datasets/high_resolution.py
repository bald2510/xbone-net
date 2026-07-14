"""Shared deterministic high-resolution preprocessing for X-ray datasets."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Callable

import torch
from PIL import Image, ImageStat


def letterbox_square(image: Image.Image) -> Image.Image:
    """Pad an image to a square without changing its aspect ratio."""
    image = image.convert("RGB")
    side = max(image.size)
    canvas = Image.new("RGB", (side, side), color="black")
    canvas.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
    return canvas


def _axis_positions(length: int, tile_size: int, stride: int) -> list[int]:
    """Return deterministic positions including an end-aligned border tile."""
    if length <= tile_size:
        return [0]
    positions = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if positions[-1] != last:
        positions.append(last)
    return positions


def _grid_count(width: int, height: int, tile_size: int, stride: int) -> int:
    return len(_axis_positions(width, tile_size, stride)) * len(
        _axis_positions(height, tile_size, stride)
    )


def _fit_to_tile_budget(
    image: Image.Image,
    tile_size: int,
    stride: int,
    max_tiles: int,
) -> tuple[Image.Image, float]:
    """Downscale only when needed so a full-coverage grid fits the tile budget."""
    source = image.convert("RGB")
    if max_tiles <= 0:
        return source, 1.0

    width, height = source.size
    count = _grid_count(max(width, tile_size), max(height, tile_size), tile_size, stride)
    if count <= max_tiles:
        return source, 1.0

    scale = min(1.0, math.sqrt(max_tiles / count))
    while True:
        resized_width = max(1, round(width * scale))
        resized_height = max(1, round(height * scale))
        if _grid_count(
            max(resized_width, tile_size),
            max(resized_height, tile_size),
            tile_size,
            stride,
        ) <= max_tiles:
            break
        scale *= 0.97

    resized = source.resize(
        (resized_width, resized_height),
        resample=Image.Resampling.LANCZOS,
    )
    return resized, scale


def make_uniform_grid_tiles(
    image: Image.Image,
    tile_size: int = 224,
    stride: int = 224,
    max_tiles: int = 96,
    uniform_std_threshold: float = 0.01,
    return_boxes: bool = False,
):
    """Cover an X-ray with a full grid and remove only near-uniform crops.

    No anatomy mask, disease score, proposal ranking, or ground-truth annotation
    is used. Images that exceed ``max_tiles`` are isotropically downscaled just
    enough for the complete grid to fit the budget.
    """
    if tile_size < 1 or stride < 1:
        raise ValueError("tile_size and stride must be positive.")
    if uniform_std_threshold < 0:
        raise ValueError("uniform_std_threshold must be non-negative.")

    original = image.convert("RGB")
    tiled_source, scale = _fit_to_tile_budget(
        original, tile_size, stride, int(max_tiles)
    )
    source_width, source_height = tiled_source.size
    canvas_width = max(source_width, tile_size)
    canvas_height = max(source_height, tile_size)
    if (canvas_width, canvas_height) != tiled_source.size:
        canvas = Image.new("RGB", (canvas_width, canvas_height), color="black")
        canvas.paste(tiled_source, (0, 0))
        tiled_source = canvas

    x_positions = _axis_positions(canvas_width, tile_size, stride)
    y_positions = _axis_positions(canvas_height, tile_size, stride)
    threshold = float(uniform_std_threshold) * 255.0
    tiles: list[Image.Image] = []
    boxes: list[tuple[int, int, int, int]] = []

    for top in y_positions:
        for left in x_positions:
            crop_box = (left, top, left + tile_size, top + tile_size)
            tile = tiled_source.crop(crop_box)
            if ImageStat.Stat(tile.convert("L")).stddev[0] < threshold:
                continue

            original_box = (
                max(0, round(left / scale)),
                max(0, round(top / scale)),
                min(original.width, round((left + tile_size) / scale)),
                min(original.height, round((top + tile_size) / scale)),
            )
            if original_box[2] <= original_box[0] or original_box[3] <= original_box[1]:
                continue
            tiles.append(tile)
            boxes.append(original_box)

    if not tiles:
        tiles = [letterbox_square(original).resize((tile_size, tile_size))]
        boxes = [(0, 0, original.width, original.height)]

    return (tiles, boxes) if return_boxes else tiles


def normalize_tile_boxes(
    boxes: list[tuple[int, int, int, int]],
    image_size: tuple[int, int],
) -> torch.Tensor:
    """Normalize source-image XYXY boxes to the [0, 1] interval."""
    width, height = image_size
    scale = torch.tensor([width, height, width, height], dtype=torch.float32)
    return torch.tensor(boxes, dtype=torch.float32) / scale.clamp_min(1.0)


def prepare_high_resolution_inputs(
    image: Image.Image,
    transform: Callable[[Image.Image], Any] | None,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the common global-image, local-tile, and spatial-box fields."""
    strategy = str(config.get("strategy", "uniform_grid"))
    if strategy != "uniform_grid":
        raise ValueError(
            f"Unsupported high-resolution strategy: {strategy!r}. "
            "Only 'uniform_grid' is implemented."
        )

    global_source = letterbox_square(image)
    global_image = transform(global_source) if transform else global_source
    raw_tiles, absolute_boxes = make_uniform_grid_tiles(
        image,
        tile_size=int(config.get("tile_size", 224)),
        stride=int(config.get("stride", 224)),
        max_tiles=int(config.get("max_tiles", 96)),
        uniform_std_threshold=float(config.get("uniform_std_threshold", 0.01)),
        return_boxes=True,
    )
    tiles = [transform(tile) if transform else tile for tile in raw_tiles]
    tile_values = torch.stack(tiles) if isinstance(tiles[0], torch.Tensor) else tiles

    return {
        "pixel_values": global_image,
        "tile_values": tile_values,
        "tile_boxes": normalize_tile_boxes(absolute_boxes, image.size),
    }
