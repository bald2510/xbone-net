"""Visualize uniform high-resolution tiling and BiomedCLIP local-token layout."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.btxrd import letterbox_square, make_uniform_grid_tiles


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
                edgecolor="magenta",
                linewidth=2.0,
            )
        )
        axis.text(
            left,
            top,
            f"GT: {label}",
            color="white",
            fontsize=8,
            bbox={"facecolor": "magenta", "alpha": 0.8, "pad": 2},
        )


def render_preprocessing(
    source: Image.Image,
    tiles: list[Image.Image],
    boxes: list[tuple[int, int, int, int]],
    annotations,
    output: Path,
    local_pool_grid: int,
    dpi: int,
):
    columns = 8
    tile_rows = math.ceil(len(tiles) / columns)
    figure = plt.figure(figsize=(20, 8 + 2.4 * tile_rows))
    layout = figure.add_gridspec(2 + tile_rows, columns)

    axis_source = figure.add_subplot(layout[0:2, 0:3])
    axis_source.imshow(source)
    for index, box in enumerate(boxes, start=1):
        left, top, right, bottom = box
        axis_source.add_patch(
            Rectangle(
                (left, top), right - left, bottom - top,
                fill=False, edgecolor="cyan", linewidth=0.7,
            )
        )
        axis_source.text(left, top, str(index), color="yellow", fontsize=6)
    draw_ground_truth(axis_source, annotations)
    axis_source.set_title(
        f"1. Original {source.width}×{source.height}\n"
        f"uniform full-coverage grid: {len(tiles)} tiles"
    )
    axis_source.axis("off")

    axis_global = figure.add_subplot(layout[0:2, 3:5])
    axis_global.imshow(letterbox_square(source))
    axis_global.set_title("2. Global branch\nletterbox → BiomedCLIP 224×224")
    axis_global.axis("off")

    axis_tokens = figure.add_subplot(layout[0:2, 5:8])
    axis_tokens.imshow(source)
    for box in boxes:
        left, top, right, bottom = box
        for row in range(local_pool_grid):
            for column in range(local_pool_grid):
                patch_left = left + (right - left) * column / local_pool_grid
                patch_top = top + (bottom - top) * row / local_pool_grid
                patch_right = left + (right - left) * (column + 1) / local_pool_grid
                patch_bottom = top + (bottom - top) * (row + 1) / local_pool_grid
                axis_tokens.add_patch(
                    Rectangle(
                        (patch_left, patch_top),
                        patch_right - patch_left,
                        patch_bottom - patch_top,
                        fill=False,
                        edgecolor="lime",
                        linewidth=0.55,
                    )
                )
    draw_ground_truth(axis_tokens, annotations)
    axis_tokens.set_title(
        f"3. Local-token coordinates\n"
        f"14×14 patch grid → {local_pool_grid}×{local_pool_grid} tokens/tile"
    )
    axis_tokens.axis("off")

    for index, (tile, box) in enumerate(zip(tiles, boxes)):
        row = 2 + index // columns
        column = index % columns
        axis = figure.add_subplot(layout[row, column])
        axis.imshow(tile)
        axis.set_title(f"T{index + 1}: {box[:2]}", fontsize=8)
        axis.axis("off")

    figure.suptitle(
        "High-resolution preprocessing: uniform tiles → BiomedCLIP local patches → learned resampler",
        fontsize=16,
    )
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--annotation")
    parser.add_argument("--output", default="outputs/visualizations/uniform_grid_preprocessing.png")
    parser.add_argument("--tile-size", type=int, default=224)
    parser.add_argument("--stride", type=int, default=224)
    parser.add_argument("--max-tiles", type=int, default=64)
    parser.add_argument("--uniform-std-threshold", type=float, default=0.01)
    parser.add_argument("--local-pool-grid", type=int, default=2)
    parser.add_argument("--dpi", type=int, default=140)
    args = parser.parse_args()

    image_path = Path(args.image).expanduser().resolve()
    with Image.open(image_path) as opened:
        source = opened.convert("RGB")
    tiles, boxes = make_uniform_grid_tiles(
        source,
        tile_size=args.tile_size,
        stride=args.stride,
        max_tiles=args.max_tiles,
        uniform_std_threshold=args.uniform_std_threshold,
        return_boxes=True,
    )
    annotation_path = (
        Path(args.annotation).expanduser().resolve()
        if args.annotation
        else PROJECT_ROOT / "data" / "BTXRD" / "Annotations" / f"{image_path.stem}.json"
    )
    annotations = load_ground_truth_annotations(annotation_path)
    output = Path(args.output).expanduser().resolve()
    render_preprocessing(
        source,
        tiles,
        boxes,
        annotations,
        output,
        args.local_pool_grid,
        args.dpi,
    )
    print(f"Saved: {output}")
    print(f"Tiles: {len(tiles)}; local tokens before resampler: {len(tiles) * args.local_pool_grid ** 2}")


if __name__ == "__main__":
    main()
