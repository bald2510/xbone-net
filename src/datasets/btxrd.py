"""BTXRD dataset with deterministic high-resolution grid tiling."""

from __future__ import annotations

import math
import os

import pandas as pd
import torch
from PIL import Image, ImageStat
from torch.utils.data import Dataset


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
    """Cover the full radiograph with uniform tiles and remove only blank crops.

    No anatomy mask, disease score, proposal ranking, or ground truth is used.
    When an image exceeds ``max_tiles``, it is isotropically downscaled just
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
    """Normalize source-image XYXY boxes to [0, 1]."""
    width, height = image_size
    scale = torch.tensor([width, height, width, height], dtype=torch.float32)
    return torch.tensor(boxes, dtype=torch.float32) / scale.clamp_min(1.0)


class BTXRDDataset(Dataset):
    """Pair BTXRD radiographs with reports and multiclass labels."""

    def __init__(
        self,
        img_dir: str,
        report_dir: str,
        csv_split_path: str,
        csv_labels_path: str,
        pathologies: list | None = None,
        classes: list | None = None,
        task_type: str = "multiclass",
        num_classes: int | None = None,
        split: str = "train",
        train_ratio: float = 1.0,
        transform=None,
        tokenizer=None,
        max_text_len: int = 256,
        clinical_subdir: str = "clinical_v2",
        k_shot: int | None = None,
        seed: int = 42,
        **kwargs,
    ):
        del num_classes
        self.img_dir = img_dir
        self.report_dir = report_dir
        self.classes = classes or pathologies or []
        self.task_type = task_type
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len

        split_frame = pd.read_csv(csv_split_path)
        label_frame = pd.read_csv(csv_labels_path)
        merged = pd.merge(split_frame, label_frame, on=["image_id"], how="inner")

        if "class_id" not in merged.columns and self.classes:
            def class_id(row):
                for index, name in enumerate(self.classes):
                    if row.get(name, 0) == 1:
                        return index
                return 0
            merged["class_id"] = merged.apply(class_id, axis=1)

        current_split = "validate" if split == "val" else split
        filtered = merged[merged["split"] == current_split].reset_index(drop=True)
        if current_split == "train" and k_shot is not None:
            filtered = filtered.groupby("class_id", group_keys=False).apply(
                lambda group: group.sample(
                    n=min(len(group), k_shot), random_state=seed
                )
            ).reset_index(drop=True)
            print(f"[Dataset] Few-shot learning: {k_shot}-shot (seed={seed}).")
        elif current_split == "train" and train_ratio < 1.0:
            filtered = filtered.sample(frac=train_ratio, random_state=seed).reset_index(
                drop=True
            )
            print(f"[Dataset] Training subset: {train_ratio * 100:.1f}%.")
        self.df = filtered

        self.xray_dir = os.path.join(report_dir, "xray")
        self.clinical_dir = os.path.join(report_dir, clinical_subdir)
        self.has_dual_reports = os.path.isdir(self.xray_dir) and os.path.isdir(
            self.clinical_dir
        )

        self.high_res_cfg = kwargs.get("high_res", {}) or {}
        self.use_high_res = bool(self.high_res_cfg.get("enabled", False))
        self.text_only = bool(kwargs.get("text_only", False))
        self.shuffle_reports = bool(kwargs.get("shuffle_reports", False))
        if self.shuffle_reports:
            import numpy as np
            self.shuffled_report_indices = np.random.default_rng(seed).permutation(
                len(self.df)
            )
        else:
            self.shuffled_report_indices = None

        print(
            f"[Dataset] Initialized '{split.upper()}' with {len(self.df)} samples. "
            f"Dual reports: {self.has_dual_reports}. Uniform high-res grid: "
            f"{self.use_high_res}."
        )

    def __len__(self):
        return len(self.df)

    def _load_and_tokenize(self, path: str, default_text: str):
        raw_text = ""
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as report_file:
                raw_text = report_file.read()
        text = raw_text.strip().lower() or default_text
        return self.tokenizer([text]).squeeze(0) if self.tokenizer else text

    def _label(self, row) -> torch.Tensor:
        if self.task_type == "multiclass":
            return torch.tensor(int(row["class_id"]), dtype=torch.long)
        return torch.tensor(
            [float(row.get(name, 0.0)) for name in self.classes],
            dtype=torch.float32,
        )

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        image_id = str(row["image_id"])
        stem = os.path.splitext(image_id)[0]
        image_path = os.path.join(self.img_dir, image_id)

        if self.text_only:
            image = Image.new("RGB", (224, 224), color="black")
        else:
            try:
                image = Image.open(image_path).convert("RGB")
            except FileNotFoundError:
                image = Image.new("RGB", (224, 224), color="black")

        report_stem = stem
        if self.shuffled_report_indices is not None:
            shuffled_id = str(
                self.df.iloc[self.shuffled_report_indices[index]]["image_id"]
            )
            report_stem = os.path.splitext(shuffled_id)[0]

        labels = self._label(row)
        high_res_fields = {}
        if self.use_high_res:
            global_source = letterbox_square(image)
            global_image = self.transform(global_source) if self.transform else global_source
            raw_tiles, absolute_boxes = make_uniform_grid_tiles(
                image,
                tile_size=int(self.high_res_cfg.get("tile_size", 224)),
                stride=int(self.high_res_cfg.get("stride", 224)),
                max_tiles=int(self.high_res_cfg.get("max_tiles", 96)),
                uniform_std_threshold=float(
                    self.high_res_cfg.get("uniform_std_threshold", 0.01)
                ),
                return_boxes=True,
            )
            tiles = [self.transform(tile) if self.transform else tile for tile in raw_tiles]
            high_res_fields = {
                "pixel_values": global_image,
                "tile_values": torch.stack(tiles)
                if isinstance(tiles[0], torch.Tensor)
                else tiles,
                "tile_boxes": normalize_tile_boxes(absolute_boxes, image.size),
            }
        else:
            image = self.transform(image) if self.transform else image

        if self.has_dual_reports:
            xray_ids = self._load_and_tokenize(
                os.path.join(self.xray_dir, f"{report_stem}.txt"),
                "no clear bone abnormalities or fracture identified.",
            )
            clinical_ids = self._load_and_tokenize(
                os.path.join(self.clinical_dir, f"{report_stem}.txt"),
                "no clinical information available.",
            )
            if self.use_high_res:
                return {
                    **high_res_fields,
                    "xray_input_ids": xray_ids,
                    "clinical_input_ids": clinical_ids,
                    "labels": labels,
                }
            return image, xray_ids, clinical_ids, labels

        input_ids = self._load_and_tokenize(
            os.path.join(self.report_dir, f"{report_stem}.txt"),
            "no clear bone abnormalities or fracture identified.",
        )
        if self.use_high_res:
            return {**high_res_fields, "input_ids": input_ids, "labels": labels}
        return image, input_ids, labels
