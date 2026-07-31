"""Batch explainability and representation audit for CTCH proposed checkpoints."""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image, ImageFilter
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.datasets.ctch import CTCHDataset
from src.models.fusion.cross_attention import reduce_attention_to_keys
from src.utils.analysis import (
    AnalysisDataCollator,
    SOURCE_EXPERIMENT,
    analysis_root,
    load_feature_archive,
    load_locked_proposed_model,
)
from src.utils.explainability import (
    COMPONENT_NAMES,
    branch_ablation,
    curve_auc,
    forward_from_local,
    global_image_perturbation_curves,
    gradient_text_relevance,
    gradient_visual_relevance,
    integrated_gradients,
    integrated_gradients_global_image,
    integrated_gradients_text,
    perturbation_curves,
    rank_correlation,
    representation_quality_metrics,
    stratified_sample_indices,
    text_input_perturbation_curves,
    text_perturbation_curves,
)


REPRESENTATIONS = (
    "fused_embeddings",
    "visual_global_embeddings",
    "visual_local_summary_embeddings",
    "text_global_embeddings",
    "image_from_text_embeddings",
    "text_from_image_embeddings",
)


def _render_representation_overview(
    train: dict[str, np.ndarray],
    test: dict[str, np.ndarray],
    metrics: dict[str, dict[str, float]],
    output: Path,
    seed: int,
) -> None:
    """Render deterministic PCA views; quantitative claims use full dimensions."""
    figure, axes = plt.subplots(2, 3, figsize=(17, 10), facecolor="white")
    labels = np.asarray(test["labels"], dtype=np.int64)
    scatter = None
    for axis, key in zip(axes.flat, REPRESENTATIONS):
        train_values = np.asarray(train[key], dtype=np.float64)
        test_values = np.asarray(test[key], dtype=np.float64)
        combined = np.vstack([train_values, test_values])
        projected = PCA(n_components=2, random_state=seed).fit_transform(combined)
        test_projected = projected[len(train_values):]
        scatter = axis.scatter(
            test_projected[:, 0],
            test_projected[:, 1],
            c=labels,
            cmap="turbo",
            s=12,
            alpha=0.68,
            linewidths=0,
        )
        axis.set_title(
            f"{key.replace('_embeddings', '')}\n"
            f"sil={metrics[key]['silhouette_cosine']:.3f}, "
            f"NC-BAcc={metrics[key]['nearest_train_centroid_balanced_accuracy']:.3f}"
        )
        axis.set_xticks([])
        axis.set_yticks([])
        axis.grid(alpha=0.15)
    if scatter is not None:
        colorbar = figure.colorbar(scatter, ax=axes.ravel().tolist(), shrink=0.72)
        colorbar.set_label("CTCH class ID")
    figure.suptitle(
        "CTCH proposed representation geometry (PCA shown; metrics use full 512-D features)",
        fontsize=14,
    )
    figure.subplots_adjust(top=0.91, wspace=0.16, hspace=0.24, right=0.92)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(figure)


def _ctch_test_dataset(loaded) -> CTCHDataset:
    params = dict(OmegaConf.to_container(loaded.cfg.dataset.params, resolve=True))
    return CTCHDataset(
        split="test",
        transform=loaded.model.backbone.preprocess,
        tokenizer=loaded.model.backbone.tokenizer_obj,
        **params,
    )


def _one_sample(sample: dict[str, Any], loaded) -> dict[str, Any]:
    batch = AnalysisDataCollator(loaded.model.backbone.tokenizer_obj)([sample])
    return {
        key: value.to(loaded.device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _cached_tokens(model, batch: dict[str, Any]) -> dict[str, Any]:
    with torch.no_grad():
        _, text_tokens = model.backbone(
            batch["pixel_values"],
            batch["clinical_input_ids"],
            attention_mask=batch["clinical_attention_mask"],
            tile_values=batch.get("tile_values"),
            tile_mask=batch.get("tile_mask"),
            tile_boxes=batch.get("tile_boxes"),
        )
    tokenizer = getattr(
        model.backbone.tokenizer_obj,
        "tokenizer",
        model.backbone.tokenizer_obj,
    )
    text_input_ids = batch["clinical_input_ids"]
    special_ids = [
        int(value)
        for value in (getattr(tokenizer, "all_special_ids", []) or [])
    ]
    special_mask = torch.zeros_like(text_input_ids, dtype=torch.bool)
    for token_id in special_ids:
        special_mask |= text_input_ids == token_id
    pad_token_id = getattr(tokenizer, "pad_token_id", 0)
    if pad_token_id is None:
        pad_token_id = 0
    return {
        "pixel_values": batch["pixel_values"].detach(),
        "global_feature": model.backbone.last_global_feature.detach(),
        "local_tokens": model.backbone.last_local_tokens.detach(),
        "local_mask": model.backbone.last_local_mask.detach(),
        "local_boxes": model.backbone.last_local_token_boxes.detach(),
        "text_tokens": text_tokens.detach(),
        "text_input_ids": text_input_ids.detach(),
        "text_attention_mask": batch["clinical_attention_mask"],
        "text_special_token_mask": special_mask.detach(),
        "text_pad_token_id": int(pad_token_id),
    }


def _decode_tokens(tokenizer, token_ids: torch.Tensor) -> list[str]:
    inner = getattr(tokenizer, "tokenizer", tokenizer)
    converter = getattr(inner, "convert_ids_to_tokens", None)
    ids = [int(value) for value in token_ids.detach().cpu().tolist()]
    if callable(converter):
        return [str(value) for value in converter(ids)]
    return [str(value) for value in ids]


def _displayable_token(token: str, tokenizer) -> bool:
    inner = getattr(tokenizer, "tokenizer", tokenizer)
    special = set(getattr(inner, "all_special_tokens", []) or [])
    if token in special:
        return False
    cleaned = token.replace("##", "").replace("▁", "").strip()
    return any(character.isalnum() for character in cleaned)


def _entropy(values: np.ndarray) -> float:
    values = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
    probability = values / max(values.sum(), 1e-12)
    return float(-(probability * np.log(np.maximum(probability, 1e-12))).sum())


def _rasterize(
    boxes: np.ndarray,
    scores: np.ndarray,
    image_size: tuple[int, int],
    max_side: int = 720,
) -> np.ndarray:
    width, height = image_size
    scale = min(1.0, max_side / max(width, height))
    out_width = max(1, round(width * scale))
    out_height = max(1, round(height * scale))
    heatmap = np.zeros((out_height, out_width), dtype=np.float32)
    counts = np.zeros_like(heatmap)
    for box, score in zip(boxes, scores):
        x1, y1 = np.floor(box[:2] * [out_width, out_height]).astype(int)
        x2, y2 = np.ceil(box[2:] * [out_width, out_height]).astype(int)
        x1 = int(np.clip(x1, 0, out_width - 1))
        y1 = int(np.clip(y1, 0, out_height - 1))
        x2 = int(np.clip(x2, x1 + 1, out_width))
        y2 = int(np.clip(y2, y1 + 1, out_height))
        heatmap[y1:y2, x1:x2] += float(score)
        counts[y1:y2, x1:x2] += 1
    heatmap /= np.maximum(counts, 1)
    heatmap -= heatmap.min()
    return heatmap / max(float(heatmap.max()), 1e-8)


def _curve_auc(fractions: np.ndarray, values: np.ndarray) -> float:
    """Return a trapezoidal AUC for a probability intervention curve."""
    return float(
        np.trapezoid(
            np.asarray(values, dtype=np.float64),
            np.asarray(fractions, dtype=np.float64),
        )
    )


def _tile_indices(
    token_count: int,
    tokens_per_tile: int,
    include_local_cls_token: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Separate tile-level CLS tokens from spatial patch tokens."""
    all_indices = np.arange(token_count, dtype=np.int64)
    if not include_local_cls_token:
        return np.asarray([], dtype=np.int64), all_indices
    cls_indices = all_indices[::tokens_per_tile]
    patch_mask = np.ones(token_count, dtype=bool)
    patch_mask[cls_indices] = False
    return cls_indices, all_indices[patch_mask]


def _draw_normalized_boxes(
    axis,
    boxes: np.ndarray,
    indices: np.ndarray,
    image_shape: tuple[int, int],
    *,
    edgecolor: str,
    linewidths: np.ndarray | float = 1.5,
    facecolor: str = "none",
    alpha: float = 1.0,
    hatch: str | None = None,
) -> None:
    """Draw normalized source-space boxes on an image axis."""
    height, width = image_shape
    widths = np.broadcast_to(
        np.asarray(linewidths, dtype=np.float64), (len(indices),)
    )
    for index, linewidth in zip(indices, widths):
        x1, y1, x2, y2 = boxes[int(index)]
        axis.add_patch(
            patches.Rectangle(
                (x1 * width, y1 * height),
                (x2 - x1) * width,
                (y2 - y1) * height,
                fill=facecolor != "none",
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=float(linewidth),
                alpha=alpha,
                hatch=hatch,
            )
        )


def _render_token_faithfulness_pipeline(
    image_path: Path,
    boxes: np.ndarray,
    visual_scores: np.ndarray,
    curves: dict[str, np.ndarray],
    record: dict[str, Any],
    output: Path,
    *,
    tokens_per_tile: int,
    include_local_cls_token: bool,
    intervention_fraction: float = 0.25,
) -> None:
    """Render a RISE-like figure while preserving token-level semantics."""
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    cls_indices, patch_indices = _tile_indices(
        len(visual_scores),
        tokens_per_tile,
        include_local_cls_token,
    )

    spatial_boxes = boxes[patch_indices] if patch_indices.size else boxes
    spatial_scores = (
        visual_scores[patch_indices] if patch_indices.size else visual_scores
    )
    heatmap = _rasterize(spatial_boxes, spatial_scores, image.size)
    resized = image.resize(
        (heatmap.shape[1], heatmap.shape[0]), Image.Resampling.LANCZOS
    )
    plot_height, plot_width = heatmap.shape

    figure, axes = plt.subplots(2, 2, figsize=(10.6, 10.0), facecolor="white")

    axes[0, 0].imshow(resized, cmap="gray")
    _draw_normalized_boxes(
        axes[0, 0],
        boxes,
        cls_indices,
        (plot_height, plot_width),
        edgecolor="#1f77b4",
        linewidths=1.8,
    )
    axes[0, 0].set_title("(a) Ảnh đầu vào và các vùng sparse-focal")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(resized, cmap="gray")
    axes[0, 1].imshow(heatmap, cmap="turbo", alpha=0.58, vmin=0.0, vmax=1.0)
    if cls_indices.size:
        cls_values = np.maximum(visual_scores[cls_indices], 0.0)
        cls_values = cls_values / max(float(cls_values.max()), 1e-8)
        _draw_normalized_boxes(
            axes[0, 1],
            boxes,
            cls_indices,
            (plot_height, plot_width),
            edgecolor="#ff7f0e",
            linewidths=1.0 + 3.0 * cls_values,
        )
    axes[0, 1].set_title(
        "(b) Attribution của token cục bộ\n"
        "Heatmap: token không gian; viền cam: token CLS"
    )
    axes[0, 1].axis("off")

    fractions = np.asarray(curves["fractions"], dtype=np.float64)
    fraction_index = int(np.argmin(np.abs(fractions - intervention_fraction)))
    shown_fraction = float(fractions[fraction_index])
    count = min(
        len(visual_scores),
        int(round(shown_fraction * len(visual_scores))),
    )
    selected = np.argsort(visual_scores)[::-1][:count]
    selected_cls = np.intersect1d(selected, cls_indices)
    selected_patch = np.setdiff1d(selected, selected_cls)

    axes[1, 0].imshow(resized, cmap="gray")
    if selected_patch.size:
        _draw_normalized_boxes(
            axes[1, 0],
            boxes,
            selected_patch,
            (plot_height, plot_width),
            edgecolor="#666666",
            linewidths=0.8,
            facecolor="#777777",
            alpha=0.62,
        )
    if selected_cls.size:
        _draw_normalized_boxes(
            axes[1, 0],
            boxes,
            selected_cls,
            (plot_height, plot_width),
            edgecolor="#222222",
            linewidths=2.0,
            facecolor="#aaaaaa",
            alpha=0.36,
            hatch="///",
        )
    axes[1, 0].set_title(
        f"(c) Vùng tương ứng với token bị vô hiệu hóa "
        f"({shown_fraction:.0%})\n"
        "Minh họa không gian; can thiệp thực hiện trên embedding"
    )
    axes[1, 0].axis("off")

    deletion = np.asarray(curves["delete_most_relevant"], dtype=np.float64)
    random_deletion = np.asarray(curves["delete_random"], dtype=np.float64)
    insertion = np.asarray(curves["insert_most_relevant"], dtype=np.float64)
    percentages = fractions * 100.0
    axes[1, 1].plot(
        percentages,
        deletion,
        "o-",
        color="#1f77b4",
        linewidth=2.0,
        label="Deletion theo attribution",
    )
    axes[1, 1].plot(
        percentages,
        random_deletion,
        "s--",
        color="#ff7f0e",
        linewidth=1.7,
        label="Deletion ngẫu nhiên",
    )
    axes[1, 1].plot(
        percentages,
        insertion,
        "^-",
        color="#2ca02c",
        linewidth=2.0,
        label="Insertion theo attribution",
    )
    axes[1, 1].set_xlim(0.0, 100.0)
    axes[1, 1].set_ylim(0.0, 1.0)
    axes[1, 1].set_xlabel("Tỷ lệ token cục bộ được can thiệp (%)")
    axes[1, 1].set_ylabel("Xác suất lớp mục tiêu")
    axes[1, 1].grid(alpha=0.25)
    axes[1, 1].legend(fontsize=8, loc="best")
    axes[1, 1].set_title("(d) Đường cong faithfulness")
    axes[1, 1].text(
        0.03,
        0.05,
        (
            f"Deletion AUC = {_curve_auc(fractions, deletion):.3f}\n"
            f"Insertion AUC = {_curve_auc(fractions, insertion):.3f}"
        ),
        transform=axes[1, 1].transAxes,
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.86, "edgecolor": "#cccccc"},
    )

    figure.suptitle(
        f"{record['image_id']} | GT={record['ground_truth']} | "
        f"Pred={record['prediction']} | Độ tin cậy={record['confidence']:.1%}",
        fontsize=12,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _render_spatial_saliency_map(
    image_path: Path,
    boxes: np.ndarray,
    visual_scores: np.ndarray,
    record: dict[str, Any],
    output: Path,
    *,
    tokens_per_tile: int,
    include_local_cls_token: bool,
) -> None:
    """Render a simple class-specific saliency map from spatial local tokens."""
    image = Image.open(image_path).convert("RGB")
    _, patch_indices = _tile_indices(
        len(visual_scores),
        tokens_per_tile,
        include_local_cls_token,
    )
    if patch_indices.size == 0:
        raise ValueError("No spatial local tokens are available for saliency.")

    heatmap = _rasterize(
        boxes[patch_indices],
        visual_scores[patch_indices],
        image.size,
    )
    blur_radius = max(4.0, 0.018 * min(heatmap.shape))
    smoothed = Image.fromarray(
        np.uint8(np.clip(heatmap, 0.0, 1.0) * 255.0),
        mode="L",
    ).filter(ImageFilter.GaussianBlur(radius=blur_radius))
    heatmap = np.asarray(smoothed, dtype=np.float32) / 255.0
    low, high = np.percentile(heatmap, [1.0, 99.0])
    heatmap = np.clip(
        (heatmap - float(low)) / max(float(high - low), 1e-8),
        0.0,
        1.0,
    )
    resized = image.resize(
        (heatmap.shape[1], heatmap.shape[0]),
        Image.Resampling.LANCZOS,
    )

    figure, axes = plt.subplots(1, 2, figsize=(9.2, 4.8), facecolor="white")
    axes[0].imshow(resized, cmap="gray")
    axes[0].set_title("(a) Ảnh X-quang đầu vào")
    axes[0].axis("off")

    axes[1].imshow(resized, cmap="gray")
    overlay = axes[1].imshow(
        heatmap,
        cmap="turbo",
        alpha=0.48,
        vmin=0.0,
        vmax=1.0,
    )
    axes[1].set_title("(b) Saliency của token ảnh cục bộ")
    axes[1].axis("off")
    colorbar = figure.colorbar(
        overlay,
        ax=axes[1],
        fraction=0.046,
        pad=0.035,
    )
    colorbar.set_label("Mức liên quan đối với lớp dự đoán")

    figure.suptitle(
        f"{record['image_id']} | GT={record['ground_truth']} | "
        f"Pred={record['prediction']} | Độ tin cậy={record['confidence']:.1%}",
        fontsize=11,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(figure)


def _cross_attention_distributions(
    model,
    cached: dict[str, torch.Tensor],
) -> dict[str, np.ndarray]:
    """Return head-wise and head-averaged bidirectional attention."""
    with torch.no_grad():
        image_tokens = model.backbone.visual_resampler(
            cached["global_feature"],
            cached["local_tokens"],
            cached["local_mask"],
            cached["local_boxes"],
        )
        image_padding = getattr(
            model.backbone.visual_resampler, "last_output_mask", None
        )
        if image_padding is None:
            image_key_padding = torch.zeros(
                image_tokens.size(0),
                image_tokens.size(1) - 1,
                dtype=torch.bool,
                device=image_tokens.device,
            )
        else:
            image_key_padding = ~image_padding.to(
                device=image_tokens.device,
                dtype=torch.bool,
            )
        text_key_padding = cached["text_attention_mask"][:, 1:] == 0
        _, attention = model.fusion(
            image_tokens,
            cached["text_tokens"],
            img_key_padding_mask=image_key_padding,
            txt_key_padding_mask=text_key_padding,
            return_attn=True,
        )
        visual = reduce_attention_to_keys(
            attention["attn_txt_to_img"], image_key_padding
        )
        text = reduce_attention_to_keys(
            attention["attn_img_to_txt"], text_key_padding
        )
        visual_heads = attention["attn_txt_to_img"][0].mean(dim=1)
        text_heads = attention["attn_img_to_txt"][0].mean(dim=1)
        visual_heads = visual_heads.masked_fill(image_key_padding[0], 0.0)
        text_heads = text_heads.masked_fill(text_key_padding[0], 0.0)
        visual_heads = visual_heads / visual_heads.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)
        text_heads = text_heads / text_heads.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)
    return {
        "visual_mean": visual[0].cpu().numpy(),
        "text_mean": text[0].cpu().numpy(),
        "visual_heads": visual_heads.cpu().numpy(),
        "text_heads": text_heads.cpu().numpy(),
        "visual_valid_mask": (~image_key_padding[0]).cpu().numpy(),
        "text_valid_mask": (~text_key_padding[0]).cpu().numpy(),
    }


def _normalized_entropy(values: np.ndarray) -> float:
    values = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
    positive = values > 0
    if int(positive.sum()) <= 1:
        return 0.0
    probability = values[positive] / values[positive].sum()
    return float(
        -(probability * np.log(np.maximum(probability, 1e-12))).sum()
        / np.log(len(probability))
    )


def _merge_display_tokens(
    tokens: list[str],
    scores: np.ndarray,
    valid_mask: np.ndarray,
    tokenizer,
) -> list[tuple[str, float]]:
    """Merge WordPiece/SentencePiece fragments for a readable attention plot."""
    merged: list[list[Any]] = []
    for token, score, valid in zip(tokens, scores, valid_mask):
        if not bool(valid) or not _displayable_token(token, tokenizer):
            continue
        is_continuation = token.startswith("##")
        cleaned = token.replace("##", "").replace("â–", "").replace("▁", "")
        cleaned = cleaned.replace("Ġ", "").strip()
        if not cleaned:
            continue
        if is_continuation and merged:
            merged[-1][0] = f"{merged[-1][0]}{cleaned}"
            merged[-1][1] += float(score)
        else:
            merged.append([cleaned, float(score)])
    return [(str(token), float(score)) for token, score in merged]


def _render_cross_attention_pipeline(
    image_path: Path,
    boxes: np.ndarray,
    visual_attention: np.ndarray,
    text_attention: np.ndarray,
    text_tokens: list[str],
    text_valid_mask: np.ndarray,
    record: dict[str, Any],
    output: Path,
    *,
    tokenizer,
    tokens_per_tile: int,
    include_local_cls_token: bool,
) -> None:
    """Render the two directional cross-attention distributions separately."""
    image = Image.open(image_path).convert("RGB")
    cls_indices, patch_indices = _tile_indices(
        len(visual_attention),
        tokens_per_tile,
        include_local_cls_token,
    )
    spatial_boxes = boxes[patch_indices] if patch_indices.size else boxes
    spatial_scores = (
        visual_attention[patch_indices]
        if patch_indices.size
        else visual_attention
    )
    heatmap = _rasterize(spatial_boxes, spatial_scores, image.size)
    resized = image.resize(
        (heatmap.shape[1], heatmap.shape[0]), Image.Resampling.LANCZOS
    )
    plot_height, plot_width = heatmap.shape

    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), facecolor="white")
    axes[0].imshow(resized, cmap="gray")
    axes[0].imshow(heatmap, cmap="Blues", alpha=0.62, vmin=0.0, vmax=1.0)
    if cls_indices.size:
        cls_values = visual_attention[cls_indices]
        cls_values = cls_values / max(float(cls_values.max()), 1e-8)
        _draw_normalized_boxes(
            axes[0],
            boxes,
            cls_indices,
            (plot_height, plot_width),
            edgecolor="#ff7f0e",
            linewidths=1.0 + 3.0 * cls_values,
        )
    axes[0].set_title(
        "(a) Văn bản truy vấn token ảnh\n"
        f"Attention entropy chuẩn hóa = "
        f"{_normalized_entropy(visual_attention):.3f}"
    )
    axes[0].axis("off")

    valid_items = _merge_display_tokens(
        text_tokens,
        text_attention,
        text_valid_mask,
        tokenizer,
    )
    valid_items.sort(key=lambda item: item[1], reverse=True)
    top_items = valid_items[:12]
    labels = [item[0] for item in top_items][::-1]
    scores = [item[1] for item in top_items][::-1]
    axes[1].barh(labels, scores, color="#ff7f0e", alpha=0.88)
    axes[1].set_xlabel("Trọng số cross-attention")
    axes[1].set_title(
        "(b) Ảnh truy vấn token bệnh sử\n"
        f"Attention entropy chuẩn hóa = "
        f"{_normalized_entropy(text_attention[text_valid_mask]):.3f}"
    )
    axes[1].grid(axis="x", alpha=0.25)

    figure.suptitle(
        f"Phân phối cross-attention của {record['image_id']}",
        fontsize=12,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _attention_head_entropies(
    attention_heads: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """Compute normalized entropy independently for every attention head."""
    valid = np.asarray(valid_mask, dtype=bool)
    return np.asarray(
        [
            _normalized_entropy(np.asarray(head, dtype=np.float64)[valid])
            for head in np.asarray(attention_heads)
        ],
        dtype=np.float64,
    )


def _draw_attention_text(
    axis,
    tokens: list[str],
    scores: np.ndarray,
    valid_mask: np.ndarray,
    tokenizer,
) -> None:
    """Draw clinical words in reading order with attention-colored boxes."""
    items = _merge_display_tokens(tokens, scores, valid_mask, tokenizer)
    maximum = max((score for _, score in items), default=1.0)
    color_map = plt.get_cmap("Oranges")
    x_position = 0.02
    y_position = 0.83
    for token, score in items:
        width = min(0.30, max(0.045, 0.014 * (len(token) + 1)))
        if x_position + width > 0.98:
            x_position = 0.02
            y_position -= 0.17
        if y_position < 0.05:
            break
        normalized = float(score) / max(float(maximum), 1e-8)
        axis.text(
            x_position,
            y_position,
            token,
            transform=axis.transAxes,
            fontsize=10,
            va="center",
            ha="left",
            bbox={
                "boxstyle": "round,pad=0.20",
                "facecolor": color_map(0.15 + 0.80 * normalized),
                "edgecolor": "none",
                "alpha": 0.95,
            },
        )
        x_position += width
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.axis("off")


def _plot_attention_intervention(
    axis,
    curves: dict[str, np.ndarray],
    *,
    title: str,
    xlabel: str,
) -> None:
    fractions = np.asarray(curves["fractions"], dtype=np.float64)
    deletion = np.asarray(curves["delete_most_relevant"], dtype=np.float64)
    random_deletion = np.asarray(curves["delete_random"], dtype=np.float64)
    insertion = np.asarray(curves["insert_most_relevant"], dtype=np.float64)
    axis.plot(
        fractions * 100.0,
        deletion,
        "o-",
        color="#1f77b4",
        linewidth=1.8,
        label="Deletion theo attention",
    )
    axis.plot(
        fractions * 100.0,
        random_deletion,
        "s--",
        color="#ff7f0e",
        linewidth=1.5,
        label="Deletion ngẫu nhiên",
    )
    axis.plot(
        fractions * 100.0,
        insertion,
        "^-",
        color="#2ca02c",
        linewidth=1.8,
        label="Insertion theo attention",
    )
    axis.set_xlim(0.0, 100.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Xác suất lớp mục tiêu")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7, loc="best")
    axis.set_title(title)
    axis.text(
        0.03,
        0.04,
        (
            f"Deletion AUC = {_curve_auc(fractions, deletion):.3f}\n"
            f"Insertion AUC = {_curve_auc(fractions, insertion):.3f}"
        ),
        transform=axis.transAxes,
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
    )


def _render_attention_evaluation(
    image_path: Path,
    boxes: np.ndarray,
    attention: dict[str, np.ndarray],
    text_tokens: list[str],
    text_valid_mask: np.ndarray,
    visual_curves: dict[str, np.ndarray],
    text_curves: dict[str, np.ndarray],
    record: dict[str, Any],
    output: Path,
    *,
    tokenizer,
    tokens_per_tile: int,
    include_local_cls_token: bool,
) -> None:
    """Render raw attention, per-head entropy, and attention-ranked fidelity."""
    image = Image.open(image_path).convert("RGB")
    visual_mean = np.asarray(attention["visual_mean"], dtype=np.float64)
    text_mean = np.asarray(attention["text_mean"], dtype=np.float64)
    cls_indices, patch_indices = _tile_indices(
        len(visual_mean),
        tokens_per_tile,
        include_local_cls_token,
    )
    heatmap = _rasterize(
        boxes[patch_indices],
        visual_mean[patch_indices],
        image.size,
    )
    blur_radius = max(4.0, 0.018 * min(heatmap.shape))
    smoothed = Image.fromarray(
        np.uint8(np.clip(heatmap, 0.0, 1.0) * 255.0),
        mode="L",
    ).filter(ImageFilter.GaussianBlur(radius=blur_radius))
    heatmap = np.asarray(smoothed, dtype=np.float32) / 255.0
    resized = image.resize(
        (heatmap.shape[1], heatmap.shape[0]),
        Image.Resampling.LANCZOS,
    )

    figure = plt.figure(figsize=(15.2, 9.0), facecolor="white")
    grid = figure.add_gridspec(
        2,
        3,
        width_ratios=(1.0, 1.15, 1.15),
        height_ratios=(1.0, 1.0),
        wspace=0.30,
        hspace=0.32,
    )
    image_axis = figure.add_subplot(grid[0, 0])
    text_axis = figure.add_subplot(grid[0, 1:])
    entropy_axis = figure.add_subplot(grid[1, 0])
    visual_curve_axis = figure.add_subplot(grid[1, 1])
    text_curve_axis = figure.add_subplot(grid[1, 2])

    image_axis.imshow(resized, cmap="gray")
    image_axis.imshow(
        heatmap,
        cmap="Blues",
        alpha=0.62,
        vmin=0.0,
        vmax=1.0,
    )
    cls_mass = float(visual_mean[cls_indices].sum()) if cls_indices.size else 0.0
    image_axis.set_title(
        "(a) Bệnh sử → token ảnh\n"
        f"Attention mass trên tile CLS = {cls_mass:.3f}"
    )
    image_axis.axis("off")

    _draw_attention_text(
        text_axis,
        text_tokens,
        text_mean,
        text_valid_mask,
        tokenizer,
    )
    displayable = np.asarray(
        [
            bool(valid) and _displayable_token(token, tokenizer)
            for token, valid in zip(text_tokens, text_valid_mask)
        ],
        dtype=bool,
    )
    special_mass = float(text_mean[text_valid_mask & ~displayable].sum())
    text_axis.set_title(
        "(b) Ảnh → token bệnh sử\n"
        f"Attention mass trên token đặc biệt/dấu câu = {special_mass:.3f}"
    )

    visual_entropy = _attention_head_entropies(
        attention["visual_heads"],
        attention["visual_valid_mask"],
    )
    text_entropy = _attention_head_entropies(
        attention["text_heads"],
        attention["text_valid_mask"],
    )
    head_ids = np.arange(1, len(visual_entropy) + 1)
    bar_width = 0.38
    entropy_axis.bar(
        head_ids - bar_width / 2,
        visual_entropy,
        width=bar_width,
        color="#1f77b4",
        label="Bệnh sử → ảnh",
    )
    entropy_axis.bar(
        head_ids + bar_width / 2,
        text_entropy,
        width=bar_width,
        color="#ff7f0e",
        label="Ảnh → bệnh sử",
    )
    entropy_axis.set_ylim(0.0, 1.05)
    entropy_axis.set_xticks(head_ids)
    entropy_axis.set_xlabel("Attention head")
    entropy_axis.set_ylabel("Entropy chuẩn hóa")
    entropy_axis.set_title(
        "(c) Mức phân tán theo từng head\n"
        f"Trung bình: {visual_entropy.mean():.3f} / {text_entropy.mean():.3f}"
    )
    entropy_axis.grid(axis="y", alpha=0.25)
    entropy_axis.legend(fontsize=7)

    _plot_attention_intervention(
        visual_curve_axis,
        visual_curves,
        title="(d) Faithfulness: token ảnh",
        xlabel="Tỷ lệ token ảnh được can thiệp (%)",
    )
    _plot_attention_intervention(
        text_curve_axis,
        text_curves,
        title="(e) Faithfulness: token bệnh sử",
        xlabel="Tỷ lệ token bệnh sử được can thiệp (%)",
    )

    figure.suptitle(
        f"Đánh giá cross-attention | {record['image_id']} | "
        f"GT={record['ground_truth']} | Pred={record['prediction']} | "
        f"Độ tin cậy={record['confidence']:.1%}",
        fontsize=13,
    )
    figure.subplots_adjust(top=0.90)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=230, bbox_inches="tight")
    plt.close(figure)


def _render_sample(
    image_path: Path,
    boxes: np.ndarray,
    visual_scores: np.ndarray,
    curves: dict[str, np.ndarray],
    text_curves: dict[str, np.ndarray],
    top_text: list[dict[str, Any]],
    record: dict[str, Any],
    output: Path,
) -> None:
    image = Image.open(image_path).convert("RGB")
    heatmap = _rasterize(boxes, visual_scores, image.size)
    resized = image.resize((heatmap.shape[1], heatmap.shape[0]), Image.Resampling.LANCZOS)
    figure, axes = plt.subplots(2, 2, figsize=(13, 10), facecolor="white")
    axes[0, 0].imshow(resized)
    for rank, index in enumerate(np.argsort(visual_scores)[::-1][:4], start=1):
        x1, y1, x2, y2 = boxes[index]
        axes[0, 0].add_patch(
            patches.Rectangle(
                (x1 * heatmap.shape[1], y1 * heatmap.shape[0]),
                (x2 - x1) * heatmap.shape[1],
                (y2 - y1) * heatmap.shape[0],
                fill=False,
                edgecolor="#ffcc00" if rank > 2 else "#ff3b30",
                linewidth=2.0,
            )
        )
    axes[0, 0].set_title("Top class-specific local regions")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(resized)
    axes[0, 1].imshow(heatmap, cmap="inferno", alpha=0.58)
    axes[0, 1].set_title("Integrated-gradients heatmap")
    axes[0, 1].axis("off")

    fractions = curves["fractions"] * 100
    axes[1, 0].plot(
        fractions,
        curves["delete_most_relevant"],
        "o-",
        label="delete most relevant",
        color="#b3261e",
    )
    axes[1, 0].plot(
        fractions,
        curves["delete_random"],
        "o-",
        label="delete random",
        color="#6b7280",
    )
    axes[1, 0].plot(
        fractions,
        curves["insert_most_relevant"],
        "o-",
        label="insert most relevant",
        color="#18732b",
    )
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].set_xlabel("Local tokens intervened (%)")
    axes[1, 0].set_ylabel("Target probability")
    axes[1, 0].grid(alpha=0.25)
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].set_title("Visual-token faithfulness")

    token_text = ", ".join(item["token"] for item in top_text[:5])
    text_fractions = text_curves["fractions"] * 100
    axes[1, 1].plot(
        text_fractions,
        text_curves["delete_most_relevant"],
        "o-",
        label="delete most relevant",
        color="#b3261e",
    )
    axes[1, 1].plot(
        text_fractions,
        text_curves["delete_random"],
        "o-",
        label="delete random",
        color="#6b7280",
    )
    axes[1, 1].plot(
        text_fractions,
        text_curves["insert_most_relevant"],
        "o-",
        label="insert most relevant",
        color="#18732b",
    )
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_xlabel("Clinical tokens intervened (%)")
    axes[1, 1].set_ylabel("Target probability")
    axes[1, 1].grid(alpha=0.25)
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].set_title(f"Clinical-token faithfulness\nTop tokens: {token_text}")
    figure.suptitle(
        f"{record['image_id']} | GT={record['ground_truth']} "
        f"Pred={record['prediction']} ({record['confidence']:.1%})"
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _sample_explanation(
    loaded,
    dataset: CTCHDataset,
    dataset_index: int,
    image_id: str,
    ig_steps: int,
    random_trials: int,
    stability_repeats: int,
    stability_steps: int,
    noise_scale: float,
    seed: int,
    run_sanity_check: bool,
    render_path: Path | None,
) -> dict[str, Any]:
    sample = dataset[dataset_index]
    batch = _one_sample(sample, loaded)
    cached = _cached_tokens(loaded.model, batch)
    with torch.no_grad():
        full_logits, _, _ = forward_from_local(loaded.model, cached)
        probabilities = torch.softmax(full_logits, dim=-1)[0]
    ground_truth = int(batch["labels"].item())
    prediction = int(probabilities.argmax())
    target_class = prediction

    branch = branch_ablation(loaded.model, cached, target_class)
    global_ig_scores, global_ig_attribution = integrated_gradients_global_image(
        loaded.model,
        cached,
        batch["pixel_values"],
        target_class,
        steps=ig_steps,
    )
    ig_scores, ig_attribution = integrated_gradients(
        loaded.model, cached, target_class, steps=ig_steps
    )
    text_ig_scores, text_ig_attribution = integrated_gradients_text(
        loaded.model,
        cached,
        target_class,
        steps=ig_steps,
    )
    gradient_scores = gradient_visual_relevance(
        loaded.model, cached, target_class
    )
    text_scores = gradient_text_relevance(loaded.model, cached, target_class)
    curves = perturbation_curves(
        loaded.model,
        cached,
        ig_scores,
        target_class,
        random_trials=random_trials,
        seed=seed,
    )
    global_curves = global_image_perturbation_curves(
        loaded.model,
        cached,
        batch["pixel_values"],
        global_ig_scores,
        target_class,
    )
    text_input_curves = text_input_perturbation_curves(
        loaded.model,
        cached,
        text_ig_scores,
        target_class,
    )
    text_curves = text_perturbation_curves(
        loaded.model,
        cached,
        text_scores,
        target_class,
        random_trials=random_trials,
        seed=seed + 1000,
    )

    baseline_tokens = cached["global_feature"].unsqueeze(1).expand_as(
        cached["local_tokens"]
    )
    with torch.no_grad():
        baseline_logits, _, _ = forward_from_local(
            loaded.model, cached, local_tokens=baseline_tokens
        )
    logit_difference = float(
        branch["logits"][0, target_class] - baseline_logits[0, target_class]
    )
    signed_sum = float(ig_attribution.sum())
    full_target_logit = float(branch["logits"][0, target_class])
    full_target_probability = float(probabilities[target_class])
    global_signed_sum = float(global_ig_attribution.sum())
    global_logit_difference = (
        full_target_logit
        - float(global_curves["delete_most_relevant_logit"][-1])
    )
    text_signed_sum = float(text_ig_attribution.sum())
    text_logit_difference = (
        full_target_logit
        - float(text_input_curves["delete_most_relevant_logit"][-1])
    )

    stability = []
    generator = torch.Generator(device=loaded.device).manual_seed(seed + 17)
    for _ in range(stability_repeats):
        noise = torch.randn(
            cached["local_tokens"].shape,
            generator=generator,
            device=loaded.device,
            dtype=cached["local_tokens"].dtype,
        )
        perturbed = {
            **cached,
            "local_tokens": cached["local_tokens"] + noise_scale * noise,
        }
        perturbed_scores, _ = integrated_gradients(
            loaded.model,
            perturbed,
            target_class,
            steps=stability_steps,
        )
        stability.append(
            rank_correlation(
                ig_scores.cpu().numpy(), perturbed_scores.cpu().numpy()
            )
        )

    sanity_correlation = None
    if run_sanity_check:
        original_centroids = loaded.model.head.centroids.detach().clone()
        try:
            with torch.no_grad():
                loaded.model.head.centroids.copy_(
                    torch.roll(original_centroids, 1, dims=0)
                )
            randomized_scores, _ = integrated_gradients(
                loaded.model,
                cached,
                target_class,
                steps=stability_steps,
            )
            sanity_correlation = rank_correlation(
                ig_scores.cpu().numpy(), randomized_scores.cpu().numpy()
            )
        finally:
            with torch.no_grad():
                loaded.model.head.centroids.copy_(original_centroids)

    token_ids = batch["clinical_input_ids"][0]
    tokens = _decode_tokens(loaded.model.backbone.tokenizer_obj, token_ids)
    valid = batch["clinical_attention_mask"][0].bool().cpu().numpy()
    text_values = text_scores.cpu().numpy()
    ranked_text = np.argsort(text_values)[::-1]
    top_text = [
        {"token": tokens[index], "score": float(text_values[index])}
        for index in ranked_text
        if valid[index]
        and _displayable_token(tokens[index], loaded.model.backbone.tokenizer_obj)
    ][:10]

    deletion_auc = curve_auc(
        curves["fractions"], curves["delete_most_relevant"]
    )
    random_auc = curve_auc(curves["fractions"], curves["delete_random"])
    low_auc = curve_auc(
        curves["fractions"], curves["delete_least_relevant"]
    )
    insertion_auc = curve_auc(
        curves["fractions"], curves["insert_most_relevant"]
    )
    targeted_logit_auc = curve_auc(
        curves["fractions"], curves["delete_most_relevant_logit"]
    )
    random_logit_auc = curve_auc(
        curves["fractions"], curves["delete_random_logit"]
    )
    low_logit_auc = curve_auc(
        curves["fractions"], curves["delete_least_relevant_logit"]
    )
    targeted_margin_auc = curve_auc(
        curves["fractions"], curves["delete_most_relevant_margin"]
    )
    random_margin_auc = curve_auc(
        curves["fractions"], curves["delete_random_margin"]
    )
    text_deletion_auc = curve_auc(
        text_curves["fractions"], text_curves["delete_most_relevant"]
    )
    text_random_auc = curve_auc(
        text_curves["fractions"], text_curves["delete_random"]
    )
    text_logit_auc = curve_auc(
        text_curves["fractions"],
        text_curves["delete_most_relevant_logit"],
    )
    text_random_logit_auc = curve_auc(
        text_curves["fractions"], text_curves["delete_random_logit"]
    )
    text_margin_auc = curve_auc(
        text_curves["fractions"],
        text_curves["delete_most_relevant_margin"],
    )
    text_random_margin_auc = curve_auc(
        text_curves["fractions"], text_curves["delete_random_margin"]
    )
    global_deletion_auc = curve_auc(
        global_curves["fractions"],
        global_curves["delete_most_relevant"],
    )
    global_insertion_auc = curve_auc(
        global_curves["fractions"],
        global_curves["insert_most_relevant"],
    )
    text_input_deletion_auc = curve_auc(
        text_input_curves["fractions"],
        text_input_curves["delete_most_relevant"],
    )
    text_input_insertion_auc = curve_auc(
        text_input_curves["fractions"],
        text_input_curves["insert_most_relevant"],
    )
    record = {
        "image_id": image_id,
        "ground_truth": ground_truth,
        "prediction": prediction,
        "correct": bool(ground_truth == prediction),
        "confidence": float(probabilities[prediction]),
        "target_class": target_class,
        "num_local_tokens": int(cached["local_tokens"].shape[1]),
        "branch_target_logit_drop": branch["target_logit_drop"],
        "branch_target_probability_drop": branch["target_probability_drop"],
        "source_target_probability_drop": {
            "global_image": (
                full_target_probability
                - float(global_curves["delete_most_relevant"][-1])
            ),
            "local_image": (
                full_target_probability
                - float(curves["delete_most_relevant"][-1])
            ),
            "clinical_text": (
                full_target_probability
                - float(text_input_curves["delete_most_relevant"][-1])
            ),
        },
        "global_image_integrated_gradients": {
            "signed_attribution_sum": global_signed_sum,
            "logit_difference_vs_zero_input": global_logit_difference,
            "completeness_error": abs(
                global_signed_sum - global_logit_difference
            ),
            "relative_completeness_error": abs(
                global_signed_sum - global_logit_difference
            ) / max(abs(global_logit_difference), 1e-8),
            "entropy": _entropy(global_ig_scores.cpu().numpy()),
        },
        "integrated_gradients": {
            "signed_attribution_sum": signed_sum,
            "local_logit_difference_vs_global_baseline": logit_difference,
            "completeness_error": abs(signed_sum - logit_difference),
            "relative_completeness_error": abs(signed_sum - logit_difference)
            / max(abs(logit_difference), 1e-8),
            "entropy": _entropy(ig_scores.cpu().numpy()),
        },
        "clinical_text_integrated_gradients": {
            "signed_attribution_sum": text_signed_sum,
            "logit_difference_vs_padding_baseline": text_logit_difference,
            "completeness_error": abs(
                text_signed_sum - text_logit_difference
            ),
            "relative_completeness_error": abs(
                text_signed_sum - text_logit_difference
            ) / max(abs(text_logit_difference), 1e-8),
            "entropy": _entropy(text_ig_scores.cpu().numpy()),
        },
        "method_agreement_spearman": rank_correlation(
            ig_scores.cpu().numpy(), gradient_scores.cpu().numpy()
        ),
        "stability_spearman_mean": float(np.mean(stability)) if stability else None,
        "classifier_randomization_spearman": sanity_correlation,
        "text_attribution_entropy": _entropy(text_values[valid]),
        "top_clinical_tokens": top_text,
        "faithfulness": {
            "deletion_auc": deletion_auc,
            "random_deletion_auc": random_auc,
            "least_relevant_deletion_auc": low_auc,
            "insertion_auc": insertion_auc,
            "random_minus_targeted_deletion_auc": random_auc - deletion_auc,
            "least_minus_targeted_deletion_auc": low_auc - deletion_auc,
            "target_logit_deletion_auc": targeted_logit_auc,
            "random_target_logit_deletion_auc": random_logit_auc,
            "random_minus_targeted_logit_auc": (
                random_logit_auc - targeted_logit_auc
            ),
            "least_minus_targeted_logit_auc": low_logit_auc - targeted_logit_auc,
            "target_margin_deletion_auc": targeted_margin_auc,
            "random_target_margin_deletion_auc": random_margin_auc,
            "random_minus_targeted_margin_auc": (
                random_margin_auc - targeted_margin_auc
            ),
        },
        "global_image_faithfulness": {
            "deletion_auc": global_deletion_auc,
            "insertion_auc": global_insertion_auc,
        },
        "clinical_text_input_faithfulness": {
            "deletion_auc": text_input_deletion_auc,
            "insertion_auc": text_input_insertion_auc,
        },
        "text_faithfulness": {
            "deletion_auc": text_deletion_auc,
            "random_deletion_auc": text_random_auc,
            "random_minus_targeted_deletion_auc": (
                text_random_auc - text_deletion_auc
            ),
            "target_logit_deletion_auc": text_logit_auc,
            "random_target_logit_deletion_auc": text_random_logit_auc,
            "random_minus_targeted_logit_auc": (
                text_random_logit_auc - text_logit_auc
            ),
            "target_margin_deletion_auc": text_margin_auc,
            "random_target_margin_deletion_auc": text_random_margin_auc,
            "random_minus_targeted_margin_auc": (
                text_random_margin_auc - text_margin_auc
            ),
        },
        "curves": {key: value.tolist() for key, value in curves.items()},
        "global_image_curves": {
            key: value.tolist() for key, value in global_curves.items()
        },
        "clinical_text_input_curves": {
            key: value.tolist()
            for key, value in text_input_curves.items()
        },
        "text_curves": {
            key: value.tolist() for key, value in text_curves.items()
        },
    }
    if render_path is not None:
        boxes = cached["local_boxes"][0].cpu().numpy()
        tokens_per_tile = int(loaded.model.backbone.local_pool_grid) ** 2 + int(
            loaded.model.backbone.include_local_cls_token
        )
        include_local_cls_token = bool(
            loaded.model.backbone.include_local_cls_token
        )
        _render_sample(
            Path(dataset.img_dir) / image_id,
            boxes,
            ig_scores.cpu().numpy(),
            curves,
            text_curves,
            top_text,
            record,
            render_path,
        )
        record["visualization"] = str(render_path)
        pipeline_path = render_path.with_name(
            f"{render_path.stem}_token_pipeline.png"
        )
        _render_token_faithfulness_pipeline(
            Path(dataset.img_dir) / image_id,
            boxes,
            ig_scores.cpu().numpy(),
            curves,
            record,
            pipeline_path,
            tokens_per_tile=tokens_per_tile,
            include_local_cls_token=include_local_cls_token,
        )
        record["token_pipeline_visualization"] = str(pipeline_path)
        saliency_path = render_path.with_name(
            f"{render_path.stem}_spatial_saliency.png"
        )
        _render_spatial_saliency_map(
            Path(dataset.img_dir) / image_id,
            boxes,
            ig_scores.cpu().numpy(),
            record,
            saliency_path,
            tokens_per_tile=tokens_per_tile,
            include_local_cls_token=include_local_cls_token,
        )
        record["spatial_saliency_visualization"] = str(saliency_path)

        attention_details = _cross_attention_distributions(
            loaded.model,
            cached,
        )
        visual_attention = attention_details["visual_mean"]
        text_attention = attention_details["text_mean"]
        if len(visual_attention) != len(boxes):
            raise ValueError(
                "Cross-attention output tokens cannot be mapped one-to-one "
                f"to local boxes: {len(visual_attention)} vs {len(boxes)}."
            )
        cross_attention_path = render_path.with_name(
            f"{render_path.stem}_cross_attention.png"
        )
        text_tokens_without_cls = _decode_tokens(
            loaded.model.backbone.tokenizer_obj,
            batch["clinical_input_ids"][0, 1:],
        )
        _render_cross_attention_pipeline(
            Path(dataset.img_dir) / image_id,
            boxes,
            visual_attention,
            text_attention,
            text_tokens_without_cls,
            batch["clinical_attention_mask"][0, 1:].bool().cpu().numpy(),
            record,
            cross_attention_path,
            tokenizer=loaded.model.backbone.tokenizer_obj,
            tokens_per_tile=tokens_per_tile,
            include_local_cls_token=include_local_cls_token,
        )
        record["cross_attention_visualization"] = str(cross_attention_path)

        visual_attention_tensor = torch.as_tensor(
            visual_attention,
            device=cached["local_tokens"].device,
            dtype=cached["local_tokens"].dtype,
        )
        attention_visual_curves = perturbation_curves(
            loaded.model,
            cached,
            visual_attention_tensor,
            target_class,
            random_trials=random_trials,
            seed=seed + 2000,
        )
        text_attention_tensor = torch.zeros_like(text_scores)
        text_attention_tensor[1:] = torch.as_tensor(
            text_attention,
            device=text_attention_tensor.device,
            dtype=text_attention_tensor.dtype,
        )
        attention_text_curves = text_perturbation_curves(
            loaded.model,
            cached,
            text_attention_tensor,
            target_class,
            random_trials=random_trials,
            seed=seed + 3000,
        )
        visual_head_entropy = _attention_head_entropies(
            attention_details["visual_heads"],
            attention_details["visual_valid_mask"],
        )
        text_head_entropy = _attention_head_entropies(
            attention_details["text_heads"],
            attention_details["text_valid_mask"],
        )
        record["cross_attention_evaluation"] = {
            "visual_head_entropy_mean": float(visual_head_entropy.mean()),
            "visual_head_entropy_std": float(visual_head_entropy.std(ddof=0)),
            "text_head_entropy_mean": float(text_head_entropy.mean()),
            "text_head_entropy_std": float(text_head_entropy.std(ddof=0)),
            "visual_attention_deletion_auc": curve_auc(
                attention_visual_curves["fractions"],
                attention_visual_curves["delete_most_relevant"],
            ),
            "visual_attention_random_deletion_auc": curve_auc(
                attention_visual_curves["fractions"],
                attention_visual_curves["delete_random"],
            ),
            "visual_attention_insertion_auc": curve_auc(
                attention_visual_curves["fractions"],
                attention_visual_curves["insert_most_relevant"],
            ),
            "text_attention_deletion_auc": curve_auc(
                attention_text_curves["fractions"],
                attention_text_curves["delete_most_relevant"],
            ),
            "text_attention_random_deletion_auc": curve_auc(
                attention_text_curves["fractions"],
                attention_text_curves["delete_random"],
            ),
            "text_attention_insertion_auc": curve_auc(
                attention_text_curves["fractions"],
                attention_text_curves["insert_most_relevant"],
            ),
        }
        attention_evaluation_path = render_path.with_name(
            f"{render_path.stem}_attention_evaluation.png"
        )
        _render_attention_evaluation(
            Path(dataset.img_dir) / image_id,
            boxes,
            attention_details,
            text_tokens_without_cls,
            batch["clinical_attention_mask"][0, 1:].bool().cpu().numpy(),
            attention_visual_curves,
            attention_text_curves,
            record,
            attention_evaluation_path,
            tokenizer=loaded.model.backbone.tokenizer_obj,
            tokens_per_tile=tokens_per_tile,
            include_local_cls_token=include_local_cls_token,
        )
        record["cross_attention_evaluation_visualization"] = str(
            attention_evaluation_path
        )
    return record


def _flatten_numeric(value: Any, prefix: str = "") -> dict[str, float]:
    result: dict[str, float] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else key
            result.update(_flatten_numeric(child, child_prefix))
    elif isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        number = float(value)
        if np.isfinite(number):
            result[prefix] = number
    return result


def _aggregate_records(records: list[dict[str, Any]], bootstrap_seed: int) -> dict[str, Any]:
    excluded = {"ground_truth", "prediction", "target_class"}
    keys = sorted(
        set().union(*(_flatten_numeric(record).keys() for record in records))
    )
    rng = np.random.default_rng(bootstrap_seed)
    output: dict[str, Any] = {}
    for key in keys:
        if key in excluded:
            continue
        values = np.asarray(
            [
                flattened[key]
                for record in records
                if key in (flattened := _flatten_numeric(record))
            ],
            dtype=np.float64,
        )
        if values.size == 0:
            continue
        bootstrap_means = np.asarray(
            [
                rng.choice(values, size=len(values), replace=True).mean()
                for _ in range(2000)
            ]
        )
        output[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "median": float(np.median(values)),
            "ci_95": [
                float(np.percentile(bootstrap_means, 2.5)),
                float(np.percentile(bootstrap_means, 97.5)),
            ],
            "n": int(len(values)),
        }
    return output


def _aggregate_subgroups(
    records: list[dict[str, Any]],
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Report whether attribution behavior changes with correctness/confidence."""
    if not records:
        return {}
    confidences = np.asarray(
        [float(record["confidence"]) for record in records],
        dtype=np.float64,
    )
    q25, q75 = np.quantile(confidences, [0.25, 0.75])
    groups = OrderedDict([
        ("correct", [record for record in records if record["correct"]]),
        ("incorrect", [record for record in records if not record["correct"]]),
        (
            "low_confidence_q1",
            [
                record
                for record in records
                if float(record["confidence"]) <= float(q25)
            ],
        ),
        (
            "high_confidence_q4",
            [
                record
                for record in records
                if float(record["confidence"]) >= float(q75)
            ],
        ),
    ])
    return {
        name: {
            "sample_count": len(group_records),
            "aggregate": _aggregate_records(
                group_records,
                bootstrap_seed=bootstrap_seed + index,
            ),
        }
        for index, (name, group_records) in enumerate(groups.items())
        if group_records
    }


def _source_role_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    sources = ("global_image", "local_image", "clinical_text")
    values = {
        source: np.asarray(
            [
                float(record["source_target_probability_drop"][source])
                for record in records
            ],
            dtype=np.float64,
        )
        for source in sources
    }
    dominant = {
        source: 0
        for source in sources
    }
    for index in range(len(records)):
        source = max(sources, key=lambda name: values[name][index])
        dominant[source] += 1
    return {
        "definition": (
            "Dominant source is the intervention whose removal causes the "
            "largest decrease in predicted-class probability."
        ),
        "dominant_source_counts": dominant,
        "per_source": {
            source: {
                "mean_probability_drop": float(source_values.mean()),
                "median_probability_drop": float(np.median(source_values)),
                "positive_drop_fraction": float((source_values > 0).mean()),
                "n": int(len(source_values)),
            }
            for source, source_values in values.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--train-features", type=Path, required=True)
    parser.add_argument("--test-features", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=44)
    parser.add_argument(
        "--all-test-samples",
        action="store_true",
        help=(
            "Run quantitative IG and interventions on every CTCH test sample. "
            "--per-class and --max-samples are ignored."
        ),
    )
    parser.add_argument("--selection-seed", type=int, default=1907)
    parser.add_argument("--ig-steps", type=int, default=24)
    parser.add_argument("--random-trials", type=int, default=16)
    parser.add_argument("--stability-repeats", type=int, default=2)
    parser.add_argument("--stability-steps", type=int, default=8)
    parser.add_argument("--noise-scale", type=float, default=0.01)
    parser.add_argument("--sanity-samples", type=int, default=5)
    parser.add_argument("--render-samples", type=int, default=6)
    parser.add_argument(
        "--visualization-only",
        action="store_true",
        help=(
            "Skip representation-space metrics and render only sample-level "
            "attention/attribution diagnostics. This supports compact feature "
            "archives that contain labels and predictions but omit optional "
            "intermediate embeddings."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument("--no-strict-fingerprint", action="store_true")
    args = parser.parse_args()

    summary_path = args.output_dir / "summary.json"
    if summary_path.is_file() and not args.overwrite:
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        train_sidecar = args.train_features.with_suffix(".json")
        test_sidecar = args.test_features.with_suffix(".json")
        feature_shas = set()
        for sidecar in (train_sidecar, test_sidecar):
            if sidecar.is_file():
                feature_shas.add(
                    json.loads(sidecar.read_text(encoding="utf-8")).get(
                        "checkpoint_sha256"
                    )
                )
        protocol = existing.get("protocol", {})
        sampling_matches = (
            bool(protocol.get("all_test_samples", False))
            == bool(args.all_test_samples)
        )
        if sampling_matches and not args.all_test_samples:
            sampling_matches = (
                int(protocol.get("per_class", -1)) == args.per_class
                and int(protocol.get("max_samples", -1)) == args.max_samples
            )
        if (
            existing.get("source_experiment") == SOURCE_EXPERIMENT
            and int(existing.get("seed", -1)) == args.seed
            and feature_shas == {existing.get("checkpoint_sha256")}
            and sampling_matches
            and int(protocol.get("selection_seed", -1)) == args.selection_seed
            and int(protocol.get("ig_steps", -1)) == args.ig_steps
            and int(protocol.get("random_trials", -1)) == args.random_trials
            and int(protocol.get("stability_repeats", -1))
            == args.stability_repeats
            and int(protocol.get("stability_steps", -1)) == args.stability_steps
            and float(protocol.get("stability_noise_scale", -1.0))
            == args.noise_scale
            and int(protocol.get("classifier_randomization_samples", -1))
            == min(args.sanity_samples, int(existing.get("sample_count", 0)))
            and int(protocol.get("render_samples", -1))
            == min(args.render_samples, int(existing.get("sample_count", 0)))
            and bool(protocol.get("visualization_only", False))
            == bool(args.visualization_only)
        ):
            print(f"[Resume] Verified explainability summary exists: {summary_path}")
            return

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable.")
        device = torch.device(args.device)
    loaded = load_locked_proposed_model(
        args.seed,
        device=device,
        strict_fingerprint=not args.no_strict_fingerprint,
    )
    train, train_provenance = load_feature_archive(args.train_features)
    test, test_provenance = load_feature_archive(args.test_features)
    for provenance, scenario in (
        (train_provenance, "ctch_train"),
        (test_provenance, "ctch_test"),
    ):
        if int(provenance.get("seed", -1)) != args.seed:
            raise ValueError("Feature seed differs from explainability seed.")
        if provenance.get("scenario") != scenario:
            raise ValueError(f"Expected {scenario} features.")
        if provenance.get("checkpoint_sha256") != loaded.checkpoint_sha256:
            raise ValueError("Feature archive and loaded checkpoint checksums differ.")
        if provenance.get("config_sha256") != loaded.provenance.get("config_sha256"):
            raise ValueError("Feature archive and loaded resolved configs differ.")

    representation: dict[str, dict[str, float]] = {}
    representation_plot: Path | None = None
    if not args.visualization_only:
        missing_representations = [
            key
            for key in REPRESENTATIONS
            if key not in train or key not in test
        ]
        if missing_representations:
            raise KeyError(
                "Feature archives omit representation spaces required by the "
                f"full protocol: {missing_representations}. Re-extract full "
                "features or pass --visualization-only."
            )
        representation = {
            key: representation_quality_metrics(
                train[key],
                train["labels"],
                test[key],
                test["labels"],
                seed=args.selection_seed,
            )
            for key in REPRESENTATIONS
        }
        representation_plot = args.output_dir / "representation_overview.png"
        _render_representation_overview(
            train,
            test,
            representation,
            representation_plot,
            seed=args.selection_seed,
        )

    dataset = _ctch_test_dataset(loaded)
    index_by_id = {
        str(row["image_id"]): int(index)
        for index, row in dataset.df.iterrows()
    }
    if args.all_test_samples:
        selected = np.arange(len(test["labels"]), dtype=np.int64)
    else:
        selected = stratified_sample_indices(
            test["labels"],
            per_class=args.per_class,
            seed=args.selection_seed,
            max_samples=args.max_samples,
        )
    selected_ids = test["image_id"].astype(str)[selected]
    missing = [image_id for image_id in selected_ids if image_id not in index_by_id]
    if missing:
        raise ValueError(f"Selected feature IDs are absent from CTCH test: {missing[:5]}")

    for parameter in loaded.model.parameters():
        parameter.requires_grad_(False)
    loaded.model.eval()
    loaded.model.backbone.explain_mode = True
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for position, image_id in enumerate(selected_ids):
        print(f"[Explain {position + 1}/{len(selected_ids)}] {image_id}")
        render_path = (
            args.output_dir / "visualizations" / f"{Path(image_id).stem}.png"
            if position < args.render_samples
            else None
        )
        record = _sample_explanation(
            loaded,
            dataset,
            index_by_id[image_id],
            image_id,
            ig_steps=args.ig_steps,
            random_trials=args.random_trials,
            stability_repeats=args.stability_repeats,
            stability_steps=args.stability_steps,
            noise_scale=args.noise_scale,
            seed=args.selection_seed + position,
            run_sanity_check=position < args.sanity_samples,
            render_path=render_path,
        )
        records.append(record)
        (args.output_dir / "samples.json").write_text(
            json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    summary = {
        "type": "locked_ctch_explainability_evaluation",
        "source_experiment": SOURCE_EXPERIMENT,
        "seed": args.seed,
        "checkpoint_sha256": loaded.checkpoint_sha256,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "protocol": {
            "target": "predicted_class",
            "sampling": (
                "all_test_samples"
                if args.all_test_samples
                else "stratified_by_ground_truth_class"
            ),
            "all_test_samples": bool(args.all_test_samples),
            "per_class": None if args.all_test_samples else args.per_class,
            "max_samples": None if args.all_test_samples else args.max_samples,
            "selection_seed": args.selection_seed,
            "ig_steps": args.ig_steps,
            "ig_baselines": {
                "global_image": "zero_in_normalized_input_space",
                "local_image": (
                    "global_image_embedding_repeated_as_local_tokens"
                ),
                "clinical_text": (
                    "padding_embedding_with_special_tokens_preserved"
                ),
            },
            "random_trials": args.random_trials,
            "faithfulness_modalities": [
                "global_image",
                "visual_local_tokens",
                "clinical_tokens",
            ],
            "stability_repeats": args.stability_repeats,
            "stability_steps": args.stability_steps,
            "stability_noise_scale": args.noise_scale,
            "classifier_randomization_samples": min(
                args.sanity_samples, len(records)
            ),
            "render_samples": min(args.render_samples, len(records)),
            "visualization_only": bool(args.visualization_only),
            "spatial_localization_metrics": (
                "not_reported: CTCH has no lesion-region annotations in the "
                "analysis workspace"
            ),
        },
        "sample_count": len(records),
        "correct_count": int(sum(record["correct"] for record in records)),
        "representation_quality": representation,
        "representation_visualization": (
            str(representation_plot) if representation_plot is not None else None
        ),
        "explanation_aggregate": _aggregate_records(
            records, bootstrap_seed=args.selection_seed + 5000
        ),
        "explanation_subgroups": _aggregate_subgroups(
            records,
            bootstrap_seed=args.selection_seed + 6000,
        ),
        "source_role_summary": _source_role_summary(records),
        "sample_records": str(args.output_dir / "samples.json"),
        "visualization_dir": str(args.output_dir / "visualizations"),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
