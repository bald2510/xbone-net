"""Visualize class-specific bidirectional cross-attention on image and text.

The image panel projects text-to-visual attention through the spatial resampler
to source-image coordinates. The report panel highlights clinical WordPiece
tokens using image-to-text attention weighted by the target-class gradient.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
from matplotlib import colors
from omegaconf import OmegaConf
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate import adapt_state_dict_keys, load_state_dict_checked
from src.datasets.builder import build_dataloader
from src.models.builder import build_model, setup_phase2_modules
from src.models.fusion.cross_attention import reduce_attention_to_keys
from src.utils.trainer import BioMedCLIPDataCollator, resolve_pad_token_id


ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path):
    OmegaConf.register_new_resolver(
        "hydra",
        lambda key: str(ROOT) if key == "runtime.cwd" else "",
        replace=True,
    )
    cfg = OmegaConf.load(path)
    OmegaConf.set_struct(cfg, False)
    return cfg


def load_model(cfg, checkpoint: Path, device: torch.device):
    model = build_model(cfg.model).to(device)
    model, _, fusion_type, _ = setup_phase2_modules(model, cfg, device)
    if fusion_type != "cross_attention":
        raise ValueError(f"Expected cross_attention fusion, got {fusion_type!r}.")
    saved = torch.load(checkpoint, map_location=device)
    state_dict = adapt_state_dict_keys(
        saved.get("model_state_dict", saved), list(model.state_dict().keys())
    )
    load_state_dict_checked(model, state_dict, context="cross-modal visualization")
    model.eval()
    return model


def one_sample_batch(sample: dict, tokenizer, device: torch.device) -> dict:
    collator = BioMedCLIPDataCollator(resolve_pad_token_id(tokenizer))
    batch = collator([sample])
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def find_annotation(image_id: str) -> Path | None:
    path = ROOT / "data" / "BTXRD" / "Annotations" / f"{Path(image_id).stem}.json"
    return path if path.exists() else None


def load_annotation(path: Path | None):
    if path is None or not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle).get("shapes", [])


def infer_dataset_index(dataset, image_id: str) -> int:
    indices = dataset.df.index[
        dataset.df["image_id"].astype(str) == str(image_id)
    ].tolist()
    if not indices:
        raise ValueError(f"{image_id!r} is not present in the BTXRD test split.")
    return int(indices[0])


def draw_ground_truth(ax, shapes, sx: float = 1.0, sy: float = 1.0):
    for shape in shapes:
        points = np.asarray(shape.get("points", []), dtype=np.float32)
        if points.size == 0:
            continue
        points[:, 0] *= sx
        points[:, 1] *= sy
        if shape.get("shape_type") == "rectangle" and len(points) >= 2:
            x1, y1 = points[0]
            x2, y2 = points[1]
            patch = mpatches.Rectangle(
                (min(x1, x2), min(y1, y2)),
                abs(x2 - x1),
                abs(y2 - y1),
                fill=False,
                edgecolor="#39ff14",
                linewidth=2.1,
                linestyle="--",
            )
        else:
            patch = mpatches.Polygon(
                points,
                closed=True,
                fill=False,
                edgecolor="#39ff14",
                linewidth=2.1,
                linestyle="--",
            )
        ax.add_patch(patch)


def annotation_mask(shapes, image_size: tuple[int, int], heat_shape: tuple[int, int]):
    width, height = image_size
    heat_h, heat_w = heat_shape
    mask = Image.new("L", (heat_w, heat_h), 0)
    drawer = ImageDraw.Draw(mask)
    sx, sy = heat_w / width, heat_h / height
    for shape in shapes:
        scaled = [
            (round(x * sx), round(y * sy))
            for x, y in shape.get("points", [])
        ]
        if shape.get("shape_type") == "rectangle" and len(scaled) >= 2:
            drawer.rectangle([scaled[0], scaled[1]], fill=1)
        elif len(scaled) >= 3:
            drawer.polygon(scaled, fill=1)
    return np.asarray(mask, dtype=bool)


def _normalize_scores(values: torch.Tensor, mask: torch.Tensor | None = None):
    values = values.float()
    if mask is not None:
        values = values * mask.to(values.dtype)
    values = torch.relu(values)
    maximum = values.amax(dim=-1, keepdim=True)
    return values / maximum.clamp_min(1e-8)


def _entropy(values: torch.Tensor):
    distribution = values.detach() / values.detach().sum().clamp_min(1e-8)
    return float(-(distribution * torch.log(distribution + 1e-8)).sum())


def cross_modal_attribution(model, batch: dict, target_class: int):
    """Return class-weighted image and clinical-token cross-attention scores."""
    model.backbone.explain_mode = True
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    with torch.no_grad():
        _, encoded_text = model.backbone(
            batch["pixel_values"],
            batch["clinical_input_ids"],
            attention_mask=batch["clinical_attention_mask"],
            tile_values=batch.get("tile_values"),
            tile_mask=batch.get("tile_mask"),
            tile_boxes=batch.get("tile_boxes"),
        )

    global_feature = model.backbone.last_global_feature.detach()
    local_tokens = model.backbone.last_local_tokens.detach().requires_grad_(True)
    local_mask = model.backbone.last_local_mask.detach()
    local_boxes = model.backbone.last_local_token_boxes.detach()
    text_tokens = encoded_text.detach().requires_grad_(True)

    image_tokens = model.backbone.visual_resampler(
        global_feature,
        local_tokens,
        local_mask,
        local_boxes,
    )
    image_padding = torch.zeros(
        image_tokens.size(0),
        image_tokens.size(1) - 1,
        dtype=torch.bool,
        device=image_tokens.device,
    )
    text_padding = batch["clinical_attention_mask"][:, 1:] == 0
    fused, attention = model.fusion(
        image_tokens,
        text_tokens,
        img_key_padding_mask=image_padding,
        txt_key_padding_mask=text_padding,
        return_attn=True,
    )
    logits = model.head(fused)
    image_gradient, text_gradient = torch.autograd.grad(
        logits[0, target_class],
        (image_tokens, text_tokens),
        retain_graph=False,
    )

    raw_visual = reduce_attention_to_keys(
        attention["attn_txt_to_img"], image_padding
    )
    visual_gradient = torch.relu(
        (image_tokens[:, 1:] * image_gradient[:, 1:]).sum(dim=-1)
    )
    if visual_gradient.sum() <= 1e-10:
        visual_gradient = (
            image_tokens[:, 1:] * image_gradient[:, 1:]
        ).abs().sum(dim=-1)
    visual_query_scores = _normalize_scores(raw_visual) * _normalize_scores(
        visual_gradient
    )
    if visual_query_scores.sum() <= 1e-10:
        visual_query_scores = _normalize_scores(raw_visual)
    visual_query_distribution = visual_query_scores / visual_query_scores.sum(
        dim=-1, keepdim=True
    ).clamp_min(1e-8)
    resampler_attention = model.backbone.visual_resampler.last_attention.to(
        visual_query_scores.device
    )
    local_scores = torch.einsum(
        "bq,bqn->bn", visual_query_distribution, resampler_attention
    )

    raw_text = reduce_attention_to_keys(
        attention["attn_img_to_txt"], text_padding
    )
    text_gradient_score = torch.relu(
        (text_tokens[:, 1:] * text_gradient[:, 1:]).sum(dim=-1)
    )
    if text_gradient_score.sum() <= 1e-10:
        text_gradient_score = (
            text_tokens[:, 1:] * text_gradient[:, 1:]
        ).abs().sum(dim=-1)
    valid_text = ~text_padding
    text_scores = _normalize_scores(raw_text, valid_text) * _normalize_scores(
        text_gradient_score, valid_text
    )
    text_scores = _normalize_scores(text_scores, valid_text)

    return {
        "logits": logits.detach(),
        "local_scores": local_scores[0].detach(),
        "local_boxes": local_boxes[0].detach(),
        "text_scores": text_scores[0].detach(),
        "text_token_ids": batch["clinical_input_ids"][0, 1:].detach(),
        "text_valid_mask": valid_text[0].detach(),
        "visual_query_entropy": _entropy(visual_query_distribution[0]),
        "text_entropy": _entropy(text_scores[0][valid_text[0]]),
        "raw_visual_attention": raw_visual[0].detach(),
        "raw_text_attention": raw_text[0].detach(),
    }


def rasterize_scores(
    boxes: np.ndarray,
    scores: np.ndarray,
    image_size: tuple[int, int],
    max_side: int = 840,
):
    width, height = image_size
    scale = min(1.0, max_side / max(width, height))
    out_w = max(1, round(width * scale))
    out_h = max(1, round(height * scale))
    values = np.zeros((out_h, out_w), dtype=np.float32)
    counts = np.zeros_like(values)
    for box, score in zip(boxes, scores):
        x1, y1 = np.floor(box[:2] * [out_w, out_h]).astype(int)
        x2, y2 = np.ceil(box[2:] * [out_w, out_h]).astype(int)
        x1, y1 = np.clip([x1, y1], 0, [out_w - 1, out_h - 1])
        x2, y2 = np.clip(
            [x2, y2], [x1 + 1, y1 + 1], [out_w, out_h]
        )
        values[y1:y2, x1:x2] += float(score)
        counts[y1:y2, x1:x2] += 1
    values /= np.maximum(counts, 1)
    values -= values.min()
    return values / (values.max() + 1e-8)


def merge_wordpieces(tokenizer, token_ids, scores, valid_mask):
    hf_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    pieces = hf_tokenizer.convert_ids_to_tokens(token_ids.tolist())
    special_tokens = set(getattr(hf_tokenizer, "all_special_tokens", []))
    words: list[str] = []
    word_scores: list[float] = []
    for piece, score, valid in zip(pieces, scores.tolist(), valid_mask.tolist()):
        if not valid or piece in special_tokens:
            continue
        if piece.startswith("##") and words:
            words[-1] += piece[2:]
            word_scores[-1] = max(word_scores[-1], float(score))
        else:
            words.append(piece.replace("▁", ""))
            word_scores.append(float(score))
    return words, np.asarray(word_scores, dtype=np.float32)


def _wrap_words(words: list[str], max_characters: int = 64):
    lines: list[list[int]] = [[]]
    used = 0
    for index, word in enumerate(words):
        length = len(word) + 1
        if lines[-1] and used + length > max_characters:
            lines.append([])
            used = 0
        lines[-1].append(index)
        used += length
    return lines


def draw_highlighted_report(ax, words, scores, top_count: int = 8):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    if not words:
        ax.text(0.02, 0.94, "No valid clinical tokens.", va="top")
        return []

    normalized = scores / (scores.max() + 1e-8)
    positive = normalized[normalized > 0]
    bold_threshold = np.quantile(positive, 0.75) if positive.size else 1.0
    lines = _wrap_words(words)
    line_height = min(0.075, 0.82 / max(len(lines), 1))
    cmap = matplotlib.colormaps["YlOrRd"]
    for row, line in enumerate(lines):
        character_position = 0
        y = 0.93 - row * line_height
        for index in line:
            score = float(normalized[index])
            x = 0.02 + character_position / 66.0
            ax.text(
                x,
                y,
                words[index],
                transform=ax.transAxes,
                ha="left",
                va="top",
                family="monospace",
                fontsize=8.7,
                fontweight="bold" if score >= bold_threshold and score > 0 else "normal",
                color="#111111",
                bbox={
                    "facecolor": cmap(0.15 + 0.8 * score),
                    "alpha": 0.12 + 0.78 * score,
                    "edgecolor": "none",
                    "pad": 1.3,
                },
            )
            character_position += len(words[index]) + 1

    unique_scores: dict[str, float] = {}
    for word, score in zip(words, normalized):
        if score > 0:
            unique_scores[word] = max(unique_scores.get(word, 0.0), float(score))
    return sorted(unique_scores.items(), key=lambda item: item[1], reverse=True)[
        :top_count
    ]


def localization_metrics(heatmap, shapes, image_size):
    mask = annotation_mask(shapes, image_size, heatmap.shape)
    if not mask.any():
        return None, None, None
    mass = float((heatmap / (heatmap.sum() + 1e-8))[mask].sum())
    area = float(mask.mean())
    return mass, area, mass / area if area > 0 else None


def render_cross_modal(
    raw_image: Image.Image,
    image_id: str,
    gt_name: str,
    pred_name: str,
    confidence: float,
    heatmap: np.ndarray,
    shapes,
    words,
    word_scores,
    visual_entropy: float,
    text_entropy: float,
    output: Path,
):
    heat_h, heat_w = heatmap.shape
    resized = raw_image.resize((heat_w, heat_h), Image.Resampling.LANCZOS)
    sx, sy = heat_w / raw_image.width, heat_h / raw_image.height
    metrics = localization_metrics(heatmap, shapes, raw_image.size)

    fig = plt.figure(figsize=(17, 7), facecolor="white")
    grid = fig.add_gridspec(1, 3, width_ratios=[0.75, 0.75, 1.65], wspace=0.08)
    input_ax = fig.add_subplot(grid[0, 0])
    heat_ax = fig.add_subplot(grid[0, 1])
    text_ax = fig.add_subplot(grid[0, 2])

    input_ax.imshow(resized, cmap="gray")
    draw_ground_truth(input_ax, shapes, sx, sy)
    input_ax.set_title("Input + ground truth", fontsize=11)
    input_ax.axis("off")

    heat_ax.imshow(resized, cmap="gray")
    heat_ax.imshow(
        heatmap,
        cmap="turbo",
        alpha=0.54,
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    draw_ground_truth(heat_ax, shapes, sx, sy)
    heat_title = "Class-specific text -> image"
    if metrics[2] is not None:
        heat_title += f"\nGT mass {metrics[0]:.1%}, lift {metrics[2]:.2f}x"
    heat_ax.set_title(heat_title, fontsize=11)
    heat_ax.axis("off")

    top_tokens = draw_highlighted_report(text_ax, words, word_scores)
    text_ax.set_title(
        "Clinical report: class-specific image -> text relevance\n"
        "darker + bold = stronger contribution",
        fontsize=11,
        loc="left",
    )
    top_text = "Top tokens: " + ", ".join(
        f"{token} ({score:.2f})" for token, score in top_tokens[:6]
    )
    text_ax.text(
        0.02,
        0.035,
        top_text,
        transform=text_ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        color="#333333",
        wrap=True,
    )
    text_ax.text(
        0.02,
        0.0,
        f"Visual-query entropy: {visual_entropy:.3f} | "
        f"Text-token entropy: {text_entropy:.3f}",
        transform=text_ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        color="#555555",
    )

    correct = gt_name == pred_name
    fig.suptitle(
        f"{image_id} | GT: {gt_name} | Pred: {pred_name} ({confidence:.1%})",
        fontsize=14,
        color="#18732b" if correct else "#a61b1b",
    )
    fig.subplots_adjust(top=0.88, bottom=0.05, left=0.02, right=0.99)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return metrics, top_tokens


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "outputs" / "2026-07-13" / "05-21-15" / ".hydra" / "config.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "checkpoints" / "btxrd" / "proposed" / "xbone_highres" / "seed_42" / "best_phase2.pth",
    )
    parser.add_argument("--image-id", default="IMG000094.jpeg")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "visualization" / "cross_attention_trained.png",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config(args.config.resolve())
    model = load_model(cfg, args.checkpoint.resolve(), device)
    loader = build_dataloader(
        cfg.dataset,
        split="test",
        transform=model.backbone.preprocess,
        tokenizer=model.backbone.tokenizer_obj,
    )
    dataset = loader.dataset
    index = infer_dataset_index(dataset, args.image_id)
    batch = one_sample_batch(dataset[index], model.backbone.tokenizer_obj, device)
    target_class = int(batch["labels"].item())
    attribution = cross_modal_attribution(model, batch, target_class)

    probabilities = torch.softmax(attribution["logits"], dim=-1)[0]
    prediction = int(probabilities.argmax())
    classes = list(cfg.dataset.params.classes)
    raw_image = Image.open(Path(dataset.img_dir) / args.image_id).convert("RGB")
    shapes = load_annotation(find_annotation(args.image_id))
    boxes = attribution["local_boxes"].cpu().numpy()
    heatmap = rasterize_scores(
        boxes,
        attribution["local_scores"].cpu().numpy(),
        raw_image.size,
    )
    words, word_scores = merge_wordpieces(
        model.backbone.tokenizer_obj,
        attribution["text_token_ids"].cpu(),
        attribution["text_scores"].cpu(),
        attribution["text_valid_mask"].cpu(),
    )
    metrics, top_tokens = render_cross_modal(
        raw_image=raw_image,
        image_id=args.image_id,
        gt_name=classes[target_class],
        pred_name=classes[prediction],
        confidence=float(probabilities[prediction]),
        heatmap=heatmap,
        shapes=shapes,
        words=words,
        word_scores=word_scores,
        visual_entropy=attribution["visual_query_entropy"],
        text_entropy=attribution["text_entropy"],
        output=args.output.resolve(),
    )

    summary = {
        "image_id": args.image_id,
        "ground_truth": classes[target_class],
        "prediction": classes[prediction],
        "confidence": float(probabilities[prediction]),
        "image_attention": {
            "gt_mass": metrics[0],
            "gt_area": metrics[1],
            "gt_lift": metrics[2],
            "visual_query_entropy": attribution["visual_query_entropy"],
        },
        "text_attention": {
            "token_entropy": attribution["text_entropy"],
            "top_tokens": [
                {"token": token, "score": score} for token, score in top_tokens
            ],
        },
        "num_local_tokens": int(attribution["local_scores"].numel()),
        "num_text_tokens": len(words),
        "output": str(args.output.resolve()),
    }
    args.output.resolve().with_suffix(".json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
