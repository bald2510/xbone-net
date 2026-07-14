"""Hierarchical causal explanation for the trained high-resolution XBone-Net."""

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
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.builder import build_dataloader
from tools.visualize_attention import (
    ROOT,
    annotation_mask,
    draw_ground_truth,
    find_annotation,
    infer_dataset_index,
    load_annotation,
    load_config,
    load_model,
    one_sample_batch,
)


COMPONENT_NAMES = [
    "Image <- text context",
    "Text <- image context",
]


def fusion_from_tokens(
    model,
    image_tokens: torch.Tensor,
    text_tokens: torch.Tensor,
    text_attention_mask: torch.Tensor,
    component_mask: torch.Tensor | None = None,
    return_attn: bool = False,
):
    """Run the exact trained fusion while exposing its two enhanced branches."""
    fusion = model.fusion
    image_features = fusion.img_proj(fusion.img_input_norm(image_tokens))
    text_features = fusion.txt_proj(fusion.txt_input_norm(text_tokens))
    image_global = fusion.norm_img_global(image_features[:, :1])
    text_global = fusion.norm_txt_global(text_features[:, :1])
    image_local = image_features[:, 1:]
    text_local = text_features[:, 1:]
    text_padding = text_attention_mask[:, 1:] == 0

    direction = getattr(fusion, "attention_direction", "bidirectional")
    uses_i2t = direction in {"bidirectional", "image_to_text"}
    uses_t2i = direction in {"bidirectional", "text_to_image"}

    attn_i2t = None
    image_from_text = fusion.norm_img(image_global)
    if uses_i2t:
        image_context, attn_i2t = fusion.img_to_txt_attn(
            query=image_global,
            key=text_local,
            value=text_local,
            key_padding_mask=text_padding,
            need_weights=return_attn,
            average_attn_weights=False,
        )
        image_from_text = fusion.norm_img(
            image_global + fusion.dropout(image_context)
        )

    attn_t2i = None
    text_from_image = fusion.norm_txt(text_global)
    if uses_t2i:
        text_context, attn_t2i = fusion.txt_to_img_attn(
            query=text_global,
            key=image_local,
            value=image_local,
            need_weights=return_attn,
            average_attn_weights=False,
        )
        text_from_image = fusion.norm_txt(
            text_global + fusion.dropout(text_context)
        )

    components = torch.stack(
        [
            image_from_text[:, 0],
            text_from_image[:, 0],
        ],
        dim=1,
    )
    if component_mask is not None:
        components = components * component_mask.view(
            1, len(COMPONENT_NAMES), 1
        )
    fused = fusion.norm_fuse(fusion.fusion_mlp(components.flatten(1)))
    logits = model.head(fused)
    return logits, components, {"i2t": attn_i2t, "t2i": attn_t2i}


def forward_from_local(
    model,
    global_feature: torch.Tensor,
    local_tokens: torch.Tensor,
    local_mask: torch.Tensor,
    local_boxes: torch.Tensor,
    text_tokens: torch.Tensor,
    text_attention_mask: torch.Tensor,
    component_mask: torch.Tensor | None = None,
    return_attn: bool = False,
):
    image_tokens = model.backbone.visual_resampler(
        global_feature,
        local_tokens,
        local_mask,
        local_boxes,
    )
    logits, components, attention = fusion_from_tokens(
        model,
        image_tokens,
        text_tokens,
        text_attention_mask,
        component_mask=component_mask,
        return_attn=return_attn,
    )
    return logits, components, attention, image_tokens


def branch_ablation(model, cached: dict, target_class: int):
    with torch.no_grad():
        full_logits, _, _, _ = forward_from_local(model, **cached)
        full_prob = torch.softmax(full_logits, dim=-1)[0, target_class]
        full_logit = full_logits[0, target_class]
        logit_drops, probability_drops = [], []
        for index in range(len(COMPONENT_NAMES)):
            component_mask = torch.ones(
                len(COMPONENT_NAMES), device=full_logits.device
            )
            component_mask[index] = 0
            logits, _, _, _ = forward_from_local(
                model, **cached, component_mask=component_mask
            )
            probability = torch.softmax(logits, dim=-1)[0, target_class]
            logit_drops.append(float(full_logit - logits[0, target_class]))
            probability_drops.append(float(full_prob - probability))
    return full_logits, np.asarray(logit_drops), np.asarray(probability_drops)


def integrated_gradients(model, cached: dict, target_class: int, steps: int = 24):
    local = cached["local_tokens"].detach()
    baseline = torch.zeros_like(local)
    gradient_sum = torch.zeros_like(local)
    alphas = torch.linspace(0.0, 1.0, steps, device=local.device)
    for step, alpha in enumerate(alphas):
        interpolated = (baseline + alpha * (local - baseline)).requires_grad_(True)
        logits, _, _, _ = forward_from_local(
            model,
            **{**cached, "local_tokens": interpolated},
        )
        gradient = torch.autograd.grad(
            logits[0, target_class], interpolated, retain_graph=False
        )[0]
        weight = 0.5 if step in (0, len(alphas) - 1) else 1.0
        gradient_sum += gradient * weight
    average_gradient = gradient_sum / max(steps - 1, 1)
    attribution = (local - baseline) * average_gradient
    signed = attribution.sum(dim=-1)
    positive = torch.relu(signed)
    if positive.sum() <= 1e-10:
        positive = attribution.abs().sum(dim=-1)
    return positive[0].detach(), attribution[0].detach()


def gradient_attention_rollout(model, cached: dict, target_class: int):
    local = cached["local_tokens"].detach().requires_grad_(True)
    logits, _, _, image_tokens = forward_from_local(
        model,
        **{**cached, "local_tokens": local},
        return_attn=True,
    )
    gradient = torch.autograd.grad(
        logits[0, target_class], image_tokens, retain_graph=False
    )[0]
    query_relevance = torch.relu((image_tokens[:, 1:] * gradient[:, 1:]).sum(dim=-1))
    if query_relevance.sum() <= 1e-10:
        query_relevance = (image_tokens[:, 1:] * gradient[:, 1:]).abs().sum(dim=-1)
    query_relevance = query_relevance / query_relevance.sum(dim=1, keepdim=True).clamp_min(1e-8)
    resampler_attention = model.backbone.visual_resampler.last_attention.to(
        query_relevance.device
    )
    local_relevance = torch.einsum("bq,bqn->bn", query_relevance, resampler_attention)
    return local_relevance[0].detach(), query_relevance[0].detach()


def probability_for_local(model, cached: dict, local_tokens: torch.Tensor, target_class: int):
    with torch.no_grad():
        logits, _, _, _ = forward_from_local(
            model, **{**cached, "local_tokens": local_tokens}
        )
        return float(torch.softmax(logits, dim=-1)[0, target_class])


def deletion_test(
    model,
    cached: dict,
    relevance: torch.Tensor,
    target_class: int,
    random_trials: int = 32,
):
    local = cached["local_tokens"].detach()
    token_count = local.size(1)
    fractions = np.asarray([0.0, 0.125, 0.25, 0.5, 0.75, 1.0])
    descending = torch.argsort(relevance, descending=True)
    ascending = descending.flip(0)
    generator = torch.Generator(device=local.device).manual_seed(42)
    top_curve, low_curve, random_curve = [], [], []
    for fraction in fractions:
        count = min(token_count, int(round(float(fraction) * token_count)))

        top_masked = local.clone()
        low_masked = local.clone()
        if count:
            top_masked[:, descending[:count]] = 0
            low_masked[:, ascending[:count]] = 0
        top_curve.append(probability_for_local(model, cached, top_masked, target_class))
        low_curve.append(probability_for_local(model, cached, low_masked, target_class))

        trial_values = []
        for _ in range(random_trials):
            random_masked = local.clone()
            if count:
                indices = torch.randperm(
                    token_count, generator=generator, device=local.device
                )[:count]
                random_masked[:, indices] = 0
            trial_values.append(
                probability_for_local(model, cached, random_masked, target_class)
            )
        random_curve.append(float(np.mean(trial_values)))
    return fractions, np.asarray(top_curve), np.asarray(random_curve), np.asarray(low_curve)


def rasterize_scores(
    boxes: np.ndarray,
    scores: np.ndarray,
    image_size: tuple[int, int],
    max_side: int = 720,
):
    width, height = image_size
    scale = min(1.0, max_side / max(width, height))
    out_w, out_h = max(1, round(width * scale)), max(1, round(height * scale))
    values = np.zeros((out_h, out_w), dtype=np.float32)
    counts = np.zeros_like(values)
    for box, score in zip(boxes, scores):
        x1, y1 = np.floor(box[:2] * [out_w, out_h]).astype(int)
        x2, y2 = np.ceil(box[2:] * [out_w, out_h]).astype(int)
        x1, y1 = np.clip([x1, y1], 0, [out_w - 1, out_h - 1])
        x2, y2 = np.clip([x2, y2], [x1 + 1, y1 + 1], [out_w, out_h])
        values[y1:y2, x1:x2] += float(score)
        counts[y1:y2, x1:x2] += 1
    values /= np.maximum(counts, 1)
    values -= values.min()
    return values / (values.max() + 1e-8)


def localization_metrics(heatmap: np.ndarray, shapes, image_size: tuple[int, int]):
    mask = annotation_mask(shapes, image_size, heatmap.shape)
    if not mask.any():
        return None, None, None
    mass = float((heatmap / (heatmap.sum() + 1e-8))[mask].sum())
    area = float(mask.mean())
    return mass, area, mass / area if area > 0 else None


def render(
    image: Image.Image,
    image_id: str,
    gt_name: str,
    pred_name: str,
    confidence: float,
    boxes: np.ndarray,
    shapes,
    ig_map: np.ndarray,
    rollout_map: np.ndarray,
    branch_drops: np.ndarray,
    fractions: np.ndarray,
    top_curve: np.ndarray,
    random_curve: np.ndarray,
    low_curve: np.ndarray,
    output: Path,
):
    height, width = ig_map.shape
    resized = image.resize((width, height), Image.Resampling.LANCZOS)
    sx, sy = width / image.width, height / image.height
    ig_metrics = localization_metrics(ig_map, shapes, image.size)
    rollout_metrics = localization_metrics(rollout_map, shapes, image.size)

    fig, axes = plt.subplots(2, 3, figsize=(16, 11), facecolor="white")
    for ax in axes.flat:
        ax.set_facecolor("white")

    axes[0, 0].imshow(resized, cmap="gray")
    draw_ground_truth(axes[0, 0], shapes, sx, sy)
    axes[0, 0].set_title("Input + ground truth")
    axes[0, 0].axis("off")

    order = np.arange(len(COMPONENT_NAMES))
    bar_colors = ["#2673b8" if value >= 0 else "#b3261e" for value in branch_drops]
    axes[0, 1].barh(order, branch_drops, color=bar_colors)
    axes[0, 1].axvline(0, color="#555555", linewidth=0.8)
    axes[0, 1].set_yticks(order, COMPONENT_NAMES)
    axes[0, 1].invert_yaxis()
    axes[0, 1].set_xlabel("Target-logit drop when removed")
    axes[0, 1].set_title("Fusion branch ablation")
    for index, value in enumerate(branch_drops):
        axes[0, 1].text(
            value,
            index,
            f" {value:+.3f}",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=9,
        )

    axes[0, 2].imshow(resized, cmap="gray")
    axes[0, 2].imshow(ig_map, cmap="inferno", alpha=0.55, interpolation="nearest")
    draw_ground_truth(axes[0, 2], shapes, sx, sy)
    ig_title = "Class-specific Integrated Gradients"
    if ig_metrics[2] is not None:
        ig_title += f"\nGT mass {ig_metrics[0]:.1%}, lift {ig_metrics[2]:.2f}x"
    axes[0, 2].set_title(ig_title)
    axes[0, 2].axis("off")

    axes[1, 0].imshow(resized, cmap="gray")
    axes[1, 0].imshow(rollout_map, cmap="turbo", alpha=0.55, interpolation="nearest")
    draw_ground_truth(axes[1, 0], shapes, sx, sy)
    rollout_title = "Gradient-weighted attention rollout"
    if rollout_metrics[2] is not None:
        rollout_title += f"\nGT mass {rollout_metrics[0]:.1%}, lift {rollout_metrics[2]:.2f}x"
    axes[1, 0].set_title(rollout_title)
    axes[1, 0].axis("off")

    axes[1, 1].imshow(resized, cmap="gray")
    ranked = np.argsort(ig_map.ravel())[::-1]
    token_scores = []
    # Recover one score per local box from the center of its rendered cell.
    for box in boxes:
        cx = min(width - 1, max(0, int((box[0] + box[2]) * 0.5 * width)))
        cy = min(height - 1, max(0, int((box[1] + box[3]) * 0.5 * height)))
        token_scores.append(ig_map[cy, cx])
    for rank, index in enumerate(np.argsort(token_scores)[::-1][:8], start=1):
        x1, y1, x2, y2 = boxes[index]
        axes[1, 1].add_patch(
            mpatches.Rectangle(
                (x1 * width, y1 * height),
                (x2 - x1) * width,
                (y2 - y1) * height,
                fill=False,
                edgecolor="#ff3b30" if rank <= 3 else "#ffcc00",
                linewidth=2.2 if rank <= 3 else 1.3,
            )
        )
        if rank <= 3:
            axes[1, 1].text(
                x1 * width + 3,
                y1 * height + 13,
                str(rank),
                color="white",
                fontsize=8,
                bbox={"facecolor": "black", "alpha": 0.65, "edgecolor": "none"},
            )
    draw_ground_truth(axes[1, 1], shapes, sx, sy)
    axes[1, 1].set_title("Top Integrated-Gradients regions")
    axes[1, 1].axis("off")

    percentage = fractions * 100
    axes[1, 2].plot(percentage, top_curve, "o-", label="Remove highest IG", color="#b3261e")
    axes[1, 2].plot(percentage, random_curve, "o-", label="Remove random", color="#6b7280")
    axes[1, 2].plot(percentage, low_curve, "o-", label="Remove lowest IG", color="#2673b8")
    axes[1, 2].set_xlabel("Local tokens removed (%)")
    axes[1, 2].set_ylabel(f"P({gt_name})")
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].grid(alpha=0.25)
    axes[1, 2].legend(fontsize=9)
    axes[1, 2].set_title(
        f"Causal deletion test\nall local tokens removed: "
        f"{top_curve[0]:.1%} -> {top_curve[-1]:.1%}"
    )

    fig.suptitle(
        f"{image_id} | GT: {gt_name} | Pred: {pred_name} ({confidence:.1%})",
        fontsize=14,
        color="#18732b" if gt_name == pred_name else "#a61b1b",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return ig_metrics, rollout_metrics


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
    parser.add_argument("--ig-steps", type=int, default=24)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "visualization" / "causal_explanation_seed42.png",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config(args.config.resolve())
    model = load_model(cfg, args.checkpoint.resolve(), device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.backbone.explain_mode = True

    loader = build_dataloader(
        cfg.dataset,
        split="test",
        transform=model.backbone.preprocess,
        tokenizer=model.backbone.tokenizer_obj,
    )
    dataset = loader.dataset
    index = infer_dataset_index(dataset, args.image_id)
    batch = one_sample_batch(dataset[index], model.backbone.tokenizer_obj, device)

    with torch.no_grad():
        _, text_tokens = model.backbone(
            batch["pixel_values"],
            batch["clinical_input_ids"],
            attention_mask=batch["clinical_attention_mask"],
            tile_values=batch["tile_values"],
            tile_mask=batch["tile_mask"],
            tile_boxes=batch["tile_boxes"],
        )
    cached = {
        "global_feature": model.backbone.last_global_feature.detach(),
        "local_tokens": model.backbone.last_local_tokens.detach(),
        "local_mask": model.backbone.last_local_mask.detach(),
        "local_boxes": model.backbone.last_local_token_boxes.detach(),
        "text_tokens": text_tokens.detach(),
        "text_attention_mask": batch["clinical_attention_mask"],
    }

    gt_id = int(batch["labels"].item())
    full_logits, branch_drops, probability_drops = branch_ablation(model, cached, gt_id)
    probabilities = torch.softmax(full_logits, dim=-1)[0]
    pred_id = int(probabilities.argmax())
    ig_scores, ig_attribution = integrated_gradients(
        model, cached, gt_id, steps=args.ig_steps
    )
    rollout_scores, query_relevance = gradient_attention_rollout(model, cached, gt_id)
    fractions, top_curve, random_curve, low_curve = deletion_test(
        model, cached, ig_scores, gt_id
    )
    with torch.no_grad():
        zero_logits, _, _, _ = forward_from_local(
            model,
            **{**cached, "local_tokens": torch.zeros_like(cached["local_tokens"])},
        )
    local_logit_difference = float(full_logits[0, gt_id] - zero_logits[0, gt_id])
    ig_signed_sum = float(ig_attribution.sum())

    boxes = cached["local_boxes"][0].cpu().numpy()
    ig_heatmap = rasterize_scores(boxes, ig_scores.cpu().numpy(), Image.open(Path(dataset.img_dir) / args.image_id).size)
    rollout_heatmap = rasterize_scores(
        boxes,
        rollout_scores.cpu().numpy(),
        Image.open(Path(dataset.img_dir) / args.image_id).size,
    )
    raw_image = Image.open(Path(dataset.img_dir) / args.image_id).convert("RGB")
    shapes = load_annotation(find_annotation(args.image_id))
    classes = list(cfg.dataset.params.classes)
    ig_metrics, rollout_metrics = render(
        image=raw_image,
        image_id=args.image_id,
        gt_name=classes[gt_id],
        pred_name=classes[pred_id],
        confidence=float(probabilities[pred_id]),
        boxes=boxes,
        shapes=shapes,
        ig_map=ig_heatmap,
        rollout_map=rollout_heatmap,
        branch_drops=branch_drops,
        fractions=fractions,
        top_curve=top_curve,
        random_curve=random_curve,
        low_curve=low_curve,
        output=args.output.resolve(),
    )

    summary = {
        "image_id": args.image_id,
        "ground_truth": classes[gt_id],
        "prediction": classes[pred_id],
        "confidence": float(probabilities[pred_id]),
        "branch_target_logit_drop": dict(zip(COMPONENT_NAMES, branch_drops.tolist())),
        "branch_target_probability_drop": dict(zip(COMPONENT_NAMES, probability_drops.tolist())),
        "integrated_gradients": {
            "gt_mass": ig_metrics[0],
            "gt_area": ig_metrics[1],
            "gt_lift": ig_metrics[2],
            "signed_attribution_sum": ig_signed_sum,
            "local_logit_difference_vs_zero": local_logit_difference,
            "completeness_error": abs(ig_signed_sum - local_logit_difference),
        },
        "gradient_attention_rollout": {
            "gt_mass": rollout_metrics[0],
            "gt_area": rollout_metrics[1],
            "gt_lift": rollout_metrics[2],
            "query_entropy": float(
                -(query_relevance * torch.log(query_relevance + 1e-8)).sum().cpu()
            ),
        },
        "deletion": {
            "fractions": fractions.tolist(),
            "remove_highest_ig": top_curve.tolist(),
            "remove_random": random_curve.tolist(),
            "remove_lowest_ig": low_curve.tolist(),
        },
        "num_local_tokens": int(cached["local_tokens"].shape[1]),
        "output": str(args.output.resolve()),
    }
    summary_path = args.output.resolve().with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
