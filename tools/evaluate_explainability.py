"""Batch explainability and representation audit for CTCH proposed checkpoints."""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets.ctch import CTCHDataset
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
    gradient_text_relevance,
    gradient_visual_relevance,
    integrated_gradients,
    perturbation_curves,
    rank_correlation,
    representation_quality_metrics,
    stratified_sample_indices,
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


def _cached_tokens(model, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        _, text_tokens = model.backbone(
            batch["pixel_values"],
            batch["clinical_input_ids"],
            attention_mask=batch["clinical_attention_mask"],
            tile_values=batch.get("tile_values"),
            tile_mask=batch.get("tile_mask"),
            tile_boxes=batch.get("tile_boxes"),
        )
    return {
        "global_feature": model.backbone.last_global_feature.detach(),
        "local_tokens": model.backbone.last_local_tokens.detach(),
        "local_mask": model.backbone.last_local_mask.detach(),
        "local_boxes": model.backbone.last_local_token_boxes.detach(),
        "text_tokens": text_tokens.detach(),
        "text_attention_mask": batch["clinical_attention_mask"],
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
    ig_scores, ig_attribution = integrated_gradients(
        loaded.model, cached, target_class, steps=ig_steps
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
        "integrated_gradients": {
            "signed_attribution_sum": signed_sum,
            "local_logit_difference_vs_global_baseline": logit_difference,
            "completeness_error": abs(signed_sum - logit_difference),
            "relative_completeness_error": abs(signed_sum - logit_difference)
            / max(abs(logit_difference), 1e-8),
            "entropy": _entropy(ig_scores.cpu().numpy()),
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
        "text_curves": {
            key: value.tolist() for key, value in text_curves.items()
        },
    }
    if render_path is not None:
        boxes = cached["local_boxes"][0].cpu().numpy()
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--train-features", type=Path, required=True)
    parser.add_argument("--test-features", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=44)
    parser.add_argument("--selection-seed", type=int, default=1907)
    parser.add_argument("--ig-steps", type=int, default=24)
    parser.add_argument("--random-trials", type=int, default=16)
    parser.add_argument("--stability-repeats", type=int, default=2)
    parser.add_argument("--stability-steps", type=int, default=8)
    parser.add_argument("--noise-scale", type=float, default=0.01)
    parser.add_argument("--sanity-samples", type=int, default=5)
    parser.add_argument("--render-samples", type=int, default=6)
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
        if (
            existing.get("source_experiment") == SOURCE_EXPERIMENT
            and int(existing.get("seed", -1)) == args.seed
            and feature_shas == {existing.get("checkpoint_sha256")}
            and int(protocol.get("per_class", -1)) == args.per_class
            and int(protocol.get("max_samples", -1)) == args.max_samples
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
            "sampling": "stratified_by_ground_truth_class",
            "per_class": args.per_class,
            "max_samples": args.max_samples,
            "selection_seed": args.selection_seed,
            "ig_steps": args.ig_steps,
            "ig_baseline": "global_image_embedding_repeated_as_local_tokens",
            "random_trials": args.random_trials,
            "faithfulness_modalities": ["visual_local_tokens", "clinical_tokens"],
            "stability_repeats": args.stability_repeats,
            "stability_steps": args.stability_steps,
            "stability_noise_scale": args.noise_scale,
            "classifier_randomization_samples": min(
                args.sanity_samples, len(records)
            ),
            "render_samples": min(args.render_samples, len(records)),
            "spatial_localization_metrics": (
                "not_reported: CTCH has no lesion-region annotations in the "
                "analysis workspace"
            ),
        },
        "sample_count": len(records),
        "correct_count": int(sum(record["correct"] for record in records)),
        "representation_quality": representation,
        "representation_visualization": str(representation_plot),
        "explanation_aggregate": _aggregate_records(
            records, bootstrap_seed=args.selection_seed + 5000
        ),
        "sample_records": str(args.output_dir / "samples.json"),
        "visualization_dir": str(args.output_dir / "visualizations"),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
