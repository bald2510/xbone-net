"""Đánh giá IG end-to-end trên ảnh letterbox và văn bản lâm sàng.

Giao thức này dùng chung cho XBone-Net và các baseline đa phương thức: ảnh được
đánh giá trong không gian pixel đã chuẩn hóa, văn bản được đánh giá trên đường
embedding của tokenizer. Benchmark không phụ thuộc vào nhánh ảnh phụ, phép gom
token thêm hay một kiến trúc fusion cụ thể.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets.ctch import CTCHDataset
from src.utils.analysis import (
    AnalysisDataCollator,
    SOURCE_EXPERIMENT,
    load_evaluated_classification_model,
    load_feature_archive,
    load_locked_proposed_model,
)
from src.utils.explainability import (
    _encode_text_embeddings,
    _text_embedding_path,
    curve_auc,
    stratified_sample_indices,
)


def _ctch_test_dataset(loaded) -> CTCHDataset:
    """Tạo tập CTCH-test với transform/tokenizer của checkpoint đang xét."""
    params = dict(OmegaConf.to_container(loaded.cfg.dataset.params, resolve=True))
    return CTCHDataset(
        split="test",
        transform=loaded.model.backbone.preprocess,
        tokenizer=loaded.model.backbone.tokenizer_obj,
        **params,
    )


def _one_sample(sample: Any, loaded) -> dict[str, Any]:
    """Collate một phần tử dữ liệu và chuyển tensor sang thiết bị model."""
    batch = AnalysisDataCollator(loaded.model.backbone.tokenizer_obj)([sample])
    return {
        key: value.to(loaded.device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _cache_modalities(model, batch: dict[str, Any]) -> dict[str, Any]:
    """Cache đặc trưng ảnh/văn bản và mask cần cho các phép can thiệp."""
    with torch.no_grad():
        encoded = model._encode_modalities(
            batch["pixel_values"],
            batch["clinical_input_ids"],
            attention_mask=batch["clinical_attention_mask"],
        )
    (
        image_features,
        text_features,
        full_image_padding,
        image_padding,
        text_padding,
    ) = encoded
    if image_features is None or text_features is None:
        raise RuntimeError("Explainability requires both image and text inputs.")

    tokenizer = getattr(
        model.backbone.tokenizer_obj,
        "tokenizer",
        model.backbone.tokenizer_obj,
    )
    input_ids = batch["clinical_input_ids"]
    special_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for token_id in getattr(tokenizer, "all_special_ids", []) or []:
        special_mask |= input_ids == int(token_id)
    pad_token_id = getattr(tokenizer, "pad_token_id", 0)
    return {
        "image_features": image_features.detach(),
        "text_features": text_features.detach(),
        "full_image_padding": full_image_padding,
        "image_padding": image_padding,
        "text_padding": text_padding,
        "text_input_ids": input_ids.detach(),
        "text_attention_mask": batch["clinical_attention_mask"].detach(),
        "text_special_token_mask": special_mask.detach(),
        "text_pad_token_id": int(0 if pad_token_id is None else pad_token_id),
    }


def _logits_from_features(
    model,
    cached: dict[str, Any],
    *,
    image_features: torch.Tensor | None = None,
    text_features: torch.Tensor | None = None,
) -> torch.Tensor:
    """Chạy fusion và linear head sau khi thay một nhánh đặc trưng."""
    fused = model._fuse_modalities(
        cached["image_features"] if image_features is None else image_features,
        cached["text_features"] if text_features is None else text_features,
        cached["full_image_padding"],
        cached["image_padding"],
        cached["text_padding"],
    )
    return model.head(fused)


def _encode_image_features(model, pixels: torch.Tensor) -> torch.Tensor:
    """Mã hóa toàn bộ đặc trưng ảnh mà classifier đã cấu hình sử dụng."""
    features, _, _, _, _ = model._encode_modalities(
        pixels,
        input_ids=None,
        attention_mask=None,
    )
    if features is None:
        raise RuntimeError("The image encoder returned no features.")
    return features


def _prepare_text_features(
    cached: dict[str, Any],
    encoded_tokens: torch.Tensor,
) -> torch.Tensor:
    """Khớp token đã mã hóa với giao diện text global hoặc token-level."""
    if cached["text_features"].ndim == 2:
        return torch.nn.functional.normalize(encoded_tokens[:, 0], dim=-1)
    return encoded_tokens


def _target_outputs(
    model,
    cached: dict[str, Any],
    target_class: int,
    *,
    image_features: torch.Tensor | None = None,
    text_features: torch.Tensor | None = None,
) -> tuple[float, float, float]:
    """Trả về xác suất, logit và margin của lớp mục tiêu."""
    with torch.no_grad():
        logits = _logits_from_features(
            model,
            cached,
            image_features=image_features,
            text_features=text_features,
        )
        probability = torch.softmax(logits, dim=-1)[0, target_class]
        target_logit = logits[0, target_class]
        alternatives = logits[0].clone()
        alternatives[target_class] = -torch.inf
        margin = target_logit - alternatives.max()
    return float(probability), float(target_logit), float(margin)


def _integrated_gradients_image(
    model,
    cached: dict[str, Any],
    pixels: torch.Tensor,
    target_class: int,
    steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tính Integrated Gradients từ baseline zero trong không gian ảnh."""
    if steps < 2:
        raise ValueError("Integrated Gradients requires at least two steps.")
    pixels = pixels.detach()
    baseline = torch.zeros_like(pixels)
    gradient_sum = torch.zeros_like(pixels)
    for index, alpha in enumerate(
        torch.linspace(0.0, 1.0, steps, device=pixels.device)
    ):
        interpolated = (baseline + alpha * (pixels - baseline)).detach()
        interpolated.requires_grad_(True)
        image_features = _encode_image_features(model, interpolated)
        logits = _logits_from_features(
            model,
            cached,
            image_features=image_features,
        )
        gradient = torch.autograd.grad(
            logits[0, target_class], interpolated, retain_graph=False
        )[0]
        gradient_sum += gradient * (0.5 if index in (0, steps - 1) else 1.0)
    attribution = (pixels - baseline) * gradient_sum / (steps - 1)
    signed = attribution.sum(dim=1)
    relevance = torch.relu(signed)
    if float(relevance.detach().sum()) <= 1e-10:
        relevance = attribution.abs().sum(dim=1)
    return relevance[0].detach(), attribution[0].detach()


def _integrated_gradients_text(
    model,
    cached: dict[str, Any],
    target_class: int,
    steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tính IG trên word embedding với baseline padding của tokenizer."""
    if steps < 2:
        raise ValueError("Integrated Gradients requires at least two steps.")
    text, baseline, transformer, projection = _text_embedding_path(model, cached)
    gradient_sum = torch.zeros_like(text)
    for index, alpha in enumerate(
        torch.linspace(0.0, 1.0, steps, device=text.device)
    ):
        interpolated = (baseline + alpha * (text - baseline)).detach()
        interpolated.requires_grad_(True)
        text_features = _prepare_text_features(
            cached,
            _encode_text_embeddings(
                transformer,
                projection,
                interpolated,
                cached["text_attention_mask"],
            ),
        )
        logits = _logits_from_features(
            model,
            cached,
            text_features=text_features,
        )
        gradient = torch.autograd.grad(
            logits[0, target_class], interpolated, retain_graph=False
        )[0]
        gradient_sum += gradient * (0.5 if index in (0, steps - 1) else 1.0)
    attribution = (text - baseline) * gradient_sum / (steps - 1)
    signed = attribution.sum(dim=-1)
    relevance = torch.relu(signed)
    if float(relevance.detach().sum()) <= 1e-10:
        relevance = attribution.abs().sum(dim=-1)
    valid = cached["text_attention_mask"].to(relevance.dtype)
    valid *= (~cached["text_special_token_mask"]).to(relevance.dtype)
    return (relevance * valid)[0].detach(), attribution[0].detach()


def _empty_curves() -> dict[str, list[float]]:
    """Khởi tạo bộ chứa các đường deletion/insertion."""
    return {
        key: []
        for key in (
            "delete_most_relevant",
            "delete_random",
            "insert_most_relevant",
            "delete_most_relevant_logit",
            "delete_random_logit",
            "insert_most_relevant_logit",
            "delete_most_relevant_margin",
            "delete_random_margin",
            "insert_most_relevant_margin",
        )
    }


def _append_curve_outputs(
    output: dict[str, list[float]],
    targeted: tuple[float, float, float],
    random_mean: np.ndarray,
    insertion: tuple[float, float, float],
) -> None:
    """Thêm kết quả probability/logit/margin vào bộ đường cong."""
    for suffix, position in (("", 0), ("_logit", 1), ("_margin", 2)):
        output[f"delete_most_relevant{suffix}"].append(targeted[position])
        output[f"delete_random{suffix}"].append(float(random_mean[position]))
        output[f"insert_most_relevant{suffix}"].append(insertion[position])


def _image_perturbation_curves(
    model,
    cached: dict[str, Any],
    pixels: torch.Tensor,
    relevance: torch.Tensor,
    target_class: int,
    random_trials: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Đánh giá deletion/insertion theo patch 16x16 của ảnh letterbox."""
    if random_trials < 1:
        raise ValueError("random_trials must be positive.")
    baseline = torch.zeros_like(pixels)
    patch_embed = getattr(
        getattr(getattr(model.backbone.model, "visual", None), "trunk", None),
        "patch_embed",
        None,
    )
    grid_size = getattr(patch_embed, "grid_size", (14, 14))
    grid_h, grid_w = (
        (int(grid_size), int(grid_size))
        if isinstance(grid_size, int)
        else (int(grid_size[0]), int(grid_size[1]))
    )
    height, width = pixels.shape[-2:]
    y_edges = np.linspace(0, height, grid_h + 1).round().astype(int)
    x_edges = np.linspace(0, width, grid_w + 1).round().astype(int)
    patches = [
        (slice(y_edges[row], y_edges[row + 1]), slice(x_edges[col], x_edges[col + 1]))
        for row in range(grid_h)
        for col in range(grid_w)
    ]
    scores = torch.stack([relevance[y, x].mean() for y, x in patches])
    ranking = torch.argsort(scores, descending=True)
    generator = torch.Generator(device=pixels.device).manual_seed(int(seed))
    random_orders = [
        torch.randperm(len(patches), generator=generator, device=pixels.device)
        for _ in range(random_trials)
    ]
    fractions = np.linspace(0.0, 1.0, 11, dtype=np.float64)
    output = _empty_curves()

    def evaluate(current: torch.Tensor) -> tuple[float, float, float]:
        """Mã hóa ảnh đã can thiệp rồi trả probability, logit và margin."""
        with torch.no_grad():
            image_features = _encode_image_features(model, current)
        return _target_outputs(
            model,
            cached,
            target_class,
            image_features=image_features,
        )

    for fraction in fractions:
        count = min(len(patches), int(round(float(fraction) * len(patches))))
        deleted = pixels.clone()
        inserted = baseline.clone()
        for patch_index in ranking[:count].tolist():
            y_slice, x_slice = patches[int(patch_index)]
            deleted[:, :, y_slice, x_slice] = baseline[:, :, y_slice, x_slice]
            inserted[:, :, y_slice, x_slice] = pixels[:, :, y_slice, x_slice]
        targeted = evaluate(deleted)
        insertion = evaluate(inserted)
        random_values = []
        for order in random_orders:
            random_pixels = pixels.clone()
            for patch_index in order[:count].tolist():
                y_slice, x_slice = patches[int(patch_index)]
                random_pixels[:, :, y_slice, x_slice] = baseline[
                    :, :, y_slice, x_slice
                ]
            random_values.append(evaluate(random_pixels))
        random_mean = np.asarray(random_values, dtype=np.float64).mean(axis=0)
        _append_curve_outputs(output, targeted, random_mean, insertion)
    return {
        "fractions": fractions,
        **{key: np.asarray(values, dtype=np.float64) for key, values in output.items()},
    }


def _text_perturbation_curves(
    model,
    cached: dict[str, Any],
    relevance: torch.Tensor,
    target_class: int,
    random_trials: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Đánh giá deletion/insertion trên các token nội dung lâm sàng."""
    if random_trials < 1:
        raise ValueError("random_trials must be positive.")
    text, baseline, transformer, projection = _text_embedding_path(model, cached)
    valid = cached["text_attention_mask"][0].bool()
    valid &= ~cached["text_special_token_mask"][0].bool()
    candidates = torch.nonzero(valid, as_tuple=False).flatten()
    if candidates.numel() < 1:
        raise ValueError("Clinical report contains no content tokens.")
    ranking = candidates[torch.argsort(relevance[candidates], descending=True)]
    generator = torch.Generator(device=text.device).manual_seed(int(seed))
    random_orders = [
        torch.randperm(len(candidates), generator=generator, device=text.device)
        for _ in range(random_trials)
    ]
    fractions = np.linspace(0.0, 1.0, 11, dtype=np.float64)
    output = _empty_curves()

    def evaluate(embeddings: torch.Tensor) -> tuple[float, float, float]:
        """Mã hóa embedding văn bản đã can thiệp và chấm lớp mục tiêu."""
        with torch.no_grad():
            features = _prepare_text_features(
                cached,
                _encode_text_embeddings(
                    transformer,
                    projection,
                    embeddings,
                    cached["text_attention_mask"],
                ),
            )
        return _target_outputs(
            model,
            cached,
            target_class,
            text_features=features,
        )

    for fraction in fractions:
        count = min(len(candidates), int(round(float(fraction) * len(candidates))))
        deleted = text.clone()
        inserted = baseline.clone()
        if count:
            selected = ranking[:count]
            deleted[:, selected] = baseline[:, selected]
            inserted[:, selected] = text[:, selected]
        targeted = evaluate(deleted)
        insertion = evaluate(inserted)
        random_values = []
        for order in random_orders:
            random_text = text.clone()
            selected = candidates[order[:count]]
            random_text[:, selected] = baseline[:, selected]
            random_values.append(evaluate(random_text))
        random_mean = np.asarray(random_values, dtype=np.float64).mean(axis=0)
        _append_curve_outputs(output, targeted, random_mean, insertion)
    return {
        "fractions": fractions,
        **{key: np.asarray(values, dtype=np.float64) for key, values in output.items()},
    }


def _entropy(values: np.ndarray) -> float:
    """Tính entropy của phân phối relevance đã chuẩn hóa."""
    weights = np.maximum(np.asarray(values, dtype=np.float64).reshape(-1), 0.0)
    if weights.sum() <= 0:
        return 0.0
    probabilities = weights / weights.sum()
    probabilities = probabilities[probabilities > 0]
    return float(-(probabilities * np.log(probabilities)).sum())


def _faithfulness(curves: dict[str, np.ndarray]) -> dict[str, float]:
    """Tóm tắt các đường cong can thiệp bằng AUC."""
    deletion = curve_auc(curves["fractions"], curves["delete_most_relevant"])
    random_deletion = curve_auc(curves["fractions"], curves["delete_random"])
    return {
        "deletion_auc": deletion,
        "random_deletion_auc": random_deletion,
        "insertion_auc": curve_auc(
            curves["fractions"], curves["insert_most_relevant"]
        ),
        "random_minus_targeted_deletion_auc": random_deletion - deletion,
        "target_logit_deletion_auc": curve_auc(
            curves["fractions"], curves["delete_most_relevant_logit"]
        ),
        "random_target_logit_deletion_auc": curve_auc(
            curves["fractions"], curves["delete_random_logit"]
        ),
    }


def _sample_explanation(
    loaded,
    dataset: CTCHDataset,
    dataset_index: int,
    image_id: str,
    ig_steps: int,
    random_trials: int,
    seed: int,
) -> dict[str, Any]:
    """Tính IG và faithfulness cho một mẫu CTCH-test."""
    batch = _one_sample(dataset[dataset_index], loaded)
    cached = _cache_modalities(loaded.model, batch)
    with torch.no_grad():
        cached_logits = _logits_from_features(loaded.model, cached)
        direct_logits = loaded.model(
            batch["pixel_values"],
            batch["clinical_input_ids"],
            attention_mask=batch["clinical_attention_mask"],
        )
    forward_error = float((cached_logits - direct_logits).abs().max())
    if forward_error > 1e-4:
        raise RuntimeError(
            "Cached forward differs from model.forward: "
            f"max_abs_error={forward_error:.6g}."
        )
    probabilities = torch.softmax(cached_logits, dim=-1)[0]
    ground_truth = int(batch["labels"].item())
    prediction = int(probabilities.argmax())
    image_scores, image_attribution = _integrated_gradients_image(
        loaded.model,
        cached,
        batch["pixel_values"],
        prediction,
        ig_steps,
    )
    text_scores, text_attribution = _integrated_gradients_text(
        loaded.model,
        cached,
        prediction,
        ig_steps,
    )
    image_curves = _image_perturbation_curves(
        loaded.model,
        cached,
        batch["pixel_values"],
        image_scores,
        prediction,
        random_trials,
        seed + 2000,
    )
    text_curves = _text_perturbation_curves(
        loaded.model,
        cached,
        text_scores,
        prediction,
        random_trials,
        seed + 3000,
    )
    full_logit = float(cached_logits[0, prediction])
    full_probability = float(probabilities[prediction])
    image_difference = full_logit - float(
        image_curves["delete_most_relevant_logit"][-1]
    )
    text_difference = full_logit - float(
        text_curves["delete_most_relevant_logit"][-1]
    )
    image_sum = float(image_attribution.sum())
    text_sum = float(text_attribution.sum())
    return {
        "image_id": image_id,
        "ground_truth": ground_truth,
        "prediction": prediction,
        "correct": bool(ground_truth == prediction),
        "confidence": full_probability,
        "target_class": prediction,
        "cached_forward_max_abs_error": forward_error,
        "source_target_probability_drop": {
            "global_image": full_probability
            - float(image_curves["delete_most_relevant"][-1]),
            "clinical_text": full_probability
            - float(text_curves["delete_most_relevant"][-1]),
        },
        "global_image_integrated_gradients": {
            "signed_attribution_sum": image_sum,
            "logit_difference_vs_zero_input": image_difference,
            "completeness_error": abs(image_sum - image_difference),
            "relative_completeness_error": abs(image_sum - image_difference)
            / max(abs(image_difference), 1e-8),
            "entropy": _entropy(image_scores.cpu().numpy()),
        },
        "clinical_text_integrated_gradients": {
            "signed_attribution_sum": text_sum,
            "logit_difference_vs_padding_baseline": text_difference,
            "completeness_error": abs(text_sum - text_difference),
            "relative_completeness_error": abs(text_sum - text_difference)
            / max(abs(text_difference), 1e-8),
            "entropy": _entropy(text_scores.cpu().numpy()),
        },
        "global_image_faithfulness": _faithfulness(image_curves),
        "clinical_text_input_faithfulness": _faithfulness(text_curves),
        "global_image_curves": {
            key: value.tolist() for key, value in image_curves.items()
        },
        "clinical_text_input_curves": {
            key: value.tolist() for key, value in text_curves.items()
        },
    }


def _flatten_numeric(value: Any, prefix: str = "") -> dict[str, float]:
    """Trải phẳng các giá trị số lồng nhau để tổng hợp bootstrap."""
    result: dict[str, float] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else key
            result.update(_flatten_numeric(child, child_prefix))
    elif isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
        value, bool
    ):
        number = float(value)
        if np.isfinite(number):
            result[prefix] = number
    return result


def _aggregate_records(
    records: list[dict[str, Any]],
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Tổng hợp mean, độ lệch chuẩn và bootstrap CI 95%."""
    if not records:
        return {}
    excluded = {"ground_truth", "prediction", "target_class"}
    keys = sorted(set().union(*(_flatten_numeric(record).keys() for record in records)))
    rng = np.random.default_rng(bootstrap_seed)
    output: dict[str, Any] = {}
    flattened_records = [_flatten_numeric(record) for record in records]
    for key in keys:
        if key in excluded:
            continue
        values = np.asarray(
            [record[key] for record in flattened_records if key in record],
            dtype=np.float64,
        )
        if values.size == 0:
            continue
        bootstrap = np.asarray(
            [rng.choice(values, size=len(values), replace=True).mean() for _ in range(2000)]
        )
        output[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "median": float(np.median(values)),
            "ci_95": [
                float(np.percentile(bootstrap, 2.5)),
                float(np.percentile(bootstrap, 97.5)),
            ],
            "n": int(len(values)),
        }
    return output


def _subgroups(records: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    """Tổng hợp riêng mẫu đúng/sai và nhóm confidence thấp/cao."""
    if not records:
        return {}
    confidence = np.asarray([record["confidence"] for record in records])
    q25, q75 = np.quantile(confidence, [0.25, 0.75])
    groups = OrderedDict(
        [
            ("correct", [record for record in records if record["correct"]]),
            ("incorrect", [record for record in records if not record["correct"]]),
            (
                "low_confidence_q1",
                [record for record in records if record["confidence"] <= q25],
            ),
            (
                "high_confidence_q4",
                [record for record in records if record["confidence"] >= q75],
            ),
        ]
    )
    return {
        name: {
            "sample_count": len(group),
            "aggregate": _aggregate_records(group, seed + index),
        }
        for index, (name, group) in enumerate(groups.items())
        if group
    }


def _source_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Mô tả modality gây giảm xác suất lớn nhất khi bị xóa."""
    if not records:
        return {}
    sources = tuple(records[0]["source_target_probability_drop"])
    values = {
        source: np.asarray(
            [record["source_target_probability_drop"][source] for record in records]
        )
        for source in sources
    }
    dominant = {source: 0 for source in sources}
    for index in range(len(records)):
        dominant[max(sources, key=lambda source: values[source][index])] += 1
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


def _parser() -> argparse.ArgumentParser:
    """Khai báo CLI, giữ các cờ cũ vô hại để lệnh hiện có vẫn chạy."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default=SOURCE_EXPERIMENT)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--train-features", type=Path, required=True)
    parser.add_argument("--test-features", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=44)
    parser.add_argument("--all-test-samples", action="store_true")
    parser.add_argument("--selection-seed", type=int, default=1907)
    parser.add_argument("--ig-steps", type=int, default=24)
    parser.add_argument("--random-trials", type=int, default=16)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--no-strict-fingerprint", action="store_true")
    parser.add_argument("--input-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--visualization-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--stability-repeats", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--stability-steps", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--noise-scale", type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument("--sanity-samples", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--render-samples", type=int, default=0, help=argparse.SUPPRESS)
    return parser


def main() -> None:
    """Chạy benchmark và ghi ``samples.json`` cùng ``summary.json``."""
    args = _parser().parse_args()
    summary_path = args.output_dir / "summary.json"
    if summary_path.is_file() and not args.overwrite:
        print(f"[Resume] Explainability summary exists: {summary_path}")
        return
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable.")
        device = torch.device(args.device)

    loaded = (
        load_locked_proposed_model(
            args.seed,
            device=device,
            strict_fingerprint=not args.no_strict_fingerprint,
        )
        if args.experiment == SOURCE_EXPERIMENT
        else load_evaluated_classification_model(
            args.experiment,
            args.seed,
            device=device,
        )
    )
    _, train_provenance = load_feature_archive(
        args.train_features,
        expected_source_experiment=args.experiment,
    )
    test, test_provenance = load_feature_archive(
        args.test_features,
        expected_source_experiment=args.experiment,
    )
    for provenance, scenario in (
        (train_provenance, "ctch_train"),
        (test_provenance, "ctch_test"),
    ):
        if int(provenance.get("seed", -1)) != args.seed:
            raise ValueError("Feature seed differs from explainability seed.")
        if provenance.get("scenario") != scenario:
            raise ValueError(f"Expected {scenario} features.")
        if provenance.get("checkpoint_sha256") != loaded.checkpoint_sha256:
            raise ValueError("Feature archive and checkpoint checksums differ.")

    dataset = _ctch_test_dataset(loaded)
    index_by_id = {
        str(row["image_id"]): int(index) for index, row in dataset.df.iterrows()
    }
    selected = (
        np.arange(len(test["labels"]), dtype=np.int64)
        if args.all_test_samples
        else stratified_sample_indices(
            test["labels"],
            per_class=args.per_class,
            seed=args.selection_seed,
            max_samples=args.max_samples,
        )
    )
    selected_ids = test["image_id"].astype(str)[selected]
    missing = [image_id for image_id in selected_ids if image_id not in index_by_id]
    if missing:
        raise ValueError(f"Selected IDs are absent from CTCH-test: {missing[:5]}")

    for parameter in loaded.model.parameters():
        parameter.requires_grad_(False)
    loaded.model.eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    samples_path = args.output_dir / "samples.json"
    for position, image_id in enumerate(selected_ids):
        print(f"[Explain {position + 1}/{len(selected_ids)}] {image_id}")
        records.append(
            _sample_explanation(
                loaded,
                dataset,
                index_by_id[image_id],
                image_id,
                ig_steps=args.ig_steps,
                random_trials=args.random_trials,
                seed=args.selection_seed + position,
            )
        )
        samples_path.write_text(
            json.dumps(records, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    summary = {
        "type": "ctch_explainability_evaluation",
        "source_experiment": args.experiment,
        "seed": args.seed,
        "checkpoint_sha256": loaded.checkpoint_sha256,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "protocol": {
            "analysis_mode": "input_only",
            "image_preprocessing": "letterbox_224_then_biomedclip_transform",
            "classifier": "linear_head",
            "target": "predicted_class",
            "sampling": "all_test_samples" if args.all_test_samples else "stratified",
            "selection_seed": args.selection_seed,
            "ig_steps": args.ig_steps,
            "ig_baselines": {
                "global_image": "zero_in_normalized_input_space",
                "clinical_text": "padding_embedding_with_special_tokens_preserved",
            },
            "random_trials": args.random_trials,
            "faithfulness_modalities": ["global_image", "clinical_text"],
            "spatial_localization_metrics": (
                "not_reported: CTCH has no lesion-region annotations"
            ),
        },
        "sample_count": len(records),
        "correct_count": int(sum(record["correct"] for record in records)),
        "explanation_aggregate": _aggregate_records(
            records, args.selection_seed + 5000
        ),
        "explanation_subgroups": _subgroups(records, args.selection_seed + 6000),
        "source_role_summary": _source_summary(records),
        "sample_records": str(samples_path),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
