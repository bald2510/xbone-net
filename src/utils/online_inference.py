"""Reusable online inference utilities for the XBone-Net Streamlit demo.

The implementation deliberately follows the locked CTCH analysis protocol:

* load the evaluated ``ctch/proposed/ours_xbone_net`` checkpoint;
* build sparse-focal high-resolution inputs with the recorded configuration;
* classify from the trained bidirectional fused representation;
* fit OOD reference statistics on CTCH train and calibrate thresholds on
  CTCH validation-ID only;
* compute Integrated Gradients over the exact local visual tokens consumed by
  the fusion module.

Larger OOD scores always indicate stronger OOD evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import colormaps
from omegaconf import OmegaConf
from PIL import Image, ImageFilter

from src.datasets.high_resolution import prepare_high_resolution_inputs
from src.utils.analysis import (
    SOURCE_EXPERIMENT,
    analysis_root,
    load_feature_archive,
    load_locked_proposed_model,
    load_proposed_experiment_model,
)
from src.utils.explainability import (
    curve_auc,
    global_image_perturbation_curves,
    integrated_gradients,
    integrated_gradients_global_image,
    integrated_gradients_text,
    perturbation_curves,
    text_input_perturbation_curves,
)
from src.utils.ood import OODDetector, calibrate_ood_threshold
from src.utils.trainer import resolve_pad_token_id


OOD_METHODS = (
    "cosine_centroids",
    "mahalanobis_centroid",
    "knn",
    "entropy",
)


@dataclass(frozen=True)
class OODCalibration:
    """Fitted train-ID references and validation-ID operating thresholds."""

    normalized_detector: OODDetector
    raw_detector: OODDetector
    thresholds: dict[str, float]
    calibration_counts: dict[str, int]
    target_id_fpr: float


@dataclass
class OnlineInferenceResult:
    """All outputs needed by the interactive demo."""

    predicted_index: int | None
    predicted_label: str | None
    probabilities: np.ndarray | None
    class_labels: list[str]
    ood_method: str
    ood_score: float
    ood_threshold: float
    is_ood: bool
    local_boxes: np.ndarray | None
    local_ig_scores: np.ndarray | None
    spatial_ig_boxes: np.ndarray | None
    spatial_ig_scores: np.ndarray | None
    global_ig_map: np.ndarray | None
    text_tokens: list[str] | None
    text_ig_scores: np.ndarray | None
    contribution_drops: dict[str, float] | None
    global_faithfulness: dict[str, Any] | None
    visual_faithfulness: dict[str, Any] | None
    text_faithfulness: dict[str, Any] | None
    seed: int
    checkpoint: str


def _labels(values: np.ndarray) -> np.ndarray:
    labels = np.asarray(values)
    if labels.ndim == 2:
        labels = labels.argmax(axis=1)
    return labels.astype(np.int64).reshape(-1)


def _score_method(
    method: str,
    normalized_detector: OODDetector,
    raw_detector: OODDetector,
    *,
    fused_embedding: np.ndarray,
    fused_embedding_raw: np.ndarray,
    logits: np.ndarray,
    knn_k: int,
) -> np.ndarray:
    if method == "cosine_centroids":
        return normalized_detector.score_cosine_centroids(fused_embedding)
    if method == "mahalanobis_centroid":
        return raw_detector.score_mahalanobis_centroid(fused_embedding_raw)
    if method == "knn":
        return normalized_detector.score_knn(fused_embedding, k=knn_k)
    if method == "entropy":
        return normalized_detector.score_entropy(logits)
    raise ValueError(f"Unsupported OOD method: {method}")


def _calibration_scores(
    method: str,
    normalized_detector: OODDetector,
    raw_detector: OODDetector,
    arrays: Mapping[str, np.ndarray],
    knn_k: int,
) -> np.ndarray:
    return _score_method(
        method,
        normalized_detector,
        raw_detector,
        fused_embedding=np.asarray(arrays["fused_embeddings"]),
        fused_embedding_raw=np.asarray(arrays["fused_embeddings_raw"]),
        logits=np.asarray(arrays["logits"]),
        knn_k=knn_k,
    )


def fit_locked_ood_calibration(
    loaded,
    *,
    target_id_fpr: float = 0.05,
    knn_k: int = 5,
) -> OODCalibration:
    """Fit the canonical OOD detector from saved CTCH train/validation features."""

    feature_root = analysis_root(loaded.provenance["seed"]) / "features"
    train, train_provenance = load_feature_archive(
        feature_root / "ctch_train.npz",
        expected_source_experiment=SOURCE_EXPERIMENT,
    )
    validation, validation_provenance = load_feature_archive(
        feature_root / "ctch_val.npz",
        expected_source_experiment=SOURCE_EXPERIMENT,
    )

    for split_name, provenance in (
        ("CTCH train", train_provenance),
        ("CTCH validation", validation_provenance),
    ):
        archive_hash = str(provenance.get("checkpoint_sha256", ""))
        if archive_hash != loaded.checkpoint_sha256:
            raise RuntimeError(
                f"{split_name} feature archive was not produced by the loaded "
                "checkpoint. Re-export the feature archives before online OOD "
                "inference."
            )

    train_labels = _labels(train["labels"])
    normalized_detector = OODDetector().fit(
        np.asarray(train["fused_embeddings"]),
        train_labels,
    )
    raw_detector = OODDetector().fit(
        np.asarray(train["fused_embeddings_raw"]),
        train_labels,
    )

    thresholds: dict[str, float] = {}
    counts: dict[str, int] = {}
    for method in OOD_METHODS:
        scores = _calibration_scores(
            method,
            normalized_detector,
            raw_detector,
            validation,
            knn_k,
        )
        thresholds[method] = calibrate_ood_threshold(
            scores,
            target_id_fpr=target_id_fpr,
        )
        counts[method] = int(len(scores))

    return OODCalibration(
        normalized_detector=normalized_detector,
        raw_detector=raw_detector,
        thresholds=thresholds,
        calibration_counts=counts,
        target_id_fpr=float(target_id_fpr),
    )


def _tokenize_text(tokenizer, text: str, device: torch.device):
    tokenized = tokenizer([text])
    attention_mask = None
    if isinstance(tokenized, Mapping):
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized.get("attention_mask")
    else:
        input_ids = tokenized
    if not isinstance(input_ids, torch.Tensor):
        input_ids = torch.as_tensor(input_ids, dtype=torch.long)
    input_ids = input_ids.to(device)
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    if attention_mask is None:
        pad_id = resolve_pad_token_id(tokenizer)
        attention_mask = (input_ids != pad_id).long()
    else:
        attention_mask = torch.as_tensor(attention_mask, dtype=torch.long, device=device)
        if attention_mask.ndim == 1:
            attention_mask = attention_mask.unsqueeze(0)
    return input_ids, attention_mask


def _build_online_inputs(loaded, image: Image.Image, clinical_text: str):
    cfg = loaded.cfg
    high_res_cfg = OmegaConf.to_container(
        cfg.dataset.params.high_res,
        resolve=True,
    )
    if not isinstance(high_res_cfg, dict) or not high_res_cfg.get("enabled", False):
        raise RuntimeError(
            "The locked proposed checkpoint is expected to use sparse-focal "
            "high-resolution preprocessing."
        )

    fields, selection = prepare_high_resolution_inputs(
        image.convert("RGB"),
        loaded.model.backbone.preprocess,
        high_res_cfg,
        return_selection=True,
    )
    pixel_values = fields["pixel_values"].unsqueeze(0).to(loaded.device)
    tile_values = fields["tile_values"].unsqueeze(0).to(loaded.device)
    tile_boxes = fields["tile_boxes"].unsqueeze(0).to(loaded.device)
    tile_mask = torch.ones(
        tile_values.shape[:2],
        dtype=torch.long,
        device=loaded.device,
    )
    input_ids, attention_mask = _tokenize_text(
        loaded.model.backbone.tokenizer_obj,
        clinical_text,
        loaded.device,
    )
    return {
        "pixel_values": pixel_values,
        "tile_values": tile_values,
        "tile_mask": tile_mask,
        "tile_boxes": tile_boxes,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "high_res_selection": selection,
    }


def _forward_and_cache(loaded, inputs: Mapping[str, Any]):
    model = loaded.model
    with torch.no_grad():
        image_tokens, text_tokens = model.backbone(
            inputs["pixel_values"],
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            tile_values=inputs["tile_values"],
            tile_mask=inputs["tile_mask"],
            tile_boxes=inputs["tile_boxes"],
        )
        full_image_padding = getattr(
            model.backbone,
            "last_image_key_padding_mask",
            None,
        )
        image_padding = (
            full_image_padding[:, 1:]
            if full_image_padding is not None
            else None
        )
        text_padding = inputs["attention_mask"][:, 1:] == 0
        fused_output = model.fusion(
            image_tokens,
            text_tokens,
            img_key_padding_mask=image_padding,
            txt_key_padding_mask=text_padding,
            return_attn=True,
        )
        if not isinstance(fused_output, tuple) or len(fused_output) != 2:
            raise RuntimeError(
                "The proposed fusion module did not return (embedding, details)."
            )
        fused, _ = fused_output
        logits = model.head(fused)

    cached = {
        "global_feature": model.backbone.last_global_feature.detach(),
        "local_tokens": model.backbone.last_local_tokens.detach(),
        "local_mask": model.backbone.last_local_mask.detach(),
        "local_boxes": model.backbone.last_local_token_boxes.detach(),
        "text_tokens": text_tokens.detach(),
        "text_attention_mask": inputs["attention_mask"],
        "text_input_ids": inputs["input_ids"],
        "global_pixel_values": inputs["pixel_values"].detach(),
        "global_foreground_box": tuple(
            inputs["high_res_selection"].foreground_box
        ),
        "text_pad_token_id": resolve_pad_token_id(
            model.backbone.tokenizer_obj
        ),
    }
    tokenizer = getattr(
        model.backbone.tokenizer_obj,
        "tokenizer",
        model.backbone.tokenizer_obj,
    )
    special_ids = {
        int(value)
        for value in (getattr(tokenizer, "all_special_ids", None) or [])
    }
    cached["text_special_token_mask"] = torch.zeros_like(
        inputs["input_ids"],
        dtype=torch.bool,
    )
    for special_id in special_ids:
        cached["text_special_token_mask"] |= (
            inputs["input_ids"] == special_id
        )
    return fused.detach(), logits.detach(), cached


def _spatial_token_subset(model, boxes: np.ndarray, scores: np.ndarray):
    tokens_per_tile = int(model.backbone.local_pool_grid) ** 2 + int(
        model.backbone.include_local_cls_token
    )
    include_cls = bool(model.backbone.include_local_cls_token)
    spatial_indices: list[int] = []
    for start in range(0, len(scores), tokens_per_tile):
        offset = 1 if include_cls else 0
        spatial_indices.extend(
            range(start + offset, min(start + tokens_per_tile, len(scores)))
        )
    indices = np.asarray(spatial_indices, dtype=np.int64)
    if indices.size == 0:
        raise RuntimeError("No spatial local tokens are available for IG rendering.")
    return boxes[indices], scores[indices]


def _decoded_text_attribution(
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    scores: np.ndarray,
) -> tuple[list[str], np.ndarray]:
    """Decode valid WordPiece tokens and merge continuation pieces."""

    inner = getattr(tokenizer, "tokenizer", tokenizer)
    converter = getattr(inner, "convert_ids_to_tokens", None)
    if not callable(converter):
        raise RuntimeError("Tokenizer does not expose convert_ids_to_tokens().")
    special_tokens = set(getattr(inner, "all_special_tokens", []) or [])
    ids = [int(value) for value in input_ids[0].detach().cpu().tolist()]
    valid = [
        bool(value) for value in attention_mask[0].detach().cpu().tolist()
    ]
    raw_tokens = [str(value) for value in converter(ids)]

    words: list[str] = []
    word_scores: list[float] = []
    for token, is_valid, score in zip(raw_tokens, valid, scores):
        if not is_valid or token in special_tokens:
            continue
        cleaned = token.replace("▁", "").strip()
        if not cleaned:
            continue
        if cleaned.startswith("##") and words:
            words[-1] += cleaned[2:]
            word_scores[-1] += float(score)
        else:
            words.append(cleaned.removeprefix("##"))
            word_scores.append(float(score))

    values = np.asarray(word_scores, dtype=np.float32)
    if values.size:
        scale = float(np.max(np.abs(values), initial=0.0))
        if scale > 0:
            values = values / scale
    return words, values


class OnlineInferenceEngine:
    """Checkpoint, OOD references, prediction, and local-token explanation."""

    def __init__(
        self,
        seed: int = 42,
        device: str | torch.device | None = None,
        *,
        experiment_name: str = SOURCE_EXPERIMENT,
        enable_ood: bool = True,
        target_id_fpr: float = 0.05,
        knn_k: int = 5,
    ) -> None:
        self.seed = int(seed)
        if device is None or str(device).lower() == "auto":
            resolved_device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            resolved_device = torch.device(device)
        self.experiment_name = str(experiment_name).strip("/")
        if self.experiment_name == SOURCE_EXPERIMENT:
            self.loaded = load_locked_proposed_model(
                self.seed,
                device=resolved_device,
                strict_fingerprint=True,
            )
        else:
            self.loaded = load_proposed_experiment_model(
                self.experiment_name,
                self.seed,
                device=resolved_device,
            )
        for parameter in self.loaded.model.parameters():
            parameter.requires_grad_(False)
        self.loaded.model.eval()
        self.loaded.model.backbone.explain_mode = True
        self.knn_k = int(knn_k)
        self.ood = (
            fit_locked_ood_calibration(
                self.loaded,
                target_id_fpr=target_id_fpr,
                knn_k=self.knn_k,
            )
            if enable_ood
            else None
        )
        self.class_labels = [
            str(value) for value in self.loaded.cfg.dataset.params.classes
        ]

    @property
    def device(self) -> torch.device:
        return self.loaded.device

    def predict(
        self,
        image: Image.Image,
        clinical_text: str,
        *,
        ood_method: str = "mahalanobis_centroid",
        ig_steps: int = 16,
        compute_faithfulness: bool = False,
        compute_global_ig: bool = False,
    ) -> OnlineInferenceResult:
        clinical_text = str(clinical_text).strip()
        if not clinical_text:
            raise ValueError("Clinical text must not be empty.")
        if ood_method not in OOD_METHODS:
            raise ValueError(f"Unsupported OOD method: {ood_method}")
        if int(ig_steps) < 2:
            raise ValueError("Integrated Gradients requires at least two steps.")

        inputs = _build_online_inputs(
            self.loaded,
            image.convert("RGB"),
            clinical_text,
        )
        fused_raw, logits, cached = _forward_and_cache(self.loaded, inputs)
        probabilities = torch.softmax(logits, dim=-1)[0]
        predicted_index = int(probabilities.argmax().item())

        fused_embedding_raw = fused_raw.cpu().numpy()
        fused_embedding = F.normalize(fused_raw, dim=-1).cpu().numpy()
        logits_array = logits.cpu().numpy()
        if self.ood is None:
            score = float("nan")
            threshold = float("nan")
            is_ood = False
        else:
            score = float(
                _score_method(
                    ood_method,
                    self.ood.normalized_detector,
                    self.ood.raw_detector,
                    fused_embedding=fused_embedding,
                    fused_embedding_raw=fused_embedding_raw,
                    logits=logits_array,
                    knn_k=self.knn_k,
                )[0]
            )
            threshold = float(self.ood.thresholds[ood_method])
            is_ood = bool(score > threshold)

        # Do not expose a closed-set class or a class-conditioned explanation
        # after the sample has been rejected by the locked OOD rule.
        if is_ood:
            return OnlineInferenceResult(
                predicted_index=None,
                predicted_label=None,
                probabilities=None,
                class_labels=self.class_labels,
                ood_method=ood_method,
                ood_score=score,
                ood_threshold=threshold,
                is_ood=True,
                local_boxes=None,
                local_ig_scores=None,
                spatial_ig_boxes=None,
                spatial_ig_scores=None,
                global_ig_map=None,
                text_tokens=None,
                text_ig_scores=None,
                contribution_drops=None,
                global_faithfulness=None,
                visual_faithfulness=None,
                text_faithfulness=None,
                seed=self.seed,
                checkpoint=str(self.loaded.checkpoint),
            )

        self.loaded.model.zero_grad(set_to_none=True)
        ig_scores, _ = integrated_gradients(
            self.loaded.model,
            cached,
            predicted_index,
            steps=int(ig_steps),
        )
        text_ig_relevance, text_ig_attribution = integrated_gradients_text(
            self.loaded.model,
            cached,
            predicted_index,
            steps=int(ig_steps),
        )
        text_ig_scores = text_ig_attribution.sum(dim=-1)
        global_ig_map = None
        global_relevance = None
        if compute_global_ig:
            global_relevance, _ = integrated_gradients_global_image(
                self.loaded.model,
                cached,
                inputs["pixel_values"],
                predicted_index,
                steps=int(ig_steps),
            )
            global_ig_map = project_global_ig_to_source(
                global_relevance.detach().cpu().numpy(),
                cached["global_foreground_box"],
                image.size,
                patch_grid=_global_patch_grid(self.loaded.model),
            )
        global_faithfulness = None
        visual_faithfulness = None
        text_faithfulness = None
        contribution_drops = None
        if compute_faithfulness:
            fractions = np.linspace(0.0, 1.0, 11, dtype=np.float64)
            global_curves = None
            if global_relevance is not None:
                global_curves = global_image_perturbation_curves(
                    self.loaded.model,
                    cached,
                    inputs["pixel_values"],
                    global_relevance,
                    predicted_index,
                    fractions=fractions,
                )
            visual_curves = perturbation_curves(
                self.loaded.model,
                cached,
                ig_scores,
                predicted_index,
                fractions=fractions,
                random_trials=1,
                seed=self.seed,
            )
            text_curves = text_input_perturbation_curves(
                self.loaded.model,
                cached,
                text_ig_relevance,
                predicted_index,
                fractions=fractions,
            )
            if global_curves is not None:
                global_faithfulness = {
                    "fractions": global_curves["fractions"],
                    "deletion": global_curves["delete_most_relevant"],
                    "insertion": global_curves["insert_most_relevant"],
                    "deletion_auc": curve_auc(
                        global_curves["fractions"],
                        global_curves["delete_most_relevant"],
                    ),
                    "insertion_auc": curve_auc(
                        global_curves["fractions"],
                        global_curves["insert_most_relevant"],
                    ),
                }
            visual_faithfulness = {
                "fractions": visual_curves["fractions"],
                "deletion": visual_curves["delete_most_relevant"],
                "insertion": visual_curves["insert_most_relevant"],
                "deletion_auc": curve_auc(
                    visual_curves["fractions"],
                    visual_curves["delete_most_relevant"],
                ),
                "insertion_auc": curve_auc(
                    visual_curves["fractions"],
                    visual_curves["insert_most_relevant"],
                ),
            }
            text_faithfulness = {
                "fractions": text_curves["fractions"],
                "deletion": text_curves["delete_most_relevant"],
                "insertion": text_curves["insert_most_relevant"],
                "deletion_auc": curve_auc(
                    text_curves["fractions"],
                    text_curves["delete_most_relevant"],
                ),
                "insertion_auc": curve_auc(
                    text_curves["fractions"],
                    text_curves["insert_most_relevant"],
                ),
            }
            if global_faithfulness is not None:
                full_probability = float(probabilities[predicted_index])
                contribution_drops = {
                    "global_visual": full_probability
                    - float(global_faithfulness["deletion"][-1]),
                    "local_visual": full_probability
                    - float(visual_faithfulness["deletion"][-1]),
                    "clinical_text": full_probability
                    - float(text_faithfulness["deletion"][-1]),
                }
        local_boxes = cached["local_boxes"][0].detach().cpu().numpy()
        local_scores = ig_scores.detach().cpu().numpy()
        spatial_boxes, spatial_scores = _spatial_token_subset(
            self.loaded.model,
            local_boxes,
            local_scores,
        )
        decoded_tokens, decoded_text_scores = _decoded_text_attribution(
            self.loaded.model.backbone.tokenizer_obj,
            inputs["input_ids"],
            inputs["attention_mask"],
            text_ig_scores.detach().cpu().numpy(),
        )

        return OnlineInferenceResult(
            predicted_index=predicted_index,
            predicted_label=self.class_labels[predicted_index],
            probabilities=probabilities.detach().cpu().numpy(),
            class_labels=self.class_labels,
            ood_method=ood_method,
            ood_score=score,
            ood_threshold=threshold,
            is_ood=False,
            local_boxes=local_boxes,
            local_ig_scores=local_scores,
            spatial_ig_boxes=spatial_boxes,
            spatial_ig_scores=spatial_scores,
            global_ig_map=global_ig_map,
            text_tokens=decoded_tokens,
            text_ig_scores=decoded_text_scores,
            contribution_drops=contribution_drops,
            global_faithfulness=global_faithfulness,
            visual_faithfulness=visual_faithfulness,
            text_faithfulness=text_faithfulness,
            seed=self.seed,
            checkpoint=str(self.loaded.checkpoint),
        )


def _global_patch_grid(model) -> tuple[int, int]:
    trunk = getattr(
        getattr(model.backbone.model, "visual", None),
        "trunk",
        None,
    )
    patch_embed = getattr(trunk, "patch_embed", None)
    grid_size = getattr(patch_embed, "grid_size", (14, 14))
    if isinstance(grid_size, int):
        return int(grid_size), int(grid_size)
    return int(grid_size[0]), int(grid_size[1])


def project_global_ig_to_source(
    heatmap: np.ndarray,
    foreground_box: tuple[int, int, int, int],
    image_size: tuple[int, int],
    *,
    patch_grid: tuple[int, int] | None = None,
) -> np.ndarray:
    """Undo foreground cropping and letterboxing for a global-view heatmap."""

    values = np.maximum(np.asarray(heatmap, dtype=np.float32), 0.0)
    if values.ndim != 2:
        raise ValueError("Global IG heatmap must be two-dimensional.")
    canvas_height, canvas_width = values.shape
    if patch_grid is not None:
        grid_height, grid_width = patch_grid
        values = np.asarray(
            Image.fromarray(values, mode="F")
            .resize(
                (grid_width, grid_height),
                Image.Resampling.BOX,
            )
            .resize(
                (canvas_width, canvas_height),
                Image.Resampling.BICUBIC,
            ),
            dtype=np.float32,
        )
        values = np.maximum(values, 0.0)
    left, top, right, bottom = (int(value) for value in foreground_box)
    foreground_width = max(1, right - left)
    foreground_height = max(1, bottom - top)
    scale = min(
        canvas_width / foreground_width,
        canvas_height / foreground_height,
    )
    resized_width = max(
        1,
        min(canvas_width, round(foreground_width * scale)),
    )
    resized_height = max(
        1,
        min(canvas_height, round(foreground_height * scale)),
    )
    x_offset = (canvas_width - resized_width) // 2
    y_offset = (canvas_height - resized_height) // 2
    content = values[
        y_offset:y_offset + resized_height,
        x_offset:x_offset + resized_width,
    ]
    content_image = Image.fromarray(content, mode="F").resize(
        (foreground_width, foreground_height),
        Image.Resampling.BICUBIC,
    )
    source_width, source_height = image_size
    source = np.zeros((source_height, source_width), dtype=np.float32)
    clipped_left = max(0, left)
    clipped_top = max(0, top)
    clipped_right = min(source_width, right)
    clipped_bottom = min(source_height, bottom)
    restored = np.asarray(content_image, dtype=np.float32)
    source[
        clipped_top:clipped_bottom,
        clipped_left:clipped_right,
    ] = restored[
        clipped_top - top:clipped_bottom - top,
        clipped_left - left:clipped_right - left,
    ]
    positive = source[source > 0]
    if positive.size:
        low, high = np.percentile(positive, [1.0, 99.0])
        source = np.clip(
            (source - float(low)) / max(float(high - low), 1e-8),
            0.0,
            1.0,
        )
    return source


def _render_heatmap_overlay(
    image: Image.Image,
    heatmap: np.ndarray,
    *,
    alpha: float,
) -> Image.Image:
    resized = image.convert("RGB").resize(
        (heatmap.shape[1], heatmap.shape[0]),
        Image.Resampling.LANCZOS,
    )
    base = np.asarray(resized, dtype=np.float32) / 255.0
    color = colormaps["turbo"](heatmap)[..., :3].astype(np.float32)
    opacity = (float(alpha) * np.power(heatmap, 0.7))[..., None]
    overlay = np.clip(base * (1.0 - opacity) + color * opacity, 0.0, 1.0)
    return Image.fromarray(np.uint8(overlay * 255.0), mode="RGB")


def render_global_ig_overlay(
    image: Image.Image,
    result: OnlineInferenceResult,
    *,
    alpha: float = 0.58,
) -> Image.Image:
    """Overlay pixel-level global-view IG in source-image coordinates."""

    if result.global_ig_map is None:
        raise ValueError("Global-image Integrated Gradients is unavailable.")
    return _render_heatmap_overlay(
        image,
        result.global_ig_map,
        alpha=alpha,
    )


def rasterize_local_ig(
    boxes: np.ndarray,
    scores: np.ndarray,
    image_size: tuple[int, int],
    *,
    max_side: int = 720,
) -> np.ndarray:
    """Rasterize local-token attribution in source-image coordinates."""

    width, height = image_size
    scale = min(1.0, float(max_side) / max(width, height))
    out_width = max(1, round(width * scale))
    out_height = max(1, round(height * scale))
    values = np.zeros((out_height, out_width), dtype=np.float32)
    counts = np.zeros_like(values)
    scores = np.maximum(np.asarray(scores, dtype=np.float32), 0.0)
    if float(scores.max(initial=0.0)) > 0:
        scores = scores / float(scores.max())

    for box, score in zip(np.asarray(boxes), scores):
        x1, y1 = np.floor(box[:2] * [out_width, out_height]).astype(int)
        x2, y2 = np.ceil(box[2:] * [out_width, out_height]).astype(int)
        x1, y1 = np.clip([x1, y1], 0, [out_width - 1, out_height - 1])
        x2, y2 = np.clip(
            [x2, y2],
            [x1 + 1, y1 + 1],
            [out_width, out_height],
        )
        values[y1:y2, x1:x2] += float(score)
        counts[y1:y2, x1:x2] += 1.0

    values /= np.maximum(counts, 1.0)
    blur_radius = max(4.0, 0.018 * min(values.shape))
    smoothed = Image.fromarray(
        np.uint8(np.clip(values, 0.0, 1.0) * 255.0),
        mode="L",
    ).filter(ImageFilter.GaussianBlur(radius=blur_radius))
    heatmap = np.asarray(smoothed, dtype=np.float32) / 255.0
    positive = heatmap[heatmap > 0]
    if positive.size:
        low, high = np.percentile(positive, [1.0, 99.0])
        heatmap = np.clip(
            (heatmap - float(low)) / max(float(high - low), 1e-8),
            0.0,
            1.0,
        )
    return heatmap


def render_local_ig_overlay(
    image: Image.Image,
    result: OnlineInferenceResult,
    *,
    alpha: float = 0.58,
) -> Image.Image:
    """Overlay token-level IG on a display-sized copy of the source image."""

    if result.spatial_ig_boxes is None or result.spatial_ig_scores is None:
        raise ValueError("Integrated Gradients is unavailable for an OOD sample.")
    heatmap = rasterize_local_ig(
        result.spatial_ig_boxes,
        result.spatial_ig_scores,
        image.size,
    )
    return _render_heatmap_overlay(image, heatmap, alpha=alpha)
