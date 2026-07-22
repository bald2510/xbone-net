"""Quantitative explainability and representation diagnostics for XBone-Net."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    balanced_accuracy_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.neighbors import KNeighborsClassifier


COMPONENT_NAMES = (
    "image_from_text_context",
    "text_from_image_context",
)


def fusion_from_tokens(
    model,
    image_tokens: torch.Tensor,
    text_tokens: torch.Tensor,
    text_attention_mask: torch.Tensor,
    component_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the exact trained fusion while exposing its two branch vectors."""
    fusion = model.fusion
    image_features = fusion.img_proj(fusion.img_input_norm(image_tokens))
    text_features = fusion.txt_proj(fusion.txt_input_norm(text_tokens))
    image_global = fusion.norm_img_global(image_features[:, :1])
    text_global = fusion.norm_txt_global(text_features[:, :1])
    image_local = image_features[:, 1:]
    text_local = text_features[:, 1:]
    text_padding = text_attention_mask[:, 1:] == 0
    image_padding = getattr(model.backbone, "last_image_key_padding_mask", None)
    image_padding = image_padding[:, 1:] if image_padding is not None else None

    direction = getattr(fusion, "attention_direction", "bidirectional")
    image_enhanced = fusion.norm_img(image_global)
    if direction in {"bidirectional", "image_to_text"}:
        context, _ = fusion.img_to_txt_attn(
            query=image_global,
            key=text_local,
            value=text_local,
            key_padding_mask=text_padding,
            need_weights=False,
        )
        image_enhanced = fusion.norm_img(
            image_global + fusion.dropout(context)
        )

    text_enhanced = fusion.norm_txt(text_global)
    if direction in {"bidirectional", "text_to_image"}:
        context, _ = fusion.txt_to_img_attn(
            query=text_global,
            key=image_local,
            value=image_local,
            key_padding_mask=image_padding,
            need_weights=False,
        )
        text_enhanced = fusion.norm_txt(text_global + fusion.dropout(context))

    components = torch.stack(
        [image_enhanced[:, 0], text_enhanced[:, 0]], dim=1
    )
    if component_mask is not None:
        components = components * component_mask.view(
            1, len(COMPONENT_NAMES), 1
        )
    cross_fused = fusion.norm_fuse(fusion.fusion_mlp(components.flatten(1)))
    if hasattr(fusion, "concat_projection") and hasattr(fusion, "gate_network"):
        image_vector = fusion._pool_image_tokens(image_tokens, image_padding)
        text_vector = text_tokens[:, 0]
        concat_fused = fusion.concat_projection(
            torch.cat([image_vector, text_vector], dim=-1)
        )
        gate = torch.sigmoid(
            fusion.gate_network(
                torch.cat([concat_fused, cross_fused], dim=-1)
            )
        )
        fused = concat_fused + gate * (cross_fused - concat_fused)
    else:
        fused = cross_fused
    return model.head(fused), components


def forward_from_local(
    model,
    cached: dict[str, torch.Tensor],
    local_tokens: Optional[torch.Tensor] = None,
    text_tokens: Optional[torch.Tensor] = None,
    component_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    local = cached["local_tokens"] if local_tokens is None else local_tokens
    text = cached["text_tokens"] if text_tokens is None else text_tokens
    image_tokens = model.backbone.visual_resampler(
        cached["global_feature"],
        local,
        cached["local_mask"],
        cached["local_boxes"],
    )
    logits, components = fusion_from_tokens(
        model,
        image_tokens,
        text,
        cached["text_attention_mask"],
        component_mask=component_mask,
    )
    return logits, components, image_tokens


def branch_ablation(
    model,
    cached: dict[str, torch.Tensor],
    target_class: int,
) -> dict[str, Any]:
    with torch.no_grad():
        full_logits, _, _ = forward_from_local(model, cached)
        full_probability = torch.softmax(full_logits, dim=-1)[0, target_class]
        full_logit = full_logits[0, target_class]
        logit_drops, probability_drops = [], []
        for index in range(len(COMPONENT_NAMES)):
            mask = torch.ones(len(COMPONENT_NAMES), device=full_logits.device)
            mask[index] = 0
            logits, _, _ = forward_from_local(
                model, cached, component_mask=mask
            )
            probability = torch.softmax(logits, dim=-1)[0, target_class]
            logit_drops.append(float(full_logit - logits[0, target_class]))
            probability_drops.append(float(full_probability - probability))
    return {
        "logits": full_logits,
        "target_logit_drop": dict(zip(COMPONENT_NAMES, logit_drops)),
        "target_probability_drop": dict(
            zip(COMPONENT_NAMES, probability_drops)
        ),
    }


def integrated_gradients(
    model,
    cached: dict[str, torch.Tensor],
    target_class: int,
    steps: int = 24,
    baseline: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if steps < 2:
        raise ValueError("Integrated gradients requires at least two steps.")
    local = cached["local_tokens"].detach()
    baseline = (
        cached["global_feature"].unsqueeze(1).expand_as(local).detach()
        if baseline is None
        else baseline.detach()
    )
    if baseline.shape != local.shape:
        raise ValueError("Integrated-gradients baseline shape differs from local tokens.")
    gradient_sum = torch.zeros_like(local)
    for index, alpha in enumerate(
        torch.linspace(0.0, 1.0, steps, device=local.device)
    ):
        interpolated = (
            baseline + alpha * (local - baseline)
        ).detach().requires_grad_(True)
        logits, _, _ = forward_from_local(
            model, cached, local_tokens=interpolated
        )
        gradient = torch.autograd.grad(
            logits[0, target_class], interpolated, retain_graph=False
        )[0]
        gradient_sum += gradient * (
            0.5 if index in (0, steps - 1) else 1.0
        )
    attribution = (local - baseline) * gradient_sum / (steps - 1)
    signed = attribution.sum(dim=-1)
    relevance = torch.relu(signed)
    if float(relevance.detach().sum()) <= 1e-10:
        relevance = attribution.abs().sum(dim=-1)
    return relevance[0].detach(), attribution[0].detach()


def gradient_visual_relevance(
    model,
    cached: dict[str, torch.Tensor],
    target_class: int,
) -> torch.Tensor:
    """Gradient×input relevance for each passthrough visual token."""
    local = cached["local_tokens"].detach().requires_grad_(True)
    logits, _, _ = forward_from_local(model, cached, local_tokens=local)
    gradient = torch.autograd.grad(
        logits[0, target_class], local, retain_graph=False
    )[0]
    relevance = torch.relu((local * gradient).sum(dim=-1))
    if float(relevance.detach().sum()) <= 1e-10:
        relevance = (local * gradient).abs().sum(dim=-1)
    return relevance[0].detach()


def gradient_text_relevance(
    model,
    cached: dict[str, torch.Tensor],
    target_class: int,
) -> torch.Tensor:
    """Gradient×input relevance for non-CLS clinical-report tokens."""
    text = cached["text_tokens"].detach().requires_grad_(True)
    local_cache = {**cached, "text_tokens": text}
    logits, _, _ = forward_from_local(model, local_cache)
    gradient = torch.autograd.grad(
        logits[0, target_class], text, retain_graph=False
    )[0]
    relevance = torch.relu((text * gradient).sum(dim=-1))
    if float(relevance.detach().sum()) <= 1e-10:
        relevance = (text * gradient).abs().sum(dim=-1)
    mask = cached["text_attention_mask"].to(relevance.dtype)
    relevance = relevance * mask
    return relevance[0].detach()


def _target_outputs(
    model,
    cached: dict[str, torch.Tensor],
    local_tokens: Optional[torch.Tensor],
    target_class: int,
    text_tokens: Optional[torch.Tensor] = None,
) -> tuple[float, float, float]:
    with torch.no_grad():
        logits, _, _ = forward_from_local(
            model,
            cached,
            local_tokens=local_tokens,
            text_tokens=text_tokens,
        )
        probabilities = torch.softmax(logits, dim=-1)[0]
        target_logit = logits[0, target_class]
        alternatives = logits[0].clone()
        alternatives[target_class] = -torch.inf
        margin = target_logit - alternatives.max()
        return (
            float(probabilities[target_class]),
            float(target_logit),
            float(margin),
        )


def perturbation_curves(
    model,
    cached: dict[str, torch.Tensor],
    relevance: torch.Tensor,
    target_class: int,
    fractions: Optional[np.ndarray] = None,
    random_trials: int = 16,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Deletion/insertion curves against random and least-relevant controls."""
    if random_trials < 1:
        raise ValueError("random_trials must be positive.")
    fractions = (
        np.asarray([0.0, 0.125, 0.25, 0.5, 0.75, 1.0])
        if fractions is None
        else np.asarray(fractions, dtype=np.float64)
    )
    local = cached["local_tokens"].detach()
    # Replacing a local token with the pretrained global image embedding removes
    # region-specific evidence while avoiding the singular zero vector before
    # the resampler's L2 normalization.
    baseline = cached["global_feature"].unsqueeze(1).expand_as(local).detach()
    token_count = local.size(1)
    descending = torch.argsort(relevance, descending=True)
    ascending = descending.flip(0)
    generator = torch.Generator(device=local.device).manual_seed(int(seed))
    top, low, random_values, insertion = [], [], [], []
    top_logits, low_logits, random_logits, insertion_logits = [], [], [], []
    top_margins, low_margins, random_margins, insertion_margins = [], [], [], []
    for fraction in fractions:
        count = min(token_count, int(round(float(fraction) * token_count)))
        top_tokens = local.clone()
        low_tokens = local.clone()
        inserted = baseline.clone()
        if count:
            top_tokens[:, descending[:count]] = 0
            low_tokens[:, ascending[:count]] = 0
            inserted[:, descending[:count]] = local[:, descending[:count]]
        top_output = _target_outputs(model, cached, top_tokens, target_class)
        low_output = _target_outputs(model, cached, low_tokens, target_class)
        insertion_output = _target_outputs(model, cached, inserted, target_class)
        top.append(top_output[0])
        top_logits.append(top_output[1])
        top_margins.append(top_output[2])
        low.append(low_output[0])
        low_logits.append(low_output[1])
        low_margins.append(low_output[2])
        insertion.append(insertion_output[0])
        insertion_logits.append(insertion_output[1])
        insertion_margins.append(insertion_output[2])

        trials = []
        for _ in range(random_trials):
            random_tokens = local.clone()
            if count:
                indices = torch.randperm(
                    token_count, generator=generator, device=local.device
                )[:count]
                random_tokens[:, indices] = 0
            trials.append(
                _target_outputs(model, cached, random_tokens, target_class)
            )
        trial_array = np.asarray(trials, dtype=np.float64)
        random_values.append(float(trial_array[:, 0].mean()))
        random_logits.append(float(trial_array[:, 1].mean()))
        random_margins.append(float(trial_array[:, 2].mean()))
    return {
        "fractions": fractions,
        "delete_most_relevant": np.asarray(top),
        "delete_random": np.asarray(random_values),
        "delete_least_relevant": np.asarray(low),
        "insert_most_relevant": np.asarray(insertion),
        "delete_most_relevant_logit": np.asarray(top_logits),
        "delete_random_logit": np.asarray(random_logits),
        "delete_least_relevant_logit": np.asarray(low_logits),
        "insert_most_relevant_logit": np.asarray(insertion_logits),
        "delete_most_relevant_margin": np.asarray(top_margins),
        "delete_random_margin": np.asarray(random_margins),
        "delete_least_relevant_margin": np.asarray(low_margins),
        "insert_most_relevant_margin": np.asarray(insertion_margins),
    }


def text_perturbation_curves(
    model,
    cached: dict[str, torch.Tensor],
    relevance: torch.Tensor,
    target_class: int,
    fractions: Optional[np.ndarray] = None,
    random_trials: int = 16,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Delete/insert clinical tokens using the report CLS embedding as baseline."""
    if random_trials < 1:
        raise ValueError("random_trials must be positive.")
    text = cached["text_tokens"].detach()
    valid = cached["text_attention_mask"][0].to(dtype=torch.bool)
    candidate_indices = torch.nonzero(valid, as_tuple=False).flatten()
    candidate_indices = candidate_indices[candidate_indices != 0]
    if candidate_indices.numel() < 1:
        raise ValueError("Clinical report contains no non-CLS valid tokens.")
    fractions = (
        np.asarray([0.0, 0.125, 0.25, 0.5, 0.75, 1.0])
        if fractions is None
        else np.asarray(fractions, dtype=np.float64)
    )
    candidate_relevance = relevance[candidate_indices]
    descending = candidate_indices[torch.argsort(candidate_relevance, descending=True)]
    ascending = descending.flip(0)
    baseline = text.clone()
    baseline[:, candidate_indices] = text[:, :1].expand(
        -1, candidate_indices.numel(), -1
    )
    generator = torch.Generator(device=text.device).manual_seed(int(seed))
    output = {
        "delete_most_relevant": [],
        "delete_random": [],
        "delete_least_relevant": [],
        "insert_most_relevant": [],
        "delete_most_relevant_logit": [],
        "delete_random_logit": [],
        "delete_least_relevant_logit": [],
        "insert_most_relevant_logit": [],
        "delete_most_relevant_margin": [],
        "delete_random_margin": [],
        "delete_least_relevant_margin": [],
        "insert_most_relevant_margin": [],
    }
    token_count = int(candidate_indices.numel())
    for fraction in fractions:
        count = min(token_count, int(round(float(fraction) * token_count)))
        top_text = text.clone()
        low_text = text.clone()
        inserted_text = baseline.clone()
        if count:
            top_text[:, descending[:count]] = baseline[:, descending[:count]]
            low_text[:, ascending[:count]] = baseline[:, ascending[:count]]
            inserted_text[:, descending[:count]] = text[:, descending[:count]]
        top = _target_outputs(
            model, cached, None, target_class, text_tokens=top_text
        )
        low = _target_outputs(
            model, cached, None, target_class, text_tokens=low_text
        )
        inserted = _target_outputs(
            model, cached, None, target_class, text_tokens=inserted_text
        )
        random_outputs = []
        for _ in range(random_trials):
            random_text = text.clone()
            if count:
                order = torch.randperm(
                    token_count, generator=generator, device=text.device
                )[:count]
                indices = candidate_indices[order]
                random_text[:, indices] = baseline[:, indices]
            random_outputs.append(
                _target_outputs(
                    model, cached, None, target_class, text_tokens=random_text
                )
            )
        random_mean = np.asarray(random_outputs, dtype=np.float64).mean(axis=0)
        for suffix, position in (("", 0), ("_logit", 1), ("_margin", 2)):
            output[f"delete_most_relevant{suffix}"].append(top[position])
            output[f"delete_random{suffix}"].append(float(random_mean[position]))
            output[f"delete_least_relevant{suffix}"].append(low[position])
            output[f"insert_most_relevant{suffix}"].append(inserted[position])
    return {
        "fractions": fractions,
        **{key: np.asarray(values) for key, values in output.items()},
    }


def curve_auc(fractions: np.ndarray, values: np.ndarray) -> float:
    fractions = np.asarray(fractions, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if fractions.shape != values.shape or fractions.size < 2:
        raise ValueError("AUC inputs must be aligned one-dimensional curves.")
    width = fractions[-1] - fractions[0]
    if width <= 0:
        raise ValueError("Curve fractions must span a positive interval.")
    integrate = getattr(np, "trapezoid", None)
    if integrate is None:  # NumPy < 2.0
        integrate = np.trapz
    return float(integrate(values, fractions) / width)


def rank_correlation(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first).reshape(-1)
    second = np.asarray(second).reshape(-1)
    if first.shape != second.shape or first.size < 2:
        return float("nan")
    if np.allclose(first, first[0]) or np.allclose(second, second[0]):
        return 0.0
    value = spearmanr(first, second).statistic
    return float(value) if np.isfinite(value) else 0.0


def stratified_sample_indices(
    labels: np.ndarray,
    per_class: int,
    seed: int,
    max_samples: Optional[int] = None,
) -> np.ndarray:
    labels = np.asarray(labels).reshape(-1)
    if per_class < 1:
        raise ValueError("per_class must be positive.")
    rng = np.random.default_rng(seed)
    selected = []
    for class_id in np.unique(labels):
        indices = np.flatnonzero(labels == class_id)
        selected.extend(
            rng.choice(indices, size=min(per_class, len(indices)), replace=False).tolist()
        )
    selected = np.asarray(selected, dtype=np.int64)
    rng.shuffle(selected)
    if max_samples is not None:
        selected = selected[: int(max_samples)]
    return selected


def representation_quality_metrics(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    test_embeddings: np.ndarray,
    test_labels: np.ndarray,
    seed: int,
    knn_k: int = 5,
) -> dict[str, float]:
    """Quantify class geometry without refitting the proposed classifier."""
    train_embeddings = np.asarray(train_embeddings, dtype=np.float64)
    test_embeddings = np.asarray(test_embeddings, dtype=np.float64)
    train_embeddings /= np.maximum(
        np.linalg.norm(train_embeddings, axis=1, keepdims=True), 1e-12
    )
    test_embeddings /= np.maximum(
        np.linalg.norm(test_embeddings, axis=1, keepdims=True), 1e-12
    )
    train_labels = np.asarray(train_labels, dtype=np.int64).reshape(-1)
    test_labels = np.asarray(test_labels, dtype=np.int64).reshape(-1)
    classes = np.unique(train_labels)
    if len(classes) < 2:
        raise ValueError("Representation metrics require at least two classes.")

    centroids = np.stack(
        [train_embeddings[train_labels == class_id].mean(axis=0) for class_id in classes]
    )
    centroids /= np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-12)
    similarities = test_embeddings @ centroids.T
    nearest = classes[similarities.argmax(axis=1)]
    class_to_position = {class_id: index for index, class_id in enumerate(classes)}
    true_positions = np.asarray([class_to_position[value] for value in test_labels])
    true_similarity = similarities[np.arange(len(test_labels)), true_positions]
    other = similarities.copy()
    other[np.arange(len(test_labels)), true_positions] = -np.inf
    margins = true_similarity - other.max(axis=1)

    knn = KNeighborsClassifier(n_neighbors=min(knn_k, len(train_embeddings)))
    knn.fit(train_embeddings, train_labels)
    knn_predictions = knn.predict(test_embeddings)
    clusters = KMeans(
        n_clusters=len(classes), random_state=seed, n_init=20
    ).fit_predict(test_embeddings)

    return {
        "silhouette_cosine": float(
            silhouette_score(test_embeddings, test_labels, metric="cosine")
        ),
        "davies_bouldin": float(davies_bouldin_score(test_embeddings, test_labels)),
        "calinski_harabasz": float(
            calinski_harabasz_score(test_embeddings, test_labels)
        ),
        "kmeans_nmi": float(normalized_mutual_info_score(test_labels, clusters)),
        "kmeans_ari": float(adjusted_rand_score(test_labels, clusters)),
        "nearest_train_centroid_balanced_accuracy": float(
            balanced_accuracy_score(test_labels, nearest)
        ),
        "knn_balanced_accuracy": float(
            balanced_accuracy_score(test_labels, knn_predictions)
        ),
        "true_centroid_margin_mean": float(margins.mean()),
        "true_centroid_margin_median": float(np.median(margins)),
        "positive_centroid_margin_fraction": float((margins > 0).mean()),
    }
