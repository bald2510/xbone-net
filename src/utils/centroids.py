"""Utilities for estimating non-parametric class centroids."""

from __future__ import annotations

from typing import Optional

import torch
from transformers import TrainerCallback


def _select_phase2_text(batch: dict, use_text: bool, report_type: str):
    if not use_text:
        return None, None
    if report_type in ("both", "xray_clinical"):
        raise NotImplementedError(
            "Empirical centroid estimation supports one Phase-2 report stream at "
            "a time. Use p2_report_type='xray' or 'clinical'."
        )
    if report_type not in ("xray", "clinical"):
        raise ValueError(
            f"Unknown p2_report_type='{report_type}'. Use xray or clinical."
        )
    prefix = "xray" if report_type == "xray" else "clinical"
    return batch[f"{prefix}_input_ids"], batch.get(f"{prefix}_attention_mask")


@torch.no_grad()
def compute_empirical_centroids(
    model,
    data_loader,
    device: torch.device,
    use_text: bool = True,
    report_type: str = "clinical",
) -> torch.Tensor:
    """Compute per-class means from the current train-set embeddings.

    The classifier head is not called, so this function can initialize a newly
    constructed empirical-centroid head. The model is evaluated without dropout
    while estimating means and its previous train/eval mode is restored.
    """
    head = getattr(model, "head", None)
    if head is None or not hasattr(head, "set_centroids"):
        raise TypeError(
            "compute_empirical_centroids requires an EmpiricalCentroidHead."
        )
    if not hasattr(model, "encode_fused"):
        raise TypeError("Model must expose encode_fused() for centroid estimation.")

    num_classes = int(head.num_classes)
    feature_dim = int(head.feature_dim)
    sums = torch.zeros(num_classes, feature_dim, dtype=torch.float32, device=device)
    counts = torch.zeros(num_classes, dtype=torch.long, device=device)

    was_training = model.training
    model.eval()
    try:
        for batch in data_loader:
            if not isinstance(batch, dict):
                raise TypeError(
                    "Centroid estimation expects dictionary-format batches from "
                    "BioMedCLIPDataCollator."
                )

            images = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            labels = labels.argmax(dim=-1) if labels.ndim > 1 else labels
            labels = labels.long().view(-1)

            if labels.numel() == 0:
                continue
            if torch.any(labels < 0) or torch.any(labels >= num_classes):
                raise ValueError(
                    f"Training labels must be in [0, {num_classes - 1}]."
                )

            input_ids, attention_mask = _select_phase2_text(
                batch, use_text=use_text, report_type=report_type
            )
            if input_ids is not None:
                input_ids = input_ids.to(device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            optional_tensors = {}
            for name in ("tile_values", "tile_mask", "tile_boxes"):
                value: Optional[torch.Tensor] = batch.get(name)
                optional_tensors[name] = value.to(device) if value is not None else None

            features = model.encode_fused(
                images,
                input_ids=input_ids,
                attention_mask=attention_mask,
                **optional_tensors,
            )
            if tuple(features.shape) != (labels.numel(), feature_dim):
                raise ValueError(
                    "Unexpected fused feature shape during centroid estimation: "
                    f"{tuple(features.shape)}; expected {(labels.numel(), feature_dim)}."
                )
            sums.index_add_(0, labels, features.detach().float())
            counts.index_add_(0, labels, torch.ones_like(labels, dtype=torch.long))

        missing = torch.nonzero(counts == 0, as_tuple=False).flatten().tolist()
        if missing:
            raise ValueError(
                "Cannot build empirical centroids because the training subset "
                f"contains no samples for classes: {missing}"
            )
        centroids = sums / counts.unsqueeze(1).to(sums.dtype)
        head.set_centroids(centroids, counts)
    finally:
        model.train(was_training)

    return counts.detach().cpu()


class EmpiricalCentroidUpdateCallback(TrainerCallback):
    """Refresh train-set centroids before validation/checkpointing each epoch."""

    def __init__(
        self,
        model,
        data_loader,
        device: torch.device,
        use_text: bool = True,
        report_type: str = "clinical",
        interval_epochs: int = 1,
    ) -> None:
        if interval_epochs < 1:
            raise ValueError("interval_epochs must be >= 1.")
        self.model = model
        self.data_loader = data_loader
        self.device = device
        self.use_text = bool(use_text)
        self.report_type = str(report_type)
        self.interval_epochs = int(interval_epochs)

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = int(round(float(state.epoch or 0.0)))
        if epoch > 0 and epoch % self.interval_epochs == 0:
            counts = compute_empirical_centroids(
                self.model,
                self.data_loader,
                self.device,
                use_text=self.use_text,
                report_type=self.report_type,
            )
            print(
                f"  [Centroids] Refreshed after epoch {epoch}; "
                f"class counts={counts.tolist()}"
            )
        return control
