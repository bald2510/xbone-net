"""
SFTrainer: Semantic Fine-tuning trainer for two-stage training.

Phase 1: Contrastive learning with X-ray reports (image-text alignment).
Phase 2: Classification with clinical reports (patient context fusion).
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import Trainer


def resolve_pad_token_id(tokenizer) -> int:
    """Resolve pad token id from an OpenCLIP HFTokenizer wrapper or HF tokenizer."""
    hf_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    pad_token_id = getattr(hf_tokenizer, "pad_token_id", None)
    if pad_token_id is not None:
        return pad_token_id
    return getattr(hf_tokenizer, "eos_token_id", 0) or 0


@dataclass
class BioMedCLIPDataCollator:
    """Collate samples with dual text fields for two-stage training."""

    pad_token_id: int = 0

    def _pad_ids(self, ids_list):
        """Pad a list of token tensors to the same length."""
        if all(ids.shape == ids_list[0].shape for ids in ids_list):
            return torch.stack(ids_list)
        max_len = max(ids.shape[0] for ids in ids_list)
        padded = []
        for ids in ids_list:
            pad_len = max_len - ids.shape[0]
            if pad_len > 0:
                padding = torch.full((pad_len,), self.pad_token_id, dtype=ids.dtype)
                ids = torch.cat([ids, padding])
            padded.append(ids)
        return torch.stack(padded)

    def __call__(self, features: list[dict]) -> dict:
        pixel_values = torch.stack([f["pixel_values"] for f in features])
        labels = torch.stack([f["labels"] for f in features])

        xray_ids = self._pad_ids([f["xray_input_ids"] for f in features])
        clinical_ids = self._pad_ids([f["clinical_input_ids"] for f in features])

        return {
            "pixel_values": pixel_values,
            "xray_input_ids": xray_ids,
            "clinical_input_ids": clinical_ids,
            "labels": labels,
        }


class SFTrainer(Trainer):
    """
    Custom trainer for two-stage training.

    Phase 1: Contrastive learning — text source controlled by p1_report_type
    Phase 2: Classification — text source controlled by p2_report_type

    Supported report types: "xray", "clinical", "both"
    """

    def __init__(self, phase, loss_fn, use_text_in_p2=True,
                 p1_report_type="xray", p2_report_type="clinical",
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.phase = phase
        self.loss_fn = loss_fn
        self.use_text_in_p2 = use_text_in_p2
        self.p1_report_type = p1_report_type   # "xray", "clinical", or "both"
        self.p2_report_type = p2_report_type    # "xray", "clinical", or "both"

    def _get_text_features(self, model, images, inputs, report_type):
        """Encode text based on report_type. Returns text_ids or averaged features."""
        if report_type == "both":
            xray_ids = inputs["xray_input_ids"]
            clinical_ids = inputs["clinical_input_ids"]
            _, xray_feat = model.backbone(images, xray_ids)
            _, clinical_feat = model.backbone(images, clinical_ids)
            return (xray_feat + clinical_feat) / 2.0
        elif report_type == "xray":
            return inputs["xray_input_ids"]
        else:  # "clinical"
            return inputs["clinical_input_ids"]

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        images = inputs["pixel_values"]
        labels = inputs["labels"]

        if self.phase == "phase1":
            # Phase 1: contrastive learning
            if self.p1_report_type == "both":
                # Encode both, average text features
                xray_ids = inputs["xray_input_ids"]
                clinical_ids = inputs["clinical_input_ids"]
                image_features, xray_feat = model.backbone(images, xray_ids)
                _, clinical_feat = model.backbone(images, clinical_ids)
                text_features = (xray_feat + clinical_feat) / 2.0
            else:
                text_key = "xray_input_ids" if self.p1_report_type == "xray" else "clinical_input_ids"
                text_ids = inputs[text_key]
                image_features, text_features = model.backbone(images, text_ids)
            loss = self.loss_fn(image_features, text_features, labels)
            outputs = {"image_features": image_features, "text_features": text_features}
        else:
            # Phase 2: classification
            if self.use_text_in_p2 and self.p2_report_type == "both":
                # Encode both texts, average features, then forward through fusion+head
                xray_ids = inputs["xray_input_ids"]
                clinical_ids = inputs["clinical_input_ids"]
                img_feat, xray_feat = model.backbone(images, xray_ids)
                _, clinical_feat = model.backbone(images, clinical_ids)
                text_feat = (xray_feat + clinical_feat) / 2.0
                # Manually run fusion + head
                if model.fusion is not None:
                    fused = model.fusion(img_feat, text_feat)
                else:
                    fused = img_feat
                text_ids = None  # signal to skip normal forward
            elif self.use_text_in_p2:
                text_key = "xray_input_ids" if self.p2_report_type == "xray" else "clinical_input_ids"
                text_ids = inputs[text_key]
            else:
                text_ids = inputs["xray_input_ids"]

            # Check if loss function needs features/prototypes (CombinedPhase2Loss)
            from utils.losses import CombinedPhase2Loss, CombinedPhase2LossMulticlass
            if self.use_text_in_p2 and self.p2_report_type == "both":
                # "both" mode: fusion already done above
                if isinstance(self.loss_fn, (CombinedPhase2Loss, CombinedPhase2LossMulticlass)):
                    logits, features = model.head(fused, return_features=True)
                    prototypes = model.head.prototypes
                    loss = self.loss_fn(logits, labels, features=fused, prototypes=prototypes)
                else:
                    logits = model.head(fused)
                    loss = self.loss_fn(logits, labels)
                outputs = logits
            elif isinstance(self.loss_fn, (CombinedPhase2Loss, CombinedPhase2LossMulticlass)):
                outputs = model(images, text_ids, return_features=True)
                if isinstance(outputs, tuple) and len(outputs) == 3:
                    logits, features, prototypes = outputs
                    loss = self.loss_fn(logits, labels, features=features, prototypes=prototypes)
                else:
                    logits = outputs[0] if isinstance(outputs, tuple) else outputs
                    loss = self.loss_fn(logits, labels)
            else:
                outputs = model(images, text_ids)
                logits = outputs[0] if isinstance(outputs, tuple) else outputs
                loss = self.loss_fn(logits, labels)

        return (loss, outputs) if return_outputs else loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: dict,
        prediction_loss_only: bool,
        ignore_keys: list | None = None,
    ):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)

        logits = None
        if not prediction_loss_only and self.phase == "phase2":
            logits = outputs[0] if isinstance(outputs, tuple) else outputs

        labels = inputs.get("labels")
        return (loss, logits, labels)
