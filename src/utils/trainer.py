"""
Custom HuggingFace Trainer Adapter for XBone-Net Two-Phase Pipeline.
===============================================================================
Extends transformers.Trainer with phase-aware loss computation for XBone-Net:
  - Phase 1 (Contrastive): Contrastive image-text alignment via dual report inputs
  - Phase 2 (Classification): Prototypical network classification & fusion
  - BioMedCLIPDataCollator: Handles dynamic padding for dual-report text fields

Supports xray, clinical, or dual-report text embedding configurations.
"""

from dataclasses import dataclass
import torch
import torch.nn as nn
from transformers import Trainer


# ============================================================
# Tokenizer Padding Utilities
# ============================================================

def resolve_pad_token_id(tokenizer) -> int:
    """Resolve the padding token ID from an OpenCLIP or HuggingFace tokenizer.

    Unwraps OpenCLIP tokenizers if necessary. Uses pad_token_id if available,
    falling back to eos_token_id or default value 0.

    Args:
        tokenizer: HuggingFace PreTrainedTokenizer or OpenCLIP wrapper.

    Returns:
        Integer token ID to use for sequence padding.
    """
    # --- Unwrap OpenCLIP's tokenizer wrapper if present ---
    hf_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    pad_token_id = getattr(hf_tokenizer, "pad_token_id", None)
    if pad_token_id is not None:
        return pad_token_id
    # --- Fallback: use EOS token for GPT-style tokenizers ---
    return getattr(hf_tokenizer, "eos_token_id", 0) or 0


# ============================================================
# Dual-Report Data Collator
# ============================================================

@dataclass
class BioMedCLIPDataCollator:
    """Data collator for XBone-Net datasets with dual-report text fields.

    Dynamically right-pads xray_input_ids and clinical_input_ids to batch maximum length.

    Attributes:
        pad_token_id (int): Token ID used for padding text sequences.

    Example:
        collator = BioMedCLIPDataCollator(pad_token_id=0)
        batch = collator([sample1, sample2])
    """

    pad_token_id: int = 0

    def _pad_ids(self, ids_list):
        """Right-pad a list of 1-D token-ID tensors to uniform length.

        Args:
            ids_list: List of 1-D torch.Tensor token-ID sequences.

        Returns:
            Batched torch.Tensor of shape (B, max_len).
        """
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

    def __call__(self, features: list) -> dict:
        """Collate tuple or dictionary features into batched tensors.

        Args:
            features: List of sample tuples or dictionary features.

        Returns:
            Dictionary containing pixel_values, xray_input_ids,
            clinical_input_ids, and labels.
        """
        if not features:
            return {}

        first = features[0]
        if isinstance(first, tuple):
            if len(first) == 4:
                # Dual-report sample tuple: (image, xray_ids, clinical_ids, labels)
                images = torch.stack([f[0] if isinstance(f[0], torch.Tensor) else F_t.to_tensor(f[0]) for f in features])
                xray_ids = self._pad_ids([f[1] for f in features])
                clinical_ids = self._pad_ids([f[2] for f in features])
                labels = torch.stack([f[3] for f in features])
                return {
                    "pixel_values": images,
                    "xray_input_ids": xray_ids,
                    "clinical_input_ids": clinical_ids,
                    "labels": labels,
                }
            elif len(first) == 3:
                # Single-report sample tuple: (image, input_ids, labels)
                images = torch.stack([f[0] for f in features])
                input_ids = self._pad_ids([f[1] for f in features])
                labels = torch.stack([f[2] for f in features])
                return {
                    "pixel_values": images,
                    "xray_input_ids": input_ids,
                    "clinical_input_ids": input_ids,
                    "labels": labels,
                }

        # Dictionary format fallback
        pixel_values = torch.stack([f["pixel_values"] for f in features])
        labels = torch.stack([f["labels"] for f in features])
        xray_ids = self._pad_ids([f.get("xray_input_ids", f.get("input_ids")) for f in features])
        clinical_ids = self._pad_ids([f.get("clinical_input_ids", f.get("input_ids")) for f in features])

        return {
            "pixel_values": pixel_values,
            "xray_input_ids": xray_ids,
            "clinical_input_ids": clinical_ids,
            "labels": labels,
        }


# ============================================================
# Phase-Aware HuggingFace Trainer Adapter
# ============================================================

class SFTrainer(Trainer):
    """Phase-aware HuggingFace Trainer for XBone-Net two-stage training.

    Overrides compute_loss and prediction_step to handle Phase 1 (contrastive)
    and Phase 2 (classification) execution modes seamlessly.

    Attributes:
        phase (str): Training phase ('phase1' or 'phase2').
        loss_fn (nn.Module): Active loss function instance.
        use_text_in_p2 (bool): Whether text features are fed into Phase 2 head.
        p1_report_type (str): Report type for Phase 1 ('xray', 'clinical', 'both').
        p2_report_type (str): Report type for Phase 2 ('xray', 'clinical', 'both').
    """

    def __init__(self, phase, loss_fn, use_text_in_p2=True,
                 p1_report_type="xray", p2_report_type="clinical",
                 *args, **kwargs):
        """Initialize the SFTrainer adapter.

        Args:
            phase: 'phase1' or 'phase2'.
            loss_fn: Loss module instance.
            use_text_in_p2: Include text input in Phase 2.
            p1_report_type: Phase 1 report selection ('xray', 'clinical', or 'both').
            p2_report_type: Phase 2 report selection ('xray', 'clinical', or 'both').
            *args: Positional arguments forwarded to Trainer.
            **kwargs: Keyword arguments forwarded to Trainer.
        """
        super().__init__(*args, **kwargs)
        self.phase = phase
        self.loss_fn = loss_fn
        self.use_text_in_p2 = use_text_in_p2
        self.p1_report_type = p1_report_type
        self.p2_report_type = p2_report_type

    def _get_text_features(self, model, images, inputs, report_type):
        """Extract text features for the specified report configuration.

        Args:
            model: XBone-Net model instance.
            images: Tensor of pixel values (B, C, H, W).
            inputs: Batch dictionary containing text fields.
            report_type: 'xray', 'clinical', or 'both'.

        Returns:
            Text features tensor or token IDs tensor.
        """
        if report_type in ("both", "xray_clinical"):
            xray_ids = inputs["xray_input_ids"]
            clinical_ids = inputs["clinical_input_ids"]
            _, xray_feat = model.backbone(images, xray_ids)
            _, clinical_feat = model.backbone(images, clinical_ids)
            return (xray_feat + clinical_feat) / 2.0
        elif report_type == "xray":
            return inputs["xray_input_ids"]
        else:
            return inputs["clinical_input_ids"]

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Compute training loss for Phase 1 or Phase 2.

        Args:
            model: The XBoneNet model instance.
            inputs: Batch dictionary with pixel_values, xray_input_ids, etc.
            return_outputs: If True, return (loss, outputs) tuple.
            **kwargs: Unused extra arguments.

        Returns:
            Scalar loss tensor or (loss, outputs) tuple.
        """
        images = inputs["pixel_values"]
        labels = inputs["labels"]

        if self.phase == "phase1":
            # --- Phase 1: Contrastive image-text alignment ---
            if self.p1_report_type in ("both", "xray_clinical"):
                xray_ids = inputs["xray_input_ids"]
                clinical_ids = inputs["clinical_input_ids"]
                image_features, xray_feat = model.backbone(images, xray_ids)
                _, clinical_feat = model.backbone(images, clinical_ids)
                text_features = (xray_feat + clinical_feat) / 2.0
            else:
                text_key = "xray_input_ids" if self.p1_report_type == "xray" else "clinical_input_ids"
                text_ids = inputs[text_key]
                image_features, text_features = model.backbone(images, text_ids)
            loss = self.loss_fn(image_features, text_features, None)
            outputs = {"image_features": image_features, "text_features": text_features}
        else:
            # --- Phase 2: Classification with optional fusion ---
            if self.use_text_in_p2 and self.p2_report_type in ("both", "xray_clinical"):
                xray_ids = inputs["xray_input_ids"]
                clinical_ids = inputs["clinical_input_ids"]
                img_feat, xray_feat = model.backbone(images, xray_ids)
                _, clinical_feat = model.backbone(images, clinical_ids)
                text_feat = (xray_feat + clinical_feat) / 2.0
                if model.fusion is not None:
                    fused = model.fusion(img_feat, text_feat)
                else:
                    fused = img_feat
                text_ids = None
            elif self.use_text_in_p2:
                text_key = "xray_input_ids" if self.p2_report_type == "xray" else "clinical_input_ids"
                text_ids = inputs[text_key]
            else:
                text_ids = inputs["xray_input_ids"]

            from src.utils.losses import CombinedPhase2Loss, CombinedPhase2LossMulticlass
            if self.use_text_in_p2 and self.p2_report_type in ("both", "xray_clinical"):
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
        """Run a single evaluation or prediction step.

        Args:
            model: Model being evaluated.
            inputs: Batch dictionary.
            prediction_loss_only: If True, only loss is returned.
            ignore_keys: Optional list of keys to ignore.

        Returns:
            Tuple of (loss, logits, labels).
        """
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)

        logits = None
        if not prediction_loss_only and self.phase == "phase2":
            logits = outputs[0] if isinstance(outputs, tuple) else outputs

        labels = inputs.get("labels")
        return (loss, logits, labels)


