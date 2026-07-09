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
import torchvision.transforms.functional as F_t


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

    def _pad_ids_with_mask(self, ids_list):
        """Right-pad a list of 1-D token-ID tensors and generate attention masks.

        Args:
            ids_list: List of 1-D torch.Tensor token-ID sequences.

        Returns:
            Tuple of (padded_ids, attention_masks) of shape (B, max_len).
            attention_masks uses 1 for real tokens and 0 for padding.
        """
        import torch
        if all(ids.shape == ids_list[0].shape for ids in ids_list):
            padded = torch.stack(ids_list)
            return padded, torch.ones_like(padded, dtype=torch.long)
            
        max_len = max(ids.shape[0] for ids in ids_list)
        padded = []
        masks = []
        for ids in ids_list:
            pad_len = max_len - ids.shape[0]
            mask = torch.ones_like(ids, dtype=torch.long)
            if pad_len > 0:
                padding = torch.full((pad_len,), self.pad_token_id, dtype=ids.dtype)
                pad_mask = torch.zeros((pad_len,), dtype=torch.long)
                ids = torch.cat([ids, padding])
                mask = torch.cat([mask, pad_mask])
            padded.append(ids)
            masks.append(mask)
        return torch.stack(padded), torch.stack(masks)

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
                images = torch.stack([f[0] if isinstance(f[0], torch.Tensor) else F_t.to_tensor(f[0]) for f in features])
                xray_ids, xray_mask = self._pad_ids_with_mask([f[1] for f in features])
                clinical_ids, clinical_mask = self._pad_ids_with_mask([f[2] for f in features])
                labels = torch.stack([f[3] for f in features])
                return {
                    "pixel_values": images,
                    "xray_input_ids": xray_ids,
                    "xray_attention_mask": xray_mask,
                    "clinical_input_ids": clinical_ids,
                    "clinical_attention_mask": clinical_mask,
                    "labels": labels,
                }
            elif len(first) == 3:
                images = torch.stack([f[0] for f in features])
                input_ids, input_mask = self._pad_ids_with_mask([f[1] for f in features])
                labels = torch.stack([f[2] for f in features])
                return {
                    "pixel_values": images,
                    "xray_input_ids": input_ids,
                    "xray_attention_mask": input_mask,
                    "clinical_input_ids": input_ids,
                    "clinical_attention_mask": input_mask,
                    "labels": labels,
                }

        # Dictionary format fallback
        pixel_values = torch.stack([f["pixel_values"] for f in features])
        labels = torch.stack([f["labels"] for f in features])
        
        xray_ids_list = [f.get("xray_input_ids", f.get("input_ids")) for f in features]
        clinical_ids_list = [f.get("clinical_input_ids", f.get("input_ids")) for f in features]
        
        xray_ids, xray_mask = self._pad_ids_with_mask(xray_ids_list)
        clinical_ids, clinical_mask = self._pad_ids_with_mask(clinical_ids_list)

        return {
            "pixel_values": pixel_values,
            "xray_input_ids": xray_ids,
            "xray_attention_mask": xray_mask,
            "clinical_input_ids": clinical_ids,
            "clinical_attention_mask": clinical_mask,
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
        labels_for_loss = labels.argmax(dim=-1) if labels.ndim > 1 else labels

        if self.phase == "phase1":
            # --- Phase 1: Contrastive image-text alignment ---
            if self.p1_report_type in ("both", "xray_clinical"):
                xray_ids = inputs["xray_input_ids"]
                xray_mask = inputs.get("xray_attention_mask")
                clinical_ids = inputs["clinical_input_ids"]
                clinical_mask = inputs.get("clinical_attention_mask")
                
                image_features, xray_feat = model.backbone(images, xray_ids, attention_mask=xray_mask)
                _, clinical_feat = model.backbone(images, clinical_ids, attention_mask=clinical_mask)
                text_features = (xray_feat + clinical_feat) / 2.0
            else:
                is_xray = self.p1_report_type == "xray"
                text_ids = inputs["xray_input_ids"] if is_xray else inputs["clinical_input_ids"]
                text_mask = inputs.get("xray_attention_mask" if is_xray else "clinical_attention_mask")
                image_features, text_features = model.backbone(images, text_ids, attention_mask=text_mask)
            loss = self.loss_fn(image_features, text_features, labels_for_loss)
            outputs = {"image_features": image_features, "text_features": text_features}
        else:
            # --- Phase 2: Classification with optional fusion ---
            if self.use_text_in_p2 and self.p2_report_type in ("both", "xray_clinical"):
                xray_ids = inputs["xray_input_ids"]
                xray_mask = inputs.get("xray_attention_mask")
                clinical_ids = inputs["clinical_input_ids"]
                clinical_mask = inputs.get("clinical_attention_mask")
                
                img_feat, xray_feat = model.backbone(images, xray_ids, attention_mask=xray_mask)
                _, clinical_feat = model.backbone(images, clinical_ids, attention_mask=clinical_mask)
                text_feat = (xray_feat + clinical_feat) / 2.0
                
                # We blend the masks for cross-attention if both are used
                # In this complex dual case, typically we use one mask or intersection.
                # To be safe, we just use xray_mask as primary if not None.
                fusion_mask = xray_mask if xray_mask is not None else clinical_mask
                
                if model.fusion is not None:
                    # In composer, fusion is called inside model(images, text_ids).
                    # Here we call model.fusion directly for the dual branch.
                    if fusion_mask is not None and text_feat.dim() == 3:
                        txt_key_padding_mask = (fusion_mask[:, 1:] == 0)
                        try:
                            fused = model.fusion(img_feat, text_feat, txt_key_padding_mask=txt_key_padding_mask)
                        except TypeError:
                            fused = model.fusion(img_feat, text_feat)
                    else:
                        fused = model.fusion(img_feat, text_feat)
                else:
                    fused = img_feat
                text_ids = None
                text_mask = None
            elif self.use_text_in_p2:
                is_xray = self.p2_report_type == "xray"
                text_ids = inputs["xray_input_ids"] if is_xray else inputs["clinical_input_ids"]
                text_mask = inputs.get("xray_attention_mask" if is_xray else "clinical_attention_mask")
            else:
                text_ids = inputs["xray_input_ids"]
                text_mask = inputs.get("xray_attention_mask")

            from src.utils.losses import CombinedPhase2Loss, CombinedPhase2LossMulticlass
            if self.use_text_in_p2 and self.p2_report_type in ("both", "xray_clinical"):
                if isinstance(self.loss_fn, (CombinedPhase2Loss, CombinedPhase2LossMulticlass)):
                    logits, features = model.head(fused, return_features=True)
                    prototypes = getattr(model.head, "prototypes", None)
                    loss = self.loss_fn(logits, labels_for_loss, features=fused, prototypes=prototypes)
                else:
                    logits = model.head(fused)
                    loss = self.loss_fn(logits, labels_for_loss)
                outputs = logits
            elif isinstance(self.loss_fn, (CombinedPhase2Loss, CombinedPhase2LossMulticlass)):
                outputs = model(images, text_ids, attention_mask=text_mask, return_features=True)
                if isinstance(outputs, tuple) and len(outputs) == 3:
                    logits, features, prototypes = outputs
                    loss = self.loss_fn(logits, labels_for_loss, features=features, prototypes=prototypes)
                else:
                    logits = outputs[0] if isinstance(outputs, tuple) else outputs
                    loss = self.loss_fn(logits, labels_for_loss)
            else:
                outputs = model(images, text_ids, attention_mask=text_mask)
                logits = outputs[0] if isinstance(outputs, tuple) else outputs
                loss = self.loss_fn(logits, labels_for_loss)

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


