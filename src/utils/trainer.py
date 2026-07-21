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
from torch.utils.data import Sampler, Subset
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


def _sampler_class_ids(dataset) -> torch.Tensor:
    """Extract integer class IDs without loading images or reports."""
    if isinstance(dataset, Subset):
        parent = _sampler_class_ids(dataset.dataset)
        indices = torch.as_tensor(dataset.indices, dtype=torch.long)
        return parent[indices]
    if hasattr(dataset, "df") and "class_id" in dataset.df.columns:
        return torch.as_tensor(
            dataset.df["class_id"].to_numpy(), dtype=torch.long
        )
    for attribute in ("labels", "targets"):
        if hasattr(dataset, attribute):
            labels = torch.as_tensor(getattr(dataset, attribute))
            if labels.ndim > 1:
                labels = labels.argmax(dim=-1)
            return labels.long().reshape(-1)
    raise AttributeError(
        "Class-aware sampling requires df['class_id'], labels, targets, or a Subset."
    )


class ClassAwareSampler(Sampler[int]):
    """Arrange indices so each full physical batch contains same-class pairs.

    This sampler is intended for Phase-1 in-batch semantic matching.  It keeps
    the epoch length unchanged while sampling classes approximately uniformly.
    Samples may repeat across batches, but are drawn without replacement within
    a pair whenever a class contains at least two examples.
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        samples_per_class: int = 2,
        seed: int = 42,
    ) -> None:
        if batch_size < 2:
            raise ValueError("Class-aware sampling requires batch_size >= 2.")
        if samples_per_class < 2 or samples_per_class > batch_size:
            raise ValueError(
                "samples_per_class must be between 2 and batch_size."
            )
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.samples_per_class = int(samples_per_class)
        self.seed = int(seed)
        self.epoch = 0
        labels = _sampler_class_ids(dataset)
        if labels.numel() != len(dataset):
            raise ValueError("Extracted labels do not match the dataset length.")
        self.class_to_indices = {
            int(class_id): torch.nonzero(
                labels == class_id, as_tuple=False
            ).flatten()
            for class_id in torch.unique(labels, sorted=True).tolist()
        }
        if len(self.class_to_indices) < 2:
            raise ValueError("Class-aware sampling requires at least two classes.")
        self.classes = torch.tensor(
            sorted(self.class_to_indices), dtype=torch.long
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        sequence: list[int] = []
        remaining_total = len(self.dataset)

        while remaining_total > 0:
            current_batch = min(self.batch_size, remaining_total)
            pair_groups = current_batch // self.samples_per_class
            selected: list[int] = []

            if pair_groups:
                replace_classes = pair_groups > len(self.classes)
                if replace_classes:
                    class_positions = torch.randint(
                        len(self.classes),
                        (pair_groups,),
                        generator=generator,
                    )
                else:
                    class_positions = torch.randperm(
                        len(self.classes), generator=generator
                    )[:pair_groups]

                for position in class_positions.tolist():
                    class_id = int(self.classes[position])
                    candidates = self.class_to_indices[class_id]
                    if len(candidates) >= self.samples_per_class:
                        chosen = candidates[
                            torch.randperm(len(candidates), generator=generator)[
                                : self.samples_per_class
                            ]
                        ]
                    else:
                        chosen = candidates[
                            torch.randint(
                                len(candidates),
                                (self.samples_per_class,),
                                generator=generator,
                            )
                        ]
                    selected.extend(int(index) for index in chosen.tolist())

            while len(selected) < current_batch:
                selected.append(
                    int(torch.randint(len(self.dataset), (1,), generator=generator))
                )
            order = torch.randperm(len(selected), generator=generator).tolist()
            sequence.extend(selected[index] for index in order)
            remaining_total -= current_batch

        return iter(sequence)


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
        """Right-pad token IDs and build a correct attention mask.

        Args:
            ids_list: List of 1-D token-ID tensors.

        Returns:
            padded_ids: Tensor [B, L].
            attention_mask: Tensor [B, L], with 1 for valid tokens
                and 0 for padding tokens.
        """
        import torch

        if not ids_list:
            raise ValueError("ids_list must not be empty.")

        if any(ids is None for ids in ids_list):
            raise ValueError(
                "ids_list contains None. Check dataset text tokenization."
            )

        max_len = max(ids.shape[0] for ids in ids_list)
        padded_list = []

        for ids in ids_list:
            if ids.ndim != 1:
                ids = ids.view(-1)

            pad_len = max_len - ids.shape[0]

            if pad_len > 0:
                padding = torch.full(
                    (pad_len,),
                    fill_value=self.pad_token_id,
                    dtype=ids.dtype,
                    device=ids.device,
                )
                ids = torch.cat([ids, padding], dim=0)

            padded_list.append(ids)

        padded_ids = torch.stack(padded_list, dim=0)

        # Padding token = 0 trong attention mask.
        attention_mask = (
            padded_ids != self.pad_token_id
        ).to(dtype=torch.long)

        return padded_ids, attention_mask


    def _pad_optional_ids_with_mask(self, ids_list):
        """Pad optional text fields; all-missing fields become a masked dummy token."""
        if all(ids is None for ids in ids_list):
            batch_size = len(ids_list)
            ids = torch.full(
                (batch_size, 1), self.pad_token_id, dtype=torch.long
            )
            mask = torch.zeros((batch_size, 1), dtype=torch.long)
            return ids, mask
        if any(ids is None for ids in ids_list):
            raise ValueError(
                "A text field is missing for only part of a batch. Ensure dataset "
                "samples use a consistent schema."
            )
        return self._pad_ids_with_mask(ids_list)

    def _pad_tiles_with_mask(self, tiles_list):
        """Pad tile tensors to have the same number of tiles (N) per image in a batch.
        
        Args:
            tiles_list: List of tensors of shape (N_i, C, H, W).
            
        Returns:
            Tuple of (padded_tiles, tile_masks).
            padded_tiles has shape (B, N_max, C, H, W).
            tile_masks has shape (B, N_max) with 1 for real tiles and 0 for padding.
        """
        import torch
        max_tiles = max(tiles.shape[0] for tiles in tiles_list)
        
        padded_tiles = []
        tile_masks = []
        
        for tiles in tiles_list:
            n = tiles.shape[0]
            pad_n = max_tiles - n
            
            mask = torch.cat([
                torch.ones(n, dtype=torch.long),
                torch.zeros(pad_n, dtype=torch.long),
            ])
            
            if pad_n > 0:
                pad = torch.zeros((pad_n, *tiles.shape[1:]), dtype=tiles.dtype, device=tiles.device)
                tiles = torch.cat([tiles, pad], dim=0)
                
            padded_tiles.append(tiles)
            tile_masks.append(mask)
            
        return torch.stack(padded_tiles), torch.stack(tile_masks)

    def _pad_tile_metadata(self, features: list, max_tiles: int):
        """Pad normalized source-image boxes for high-resolution tiles."""
        padded_boxes = []
        for feature in features:
            tile_count = feature["tile_values"].shape[0]
            boxes = feature.get("tile_boxes")
            if boxes is None:
                boxes = torch.zeros((tile_count, 4), dtype=torch.float32)
            if boxes.shape != (tile_count, 4):
                raise ValueError(
                    f"tile_boxes must be {(tile_count, 4)}, got {tuple(boxes.shape)}"
                )
            pad_count = max_tiles - tile_count
            if pad_count:
                boxes = torch.cat(
                    [boxes, torch.zeros((pad_count, 4), dtype=boxes.dtype)], dim=0
                )
            padded_boxes.append(boxes)
        return torch.stack(padded_boxes)

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
                labels = torch.stack([f[3] if isinstance(f[3], torch.Tensor) else torch.tensor(f[3]) for f in features])
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
                labels = torch.stack([f[2] if isinstance(f[2], torch.Tensor) else torch.tensor(f[2]) for f in features])
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
        labels = torch.stack([f["labels"] if isinstance(f["labels"], torch.Tensor) else torch.tensor(f["labels"]) for f in features])
        
        xray_ids_list = [f.get("xray_input_ids", f.get("input_ids")) for f in features]
        clinical_ids_list = [f.get("clinical_input_ids", f.get("input_ids")) for f in features]
        
        xray_ids, xray_mask = self._pad_optional_ids_with_mask(xray_ids_list)
        clinical_ids, clinical_mask = self._pad_optional_ids_with_mask(clinical_ids_list)

        
        batch = {
            "pixel_values": pixel_values,
            "xray_input_ids": xray_ids,
            "xray_attention_mask": xray_mask,
            "clinical_input_ids": clinical_ids,
            "clinical_attention_mask": clinical_mask,
            "labels": labels,
        }
        
        # High-Res Tiling Support
        if "tile_values" in features[0]:
            tile_values_list = [f["tile_values"] for f in features]
            padded_tiles, tile_masks = self._pad_tiles_with_mask(tile_values_list)
            batch["tile_values"] = padded_tiles
            batch["tile_mask"] = tile_masks
            tile_boxes = self._pad_tile_metadata(
                features, padded_tiles.shape[1]
            )
            batch["tile_boxes"] = tile_boxes
            
        return batch


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
                 class_aware_sampling=None,
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
        self.phase = phase
        self.class_aware_sampling = dict(class_aware_sampling or {})
        super().__init__(*args, **kwargs)
        self.loss_fn = loss_fn
        self.use_text_in_p2 = use_text_in_p2
        self.p1_report_type = p1_report_type
        self.p2_report_type = p2_report_type

    def _get_train_sampler(self, train_dataset=None):
        dataset = train_dataset if train_dataset is not None else self.train_dataset
        enabled = bool(self.class_aware_sampling.get("enabled", False))
        if self.phase == "phase1" and enabled and dataset is not None:
            return ClassAwareSampler(
                dataset=dataset,
                batch_size=int(self.args.train_batch_size),
                samples_per_class=int(
                    self.class_aware_sampling.get("samples_per_class", 2)
                ),
                seed=int(self.class_aware_sampling.get("seed", self.args.seed)),
            )
        return super()._get_train_sampler(train_dataset)

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
        
        tile_values = inputs.get("tile_values")
        tile_mask = inputs.get("tile_mask")
        tile_boxes = inputs.get("tile_boxes")

        if self.phase == "phase1":
            # --- Phase 1: Contrastive image-text alignment ---
            if self.p1_report_type in ("both", "xray_clinical"):
                raise NotImplementedError(
                    "Simultaneous token-level X-ray and clinical report alignment is "
                    "not implemented. Configure p1_report_type='xray' or 'clinical'."
                )

            is_xray = self.p1_report_type == "xray"
            text_ids = inputs["xray_input_ids"] if is_xray else inputs["clinical_input_ids"]
            text_mask = inputs.get(
                "xray_attention_mask" if is_xray else "clinical_attention_mask"
            )
            image_features, text_features = model.backbone(
                images,
                text_ids,
                attention_mask=text_mask,
                tile_values=tile_values,
                tile_mask=tile_mask,
                tile_boxes=tile_boxes,
            )

            # Semantic matching requires one vector per image. High-resolution
            # backbones may provide a global-anchored local residual pooler;
            # generic backbones retain the masked-mean fallback.
            if image_features.dim() == 3:
                contrastive_pooler = getattr(
                    model.backbone,
                    "pool_contrastive_image_features",
                    None,
                )
                if callable(contrastive_pooler):
                    image_features = contrastive_pooler(image_features)
                else:
                    image_padding_mask = getattr(
                        model.backbone, "last_image_key_padding_mask", None
                    )
                    if image_padding_mask is None:
                        image_features = image_features.mean(dim=1)
                    else:
                        valid = (~image_padding_mask).unsqueeze(-1).to(
                            image_features.dtype
                        )
                        image_features = (image_features * valid).sum(
                            dim=1
                        ) / valid.sum(dim=1).clamp_min(1.0)
                
            loss = self.loss_fn(image_features, text_features, labels_for_loss)
            outputs = {"image_features": image_features, "text_features": text_features}
        else:
            # --- Phase 2: Classification with optional fusion ---
            if self.use_text_in_p2 and self.p2_report_type in ("both", "xray_clinical"):
                raise NotImplementedError(
                    "Simultaneous token-level X-ray and clinical report fusion is "
                    "not implemented. Configure p2_report_type='xray' or 'clinical'."
                )
            elif self.use_text_in_p2:
                is_xray = self.p2_report_type == "xray"
                text_ids = inputs["xray_input_ids"] if is_xray else inputs["clinical_input_ids"]
                text_mask = inputs.get("xray_attention_mask" if is_xray else "clinical_attention_mask")
            else:
                # True image-only mode: do not pass any text to the backbone.
                text_ids = None
                text_mask = None

            outputs = model(
                images,
                text_ids,
                attention_mask=text_mask,
                tile_values=tile_values,
                tile_mask=tile_mask,
                tile_boxes=tile_boxes,
            )
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
