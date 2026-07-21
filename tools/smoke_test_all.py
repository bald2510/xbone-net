#!/usr/bin/env python
"""
One-command smoke-test suite for all XBone-Net experiments.

Run from the project root:

    python tools/smoke_test_all.py \
        --config-root configs \
        --manifest tools/smoke_manifest.yaml \
        --device cuda

The runner:
  1. Locates every planned experiment config.
  2. Resolves each Hydra config exactly once.
  3. Builds the model and train/validation data loaders.
  4. Runs one Phase-1 forward/backward step when enabled.
  5. Runs one Phase-2 forward/backward step when enabled.
  6. Checks high-resolution tile shapes and fixed visual-token count.
  7. Runs one validation batch and metric computation.
  8. Saves, rebuilds, reloads, and compares a checkpoint.
  9. Runs global metric/bootstrap/OOD unit checks.
 10. Writes JSON, CSV, and Markdown reports.

This is a smoke test, not a performance benchmark.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import fnmatch
import gc
import io
import json
import os
import random
import sys
import tempfile
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SmokeResult:
    experiment_id: str
    config_file: str
    status: str = "FAIL"
    compose_method: str = ""
    backbone_type: str = ""
    peft_type: str = ""
    fusion_type: str = ""
    classifier_type: str = ""
    phase1_enabled: bool = False
    phase2_enabled: bool = False
    high_res_enabled: bool = False
    phase1_loss: float | None = None
    phase2_loss: float | None = None
    eval_loss: float | None = None
    logits_shape: str = ""
    image_feature_shape: str = ""
    text_feature_shape: str = ""
    tile_shape: str = ""
    trainable_backbone: int = 0
    trainable_fusion: int = 0
    trainable_head: int = 0
    checkpoint_reload_match: bool = False
    warnings: list[str] = field(default_factory=list)
    error_stage: str = ""
    error: str = ""
    traceback: str = ""
    elapsed_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tensor_shape(value: Any) -> str:
    if isinstance(value, torch.Tensor):
        return str(tuple(value.shape))
    return ""


def move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        moved = [move_to_device(item, device) for item in value]
        return type(value)(moved)
    return value


def finite_scalar(value: torch.Tensor, name: str) -> float:
    if not isinstance(value, torch.Tensor) or value.ndim != 0:
        raise AssertionError(f"{name} must be a scalar tensor, got {type(value)}.")
    if not torch.isfinite(value):
        raise AssertionError(f"{name} is NaN or infinity: {value.detach().cpu().item()}.")
    return float(value.detach().cpu().item())


def trainable_count(module: nn.Module | None) -> int:
    if module is None:
        return 0
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def gradient_count(module: nn.Module | None) -> int:
    if module is None:
        return 0
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
        and parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
    )


def parse_logits(outputs: Any) -> torch.Tensor:
    if isinstance(outputs, tuple):
        outputs = outputs[0]
    if isinstance(outputs, dict):
        if "logits" in outputs:
            outputs = outputs["logits"]
        else:
            raise AssertionError(f"Cannot extract logits from output keys: {list(outputs)}")
    if not isinstance(outputs, torch.Tensor):
        raise AssertionError(f"Expected logits tensor, got {type(outputs)}.")
    return outputs


def get_nested(cfg: Any, path: str, default: Any = None) -> Any:
    current = cfg
    for part in path.split("."):
        try:
            if current is None:
                return default
            if hasattr(current, "get"):
                current = current.get(part, default)
            else:
                current = getattr(current, part)
        except Exception:
            return default
    return current


def set_if_present(cfg: Any, path: str, value: Any) -> None:
    """Set an OmegaConf path when its parent exists."""
    from omegaconf import open_dict

    parts = path.split(".")
    current = cfg
    with open_dict(cfg):
        for part in parts[:-1]:
            if current is None:
                return
            if hasattr(current, "get"):
                next_value = current.get(part, None)
            else:
                next_value = getattr(current, part, None)
            if next_value is None:
                return
            current = next_value
        if hasattr(current, "__setitem__"):
            current[parts[-1]] = value
        else:
            setattr(current, parts[-1], value)


# ---------------------------------------------------------------------------
# Manifest and config discovery
# ---------------------------------------------------------------------------

def load_manifest(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required for smoke_manifest.yaml.") from exc
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if "experiments" not in data:
        raise ValueError("Manifest must contain an 'experiments' list.")
    return data


def all_yaml_files(config_root: Path) -> list[Path]:
    return sorted(
        file
        for file in config_root.rglob("*.yaml")
        if "__pycache__" not in file.parts
    )


def matches_any(relative_lower: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatch(relative_lower, pattern.lower()) for pattern in patterns)


def resolve_manifest_configs(
    config_root: Path,
    manifest: dict[str, Any],
) -> tuple[list[tuple[dict[str, Any], Path]], list[str]]:
    files = all_yaml_files(config_root)
    relative_map = {
        file: file.relative_to(config_root).as_posix().lower()
        for file in files
    }

    resolved: list[tuple[dict[str, Any], Path]] = []
    missing: list[str] = []
    used_files: set[Path] = set()

    for experiment in manifest["experiments"]:
        experiment_id = str(experiment["id"])
        patterns = experiment.get("patterns", [])
        excludes = [text.lower() for text in experiment.get("exclude_contains", [])]

        candidates = []
        for file, relative_lower in relative_map.items():
            if not matches_any(relative_lower, patterns):
                continue
            if any(excluded in relative_lower for excluded in excludes):
                continue
            candidates.append(file)

        if not candidates:
            missing.append(experiment_id)
            continue

        # Prefer the shortest path and then lexicographic order.
        candidates.sort(key=lambda item: (len(item.relative_to(config_root).parts), str(item)))
        selected = candidates[0]

        if selected in used_files:
            # A duplicated file assignment usually indicates overly broad patterns.
            missing.append(
                f"{experiment_id} (matched duplicate config: "
                f"{selected.relative_to(config_root).as_posix()})"
            )
            continue

        used_files.add(selected)
        resolved.append((experiment, selected))

    return resolved, missing


def compose_config(config_root: Path, config_file: Path):
    """Resolve a config using direct and base-config Hydra strategies."""
    try:
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
    except ImportError as exc:
        raise RuntimeError("hydra-core is required to resolve experiment configs.") from exc

    relative = config_file.relative_to(config_root).with_suffix("").as_posix()
    parts = relative.split("/")

    attempts: list[tuple[str, str, list[str]]] = [
        ("direct", relative, []),
    ]

    if parts[0] in {"experiment", "experiments"} and len(parts) > 1:
        group = parts[0]
        value = "/".join(parts[1:])
        attempts.extend(
            [
                (f"config+{group}", "config", [f"+{group}={value}"]),
                (f"config={group}", "config", [f"{group}={value}"]),
            ]
        )
    else:
        attempts.extend(
            [
                ("config+experiment", "config", [f"+experiment={relative}"]),
                ("config=experiment", "config", [f"experiment={relative}"]),
                ("config+experiments", "config", [f"+experiments={relative}"]),
                ("config=experiments", "config", [f"experiments={relative}"]),
            ]
        )

    errors = []
    for method, config_name, overrides in attempts:
        try:
            GlobalHydra.instance().clear()
            with initialize_config_dir(
                version_base="1.3",
                config_dir=str(config_root.resolve()),
            ):
                cfg = compose(
                    config_name=config_name,
                    overrides=overrides,
                    return_hydra_config=True,
                )
            return cfg, method
        except Exception as exc:
            errors.append(f"{method}: {type(exc).__name__}: {exc}")

    raise RuntimeError(
        f"Could not compose {relative}. Attempts:\n" + "\n".join(errors)
    )


def patch_for_smoke(
    cfg: Any,
    *,
    batch_size: int,
    seed: int,
    temp_dir: Path,
) -> Any:
    """Apply runtime-only smoke overrides without altering experiment semantics."""
    from omegaconf import OmegaConf, open_dict, DictConfig, ListConfig
    from hydra.core.hydra_config import HydraConfig

    def make_writable(node):
        OmegaConf.set_readonly(node, False)
        if isinstance(node, DictConfig):
            for k in node.keys():
                if k == "hydra":
                    continue
                child = node._get_node(k)
                if isinstance(child, (DictConfig, ListConfig)):
                    make_writable(child)
        elif isinstance(node, ListConfig):
            for child in node:
                if isinstance(child, (DictConfig, ListConfig)):
                    make_writable(child)

    make_writable(cfg)
    HydraConfig.instance().set_config(cfg)

    with open_dict(cfg):
        cfg.pop("hydra", None)
        cfg.seed = seed

        if cfg.get("dataset") is None:
            raise ValueError("Resolved config has no dataset section.")
        cfg.dataset.batch_size = batch_size
        cfg.dataset.num_workers = 0
        if cfg.dataset.get("pin_memory") is not None:
            cfg.dataset.pin_memory = False
        if cfg.dataset.get("persistent_workers") is not None:
            cfg.dataset.persistent_workers = False

        if cfg.get("params") is None:
            raise ValueError("Resolved config has no params section.")

        cfg.params.run_ood = False
        cfg.params.gradient_checkpointing = False
        cfg.params.model_dir = str(temp_dir)

        p1 = cfg.params.get("phase1", None)
        if p1 is not None:
            p1.epochs = 1
            p1.early_stopping_patience = 1
            p1.checkpoint_path = str(temp_dir / "best_phase1.pth")
            p1.merged_checkpoint_path = str(temp_dir / "merged_phase1.pth")

        p2 = cfg.params.get("phase2", None)
        if p2 is not None:
            p2.epochs = 1
            p2.early_stopping_patience = 1
            p2.checkpoint_path = str(temp_dir / "best_phase2.pth")

    OmegaConf.resolve(cfg)
    return cfg


# ---------------------------------------------------------------------------
# Project-specific setup
# ---------------------------------------------------------------------------

def import_project_api():
    try:
        from src.models.builder import build_model, setup_phase2_modules
        from src.datasets.builder import build_dataloader
        from src.utils.trainer import (
            BioMedCLIPDataCollator,
            SFTrainer,
            resolve_pad_token_id,
        )
        from src.utils.losses import build_loss, build_phase2_loss
        from src.utils.metrics import (
            compute_metrics_multiclass,
            bootstrap_confidence_intervals,
        )
        from src.utils.ood import OODDetector, evaluate_ood
    except Exception as exc:
        raise RuntimeError(
            "Could not import project modules. Run this script from the project root "
            "after copying the corrected pipeline files."
        ) from exc

    return {
        "build_model": build_model,
        "setup_phase2_modules": setup_phase2_modules,
        "build_dataloader": build_dataloader,
        "BioMedCLIPDataCollator": BioMedCLIPDataCollator,
        "SFTrainer": SFTrainer,
        "resolve_pad_token_id": resolve_pad_token_id,
        "build_loss": build_loss,
        "build_phase2_loss": build_phase2_loss,
        "compute_metrics_multiclass": compute_metrics_multiclass,
        "bootstrap_confidence_intervals": bootstrap_confidence_intervals,
        "OODDetector": OODDetector,
        "evaluate_ood": evaluate_ood,
    }


def make_compute_loss_adapter(
    SFTrainer,
    *,
    phase: str,
    loss_fn: nn.Module,
    use_text_in_p2: bool = True,
    p1_report_type: str = "clinical",
    p2_report_type: str = "clinical",
):
    """Create a lightweight object that executes the exact SFTrainer.compute_loss."""
    adapter = object.__new__(SFTrainer)
    adapter.phase = phase
    adapter.loss_fn = loss_fn
    adapter.use_text_in_p2 = use_text_in_p2
    adapter.p1_report_type = p1_report_type
    adapter.p2_report_type = p2_report_type
    return adapter


def build_loaders(cfg: Any, model: nn.Module, api: dict[str, Any]):
    tokenizer = getattr(
        model.backbone,
        "tokenizer_obj",
        getattr(model.backbone, "tokenizer", None),
    )
    preprocess = getattr(model.backbone, "preprocess", None)

    train_loader = api["build_dataloader"](
        cfg.dataset,
        split="train",
        transform=preprocess,
        tokenizer=tokenizer,
    )
    val_loader = api["build_dataloader"](
        cfg.dataset,
        split="val",
        transform=preprocess,
        tokenizer=tokenizer,
    )

    pad_id = api["resolve_pad_token_id"](tokenizer) if tokenizer is not None else 0
    collator = api["BioMedCLIPDataCollator"](pad_token_id=pad_id)

    # Match train.py exactly.
    train_loader.collate_fn = collator
    val_loader.collate_fn = collator

    return train_loader, val_loader


def first_batch(loader, name: str) -> dict[str, Any]:
    try:
        batch = next(iter(loader))
    except StopIteration as exc:
        raise RuntimeError(f"{name} loader is empty.") from exc
    if not isinstance(batch, dict):
        raise TypeError(
            f"{name} loader must yield dictionary batches, got {type(batch)}."
        )
    required = {"pixel_values", "labels"}
    missing = required.difference(batch)
    if missing:
        raise KeyError(f"{name} batch is missing keys: {sorted(missing)}")
    return batch


def phase_enabled(cfg: Any, phase: str) -> bool:
    params = cfg.params
    phase_cfg = params.get(phase, {}) or {}
    fallback = params.get(f"run_{phase}", True)
    return bool(phase_cfg.get("enabled", fallback))


def class_metadata(cfg: Any) -> tuple[list[str], int]:
    dataset_params = cfg.dataset.get("params", {}) or {}
    names = dataset_params.get(
        "classes",
        dataset_params.get("pathologies", []),
    )
    names = list(names) if names is not None else []
    num_classes = int(dataset_params.get("num_classes", len(names)))
    if num_classes < 2:
        raise ValueError(f"Expected at least two classes, got {num_classes}.")
    if len(names) != num_classes:
        names = [f"class_{index}" for index in range(num_classes)]
    return names, num_classes


def configure_phase2_trainability(model: nn.Module, cfg: Any) -> None:
    p1_cfg = cfg.params.get("phase1", {}) or {}
    p2_cfg = cfg.params.get("phase2", {}) or {}

    do_phase1 = phase_enabled(cfg, "phase1")
    merge_after_p1 = bool(p1_cfg.get("merge_lora_after_training", False))
    init_from_merged = bool(p2_cfg.get("init_from_phase1_merged", False))

    is_merged = (do_phase1 and merge_after_p1) or init_from_merged
    if is_merged:
        for parameter in model.backbone.parameters():
            parameter.requires_grad = False

    for parameter in model.fusion.parameters():
        parameter.requires_grad = True
    for parameter in model.head.parameters():
        parameter.requires_grad = True


def choose_text_batch(batch: dict[str, Any], report_type: str):
    if report_type == "xray":
        return batch.get("xray_input_ids"), batch.get("xray_attention_mask")
    if report_type == "clinical":
        return batch.get("clinical_input_ids"), batch.get("clinical_attention_mask")
    raise ValueError(
        f"Smoke test supports report_type='xray' or 'clinical', got {report_type!r}."
    )


# ---------------------------------------------------------------------------
# Per-experiment smoke
# ---------------------------------------------------------------------------

def smoke_one_experiment(
    experiment: dict[str, Any],
    config_file: Path,
    *,
    config_root: Path,
    device: torch.device,
    batch_size: int,
    seed: int,
    work_root: Path,
    api: dict[str, Any],
) -> SmokeResult:
    start = time.perf_counter()
    result = SmokeResult(
        experiment_id=str(experiment["id"]),
        config_file=config_file.relative_to(config_root).as_posix(),
    )
    stage = "compose"

    try:
        exp_temp = work_root / result.experiment_id
        exp_temp.mkdir(parents=True, exist_ok=True)

        cfg, compose_method = compose_config(config_root, config_file)
        result.compose_method = compose_method
        cfg = patch_for_smoke(
            cfg,
            batch_size=batch_size,
            seed=seed,
            temp_dir=exp_temp,
        )

        result.backbone_type = str(cfg.model.get("backbone_type", ""))
        result.peft_type = str((cfg.model.get("peft", {}) or {}).get("type", "none"))
        result.phase1_enabled = phase_enabled(cfg, "phase1")
        result.phase2_enabled = phase_enabled(cfg, "phase2")

        high_res_cfg = (
            ((cfg.dataset.get("params", {}) or {}).get("high_res", {}) or {})
        )
        result.high_res_enabled = bool(high_res_cfg.get("enabled", False))

        stage = "build_model"
        model = api["build_model"](cfg.model).to(device)
        initial_trainable_names = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]

        stage = "build_dataloaders"
        train_loader, val_loader = build_loaders(cfg, model, api)
        train_batch = move_to_device(first_batch(train_loader, "train"), device)
        val_batch = move_to_device(first_batch(val_loader, "validation"), device)

        result.tile_shape = tensor_shape(train_batch.get("tile_values"))

        if result.high_res_enabled:
            if "tile_values" not in train_batch or "tile_mask" not in train_batch:
                raise AssertionError(
                    "High-resolution config did not produce tile_values and tile_mask."
                )
            tile_values = train_batch["tile_values"]
            tile_mask = train_batch["tile_mask"]
            if tile_values.ndim != 5:
                raise AssertionError(
                    f"tile_values must be [B,N,C,H,W], got {tuple(tile_values.shape)}"
                )
            if tile_mask.shape != tile_values.shape[:2]:
                raise AssertionError(
                    f"tile_mask {tuple(tile_mask.shape)} does not match "
                    f"tile_values {tuple(tile_values.shape[:2])}."
                )
            if torch.any(tile_mask.sum(dim=1) < 1):
                raise AssertionError("At least one high-resolution sample has zero valid tiles.")
        else:
            if "tile_values" in train_batch:
                result.warnings.append(
                    "High-res is disabled but the dataset still returned tile_values."
                )

        classes, num_classes = class_metadata(cfg)

        # ---------------------------------------------------------------
        # Phase 1
        # ---------------------------------------------------------------
        if result.phase1_enabled:
            stage = "phase1_setup"
            for parameter in model.fusion.parameters():
                parameter.requires_grad = False
            for parameter in model.head.parameters():
                parameter.requires_grad = False

            p1_cfg = cfg.params.phase1
            clip_model = getattr(model.backbone, "model", model.backbone)
            phase1_loss_fn = api["build_loss"](
                p1_cfg.get("loss_type", "semantic_matching"),
                clip_model=clip_model,
                temperature=p1_cfg.get("temperature", 0.07),
                target_similarity=p1_cfg.get("target_similarity", 0.7),
            )
            phase1_adapter = make_compute_loss_adapter(
                api["SFTrainer"],
                phase="phase1",
                loss_fn=phase1_loss_fn,
                p1_report_type=p1_cfg.get("p1_report_type", "clinical"),
            )

            stage = "phase1_forward_backward"
            model.train()
            model.zero_grad(set_to_none=True)
            phase1_loss, phase1_outputs = api["SFTrainer"].compute_loss(
                phase1_adapter,
                model,
                train_batch,
                return_outputs=True,
            )
            result.phase1_loss = finite_scalar(phase1_loss, "Phase-1 loss")

            image_features = phase1_outputs["image_features"]
            text_features = phase1_outputs["text_features"]
            result.image_feature_shape = tensor_shape(image_features)
            result.text_feature_shape = tensor_shape(text_features)

            if image_features.ndim != 2 or text_features.ndim != 2:
                raise AssertionError(
                    "Phase 1 must return [B,D] image and text embeddings, got "
                    f"{tuple(image_features.shape)} and {tuple(text_features.shape)}."
                )
            if image_features.shape != text_features.shape:
                raise AssertionError(
                    "Phase-1 image/text embedding shapes differ: "
                    f"{tuple(image_features.shape)} vs {tuple(text_features.shape)}."
                )

            phase1_loss.backward()
            if gradient_count(model.backbone) == 0:
                raise AssertionError("Phase 1 produced no finite backbone/adapter gradients.")

            trainable = [
                parameter
                for parameter in model.backbone.parameters()
                if parameter.requires_grad
            ]
            if trainable:
                optimizer = torch.optim.AdamW(trainable, lr=1e-7)
                optimizer.step()
            model.zero_grad(set_to_none=True)

        # ---------------------------------------------------------------
        # Phase 2
        # ---------------------------------------------------------------
        if result.phase2_enabled:
            stage = "phase2_setup"
            model, classifier_type, fusion_type, resolved_num_classes = (
                api["setup_phase2_modules"](model, cfg, device)
            )
            result.classifier_type = str(classifier_type)
            result.fusion_type = str(fusion_type)
            if int(resolved_num_classes) != num_classes:
                raise AssertionError(
                    f"Builder resolved {resolved_num_classes} classes; config has {num_classes}."
                )

            configure_phase2_trainability(model, cfg)
            result.trainable_backbone = trainable_count(model.backbone)
            result.trainable_fusion = trainable_count(model.fusion)
            result.trainable_head = trainable_count(model.head)

            # The production training loop estimates empirical centroids from
            # the complete labelled training split before the first Phase-2
            # forward pass.  A smoke test deliberately uses only one tiny
            # batch, so install deterministic, non-degenerate centroids here
            # to exercise the classifier (including an optional class bias)
            # without pretending that the batch is a valid estimate.
            if (
                classifier_type == "empirical_centroid"
                and hasattr(model.head, "set_centroids")
            ):
                stage = "initialize_smoke_centroids"
                centroids = torch.randn(
                    num_classes,
                    int(model.head.feature_dim),
                    device=device,
                )
                counts = torch.ones(num_classes, dtype=torch.long, device=device)
                model.head.set_centroids(centroids, counts)

            p2_cfg = cfg.params.phase2
            loss_cfg = p2_cfg.get("loss", {}) or {}
            phase2_loss_fn = api["build_phase2_loss"](
                loss_type=p2_cfg.get("loss_type", None),
                classifier_type=classifier_type,
                class_weights=None,
                label_smoothing=float(loss_cfg.get("label_smoothing", 0.0)),
            )

            phase2_adapter = make_compute_loss_adapter(
                api["SFTrainer"],
                phase="phase2",
                loss_fn=phase2_loss_fn,
                use_text_in_p2=bool(p2_cfg.get("use_text", True)),
                p2_report_type=p2_cfg.get("p2_report_type", "clinical"),
            )

            stage = "phase2_forward_backward"
            model.train()
            model.zero_grad(set_to_none=True)
            phase2_loss, phase2_outputs = api["SFTrainer"].compute_loss(
                phase2_adapter,
                model,
                train_batch,
                return_outputs=True,
            )
            result.phase2_loss = finite_scalar(phase2_loss, "Phase-2 loss")
            train_logits = parse_logits(phase2_outputs)
            result.logits_shape = tensor_shape(train_logits)

            if train_logits.ndim != 2:
                raise AssertionError(
                    f"Phase-2 logits must be [B,C], got {tuple(train_logits.shape)}."
                )
            if train_logits.size(1) != num_classes:
                raise AssertionError(
                    f"Logit classes={train_logits.size(1)}, expected {num_classes}."
                )

            phase2_loss.backward()

            if result.trainable_head > 0 and gradient_count(model.head) == 0:
                raise AssertionError("Classifier head received no finite gradients.")
            if result.trainable_fusion > 0 and gradient_count(model.fusion) == 0:
                raise AssertionError("Fusion module received no finite gradients.")

            trainable = [
                parameter
                for parameter in model.parameters()
                if parameter.requires_grad
            ]
            if not trainable:
                raise AssertionError("Phase 2 has no trainable parameters.")
            optimizer = torch.optim.AdamW(trainable, lr=1e-7)
            optimizer.step()
            model.zero_grad(set_to_none=True)

            # -----------------------------------------------------------
            # Fixed-K high-resolution assertions
            # -----------------------------------------------------------
            if result.high_res_enabled and hasattr(model.backbone, "num_visual_tokens"):
                stage = "high_res_shape_check"
                p2_report_type = p2_cfg.get("p2_report_type", "clinical")
                use_text = bool(p2_cfg.get("use_text", True))
                text_ids, text_mask = (
                    choose_text_batch(train_batch, p2_report_type)
                    if use_text
                    else (None, None)
                )

                model.eval()
                with torch.no_grad():
                    image_features, text_features = model.backbone(
                        train_batch["pixel_values"],
                        text_ids,
                        attention_mask=text_mask,
                        tile_values=train_batch.get("tile_values"),
                        tile_mask=train_batch.get("tile_mask"),
                    )

                expected_k = int(model.backbone.num_visual_tokens)
                if image_features.ndim != 3:
                    raise AssertionError(
                        f"High-res Phase-2 image features must be [B,K+1,D], got "
                        f"{tuple(image_features.shape)}."
                    )
                if image_features.size(1) != expected_k + 1:
                    raise AssertionError(
                        f"Fixed-K violation: got {image_features.size(1)-1} local tokens, "
                        f"expected {expected_k}."
                    )
                image_padding_mask = getattr(
                    model.backbone,
                    "last_image_key_padding_mask",
                    None,
                )
                if image_padding_mask is None:
                    raise AssertionError("High-res backbone did not expose an image padding mask.")
                if image_padding_mask.shape != image_features.shape[:2]:
                    raise AssertionError(
                        f"Image mask {tuple(image_padding_mask.shape)} does not match "
                        f"image features {tuple(image_features.shape[:2])}."
                    )

            # -----------------------------------------------------------
            # Validation batch + metrics
            # -----------------------------------------------------------
            stage = "validation_batch"
            model.eval()
            with torch.no_grad():
                eval_loss, eval_outputs = api["SFTrainer"].compute_loss(
                    phase2_adapter,
                    model,
                    val_batch,
                    return_outputs=True,
                )
            result.eval_loss = finite_scalar(eval_loss, "Validation loss")
            eval_logits = parse_logits(eval_outputs)

            if not torch.isfinite(eval_logits).all():
                raise AssertionError("Validation logits contain NaN or infinity.")
            probabilities = torch.softmax(eval_logits, dim=-1)
            if not torch.allclose(
                probabilities.sum(dim=-1),
                torch.ones(probabilities.size(0), device=device),
                atol=1e-5,
            ):
                raise AssertionError("Softmax probabilities do not sum to one.")

            labels = val_batch["labels"]
            if labels.ndim > 1:
                labels = labels.argmax(dim=-1)

            # The corrected metric function must tolerate a batch missing classes.
            with contextlib.redirect_stdout(io.StringIO()):
                metrics = api["compute_metrics_multiclass"](
                    probabilities.detach().cpu().numpy(),
                    labels.detach().cpu().numpy(),
                    classes,
                )
            for metric_name in ("f1_macro", "accuracy"):
                if metric_name in metrics and not np.isfinite(metrics[metric_name]):
                    raise AssertionError(f"Metric {metric_name} is not finite.")

            # -----------------------------------------------------------
            # Save -> rebuild -> reload -> numerical equality
            # -----------------------------------------------------------
            stage = "checkpoint_roundtrip"
            checkpoint_path = exp_temp / "smoke_roundtrip.pth"
            torch.save(model.state_dict(), checkpoint_path)

            reference_logits = eval_logits.detach().cpu()
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            reloaded_model = api["build_model"](cfg.model).to(device)
            reloaded_model, _, _, _ = api["setup_phase2_modules"](
                reloaded_model,
                cfg,
                device,
            )
            checkpoint = torch.load(checkpoint_path, map_location=device)
            load_result = reloaded_model.load_state_dict(checkpoint, strict=False)

            critical_tokens = (
                "fusion.",
                "head.",
                "visual_resampler",
                "lora_A",
                "lora_B",
            )
            critical_missing = [
                key
                for key in load_result.missing_keys
                if any(token in key for token in critical_tokens)
            ]
            if critical_missing:
                raise AssertionError(
                    "Critical checkpoint parameters were not reloaded: "
                    + ", ".join(critical_missing[:12])
                )

            reloaded_model.eval()
            with torch.no_grad():
                _, reloaded_outputs = api["SFTrainer"].compute_loss(
                    phase2_adapter,
                    reloaded_model,
                    val_batch,
                    return_outputs=True,
                )
            reloaded_logits = parse_logits(reloaded_outputs).detach().cpu()

            result.checkpoint_reload_match = bool(
                torch.allclose(
                    reference_logits,
                    reloaded_logits,
                    atol=2e-4,
                    rtol=2e-4,
                )
            )
            if not result.checkpoint_reload_match:
                max_delta = float((reference_logits - reloaded_logits).abs().max())
                raise AssertionError(
                    f"Checkpoint reload changed logits; max |delta|={max_delta:.6g}."
                )

            del reloaded_model

        else:
            result.warnings.append("Phase 2 is disabled; classification checks were skipped.")

        # Parameter-regime assertions.
        if result.peft_type == "lora":
            if not any("lora" in name.lower() for name in initial_trainable_names):
                raise AssertionError(
                    f"{result.peft_type} config has no trainable LoRA parameter."
                )

        result.status = "PASS"

    except Exception as exc:
        result.status = "FAIL"
        result.error_stage = stage
        result.error = f"{type(exc).__name__}: {exc}"
        result.traceback = traceback.format_exc()

    finally:
        result.elapsed_seconds = round(time.perf_counter() - start, 3)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return result


# ---------------------------------------------------------------------------
# Global corrected-utility checks
# ---------------------------------------------------------------------------

def run_global_utility_checks(api: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    # Missing-class multiclass metrics + bootstrap argmax consistency.
    try:
        rng = np.random.RandomState(seed)
        num_classes = 4
        labels = np.array([0, 2, 3, 0, 2, 3, 0, 2], dtype=np.int64)
        logits = rng.normal(size=(len(labels), num_classes))
        logits[np.arange(len(labels)), labels] += 2.0
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
        names = [f"class_{i}" for i in range(num_classes)]

        with contextlib.redirect_stdout(io.StringIO()):
            metrics = api["compute_metrics_multiclass"](probs, labels, names)
            ci = api["bootstrap_confidence_intervals"](
                probs,
                labels,
                names,
                is_multilabel=False,
                n_bootstrap=25,
                seed=seed,
            )

        if "f1_macro" not in metrics:
            raise AssertionError("Multiclass metrics did not return f1_macro.")
        if "f1_macro" not in ci:
            raise AssertionError("Bootstrap did not return a Macro-F1 interval.")
        checks.append({"name": "metrics_and_bootstrap", "status": "PASS", "error": ""})
    except Exception as exc:
        checks.append(
            {
                "name": "metrics_and_bootstrap",
                "status": "FAIL",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )

    # Leakage-free OOD API and all scoring methods.
    try:
        rng = np.random.RandomState(seed + 1)
        dim = 16
        ref_labels = np.repeat(np.arange(3), 6)
        centers = rng.normal(size=(3, dim))
        ref_embeddings = np.vstack(
            [
                centers[label] + 0.15 * rng.normal(size=dim)
                for label in ref_labels
            ]
        )
        id_test = np.vstack(
            [
                centers[label] + 0.20 * rng.normal(size=dim)
                for label in np.repeat(np.arange(3), 3)
            ]
        )
        ood_test = 4.0 + rng.normal(size=(9, dim))

        detector = api["OODDetector"]().fit(ref_embeddings, ref_labels)
        id_maha = detector.score_mahalanobis(id_test)
        ood_maha = detector.score_mahalanobis(ood_test)
        id_knn = detector.score_knn(id_test, k=len(ref_embeddings))
        logits = rng.normal(size=(len(id_test), 3))
        energy = detector.score_energy(logits)

        for name, array in (
            ("id_maha", id_maha),
            ("ood_maha", ood_maha),
            ("id_knn", id_knn),
            ("energy", energy),
        ):
            if not np.isfinite(array).all():
                raise AssertionError(f"{name} contains NaN or infinity.")

        ood_metrics = api["evaluate_ood"](id_maha, ood_maha)
        for key in ("auroc", "aupr_out", "fpr_at_95tpr"):
            if key not in ood_metrics:
                raise AssertionError(f"OOD metrics are missing {key}.")
        checks.append({"name": "ood_scoring", "status": "PASS", "error": ""})
    except Exception as exc:
        checks.append(
            {
                "name": "ood_scoring",
                "status": "FAIL",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )

    return checks


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def write_reports(
    report_dir: Path,
    results: list[SmokeResult],
    missing: list[str],
    utility_checks: list[dict[str, Any]],
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "summary": {
            "total": len(results),
            "passed": sum(result.status == "PASS" for result in results),
            "failed": sum(result.status == "FAIL" for result in results),
            "missing_expected": missing,
            "utility_checks": utility_checks,
        },
        "results": [asdict(result) for result in results],
    }
    (report_dir / "smoke_results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    csv_fields = [
        "experiment_id",
        "config_file",
        "status",
        "error_stage",
        "error",
        "backbone_type",
        "peft_type",
        "fusion_type",
        "classifier_type",
        "phase1_enabled",
        "phase2_enabled",
        "high_res_enabled",
        "phase1_loss",
        "phase2_loss",
        "eval_loss",
        "logits_shape",
        "image_feature_shape",
        "text_feature_shape",
        "tile_shape",
        "trainable_backbone",
        "trainable_fusion",
        "trainable_head",
        "checkpoint_reload_match",
        "elapsed_seconds",
    ]
    with (report_dir / "smoke_results.csv").open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for result in results:
            row = asdict(result)
            writer.writerow({key: row.get(key) for key in csv_fields})

    lines = [
        "# XBone-Net smoke-test report",
        "",
        f"- Total resolved experiments: **{len(results)}**",
        f"- Passed: **{sum(result.status == 'PASS' for result in results)}**",
        f"- Failed: **{sum(result.status == 'FAIL' for result in results)}**",
        f"- Missing expected configs: **{len(missing)}**",
        "",
    ]

    if missing:
        lines.extend(["## Missing expected configs", ""])
        lines.extend(f"- `{item}`" for item in missing)
        lines.append("")

    lines.extend(
        [
            "## Experiment results",
            "",
            "| Experiment | Config | Status | Failed stage | Main error |",
            "|---|---|---:|---|---|",
        ]
    )
    for result in results:
        error = result.error.replace("|", "\\|").replace("\n", " ")[:180]
        lines.append(
            f"| `{result.experiment_id}` | `{result.config_file}` | "
            f"**{result.status}** | `{result.error_stage}` | {error} |"
        )

    lines.extend(["", "## Global utility checks", ""])
    for check in utility_checks:
        lines.append(
            f"- **{check['name']}**: {check['status']}"
            + (f" — {check['error']}" if check["error"] else "")
        )

    (report_dir / "smoke_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one-batch smoke tests for every XBone-Net experiment."
    )
    parser.add_argument("--config-root", default="configs", type=Path)
    parser.add_argument(
        "--manifest",
        default=Path("tools/smoke_manifest.yaml"),
        type=Path,
    )
    parser.add_argument(
        "--report-dir",
        default=Path("results/smoke_test"),
        type=Path,
    )
    parser.add_argument(
        "--work-dir",
        default=Path(".smoke_checkpoints"),
        type=Path,
    )
    parser.add_argument("--batch-size", default=2, type=int)
    parser.add_argument("--seed", default=123, type=int)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
    )
    parser.add_argument(
        "--allow-missing-configs",
        action="store_true",
        help="Do not fail the suite when a planned config is absent.",
    )
    parser.add_argument(
        "--stop-on-first-failure",
        action="store_true",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[ERROR] --device cuda was requested but CUDA is unavailable.")
        return 2
    if args.batch_size < 2:
        print("[ERROR] Use batch-size >= 2 because Phase-1 pairwise alignment needs a batch.")
        return 2

    project_root = Path.cwd()
    config_root = (project_root / args.config_root).resolve()
    manifest_path = (project_root / args.manifest).resolve()
    report_dir = (project_root / args.report_dir).resolve()
    work_dir = (project_root / args.work_dir).resolve()

    if not config_root.is_dir():
        print(f"[ERROR] Config root does not exist: {config_root}")
        return 2
    if not manifest_path.is_file():
        print(f"[ERROR] Manifest does not exist: {manifest_path}")
        return 2

    sys.path.insert(0, str(project_root))
    seed_everything(args.seed)

    manifest = load_manifest(manifest_path)
    resolved, missing = resolve_manifest_configs(config_root, manifest)

    print("=" * 78)
    print("XBONE-NET COMPLETE EXPERIMENT SMOKE TEST")
    print("=" * 78)
    print(f"Project root       : {project_root}")
    print(f"Config root        : {config_root}")
    print(f"Resolved configs   : {len(resolved)}")
    print(f"Missing configs    : {len(missing)}")
    print(f"Device             : {args.device}")
    print(f"Smoke batch size   : {args.batch_size}")
    print("=" * 78)

    if missing:
        print("\nMissing planned configurations:")
        for item in missing:
            print(f"  - {item}")

    try:
        api = import_project_api()
    except Exception as exc:
        print(f"\n[ERROR] {exc}")
        traceback.print_exc()
        return 2

    device = torch.device(args.device)
    work_dir.mkdir(parents=True, exist_ok=True)

    results: list[SmokeResult] = []
    for index, (experiment, config_file) in enumerate(resolved, start=1):
        experiment_id = experiment["id"]
        relative = config_file.relative_to(config_root)
        print(
            f"\n[{index:02d}/{len(resolved):02d}] "
            f"{experiment_id} <- {relative}"
        )
        result = smoke_one_experiment(
            experiment,
            config_file,
            config_root=config_root,
            device=device,
            batch_size=args.batch_size,
            seed=args.seed,
            work_root=work_dir,
            api=api,
        )
        results.append(result)

        if result.status == "PASS":
            print(
                f"  PASS | P1={result.phase1_loss} | "
                f"P2={result.phase2_loss} | eval={result.eval_loss} | "
                f"{result.elapsed_seconds:.1f}s"
            )
        else:
            print(
                f"  FAIL at {result.error_stage}: {result.error}"
            )
            if args.stop_on_first_failure:
                break

    print("\nRunning corrected metric/bootstrap/OOD utility checks...")
    utility_checks = run_global_utility_checks(api, args.seed)
    for check in utility_checks:
        print(f"  {check['status']:4s} | {check['name']} {check['error']}")

    write_reports(report_dir, results, missing, utility_checks)

    failures = sum(result.status == "FAIL" for result in results)
    utility_failures = sum(check["status"] == "FAIL" for check in utility_checks)
    missing_failure = bool(missing and not args.allow_missing_configs)

    print("\n" + "=" * 78)
    print("SMOKE TEST SUMMARY")
    print("=" * 78)
    print(f"PASS experiments : {sum(result.status == 'PASS' for result in results)}")
    print(f"FAIL experiments : {failures}")
    print(f"Missing configs  : {len(missing)}")
    print(f"Utility failures : {utility_failures}")
    print(f"Reports          : {report_dir}")
    print("=" * 78)

    suite_ok = failures == 0 and utility_failures == 0 and not missing_failure
    if suite_ok:
        print("\nALL SMOKE TESTS PASSED.")
        return 0

    print("\nSMOKE TEST SUITE FAILED. Open smoke_report.md for the first failed stage.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
