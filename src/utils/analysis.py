"""Locked post-hoc analysis helpers for the canonical CTCH proposed model.

The OOD and explainability pipelines use this module instead of the training
entry point.  It deliberately fixes the source experiment, checkpoint layout,
and Phase-2 architecture so an analysis run cannot silently evaluate an
ablation, a baseline, or freshly initialized weights.
"""

from __future__ import annotations

import datetime as _datetime
import hashlib
import json
import os
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_EXPERIMENT = "ctch/proposed/ours_xbone_net"
ZEROSHOT_BIOMEDCLIP_EXPERIMENT = (
    "ctch/baselines/zeroshot/biomedclip_zeroshot"
)
SOURCE_SEEDS = (42, 123, 456)
EXPECTED_NUM_CLASSES = 22
EXPECTED_TOTAL_PARAMETERS = 204_535_320
EXPECTED_TRAINABLE_PARAMETERS = 8_632_599

ANALYSIS_METADATA_KEYS = (
    "image_id",
    "group",
    "patient_id",
    "report_source_id",
    "scenario",
)


def seed_everything(seed: int) -> None:
    """Seed post-hoc analysis without enabling any training behavior."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def locked_checkpoint_path(seed: int) -> Path:
    if int(seed) not in SOURCE_SEEDS:
        raise ValueError(
            f"Analysis is locked to seeds {list(SOURCE_SEEDS)}, got {seed}."
        )
    return (
        PROJECT_ROOT
        / "checkpoints"
        / "ctch"
        / "proposed"
        / "ours_xbone_net"
        / f"seed_{int(seed)}"
        / "best_phase2.pth"
    )


def analysis_root(seed: Optional[int] = None) -> Path:
    root = PROJECT_ROOT / "results" / SOURCE_EXPERIMENT
    if seed is not None:
        root = root / f"seed_{int(seed)}"
    return root / "analysis"


def _absolute_dataset_paths(cfg: DictConfig) -> None:
    """Replace Hydra runtime interpolations for standalone analysis scripts."""
    params = cfg.dataset.params
    dataset_name = str(cfg.dataset.name)
    if dataset_name == "ctch":
        params.img_dir = str(PROJECT_ROOT / "data" / "CTCH" / "images")
        params.xray_report_dir = str(
            PROJECT_ROOT / "data" / "CTCH" / "reports" / "xray"
        )
        params.clinical_report_dir = str(
            PROJECT_ROOT / "data" / "CTCH" / "reports" / "clinical"
        )
        params.csv_split_path = str(PROJECT_ROOT / "data" / "CTCH" / "ctch-split.csv")
        params.csv_labels_path = str(PROJECT_ROOT / "data" / "CTCH" / "ctch-labels.csv")


def compose_source_config(seed: int) -> DictConfig:
    """Load the immutable evaluated config for the canonical CTCH checkpoint."""
    if int(seed) not in SOURCE_SEEDS:
        raise ValueError(
            f"Analysis is locked to seeds {list(SOURCE_SEEDS)}, got {seed}."
        )
    metrics_path = (
        PROJECT_ROOT
        / "results"
        / SOURCE_EXPERIMENT
        / f"seed_{int(seed)}"
        / "metrics.json"
    )
    if not metrics_path.is_file():
        raise FileNotFoundError(
            "Missing evaluated configuration for canonical analysis: "
            f"{metrics_path}"
        )
    metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    resolved = metrics_payload.get("config")
    if not isinstance(resolved, dict):
        raise ValueError(f"Evaluation result has no resolved config: {metrics_path}")
    cfg = OmegaConf.create(resolved)
    OmegaConf.set_struct(cfg, False)
    recorded_experiment = str(cfg.get("experiment_name", "")).strip("/")
    if recorded_experiment != SOURCE_EXPERIMENT:
        raise ValueError(
            f"Resolved config records {recorded_experiment!r}, expected "
            f"{SOURCE_EXPERIMENT!r}."
        )
    recorded_seed = int(cfg.get("seed", cfg.get("params", {}).get("seed", seed)))
    if recorded_seed != int(seed):
        raise ValueError(
            f"Resolved config records seed {recorded_seed}, expected {int(seed)}."
        )
    _absolute_dataset_paths(cfg)
    cfg.params.model_dir = str(locked_checkpoint_path(seed).parent)
    cfg.params.phase2.checkpoint_path = str(locked_checkpoint_path(seed))
    return cfg


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def adapt_state_dict_keys(
    state_dict: dict[str, torch.Tensor],
    model_keys: Iterable[str],
) -> dict[str, torch.Tensor]:
    """Align raw OpenCLIP checkpoint prefixes with the XBone wrapper."""
    model_keys = list(model_keys)
    checkpoint_keys = list(state_dict)
    if not checkpoint_keys:
        return state_dict
    model_has_backbone = any(key.startswith("backbone.model.") for key in model_keys)
    first_key = checkpoint_keys[0]
    if model_has_backbone and not first_key.startswith("backbone.model."):
        if first_key.startswith("model."):
            return {
                key.replace("model.", "backbone.model.", 1): value
                for key, value in state_dict.items()
            }
        if first_key.startswith(("visual.", "transformer.", "text.")):
            return {f"backbone.model.{key}": value for key, value in state_dict.items()}
    return state_dict


def _load_checkpoint_payload(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location=device)
    if not isinstance(payload, dict):
        raise TypeError(f"Checkpoint must contain a mapping, got {type(payload).__name__}.")
    return payload


def _validate_checkpoint_state(state_dict: dict[str, torch.Tensor]) -> None:
    centroid_items = [
        (key, value)
        for key, value in state_dict.items()
        if key.endswith("head.centroids") or key == "head.centroids"
    ]
    if len(centroid_items) != 1:
        raise RuntimeError(
            "Canonical Phase-2 checkpoint must contain exactly one head.centroids tensor."
        )
    key, centroids = centroid_items[0]
    expected = (EXPECTED_NUM_CLASSES, 512)
    if tuple(centroids.shape) != expected:
        raise RuntimeError(
            f"Checkpoint {key} has shape {tuple(centroids.shape)}, expected {expected}."
        )
    initialized = [
        value
        for key, value in state_dict.items()
        if key.endswith("head.centroids_initialized")
    ]
    if initialized and not bool(initialized[0].item()):
        raise RuntimeError("Checkpoint empirical centroids are not initialized.")


@dataclass(frozen=True)
class LoadedAnalysisModel:
    model: torch.nn.Module
    cfg: DictConfig
    checkpoint: Path
    checkpoint_sha256: str
    device: torch.device
    provenance: dict[str, Any]


def load_locked_proposed_model(
    seed: int,
    device: Optional[torch.device] = None,
    strict_fingerprint: bool = True,
) -> LoadedAnalysisModel:
    """Build and load the exact CTCH proposed Phase-2 model."""
    seed = int(seed)
    from src.models.builder import build_model, setup_phase2_modules

    seed_everything(seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = compose_source_config(seed)
    if str(cfg.experiment_name) != SOURCE_EXPERIMENT:
        raise RuntimeError(f"Unexpected source experiment: {cfg.experiment_name}")
    if str(cfg.model.backbone_type) != "biomedclip":
        raise RuntimeError("CTCH analysis requires the BiomedCLIP proposed backbone.")
    if str(cfg.model.visual_resampler.aggregation) != "passthrough":
        raise RuntimeError("CTCH analysis requires the passthrough visual resampler.")

    checkpoint = locked_checkpoint_path(seed).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Missing trained CTCH proposed checkpoint: {checkpoint}"
        )

    model = build_model(cfg.model).to(device)
    model, classifier_type, fusion_type, num_classes = setup_phase2_modules(
        model, cfg, device
    )
    if (classifier_type, fusion_type, int(num_classes)) != (
        "empirical_centroid",
        "cross_attention",
        EXPECTED_NUM_CLASSES,
    ):
        raise RuntimeError(
            "Canonical architecture mismatch: expected empirical_centroid + "
            f"cross_attention + 22 classes, got {classifier_type} + "
            f"{fusion_type} + {num_classes}."
        )

    payload = _load_checkpoint_payload(checkpoint, device)
    state_dict = payload.get("model_state_dict", payload)
    if not isinstance(state_dict, dict):
        raise TypeError("Checkpoint model_state_dict must be a mapping.")
    state_dict = adapt_state_dict_keys(state_dict, model.state_dict().keys())
    _validate_checkpoint_state(state_dict)
    result = model.load_state_dict(state_dict, strict=False)
    critical_tokens = ("lora_A", "lora_B", "visual_resampler")
    critical_prefixes = ("fusion.", "head.")
    critical_missing = [
        key
        for key in result.missing_keys
        if key.startswith(critical_prefixes)
        or any(token in key for token in critical_tokens)
    ]
    if critical_missing:
        raise RuntimeError(
            "Critical trained weights were not loaded:\n" + "\n".join(critical_missing[:30])
        )
    critical_unexpected = [
        key
        for key in result.unexpected_keys
        if key.startswith(critical_prefixes)
        or any(token in key for token in critical_tokens)
    ]
    if critical_unexpected:
        raise RuntimeError(
            "Checkpoint contains incompatible critical weights:\n"
            + "\n".join(critical_unexpected[:30])
        )

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if strict_fingerprint and (
        total_parameters != EXPECTED_TOTAL_PARAMETERS
        or trainable_parameters != EXPECTED_TRAINABLE_PARAMETERS
    ):
        raise RuntimeError(
            "Model fingerprint differs from the validated CTCH proposed architecture: "
            f"total={total_parameters:,}, trainable={trainable_parameters:,}; expected "
            f"{EXPECTED_TOTAL_PARAMETERS:,} and {EXPECTED_TRAINABLE_PARAMETERS:,}."
        )

    model.eval()
    resolved_config_yaml = OmegaConf.to_yaml(cfg, resolve=True, sort_keys=True)
    provenance = {
        "source_experiment": SOURCE_EXPERIMENT,
        "seed": seed,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "git_revision": _git_revision(),
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "num_classes": int(num_classes),
        "fusion_type": fusion_type,
        "classifier_type": classifier_type,
        "visual_resampler": str(cfg.model.visual_resampler.aggregation),
        "config_sha256": hashlib.sha256(
            resolved_config_yaml.encode("utf-8")
        ).hexdigest(),
        "resolved_config": OmegaConf.to_container(cfg, resolve=True),
        "created_at": _datetime.datetime.now(_datetime.timezone.utc).isoformat(),
    }
    return LoadedAnalysisModel(
        model=model,
        cfg=cfg,
        checkpoint=checkpoint,
        checkpoint_sha256=provenance["checkpoint_sha256"],
        device=device,
        provenance=provenance,
    )


def load_proposed_experiment_model(
    experiment_name: str,
    seed: int,
    device: Optional[torch.device] = None,
) -> LoadedAnalysisModel:
    """Load a proposed checkpoint from its immutable evaluated configuration.

    Unlike :func:`load_locked_proposed_model`, this helper is intended only for
    explicitly named cross-version audits.  It rebuilds the architecture from
    the resolved configuration stored beside the model's evaluation metrics,
    so later YAML edits cannot silently change the evaluated network.
    """
    experiment_name = str(experiment_name).strip("/")
    if not (
        experiment_name.startswith("ctch/proposed/ours_xbone_net")
        or experiment_name == "ctch/proposed/proposed_v6"
    ):
        raise ValueError(
            "Cross-version analysis only accepts CTCH proposed experiments."
        )
    seed = int(seed)
    metrics_path = (
        PROJECT_ROOT / "results" / experiment_name / f"seed_{seed}" / "metrics.json"
    )
    config_reference_path = metrics_path
    if metrics_path.is_file():
        config_reference = json.loads(metrics_path.read_text(encoding="utf-8"))
        resolved = config_reference.get("config")
    else:
        # Older cross-version exports retained the complete immutable config in
        # each feature-archive sidecar even when the original metrics.json was
        # later moved. Accept only a sidecar tied to this exact experiment and
        # seed; the checkpoint hash is verified below before the model is used.
        config_reference = None
        for candidate in sorted(PROJECT_ROOT.glob("results/**/ctch_train.json")):
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if (
                str(payload.get("source_experiment", "")).strip("/")
                == experiment_name
                and int(payload.get("seed", -1)) == seed
                and isinstance(payload.get("resolved_config"), dict)
            ):
                config_reference = payload
                config_reference_path = candidate
                break
        if config_reference is None:
            raise FileNotFoundError(
                f"Missing evaluated configuration for {experiment_name}: "
                f"{metrics_path}"
            )
        resolved = config_reference.get("resolved_config")
    if not isinstance(resolved, dict):
        raise ValueError(
            "Evaluation artifact has no resolved config: "
            f"{config_reference_path}"
        )
    cfg = OmegaConf.create(resolved)
    OmegaConf.set_struct(cfg, False)
    recorded_experiment = str(cfg.get("experiment_name", "")).strip("/")
    if recorded_experiment != experiment_name:
        raise ValueError(
            f"Resolved config records {recorded_experiment!r}, expected "
            f"{experiment_name!r}."
        )
    if str(cfg.dataset.name) != "ctch":
        raise ValueError("Proposed comparison requires the CTCH dataset config.")
    _absolute_dataset_paths(cfg)

    phase3_enabled = bool(
        (cfg.get("params", {}) or {}).get("phase3", {}).get("enabled", False)
    )
    checkpoint_name = "best_phase3.pth" if phase3_enabled else "best_phase2.pth"
    checkpoint = (
        PROJECT_ROOT
        / "checkpoints"
        / experiment_name
        / f"seed_{seed}"
        / checkpoint_name
    ).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing proposed checkpoint: {checkpoint}")
    recorded_checkpoint_hash = str(
        config_reference.get("checkpoint_sha256", "")
    ).strip()
    if recorded_checkpoint_hash:
        current_checkpoint_hash = sha256_file(checkpoint)
        if current_checkpoint_hash != recorded_checkpoint_hash:
            raise RuntimeError(
                "Checkpoint hash differs from the evaluated configuration "
                f"reference: {checkpoint}"
            )
    cfg.params.model_dir = str(checkpoint.parent)
    cfg.params.phase2.checkpoint_path = str(checkpoint)
    if phase3_enabled:
        cfg.params.phase3.checkpoint_path = str(checkpoint)

    from src.models.builder import (
        build_model,
        setup_phase2_modules,
        setup_phase3_modules,
    )

    seed_everything(seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg.model).to(device)
    model, classifier_type, fusion_type, num_classes = setup_phase2_modules(
        model, cfg, device
    )
    model, _ = setup_phase3_modules(model, cfg, device)
    if classifier_type != "empirical_centroid" or int(num_classes) != EXPECTED_NUM_CLASSES:
        raise RuntimeError(
            "Proposed comparison requires empirical_centroid with 22 classes; "
            f"got {classifier_type} with {num_classes}."
        )
    if fusion_type not in {"cross_attention", "gated_cross_attention"}:
        raise RuntimeError(f"Unsupported proposed fusion type: {fusion_type}")

    payload = _load_checkpoint_payload(checkpoint, device)
    state_dict = payload.get("model_state_dict", payload)
    if not isinstance(state_dict, dict):
        raise TypeError("Checkpoint model_state_dict must be a mapping.")
    state_dict = adapt_state_dict_keys(state_dict, model.state_dict().keys())
    _validate_checkpoint_state(state_dict)
    result = model.load_state_dict(state_dict, strict=False)
    critical_tokens = ("lora_A", "lora_B", "visual_resampler")
    critical_prefixes = ("fusion.", "head.", "drl_auxiliary.")
    critical_missing = [
        key for key in result.missing_keys
        if key.startswith(critical_prefixes)
        or any(token in key for token in critical_tokens)
    ]
    critical_unexpected = [
        key for key in result.unexpected_keys
        if key.startswith(critical_prefixes)
        or any(token in key for token in critical_tokens)
    ]
    if critical_missing or critical_unexpected:
        raise RuntimeError(
            "Checkpoint architecture mismatch. Missing critical keys: "
            f"{critical_missing[:20]}; unexpected critical keys: "
            f"{critical_unexpected[:20]}."
        )

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters()
        if parameter.requires_grad
    )
    recorded_metrics = config_reference.get("metrics", {})
    expected_total = recorded_metrics.get(
        "param_total",
        config_reference.get("total_parameters"),
    )
    expected_trainable = recorded_metrics.get(
        "param_trainable",
        config_reference.get("trainable_parameters"),
    )
    if expected_total is not None and total_parameters != int(expected_total):
        raise RuntimeError(
            f"Parameter fingerprint mismatch for {experiment_name}: "
            f"total={total_parameters:,}, evaluated={int(expected_total):,}."
        )
    if expected_trainable is not None and trainable_parameters != int(expected_trainable):
        raise RuntimeError(
            f"Trainable-parameter fingerprint mismatch for {experiment_name}: "
            f"current={trainable_parameters:,}, evaluated={int(expected_trainable):,}."
        )

    model.eval()
    resolved_config_yaml = OmegaConf.to_yaml(cfg, resolve=True, sort_keys=True)
    checkpoint_sha256 = sha256_file(checkpoint)
    provenance = {
        "source_experiment": experiment_name,
        "seed": seed,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "config_reference_path": str(config_reference_path.resolve()),
        "config_reference_sha256": sha256_file(config_reference_path),
        "metrics_path": (
            str(metrics_path.resolve()) if metrics_path.is_file() else None
        ),
        "git_revision": _git_revision(),
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "num_classes": int(num_classes),
        "fusion_type": fusion_type,
        "classifier_type": classifier_type,
        "visual_resampler": str(cfg.model.visual_resampler.aggregation),
        "config_sha256": hashlib.sha256(
            resolved_config_yaml.encode("utf-8")
        ).hexdigest(),
        "resolved_config": OmegaConf.to_container(cfg, resolve=True),
        "created_at": _datetime.datetime.now(_datetime.timezone.utc).isoformat(),
    }
    return LoadedAnalysisModel(
        model=model,
        cfg=cfg,
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha256,
        device=device,
        provenance=provenance,
    )


def load_evaluated_ctch_zeroshot_model(
    experiment_name: str,
    seed: int,
    device: Optional[torch.device] = None,
) -> LoadedAnalysisModel:
    """Rebuild the evaluated deterministic BioMedCLIP foundation baseline."""
    experiment_name = str(experiment_name).strip("/")
    seed = int(seed)
    if experiment_name != ZEROSHOT_BIOMEDCLIP_EXPERIMENT:
        raise ValueError(f"Unsupported zero-shot experiment: {experiment_name}")

    metrics_path = (
        PROJECT_ROOT / "results" / experiment_name / f"seed_{seed}" / "metrics.json"
    )
    if not metrics_path.is_file():
        raise FileNotFoundError(
            f"Missing evaluated zero-shot configuration: {metrics_path}"
        )
    metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    resolved = metrics_payload.get("config")
    if not isinstance(resolved, dict):
        raise ValueError(f"Evaluation result has no resolved config: {metrics_path}")
    cfg = OmegaConf.create(resolved)
    OmegaConf.set_struct(cfg, False)
    if str(cfg.get("experiment_name", "")).strip("/") != experiment_name:
        raise ValueError("Zero-shot metrics record a different experiment.")
    if str(cfg.dataset.name) != "ctch":
        raise ValueError("Zero-shot OOD baseline must use the CTCH dataset config.")
    _absolute_dataset_paths(cfg)
    # The original foundation baseline consumes one global 224x224 image only.
    # Supplying this field also lets shared CTCH/OOD dataset builders avoid
    # creating high-resolution tiles.
    cfg.dataset.params.high_res = {"enabled": False}

    from src.models.builder import build_model, setup_phase2_modules

    seed_everything(seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg.model).to(device)
    model, classifier_type, fusion_type, num_classes = setup_phase2_modules(
        model,
        cfg,
        device,
    )
    if classifier_type != "none" or fusion_type != "none":
        raise RuntimeError(
            "BioMedCLIP zero-shot baseline must not build a trained fusion/head."
        )
    if int(num_classes) != EXPECTED_NUM_CLASSES:
        raise RuntimeError(
            f"Expected {EXPECTED_NUM_CLASSES} CTCH classes, got {num_classes}."
        )
    model.eval()

    resolved_config_yaml = OmegaConf.to_yaml(cfg, resolve=True, sort_keys=True)
    config_sha256 = hashlib.sha256(
        resolved_config_yaml.encode("utf-8")
    ).hexdigest()
    foundation_fingerprint = hashlib.sha256(
        (
            f"official_pretrained_foundation:{experiment_name}:"
            f"{config_sha256}"
        ).encode("utf-8")
    ).hexdigest()
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    provenance = {
        "source_experiment": experiment_name,
        "analysis_scope": "ctch_zeroshot_ood",
        "seed": seed,
        "checkpoint": None,
        # Shared OOD validation historically calls this field a checkpoint hash.
        # For a checkpoint-free baseline it is a deterministic fingerprint of
        # the official foundation source and the evaluated resolved config.
        "checkpoint_sha256": foundation_fingerprint,
        "weight_source": "official_pretrained_foundation",
        "metrics_path": str(metrics_path.resolve()),
        "metrics_sha256": sha256_file(metrics_path),
        "git_revision": _git_revision(),
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "num_classes": int(num_classes),
        "fusion_type": fusion_type,
        "classifier_type": classifier_type,
        "config_sha256": config_sha256,
        "resolved_config": OmegaConf.to_container(cfg, resolve=True),
        "created_at": _datetime.datetime.now(_datetime.timezone.utc).isoformat(),
    }
    return LoadedAnalysisModel(
        model=model,
        cfg=cfg,
        checkpoint=metrics_path.resolve(),
        checkpoint_sha256=foundation_fingerprint,
        device=device,
        provenance=provenance,
    )


def load_evaluated_ctch_model(
    experiment_name: str,
    seed: int,
    device: Optional[torch.device] = None,
) -> LoadedAnalysisModel:
    """Load a canonical, zero-shot, or ablation CTCH evaluated model.

    This entry point is intentionally narrower than the training CLI.  It only
    accepts the canonical proposed model and experiments below
    ``ctch/ablation_study``.  The architecture is reconstructed from the
    resolved config embedded in ``metrics.json`` and checked against the saved
    parameter fingerprint before features are exported.
    """
    experiment_name = str(experiment_name).strip("/")
    seed = int(seed)
    if experiment_name == SOURCE_EXPERIMENT:
        return load_locked_proposed_model(
            seed,
            device=device,
            strict_fingerprint=True,
        )
    if experiment_name == ZEROSHOT_BIOMEDCLIP_EXPERIMENT:
        return load_evaluated_ctch_zeroshot_model(
            experiment_name,
            seed,
            device=device,
        )
    if not experiment_name.startswith("ctch/ablation_study/"):
        raise ValueError(
            "Representation analysis accepts only the canonical CTCH model, "
            "the BioMedCLIP zero-shot baseline, or experiments below "
            "'ctch/ablation_study/'."
        )

    metrics_path = (
        PROJECT_ROOT / "results" / experiment_name / f"seed_{seed}" / "metrics.json"
    )
    if not metrics_path.is_file():
        raise FileNotFoundError(
            f"Missing evaluated configuration for {experiment_name}: {metrics_path}"
        )
    metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    resolved = metrics_payload.get("config")
    if not isinstance(resolved, dict):
        raise ValueError(f"Evaluation result has no resolved config: {metrics_path}")
    cfg = OmegaConf.create(resolved)
    OmegaConf.set_struct(cfg, False)
    recorded_experiment = str(cfg.get("experiment_name", "")).strip("/")
    if recorded_experiment != experiment_name:
        raise ValueError(
            f"Resolved config records {recorded_experiment!r}, expected "
            f"{experiment_name!r}."
        )
    recorded_seed = int(cfg.get("seed", cfg.get("params", {}).get("seed", seed)))
    if recorded_seed != seed:
        raise ValueError(
            f"Resolved config records seed {recorded_seed}, expected {seed}."
        )
    if str(cfg.dataset.name) != "ctch":
        raise ValueError("Ablation OOD analysis requires the CTCH dataset config.")
    _absolute_dataset_paths(cfg)

    checkpoint = (
        PROJECT_ROOT
        / "checkpoints"
        / experiment_name
        / f"seed_{seed}"
        / "best_phase2.pth"
    ).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing ablation checkpoint: {checkpoint}")
    cfg.params.model_dir = str(checkpoint.parent)
    cfg.params.phase2.checkpoint_path = str(checkpoint)

    from src.models.builder import build_model, setup_phase2_modules

    seed_everything(seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg.model).to(device)
    model, classifier_type, fusion_type, num_classes = setup_phase2_modules(
        model, cfg, device
    )
    if int(num_classes) != EXPECTED_NUM_CLASSES:
        raise RuntimeError(
            f"CTCH ablation must have {EXPECTED_NUM_CLASSES} classes, "
            f"got {num_classes}."
        )

    payload = _load_checkpoint_payload(checkpoint, device)
    state_dict = payload.get("model_state_dict", payload)
    if not isinstance(state_dict, dict):
        raise TypeError("Checkpoint model_state_dict must be a mapping.")
    state_dict = adapt_state_dict_keys(state_dict, model.state_dict().keys())
    if classifier_type == "empirical_centroid":
        _validate_checkpoint_state(state_dict)
    result = model.load_state_dict(state_dict, strict=False)
    critical_tokens = ("lora_A", "lora_B", "visual_resampler")
    critical_prefixes = ("fusion.", "head.")
    critical_missing = [
        key
        for key in result.missing_keys
        if key.startswith(critical_prefixes)
        or any(token in key for token in critical_tokens)
    ]
    critical_unexpected = [
        key
        for key in result.unexpected_keys
        if key.startswith(critical_prefixes)
        or any(token in key for token in critical_tokens)
    ]
    if critical_missing or critical_unexpected:
        raise RuntimeError(
            "Ablation checkpoint architecture mismatch. Missing critical keys: "
            f"{critical_missing[:20]}; unexpected critical keys: "
            f"{critical_unexpected[:20]}."
        )

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    recorded_metrics = metrics_payload.get("metrics", {})
    expected_total = recorded_metrics.get("param_total")
    expected_trainable = recorded_metrics.get("param_trainable")
    if expected_total is not None and total_parameters != int(expected_total):
        raise RuntimeError(
            f"Parameter fingerprint mismatch for {experiment_name}: "
            f"total={total_parameters:,}, evaluated={int(expected_total):,}."
        )
    if expected_trainable is not None and trainable_parameters != int(
        expected_trainable
    ):
        raise RuntimeError(
            f"Trainable-parameter fingerprint mismatch for {experiment_name}: "
            f"current={trainable_parameters:,}, "
            f"evaluated={int(expected_trainable):,}."
        )

    model.eval()
    resolved_config_yaml = OmegaConf.to_yaml(cfg, resolve=True, sort_keys=True)
    checkpoint_sha256 = sha256_file(checkpoint)
    provenance = {
        "source_experiment": experiment_name,
        "analysis_scope": "ctch_ablation_ood",
        "seed": seed,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "metrics_path": str(metrics_path.resolve()),
        "metrics_sha256": sha256_file(metrics_path),
        "git_revision": _git_revision(),
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "num_classes": int(num_classes),
        "fusion_type": fusion_type,
        "classifier_type": classifier_type,
        "visual_resampler": str(
            cfg.model.get("visual_resampler", {}).get("aggregation", "none")
        ),
        "config_sha256": hashlib.sha256(
            resolved_config_yaml.encode("utf-8")
        ).hexdigest(),
        "resolved_config": OmegaConf.to_container(cfg, resolve=True),
        "created_at": _datetime.datetime.now(_datetime.timezone.utc).isoformat(),
    }
    return LoadedAnalysisModel(
        model=model,
        cfg=cfg,
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha256,
        device=device,
        provenance=provenance,
    )


class AnalysisDataCollator:
    """Use the training collator while preserving sample provenance fields."""

    def __init__(self, tokenizer) -> None:
        from src.utils.trainer import BioMedCLIPDataCollator, resolve_pad_token_id

        pad_id = resolve_pad_token_id(tokenizer) if tokenizer is not None else 0
        self.base = BioMedCLIPDataCollator(pad_token_id=pad_id)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        metadata = {
            key: [feature.get(key, "") for feature in features]
            for key in ANALYSIS_METADATA_KEYS
            if any(key in feature for feature in features)
        }
        batch = self.base(features)
        batch.update(metadata)
        return batch


class MetadataDataset(Dataset):
    """Add stable IDs/groups to a standard repository dataset."""

    def __init__(self, dataset: Dataset, scenario: str) -> None:
        self.dataset = dataset
        self.scenario = str(scenario)
        self.df = getattr(dataset, "df", None)

    def __len__(self) -> int:
        return len(self.dataset)

    @staticmethod
    def _as_dict(sample: Any) -> dict[str, Any]:
        if isinstance(sample, dict):
            return dict(sample)
        if isinstance(sample, tuple) and len(sample) == 4:
            return {
                "pixel_values": sample[0],
                "xray_input_ids": sample[1],
                "clinical_input_ids": sample[2],
                "labels": sample[3],
            }
        if isinstance(sample, tuple) and len(sample) == 3:
            return {
                "pixel_values": sample[0],
                "input_ids": sample[1],
                "labels": sample[2],
            }
        raise TypeError(f"Unsupported dataset sample type: {type(sample).__name__}")

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self._as_dict(self.dataset[index])
        row = self.df.iloc[index] if self.df is not None else {}
        image_id = str(row.get("image_id", index))
        group = row.get("class_id", row.get("fractured", ""))
        patient = ""
        for candidate in (
            "patient_key",
            "patient_id",
            "Mã bệnh nhân",
            "MÃ BỆNH NHÂN",
        ):
            if candidate in row:
                patient = str(row[candidate])
                break
        sample.update(
            {
                "image_id": image_id,
                "group": str(group),
                "patient_id": patient,
                "report_source_id": image_id,
                "scenario": self.scenario,
            }
        )
        return sample


def build_analysis_loader(
    dataset: Dataset,
    tokenizer,
    batch_size: int,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
        collate_fn=AnalysisDataCollator(tokenizer),
    )


def _to_device(value: Any, device: torch.device) -> Any:
    return value.to(device) if isinstance(value, torch.Tensor) else value


def _attention_distribution_statistics(
    attention: Optional[torch.Tensor],
    key_padding_mask: Optional[torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Summarize per-sample attention concentration without retaining maps."""
    if attention is None:
        return {}
    from src.models.fusion.cross_attention import reduce_attention_to_keys

    probability = reduce_attention_to_keys(attention, key_padding_mask)
    if key_padding_mask is None:
        valid_count = torch.full(
            (probability.size(0),),
            probability.size(1),
            device=probability.device,
            dtype=probability.dtype,
        )
    else:
        valid_count = (~key_padding_mask).sum(dim=1).to(probability.dtype)
    entropy = -(
        probability * probability.clamp_min(1e-12).log()
    ).sum(dim=1)
    entropy_denominator = valid_count.clamp_min(2.0).log()
    normalized_entropy = torch.where(
        valid_count > 1,
        entropy / entropy_denominator,
        torch.zeros_like(entropy),
    )
    top_k = min(3, probability.size(1))
    return {
        "entropy_normalized": normalized_entropy,
        "top1_mass": probability.max(dim=1).values,
        "top3_mass": probability.topk(top_k, dim=1).values.sum(dim=1),
        "effective_tokens": entropy.exp(),
    }


def forward_analysis_batch(
    model: torch.nn.Module,
    batch: dict[str, Any],
    device: torch.device,
    report_type: str = "clinical",
) -> dict[str, torch.Tensor]:
    """Export modality and fusion representations through the trained path."""
    images = _to_device(batch["pixel_values"], device)
    tile_values = _to_device(batch.get("tile_values"), device)
    tile_mask = _to_device(batch.get("tile_mask"), device)
    tile_boxes = _to_device(batch.get("tile_boxes"), device)
    prefix = "xray" if report_type == "xray" else "clinical"
    input_ids = _to_device(batch.get(f"{prefix}_input_ids"), device)
    attention_mask = _to_device(batch.get(f"{prefix}_attention_mask"), device)

    image_tokens, text_tokens = model.backbone(
        images,
        input_ids,
        attention_mask=attention_mask,
        tile_values=tile_values,
        tile_mask=tile_mask,
        tile_boxes=tile_boxes,
    )
    if image_tokens.ndim != 3 or text_tokens is None or text_tokens.ndim != 3:
        raise RuntimeError(
            "CTCH proposed analysis expects local image and text token sequences."
        )

    image_padding = getattr(model.backbone, "last_image_key_padding_mask", None)
    image_local_padding = image_padding[:, 1:] if image_padding is not None else None
    text_local_padding = attention_mask[:, 1:] == 0 if attention_mask is not None else None
    fused, details = model.fusion(
        image_tokens,
        text_tokens,
        img_key_padding_mask=image_local_padding,
        txt_key_padding_mask=text_local_padding,
        return_attn=True,
    )
    logits = model.head(fused)

    text_attention_stats = _attention_distribution_statistics(
        details.get("attn_img_to_txt"), text_local_padding
    )
    visual_attention_stats = _attention_distribution_statistics(
        details.get("attn_txt_to_img"), image_local_padding
    )

    if image_tokens.size(1) > 1:
        local = image_tokens[:, 1:]
        if image_local_padding is None:
            visual_local_summary = local.mean(dim=1)
        else:
            valid = (~image_local_padding).unsqueeze(-1).to(local.dtype)
            visual_local_summary = (local * valid).sum(dim=1) / valid.sum(
                dim=1
            ).clamp_min(1.0)
    else:
        visual_local_summary = image_tokens[:, 0]

    output = {
        "fused_embeddings_raw": fused,
        "fused_embeddings": F.normalize(fused, dim=-1),
        "label_discriminative_embeddings": F.normalize(fused, dim=-1),
        "visual_global_embeddings": F.normalize(image_tokens[:, 0], dim=-1),
        "visual_local_summary_embeddings": F.normalize(
            visual_local_summary, dim=-1
        ),
        "text_global_embeddings": F.normalize(text_tokens[:, 0], dim=-1),
        "image_from_text_embeddings": F.normalize(
            details["txt_context_for_image"], dim=-1
        ),
        "text_from_image_embeddings": F.normalize(
            details["img_context_for_text"], dim=-1
        ),
        "logits": logits,
        "probabilities": torch.softmax(logits, dim=-1),
    }
    if getattr(model, "drl_auxiliary", None) is not None:
        from src.models.drl import drl_ood_score

        auxiliary_logits, distribution_features, auxiliary_details = model.drl_auxiliary(
            image_tokens,
            text_tokens,
            fused,
            image_local_padding_mask=image_local_padding,
            return_details=True,
        )
        output.update({
            "distribution_discriminative_embeddings": F.normalize(
                distribution_features, dim=-1
            ),
            "drl_auxiliary_logits": auxiliary_logits,
            "drl_auxiliary_probabilities": torch.softmax(auxiliary_logits, dim=-1),
            "drl_ood_scores": drl_ood_score(logits, auxiliary_logits),
            "drl_component_weights": auxiliary_details["component_weights"],
        })
    output.update({
        f"attention_text_{key}": value
        for key, value in text_attention_stats.items()
    })
    output.update({
        f"attention_visual_{key}": value
        for key, value in visual_attention_stats.items()
    })
    output["attention_context_cosine"] = F.cosine_similarity(
        details["txt_context_for_image"],
        details["img_context_for_text"],
        dim=-1,
    )
    if "fusion_gate" in details:
        output["fusion_gate"] = details["fusion_gate"].squeeze(-1)
    return output


def save_feature_archive(
    output: Path,
    arrays: dict[str, np.ndarray],
    provenance: dict[str, Any],
) -> Path:
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(arrays)
    payload["provenance_json"] = np.asarray(
        json.dumps(provenance, ensure_ascii=False, sort_keys=True)
    )
    np.savez_compressed(output, **payload)
    output.with_suffix(".json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return output


def load_feature_archive(
    path: Path,
    expected_source_experiment: str = SOURCE_EXPERIMENT,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {
            key: archive[key]
            for key in archive.files
            if key != "provenance_json"
        }
        if "provenance_json" not in archive.files:
            raise ValueError(f"Feature archive has no provenance_json: {path}")
        provenance = json.loads(str(archive["provenance_json"].item()))
    expected_source_experiment = str(expected_source_experiment).strip("/")
    if provenance.get("source_experiment") != expected_source_experiment:
        raise ValueError(
            f"Archive source is not {expected_source_experiment}: {path}"
        )
    return arrays, provenance


def collect_feature_batches(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    report_type: str = "clinical",
) -> dict[str, np.ndarray]:
    tensor_parts: dict[str, list[np.ndarray]] = {}
    metadata_parts: dict[str, list[str]] = {
        key: [] for key in ANALYSIS_METADATA_KEYS
    }
    labels: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            output = forward_analysis_batch(
                model, batch, device=device, report_type=report_type
            )
            for key, value in output.items():
                tensor_parts.setdefault(key, []).append(value.detach().cpu().numpy())
            label_values = batch["labels"]
            labels.append(label_values.detach().cpu().numpy())
            for key in ANALYSIS_METADATA_KEYS:
                values = batch.get(key, [""] * len(label_values))
                metadata_parts[key].extend(str(value) for value in values)

    if not labels:
        raise RuntimeError("Feature export received an empty dataset.")
    arrays = {
        key: np.concatenate(parts, axis=0)
        for key, parts in tensor_parts.items()
    }
    arrays["labels"] = np.concatenate(labels, axis=0)
    for key, values in metadata_parts.items():
        arrays[key] = np.asarray(values, dtype=str)
    arrays["predictions"] = arrays["logits"].argmax(axis=1).astype(np.int64)
    return arrays
