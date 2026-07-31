"""Export CTCH/OOD features for an evaluated representation baseline.

Unlike ``export_analysis_features.py``, this script is not locked to the
canonical proposed architecture. It accepts the canonical CTCH model, the
deterministic BioMedCLIP zero-shot baseline, or an experiment below
``ctch/ablation_study`` and reconstructs the network from the resolved
configuration saved in that experiment's ``metrics.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

from src.utils.analysis import (
    ANALYSIS_METADATA_KEYS,
    build_analysis_loader,
    load_evaluated_ctch_model,
    load_feature_archive,
    save_feature_archive,
)
from src.utils.prompts import generate_clip_class_prompts
from tools.export_analysis_features import build_scenario_dataset


SCENARIOS = (
    "ctch_train",
    "ctch_val",
    "ctch_test",
    "ctch_ood",
    "btxrd_test",
)


def experiment_analysis_root(experiment: str, seed: int) -> Path:
    return ROOT / "results" / experiment / f"seed_{int(seed)}" / "analysis"


def _to_device(value: Any, device: torch.device) -> Any:
    return value.to(device) if isinstance(value, torch.Tensor) else value


def _collect_fused_feature_batches(
    model: torch.nn.Module,
    loader,
    *,
    device: torch.device,
    report_type: str,
) -> dict[str, np.ndarray]:
    """Collect a representation shared by multimodal and unimodal ablations."""
    tensor_parts: dict[str, list[np.ndarray]] = {}
    metadata_parts: dict[str, list[str]] = {
        key: [] for key in ANALYSIS_METADATA_KEYS
    }
    labels: list[np.ndarray] = []
    prefix = "xray" if report_type == "xray" else "clinical"
    model.eval()

    with torch.no_grad():
        for batch in loader:
            images = _to_device(batch["pixel_values"], device)
            tile_values = _to_device(batch.get("tile_values"), device)
            tile_mask = _to_device(batch.get("tile_mask"), device)
            tile_boxes = _to_device(batch.get("tile_boxes"), device)
            input_ids = _to_device(batch.get(f"{prefix}_input_ids"), device)
            attention_mask = _to_device(
                batch.get(f"{prefix}_attention_mask"),
                device,
            )

            encoded = model._encode_modalities(
                images,
                input_ids=input_ids,
                attention_mask=attention_mask,
                tile_values=tile_values,
                tile_mask=tile_mask,
                tile_boxes=tile_boxes,
            )
            (
                image_features,
                text_features,
                full_image_padding,
                image_local_padding,
                text_local_padding,
            ) = encoded
            fused = model._fuse_modalities(
                image_features,
                text_features,
                full_image_padding,
                image_local_padding,
                text_local_padding,
            )
            logits = model.head(fused)
            outputs = {
                "fused_embeddings_raw": fused,
                "fused_embeddings": F.normalize(fused, dim=-1),
                "logits": logits,
                "probabilities": torch.softmax(logits, dim=-1),
            }
            if image_features is not None:
                image_global = (
                    image_features[:, 0]
                    if image_features.ndim == 3
                    else image_features
                )
                outputs["visual_global_embeddings"] = F.normalize(
                    image_global,
                    dim=-1,
                )
            if text_features is not None:
                text_global = (
                    text_features[:, 0]
                    if text_features.ndim == 3
                    else text_features
                )
                outputs["text_global_embeddings"] = F.normalize(
                    text_global,
                    dim=-1,
                )

            for key, value in outputs.items():
                tensor_parts.setdefault(key, []).append(
                    value.detach().cpu().numpy()
                )
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


def _collect_zeroshot_feature_batches(
    model: torch.nn.Module,
    loader,
    *,
    device: torch.device,
    prompt_classes: list[str],
    temperature: float,
) -> dict[str, np.ndarray]:
    """Export original BioMedCLIP global-image features and prompt logits."""
    if temperature <= 0:
        raise ValueError("Zero-shot temperature must be positive.")
    tokenizer = model.backbone.tokenizer_obj
    prompt_dict = generate_clip_class_prompts(prompt_classes)
    prompt_batch = tokenizer(list(prompt_dict.values()))
    prompt_attention = None
    if isinstance(prompt_batch, dict):
        prompt_attention = prompt_batch.get("attention_mask")
        prompt_ids = prompt_batch["input_ids"]
    else:
        prompt_ids = prompt_batch
    prompt_ids = prompt_ids.to(device)
    if prompt_attention is not None:
        prompt_attention = prompt_attention.to(device)

    text_encoder = getattr(model.backbone, "encode_text", None)
    if text_encoder is None:
        text_encoder = getattr(model.backbone.model, "encode_text", None)
    image_encoder = getattr(model.backbone.model, "encode_image", None)
    if text_encoder is None or image_encoder is None:
        raise RuntimeError("BioMedCLIP must expose encode_image and encode_text.")
    with torch.no_grad():
        try:
            prompt_features = text_encoder(
                prompt_ids,
                attention_mask=prompt_attention,
            )
        except TypeError:
            prompt_features = text_encoder(prompt_ids)
        prompt_features = F.normalize(prompt_features, dim=-1)

    tensor_parts: dict[str, list[np.ndarray]] = {}
    metadata_parts: dict[str, list[str]] = {
        key: [] for key in ANALYSIS_METADATA_KEYS
    }
    labels: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            images = _to_device(batch["pixel_values"], device)
            raw_image_features = image_encoder(images)
            image_features = F.normalize(raw_image_features, dim=-1)
            logits = image_features @ prompt_features.T / float(temperature)
            outputs = {
                "fused_embeddings_raw": raw_image_features,
                "fused_embeddings": image_features,
                "visual_global_embeddings": image_features,
                "logits": logits,
                "probabilities": torch.softmax(logits, dim=-1),
            }
            for key, value in outputs.items():
                tensor_parts.setdefault(key, []).append(
                    value.detach().cpu().numpy()
                )
            label_values = batch["labels"]
            labels.append(label_values.detach().cpu().numpy())
            for key in ANALYSIS_METADATA_KEYS:
                values = batch.get(key, [""] * len(label_values))
                metadata_parts[key].extend(str(value) for value in values)

    if not labels:
        raise RuntimeError("Zero-shot feature export received an empty dataset.")
    arrays = {
        key: np.concatenate(parts, axis=0)
        for key, parts in tensor_parts.items()
    }
    arrays["labels"] = np.concatenate(labels, axis=0)
    arrays["predictions"] = arrays["logits"].argmax(axis=1).astype(np.int64)
    for key, values in metadata_parts.items():
        arrays[key] = np.asarray(values, dtype=str)
    return arrays


def _can_resume(
    path: Path,
    *,
    experiment: str,
    seed: int,
    checkpoint_sha256: str,
    config_sha256: str,
    scenario: str,
    allow_incomplete_ood: bool,
) -> bool:
    if not path.is_file():
        return False
    try:
        arrays, provenance = load_feature_archive(
            path,
            expected_source_experiment=experiment,
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    if "fused_embeddings_raw" not in arrays:
        return False
    return (
        int(provenance.get("seed", -1)) == int(seed)
        and provenance.get("scenario") == scenario
        and provenance.get("checkpoint_sha256") == checkpoint_sha256
        and provenance.get("config_sha256") == config_sha256
        and (
            scenario != "ctch_ood"
            or bool(provenance.get("allow_incomplete_ood", False))
            == bool(allow_incomplete_ood)
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=SCENARIOS,
        default=list(SCENARIOS),
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--allow-incomplete-ood", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    args = parser.parse_args()

    experiment = str(args.experiment).strip("/")
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable.")
        device = torch.device(args.device)

    loaded = load_evaluated_ctch_model(
        experiment,
        args.seed,
        device=device,
    )
    feature_root = experiment_analysis_root(
        experiment,
        args.seed,
    ) / "features"
    feature_root.mkdir(parents=True, exist_ok=True)
    print(
        f"[Source] {experiment} seed={args.seed} "
        f"sha256={loaded.checkpoint_sha256[:12]} device={device}"
    )

    for scenario in args.scenarios:
        output = feature_root / f"{scenario}.npz"
        if (
            not args.overwrite
            and _can_resume(
                output,
                experiment=experiment,
                seed=args.seed,
                checkpoint_sha256=loaded.checkpoint_sha256,
                config_sha256=str(loaded.provenance["config_sha256"]),
                scenario=scenario,
                allow_incomplete_ood=args.allow_incomplete_ood,
            )
        ):
            print(f"[Resume] {scenario}: verified archive already exists.")
            continue

        print(f"[Export] {scenario}")
        dataset, scenario_metadata = build_scenario_dataset(
            scenario,
            loaded,
            allow_incomplete_ood=args.allow_incomplete_ood,
            mismatch_seed=2025,
        )
        loader = build_analysis_loader(
            dataset,
            tokenizer=loaded.model.backbone.tokenizer_obj,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        if loaded.provenance.get("classifier_type") == "none":
            arrays = _collect_zeroshot_feature_batches(
                loaded.model,
                loader,
                device=loaded.device,
                prompt_classes=list(
                    loaded.cfg.dataset.params.get(
                        "prompt_classes",
                        loaded.cfg.dataset.params.classes,
                    )
                ),
                temperature=float(loaded.cfg.params.get("temperature", 0.07)),
            )
        else:
            arrays = _collect_fused_feature_batches(
                loaded.model,
                loader,
                device=loaded.device,
                report_type=str(loaded.cfg.params.phase2.p2_report_type),
            )
        provenance = {
            **loaded.provenance,
            "scenario": scenario,
            "sample_count": int(len(arrays["labels"])),
            "allow_incomplete_ood": bool(args.allow_incomplete_ood),
            "feature_dimensions": {
                key: list(value.shape)
                for key, value in arrays.items()
                if "embeddings" in key
                or key in {"logits", "probabilities"}
            },
            **scenario_metadata,
        }
        save_feature_archive(output, arrays, provenance)
        print(f"  -> {output} ({len(dataset)} samples)")


if __name__ == "__main__":
    main()
