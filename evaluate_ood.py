"""Evaluate XBone-Net out-of-distribution detection without reference leakage."""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from typing import Optional

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from src.utils.ood import OODDetector, evaluate_ood


def load_embeddings(path: str) -> dict:
    data = np.load(path, allow_pickle=True)
    embedding_key = "fused_embeddings" if "fused_embeddings" in data.files else "image_embeddings"
    result = {
        "image_embeddings": data[embedding_key],
        "labels": data["labels"],
    }
    if "logits" in data.files:
        result["logits"] = data["logits"]
    if "probabilities" in data.files:
        result["probabilities"] = data["probabilities"]
    text_keys = [key for key in data.files if key.startswith("text_embeddings_")]
    if text_keys:
        result["text_embeddings"] = {
            key.replace("text_embeddings_", ""): data[key] for key in text_keys
        }
    return result


def _integer_labels(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    if labels.ndim == 2:
        labels = np.argmax(labels, axis=1)
    return labels.astype(np.int64).reshape(-1)


def _normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    embeddings = np.asarray(embeddings, dtype=np.float64)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.maximum(norms, 1e-12)


def split_reference_and_id_test(
    embeddings: np.ndarray,
    labels: np.ndarray,
    logits: Optional[np.ndarray] = None,
    ref_fraction: float = 0.5,
    seed: int = 42,
):
    """Create disjoint stratified reference and ID-test sets."""
    labels = _integer_labels(labels)
    counts = np.bincount(labels)
    present_counts = counts[counts > 0]
    if present_counts.size == 0 or np.any(present_counts < 2):
        raise ValueError(
            "A leakage-free stratified split requires at least two samples per present class."
        )
    if not 0.0 < ref_fraction < 1.0:
        raise ValueError("ref_fraction must be between 0 and 1.")

    indices = np.arange(len(labels))
    ref_idx, test_idx = train_test_split(
        indices,
        train_size=ref_fraction,
        random_state=seed,
        stratify=labels,
    )
    return {
        "ref_embeddings": embeddings[ref_idx],
        "ref_labels": labels[ref_idx],
        "id_test_embeddings": embeddings[test_idx],
        "id_test_labels": labels[test_idx],
        "id_test_logits": logits[test_idx] if logits is not None else None,
    }


def stratified_reference_subset(
    embeddings: np.ndarray,
    labels: np.ndarray,
    n_ref: int,
    seed: int = 42,
):
    labels = _integer_labels(labels)
    if n_ref <= 0 or n_ref >= len(labels):
        return embeddings, labels
    n_classes = len(np.unique(labels))
    if n_ref < n_classes:
        raise ValueError(
            f"n_ref={n_ref} is smaller than the {n_classes} reference classes."
        )
    indices = np.arange(len(labels))
    subset_idx, _ = train_test_split(
        indices,
        train_size=n_ref,
        random_state=seed,
        stratify=labels,
    )
    return embeddings[subset_idx], labels[subset_idx]


def load_prototypes_from_checkpoint(checkpoint_path: Optional[str]):
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return None
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    for key, value in state_dict.items():
        if key.endswith("head.prototypes") or "head.prototypes" in key:
            prototypes = value.detach().cpu().numpy()
            print(f"Loaded prototypes {prototypes.shape} from '{key}'.")
            return prototypes
    print("[Warning] No prototype tensor was found in the checkpoint.")
    return None


def encode_text_anchors(
    checkpoint_path: str,
    config_path: str,
    pathologies: list[str],
    device: torch.device,
    image_context: str = "a bone x-ray",
) -> np.ndarray:
    """Encode anchors using the exact model/PEFT configuration used in training."""
    from omegaconf import OmegaConf
    from src.models.builder import build_model

    if not config_path or not os.path.exists(config_path):
        raise FileNotFoundError(
            "Text-anchor evaluation requires --config pointing to the training YAML."
        )
    cfg = OmegaConf.load(config_path)
    model_cfg = cfg.model if "model" in cfg else cfg
    model = build_model(model_cfg).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    result = model.load_state_dict(state_dict, strict=False)
    critical_missing = [
        key for key in result.missing_keys
        if "lora_A" in key or "lora_B" in key
    ]
    if critical_missing:
        raise RuntimeError(
            "Text-anchor model did not load PEFT weights:\n"
            + "\n".join(critical_missing[:20])
        )

    model.eval()
    tokenizer = getattr(model.backbone, "tokenizer", None)
    encoder = getattr(getattr(model.backbone, "model", None), "encode_text", None)
    if tokenizer is None or encoder is None:
        raise RuntimeError("The configured backbone cannot encode text anchors.")

    anchors = []
    with torch.no_grad():
        for pathology in pathologies:
            prompt = (
                f"this is an image of {image_context}; "
                f"{pathology.lower()} presented in image"
            )
            tokens = tokenizer([prompt])
            if isinstance(tokens, torch.Tensor):
                tokens = tokens.to(device)
            feature = torch.nn.functional.normalize(encoder(tokens), dim=-1)
            anchors.append(feature.cpu().numpy())
    return np.vstack(anchors)


def run_ood_evaluation(
    ref_embeddings: np.ndarray,
    ref_labels: np.ndarray,
    id_test_embeddings: np.ndarray,
    ood_embeddings: np.ndarray,
    methods: list[str],
    n_ref_sizes: list[int],
    id_test_logits: Optional[np.ndarray] = None,
    ood_logits: Optional[np.ndarray] = None,
    text_anchor_embeddings: Optional[np.ndarray] = None,
    prototypes: Optional[np.ndarray] = None,
    seed: int = 42,
) -> dict:
    results = {}
    for n_ref in n_ref_sizes:
        current_ref, current_labels = stratified_reference_subset(
            ref_embeddings, ref_labels, n_ref, seed=seed
        )
        ref_key = "full" if n_ref <= 0 or n_ref >= len(ref_embeddings) else f"n{len(current_ref)}"

        detector = OODDetector().fit(
            current_ref,
            current_labels,
            prototypes=prototypes,
        )

        for method in methods:
            key = f"{method}_{ref_key}"
            if method == "mahalanobis":
                id_scores = detector.score_mahalanobis(id_test_embeddings)
                ood_scores = detector.score_mahalanobis(ood_embeddings)
            elif method == "knn":
                id_scores = detector.score_knn(id_test_embeddings, k=5)
                ood_scores = detector.score_knn(ood_embeddings, k=5)
            elif method == "text_anchor":
                if text_anchor_embeddings is None:
                    print("  [Skip] text_anchor requires --checkpoint, --config, and --id-pathologies.")
                    continue
                id_scores = detector.score_text_anchor(id_test_embeddings, text_anchor_embeddings)
                ood_scores = detector.score_text_anchor(ood_embeddings, text_anchor_embeddings)
            elif method == "energy":
                if id_test_logits is None or ood_logits is None:
                    print("  [Skip] energy requires logits in both embedding archives.")
                    continue
                id_scores = detector.score_energy(id_test_logits)
                ood_scores = detector.score_energy(ood_logits)
            else:
                print(f"  [Warning] Unknown method: {method}")
                continue

            metrics = evaluate_ood(id_scores, ood_scores)
            results[key] = metrics
            print(
                f"  {key:30s} | AUROC: {metrics['auroc']:.4f} | "
                f"AUPR-Out: {metrics['aupr_out']:.4f} | "
                f"FPR@95: {metrics['fpr_at_95tpr']:.4f}"
            )
    return results


def main():
    parser = argparse.ArgumentParser(description="Leakage-free OOD detection evaluation")
    parser.add_argument(
        "--id-embeddings",
        required=True,
        help="ID test embeddings, or an ID pool that will be split when --ref-embeddings is omitted.",
    )
    parser.add_argument(
        "--ref-embeddings",
        default=None,
        help="Optional disjoint ID reference/calibration embeddings.",
    )
    parser.add_argument("--ood-embeddings", required=True)
    parser.add_argument("--methods", default="mahalanobis,knn,text_anchor,energy")
    parser.add_argument("--n-ref", default="0")
    parser.add_argument("--ref-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--config", default=None, help="Training YAML required for text anchors.")
    parser.add_argument("--id-pathologies", default=None)
    parser.add_argument("--image-context", default="a bone x-ray")
    parser.add_argument("--output-dir", default="results/ood/")
    args = parser.parse_args()

    methods = [value.strip() for value in args.methods.split(",") if value.strip()]
    n_ref_sizes = [int(value.strip()) for value in args.n_ref.split(",")]

    id_data = load_embeddings(args.id_embeddings)
    id_embeddings = _normalize_embeddings(id_data["image_embeddings"])
    id_labels = _integer_labels(id_data["labels"])
    id_logits = id_data.get("logits")

    if args.ref_embeddings:
        ref_data = load_embeddings(args.ref_embeddings)
        ref_embeddings = _normalize_embeddings(ref_data["image_embeddings"])
        ref_labels = _integer_labels(ref_data["labels"])
        id_test_embeddings = id_embeddings
        id_test_logits = id_logits
    else:
        split = split_reference_and_id_test(
            id_embeddings,
            id_labels,
            logits=id_logits,
            ref_fraction=args.ref_fraction,
            seed=args.seed,
        )
        ref_embeddings = split["ref_embeddings"]
        ref_labels = split["ref_labels"]
        id_test_embeddings = split["id_test_embeddings"]
        id_test_logits = split["id_test_logits"]
        print(
            "[Info] --ref-embeddings was omitted; created a disjoint stratified "
            f"split: reference={len(ref_embeddings)}, ID-test={len(id_test_embeddings)}."
        )

    ood_data = load_embeddings(args.ood_embeddings)
    ood_embeddings = _normalize_embeddings(ood_data["image_embeddings"])
    ood_logits = ood_data.get("logits")

    text_anchors = None
    if "text_anchor" in methods:
        if args.checkpoint and args.config and args.id_pathologies:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            pathologies = [value.strip() for value in args.id_pathologies.split(",")]
            text_anchors = encode_text_anchors(
                args.checkpoint,
                args.config,
                pathologies,
                device,
                args.image_context,
            )
        else:
            print("[Warning] text_anchor requested without checkpoint/config/pathology names.")

    prototypes = load_prototypes_from_checkpoint(args.checkpoint)
    results = run_ood_evaluation(
        ref_embeddings=ref_embeddings,
        ref_labels=ref_labels,
        id_test_embeddings=id_test_embeddings,
        ood_embeddings=ood_embeddings,
        methods=methods,
        n_ref_sizes=n_ref_sizes,
        id_test_logits=id_test_logits,
        ood_logits=ood_logits,
        text_anchor_embeddings=text_anchors,
        prototypes=prototypes,
        seed=args.seed,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    output = {
        "type": "ood_evaluation",
        "reference_source": args.ref_embeddings or "stratified split from id_source",
        "id_source": args.id_embeddings,
        "ood_source": args.ood_embeddings,
        "methods": methods,
        "n_ref_sizes": n_ref_sizes,
        "seed": args.seed,
        "timestamp": datetime.datetime.now().isoformat(),
        "results": results,
    }
    json_path = os.path.join(args.output_dir, "ood_metrics.json")
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, ensure_ascii=False)
    print(f"\nOOD results saved to: {json_path}")


if __name__ == "__main__":
    main()