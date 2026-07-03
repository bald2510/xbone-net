"""
XBone-Net Out-of-Distribution Detection Evaluation.
===============================================================================
Evaluates OOD detection performance on pre-extracted image and text embeddings:
  - Data Loading: Loads L2-normalized ID and OOD embeddings from .npz files.
  - Prototypes & Anchors: Extracts class prototypes from Phase 2 checkpoints and zero-shot text anchors.
  - OOD Scoring: Fits OODDetector on reference ID samples using Mahalanobis, k-NN, or text-anchor.
  - Metrics Export: Computes AUROC and FPR@95TPR and saves metrics to JSON.
"""

import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import os
import json
import argparse
import datetime

import torch
import numpy as np

from src.utils.ood import OODDetector, evaluate_ood


# ============================================================
# Data Loading & Anchor Extraction
# ============================================================


def load_embeddings(path: str) -> dict:
    """Load image and optional text embeddings from a .npz archive.

    Args:
        path: File path to .npz file produced by evaluate.py --save-embeddings.

    Returns:
        dict: Dictionary containing image_embeddings, labels, and optional text_embeddings.
    """
    data = np.load(path, allow_pickle=True)
    result = {
        "image_embeddings": data["image_embeddings"],
        "labels": data["labels"],
    }
    text_keys = [k for k in data.files if k.startswith("text_embeddings_")]
    if text_keys:
        result["text_embeddings"] = {
            k.replace("text_embeddings_", ""): data[k] for k in text_keys
        }
    return result


def load_prototypes_from_checkpoint(checkpoint_path: str):
    """Extract learned class prototypes from a Phase 2 checkpoint.

    Scans the checkpoint state_dict for keys containing head.prototypes and
    returns the prototype weight matrix as a NumPy array.

    Args:
        checkpoint_path: File path to a PyTorch .pth checkpoint.

    Returns:
        np.ndarray: Prototype matrix of shape (num_classes, embed_dim) if present; None otherwise.
    """
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return None
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint)

        proto_key = None
        for key in state_dict.keys():
            if "head.prototypes" in key:
                proto_key = key
                break

        if proto_key is not None:
            prototypes = state_dict[proto_key].numpy()
            print(f"Loaded prototypes of shape {prototypes.shape} from '{proto_key}'.")
            return prototypes
    except Exception as e:
        print(f"[Warning] Failed to load prototypes from checkpoint: {e}")
    return None


def encode_text_anchors(
    checkpoint_path: str,
    pathologies: list[str],
    device: torch.device,
    image_context: str = "a bone x-ray",
) -> np.ndarray:
    """Encode text prompts into L2-normalized text anchor embeddings.

    Builds model, loads checkpoint weights, and encodes positive prompts for each
    pathology to produce anchor vectors for text-anchor OOD scoring.

    Args:
        checkpoint_path: File path to model checkpoint.
        pathologies: List of target pathology class names.
        device: Target torch.device for model encoding.
        image_context: Context phrase for prompt construction (default: 'a bone x-ray').

    Returns:
        np.ndarray: Array of shape (len(pathologies), embed_dim) containing text anchors.
    """
    from src.models.builder import build_model
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({
        "peft": {"type": "none", "params": {}},
        "fusion": {"type": "none", "params": {}},
        "classifier": {"type": "none", "params": {}},
    })
    model = build_model(cfg).to(device)

    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model_keys = list(model.state_dict().keys())
        has_backbone = any(k.startswith("backbone.model.") for k in model_keys)
        ckpt_keys = list(state_dict.keys())
        if ckpt_keys and has_backbone and not ckpt_keys[0].startswith("backbone.model."):
            if ckpt_keys[0].startswith("model."):
                state_dict = {k.replace("model.", "backbone.model.", 1): v for k, v in state_dict.items()}
            elif ckpt_keys[0].startswith(("visual.", "transformer.", "text.")):
                state_dict = {"backbone.model." + k: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)

    model.eval()
    tokenizer = model.backbone.tokenizer

    anchors = []
    with torch.no_grad():
        for path in pathologies:
            prompt = f"this is an image of {image_context}; {path.lower()} presented in image"
            tokens = tokenizer([prompt]).to(device)
            feat = model.backbone.model.encode_text(tokens)
            feat /= feat.norm(dim=-1, keepdim=True)
            anchors.append(feat.cpu().numpy())

    return np.vstack(anchors)


# ============================================================
# OOD Evaluation Core
# ============================================================

def run_ood_evaluation(
    id_embeddings: np.ndarray,
    id_labels: np.ndarray,
    ood_embeddings: np.ndarray,
    methods: list[str],
    n_ref_sizes: list[int],
    text_anchor_embeddings: np.ndarray = None,
    prototypes: np.ndarray = None,
) -> dict:
    """Compute OOD detection metrics across scoring methods and reference sizes.

    Fits OODDetector on reference ID samples, computes scores for ID and OOD
    embeddings, and calculates AUROC and FPR@95TPR metrics.

    Args:
        id_embeddings: L2-normalized ID image embeddings of shape (N_id, D).
        id_labels: Integer class labels for ID samples of shape (N_id,).
        ood_embeddings: L2-normalized OOD image embeddings of shape (N_ood, D).
        methods: List of scoring method names ('mahalanobis', 'knn', 'text_anchor', 'energy').
        n_ref_sizes: Reference subset sizes to evaluate (0 = full ID set).
        text_anchor_embeddings: Optional text anchor matrix of shape (C, D).
        prototypes: Optional class prototypes matrix of shape (C, D).

    Returns:
        dict: Mapping '{method}_{ref_key}' to evaluation metrics (auroc, fpr_at_95tpr).
    """
    results = {}

    for n_ref in n_ref_sizes:
        if n_ref == 0:
            ref_embeds = id_embeddings
            ref_labels = id_labels
            ref_key = "full"
        else:
            n_ref = min(n_ref, len(id_embeddings))
            idx = np.random.RandomState(42).choice(len(id_embeddings), n_ref, replace=False)
            ref_embeds = id_embeddings[idx]
            ref_labels = id_labels[idx]
            ref_key = f"n{n_ref}"

        detector = OODDetector()
        detector.fit(ref_embeds, ref_labels, prototypes=prototypes)

        for method in methods:
            key = f"{method}_{ref_key}"

            if method == "text_anchor":
                if text_anchor_embeddings is None:
                    print("  [Skip] text_anchor requires --checkpoint and --id-pathologies")
                    continue
                id_scores = detector.score_text_anchor(id_embeddings, text_anchor_embeddings)
                ood_scores = detector.score_text_anchor(ood_embeddings, text_anchor_embeddings)
            elif method == "mahalanobis":
                id_scores = detector.score_mahalanobis(id_embeddings)
                ood_scores = detector.score_mahalanobis(ood_embeddings)
            elif method == "knn":
                id_scores = detector.score_knn(id_embeddings, k=5)
                ood_scores = detector.score_knn(ood_embeddings, k=5)
            elif method == "energy":
                print("  [Skip] Energy score requires logits, not embeddings.")
                continue
            else:
                print(f"  [Warning] Unknown method: {method}")
                continue

            eval_result = evaluate_ood(id_scores, ood_scores)
            results[key] = eval_result

            print(f"  {key:30s} | AUROC: {eval_result['auroc']:.4f} | "
                  f"FPR@95: {eval_result['fpr_at_95tpr']:.4f}")

    return results


# ============================================================
# Main Entry Point & CLI Parsing
# ============================================================

def main():
    """CLI entry-point: parse arguments, load data, and execute OOD evaluation."""
    parser = argparse.ArgumentParser(description="OOD detection evaluation")
    parser.add_argument("--id-embeddings", required=True, help="Path to ID embeddings .npz")
    parser.add_argument("--ood-embeddings", required=True, help="Path to OOD embeddings .npz")
    parser.add_argument("--methods", default="mahalanobis,knn,text_anchor",
                        help="Comma-separated OOD methods")
    parser.add_argument("--n-ref", default="0",
                        help="Comma-separated reference set sizes (0=full)")
    parser.add_argument("--checkpoint", default=None,
                        help="Model checkpoint for text anchor encoding")
    parser.add_argument("--id-pathologies", default=None,
                        help="Comma-separated ID pathology names for text anchors")
    parser.add_argument("--image-context", default="a bone x-ray")
    parser.add_argument("--output-dir", default="results/ood/")
    args = parser.parse_args()

    print("=" * 60)
    print("OOD DETECTION EVALUATION")
    print("=" * 60)

    methods = [m.strip() for m in args.methods.split(",")]
    n_ref_sizes = [int(n.strip()) for n in args.n_ref.split(",")]

    print(f"\nLoading ID embeddings from: {args.id_embeddings}")
    id_data = load_embeddings(args.id_embeddings)
    id_embeds_norm = np.linalg.norm(id_data["image_embeddings"], axis=1, keepdims=True)
    id_data["image_embeddings"] = id_data["image_embeddings"] / (id_embeds_norm + 1e-8)
    print(f"  Shape: {id_data['image_embeddings'].shape}")

    print(f"Loading OOD embeddings from: {args.ood_embeddings}")
    ood_data = load_embeddings(args.ood_embeddings)
    ood_embeds_norm = np.linalg.norm(ood_data["image_embeddings"], axis=1, keepdims=True)
    ood_data["image_embeddings"] = ood_data["image_embeddings"] / (ood_embeds_norm + 1e-8)
    print(f"  Shape: {ood_data['image_embeddings'].shape}")

    text_anchors = None
    if "text_anchor" in methods and args.checkpoint and args.id_pathologies:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pathologies = [p.strip() for p in args.id_pathologies.split(",")]
        print(f"\nEncoding text anchors for: {pathologies}")
        text_anchors = encode_text_anchors(
            args.checkpoint, pathologies, device, args.image_context
        )
        print(f"  Text anchor shape: {text_anchors.shape}")

    prototypes = None
    if args.checkpoint:
        prototypes = load_prototypes_from_checkpoint(args.checkpoint)

    print(f"\nMethods: {methods}")
    print(f"Reference sizes: {n_ref_sizes}")
    print("-" * 60)

    results = run_ood_evaluation(
        id_embeddings=id_data["image_embeddings"],
        id_labels=id_data["labels"],
        ood_embeddings=ood_data["image_embeddings"],
        methods=methods,
        n_ref_sizes=n_ref_sizes,
        text_anchor_embeddings=text_anchors,
        prototypes=prototypes,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    output = {
        "type": "ood_evaluation",
        "id_source": args.id_embeddings,
        "ood_source": args.ood_embeddings,
        "methods": methods,
        "n_ref_sizes": n_ref_sizes,
        "timestamp": datetime.datetime.now().isoformat(),
        "results": results,
    }

    json_path = os.path.join(args.output_dir, "ood_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nOOD results saved to: {json_path}")


if __name__ == "__main__":
    main()

