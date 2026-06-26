"""
OOD Detection Evaluation for XBone-Net
========================================
Evaluates OOD detection using pre-extracted embeddings from evaluate_model.py.

Usage:
    # Cross-dataset OOD: FracAtlas (ID) vs BTXRD (OOD)
    python evaluate_ood.py \
        --id-embeddings results/claim2_loss_semantic_fracatlas/seed_42/embeddings.npz \
        --ood-embeddings results/claim2_loss_semantic_btxrd/seed_42/embeddings.npz \
        --methods mahalanobis,knn,text_anchor \
        --output-dir results/ood/fracatlas_vs_btxrd

    # Few-shot reference set study
    python evaluate_ood.py \
        --id-embeddings results/.../embeddings.npz \
        --ood-embeddings results/.../embeddings.npz \
        --methods mahalanobis,knn \
        --n-ref 5,10,25,50,100,0 \
        --output-dir results/ood/fewshot_study

    # With text anchors (requires model checkpoint for encoding prompts)
    python evaluate_ood.py \
        --id-embeddings results/.../embeddings.npz \
        --ood-embeddings results/.../embeddings.npz \
        --methods text_anchor \
        --checkpoint checkpoints/.../best_phase1.pth \
        --id-pathologies "fractured" \
        --output-dir results/ood/text_anchor
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

from utils.ood import OODDetector, evaluate_ood


def load_embeddings(path: str) -> dict:
    """Load embeddings from .npz file created by evaluate_model.py."""
    data = np.load(path, allow_pickle=True)
    result = {
        "image_embeddings": data["image_embeddings"],
        "labels": data["labels"],
    }
    # Load text embeddings if present
    text_keys = [k for k in data.files if k.startswith("text_embeddings_")]
    if text_keys:
        result["text_embeddings"] = {
            k.replace("text_embeddings_", ""): data[k] for k in text_keys
        }
    return result


def encode_text_anchors(
    checkpoint_path: str,
    pathologies: list[str],
    device: torch.device,
    image_context: str = "a bone x-ray",
) -> np.ndarray:
    """Encode text prompts as OOD anchors using a BiomedCLIP model."""
    from models.builder import build_model
    from omegaconf import OmegaConf

    # Build minimal model just for text encoding
    cfg = OmegaConf.create({
        "peft": {"type": "none", "params": {}},
        "fusion": {"type": "none", "params": {}},
        "classifier": {"type": "none", "params": {}},
    })
    model = build_model(cfg).to(device)

    # Load checkpoint if provided
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

    # Encode positive text prompts for each ID pathology
    anchors = []
    with torch.no_grad():
        for path in pathologies:
            prompt = f"this is an image of {image_context}; {path.lower()} presented in image"
            tokens = tokenizer([prompt]).to(device)
            feat = model.backbone.model.encode_text(tokens)
            feat /= feat.norm(dim=-1, keepdim=True)
            anchors.append(feat.cpu().numpy())

    return np.vstack(anchors)  # (K, D)


def run_ood_evaluation(
    id_embeddings: np.ndarray,
    id_labels: np.ndarray,
    ood_embeddings: np.ndarray,
    methods: list[str],
    n_ref_sizes: list[int],
    text_anchor_embeddings: np.ndarray = None,
) -> dict:
    """
    Run OOD detection with multiple methods and reference set sizes.

    Returns structured results for all method × n_ref combinations.
    """
    results = {}

    for n_ref in n_ref_sizes:
        # Subsample reference set
        if n_ref == 0:
            # Use full ID set
            ref_embeds = id_embeddings
            ref_labels = id_labels
            ref_key = "full"
        else:
            n_ref = min(n_ref, len(id_embeddings))
            idx = np.random.RandomState(42).choice(len(id_embeddings), n_ref, replace=False)
            ref_embeds = id_embeddings[idx]
            ref_labels = id_labels[idx]
            ref_key = f"n{n_ref}"

        # Fit OOD detector on reference set
        detector = OODDetector()
        detector.fit(ref_embeds, ref_labels)

        for method in methods:
            key = f"{method}_{ref_key}"

            if method == "text_anchor":
                if text_anchor_embeddings is None:
                    print(f"  [Skip] text_anchor requires --checkpoint and --id-pathologies")
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
                # Energy requires logits — skip if not available
                print(f"  [Skip] Energy score requires logits, not embeddings.")
                continue
            else:
                print(f"  [Warning] Unknown method: {method}")
                continue

            # Evaluate
            eval_result = evaluate_ood(id_scores, ood_scores)
            results[key] = eval_result

            print(f"  {key:30s} | AUROC: {eval_result['auroc']:.4f} | "
                  f"FPR@95: {eval_result['fpr_at_95tpr']:.4f}")

    return results


def main():
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

    # Parse arguments
    methods = [m.strip() for m in args.methods.split(",")]
    n_ref_sizes = [int(n.strip()) for n in args.n_ref.split(",")]

    # Load embeddings
    print(f"\nLoading ID embeddings from: {args.id_embeddings}")
    id_data = load_embeddings(args.id_embeddings)
    # L2-normalize image embeddings
    id_embeds_norm = np.linalg.norm(id_data["image_embeddings"], axis=1, keepdims=True)
    id_data["image_embeddings"] = id_data["image_embeddings"] / (id_embeds_norm + 1e-8)
    print(f"  Shape: {id_data['image_embeddings'].shape}")

    print(f"Loading OOD embeddings from: {args.ood_embeddings}")
    ood_data = load_embeddings(args.ood_embeddings)
    # L2-normalize image embeddings
    ood_embeds_norm = np.linalg.norm(ood_data["image_embeddings"], axis=1, keepdims=True)
    ood_data["image_embeddings"] = ood_data["image_embeddings"] / (ood_embeds_norm + 1e-8)
    print(f"  Shape: {ood_data['image_embeddings'].shape}")

    # Encode text anchors if needed
    text_anchors = None
    if "text_anchor" in methods and args.checkpoint and args.id_pathologies:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pathologies = [p.strip() for p in args.id_pathologies.split(",")]
        print(f"\nEncoding text anchors for: {pathologies}")
        text_anchors = encode_text_anchors(
            args.checkpoint, pathologies, device, args.image_context
        )
        print(f"  Text anchor shape: {text_anchors.shape}")

    # Run OOD evaluation
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
    )

    # Save results
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

    print(f"\n✅ OOD results saved to: {json_path}")


if __name__ == "__main__":
    main()
