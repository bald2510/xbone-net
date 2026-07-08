"""
XBone-Net Model Evaluation Pipeline.
===============================================================================
Executes model evaluation, metric computation, bootstrap 95% CIs, and embedding exports:

  - Model Reconstruction: Reconstructs architecture from Hydra config and loads
    fine-tuned Phase 2/Phase 1 weights or pre-trained foundation weights.
  - Test Set Inference: Evaluates test split in zero-shot mode (CLIP text prompts)
    or classifier mode (cross-attention fusion + prototypical head).
  - Metric Computation: Calculates AUROC, F1-Macro, Accuracy, Sensitivity, Specificity,
    Precision, and optional 95% bootstrap confidence intervals.
  - Embedding Export: Option to export intermediate image/text embeddings to .npz.

Outputs evaluation metrics to JSON and optional .npz embeddings file.
"""

import os
import sys
import json
import argparse
import datetime
from typing import Optional, Tuple, Dict, Any

# Safely force UTF-8 stdout/stderr on Windows environments
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

import torch
import torch.nn as nn
import numpy as np
import hydra
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf

from src.models.builder import build_model, setup_phase2_modules
from src.datasets.builder import build_dataloader
from src.utils.prompts import generate_custom_prompts
from src.utils.metrics import (
    compute_metrics,
    compute_metrics_multiclass,
    bootstrap_confidence_intervals,
)


# ============================================================
# Helper Functions & Checkpoint Loading
# ============================================================

def seed_everything(seed: int = 42) -> None:
    """Set random seeds across Python, NumPy, and PyTorch for deterministic evaluation.

    Args:
        seed (int): Integer seed value (default: 42).
    """
    import random
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def adapt_state_dict_keys(
    state_dict: Dict[str, torch.Tensor],
    model_keys: list[str],
) -> Dict[str, torch.Tensor]:
    """Adapt checkpoint state-dict key prefixes for XBone-Net compatibility.

    Checkpoints saved by raw OpenCLIP use prefixes like 'model.' or bare sub-module
    names ('visual.', 'text.'), whereas the XBone-Net wrapper nests everything under
    'backbone.model.'. This function detects mismatches and re-maps keys.

    Args:
        state_dict (Dict[str, torch.Tensor]): Loaded checkpoint state_dict mapping.
        model_keys (list[str]): Target model parameter name keys.

    Returns:
        Dict[str, torch.Tensor]: Adapted state_dict with aligned key prefixes.
    """
    model_has_backbone = any(k.startswith("backbone.model.") for k in model_keys)
    checkpoint_keys = list(state_dict.keys())
    if not checkpoint_keys:
        return state_dict

    first_key = checkpoint_keys[0]

    if model_has_backbone and not first_key.startswith("backbone.model."):
        if first_key.startswith("model."):
            print(" -> Converting 'model.' prefix in checkpoint keys to 'backbone.model.'")
            return {k.replace("model.", "backbone.model.", 1): v for k, v in state_dict.items()}
        elif first_key.startswith(("visual.", "transformer.", "text.")):
            print(" -> Prepending 'backbone.model.' prefix to raw OpenCLIP checkpoint keys.")
            return {"backbone.model." + k: v for k, v in state_dict.items()}

    return state_dict


def load_model_checkpoint(
    model: nn.Module,
    cfg: DictConfig,
    device: torch.device,
    is_zero_shot: bool = False,
) -> bool:
    """Load model checkpoint weights using paths resolved from config.

    Searches in priority order: params.model_dir, then checkpoint_path from
    various config levels (root, params, params.phase2, params.phase1).

    Args:
        model (nn.Module): Target PyTorch model instance.
        cfg (DictConfig): Complete Hydra configuration object.
        device (torch.device): Computation device.
        is_zero_shot (bool): Flag indicating zero-shot evaluation mode.

    Returns:
        bool: True if checkpoint loaded or zero-shot foundation weights are active.
    """
    if is_zero_shot:
        print(" -> [Zero-Shot Baseline] Using official pre-trained foundation weights loaded at model build time.\n")
        return True

    params_cfg = cfg.get("params", {}) or {}
    model_dir = params_cfg.get("model_dir", None)
    experiment_name = str(cfg.get("experiment_name", "default_experiment"))
    seed_val = params_cfg.get("seed", cfg.get("seed", 42))

    default_dir = os.path.join("checkpoints", experiment_name, f"seed_{seed_val}")
    if not model_dir or "//" in model_dir or "seed_/" in str(model_dir):
        model_dir = default_dir

    checkpoint_path = None
    if model_dir and os.path.isdir(model_dir):
        for candidate in ["best_phase2.pth", "best_phase1.pth", "checkpoint_epoch_10.pth"]:
            p = os.path.join(model_dir, candidate)
            if os.path.exists(p):
                checkpoint_path = p
                break

    if not checkpoint_path:
        checkpoint_path = cfg.get("checkpoint_path", None) or params_cfg.get("checkpoint_path", None)
        if not checkpoint_path:
            p2_cfg = params_cfg.get("phase2", {}) or {}
            checkpoint_path = p2_cfg.get("checkpoint_path", None)
            if not checkpoint_path:
                p1_cfg = params_cfg.get("phase1", {}) or {}
                checkpoint_path = p1_cfg.get("checkpoint_path", None)

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading weights from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        state_dict = adapt_state_dict_keys(state_dict, list(model.state_dict().keys()))

        # --- Filter out head weights that mismatch config num_classes ---
        dataset_params = cfg.dataset.get('params', {})
        classes = dataset_params.get('classes', dataset_params.get('pathologies', []))
        expected_num_classes = dataset_params.get('num_classes', len(classes))

        filtered_state_dict = {}
        for k, v in state_dict.items():
            # Only filter classification head keys (head.prototypes, head.fc, etc.)
            # NOT backbone keys like backbone.model.visual.*.head.proj.*
            if k.startswith('head.') and v.shape[0] != expected_num_classes:
                print(f"  [Checkpoint] Skipping '{k}' (shape {v.shape}) — "
                      f"mismatches expected num_classes={expected_num_classes}")
                continue
            filtered_state_dict[k] = v

        model.load_state_dict(filtered_state_dict, strict=False)
        print(" -> Checkpoint loaded successfully!\n")
        return True
    elif is_zero_shot:
        print(" -> [Zero-Shot Baseline] Using official pre-trained foundation weights loaded at model build time.\n")
        return True

    print(f"[Warning] No checkpoint file (.pth) found. Using initial weights.\n")
    return False


# ============================================================
# Inference & Evaluation Core
# ============================================================

def run_evaluation(
    model: nn.Module,
    test_loader,
    pathologies: list[str],
    is_classifier: bool,
    device: torch.device,
    temperature: float = 0.07,
    image_context: str = "a bone x-ray",
    p2_report_type: str = "clinical",
    save_embeddings_flag: bool = False,
) -> Dict[str, Any]:
    """Run model inference over test DataLoader and collect probabilities.

    Operates in classifier mode (forward pass through backbone, fusion, head) or
    zero-shot mode (cosine similarity against pre-encoded text prompts).

    Args:
        model (nn.Module): Model instance in evaluation mode.
        test_loader: DataLoader for test split.
        pathologies (list[str]): Target class / pathology names.
        is_classifier (bool): True if using classification head; False if zero-shot.
        device (torch.device): Computation device.
        temperature (float): Temperature factor for zero-shot cosine logits (default: 0.07).
        image_context (str): Context prompt string for text prompts.
        p2_report_type (str): Report type for Phase 2 ('xray', 'clinical', 'both').
        save_embeddings_flag (bool): Flag to save intermediate embeddings.

    Returns:
        Dict[str, Any]: Dictionary containing 'all_probs', 'all_ground_truths',
            and optional 'image_embeddings' / 'text_embeddings'.
    """
    tokenizer = model.backbone.tokenizer
    text_features_dict: Dict[str, torch.Tensor] = {}
    text_embeddings_np: Optional[Dict[str, np.ndarray]] = None

    if not is_classifier:
        print("\nZero-shot mode: Pre-extracting text prompt features using text encoder...")
        prompt_dict = generate_custom_prompts(pathologies, image_context)
        print(f"Generated text prompts for classes: {list(prompt_dict.values())}")

        with torch.no_grad():
            for path, pair in prompt_dict.items():
                pos_tokens = tokenizer([pair["positive"]])
                neg_tokens = tokenizer([pair["negative"]])
                if isinstance(pos_tokens, torch.Tensor):
                    pos_tokens = pos_tokens.to(device)
                if isinstance(neg_tokens, torch.Tensor):
                    neg_tokens = neg_tokens.to(device)

                if hasattr(model.backbone, "encode_text"):
                    pos_feat = model.backbone.encode_text(pos_tokens)
                    neg_feat = model.backbone.encode_text(neg_tokens)
                elif hasattr(model.backbone.model, "encode_text"):
                    pos_feat = model.backbone.model.encode_text(pos_tokens)
                    neg_feat = model.backbone.model.encode_text(neg_tokens)
                else:
                    raise AttributeError(f"Model backbone {type(model.backbone)} lacks encode_text method.")

                pos_feat = pos_feat / pos_feat.norm(dim=-1, keepdim=True)
                neg_feat = neg_feat / neg_feat.norm(dim=-1, keepdim=True)
                text_features_dict[path] = torch.cat([pos_feat, neg_feat], dim=0)

        if save_embeddings_flag:
            text_embeddings_np = {
                path: text_features_dict[path].cpu().numpy()
                for path in pathologies
            }

    all_probs = []
    all_ground_truths = []
    all_image_embeds = [] if save_embeddings_flag else None

    print("\nScanning test dataset...")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            if len(batch) == 4:
                images, xray_ids, clinical_ids, labels = batch
                if p2_report_type in ("both", "xray_clinical"):
                    input_ids = None
                elif p2_report_type == "xray":
                    input_ids = xray_ids
                else:
                    input_ids = clinical_ids
            elif len(batch) == 3:
                images, input_ids, labels = batch
                xray_ids, clinical_ids = None, None
            else:
                raise ValueError(f"Unexpected batch format of length {len(batch)}")

            images = images.to(device)
            if input_ids is not None and isinstance(input_ids, torch.Tensor):
                input_ids = input_ids.to(device)

            if is_classifier:
                if p2_report_type in ("both", "xray_clinical") and len(batch) == 4:
                    img_feat, xray_feat = model.backbone(images, xray_ids.to(device))
                    _, clinical_feat = model.backbone(images, clinical_ids.to(device))
                    text_feat = (xray_feat + clinical_feat) / 2.0
                    fused = model.fusion(img_feat, text_feat) if model.fusion is not None else img_feat
                    if save_embeddings_flag:
                        all_image_embeds.append(fused.cpu().numpy())
                    logits = model.head(fused)
                elif save_embeddings_flag:
                    img_feats, txt_feats = model.backbone(images, input_ids)
                    fused = model.fusion(img_feats, txt_feats) if model.fusion is not None else img_feats
                    all_image_embeds.append(fused.cpu().numpy())
                    logits = model.head(fused)
                else:
                    outputs = model(images, input_ids)
                    logits = outputs[0] if isinstance(outputs, tuple) else outputs

                batch_probs = torch.softmax(logits, dim=-1)
            else:
                img_feat = model.backbone.model.encode_image(images)
                img_feat /= img_feat.norm(dim=-1, keepdim=True)

                if save_embeddings_flag:
                    all_image_embeds.append(img_feat.cpu().numpy())

                batch_probs_list = []
                for path in pathologies:
                    text_weights = text_features_dict[path]
                    logits = img_feat @ text_weights.T
                    probs = torch.softmax(logits / temperature, dim=-1)[:, 0].unsqueeze(1)
                    batch_probs_list.append(probs)
                batch_probs = torch.cat(batch_probs_list, dim=1)

            all_probs.append(batch_probs.cpu())
            all_ground_truths.append(labels.cpu())

    all_probs_np = torch.cat(all_probs, dim=0).numpy()
    all_gt_np = torch.cat(all_ground_truths, dim=0).numpy()

    image_embeddings_np = None
    if save_embeddings_flag and all_image_embeds:
        image_embeddings_np = np.concatenate(all_image_embeds, axis=0)

    return {
        "all_probs": all_probs_np,
        "all_ground_truths": all_gt_np,
        "image_embeddings": image_embeddings_np,
        "text_embeddings": text_embeddings_np,
    }


# ============================================================
# Result Serialization & Export
# ============================================================

def save_results_json(
    metrics: Dict[str, Any],
    cfg: DictConfig,
    output_dir: str,
    ci_95: Optional[Dict[str, Any]] = None,
) -> str:
    """Save evaluation metrics and experiment configuration to JSON.

    Args:
        metrics (Dict[str, Any]): Dictionary of computed metric values.
        cfg (DictConfig): Complete Hydra configuration.
        output_dir (str): Output directory path for metrics.json.
        ci_95 (Optional[Dict[str, Any]]): Optional 95% bootstrap confidence intervals.

    Returns:
        str: Absolute file path to saved metrics.json file.
    """
    os.makedirs(output_dir, exist_ok=True)
    params_cfg = cfg.get("params", {}) or {}
    experiment_name = params_cfg.get("experiment_name", cfg.get("experiment_name", "unknown"))
    dataset_name = cfg.dataset.get("name", "unknown")
    seed = params_cfg.get("seed", cfg.get("seed", -1))

    result = {
        "experiment_name": str(experiment_name),
        "dataset": str(dataset_name),
        "seed": int(seed),
        "timestamp": datetime.datetime.now().isoformat(),
        "metrics": metrics,
        "config": OmegaConf.to_container(cfg, resolve=True),
    }

    if ci_95 is not None:
        result["ci_95"] = ci_95

    json_path = os.path.join(output_dir, "metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nResults saved to: {json_path}")
    return json_path


def export_embeddings(
    image_embeddings: Optional[np.ndarray],
    text_embeddings: Optional[Dict[str, np.ndarray]],
    labels: np.ndarray,
    output_dir: str,
) -> Optional[str]:
    """Save extracted image and text embeddings to a compressed .npz file.

    Args:
        image_embeddings (Optional[np.ndarray]): Image embeddings array of shape (N, D).
        text_embeddings (Optional[Dict[str, np.ndarray]]): Text embeddings dictionary.
        labels (np.ndarray): Ground-truth labels array.
        output_dir (str): Output directory.

    Returns:
        Optional[str]: Saved filepath or None if image_embeddings is None.
    """
    if image_embeddings is None:
        print("[Warning] No image embeddings to save.")
        return None

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "embeddings.npz")

    save_dict = {
        "image_embeddings": image_embeddings,
        "labels": labels,
    }

    if text_embeddings is not None:
        for path_name, emb in text_embeddings.items():
            safe_key = f"text_embeddings_{path_name.replace(' ', '_')}"
            save_dict[safe_key] = emb

    np.savez_compressed(save_path, **save_dict)
    print(f"Embeddings saved to: {save_path}")
    print(f"  Image embeddings shape: {image_embeddings.shape}")
    print(f"  Labels shape:           {labels.shape}")
    return save_path


# ============================================================
# Main Entry Point & CLI Parsing
# ============================================================

def parse_extra_args() -> argparse.Namespace:
    """Parse non-Hydra CLI flags consumed before Hydra initialization.

    Returns:
        argparse.Namespace: CLI arguments namespace.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--bootstrap", action="store_true", default=False, help="Compute bootstrap 95% confidence intervals")
    parser.add_argument("--n-bootstrap", type=int, default=10_000, help="Number of bootstrap resamples (default: 10000)")
    parser.add_argument("--save-embeddings", action="store_true", default=False, help="Save image/text embeddings to .npz")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save JSON results and embeddings")
    parser.add_argument("--debug", action="store_true", default=False, help="Print detailed model architecture layers for debugging")

    extra_args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return extra_args


extra_args = parse_extra_args()


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Main execution orchestrator for XBone-Net evaluation pipeline.

    Args:
        cfg (DictConfig): Complete Hydra configuration object.
    """
    params_cfg = cfg.get("params", {}) or {}
    seed_val = params_cfg.get("seed", cfg.get("seed", 42))
    seed_everything(seed_val)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting evaluation on device: {device} (seed: {seed_val})")

    print("Building model architecture...")
    model = build_model(cfg.model).to(device)

    p2_phase_cfg = params_cfg.get("phase2", {}) or {}
    model, classifier_type, fusion_type, _ = setup_phase2_modules(model, cfg, device)

    debug_mode = extra_args.debug or params_cfg.get("debug", cfg.get("debug", False))
    model.print_architecture(verbose=debug_mode)

    pathologies = cfg.dataset.params.get("classes", cfg.dataset.params.get("pathologies", []))
    is_classifier = classifier_type != "none"
    exp_name = str(params_cfg.get("experiment_name", cfg.get("experiment_name", ""))).lower()
    is_zero_shot = (not is_classifier) or ("zeroshot" in exp_name)

    if is_zero_shot:
        backbone_type = cfg.model.get("backbone_type", "biomedclip")
        print(f"[Zero-Shot Baseline] Using official pre-trained foundation weights for '{backbone_type}'.")

    model.eval()

    # --- Report parameter counts ---
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    print(f"\n  [Params] Total: {total_params:,} | Trainable: {trainable_params:,} ({100*trainable_params/total_params:.2f}%) | Frozen: {frozen_params:,}")

    ckpt_loaded = load_model_checkpoint(model, cfg, device, is_zero_shot=is_zero_shot)
    if not ckpt_loaded and not is_zero_shot:
        print("[ERROR] Cannot run evaluation for fine-tuned experiment because no trained checkpoint (.pth) was found!")
        print("        Please ensure train.py completes successfully before running evaluate.py.\n")
        sys.exit(1)

    print(f"Loading test dataset: {cfg.dataset.name}...")
    tokenizer_func = getattr(model.backbone, "tokenizer_obj", getattr(model.backbone, "tokenizer", None))
    test_loader = build_dataloader(
        cfg=cfg.dataset,
        split="test",
        transform=model.backbone.preprocess,
        tokenizer=tokenizer_func,
    )

    task_type = cfg.dataset.params.get("task_type", "multiclass")
    is_multilabel = (task_type == "multilabel")

    eval_output = run_evaluation(
        model=model,
        test_loader=test_loader,
        pathologies=list(pathologies),
        is_classifier=is_classifier,
        device=device,
        temperature=params_cfg.get("temperature", 0.07),
        image_context=params_cfg.get("image_context", "a bone x-ray"),
        # Always use clinical reports for evaluation to prevent data leakage
        # (xray reports may contain diagnosis labels)
        p2_report_type="clinical",
        save_embeddings_flag=extra_args.save_embeddings,
    )

    all_probs = eval_output["all_probs"]
    all_gt = eval_output["all_ground_truths"]

    if is_multilabel:
        metrics = compute_metrics(all_probs, all_gt, list(pathologies), is_multilabel)
    else:
        metrics = compute_metrics_multiclass(all_probs, all_gt, list(pathologies))

    ci_95 = None
    if extra_args.bootstrap:
        ci_95 = bootstrap_confidence_intervals(
            all_probs,
            all_gt,
            pathologies=list(pathologies),
            is_multilabel=is_multilabel,
            n_bootstrap=extra_args.n_bootstrap,
            seed=seed_val,
        )

    if extra_args.output_dir:
        output_dir = extra_args.output_dir
    else:
        exp_name_str = params_cfg.get("experiment_name", cfg.get("experiment_name", "default"))
        output_dir = os.path.join("results", str(exp_name_str), f"seed_{seed_val}")

    # Add parameter counts to metrics
    metrics["param_total"] = total_params
    metrics["param_trainable"] = trainable_params
    metrics["param_trainable_pct"] = round(100 * trainable_params / total_params, 2) if total_params > 0 else 0

    save_results_json(metrics, cfg, output_dir, ci_95=ci_95)

    if extra_args.save_embeddings:
        export_embeddings(
            image_embeddings=eval_output["image_embeddings"],
            text_embeddings=eval_output["text_embeddings"],
            labels=all_gt,
            output_dir=output_dir,
        )

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
