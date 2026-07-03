"""
XBone-Net Evaluation Pipeline.
===============================================================================
Executes model evaluation, metric computation, bootstrap CIs, and embedding export:
  - Model Loading: Reconstructs architecture from Hydra config and loads Phase 2/Phase 1 weights.
  - Inference: Evaluates test set in zero-shot or classifier mode to obtain probabilities.
  - Metrics & CIs: Computes AUROC, F1, Accuracy, Sensitivity, Specificity, Precision, and optional 95% CIs.
  - Embedding Export: Option to save intermediate image/text embeddings to .npz for downstream analysis.

Outputs results to JSON and optional .npz embeddings.
"""

import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import os
import json
import argparse
import datetime
from typing import Optional

import torch
import hydra
import numpy as np
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from src.models.builder import build_model, setup_phase2_modules
from src.datasets.builder import build_dataloader
from src.utils.prompts import generate_custom_prompts
from src.utils.metrics import (compute_metrics, compute_metrics_multiclass,
                                bootstrap_confidence_intervals)


# ============================================================
# Helper Functions & Checkpoint Loading
# ============================================================


def seed_everything(seed: int = 42) -> None:
    """Set random seeds across Python, NumPy, and PyTorch for deterministic evaluation.

    Args:
        seed: Integer seed value (default: 42).

    Returns:
        None
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
    state_dict: dict[str, torch.Tensor],
    model_keys: list[str],
) -> dict[str, torch.Tensor]:
    """Adapt checkpoint state-dict key prefixes for XBone-Net compatibility.

    Checkpoints saved by raw OpenCLIP use prefixes like 'model.' or bare sub-module
    names ('visual.', 'text.'), whereas the XBone-Net wrapper nests everything under
    'backbone.model.'. This function detects mismatches and re-maps keys so that
    model.load_state_dict succeeds.

    Args:
        state_dict: Loaded checkpoint OrderedDict mapping parameter names to tensors.
        model_keys: List of parameter names from target model state_dict keys.

    Returns:
        dict[str, torch.Tensor]: Adapted state-dict with corrected key prefixes.
    """
    model_has_backbone_model = any(k.startswith("backbone.model.") for k in model_keys)
    checkpoint_keys = list(state_dict.keys())
    if not checkpoint_keys:
        return state_dict

    first_ckpt_key = checkpoint_keys[0]

    if model_has_backbone_model and not first_ckpt_key.startswith("backbone.model."):
        if first_ckpt_key.startswith("model."):
            print("Detected 'model.' prefix in checkpoint keys. Converting to 'backbone.model.' for compatibility.")
            return {k.replace("model.", "backbone.model.", 1): v for k, v in state_dict.items()}

        elif first_ckpt_key.startswith(("visual.", "transformer.", "text.")):
            print("Detected raw OpenCLIP submodule prefix in checkpoint keys. Prepending 'backbone.model.' for compatibility.")
            return {"backbone.model." + k: v for k, v in state_dict.items()}

    return state_dict


def load_checkpoint_from_dir(
    model: torch.nn.Module,
    model_dir: str,
    device: torch.device,
) -> bool:
    """Load model weights from a checkpoint directory.

    Searches model_dir for checkpoint files in priority order:
    best_phase2.pth > best_phase1.pth > best_semantic_lora.pth
    > open_clip_pytorch_model.bin > checkpoint_epoch_10.pth.
    Falls back to any .pth / .bin file found in the directory.

    Args:
        model: Target nn.Module to load weights into.
        model_dir: Directory path containing checkpoints.
        device: Target torch.device for weight loading.

    Returns:
        bool: True if a checkpoint was found and loaded successfully; False otherwise.
    """
    if not os.path.isdir(model_dir):
        print(f"[Warning] model_dir '{model_dir}' does not exist or is not a directory.")
        return False

    checkpoint_filenames = [
        "best_phase2.pth",
        "best_phase1.pth",
        "best_semantic_lora.pth",
        "open_clip_pytorch_model.bin",
        "checkpoint_epoch_10.pth",
    ]

    checkpoint_path = None
    for filename in checkpoint_filenames:
        path = os.path.join(model_dir, filename)
        if os.path.exists(path):
            checkpoint_path = path
            break

    if not checkpoint_path:
        for file in os.listdir(model_dir):
            if file.endswith((".pth", ".bin")):
                checkpoint_path = os.path.join(model_dir, file)
                break

    if checkpoint_path:
        print(f"Loading weights from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        state_dict = adapt_state_dict_keys(state_dict, list(model.state_dict().keys()))
        model.load_state_dict(state_dict, strict=False)
        print("-> Checkpoint loaded successfully from directory!\n")
        return True
    else:
        print(f"[Warning] No checkpoint file (.pth or .bin) found in directory '{model_dir}'")
        return False


def load_model_checkpoint(
    model: torch.nn.Module,
    cfg: DictConfig,
    device: torch.device,
    is_zero_shot: bool = False,
) -> bool:
    """Load model checkpoint weights using paths resolved from config.

    Attempts loading in order: params.model_dir, then checkpoint_path from
    various config levels (root, params, params.phase2, params.phase1).

    Args:
        model: Target nn.Module to load weights into.
        cfg: Hydra DictConfig containing checkpoint fields.
        device: Target torch.device for weight mapping.
        is_zero_shot: Whether the evaluation is a zero-shot baseline.

    Returns:
        bool: True if checkpoint loaded or official foundation weights are used.
    """
    loaded = False
    params_cfg = cfg.get("params", {}) or {}
    model_dir = params_cfg.get("model_dir", None)

    if model_dir and os.path.isdir(model_dir):
        print(f"Searching for checkpoints in folder: {model_dir}")
        loaded = load_checkpoint_from_dir(model, model_dir, device)

    if not loaded:
        checkpoint_path = cfg.get("checkpoint_path", None)
        if not checkpoint_path:
            checkpoint_path = params_cfg.get("checkpoint_path", None)
            if not checkpoint_path:
                phase2_cfg = params_cfg.get("phase2", {}) or {}
                checkpoint_path = phase2_cfg.get("checkpoint_path", None)
                if not checkpoint_path:
                    phase1_cfg = params_cfg.get("phase1", {}) or {}
                    checkpoint_path = phase1_cfg.get("checkpoint_path", None)

        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Loading weights from file: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            state_dict = adapt_state_dict_keys(state_dict, list(model.state_dict().keys()))
            model.load_state_dict(state_dict, strict=False)
            print("-> Checkpoint loaded successfully from file!\n")
            loaded = True
        elif is_zero_shot:
            print("-> [Zero-Shot Baseline] Using official pre-trained foundation weights (CLIP / PubMedCLIP / MedCLIP / BiomedCLIP) loaded at model build time.\n")
            return True
        else:
            print("-> [Info] No fine-tuned checkpoint found. Model is running with pre-trained foundation weights.\n")
            return False

    return loaded


# ============================================================
# Inference & Evaluation Core
# ============================================================

def run_evaluation(
    model: torch.nn.Module,
    test_loader,
    pathologies: list[str],
    is_classifier: bool,
    device: torch.device,
    temperature: float = 0.07,
    image_context: str = "a bone x-ray",
    p2_report_type: str = "clinical",
    save_embeddings: bool = False,
) -> dict:
    """Run model inference over test DataLoader and collect predictions.

    Operates in classifier mode (forward pass through backbone, fusion, head) or
    zero-shot mode (cosine similarity against pre-encoded text prompts).

    Args:
        model: XBone-Net model in evaluation mode.
        test_loader: DataLoader for the test split.
        pathologies: List of target class names.
        is_classifier: If True, use classification head; else use zero-shot prompts.
        device: Target torch.device for inference.
        temperature: Temperature scaling factor for zero-shot cosine logits (default: 0.07).
        image_context: Context phrase for text prompts (default: 'a bone x-ray').
        p2_report_type: Report branch ('xray', 'clinical', or 'both').
        save_embeddings: If True, collect and return intermediate embeddings.

    Returns:
        dict: Dictionary containing predicted probabilities, ground truths, and optional embeddings.
    """
    if is_classifier and p2_report_type != "clinical":
        print(f"[Info] Overriding p2_report_type='{p2_report_type}' to 'clinical' for inference.")
        p2_report_type = "clinical"

    tokenizer = model.backbone.tokenizer

    text_features_dict: dict[str, torch.Tensor] = {}
    text_embeddings_np: Optional[dict[str, np.ndarray]] = None

    if not is_classifier:
        print("\nZero-shot mode: Pre-extracting text prompt features using pre-trained text encoder...")
        prompt_dict = generate_custom_prompts(pathologies, image_context)
        print(f"Generated prompts for pathologies: {list(prompt_dict.values())}")

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
                    raise AttributeError(f"Model backbone {type(model.backbone)} does not support encode_text.")

                pos_feat = pos_feat / pos_feat.norm(dim=-1, keepdim=True)
                neg_feat = neg_feat / neg_feat.norm(dim=-1, keepdim=True)

                text_features_dict[path] = torch.cat([pos_feat, neg_feat], dim=0)

        if save_embeddings:
            text_embeddings_np = {
                path: text_features_dict[path].cpu().numpy()
                for path in pathologies
            }

    all_probs = []
    all_ground_truths = []
    all_image_embeds = [] if save_embeddings else None

    print("\nScanning test dataset...")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            if len(batch) == 4:
                images, xray_ids, clinical_ids, labels = batch
                if p2_report_type == "both":
                    input_ids = None
                elif p2_report_type == "xray":
                    input_ids = xray_ids
                else:
                    input_ids = clinical_ids
            elif len(batch) == 3:
                images, input_ids, labels = batch
                xray_ids = None
                clinical_ids = None
            else:
                raise ValueError(f"Unexpected batch format with length {len(batch)}")

            images = images.to(device)
            if input_ids is not None and isinstance(input_ids, torch.Tensor):
                input_ids = input_ids.to(device)

            if is_classifier:
                if p2_report_type == "both" and len(batch) == 4:
                    xray_ids = xray_ids.to(device)
                    clinical_ids = clinical_ids.to(device)
                    img_feat, xray_feat = model.backbone(images, xray_ids)
                    _, clinical_feat = model.backbone(images, clinical_ids)
                    text_feat = (xray_feat + clinical_feat) / 2.0
                    if model.fusion is not None:
                        fused = model.fusion(img_feat, text_feat)
                    else:
                        fused = img_feat
                    if save_embeddings:
                        all_image_embeds.append(fused.cpu().numpy())
                    logits = model.head(fused)
                elif save_embeddings:
                    img_feats, txt_feats = model.backbone(images, input_ids)
                    if txt_feats is None:
                        fused = img_feats
                    else:
                        fused = model.fusion(img_feats, txt_feats)
                    all_image_embeds.append(fused.cpu().numpy())
                    logits = model.head(fused)
                else:
                    outputs = model(images, input_ids)
                    if isinstance(outputs, tuple):
                        logits = outputs[0]
                    else:
                        logits = outputs
                batch_probs = torch.softmax(logits, dim=-1)
            else:
                img_feat = model.backbone.model.encode_image(images)
                img_feat /= img_feat.norm(dim=-1, keepdim=True)

                if save_embeddings:
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
    if save_embeddings and all_image_embeds:
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
    metrics: dict,
    cfg: DictConfig,
    output_dir: str,
    ci_95: Optional[dict] = None,
) -> str:
    """Save evaluation metrics and experiment configuration to JSON.

    Args:
        metrics: Dictionary of computed metric values.
        cfg: Hydra DictConfig to serialize alongside results.
        output_dir: Output directory path for metrics.json.
        ci_95: Optional dictionary of 95% bootstrap confidence intervals.

    Returns:
        str: Absolute file path to saved metrics.json.
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


def save_embeddings(
    image_embeddings: Optional[np.ndarray],
    text_embeddings: Optional[dict[str, np.ndarray]],
    labels: np.ndarray,
    output_dir: str,
) -> Optional[str]:
    """Save extracted image and text embeddings to a compressed .npz file.

    Args:
        image_embeddings: Image embeddings array of shape (N, D) or None.
        text_embeddings: Optional dict mapping pathology to (2, D) text embeddings.
        labels: Ground-truth labels array of shape (N,) or (N, C).
        output_dir: Directory path to save embeddings.npz.

    Returns:
        Optional[str]: Saved file path or None if image_embeddings is None.
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
    parser.add_argument("--bootstrap", action="store_true", default=False,
                        help="Compute bootstrap 95%% confidence intervals")
    parser.add_argument("--n-bootstrap", type=int, default=10_000,
                        help="Number of bootstrap resamples (default: 10000)")
    parser.add_argument("--save-embeddings", action="store_true", default=False,
                        help="Save image/text embeddings to .npz")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save JSON results and embeddings")
    parser.add_argument("--debug", action="store_true", default=False,
                        help="Print detailed model architecture layers for debugging")

    extra_args, remaining = parser.parse_known_args()

    sys.argv = [sys.argv[0]] + remaining

    return extra_args


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    """Execute main evaluation workflow.

    Builds model, loads checkpoint, executes test set inference, computes metrics
    and optional bootstrap CIs, and exports results.

    Args:
        cfg: Hydra DictConfig resolved from configs/config.yaml.

    Returns:
        None
    """
    params_cfg = cfg.get("params", {}) or {}
    seed_val = params_cfg.get("seed", cfg.get("seed", 42))
    seed_everything(seed_val)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting evaluation on device: {device} (seed: {seed_val})")

    print("Building model architecture...")
    model = build_model(cfg.model).to(device)

    p2_phase_cfg = params_cfg.get("phase2", {}) or {}
    _, classifier_type, fusion_type, _ = setup_phase2_modules(model, cfg, device)

    debug_mode = extra_args.debug or params_cfg.get("debug", cfg.get("debug", False))
    model.print_architecture(verbose=debug_mode)

    pathologies = cfg.dataset.params.get('classes', cfg.dataset.params.get('pathologies', []))
    is_classifier = classifier_type != "none"
    exp_name = str(params_cfg.get("experiment_name", cfg.get("experiment_name", ""))).lower()
    is_zero_shot = (not is_classifier) or ("zeroshot" in exp_name)

    if is_zero_shot:
        backbone_type = cfg.model.get("backbone_type", "biomedclip")
        print(f"[Zero-Shot Baseline] Using official pre-trained foundation weights for '{backbone_type}'.")

    model.eval()
    ckpt_loaded = load_model_checkpoint(model, cfg, device, is_zero_shot=is_zero_shot)
    if not ckpt_loaded and not is_zero_shot:
        print("[ERROR] Cannot run evaluation for fine-tuned experiment because no trained checkpoint (.pth) was found!")
        print("        Please ensure train.py completes successfully and saves best_phase2.pth before running evaluate.py.\n")
        sys.exit(1)

    preprocess = model.backbone.preprocess
    tokenizer = model.backbone.tokenizer

    print(f"Loading test dataset: {cfg.dataset.name}...")
    test_loader = build_dataloader(
        cfg=cfg.dataset,
        split="test",
        transform=preprocess,
        tokenizer=tokenizer,
    )

    task_type = cfg.dataset.params.get('task_type', 'multiclass')
    is_multilabel = (task_type == 'multilabel')

    temperature = params_cfg.get("temperature", 0.07)
    image_context = params_cfg.get("image_context", "a bone x-ray")
    p2_report_type = p2_phase_cfg.get("p2_report_type", "clinical")

    eval_output = run_evaluation(
        model=model,
        test_loader=test_loader,
        pathologies=list(pathologies),
        is_classifier=is_classifier,
        device=device,
        temperature=temperature,
        image_context=image_context,
        p2_report_type=p2_report_type,
        save_embeddings=extra_args.save_embeddings,
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
            all_probs, all_gt,
            pathologies=list(pathologies),
            is_multilabel=is_multilabel,
            n_bootstrap=extra_args.n_bootstrap,
            seed=seed_val,
        )

    if extra_args.output_dir:
        output_dir = extra_args.output_dir
    else:
        exp_name = params_cfg.get("experiment_name", cfg.get("experiment_name", "default"))
        seed_val = params_cfg.get("seed", cfg.get("seed", 0))
        output_dir = os.path.join("results", str(exp_name), f"seed_{seed_val}")

    save_results_json(metrics, cfg, output_dir, ci_95=ci_95)

    if extra_args.save_embeddings:
        save_embeddings(
            image_embeddings=eval_output["image_embeddings"],
            text_embeddings=eval_output["text_embeddings"],
            labels=all_gt,
            output_dir=output_dir,
        )

    print("\nEvaluation complete!")


extra_args = parse_extra_args()

if __name__ == "__main__":
    main()

