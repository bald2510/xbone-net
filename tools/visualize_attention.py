"""
XBone-Net Attention Overlay Visualizer.
===============================================================================
Extracts ViT self-attention weights and cross-attention fusion maps, generates
diagnostic heatmap overlays:
  - Attention Monkey-Patching: Intercepts last ViT block self-attention matrices.
  - Cross-Attention Fusion: Extracts image-text cross-attention from fusion module.
  - Heatmap Rendering: Maps CLS-token attention to 14x14 grid and overlays on X-ray.
  - Annotation Overlay: Draws ground-truth bounding box/polygon annotations.
  - Figure Export: Saves to results/visualization and report images folder.
"""

import sys
import os
sys.path.append(os.path.abspath('.'))

import types
import json
import shutil
import argparse
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from omegaconf import OmegaConf

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from src.models.builder import build_model, setup_phase2_modules
from src.datasets.builder import build_dataloader
import importlib.util

# Load project's evaluate.py (not pip 'evaluate' package)
_eval_spec = importlib.util.spec_from_file_location("proj_evaluate", os.path.join(os.path.abspath('.'), "evaluate.py"))
_eval_module = importlib.util.module_from_spec(_eval_spec)
_eval_spec.loader.exec_module(_eval_module)
load_model_checkpoint = _eval_module.load_model_checkpoint
adapt_state_dict_keys = _eval_module.adapt_state_dict_keys


# ============================================================
# ViT Self-Attention Forward Hooks
# ============================================================

captured_attention = []


def patched_attn_forward(self, x, attn_mask=None, is_causal=False):
    """Patched self-attention forward method capturing internal attention weights.

    Intercepts the ViT self-attention computation to extract and store the
    raw attention weight matrix (after softmax, before dropout) for visualization.

    Args:
        self: timm.models.vision_transformer.Attention instance.
        x: Input tensor of shape (B, N, C).
        attn_mask: Optional attention mask (unused).
        is_causal: Causal mask flag (unused).

    Returns:
        torch.Tensor: Output tensor of shape (B, N, C).
    """
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q, k = self.q_norm(q), self.k_norm(k)

    q = q * self.scale
    attn = q @ k.transpose(-2, -1)
    attn = attn.softmax(dim=-1)

    captured_attention.append(attn.detach().cpu())

    attn = self.attn_drop(attn)
    x = attn @ v

    x = x.transpose(1, 2).reshape(B, N, C)
    x = self.norm(x)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


# ============================================================
# Config & Model Loading
# ============================================================

def load_experiment_config(experiment_path: str) -> OmegaConf:
    """Load a full Hydra experiment config by path.

    Args:
        experiment_path: Relative experiment config path (e.g., 'btxrd/proposed/ours_xbone_net').

    Returns:
        OmegaConf: Fully resolved Hydra config.
    """
    config_dir = os.path.abspath("configs")
    GlobalHydra.instance().clear()

    # Register hydra resolver for standalone usage (outside hydra.main)
    try:
        OmegaConf.register_new_resolver("hydra", lambda path: os.getcwd() if "cwd" in path else "")
    except ValueError:
        pass  # Already registered

    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="config", overrides=[f"+experiment={experiment_path}"])

    # Force-resolve any remaining hydra:runtime.cwd interpolations
    OmegaConf.set_struct(cfg, False)
    cwd = os.getcwd()
    cfg_dict = OmegaConf.to_container(cfg, resolve=False)

    def resolve_hydra_cwd(obj):
        if isinstance(obj, str) and "${hydra:runtime.cwd}" in obj:
            return obj.replace("${hydra:runtime.cwd}", cwd)
        elif isinstance(obj, dict):
            return {k: resolve_hydra_cwd(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [resolve_hydra_cwd(v) for v in obj]
        return obj

    cfg_resolved = resolve_hydra_cwd(cfg_dict)
    return OmegaConf.create(cfg_resolved)


def build_and_load_model(cfg, device):
    """Build model from config and load checkpoint.

    Args:
        cfg: Hydra config with model/dataset/params.
        device: torch.device.

    Returns:
        nn.Module: Loaded model in eval mode.
    """
    print("Building model from config...")
    model = build_model(cfg.model).to(device)
    model, classifier_type, fusion_type, _ = setup_phase2_modules(model, cfg, device)
    model.eval()

    print("Loading checkpoint...")
    ckpt_loaded = load_model_checkpoint(model, cfg, device, is_zero_shot=False)
    if not ckpt_loaded:
        print("[ERROR] Could not load checkpoint!")
        sys.exit(1)

    return model


# ============================================================
# Visualization Functions
# ============================================================

def generate_attention_overlay(
    raw_img,
    attn_map,
    gt_class_name,
    pred_class_name,
    pred_prob,
    annotation_path=None,
    output_path=None,
    cross_attn_info=None,
):
    """Generate attention overlay visualization figure.

    Creates a multi-panel figure with the original X-ray, ViT self-attention
    overlay, and optionally cross-attention fusion visualization.

    Args:
        raw_img: PIL Image of the original X-ray.
        attn_map: 2D numpy array of attention weights (14x14).
        gt_class_name: Ground truth class name string.
        pred_class_name: Predicted class name string.
        pred_prob: Prediction probability float.
        annotation_path: Optional JSON annotation file path.
        cross_attn_info: Optional dict with cross-attention maps from fusion module.
        output_path: Output file path for the saved figure.

    Returns:
        None
    """
    has_cross_attn = cross_attn_info is not None
    ncols = 3 if has_cross_attn else 2
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5))

    # Panel 1: Original X-ray
    axes[0].imshow(raw_img)
    axes[0].set_title("Original X-ray Scan", fontsize=11)
    axes[0].axis('off')

    # Panel 2: ViT Self-Attention Overlay
    axes[1].imshow(raw_img)
    norm_attn = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
    axes[1].imshow(
        norm_attn, cmap='jet', alpha=0.5,
        extent=(0, raw_img.width, raw_img.height, 0),
        interpolation='bilinear'
    )
    axes[1].set_title("ViT Self-Attention (CLS→patches)", fontsize=11)
    axes[1].axis('off')

    # Panel 3: Cross-Attention Fusion (if available)
    if has_cross_attn:
        # top_attn_weights: Image→Text attention [B, heads, 1, 1] (pooled features are 1-token)
        # We visualize the attention magnitude per head as a bar chart
        top_w = cross_attn_info["top_attn_weights"].squeeze().detach().cpu().numpy()
        bottom_w = cross_attn_info["bottom_attn_weights"].squeeze().detach().cpu().numpy()

        # If both are scalar (1→1 attention), show per-head magnitude comparison
        if top_w.ndim == 1:
            # Each head has a single attention weight (image→text for 1-token each)
            x_pos = np.arange(len(top_w))
            width = 0.35
            axes[2].bar(x_pos - width/2, top_w, width, label='Img→Txt', color='#2196F3', alpha=0.8)
            axes[2].bar(x_pos + width/2, bottom_w, width, label='Txt→Img', color='#FF5722', alpha=0.8)
            axes[2].set_xlabel('Attention Head', fontsize=10)
            axes[2].set_ylabel('Attention Weight', fontsize=10)
            axes[2].set_title('Cross-Attention Fusion\n(per head)', fontsize=11)
            axes[2].legend(fontsize=9)
            axes[2].set_xticks(x_pos)
        else:
            # Fallback: show as heatmap
            combined = np.stack([top_w.mean(axis=0), bottom_w.mean(axis=0)])
            axes[2].imshow(combined, cmap='viridis', aspect='auto')
            axes[2].set_yticks([0, 1])
            axes[2].set_yticklabels(['Img→Txt', 'Txt→Img'])
            axes[2].set_title('Cross-Attention Weights', fontsize=11)

    # Title
    correct = gt_class_name.lower() == pred_class_name.lower()
    title_color = '#2E7D32' if correct else '#C62828'
    status = '✓' if correct else '✗'
    fig.suptitle(
        f"{status} GT: {gt_class_name.upper()} | Pred: {pred_class_name.upper()} ({pred_prob*100:.1f}%)",
        fontsize=13, fontweight='bold', y=0.98, color=title_color
    )

    # Draw annotations if available
    if annotation_path and os.path.exists(annotation_path):
        try:
            with open(annotation_path, 'r', encoding='utf-8') as f:
                anno_data = json.load(f)
            shapes = anno_data.get('shapes', [])
            for shape in shapes:
                points = shape.get('points', [])
                shape_type = shape.get('shape_type', 'polygon')
                if not points:
                    continue
                pts = np.array(points)
                # Draw on panels 0 and 1 (original + attention)
                for ax in axes[:2]:
                    if shape_type == 'rectangle':
                        x1, y1 = pts[0]
                        x2, y2 = pts[1]
                        rect = patches.Rectangle(
                            (min(x1, x2), min(y1, y2)),
                            abs(x2 - x1), abs(y2 - y1),
                            linewidth=2.5, edgecolor='#00FF00',
                            facecolor='none', linestyle='--'
                        )
                        ax.add_patch(rect)
                    else:
                        poly = patches.Polygon(
                            pts, closed=True, linewidth=2.5,
                            edgecolor='#00FF00', facecolor='none', linestyle='--'
                        )
                        ax.add_patch(poly)
            print(f"  [GT] Drew {len(shapes)} annotation shapes")
        except Exception as e:
            print(f"  [Warning] Failed to draw annotation: {e}")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_path}")
    plt.close()


# ============================================================
# Main Entry Point
# ============================================================

def main():
    """Run attention visualization pipeline for XBone-Net proposed model."""
    parser = argparse.ArgumentParser(description="XBone-Net Attention Visualizer")
    parser.add_argument("--experiment", "-e", type=str,
                        default="btxrd/proposed/ours_xbone_net",
                        help="Experiment config path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-samples", "-n", type=int, default=10,
                        help="Max candidates to scan per class for best prediction")
    parser.add_argument("--classes", nargs="+", type=str, default=None,
                        help="Specific class names to visualize (default: all)")
    parser.add_argument("--output-dir", type=str, default="results/visualization",
                        help="Output directory for saved figures")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # --- Load config via Hydra ---
    cfg = load_experiment_config(args.experiment)

    # Disable struct mode to allow overrides
    OmegaConf.set_struct(cfg, False)
    cfg.seed = args.seed

    # Override model_dir for correct checkpoint discovery
    experiment_name = str(cfg.get("experiment_name", args.experiment))
    model_dir = os.path.join("checkpoints", experiment_name, f"seed_{args.seed}")
    if "params" not in cfg:
        cfg.params = {}
    cfg.params.model_dir = model_dir
    cfg.params.seed = args.seed

    # --- Build and load model ---
    model = build_and_load_model(cfg, device)

    # --- Patch ViT self-attention for extraction ---
    print("Patching ViT attention layer...")
    attn_module = model.backbone.model.visual.trunk.blocks[-1].attn
    attn_module.fused_attn = False
    attn_module.forward = types.MethodType(patched_attn_forward, attn_module)

    # --- Check if model has cross-attention fusion ---
    has_cross_attn_fusion = hasattr(model, 'fusion') and hasattr(model.fusion, 'top_cross_attn')
    if has_cross_attn_fusion:
        print("[INFO] Model has Cross-Attention fusion — will extract fusion attention maps")

    # --- Build test dataloader ---
    tokenizer_func = getattr(model.backbone, "tokenizer_obj", getattr(model.backbone, "tokenizer", None))
    test_loader = build_dataloader(
        cfg.dataset, split="test",
        transform=model.backbone.preprocess,
        tokenizer=tokenizer_func,
    )
    dataset = test_loader.dataset

    # --- Determine target classes ---
    classes_list = list(cfg.dataset.params.classes)
    if args.classes:
        target_classes = {classes_list.index(c): c for c in args.classes if c in classes_list}
    else:
        target_classes = {i: c for i, c in enumerate(classes_list)}

    # --- Build candidate sample index per class ---
    candidates = {c: [] for c in target_classes}
    for idx in range(len(dataset)):
        row = dataset.df.iloc[idx]
        class_id = int(row['class_id'])
        if class_id in target_classes:
            candidates[class_id].append(idx)

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Generate visualizations ---
    print(f"\nGenerating attention maps for {len(target_classes)} classes...\n")
    for class_id, class_name in target_classes.items():
        indices = candidates[class_id]
        if not indices:
            print(f"[{class_name}] No samples found, skipping.")
            continue

        # Find best correctly-predicted sample
        selected_idx = None
        best_prob = 0.0

        print(f"[{class_name}] Scanning {min(len(indices), args.num_samples)} candidates...")
        for idx in indices[:args.num_samples]:
            sample = dataset[idx]
            # Handle both dual-report (4-tuple) and single-report (3-tuple) modes
            if len(sample) == 4:
                image_tensor, xray_ids, clinical_ids, labels = sample
                text_input = clinical_ids  # Use clinical report for evaluation
            else:
                image_tensor, text_input, labels = sample

            img_in = image_tensor.unsqueeze(0).to(device)
            txt_in = text_input.unsqueeze(0).to(device)

            captured_attention.clear()
            with torch.no_grad():
                logits = model(img_in, txt_in)
                probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

            pred_class = np.argmax(probs)
            if pred_class == class_id and probs[class_id] > best_prob:
                best_prob = probs[class_id]
                selected_idx = idx

        if selected_idx is None:
            selected_idx = indices[0]
            print(f"  [Fallback] No correct prediction. Using index {selected_idx}")
        else:
            print(f"  Selected index {selected_idx}, confidence: {best_prob:.4f}")

        # --- Final forward pass on selected sample ---
        sample = dataset[selected_idx]
        if len(sample) == 4:
            image_tensor, xray_ids, clinical_ids, labels = sample
            text_input = clinical_ids
        else:
            image_tensor, text_input, labels = sample

        img_in = image_tensor.unsqueeze(0).to(device)
        txt_in = text_input.unsqueeze(0).to(device)

        captured_attention.clear()

        # Run forward with cross-attention extraction if available
        cross_attn_info = None
        with torch.no_grad():
            if has_cross_attn_fusion:
                # Manual forward to capture cross-attention
                img_feats, txt_feats = model.backbone(img_in, txt_in)
                fused_feats, cross_attn_info = model.fusion(img_feats, txt_feats, return_attn=True)
                logits = model.head(fused_feats)
            else:
                logits = model(img_in, txt_in)

            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        # --- Extract image path ---
        row = dataset.df.iloc[selected_idx]
        image_id = str(row['image_id'])
        raw_img_path = os.path.join(dataset.img_dir, image_id)
        raw_img = Image.open(raw_img_path).convert('RGB')

        # --- Extract ViT self-attention map ---
        if captured_attention:
            attn_matrix = captured_attention[0].squeeze(0)  # [heads, N, N]
            cls_attn = attn_matrix[:, 0, 1:].mean(dim=0).numpy()  # Average over heads, CLS→patches
            attn_map = cls_attn.reshape(14, 14)
        else:
            print(f"  [Warning] No ViT attention captured, using zeros")
            attn_map = np.zeros((14, 14))

        # --- Class info ---
        gt_class_name = classes_list[class_id]
        pred_class_id = int(np.argmax(probs))
        pred_class_name = classes_list[pred_class_id]
        pred_prob = probs[pred_class_id]

        # --- Annotation path ---
        file_name_without_ext = os.path.splitext(image_id)[0]
        annotation_path = os.path.join("data/BTXRD/Annotations", f"{file_name_without_ext}.json")

        # --- Generate and save ---
        safe_class_name = class_name.replace(" ", "_")
        output_file = os.path.join(args.output_dir, f"attn_{safe_class_name}.png")

        generate_attention_overlay(
            raw_img=raw_img,
            attn_map=attn_map,
            gt_class_name=gt_class_name,
            pred_class_name=pred_class_name,
            pred_prob=pred_prob,
            annotation_path=annotation_path,
            output_path=output_file,
            cross_attn_info=cross_attn_info,
        )

        # Copy to report images
        report_dest = os.path.join("docs", "report", "images", f"attn_{safe_class_name}.png")
        os.makedirs(os.path.dirname(report_dest), exist_ok=True)
        if os.path.exists(output_file):
            shutil.copy(output_file, report_dest)
            print(f"  Copied to: {report_dest}")

    print(f"\nAll attention maps saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
