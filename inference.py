"""
XBone-Net Single-Sample Inference & OOD Visualizer.
===============================================================================
Executes single-sample CLI inference, attention map visualization, and OOD scoring:
  - Model Loading: Loads model architecture and Phase 2 checkpoint.
  - OOD Detection: Scores sample against ID calibration using Mahalanobis, k-NN, or text-anchor.
  - Spatial Attention: Captures ViT self-attention and renders 14x14 heatmap overlay on X-ray image.
  - Cross-Attention: Analyzes bi-directional cross-attention head weights and cross-modal affinity.

Outputs prediction probabilities, OOD status, and PNG visualization of attention maps.
"""

import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import os
import types
import argparse
import torch
import numpy as np
from PIL import Image
import hydra
from omegaconf import DictConfig
import matplotlib.pyplot as plt

from src.models.builder import build_model, setup_phase2_modules
from src.utils.ood import OODDetector


# ============================================================
# ViT Attention Hooks & Renderers
# ============================================================

# Global buffer for captured ViT self-attention matrices
captured_spatial_attention = []


def patched_attn_forward(self, x, attn_mask=None, is_causal=False):
    """Monkey-patched forward for ViT self-attention to capture attention weights.

    Replaces default fused-attention forward of the last ViT block so that raw attention
    matrices are appended to captured_spatial_attention.

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
    if hasattr(self, "q_norm") and self.q_norm is not None:
        q, k = self.q_norm(q), self.k_norm(k)
    q = q * self.scale
    attn = q @ k.transpose(-2, -1)
    attn = attn.softmax(dim=-1)
    captured_spatial_attention.append(attn.detach().cpu())
    attn = self.attn_drop(attn)
    x = attn @ v
    x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
    if hasattr(self, "norm") and self.norm is not None:
        x = self.norm(x)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


def render_and_save_spatial_attention(raw_img, captured_attn, output_path="results/attention_map.png"):
    """Render CLS-token spatial attention as a heatmap overlay on the input image.

    Extracts CLS token attention to spatial patches from the last ViT block, averages
    across heads, and reshapes into a 14x14 grid. Normalizes map and renders terminal grid
    and saved side-by-side matplotlib figure.

    Args:
        raw_img: Original PIL.Image object.
        captured_attn: List of captured attention tensors.
        output_path: Output file path for saved PNG figure.

    Returns:
        None
    """
    if not captured_attn:
        return
    attn_matrix = captured_attn[0].squeeze(0)
    cls_attn = attn_matrix[:, 0, 1:].mean(dim=0).numpy()
    spatial_map = cls_attn.reshape(14, 14)
    norm_map = (spatial_map - spatial_map.min()) / (spatial_map.max() - spatial_map.min() + 1e-8)

    chars = ["  ", "░░", "▒▒", "▓▓", "██"]
    print("\nSPATIAL ATTENTION GRID (14x14 ViT ATTENTION):")
    print("┌" + "─" * 28 + "┐")
    for row in norm_map:
        row_str = ""
        for val in row:
            char_idx = min(int(val * len(chars)), len(chars) - 1)
            row_str += chars[char_idx]
        print("│" + row_str + "│")
    print("└" + "─" * 28 + "┘")

    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(raw_img)
        axes[0].set_title("Original X-ray Scan", fontsize=11)
        axes[0].axis("off")

        axes[1].imshow(raw_img)
        axes[1].imshow(norm_map, cmap="jet", alpha=0.5, extent=(0, raw_img.width, raw_img.height, 0), interpolation="bilinear")
        axes[1].set_title("Attention Heatmap Overlay (Focused ROI)", fontsize=11)
        axes[1].axis("off")

        plt.suptitle("Visual Spatial Attention Map (BioMedCLIP ViT)", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved attention map visualization to: {output_path}")
    except Exception as e:
        print(f"[Warning] Could not save attention map plot: {e}")


# ============================================================
# Helper Functions & Checkpoint Loading
# ============================================================

def adapt_state_dict_keys(
    state_dict: dict[str, torch.Tensor],
    model_keys: list[str],
) -> dict[str, torch.Tensor]:
    """Adapt checkpoint key prefixes for XBone-Net wrapper compatibility.

    Args:
        state_dict: Checkpoint OrderedDict of parameter tensors.
        model_keys: Parameter names from target model state_dict.

    Returns:
        dict[str, torch.Tensor]: State dict with corrected key prefixes.
    """
    model_has_backbone_model = any(k.startswith("backbone.model.") for k in model_keys)
    checkpoint_keys = list(state_dict.keys())
    if not checkpoint_keys:
        return state_dict

    first_ckpt_key = checkpoint_keys[0]

    if model_has_backbone_model and not first_ckpt_key.startswith("backbone.model."):
        if first_ckpt_key.startswith("model."):
            print("Detected 'model.' prefix in checkpoint keys. Converting to 'backbone.model.'...")
            return {k.replace("model.", "backbone.model.", 1): v for k, v in state_dict.items()}
        elif first_ckpt_key.startswith(("visual.", "transformer.", "text.")):
            print("Detected raw submodule prefix. Prepending 'backbone.model.'...")
            return {"backbone.model." + k: v for k, v in state_dict.items()}

    return state_dict


def load_checkpoint_from_dir(
    model: torch.nn.Module,
    model_dir: str,
    device: torch.device,
) -> bool:
    """Load checkpoint weights from directory by searching standard filenames.

    Args:
        model: Target nn.Module.
        model_dir: Directory path containing checkpoints.
        device: Target torch.device for weight loading.

    Returns:
        bool: True if a checkpoint was found and loaded; False otherwise.
    """
    if not os.path.isdir(model_dir):
        return False

    checkpoint_filenames = [
        "best_phase2.pth",
        "best_phase1.pth",
        "best_semantic_lora.pth",
        "open_clip_pytorch_model.bin",
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
        print(f"Loading weights from checkpoint folder: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        state_dict = adapt_state_dict_keys(state_dict, list(model.state_dict().keys()))
        model.load_state_dict(state_dict, strict=False)
        print("-> Checkpoint loaded successfully from folder!\n")
        return True
    return False


def load_model_checkpoint(
    model: torch.nn.Module,
    cfg: DictConfig,
    device: torch.device,
    custom_checkpoint_path: str = None
) -> None:
    """Load model checkpoint weights with fallback search strategy.

    Args:
        model: Target nn.Module.
        cfg: Hydra DictConfig containing checkpoint path fields.
        device: Target torch.device for weight loading.
        custom_checkpoint_path: If provided, used directly.

    Returns:
        None
    """
    if custom_checkpoint_path and os.path.exists(custom_checkpoint_path):
        print(f"Loading weights from custom path: {custom_checkpoint_path}")
        checkpoint = torch.load(custom_checkpoint_path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        state_dict = adapt_state_dict_keys(state_dict, list(model.state_dict().keys()))
        model.load_state_dict(state_dict, strict=False)
        print("-> Checkpoint loaded successfully from custom path!\n")
        return

    loaded = False
    params_cfg = cfg.get("params", {}) or {}
    model_dir = params_cfg.get("model_dir", None)

    if model_dir:
        loaded = load_checkpoint_from_dir(model, model_dir, device)

    if not loaded:
        checkpoint_path = cfg.get("checkpoint_path", None)
        if not checkpoint_path:
            checkpoint_path = params_cfg.get("checkpoint_path", None)
            if not checkpoint_path:
                phase2_cfg = params_cfg.get("phase2", {}) or {}
                checkpoint_path = phase2_cfg.get("checkpoint_path", None)

        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Loading weights from config checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            state_dict = adapt_state_dict_keys(state_dict, list(model.state_dict().keys()))
            model.load_state_dict(state_dict, strict=False)
            print("-> Checkpoint loaded successfully!\n")
        else:
            print("-> [Info] No checkpoint found. Running model with initial/random weights.\n")


# ============================================================
# Main Entry Point & CLI Parsing
# ============================================================

def parse_args():
    """Parse non-Hydra CLI flags for single-sample inference.

    Returns:
        argparse.Namespace: CLI arguments namespace.
    """
    parser = argparse.ArgumentParser(description="Multi-Modal Inference & OOD Detection")
    parser.add_argument("--image", required=True, type=str, help="Path to test image file")
    parser.add_argument("--clinical-text", type=str, default="", help="Clinical text of the patient (if applicable)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Custom checkpoint path (.pth)")
    parser.add_argument("--id-embeddings", type=str, default=None, 
                        help="Path to In-Distribution reference embeddings (.npz)")
    parser.add_argument("--ood-params", type=str, default=None,
                        help="Path to pre-computed OOD calibration parameters (.npz)")
    parser.add_argument("--method", type=str, default="mahalanobis", choices=["mahalanobis", "knn", "text_anchor"],
                        help="OOD score method")
    parser.add_argument("--fpr-threshold", type=float, default=0.05, 
                        help="Target False Positive Rate to calculate OOD threshold (default: 0.05)")
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda")
    parser.add_argument("--debug", action="store_true", default=False,
                        help="Print detailed model architecture layers for debugging")

    extra_args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return extra_args


args = parse_args()


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run single-sample inference with OOD detection and attention maps.

    Loads model and checkpoint, processes one image (with optional clinical text),
    runs OOD scoring, generates spatial/cross-attention visualizations, and prints results.

    Args:
        cfg: Hydra DictConfig resolved from configs/config.yaml.

    Returns:
        None
    """
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Building model architecture...")
    model = build_model(cfg.model).to(device)

    _, classifier_type, fusion_type, _ = setup_phase2_modules(model, cfg, device)

    params_cfg = cfg.get("params", {}) or {}
    debug_mode = getattr(args, "debug", False) or params_cfg.get("debug", cfg.get("debug", False))
    model.print_architecture(verbose=debug_mode)

    model.eval()
    load_model_checkpoint(model, cfg, device, custom_checkpoint_path=args.checkpoint)

    dataset_params = cfg.dataset.get('params', {}) or {}
    classes = dataset_params.get('classes', dataset_params.get('pathologies', []))
    num_classes = dataset_params.get('num_classes', len(classes))
    is_multilabel = (dataset_params.get('task_type', 'multiclass') == 'multilabel')

    print(f"Dataset classes: {list(classes)}")
    print(f"Task type: {'Multi-label' if is_multilabel else 'Multi-class'}")

    ood_params_path = args.ood_params
    if not ood_params_path and args.checkpoint:
        checkpoint_dir = os.path.dirname(args.checkpoint)
        default_params_path = os.path.join(checkpoint_dir, "ood_parameters.npz")
        if os.path.exists(default_params_path):
            ood_params_path = default_params_path
            print(f"Auto-resolved OOD parameters to: {ood_params_path}")

    model_prototypes = None
    if hasattr(model, 'head') and hasattr(model.head, 'prototypes'):
        model_prototypes = model.head.prototypes.detach().cpu().numpy()
        print("Successfully extracted trained class prototypes from Prototypical Head for OOD centering.")

    detector = None
    threshold = None

    id_scores_mean = None
    id_scores_std = None

    if args.method == "mahalanobis" and ood_params_path and os.path.exists(ood_params_path):
        print(f"\nLoading pre-computed OOD calibration parameters from: {ood_params_path}...")
        try:
            ood_data = np.load(ood_params_path, allow_pickle=True)
            detector = OODDetector()
            detector.shared_cov_inv = ood_data["shared_cov_inv"]

            detector.class_means = {}
            if model_prototypes is not None:
                norms = np.linalg.norm(model_prototypes, axis=1, keepdims=True)
                normed_prototypes = model_prototypes / (norms + 1e-8)
                for c, proto in enumerate(normed_prototypes):
                    detector.class_means[c] = proto
            detector._fitted = True
            threshold = float(ood_data["threshold"])

            if "id_scores_mean" in ood_data and "id_scores_std" in ood_data:
                id_scores_mean = float(ood_data["id_scores_mean"])
                id_scores_std = float(ood_data["id_scores_std"])

            print(f"OOD Detector successfully initialized from pre-computed parameters (Threshold={threshold:.4f})!")
        except Exception as e:
            print(f"[Warning] Failed to load pre-computed OOD parameters: {e}. Falling back to fitting from ID embeddings.")
            detector = None

    if detector is None:
        id_embed_path = args.id_embeddings
        if not id_embed_path:
            exp_name = cfg.get("experiment_name", "default")
            seed_val = cfg.get("seed", 42)
            default_path = os.path.join("results", str(exp_name), f"seed_{seed_val}", "embeddings.npz")
            if os.path.exists(default_path):
                id_embed_path = default_path
                print(f"Auto-resolved ID reference embeddings to: {id_embed_path}")
            else:
                print(f"\n[Warning] No In-Distribution reference embeddings (.npz) found at default path '{default_path}'")
                print("OOD Detection CANNOT be run without reference embeddings or pre-computed OOD parameters. Skipping OOD checks.")

        if id_embed_path and os.path.exists(id_embed_path):
            print(f"\nFitting OOD Detector using reference: {id_embed_path}...")
            id_data = np.load(id_embed_path, allow_pickle=True)
            id_embeddings = id_data["image_embeddings"]
            id_labels = id_data["labels"]

            id_embeds_norm = np.linalg.norm(id_embeddings, axis=1, keepdims=True)
            id_embeddings = id_embeddings / (id_embeds_norm + 1e-8)

            detector = OODDetector()
            detector.fit(id_embeddings, id_labels, prototypes=model_prototypes)

            if args.method == "mahalanobis":
                id_scores = detector.score_mahalanobis(id_embeddings)
            elif args.method == "knn":
                id_scores = detector.score_knn(id_embeddings, k=5)
            elif args.method == "text_anchor":
                anchors = []
                tokenizer = model.backbone.tokenizer
                with torch.no_grad():
                    for path in classes:
                        prompt = f"this is an image of a bone x-ray; {path.lower()} presented in image"
                        tokens = tokenizer([prompt]).to(device)
                        feat = model.backbone.model.encode_text(tokens)
                        feat = feat / feat.norm(dim=-1, keepdim=True)
                        anchors.append(feat.cpu().numpy())
                text_anchors = np.vstack(anchors)
                id_scores = detector.score_text_anchor(id_embeddings, text_anchors)

            id_scores_mean = float(np.mean(id_scores))
            id_scores_std = float(np.std(id_scores))
            threshold = np.percentile(id_scores, 100 * (1.0 - args.fpr_threshold))
            print(f"OOD Detector ({args.method}) fitted successfully! OOD threshold (FPR={args.fpr_threshold:.2f}): {threshold:.4f}")

    print(f"\nProcessing input image: {args.image}...")
    if not os.path.exists(args.image):
        print(f"[Error] Image file not found: {args.image}")
        return

    image = Image.open(args.image).convert("RGB")
    preprocess = model.backbone.preprocess
    image_tensor = preprocess(image).unsqueeze(0).to(device)

    backbone_type = cfg.model.get("backbone_type", "biomedclip")
    is_image_only = backbone_type.startswith("resnet50") or backbone_type.startswith("densenet121")

    text_tokens = None
    if not is_image_only:
        clinical_text = args.clinical_text
        if not clinical_text:
            clinical_text = "no abnormal findings or fractures visible"
            print(f"[Info] No clinical text provided. Using default: '{clinical_text}'")
        tokenizer = model.backbone.tokenizer
        text_tokens = tokenizer([clinical_text]).to(device)

    print("Running forward pass...")
    attn_info = None
    captured_spatial_attention.clear()
    if hasattr(model.backbone, "model") and hasattr(model.backbone.model, "visual"):
        try:
            attn_module = model.backbone.model.visual.trunk.blocks[-1].attn
            attn_module.fused_attn = False
            attn_module.forward = types.MethodType(patched_attn_forward, attn_module)
        except Exception:
            pass

    with torch.no_grad():
        img_feats, txt_feats = model.backbone(image_tensor, text_tokens)

        if txt_feats is None or is_image_only:
            fused_feats = img_feats
        else:
            if hasattr(model.fusion, "top_cross_attn"):
                fused_feats, attn_info = model.fusion(img_feats, txt_feats, return_attn=True)
            else:
                fused_feats = model.fusion(img_feats, txt_feats)

        test_embed = fused_feats.cpu().numpy()
        test_embed = test_embed / (np.linalg.norm(test_embed, axis=1, keepdims=True) + 1e-8)

        logits = model.head(fused_feats)
        if is_multilabel:
            probs = torch.sigmoid(logits).cpu().numpy()[0]
        else:
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

    is_ood = False
    test_score = None
    if detector is not None:
        if args.method == "mahalanobis":
            test_score = detector.score_mahalanobis(test_embed)[0]
        elif args.method == "knn":
            test_score = detector.score_knn(test_embed, k=5)[0]
        elif args.method == "text_anchor":
            test_score = detector.score_text_anchor(test_embed, text_anchors)[0]

        is_ood = test_score > threshold

    render_and_save_spatial_attention(image, captured_spatial_attention, output_path="results/attention_map.png")
    if attn_info is not None:
        top_attn = attn_info["top_attn_weights"].cpu().squeeze(0).squeeze(-1).squeeze(-1).numpy()
        bottom_attn = attn_info["bottom_attn_weights"].cpu().squeeze(0).squeeze(-1).squeeze(-1).numpy()

        top_out = attn_info["top_out"]
        bottom_out = attn_info["bottom_out"]

        top_norm = top_out / (top_out.norm(dim=-1, keepdim=True) + 1e-8)
        bottom_norm = bottom_out / (bottom_out.norm(dim=-1, keepdim=True) + 1e-8)
        mutual_affinity = (top_norm @ bottom_norm.T).item()

        print("\n" + "=" * 60)
        print("BI-DIRECTIONAL CROSS-ATTENTION MAP")
        print("=" * 60)
        print("1. Top Branch (Image Query -> Text Key/Value):")
        head_str_top = " | ".join([f"H{i+1}: {w:.4f}" for i, w in enumerate(top_attn)])
        print(f"   - 8 Attention heads : [ {head_str_top} ]")
        print(f"   - Average attention  : {top_attn.mean():.4f}")

        print("\n2. Bottom Branch (Text Query -> Image Key/Value):")
        head_str_bottom = " | ".join([f"H{i+1}: {w:.4f}" for i, w in enumerate(bottom_attn)])
        print(f"   - 8 Attention heads : [ {head_str_bottom} ]")
        print(f"   - Average attention  : {bottom_attn.mean():.4f}")

        print("\n3. Cross-Modal Mutual Affinity:")
        print(f"   - Top & Bottom stream alignment: {mutual_affinity:.4f}")
        print("=" * 60)

    print("\n" + "=" * 60)
    print("PREDICTION RESULTS & OOD DETECTION")
    print("=" * 60)

    if detector is not None:
        embed_dim = test_embed.shape[-1]
        norm_score = test_score / embed_dim
        norm_threshold = threshold / embed_dim

        print(f"OOD Method          : {args.method.upper()}")
        print(f"Raw Score (D²)      : {test_score:10.4f}  (Threshold: {threshold:.4f})")
        print(f"Dimension-Norm (D²/D): {norm_score:10.4f}  (Threshold: {norm_threshold:.4f})")

        z_str = ""
        if id_scores_mean is not None and id_scores_std is not None and id_scores_std > 0:
            z_score = (test_score - id_scores_mean) / id_scores_std
            z_thresh = (threshold - id_scores_mean) / id_scores_std
            print(f"Z-Score             : {z_score:+10.4f} σ (Threshold: {z_thresh:+.4f} σ)")
            print(f"                      [Ref ID Mean: {id_scores_mean:.2f}, Std: {id_scores_std:.2f}]")
            z_str = f" ({z_score:+.2f} σ from ID mean)"
        else:
            theo_mean = float(embed_dim)
            theo_std = float(np.sqrt(2 * embed_dim))
            z_score = (test_score - theo_mean) / theo_std
            z_thresh = (threshold - theo_mean) / theo_std
            print(f"Z-Score (theoretical): {z_score:+10.4f} σ (Threshold: {z_thresh:+.4f} σ)")
            z_str = f" ({z_score:+.2f} σ from theoretical mean)"

        print("-" * 60)

        if is_ood:
            print("STATUS       : OUT-OF-DISTRIBUTION (OOD)")
            print(f"WARNING      : Sample is out-of-distribution{z_str}. Skipping disease prediction.")
            print("=" * 60 + "\n")
            return
        else:
            print("STATUS       : IN-DISTRIBUTION (ID)")
            print(f"               Valid sample{z_str}, proceeding with classification.")
    else:
        print("STATUS       : UNKNOWN OOD (No reference loaded)")

    print("-" * 60)
    print("Classification Details:")
    if is_multilabel:
        predicted_classes = []
        for idx, prob in enumerate(probs):
            class_name = classes[idx]
            status_flag = "[PREDICTED]" if prob >= 0.5 else "           "
            print(f"{status_flag} {class_name:25s} : Prob {prob*100:6.2f}%")
            if prob >= 0.5:
                predicted_classes.append((class_name, prob))

        print("-" * 60)
        if predicted_classes:
            predicted_str = ", ".join([f"{name} ({prob*100:.1f}%)" for name, prob in predicted_classes])
            print(f"Clinical Diagnosis: Detected {predicted_str}")
        else:
            print("Clinical Diagnosis: No abnormalities detected (Normal)")
    else:
        pred_idx = int(np.argmax(probs))
        for idx, prob in enumerate(probs):
            class_name = classes[idx]
            status_flag = "[PREDICTED]" if idx == pred_idx else "           "
            print(f"{status_flag} {class_name:25s} : Prob {prob*100:6.2f}%")

        print("-" * 60)
        print(f"Clinical Diagnosis: '{classes[pred_idx]}' ({probs[pred_idx]*100:.2f}%)")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

