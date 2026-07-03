"""
XBone-Net Attention Overlay Visualizer.
===============================================================================
Extracts ViT self-attention weights and generates diagnostic heatmap overlays:
  - Attention Monkey-Patching: Intercepts last ViT block self-attention matrices.
  - Heatmap Rendering: Maps CLS-token attention to 14x14 grid and overlays on raw X-ray scans.
  - Annotation Overlay: Draws ground-truth bounding box/polygon annotations from JSON files.
  - Figure Copying: Exports visualization figures to results and document report folders.
"""

import sys
import os
sys.path.append(os.path.abspath('.'))

import types
import json
import shutil
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from omegaconf import OmegaConf

from src.models.builder import build_model, setup_phase2_modules
from src.datasets.builder import build_dataloader
from evaluate import adapt_state_dict_keys


# ============================================================
# ViT Attention Forward Hooks
# ============================================================

captured_attention = []


def patched_attn_forward(self, x, attn_mask=None, is_causal=False):
    """Patched self-attention forward method capturing internal attention weights.

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

    x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
    x = self.norm(x)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


# ============================================================
# Main Entry Point & Visualization Pipeline
# ============================================================

def main():
    """Run visual attention map extraction and overlay plot generation.

    Returns:
        None
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    cfg = OmegaConf.create({
        'peft': {'type': 'none', 'params': {}},
        'fusion': {'type': 'none', 'params': {}},
        'classifier': {'type': 'none', 'params': {}},
        'dataset': {
            'name': 'btxrd',
            'batch_size': 1,
            'num_workers': 0,
            'params': {
                'img_dir': 'data/BTXRD/images',
                'report_dir': 'data/BTXRD/reports',
                'csv_split_path': 'data/BTXRD/btxrd-split.csv',
                'csv_labels_path': 'data/BTXRD/btxrd-labels.csv',
                'task_type': 'multiclass',
                'num_classes': 10,
                'classes': ['normal', 'osteochondroma', 'osteosarcoma', 'multiple osteochondromas', 'simple bone cyst', 'other bt', 'giant cell tumor', 'synovial osteochondroma', 'other mt', 'osteofibroma']
            }
        },
        'model': {
            'backbone_type': 'biomedclip',
            'freeze_backbone': True
        },
        'experiment_name': 'ours_xbone_net',
        'seed': 42,
        'params': {
            'phase2': {
                'fusion_type': 'cross_attention',
                'classifier_type': 'prototypical'
            }
        }
    })

    print("Building model...")
    model = build_model(cfg.model).to(device)
    setup_phase2_modules(model, cfg, device)

    checkpoint_path = 'checkpoints/1_baseline/ours_xbone_net/seed_42/best_phase2.pth'
    print(f"Loading weights from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    state_dict = adapt_state_dict_keys(state_dict, list(model.state_dict().keys()))
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    print("Patching attention layer...")
    attn_module = model.backbone.model.visual.trunk.blocks[-1].attn
    attn_module.fused_attn = False
    attn_module.forward = types.MethodType(patched_attn_forward, attn_module)

    test_loader = build_dataloader(cfg.dataset, split="test", transform=model.backbone.preprocess, tokenizer=model.backbone.tokenizer)
    dataset = test_loader.dataset

    target_classes = {
        0: "normal",
        1: "osteochondroma",
        2: "osteosarcoma",
        6: "giant cell tumor"
    }

    candidates = {c: [] for c in target_classes}
    for idx in range(len(dataset)):
        row = dataset.df.iloc[idx]
        class_id = int(row['class_id'])
        if class_id in target_classes:
            candidates[class_id].append(idx)

    os.makedirs('results/visualization', exist_ok=True)

    print("\nStarting visual map generation...")
    for class_id, class_name in target_classes.items():
        indices = candidates[class_id]
        if not indices:
            print(f"No samples found for class: {class_name}")
            continue

        selected_idx = None
        best_prob = 0.0

        print(f"Analyzing candidates for class: {class_name} ({len(indices)} samples)...")
        for idx in indices[:10]:
            image_tensor, xray_ids, clinical_ids, labels = dataset[idx]

            img_in = image_tensor.unsqueeze(0).to(device)
            clin_in = clinical_ids.unsqueeze(0).to(device)

            captured_attention.clear()
            with torch.no_grad():
                logits = model(img_in, clin_in)
                probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

            pred_class = np.argmax(probs)
            if pred_class == class_id and probs[class_id] > best_prob:
                best_prob = probs[class_id]
                selected_idx = idx

        if selected_idx is None:
            selected_idx = indices[0]
            print(f"  [Fallback] No correct high-confidence prediction found. Using index {selected_idx}")
        else:
            print(f"  Selected index: {selected_idx} with confidence: {best_prob:.4f}")

        image_tensor, xray_ids, clinical_ids, labels = dataset[selected_idx]
        img_in = image_tensor.unsqueeze(0).to(device)
        clin_in = clinical_ids.unsqueeze(0).to(device)

        captured_attention.clear()
        with torch.no_grad():
            logits = model(img_in, clin_in)
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        row = dataset.df.iloc[selected_idx]
        image_id = str(row['image_id'])
        raw_img_path = os.path.join(dataset.img_dir, image_id)

        raw_img = Image.open(raw_img_path).convert('RGB')

        attn_matrix = captured_attention[0].squeeze(0)
        cls_attn = attn_matrix[:, 0, 1:].mean(dim=0).numpy()

        attn_map = cls_attn.reshape(14, 14)

        classes_list = cfg.dataset.params.classes
        gt_class_name = classes_list[class_id]
        pred_class_id = int(np.argmax(probs))
        pred_class_name = classes_list[pred_class_id]
        pred_prob = probs[pred_class_id]

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(raw_img)
        axes[0].set_title("Original X-ray Scan", fontsize=11)
        axes[0].axis('off')

        axes[1].imshow(raw_img)
        norm_attn = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
        axes[1].imshow(norm_attn, cmap='jet', alpha=0.5, extent=(0, raw_img.width, raw_img.height, 0), interpolation='bilinear')
        axes[1].set_title("Attention Overlay", fontsize=11)
        axes[1].axis('off')

        fig.suptitle(f"GT: {gt_class_name.upper()} | Pred: {pred_class_name.upper()} ({pred_prob*100:.1f}%)", 
                     fontsize=13, fontweight='bold', y=0.98, color='#1F2937')

        file_name_without_ext = os.path.splitext(image_id)[0]
        json_path = os.path.join("data/BTXRD/Annotations", f"{file_name_without_ext}.json")

        if os.path.exists(json_path):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    anno_data = json.load(f)
                shapes = anno_data.get('shapes', [])
                for shape in shapes:
                    points = shape.get('points', [])
                    shape_type = shape.get('shape_type', 'polygon')
                    if not points:
                        continue
                    pts = np.array(points)
                    for ax in axes:
                        if shape_type == 'rectangle':
                            x1, y1 = pts[0]
                            x2, y2 = pts[1]
                            rect = patches.Rectangle(
                                (min(x1, x2), min(y1, y2)),
                                abs(x2 - x1),
                                abs(y2 - y1),
                                linewidth=2.5,
                                edgecolor='#00FF00',
                                facecolor='none',
                                linestyle='--'
                            )
                            ax.add_patch(rect)
                        else:
                            poly = patches.Polygon(
                                pts,
                                closed=True,
                                linewidth=2.5,
                                edgecolor='#00FF00',
                                facecolor='none',
                                linestyle='--'
                            )
                            ax.add_patch(poly)
                print(f"  [GT Annotations] Drew {len(shapes)} shapes from {json_path}")
            except Exception as e:
                print(f"  [Warning] Failed to parse/draw annotation from {json_path}: {e}")

        safe_class_name = class_name.replace(" ", "_")
        output_file = f"results/visualization/attn_{safe_class_name}.png"
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  Saved visual map to: {output_file}")

        report_dest = f"document/report/images/attn_{safe_class_name}.png"
        shutil.copy(output_file, report_dest)
        print(f"  Copied to report images: {report_dest}")

        artifact_dir = "C:/Users/lebat/.gemini/antigravity/brain/9e6359b9-f81e-4515-b344-977de9a36b81"
        if os.path.exists(artifact_dir):
            artifact_dest = os.path.join(artifact_dir, f"attn_{safe_class_name}.png")
            shutil.copy(output_file, artifact_dest)
            print(f"  Copied to artifacts: {artifact_dest}")

    print("\nVisual maps successfully generated in results/visualization/")


if __name__ == "__main__":
    main()

