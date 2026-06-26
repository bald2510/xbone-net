import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import os
import torch
import numpy as np

# Apply monkeypatches first
import transformers
from transformers import CLIPImageProcessor
import transformers.processing_utils

if 'feature_extractor' in transformers.processing_utils.MODALITY_TO_BASE_CLASS_MAPPING:
    transformers.processing_utils.MODALITY_TO_BASE_CLASS_MAPPING['feature_extractor'] = (
        'FeatureExtractionMixin', 'ImageProcessingMixin'
    )

original_clip_init = CLIPImageProcessor.__init__
def wrapped_clip_init(self, *args, **kwargs):
    arg_names = [
        "do_resize", "size", "resample", "do_center_crop", 
        "crop_size", "do_normalize", "image_mean", "image_std", 
        "do_convert_rgb"
    ]
    new_kwargs = dict(kwargs)
    for name, val in zip(arg_names, args):
        new_kwargs[name] = val
    return original_clip_init(self, **new_kwargs)

CLIPImageProcessor.__init__ = wrapped_clip_init
sys.modules['transformers'].CLIPFeatureExtractor = CLIPImageProcessor
transformers.CLIPFeatureExtractor = CLIPImageProcessor

# Set up python path to import models
sys.path.append(".")

from models.builder import build_model, setup_phase2_modules
from local_datasets.builder import build_dataloader
from omegaconf import OmegaConf

# Load configs
cfg = OmegaConf.load("configs/experiment/1_baseline/B2_medclip.yaml")
dataset_cfg = OmegaConf.load("configs/dataset/btxrd.yaml")

# Merge
cfg.dataset = dataset_cfg

# Manually populate paths in config
cfg.dataset.params.img_dir = "data/BTXRD/images"
cfg.dataset.params.report_dir = "data/BTXRD/reports"
cfg.dataset.params.csv_split_path = "data/BTXRD/btxrd-split.csv"
cfg.dataset.params.csv_labels_path = "data/BTXRD/btxrd-labels.csv"
cfg.seed = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Build model
model = build_model(cfg.model).to(device)
model, classifier_type, fusion_type, num_classes = setup_phase2_modules(model, cfg, device)
model.eval()

checkpoint_path = "checkpoints/1_baseline/B2_medclip/seed_42/best_phase2.pth"
if os.path.exists(checkpoint_path):
    print("Loading checkpoint from:", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    # Adapt keys
    adapted_dict = {}
    model_keys = list(model.state_dict().keys())
    for k, v in state_dict.items():
        if not k.startswith("backbone.model.") and any(mk.startswith("backbone.model.") for mk in model_keys):
            if k.startswith("model."):
                adapted_dict[k.replace("model.", "backbone.model.", 1)] = v
            elif k.startswith(("visual.", "transformer.", "text.")):
                adapted_dict["backbone.model." + k] = v
            else:
                adapted_dict[k] = v
        else:
            adapted_dict[k] = v
    model.load_state_dict(adapted_dict, strict=False)
    print("Checkpoint loaded successfully.")

# Dataloader
test_loader = build_dataloader(
    cfg=cfg.dataset,
    split="test",
    transform=model.backbone.preprocess,
    tokenizer=model.backbone.tokenizer,
)

print("\n--- Testing 3 Samples ---")
with torch.no_grad():
    for i, batch in enumerate(test_loader):
        if i >= 1:
            break
        images, xray_ids, clinical_ids, labels = batch
        images = images.to(device)
        clinical_ids = clinical_ids.to(device)
        labels = labels.to(device)
        
        # Test sample 0
        img = images[0:1]
        txt = clinical_ids[0:1]
        lbl = labels[0:1]
        
        # 1. Normal forward
        img_feat, txt_feat = model.backbone(img, txt)
        fused_feat = model.fusion(img_feat, txt_feat)
        logits = model.head(fused_feat)
        probs = torch.softmax(logits, dim=-1)
        
        print(f"\n[Sample 0] Ground Truth Class: {lbl.item()}")
        print(f"Probabilities (Normal): {probs[0].cpu().numpy()}")
        print(f"Predicted Class (Normal): {torch.argmax(probs, dim=-1).item()}")
        
        # 2. No image (zero image features)
        zero_img_feat = torch.zeros_like(img_feat)
        fused_feat_no_img = model.fusion(zero_img_feat, txt_feat)
        logits_no_img = model.head(fused_feat_no_img)
        probs_no_img = torch.softmax(logits_no_img, dim=-1)
        print(f"Probabilities (No Image): {probs_no_img[0].cpu().numpy()}")
        print(f"Predicted Class (No Image): {torch.argmax(probs_no_img, dim=-1).item()}")
        
        # 3. No text (zero text features)
        zero_txt_feat = torch.zeros_like(txt_feat)
        fused_feat_no_txt = model.fusion(img_feat, zero_txt_feat)
        logits_no_txt = model.head(fused_feat_no_txt)
        probs_no_txt = torch.softmax(logits_no_txt, dim=-1)
        print(f"Probabilities (No Text): {probs_no_txt[0].cpu().numpy()}")
        print(f"Predicted Class (No Text): {torch.argmax(probs_no_txt, dim=-1).item()}")

        # 4. Check norms of features
        print(f"Image features norm: {img_feat.norm(dim=-1).item():.4f}")
        print(f"Text features norm: {txt_feat.norm(dim=-1).item():.4f}")
