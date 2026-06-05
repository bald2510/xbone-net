import os
import torch
import hydra
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import confusion_matrix, hamming_loss
import evaluate

from models.builder import build_model
from local_datasets.builder import build_dataloader

def generate_custom_prompts(pathologies, image_context="a bone x-ray"):
    """
    Generate automatic prompts based on the list of pathologies.
    Structure: 'this is an image of {image_context}; {pathology} presented in image'
    """
    return {
        path: {
            "positive": f"this is an image of {image_context}; {path.lower()} presented in image",
            "negative": f"this is an image of {image_context}; no {path.lower()} presented in image"
        } for path in pathologies
    }

def adapt_state_dict_keys(state_dict, model_keys):
    """
    Automatically adjusts mismatched key prefixes between standard OpenCLIP/BiomedCLIP 
    and the wrapper XBone multimodal model.
    e.g., converts 'model.visual...' or 'visual...' to 'backbone.model.visual...'.
    """
    model_has_backbone_model = any(k.startswith("backbone.model.") for k in model_keys)
    checkpoint_keys = list(state_dict.keys())
    if not checkpoint_keys:
        return state_dict
        
    first_ckpt_key = checkpoint_keys[0]
    
    # Check if model has wrapper backbone structure but checkpoint doesn't
    if model_has_backbone_model and not first_ckpt_key.startswith("backbone.model."):
        # Case A: checkpoint keys start with 'model.' (e.g. standard open_clip)
        if first_ckpt_key.startswith("model."):
            print("Detected 'model.' prefix in checkpoint keys. Converting to 'backbone.model.' for compatibility.")
            adapted_state_dict = {}
            for k, v in state_dict.items():
                new_key = k.replace("model.", "backbone.model.", 1)
                adapted_state_dict[new_key] = v
            return adapted_state_dict
            
        # Case B: checkpoint keys start with 'visual.', 'transformer.', or 'text.'
        elif first_ckpt_key.startswith("visual.") or first_ckpt_key.startswith("transformer.") or first_ckpt_key.startswith("text."):
            print("Detected raw OpenCLIP submodule prefix in checkpoint keys. Prepending 'backbone.model.' for compatibility.")
            adapted_state_dict = {}
            for k, v in state_dict.items():
                new_key = "backbone.model." + k
                adapted_state_dict[new_key] = v
            return adapted_state_dict
            
    return state_dict

def load_checkpoint_from_dir(model, model_dir, device):
    """
    Loads model weights from a directory containing checkpoints.
    Searches for common checkpoint filenames (best_phase2.pth, best_phase1.pth, etc.).
    """
    if not os.path.isdir(model_dir):
        print(f"[Warning] model_dir '{model_dir}' does not exist or is not a directory.")
        return False
        
    # Search filenames in order of preference
    checkpoint_filenames = [
        "best_phase2.pth",       # Phase 2 classifier checkpoint (Full Model)
        "best_phase1.pth",       # Phase 1 LoRA checkpoint
        "best_semantic_lora.pth",# Legacy naming
        "open_clip_pytorch_model.bin", # Baseline open_clip weights
        "checkpoint_epoch_10.pth" # Default name
    ]
    
    checkpoint_path = None
    for filename in checkpoint_filenames:
        path = os.path.join(model_dir, filename)
        if os.path.exists(path):
            checkpoint_path = path
            break
            
    # Fallback to any .pth or .bin file in the directory
    if not checkpoint_path:
        for file in os.listdir(model_dir):
            if file.endswith(".pth") or file.endswith(".bin"):
                checkpoint_path = os.path.join(model_dir, file)
                break
                
    if checkpoint_path:
        print(f"Loading weights from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        # Adapt keys for OpenCLIP/backbone compatibility
        state_dict = adapt_state_dict_keys(state_dict, model.state_dict().keys())
        model.load_state_dict(state_dict, strict=False)
        print("-> Checkpoint loaded successfully from directory!\n")
        return True
    else:
        print(f"[Warning] No checkpoint file (.pth or .bin) found in directory '{model_dir}'")
        return False

@hydra.main(version_base=None, config_path="configs", config_name="experiment/experiment2/baseline/fracatlas_biomedclip_baseline.yaml")
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting inference evaluation on device: {device}")
    
    # =====================================================================
    # 1. INITIALIZE MODEL & LOAD CHECKPOINT
    # =====================================================================
    print("Building model architecture...")
    model = build_model(cfg.model).to(device)
    model.eval()
    
    loaded = False
    params_cfg = cfg.get("params", {}) or {}
    model_dir = params_cfg.get("model_dir", None)
    
    # Try loading from specified directory first
    if model_dir:
        print(f"Searching for checkpoints in folder: {model_dir}")
        loaded = load_checkpoint_from_dir(model, model_dir, device)
        
    if not loaded:
        # Fallback to traditional checkpoint paths if model_dir is not specified or load fails
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
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            # Adapt keys for OpenCLIP/backbone compatibility
            state_dict = adapt_state_dict_keys(state_dict, model.state_dict().keys())
            model.load_state_dict(state_dict, strict=False)
            print("-> Checkpoint loaded successfully from file!\n")
        else:
            print("-> [Info] No checkpoint found or defined. Running model with initial/random weights.\n")
        
    preprocess = model.backbone.preprocess
    tokenizer = model.backbone.tokenizer
    
    # =====================================================================
    # 2. INITIALIZE DATALOADER
    # =====================================================================
    print(f"Loading test dataset: {cfg.dataset.name}...")
    test_loader = build_dataloader(
        cfg=cfg.dataset, 
        split="test", 
        transform=preprocess, 
        tokenizer=tokenizer
    )
    
    pathologies = cfg.dataset.params.pathologies
    is_classifier = cfg.model.classifier.type != "none"
    
    # =====================================================================
    # 3. PRE-EXTRACT PROMPTS (Zero-shot / LoRA semantic alignment mode)
    # =====================================================================
    text_features_dict = {}
    if not is_classifier:
        print("\nZero-shot mode: Pre-extracting text prompt features...")
        image_context = cfg.get("params", {}).get("image_context", "a bone x-ray")
        prompt_dict = generate_custom_prompts(pathologies, image_context)
        print(f"Generated prompts for pathologies: {list(prompt_dict.values())}")
        with torch.no_grad():
            for path, pair in prompt_dict.items():
                pos_tokens = tokenizer([pair["positive"]]).to(device)
                neg_tokens = tokenizer([pair["negative"]]).to(device)
                
                pos_feat = model.backbone.model.encode_text(pos_tokens)
                neg_feat = model.backbone.model.encode_text(neg_tokens)
                
                pos_feat /= pos_feat.norm(dim=-1, keepdim=True)
                neg_feat /= neg_feat.norm(dim=-1, keepdim=True)
                
                text_features_dict[path] = torch.cat([pos_feat, neg_feat], dim=0)

    # =====================================================================
    # 4. RUN TEST EVALUATION
    # =====================================================================
    all_probs = []
    all_ground_truths = []
    
    temp = cfg.get("params", {}).get("temperature", 0.07)
    
    print("\nScanning test dataset...")
    with torch.no_grad():
        for images, input_ids, labels in tqdm(test_loader):
            images = images.to(device)
            if isinstance(input_ids, torch.Tensor):
                input_ids = input_ids.to(device)
            
            if is_classifier:
                # Forward through the full model
                outputs = model(images, input_ids)
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs
                batch_probs = torch.sigmoid(logits)
            else:
                # Cosine similarity matching
                img_feat = model.backbone.model.encode_image(images)
                img_feat /= img_feat.norm(dim=-1, keepdim=True)
                
                batch_probs = []
                for path in pathologies:
                    text_weights = text_features_dict[path]
                    logits = img_feat @ text_weights.T
                    probs = torch.softmax(logits / temp, dim=-1)[:, 0].unsqueeze(1)
                    batch_probs.append(probs)
                
                batch_probs = torch.cat(batch_probs, dim=1) 
    
            all_probs.append(batch_probs.cpu())
            all_ground_truths.append(labels.cpu())

    all_probs = torch.cat(all_probs, dim=0).numpy()
    all_ground_truths = torch.cat(all_ground_truths, dim=0).numpy()

    # =====================================================================
    # 5. METRIC CALCULATION (USING HUGGING FACE EVALUATE LIBRARY)
    # =====================================================================
    results = []
    print("\n=== DETAILED METRICS EVALUATION ===")
    
    # Load Hugging Face evaluate metrics
    accuracy_metric = evaluate.load("accuracy")
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    f1_metric = evaluate.load("f1")
    roc_auc_metric = evaluate.load("roc_auc")
    
    # Determine if dataset is multi-label or single-label
    is_multilabel = (cfg.dataset.name.lower() == 'btxrd') or (len(pathologies) > 1)
    
    pathology_results = []
    for idx, path in enumerate(pathologies):
        gt_labels = all_ground_truths[:, idx]
        probs = all_probs[:, idx]
        
        if len(np.unique(gt_labels)) < 2:
            print(f"[Warning] Pathology '{path}' skipped: only 1 class present in test set.")
            continue
            
        preds = (probs >= 0.5).astype(int)
        
        # Specificity via confusion matrix
        tn, fp, fn, tp = confusion_matrix(gt_labels, preds).ravel()
        print(f"\nConfusion Matrix for {path}:")
        print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        # Compute metrics using Hugging Face evaluate
        accuracy = accuracy_metric.compute(predictions=preds, references=gt_labels)["accuracy"]
        precision = precision_metric.compute(predictions=preds, references=gt_labels)["precision"]
        recall = recall_metric.compute(predictions=preds, references=gt_labels)["recall"]
        f1 = f1_metric.compute(predictions=preds, references=gt_labels)["f1"]
        auroc = roc_auc_metric.compute(prediction_scores=probs, references=gt_labels)["roc_auc"]
        
        pathology_results.append({
            "Pathology": path, 
            "AUROC": auroc,
            "Accuracy": accuracy,
            "F1_Score": f1,
            "Precision": precision,
            "Recall_Sens": recall,
            "Specificity": specificity
        })
        
        print(f"{path:25s} | AUC: {auroc:.3f} | Acc: {accuracy:.3f} | F1: {f1:.3f} | Prec: {precision:.3f} | Rec: {recall:.3f} | Spec: {specificity:.3f}")

    if is_multilabel:
        print("\n=== OVERALL MULTI-LABEL METRICS ===")
        all_preds = (all_probs >= 0.5).astype(int)
        
        # Compute overall multi-label metrics using HF evaluate
        subset_acc = accuracy_metric.compute(predictions=all_preds, references=all_ground_truths)["accuracy"]
        h_loss = hamming_loss(all_ground_truths, all_preds)
        
        micro_f1 = f1_metric.compute(predictions=all_preds, references=all_ground_truths, average='micro', zero_division=0)["f1"]
        macro_f1 = f1_metric.compute(predictions=all_preds, references=all_ground_truths, average='macro', zero_division=0)["f1"]
        
        try:
            micro_auroc = roc_auc_metric.compute(prediction_scores=all_probs, references=all_ground_truths, average='micro')["roc_auc"]
        except ValueError:
            micro_auroc = float('nan')
            
        try:
            macro_auroc = roc_auc_metric.compute(prediction_scores=all_probs, references=all_ground_truths, average='macro')["roc_auc"]
        except ValueError:
            macro_auroc = float('nan')
            
        print(f"Subset Accuracy (Exact Match) : {subset_acc:.4f}")
        print(f"Hamming Loss                  : {h_loss:.4f}")
        print(f"Micro F1-Score                : {micro_f1:.4f}")
        print(f"Macro F1-Score                : {macro_f1:.4f}")
        print(f"Micro AUROC                   : {micro_auroc:.4f}")
        print(f"Macro AUROC                   : {macro_auroc:.4f}")
    else:
        print("\n=== SINGLE-LABEL METRICS SUMMARY ===")
        if len(pathology_results) > 0:
            res = pathology_results[0]
            print(f"Accuracy            : {res['Accuracy']:.4f}")
            print(f"Precision           : {res['Precision']:.4f}")
            print(f"Recall (Sensitivity): {res['Recall_Sens']:.4f}")
            print(f"Specificity         : {res['Specificity']:.4f}")
            print(f"F1-Score            : {res['F1_Score']:.4f}")
            print(f"AUROC               : {res['AUROC']:.4f}")
        else:
            print("No evaluation results computed.")

    # =====================================================================
    # 6. PRINT SAMPLE PREDICTIONS
    # =====================================================================
    num_samples = len(all_ground_truths)
    sample_idx = random.randint(0, num_samples - 1)
    
    print(f"\n=== DETAILED PREDICTIONS FOR SAMPLE #{sample_idx} ===")
    print(f"{'Pathology / Property':<25} | {'Ground Truth':<12} | {'Prediction':<10} | {'Probability':<10} | {'Status'}")
    print("-" * 75)
    
    for i, path in enumerate(pathologies):
        prob = all_probs[sample_idx, i]
        gt = int(all_ground_truths[sample_idx, i])
        pred = 1 if prob >= 0.5 else 0
        
        match_status = "Correct" if pred == gt else "Incorrect"
        
        if gt == 1 or pred == 1:
            print(f"> {path:<23} | {gt:<12} | {pred:<10} | {prob:<10.4f} | {match_status}")
        else:
            print(f"  {path:<23} | {gt:<12} | {pred:<10} | {prob:<10.4f} | {match_status}")

if __name__ == "__main__":
    main()
