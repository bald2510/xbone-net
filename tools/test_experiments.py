import os
import torch
import sys
import warnings
from omegaconf import OmegaConf
from hydra import initialize, compose

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

from src.datasets.builder import build_dataloader
from src.models.builder import build_model
from src.utils.losses import build_loss
from tools.run_all import EXPERIMENTS
from hydra.core.hydra_config import HydraConfig
from hydra.conf import HydraConf, RuntimeConf

# Initialize HydraConfig globally so interpolations work
hc = HydraConf(runtime=RuntimeConf(cwd=os.getcwd()))
HydraConfig.instance().set_config(OmegaConf.create({'hydra': hc}))

def dry_run_experiment(experiment_path: str):
    """Perform a dry-run of a single experiment to verify model and pipeline."""
    print(f"[{experiment_path}] Initialization...")
    try:
        # Load Hydra config for this experiment
        cfg = compose(config_name="config", overrides=[f"+experiment={experiment_path}"])
        OmegaConf.resolve(cfg)
        
        # 1. Build Model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = build_model(cfg.model)
        model.to(device)
        model.train()
        
        tokenizer_func = getattr(model.backbone, "tokenizer_obj", getattr(model.backbone, "tokenizer", None))
        preprocess_func = getattr(model.backbone, "preprocess", None)
        print(f"[{experiment_path}] Extracted preprocess_func: {preprocess_func}")
        print(f"[{experiment_path}] Extracted tokenizer_func: {tokenizer_func}")

        # 2. Build Dataloader (use a tiny batch size and 0 workers to save time)
        cfg.dataset.batch_size = 2
        cfg.dataset.num_workers = 0
        train_loader = build_dataloader(cfg.dataset, split="train", transform=preprocess_func, tokenizer=tokenizer_func)
        
        # Get one batch
        batch = next(iter(train_loader))
        
        # Move batch to device
        if len(batch) == 4:
            images, xray_ids, clinical_ids, labels = batch
            texts = {"xray_input_ids": xray_ids, "clinical_input_ids": clinical_ids}
        else:
            images, texts, labels = batch
        
        images = images.to(device)
        if isinstance(texts, dict):
            texts = {k: v.to(device) for k, v in texts.items()}
        elif isinstance(texts, torch.Tensor):
            texts = texts.to(device)
        labels = labels.to(device)
        
        # 3. Test Phase 1 (if enabled)
        p1_cfg = cfg.params.get("phase1", {})
        if p1_cfg.get("run_phase1", False):
            print(f"[{experiment_path}] Testing Phase 1 (Contrastive)...")
            
            # Freeze fusion and head
            for param in model.fusion.parameters():
                param.requires_grad = False
            for param in model.head.parameters():
                param.requires_grad = False
                
            loss_fn_p1 = build_loss(
                p1_cfg.get("loss_type", "semantic_matching"), 
                clip_model=model.backbone.model if hasattr(model.backbone, "model") else None, 
                temperature=p1_cfg.get("temperature", 0.07),
                target_similarity=p1_cfg.get("target_similarity", 0.7)
            )
            
            p1_report_type = p1_cfg.get("p1_report_type", "both")
            labels_for_loss = labels if p1_cfg.get("loss_type") == "supervised_contrastive" else None
            
            # Forward pass
            if p1_report_type == "both":
                xray_input_ids = texts["xray_input_ids"]
                clinical_input_ids = texts["clinical_input_ids"]
                
                _, xray_feat = model.backbone(images, xray_input_ids)
                _, clinical_feat = model.backbone(images, clinical_input_ids)
                text_features = (xray_feat + clinical_feat) / 2.0
                image_features = xray_feat # Just need some valid tensor for loss test
            else:
                is_xray = (p1_report_type == "xray")
                text_ids = inputs["xray_input_ids"] if is_xray else inputs["clinical_input_ids"]
                text_mask = inputs.get("xray_attention_mask" if is_xray else "clinical_attention_mask")
                image_features, text_features = model.backbone(images, text_ids, attention_mask=text_mask)
                
            loss = loss_fn_p1(image_features, text_features, labels_for_loss)
            
            # Backward pass
            loss.backward()
            
            # Check if any backbone parameter received gradients
            has_grads = any(p.grad is not None for p in model.backbone.parameters() if p.requires_grad)
            if not has_grads and any(p.requires_grad for p in model.backbone.parameters()):
                print(f"[{experiment_path}] WARNING: Phase 1 backward pass produced no gradients in backbone!")
        
        # 4. Test Phase 2 (if enabled)
        p2_cfg = cfg.params.get("phase2", {})
        if cfg.params.get("run_phase2", False) and p2_cfg.get("enabled", True):
            print(f"[{experiment_path}] Testing Phase 2 (Classification)...")
            
            # Freeze backbone
            for param in model.backbone.parameters():
                param.requires_grad = False
                
            # Unfreeze fusion and head
            for param in model.fusion.parameters():
                param.requires_grad = True
            for param in model.head.parameters():
                param.requires_grad = True
                
            loss_fn_p2 = torch.nn.CrossEntropyLoss()
            
            # Forward pass
            use_text_in_p2 = p2_cfg.get("use_text", True)
            p2_report_type = p2_cfg.get("p2_report_type", "both")
            
            if use_text_in_p2:
                if p2_report_type == "both":
                    # Simple average for dual reports, same as trainer
                    xray_input_ids = texts["xray_input_ids"]
                    clinical_input_ids = texts["clinical_input_ids"]
                    _, xray_feat = model.backbone(images, xray_input_ids)
                    _, clinical_feat = model.backbone(images, clinical_input_ids)
                    text_input = (xray_feat + clinical_feat) / 2.0
                else:
                    text_input = texts["xray_input_ids"] if p2_report_type == "xray" else texts["clinical_input_ids"]
                logits = model(images, text_input)
            else:
                logits = model(images)
                
            loss = loss_fn_p2(logits, labels)
            
            # Backward pass
            loss.backward()
            
            has_grads = any(p.grad is not None for p in model.fusion.parameters()) or any(p.grad is not None for p in model.head.parameters())
            if not has_grads:
                print(f"[{experiment_path}] WARNING: Phase 2 backward pass produced no gradients in head/fusion!")

        print(f"[{experiment_path}] OK!")
        return True
    
    except Exception as e:
        print(f"[{experiment_path}] FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("==================================================")
    print(" XBone-Net Dry-Run Experiment Tester")
    print("==================================================")
    
    initialize(config_path="../configs", version_base=None)
    
    success_count = 0
    total_count = 0
    failed_experiments = []
    
    # We will only test 1 representative experiment from each group to save time,
    # or you can test all of them. Let's test all of them quickly!
    for group_name, exp_list in EXPERIMENTS.items():
        print(f"\n--- Group: {group_name} ---")
        for exp in exp_list:
            total_count += 1
            if dry_run_experiment(exp):
                success_count += 1
            else:
                failed_experiments.append(exp)
                
    print("\n==================================================")
    print(f"Summary: {success_count}/{total_count} experiments passed dry-run.")
    if failed_experiments:
        print("Failed experiments:")
        for exp in failed_experiments:
            print(f"  - {exp}")
    else:
        print("All experiments passed!")
    print("==================================================")


if __name__ == "__main__":
    main()
