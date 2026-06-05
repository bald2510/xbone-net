import os
import torch
import torch.nn as nn
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.optim import AdamW
from transformers import Trainer, TrainingArguments

# Import builders from project setup
from models.builder import build_model
from local_datasets.builder import build_dataloader
from utils.losses import build_loss

def seed_everything(seed=42):
    """Cố định random seed để đảm bảo tính tái lập."""
    import random
    import numpy as np
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

class HuggingFaceDatasetWrapper(torch.utils.data.Dataset):
    """
    Wrapper to convert standard tuple-based PyTorch Datasets
    into dictionary-based Datasets expected by Hugging Face Trainer.
    """
    def __init__(self, dataset):
        self.dataset = dataset
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        image, input_ids, labels = self.dataset[idx]
        return {
            "pixel_values": image,
            "input_ids": input_ids,
            "labels": labels
        }

class XBoneTrainer(Trainer):
    """
    Custom Hugging Face Trainer subclass that overrides compute_loss
    to support two-stage training logic without changing the core model.
    """
    def __init__(self, phase, loss_fn, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.phase = phase
        self.loss_fn = loss_fn

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        images = inputs["pixel_values"]
        input_ids = inputs["input_ids"]
        labels = inputs["labels"]
        
        if self.phase == "phase1":
            # Phase 1: Contrastive semantic matching from backbone features
            image_features, text_features = model.backbone(images, input_ids)
            loss = self.loss_fn(image_features, text_features, labels)
            outputs = {"image_features": image_features, "text_features": text_features}
        else:
            # Phase 2: Classification using fusion & head logits
            outputs = model(images, input_ids)
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
            loss = self.loss_fn(logits, labels)
            
        return (loss, outputs) if return_outputs else loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: dict,
        prediction_loss_only: bool,
        ignore_keys: list = None,
    ):
        """
        Overrides Trainer.prediction_step to compute validation predictions and loss
        using our custom compute_loss, bypassing model(**inputs).
        """
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
            
        logits = None
        if not prediction_loss_only:
            if self.phase == "phase2":
                logits = outputs[0] if isinstance(outputs, tuple) else outputs
                
        labels = inputs.get("labels", None)
        return (loss, logits, labels)


# =====================================================================
# HÀM CHÍNH (MAIN FUNCTION)
# =====================================================================
@hydra.main(version_base=None, config_path="configs", config_name="experiment/experiment2/finetune/fracatlas_biomedclip_lora_r16.yaml")
def main(cfg: DictConfig):
    print("=== CẤU HÌNH THÍ NGHIỆM ĐÃ ĐƯỢC GỘP ===")
    print(OmegaConf.to_yaml(cfg))
    print("========================================\n")

    seed_everything(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}\n")

    # 1. Khởi tạo mô hình đầy đủ từ Config (Có peft, fusion, classifier)
    print("Đang lắp ráp mô hình...")
    model = build_model(cfg.model).to(device)
    print("Lắp ráp mô hình thành công!\n")

    # 2. Khởi tạo DataLoaders
    print("Đang khởi tạo DataLoaders...")
    preprocess = model.backbone.preprocess
    tokenizer = model.backbone.tokenizer
    
    train_loader = build_dataloader(cfg=cfg.dataset, split="train", transform=preprocess, tokenizer=tokenizer)
    val_loader = build_dataloader(cfg=cfg.dataset, split="val", transform=preprocess, tokenizer=tokenizer)
    print("Khởi tạo dữ liệu hoàn tất!\n")

    # 2.5 Sanity Check: Kiểm tra dữ liệu đầu vào của dataloader
    print("=== KIỂM TRA ĐẦU VÀO DATALOADER (SANITY CHECK) ===")
    try:
        sample_batch = next(iter(train_loader))
        images_check, input_ids_check, labels_check = sample_batch
        print(f" -> Kích thước Batch: Ảnh {images_check.shape} | Token IDs {input_ids_check.shape} | Nhãn {labels_check.shape}")
        
        # Decode và in thử prompt của mẫu đầu tiên
        if hasattr(tokenizer, 'decode'):
            decoded_prompt = tokenizer.decode(input_ids_check[0])
            print(f" -> Mẫu Text đầu tiên (Decoded): '{decoded_prompt}'")
        else:
            print(f" -> Mẫu Text đầu tiên (Raw): '{input_ids_check[0]}'")
            
        print(f" -> Giá trị nhãn mẫu đầu tiên: {labels_check[0].tolist()}")
    except Exception as e:
        print(f" -> [LỖI] Không thể đọc mẫu từ dataloader: {e}")
    print("==================================================\n")

    # Lấy các tham số cấu hình của từng Phase (hỗ trợ fallback nếu thiếu)
    params_cfg = cfg.get("params", {}) or {}
    run_phase1 = params_cfg.get("run_phase1", True)
    run_phase2 = params_cfg.get("run_phase2", True)

    p1_cfg = params_cfg.get("phase1", {}) or {}
    epochs_p1 = p1_cfg.get("epochs", params_cfg.get("epochs", 100))
    lr_p1 = p1_cfg.get("lr", params_cfg.get("lr", 1e-4))
    wd_p1 = p1_cfg.get("weight_decay", params_cfg.get("weight_decay", 1e-2))
    cp_p1 = p1_cfg.get("checkpoint_path", "checkpoints/best_phase1.pth")
    temp_p1 = p1_cfg.get("temperature", params_cfg.get("temperature", 0.07))

    p2_cfg = params_cfg.get("phase2", {}) or {}
    epochs_p2 = p2_cfg.get("epochs", params_cfg.get("epochs", 10))
    lr_p2 = p2_cfg.get("lr", params_cfg.get("lr", 1e-4))
    wd_p2 = p2_cfg.get("weight_decay", params_cfg.get("weight_decay", 1e-2))
    cp_p2 = p2_cfg.get("checkpoint_path", params_cfg.get("checkpoint_path", "checkpoints/best_phase2.pth"))

    # Create checkpoint directories if they do not exist
    os.makedirs(os.path.dirname(cp_p1) if os.path.dirname(cp_p1) else ".", exist_ok=True)
    os.makedirs(os.path.dirname(cp_p2) if os.path.dirname(cp_p2) else ".", exist_ok=True)

    # Tự động phát hiện hỗ trợ kiểu dữ liệu (precision auto-detection)
    use_bf16 = False
    use_fp16 = False
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            use_bf16 = True
            print(" -> [Config] Phát hiện phần cứng hỗ trợ BF16. Kích hoạt BF16 training.")
        else:
            use_fp16 = True
            print(" -> [Config] Phát hiện phần cứng hỗ trợ FP16. Kích hoạt FP16 training.")
    else:
        print(" -> [Config] Không có GPU khả dụng. Chạy ở chế độ FP32 mặc định.")

    # Đọc cấu hình gradient checkpointing từ params (mặc định là True để tiết kiệm bộ nhớ)
    gradient_checkpointing = params_cfg.get("gradient_checkpointing", True)
    print(f" -> [Config] Gradient Checkpointing: {gradient_checkpointing}\n")

    # Wrap dataset instances for HF Trainer compatibility
    train_dataset = HuggingFaceDatasetWrapper(train_loader.dataset)
    val_dataset = HuggingFaceDatasetWrapper(val_loader.dataset)

    # =====================================================================
    # PHASE 1: LO-RA FINE-TUNING (SEMANTIC MATCHING)
    # =====================================================================
    if run_phase1:
        print("\n" + "="*50)
        print("🚀 BẮT ĐẦU PHA 1: HUẤN LUYỆN SEMANTIC MATCHING VỚI LoRA (HF TRAINER)")
        print("="*50)

        # Cấu hình requires_grad: Chỉ cho phép LoRA weights được cập nhật
        for name, param in model.named_parameters():
            if "lora" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        print("Tóm tắt thông số cho Pha 1:")
        model.print_parameter_summary()

        loss_type = p1_cfg.get("loss_type", "semantic_matching")
        print(f"Khởi tạo loss function: {loss_type}")
        criterion_p1 = build_loss(loss_type, temperature=temp_p1)
        
        # Chỉ tối ưu hóa các tham số LoRA (requires_grad=True) + logit_scale & bias của loss (nếu có)
        trainable_params_p1 = [p for p in model.parameters() if p.requires_grad]
        if hasattr(criterion_p1, 'logit_scale') and isinstance(criterion_p1.logit_scale, nn.Parameter):
            trainable_params_p1.append(criterion_p1.logit_scale)
        if hasattr(criterion_p1, 'bias') and isinstance(criterion_p1.bias, nn.Parameter):
            trainable_params_p1.append(criterion_p1.bias)

        optimizer_p1 = AdamW(trainable_params_p1, lr=lr_p1, weight_decay=wd_p1)

        # Configure TrainingArguments for Phase 1
        p1_args = TrainingArguments(
            output_dir=os.path.dirname(cp_p1) if os.path.dirname(cp_p1) else "./checkpoints",
            num_train_epochs=epochs_p1,
            learning_rate=lr_p1,
            weight_decay=wd_p1,
            per_device_train_batch_size=cfg.dataset.batch_size,
            per_device_eval_batch_size=cfg.dataset.batch_size,
            eval_strategy="epoch" if epochs_p1 > 0 else "no",
            save_strategy="epoch" if epochs_p1 > 0 else "no",
            logging_strategy="epoch" if epochs_p1 > 0 else "no",
            load_best_model_at_end=True if epochs_p1 > 0 else False,
            metric_for_best_model="loss",
            greater_is_better=False,
            remove_unused_columns=False,
            seed=cfg.get("seed", 42),
            dataloader_num_workers=cfg.dataset.num_workers,
            lr_scheduler_type="constant",
            bf16=use_bf16,
            fp16=use_fp16,
            gradient_checkpointing=gradient_checkpointing,
        )

        trainer_p1 = XBoneTrainer(
            phase="phase1",
            loss_fn=criterion_p1,
            model=model,
            args=p1_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            optimizers=(optimizer_p1, None)
        )

        if epochs_p1 > 0:
            trainer_p1.train()
            
        # Lưu checkpoint Phase 1 tốt nhất để dùng cho Phase 2 hoặc inference
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': OmegaConf.to_container(cfg, resolve=True)
        }, cp_p1)
        print(f" [*] Đã lưu checkpoint Phase 1 tốt nhất tại: {cp_p1}")
        print("Pha 1 hoàn tất!\n")
    else:
        print("\n[Bỏ qua] Không chạy Pha 1 huấn luyện LoRA.")

    # =====================================================================
    # TRANSITION: TẢI TRỌNG SỐ TỐT NHẤT TỪ PHASE 1
    # =====================================================================
    if os.path.exists(cp_p1):
        print(f"Đang tải trọng số tốt nhất từ Pha 1 tại: {cp_p1}")
        checkpoint = torch.load(cp_p1, map_location=device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        # Sử dụng strict=False để có thể tải LoRA weights ngay cả khi model có thêm classifier/fusion module
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        print(" -> Tải thành công weights từ Pha 1!")
        if len(missing_keys) > 0:
            print(f"    (Lưu ý: Thiếu các weights: {list(missing_keys)[:5]}... - Điều này bình thường vì Phase 1 không lưu classifier head)")
    else:
        print(f"\n[Cảnh báo] Không tìm thấy checkpoint Pha 1 tại: {cp_p1}. Sử dụng trọng số khởi tạo ban đầu.")

    # =====================================================================
    # PHASE 2: CLASSIFIER TRAINING
    # =====================================================================
    if run_phase2:
        print("\n" + "="*50)
        print("🚀 BẮT ĐẦU PHA 2: HUẤN LUYỆN CLASSIFIER HEAD & FUSION MODULE (HF TRAINER)")
        print("="*50)

        # Cấu hình requires_grad: Đóng băng toàn bộ backbone (bao gồm cả LoRA), chỉ unfreeze fusion và head
        for param in model.backbone.parameters():
            param.requires_grad = False
            
        if model.fusion is not None:
            for param in model.fusion.parameters():
                param.requires_grad = True
                
        if model.head is not None:
            for param in model.head.parameters():
                param.requires_grad = True

        print("Tóm tắt thông số cho Pha 2 (Chỉ train Classifier & Fusion):")
        model.print_parameter_summary()

        criterion_p2 = nn.BCEWithLogitsLoss()
        
        trainable_params_p2 = [p for p in model.parameters() if p.requires_grad]
        optimizer_p2 = AdamW(trainable_params_p2, lr=lr_p2, weight_decay=wd_p2)

        # Configure TrainingArguments for Phase 2
        p2_args = TrainingArguments(
            output_dir=os.path.dirname(cp_p2) if os.path.dirname(cp_p2) else "./checkpoints",
            num_train_epochs=epochs_p2,
            learning_rate=lr_p2,
            weight_decay=wd_p2,
            per_device_train_batch_size=cfg.dataset.batch_size,
            per_device_eval_batch_size=cfg.dataset.batch_size,
            eval_strategy="epoch" if epochs_p2 > 0 else "no",
            save_strategy="epoch" if epochs_p2 > 0 else "no",
            logging_strategy="epoch" if epochs_p2 > 0 else "no",
            load_best_model_at_end=True if epochs_p2 > 0 else False,
            metric_for_best_model="loss",
            greater_is_better=False,
            remove_unused_columns=False,
            seed=cfg.get("seed", 42),
            dataloader_num_workers=cfg.dataset.num_workers,
            lr_scheduler_type="constant",
            bf16=use_bf16,
            fp16=use_fp16,
            gradient_checkpointing=gradient_checkpointing,
        )

        trainer_p2 = XBoneTrainer(
            phase="phase2",
            loss_fn=criterion_p2,
            model=model,
            args=p2_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            optimizers=(optimizer_p2, None)
        )

        if epochs_p2 > 0:
            trainer_p2.train()
            
        # Lưu checkpoint Phase 2 tốt nhất
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': OmegaConf.to_container(cfg, resolve=True)
        }, cp_p2)
        print(f" [*] Đã lưu checkpoint Phase 2 tốt nhất tại: {cp_p2}")
        print("Pha 2 hoàn tất thành công!\n")
    else:
        print("\n[Bỏ qua] Không chạy Pha 2 huấn luyện Classifier.")

    print("Toàn bộ quá trình chạy hoàn tất thành công!")

if __name__ == "__main__":
    main()
