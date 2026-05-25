import os
import torch
import hydra
import numpy as np
import pandas as pd
from tqdm import tqdm
from omegaconf import DictConfig
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

from models.builder import build_model
from datasets.builder import build_dataloader

def generate_medical_prompts(pathologies):
    """Tạo câu lệnh tự động dựa vào danh sách bệnh truyền vào (Dùng cho Zero-shot)"""
    return {
        path: {
            "positive": f"Findings consistent with {path.lower()}",
            "negative": f"No findings consistent with {path.lower()}"
        } for path in pathologies
    }

@hydra.main(version_base=None, config_path="configs", config_name="experiment/eval_baseline")
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Bắt đầu đánh giá trên thiết bị: {device}")
    
    # =====================================================================
    # 1. KHỞI TẠO MODEL VÀ TẢI TRỌNG SỐ (LOAD CHECKPOINT)
    # =====================================================================
    print("Đang lắp ráp kiến trúc mô hình...")
    model = build_model(cfg.model).to(device)
    
    model.print_parameter_summary()

    # THÊM MỚI: Tải trọng số nếu có đường dẫn checkpoint trong config
    checkpoint_path = cfg.get("checkpoint_path", None)
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Đang tải trọng số từ: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Hỗ trợ cả 2 dạng: file .pth lưu nguyên state_dict hoặc file lưu dạng dict có key 'model_state_dict' (như ta đã cấu hình ở train.py)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict)
        print("-> Tải trọng số thành công!\n")
    else:
        print("-> [Lưu ý] Không có checkpoint_path. Chạy mô hình gốc (Zero-shot / Khởi tạo ngẫu nhiên).\n")

    model.eval()
    
    preprocess = model.backbone.preprocess
    tokenizer = model.backbone.tokenizer
    
    # =====================================================================
    # 2. KHỞI TẠO DATALOADER
    # =====================================================================
    print(f"Đang tải dataset: {cfg.dataset.name}...")
    test_loader = build_dataloader(
        cfg=cfg.dataset, 
        split="test", 
        transform=preprocess, 
        tokenizer=tokenizer
    )
    pathologies = cfg.dataset.params.pathologies
    
    # =====================================================================
    # 3. CHUẨN BỊ CHO ZERO-SHOT (Nếu mô hình đang chạy chế độ baseline)
    # =====================================================================
    text_features_dict = {}
    is_zero_shot = (cfg.get("phase") == "zero_shot_inference")
    
    if is_zero_shot:
        print("Chế độ Zero-shot: Đang tiền trích xuất đặc trưng Prompts...")
        prompt_dict = generate_medical_prompts(pathologies)
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
    # 4. CHẠY VÒNG LẶP TEST TRÊN ẢNH THẬT
    # =====================================================================
    all_probs = []
    all_ground_truths = []
    
    print("\nĐang quét qua tập Test...")
    with torch.no_grad():
        for images, input_ids, labels in tqdm(test_loader):
            images = images.to(device)
            # Có model cần input_ids, có model (zero-shot) thì không, nhưng cứ đẩy lên device cho an toàn
            if isinstance(input_ids, torch.Tensor):
                input_ids = input_ids.to(device)
            
            if is_zero_shot:
                # LOGIC CHUYÊN CHO ZERO-SHOT (Tính thủ công Cosine Sim)
                img_feat = model.backbone.model.encode_image(images)
                img_feat /= img_feat.norm(dim=-1, keepdim=True)
                
                batch_probs = []
                for path in pathologies:
                    text_weights = text_features_dict[path]
                    logits = img_feat @ text_weights.T
                    
                    # Lấy nhiệt độ từ config, mặc định là 0.07 nếu không truyền
                    temp = cfg.get("params", {}).get("temperature", 0.07)
                    probs = torch.softmax(logits / temp, dim=-1)[:, 0].unsqueeze(1)
                    batch_probs.append(probs)
                
                batch_probs = torch.cat(batch_probs, dim=1) # [Batch, 14]
            
            else:
                # LOGIC CHUYÊN CHO MÔ HÌNH ĐÃ TRAIN (FiLM, ProtoHead, v.v.)
                # Ở train.py chúng ta dùng BCEWithLogitsLoss, nên đầu ra của model là Logits chưa chuẩn hóa
                logits = model(images, input_ids) # [Batch, 14]
                
                # Áp dụng Sigmoid để đưa Logits về dải xác suất [0, 1] cho bài toán Multi-label
                batch_probs = torch.sigmoid(logits)

            all_probs.append(batch_probs.cpu())
            all_ground_truths.append(labels.cpu())

    # Gộp tensor của tất cả các batch lại
    all_probs = torch.cat(all_probs, dim=0).numpy()
    all_ground_truths = torch.cat(all_ground_truths, dim=0).numpy()

    # =====================================================================
    # 5. TÍNH TOÁN METRICS TỰ ĐỘNG
    # =====================================================================
    results = []
    print("\n=== KẾT QUẢ ĐÁNH GIÁ CHI TIẾT ===")
    for idx, path in enumerate(pathologies):
        gt_labels = all_ground_truths[:, idx]
        probs = all_probs[:, idx]
        
        if len(np.unique(gt_labels)) < 2:
            print(f"[Cảnh báo] Bệnh '{path}' bị bỏ qua do chỉ có 1 class trong tập Test.")
            continue
            
        auroc = roc_auc_score(gt_labels, probs)
        preds = (probs >= 0.5).astype(int)
        
        tn, fp, fn, tp = confusion_matrix(gt_labels, preds).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        accuracy = accuracy_score(gt_labels, preds)
        precision = precision_score(gt_labels, preds, zero_division=0)
        recall = recall_score(gt_labels, preds, zero_division=0) 
        f1 = f1_score(gt_labels, preds, zero_division=0)
        
        results.append({
            "Pathology": path, 
            "AUROC": auroc,
            "Accuracy": accuracy,
            "F1_Score": f1,
            "Precision": precision,
            "Recall_Sens": recall,
            "Specificity": specificity
        })
        
        print(f"{path:25s} | AUC: {auroc:.3f} | Acc: {accuracy:.3f} | F1: {f1:.3f} | Prec: {precision:.3f} | Rec: {recall:.3f} | Spec: {specificity:.3f}")

    # =====================================================================
    # 6. LƯU BÁO CÁO
    # =====================================================================
    df_res = pd.DataFrame(results).round(4)
    save_path = os.path.join("runs", cfg.experiment_name, f"metrics_{cfg.dataset.name}_test.csv")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df_res.to_csv(save_path, index=False)
    print(f"\n[Thành công] Đã lưu bảng kết quả chi tiết vào: {save_path}")

if __name__ == "__main__":
    main()