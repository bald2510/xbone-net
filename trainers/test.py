import os
import sys
import glob
from pathlib import Path
import pandas as pd
import torch
from PIL import Image
import open_clip  # Bắt buộc phải có thư viện này
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="timm")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # =========================================================================
    # 1. Đọc CSV và Khởi tạo Data thực tế
    # =========================================================================
    csv_path = r'C:\Users\lebat\Documents\Github\xbone-net\data\19052026\mimic-cxr-2.0.0-chexpert.csv' 
    image_dir = r'C:\Users\lebat\Documents\Github\xbone-net\data\19052026\images'
    prompt_dir = r'C:\Users\lebat\Documents\Github\xbone-net\data\19052026\reports_xray_prompts'
    
    print("Đang đọc dữ liệu từ file CSV...")
    df = pd.read_csv(csv_path)
    label_cols = [col for col in df.columns if col not in ['subject_id', 'study_id']]

    # =========================================================================
    # 2. Khởi tạo Mô hình GỐC (Base Model Baseline)
    # =========================================================================
    model_name = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
    print(f"\nĐang tải trực tiếp mô hình gốc từ HuggingFace: {model_name}")
    print("Lưu ý: Quá trình này sẽ không dùng bất kỳ trọng số LoRA nào của bạn...")
    
    # Load model gốc, hàm tiền xử lý và tokenizer
    model, _, preprocess = open_clip.create_model_and_transforms(model_name)
    tokenizer = open_clip.get_tokenizer(model_name)
    
    model.to(device)
    model.eval()
    context_length = 256

    # =========================================================================
    # 3. Chuẩn bị Batch Test (ĐỌC TEXT TỪ FILE)
    # =========================================================================
    test_df = df.head(5)

    images_list = []
    texts_list = [] 
    valid_test_data = [] 

    for idx, row in test_df.iterrows():
        subject_id = f"p{int(row['subject_id'])}"
        study_id = f"s{int(row['study_id'])}"
        
        study_dir = os.path.join(image_dir, subject_id, study_id)
        jpg_files = glob.glob(os.path.join(study_dir, "*.jpg"))
        
        txt_path = os.path.join(prompt_dir, subject_id, f"{study_id}.txt")
        report_text = ""
        
        if os.path.exists(txt_path):
            with open(txt_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content and "Không có prompt" not in content:
                    report_text = content
                    
        if not jpg_files or not report_text:
            continue
            
        img_path = jpg_files[0] 
        try:
            img_tensor = preprocess(Image.open(img_path).convert('RGB'))
            
            images_list.append(img_tensor)
            texts_list.append(report_text) 
            
            gt_diseases = [label_cols[i] for i, val in enumerate(row[label_cols]) if val > 0]
            if not gt_diseases: gt_diseases = ["No Finding"]
            
            valid_test_data.append({
                "image_name": os.path.basename(img_path),
                "study_id": study_id,
                "ground_truth": gt_diseases,
                "report": report_text 
            })
        except Exception as e:
            print(f"Lỗi đọc dữ liệu {study_id}: {e}")

    if not images_list:
        print("Không tìm thấy bộ dữ liệu (Ảnh + Text) hợp lệ nào để test!")
        return

    images = torch.stack(images_list).to(device)
    
    texts = tokenizer(texts_list, context_length=context_length).to(device)
    # Base open_clip tokenizer thường trả về Tensor trực tiếp
    if isinstance(texts, dict):
        texts = texts['input_ids']
    if len(texts.shape) > 2:
        texts = texts.squeeze()

    # =========================================================================
    # 4. Chạy suy luận (Zero-shot Inference bằng Model Gốc)
    # =========================================================================
    print(f"\nĐang đo độ tương đồng giữa X-quang và Báo cáo y khoa của {len(images_list)} ca...\n")
    with torch.no_grad():
        # KHÁC BIỆT CHÍNH Ở ĐÂY: Dùng model() thay vì model.backbone.model()
        image_features = model.encode_image(images)
        text_features = model.encode_text(texts)

        # Tính toán Cosine Similarity
        image_features_norm = torch.nn.functional.normalize(image_features, dim=-1)
        text_features_norm = torch.nn.functional.normalize(text_features, dim=-1)
        similarity_matrix = (image_features_norm @ text_features_norm.t()).cpu().numpy()

    # =========================================================================
    # 5. In Kết quả (Baseline)
    # =========================================================================
    print("=" * 80)
    print("KẾT QUẢ ĐÁNH GIÁ TRÊN MÔ HÌNH GỐC (BASELINE)")
    print("=" * 80)
    for i, data in enumerate(valid_test_data):
        matching_score = similarity_matrix[i][i] * 100 
        
        print(f"📸 --- ẢNH: {data['image_name']} (Lần chụp: {data['study_id']}) ---")
        print(f"✅ BỆNH THỰC TẾ (CSV): {', '.join(data['ground_truth'])}")
        print(f"📝 BÁO CÁO Y KHOA: {data['report']}")
        print(f"🎯 ĐỘ TƯƠNG ĐỒNG GỐC (Baseline): {matching_score:.2f}%")
        print("-" * 80)

if __name__ == "__main__":
    main()