import torch
import os
from PIL import Image
from src.models.build_model import MyMultiModalModel

def test_zero_shot_modular():
    # 1. Cấu hình (Giống hệt nãy)
    config = {
        'model': {
            'config_path': 'checkpoints/biomedclip/open_clip_config.json',
            'checkpoint_path': 'checkpoints/biomedclip/open_clip_pytorch_model.bin',
            'num_classes': 9, # Số lượng labels
            'hidden_dim': 512,
            'dropout': 0.3
        }
    }

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # 2. Khởi tạo Model Modular
    # Lưu ý: Ở đây ta chỉ dùng phần Backbone để lấy features giống HF
    model = MyMultiModalModel(config).to(device)
    model.eval()

    # 3. Chuẩn bị Labels & Images từ máy
    labels = [
        'adenocarcinoma histopathology', 'brain MRI', 'covid line chart',
        'squamous cell carcinoma histopathology', 'immunohistochemistry histopathology',
        'bone X-ray', 'chest X-ray', 'pie chart', 'hematoxylin and eosin histopathology'
    ]
    template = 'this is a photo of '
    
    image_folder = "data/raw/"
    test_imgs = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not test_imgs:
        print("Folder data/raw trống")
        return

    # 4. Tokenize Text & Load Images
    texts = model.backbone.tokenizer([template + l for l in labels], context_length=256).to(device)
    
    input_images = []
    for img_name in test_imgs:
        img_path = os.path.join(image_folder, img_name)
        img = Image.open(img_path).convert('RGB')
        input_images.append(model.backbone.preprocess(img))
    
    images_tensor = torch.stack(input_images).to(device)

    # 5. Logic Zero-shot (Giống code HF nhưng dùng model của ông)
    print(f"Đang chạy test Zero-shot với model...")
    with torch.no_grad():
        # Lấy feature từ backbone (phần lõi của model mới)
        image_features, text_features = model.backbone(images_tensor, texts)
        
        # Lấy logit_scale từ model gốc (open_clip lưu trong model.model)
        logit_scale = model.backbone.model.logit_scale.exp()

        # Tính toán xác suất (Dot product + Softmax)
        logits = (logit_scale * image_features @ text_features.t()).detach().softmax(dim=-1)
        sorted_indices = torch.argsort(logits, dim=-1, descending=True)

    # 6. In kết quả
    logits = logits.cpu().numpy()
    sorted_indices = sorted_indices.cpu().numpy()

    for i, img_name in enumerate(test_imgs):
        print(f"👉 File: {img_name}")
        for j in range(3): # Xem Top 3
            idx = sorted_indices[i][j]
            print(f"   [{labels[idx]}]: {logits[i][idx]:.4f}")
        print("-" * 20)

if __name__ == "__main__":
    # Đảm bảo PYTHONPATH đã được set để nhận diện thư mục 'src'
    test_zero_shot_modular()