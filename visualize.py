import os
import torch
import hydra
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from omegaconf import DictConfig
from sklearn.manifold import TSNE

from models.builder import build_model
from datasets.builder import build_dataloader

@hydra.main(version_base=None, config_path="configs", config_name="experiment/exp_baseline_flim_prototypical")
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Bắt đầu Visualization trên thiết bị: {device}")
    
    # 1. TẢI MÔ HÌNH VÀ TRỌNG SỐ
    model = build_model(cfg.model).to(device)
    checkpoint_path = cfg.get("checkpoint_path", "./checkpoints/best_model.pth")
    
    if os.path.exists(checkpoint_path):
        print(f"Đang tải trọng số từ: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
    else:
        raise FileNotFoundError(f"Không tìm thấy file trọng số tại {checkpoint_path}!")
        
    model.eval()
    
    # 2. TẢI DỮ LIỆU
    test_loader = build_dataloader(
        cfg=cfg.dataset, split="test", 
        transform=model.backbone.preprocess, 
        tokenizer=model.backbone.tokenizer
    )
    pathologies = cfg.dataset.params.pathologies
    
    # Để đồ thị bớt rối rắm (do multi-label), ta sẽ chỉ chọn 5 bệnh lý nổi bật nhất để tô màu
    TARGET_CLASSES = ['Cardiomegaly', 'Edema', 'Consolidation', 'Pneumothorax', 'Pleural Effusion']
    target_indices = [pathologies.index(cls) for cls in TARGET_CLASSES]

    # 3. TRÍCH XUẤT ĐẶC TRƯNG VÀ PROTOTYPES
    all_fused_features = []
    all_labels = []
    
    print("Đang trích xuất đặc trưng từ tập Test (giới hạn 1000 mẫu để vẽ t-SNE)...")
    max_samples = 1000
    sample_count = 0
    
    with torch.no_grad():
        for images, input_ids, labels in tqdm(test_loader):
            images, input_ids = images.to(device), input_ids.to(device)
            
            # Trích xuất thủ công để lấy feature trung gian (thay vì lấy logits)
            img_feats = model.backbone.model.encode_image(images)
            txt_feats = model.backbone.model.encode_text(input_ids)
            
            # Đưa qua FiLM Fusion
            fused_feats = model.fusion(img_feats, txt_feats)
            
            # QUAN TRỌNG: L2 Normalization (Vì ProtoHead dùng Cosine Similarity)
            fused_feats = fused_feats / fused_feats.norm(dim=-1, keepdim=True)
            
            all_fused_features.append(fused_feats.cpu())
            all_labels.append(labels.cpu())
            
            sample_count += images.size(0)
            if sample_count >= max_samples:
                break

    all_fused_features = torch.cat(all_fused_features, dim=0)[:max_samples]
    all_labels = torch.cat(all_labels, dim=0)[:max_samples].numpy()
    
    # Lấy và Normalize ma trận Prototypes từ mô hình
    prototypes = model.head.prototypes.detach()
    prototypes = prototypes / prototypes.norm(dim=-1, keepdim=True)
    prototypes = prototypes.cpu()

    # 4. CHẠY T-SNE CHO TOÀN BỘ KHÔNG GIAN
    print("Đang chạy t-SNE để giảm chiều từ 512D -> 2D (Sẽ mất khoảng 10-30 giây)...")
    # Gộp features (1000, 512) và prototypes (14, 512) thành một cục (1014, 512)
    combined_features = torch.cat([all_fused_features, prototypes], dim=0).numpy()
    
    tsne = TSNE(n_components=2, perplexity=20, random_state=42, init='pca', learning_rate='auto', metric="cosine")
    combined_2d = tsne.fit_transform(combined_features)
    
    # Tách ngược lại thành features_2d và prototypes_2d
    features_2d = combined_2d[:max_samples]
    prototypes_2d = combined_2d[max_samples:]

    # 5. LỌC DỮ LIỆU ĐỂ VẼ (Xử lý bài toán Multi-label)
    # Vì một ảnh có thể có nhiều bệnh, ta chọn ảnh nào có duy nhất 1 bệnh trong TARGET_CLASSES để tô màu cho rõ
    plot_data = []
    for i in range(max_samples):
        row_labels = all_labels[i]
        active_targets = [cls for idx, cls in zip(target_indices, TARGET_CLASSES) if row_labels[idx] == 1]
        
        # Chỉ vẽ các điểm có đúng 1 nhãn dương tính trong tập mục tiêu (để phân cụm rõ ràng)
        if len(active_targets) == 1:
            plot_data.append({
                'x': features_2d[i, 0],
                'y': features_2d[i, 1],
                'Label': active_targets[0]
            })
            
    df_plot = pd.DataFrame(plot_data)

    # 6. VẼ ĐỒ THỊ BẰNG SEABORN & MATPLOTLIB
    print("Đang kết xuất biểu đồ...")
    plt.figure(figsize=(12, 10))
    sns.set_style("whitegrid")
    
    # Chọn bảng màu rực rỡ
    palette = sns.color_palette("bright", len(TARGET_CLASSES))
    
    # Vẽ các điểm ảnh (Image Features)
    sns.scatterplot(
        data=df_plot, x='x', y='y', hue='Label', 
        palette=palette, alpha=0.6, s=50, edgecolor=None
    )
    
    # Vẽ các Prototypes (Neo tâm cụm)
    for idx, cls in zip(target_indices, TARGET_CLASSES):
        color = palette[TARGET_CLASSES.index(cls)]
        plt.scatter(
            prototypes_2d[idx, 0], prototypes_2d[idx, 1], 
            color=color, marker='*', s=800, edgecolor='black', linewidth=1.5,
            label=f"{cls} (Prototype)"
        )

    plt.title("Không gian t-SNE: Image Features và Prototypes (Baseline + FiLM)", fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("t-SNE Dimension 1", fontsize=12)
    plt.ylabel("t-SNE Dimension 2", fontsize=12)
    
    # Sắp xếp lại chú thích (Legend) ở ngoài biểu đồ
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    
    save_path = f"runs/{cfg.experiment_name}/tsne_clusters.png" 
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n[Thành công] Đã lưu hình ảnh trực quan hóa độ phân giải cao tại: {save_path}")

if __name__ == "__main__":
    main()