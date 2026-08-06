import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path("c:/Users/lebat/Documents/Github/xbone-net")
sys.path.insert(0, str(ROOT))

from src.utils.ood import OODDetector
from src.utils.centroid_visualization import l2_normalize

def get_image_path(image_id, dataset="ctch"):
    if dataset == "ctch":
        img_dir = ROOT / "data" / "CTCH" / "images"
    else:
        img_dir = ROOT / "data" / "BTXRD" / "images"
        
    p = img_dir / image_id
    if p.exists():
        return p
    for ext in [".png", ".jpg", ".jpeg"]:
        p = img_dir / f"{image_id}{ext}"
        if p.exists():
            return p
    return None

def main():
    seed = 42
    ood_split = "ctch_ood"
    ood_index = 0

    features_dir = ROOT / f"results/ctch/proposed/ours_xbone_net/seed_{seed}/analysis/features"
    ood_file = features_dir / f"{ood_split}.npz"
    train_file = features_dir / "ctch_train.npz"
    val_file = features_dir / "ctch_val.npz"

    ood_data = np.load(ood_file, allow_pickle=True)
    train_data = np.load(train_file, allow_pickle=True)
    val_data = np.load(val_file, allow_pickle=True)

    ood_fused = ood_data["fused_embeddings"]
    ood_raw = ood_data["fused_embeddings_raw"]
    ood_ids = ood_data["image_id"]

    train_fused = train_data["fused_embeddings"]
    train_raw = train_data["fused_embeddings_raw"]
    train_ids = train_data["image_id"]
    train_labels = train_data["labels"]
    
    val_raw = val_data["fused_embeddings_raw"]
    if train_labels.ndim == 2:
        train_labels = train_labels.argmax(axis=1)

    # 1. Get OOD sample
    ood_sample_fused = ood_fused[ood_index:ood_index+1]
    ood_sample_raw = ood_raw[ood_index:ood_index+1]
    ood_id = str(ood_ids[ood_index])

    # 2. Nearest ID sample (Cosine similarity on fused_embeddings)
    norm_ood = l2_normalize(ood_sample_fused)
    norm_train = l2_normalize(train_fused)
    sim = norm_ood @ norm_train.T
    nearest_idx = np.argmax(sim[0])
    nearest_id = str(train_ids[nearest_idx])
    nearest_sim = sim[0, nearest_idx]
    nearest_class = train_labels[nearest_idx]

    # 3. Mahalanobis distance
    detector = OODDetector().fit(train_raw, train_labels)
    centroids = np.array([detector.class_means[i] for i in range(len(detector.class_means))])
    prec = detector.shared_cov_inv
    
    diff = ood_sample_raw - centroids
    m_dists = np.sum((diff @ prec) * diff, axis=1)
    m_dists = np.sqrt(np.maximum(m_dists, 0))
    confused_class = np.argmin(m_dists)
    min_m_dist = m_dists[confused_class]

    print(f"OOD Image ID: {ood_id}")
    print(f"Nearest ID Image ID: {nearest_id} (Class {nearest_class}) - Cosine Sim: {nearest_sim:.4f}")
    print(f"Confused Class: {confused_class} - Mahalanobis Dist: {min_m_dist:.4f}")

    # Calculate ID threshold (95% TPR on validation set)
    val_scores = detector.score_mahalanobis_centroid(val_raw)
    threshold = np.percentile(val_scores, 95)
    print(f"ID Threshold (95% TPR): {threshold:.4f}")
    
    is_ood = min_m_dist > threshold
    ood_status = "OOD" if is_ood else "ID (Bỏ sót)"

    # Read class names
    try:
        import yaml
        with open(ROOT / "configs/dataset/ctch.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            classes = cfg["params"]["classes"]
    except Exception as e:
        print(e)
        classes = [f"Class {i}" for i in range(22)]

    # Plot
    img_ood_path = get_image_path(ood_id, "ctch")
    img_id_path = get_image_path(nearest_id, "ctch")

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    if img_ood_path:
        img_ood = plt.imread(img_ood_path)
        axes[0].imshow(img_ood, cmap="gray")
    
    title_ood = (
        f"Mẫu OOD\n"
        f"{ood_id}\n"
        f"Cụm nhầm lẫn: {classes[confused_class]}\n"
        f"Mahalanobis Dist: {min_m_dist:.2f} < {threshold:.2f}\n"
    )
    axes[0].set_title(title_ood)
    axes[0].axis("off")

    if img_id_path:
        img_id = plt.imread(img_id_path)
        axes[1].imshow(img_id, cmap="gray")
        
    title_id = (
        f"Mẫu ID gần nhất\n"
        f"ID: {nearest_id}\n"
        f"Lớp: {classes[nearest_class]}\n"
        f"Cosine Sim: {nearest_sim:.3f}\n"
    )
    axes[1].set_title(title_id)
    axes[1].axis("off")

    output_path = ROOT / "results/ctch/proposed/ours_xbone_net/seed_42/centroid_visualization/ood_sample_comparison.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved plot to {output_path}")

if __name__ == "__main__":
    main()
