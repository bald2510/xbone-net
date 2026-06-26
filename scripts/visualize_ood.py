import sys
import os
sys.path.append(os.path.abspath('.'))

import numpy as np
import matplotlib.pyplot as plt
from utils.ood import OODDetector

def main():
    print("Loading embeddings...")
    # Load ID embeddings
    id_data = np.load("results/1_baseline/ours_xbone_net/seed_42/embeddings.npz", allow_pickle=True)
    id_embeds = id_data["image_embeddings"]
    id_labels = id_data["labels"]

    # Load OOD embeddings
    ood_data = np.load("results/ood/btxrd_vs_fracatlas/fracatlas/embeddings.npz", allow_pickle=True)
    ood_embeds = ood_data["image_embeddings"]

    # L2-normalize image embeddings (matching evaluate_ood.py)
    print("L2-normalizing embeddings...")
    id_embeds_norm = id_embeds / (np.linalg.norm(id_embeds, axis=1, keepdims=True) + 1e-8)
    ood_embeds_norm = ood_embeds / (np.linalg.norm(ood_embeds, axis=1, keepdims=True) + 1e-8)

    print("Fitting OOD detector on ID set...")
    detector = OODDetector()
    detector.fit(id_embeds_norm, id_labels)

    print("Computing OOD scores...")
    # Mahalanobis
    id_maha = detector.score_mahalanobis(id_embeds_norm)
    ood_maha = detector.score_mahalanobis(ood_embeds_norm)

    # k-NN
    id_knn = detector.score_knn(id_embeds_norm, k=5)
    ood_knn = detector.score_knn(ood_embeds_norm, k=5)

    print("Plotting OOD score distributions...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Mahalanobis Distance (using log10 due to massive OOD separation)
    axes[0].hist(np.log10(id_maha), bins=30, alpha=0.6, color='#1f77b4', label='ID (BTXRD)', density=True, edgecolor='black')
    axes[0].hist(np.log10(ood_maha), bins=30, alpha=0.6, color='#d62728', label='OOD (FracAtlas)', density=True, edgecolor='black')
    axes[0].set_xlabel(r'$\log_{10}$ Khoảng cách Mahalanobis', fontsize=11)
    axes[0].set_ylabel('Mật độ phân bổ (Density)', fontsize=11)
    axes[0].set_title('Phân bổ Khoảng cách Mahalanobis (Log-scale)', fontsize=12, fontweight='bold')
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].legend(fontsize=10)

    # Plot 2: k-NN Cosine Distance
    axes[1].hist(id_knn, bins=30, alpha=0.6, color='#1f77b4', label='ID (BTXRD)', density=True, edgecolor='black')
    axes[1].hist(ood_knn, bins=30, alpha=0.6, color='#d62728', label='OOD (FracAtlas)', density=True, edgecolor='black')
    axes[1].set_xlabel('Khoảng cách Cosine k-NN (k=5)', fontsize=11)
    axes[1].set_ylabel('Mật độ phân bổ (Density)', fontsize=11)
    axes[1].set_title('Phân bổ Khoảng cách Cosine k-NN', fontsize=12, fontweight='bold')
    axes[1].grid(True, linestyle='--', alpha=0.5)
    axes[1].legend(fontsize=10)

    plt.tight_layout()
    output_path = 'results/visualization/ood_distribution.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ OOD distribution plot successfully saved to: {output_path}")

if __name__ == "__main__":
    main()
