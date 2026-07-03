# XBone-Net: Multimodal Bone Tumor Classification and Out-of-Distribution Detection

XBone-Net is a state-of-the-art multimodal vision-language framework designed for bone tumor classification and Out-of-Distribution (OOD) anomaly detection on musculoskeletal radiographs. It addresses the clinical challenge of label leakage by decoupling xray reports at training from historical clinical data at inference, and deploys a learnable Prototypical network to construct robust, clinically explainable decision boundaries.

---

## 1. Key Features

- **Asymmetric Two-Stage Training:**
  - **Phase 1 (Contrastive Alignment):** Align image embeddings with text embeddings using PEFT (QLoRA) and Semantic Matching Contrastive Loss.
  - **Phase 2 (Multimodal Classification):** Fuse image and text (clinical history) features using a Cross-Attention Transformer and classify via a learnable Prototypical Head.
- **Label Leakage Prevention:** Compels the model to use **only clinical history text** at inference, preventing cheating from pre-existing X-ray reports.
- **Robust OOD Detection:** Supports 4 OOD scoring algorithms (Mahalanobis Distance, Cosine-KNN, Energy Score, and Text-Anchor Distance) to identify anomalous radiographs.
- **Explainability:** Built-in cross-attention map extraction to visualize which regions of the X-ray image align with specific clinical keywords.

---

## 2. Directory Structure

```text
xbone-net/
├── configs/                # Hydra configuration files
│   ├── dataset/            # Dataset-specific configs (btxrd, ctch)
│   ├── model/              # Model, PEFT, fusion, and classifier configs
│   └── experiment/         # Grouped experiment configs (Baselines & Ours)
├── src/                    # Consolidated Python source code package
│   ├── datasets/           # PyTorch Dataset loaders for BTXRD and CTCH
│   ├── models/             # PyTorch Modules (Backbones, Fusion, Prototypical Head)
│   └── utils/              # Helper utilities (losses, metrics, prompts, etc.)
├── tools/                  # Script utilities, training execution runners, and plotting
├── train.py                # Main two-stage training script
├── evaluate.py             # Main classification evaluation script
├── evaluate_ood.py         # Main OOD detection evaluation script
└── README.md               # This documentation file
```

---

## 3. Installation

We recommend using Anaconda or Miniconda to manage environments.

```bash
# 1. Create and activate the conda environment
conda create -n xbone python=3.10 -y
conda activate xbone

# 2. Install PyTorch with CUDA support (adjust CUDA version if necessary)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 3. Install other required packages
pip install -r requirements.txt
```

---

## 4. Dataset Setup

Place your dataset images and split CSV files in a `data/` folder in the project directory (or adjust path settings in `configs/dataset/btxrd.yaml` or `configs/dataset/ctch.yaml`):

```text
data/
├── btxrd/
│   ├── images/
│   │   ├── patient_001.png
│   │   └── ...
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
└── ctch/
    ├── images/
    │   ├── patient_001.png
    │   └── ...
    ├── train.csv
    ├── val.csv
    └── test.csv
```

---

## 5. Usage

### 5.1. Training a Model
To train XBone-Net or a baseline model on a specific experiment config (for example, the proposed `ours_xbone_net` on the BTXRD dataset):

```bash
python train.py +experiment=btxrd/proposed/ours_xbone_net
```

For the CTCH dataset baseline:
```bash
python train.py +experiment=ctch/proposed/ours_xbone_net
```

### 5.2. Running Evaluation
To evaluate a trained model checkpoint and calculate metrics (Accuracy, F1-macro, Specificity, AUROC, Confusion Matrix) along with **95% Bootstrap Confidence Intervals**:

```bash
python evaluate.py +experiment=btxrd/proposed/ours_xbone_net --bootstrap --save-embeddings
```
*(Using `--save-embeddings` saves the test embeddings and probabilities to an `.npz` file, which is required for running OOD detection).*

### 5.3. Out-of-Distribution (OOD) Detection
Once you have saved the ID embeddings using `evaluate.py`, run OOD detection against an OOD dataset (e.g., using `fracatlas` as OOD):

```bash
python evaluate_ood.py \
    --id-path checkpoints/ours_xbone_net/seed_42/test_embeddings.npz \
    --ood-path checkpoints/fracatlas/seed_42/test_embeddings.npz \
    --method mahalanobis
```

### 5.4. Running the Entire Experiment Suite
To run all organized experiment groups (Zero-shot, Fine-tuned Baselines, Proposed Model):

```bash
python tools/run_all.py --group zero_shot_baselines
python tools/run_all.py --group finetuned_baselines
python tools/run_all.py --group proposed
```
