# XBone-Net: Multimodal Bone Tumor Classification and Out-of-Distribution Detection

XBone-Net is a multimodal vision-language framework for bone-tumor classification and out-of-distribution (OOD) detection on musculoskeletal radiographs. The canonical pipeline uses radiographs and pre-imaging clinical history in both training phases; radiology findings are excluded because they can contain the target diagnosis and cause label leakage.

---

## 1. Key Features

- **High-resolution image path:** Cover the full radiograph with a deterministic 224-pixel grid, pool local patch tokens, and compress them with a coordinate-aware learned resampler. Local tiles propagate gradients to visual LoRA adapters during training, with chunk-level activation checkpointing to control memory.
- **Two-stage training:**
  - **Phase 1 (semantic alignment):** Align image and clinical-history embeddings with LoRA and soft-target Semantic Matching Loss.
  - **Phase 2 (multimodal classification):** Continue optimizing the LoRA adapters together with bidirectional cross-attention. Classification uses cosine similarity to empirical class centroids recomputed from the training embeddings and effective-number weighted cross-entropy.
- **Label-leakage prevention:** Use **clinical history only** as the text modality in both phases and at inference; X-ray reports are reserved for ablation analysis.
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
│   ├── models/             # PyTorch modules (backbones, fusion, centroid/ablation heads)
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

# Required only for experiments using the MedCLIP backbone. Its upstream
# metadata pins an obsolete Transformers version; XBone-Net provides the
# compatibility layer, so install the package without its dependency pins.
python -m pip install --no-deps MedCLIP==0.0.3
```

`requirements.txt` is generated from direct imports in the repository. After
adding or removing a dependency, regenerate it from the active environment:

```bash
python tools/generate_requirements.py

# CI/read-only validation: exits with code 1 when the file is stale
python tools/generate_requirements.py --check
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
    --id-path checkpoints/btxrd/proposed/ours_xbone_net/seed_42/test_embeddings.npz \
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

### 5.5. Canonical Pipeline and Ablations

BTXRD and CTCH both provide baseline and proposed-model evaluations. Component ablations are performed on the real-world CTCH dataset under `configs/experiment/ctch/ablation_study/` and are organized into exactly four groups: `input` (resolution and modality), `backbone` (PEFT and backbone adaptation), `fusion`, and `classifier`. Every ablation inherits from `configs/experiment/ctch/proposed/ours_xbone_net.yaml` and overrides only the component being tested. The LoRA adapters are intentionally not merged after Phase 1 so that they remain trainable during Phase 2; adapter merging/freezing is treated as a backbone/PEFT ablation.

The proposed pipeline is fixed before interpreting ablations. If an ablation performs better on a metric, report the result directly as a limitation or trade-off of the proposed component rather than relabeling that ablation as the proposed model after seeing test results.

### 5.6. CTCH Few-Shot Experiments

CTCH follows the same per-class sampling rule as BTXRD: retain at most `k_shot` training samples from each available class with deterministic sampling by `seed`; validation and test remain unchanged. Configurations are provided for 1, 10, and 20 shots with LoRA-BiomedCLIP, LoRA-PubMedCLIP, and XBone-Net. For example:

```bash
python train.py +experiment=ctch/few_shot/1_shot/ours_xbone_net
python train.py +experiment=ctch/few_shot/10_shot/lora_biomedclip
python train.py +experiment=ctch/few_shot/20_shot/lora_pubmedclip
```
