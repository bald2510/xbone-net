# XBone-Net: Multimodal Bone Tumor Classification and Out-of-Distribution Detection

XBone-Net is a multimodal vision-language framework for bone-tumor classification and out-of-distribution (OOD) detection on musculoskeletal radiographs. The canonical pipeline uses radiographs and pre-imaging clinical history in both training phases; radiology findings are excluded because they can contain the target diagnosis and cause label leakage.

---

## 1. Key Features

- **High-resolution image path:** Conservatively crop scanner padding, retain an aspect-preserving 224-pixel global view, and select exactly four 224-pixel local views using image-only coverage and focal texture scores. Each local view contributes one pretrained CLS token and a pooled `2×2` patch grid, producing 20 local visual tokens without changing the BiomedCLIP input resolution.
- **Two-stage training:**
  - **Phase 1 (semantic alignment):** Align image and clinical-history embeddings with LoRA and soft-target Semantic Matching Loss.
  - **Phase 2 (multimodal classification):** Continue optimizing the LoRA adapters together with bidirectional cross-attention. Classification uses cosine similarity to empirical class centroids recomputed from the training embeddings and effective-number weighted cross-entropy.
- **Label-leakage prevention:** Use **clinical history only** as the text modality in both phases and at inference; X-ray reports are reserved for ablation analysis.
- **Post-hoc OOD detection:** Supports cosine distance to empirical class centroids, class-centroid Mahalanobis distance, cosine-KNN, and predictive entropy. Detectors are fitted on CTCH train features and thresholds are calibrated on CTCH validation-ID data.
- **Explainability:** Provides Integrated Gradients for the global image, sparse-focal local tokens and clinical-text tokens, together with perturbation-based faithfulness analysis.

---

## 2. Directory Structure

```text
xbone-net/
├── benchmark/              # Efficiency, OOD and explainability benchmarks
├── configs/                # Hydra dataset, model and experiment configurations
├── data/                   # Local datasets and dataset-specific preprocessing
├── demo/                   # Streamlit research demo and its dependencies
├── docs/
│   ├── assets/             # Documentation assets
│   ├── paper/              # Conference-paper sources
│   └── report/             # Thesis sources and generated report artifacts
├── scripts/                # Windows launchers for training, evaluation and reports
├── src/                    # Reusable datasets, models and utility modules
├── tools/                  # Experiment orchestration, exports and visualizations
├── train.py                # Two-stage training entry point
├── evaluate.py             # Classification evaluation entry point
├── evaluate_ood.py         # OOD evaluation stage
├── inference.py            # Single-sample inference entry point
├── HuongDanCaiDat.txt      # Vietnamese installation guide
└── HuongDanSuDung.txt      # Vietnamese execution guide
```

Vietnamese setup and execution instructions are available in
[`HuongDanCaiDat.txt`](HuongDanCaiDat.txt) and
[`HuongDanSuDung.txt`](HuongDanSuDung.txt).

---

## 3. Installation

We recommend using Anaconda or Miniconda to manage environments. The versions
below match the validated project environment; use the CPU PyTorch index when
CUDA is unavailable.

```bash
# 1. Create and activate the conda environment
conda create -n Thesis python=3.11.15 -y
conda activate Thesis

# 2. Install the validated CUDA build of PyTorch
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128

# 3. Install other required packages
python -m pip install -r requirements.txt

# Required only for experiments using the MedCLIP backbone. Its upstream
# metadata pins an obsolete Transformers version; XBone-Net provides the
# compatibility layer, so install the package without its dependency pins.
python -m pip install --no-deps MedCLIP==0.0.3
```

`requirements.txt` is generated from direct imports in the repository. After
adding or removing a dependency, regenerate it from the active environment:

```bash
python tools/generate_requirement.py

# CI/read-only validation: exits with code 1 when the file is stale
python tools/generate_requirement.py --check
```

---

## 4. Dataset Setup

Place dataset payloads under `data/` or change the paths in
`configs/dataset/btxrd.yaml` and `configs/dataset/ctch.yaml`. Dataset images,
reports and patient data are intentionally excluded from Git.

```text
data/
├── BTXRD/
│   ├── images/
│   ├── reports/
│   ├── btxrd-split.csv
│   └── btxrd-labels.csv
└── CTCH/
    ├── images/
    ├── reports/clinical/
    ├── reports/clinical_vi/
    ├── ctch-split.csv
    ├── ctch-labels.csv
    ├── ctch-ood.csv
    └── labels.txt

```

---

## 5. Usage

### 5.1. Training a Model
To train XBone-Net or a baseline model on a specific experiment config (for example, the proposed `ours_xbone_net` on the BTXRD dataset):

```bash
python train.py +experiment=btxrd/proposed/ours_xbone_net
```

For the proposed model on CTCH:
```bash
python train.py +experiment=ctch/proposed/ours_xbone_net
```

### 5.2. Running Evaluation
To evaluate a trained model checkpoint and calculate metrics (Accuracy, F1-macro, Specificity, AUROC, Confusion Matrix) along with **95% Bootstrap Confidence Intervals**:

```bash
python evaluate.py +experiment=btxrd/proposed/ours_xbone_net --bootstrap --save-embeddings
```
`--save-embeddings` is optional for ordinary classification evaluation.

### 5.3. Out-of-Distribution (OOD) Detection
OOD and explainability are locked to the proposed model trained on CTCH. The
orchestrator exports the required feature archives, fits detectors only on CTCH
train/validation data, and evaluates all configured scenarios for three seeds:

```bash
python benchmark/ood_analysis.py
```

See Section 9 of [`HuongDanSuDung.txt`](HuongDanSuDung.txt) for feature-only,
OOD-only, explainability-only and table-generation commands. `evaluate_ood.py`
is an internal stage of this locked workflow and should not be invoked with
unrelated BTXRD checkpoints.

### 5.4. Running the Entire Experiment Suite
`tools/training.py` discovers every YAML under `configs/experiment/`; the Python
registry no longer needs to be edited. Enable a config/group with `+` and
disable it with `-` in `tools/experiments.txt` (a disabled selector always
wins). Inspect the available selectors and preview the final queue before
launching training:

```bash
python tools/training.py --list-configs
python tools/training.py --dry-run
python tools/training.py --seeds 42 123 456 --bootstrap
python tools/training.py --group zero_shot_baselines
python tools/training.py --group finetuned_baselines
python tools/training.py --group proposed
python tools/training.py --group ablation
```

`--group` explicitly selects a subset while retaining the manifest's disabled
list. `--ignore-experiment-file` performs direct CLI-only selection.

Trainable experiments use every requested seed. Deterministic zero-shot configs
are evaluated once because repeating them under different seed labels would be
pseudo-replication; their uncertainty is estimated by test-sample bootstrap.

Review mode is independent from the training switches. Without `--group`, it
scans every existing result and prints separate BTXRD/CTCH sections grouped by
configuration family. It reports F1, balanced accuracy, accuracy, sensitivity,
specificity, precision, AUROC, AUPRC, ECE, adaptive ECE, NLL, Brier score, and
parameter counts; the same data are exported to
`results/summary/run_all_table.csv`.

```bash
python tools/training.py --table
python tools/training.py --table --group ctch
python tools/training.py --table --group ctch_proposed
python tools/training.py --table --table-selected-only
```

### 5.5. Canonical Pipeline and Ablations

BTXRD and CTCH both provide baseline and proposed-model evaluations. Component ablations are performed on the real-world CTCH dataset under `configs/experiment/ctch/ablation_study/` and are organized into `modality`, `finetune`, and `architecture` (`preprocess`, `phase`, `fusion`, and `classifier`). Every ablation inherits from `configs/experiment/ctch/proposed/ours_xbone_net.yaml` and overrides only the component being tested. The `shuffled_report` experiment trains normally and applies a one-to-one cross-class report derangement only on the test split. The `xbone_nohighres` and `xbone_letterbox` controls use one encoder view while preserving the proposed fusion interface of one global token plus 20 pooled visual tokens.

The proposed pipeline is fixed before interpreting ablations. If an ablation performs better on a metric, report the result directly as a limitation or trade-off of the proposed component rather than relabeling that ablation as the proposed model after seeing test results.

### 5.6. CTCH Few-Shot Experiments

CTCH follows the same per-class sampling rule as BTXRD: retain at most `k_shot` training samples from each available class with deterministic sampling by `seed`; validation and test remain unchanged. Configurations are provided for 1, 10, and 20 shots with LoRA-BiomedCLIP, LoRA-PubMedCLIP, and XBone-Net. For example:

```bash
python train.py +experiment=ctch/few_shot/1_shot/ours_xbone_net
python train.py +experiment=ctch/few_shot/10_shot/lora_biomedclip
python train.py +experiment=ctch/few_shot/20_shot/lora_pubmedclip
```

### 5.7. Streamlit Research Demo

After preparing the canonical CTCH proposed checkpoint and OOD reference
features, launch the local interface with:

```bash
streamlit run demo/streamlit_app.py
```

The app displays preprocessing, class probabilities, maximum-softmax
confidence, the OOD score and threshold, nearest CTCH reference images, and
global/local/text Integrated Gradients. It is a research demonstration and not
a medical device.
