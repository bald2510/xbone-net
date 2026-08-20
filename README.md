# XBone-Net: Multimodal Bone Tumor Classification and Out-of-Distribution Detection

XBone-Net is a multimodal vision-language framework for bone-tumor classification and out-of-distribution (OOD) detection on musculoskeletal radiographs. The canonical pipeline uses radiographs and pre-imaging clinical history in both training phases; radiology findings are excluded because they can contain the target diagnosis and cause label leakage.

---

## 1. Key Features

- **Canonical image path:** Pass each radiograph directly to the original BiomedCLIP resize/crop/normalization transform (`direct_resize` in the locked experiment config); no high-resolution tile branch or visual resampler is used.
- **Two-stage training:**
  - **Phase 1 (semantic alignment):** Align image and clinical-history embeddings with LoRA and soft-target Semantic Matching Loss.
  - **Phase 2 (multimodal classification):** Continue optimizing the LoRA adapters together with bidirectional cross-attention and a 512-to-class linear head using class-weighted cross-entropy.
- **Label-leakage prevention:** Use **clinical history only** as the text modality in both phases and at inference; X-ray reports are reserved for ablation analysis.
- **Post-hoc OOD detection:** Uses only `MultimodalEnsembleOODDetector`, which combines validation-standardized image and clinical-text kNN distances with max fusion. The detector is fitted on CTCH train features and its operating threshold is calibrated on CTCH validation-ID data.
- **Explainability:** Provides end-to-end Integrated Gradients for the image tensor produced by the locked preprocessing config and for clinical-text embeddings, together with perturbation-based faithfulness analysis.

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
orchestrator exports the required feature archives, fits the ensemble detector on CTCH
train/validation data, and evaluates all configured scenarios for three seeds:

```bash
python benchmark/ood_analysis.py
```

See Section 9 of [`HuongDanSuDung.txt`](HuongDanSuDung.txt) for the OOD and
explainability commands. Feature extraction, calibration and scoring are
coordinated by `benchmark/ood_analysis.py`.

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

BTXRD and CTCH both provide baseline and proposed-model evaluations. Component ablations are performed on CTCH under `configs/experiment/ctch/ablation_study/`. Every current-architecture ablation inherits from `configs/experiment/ctch/proposed/ours_xbone_net.yaml` and overrides one decision, such as phase-2-only training, concatenation or one-way attention instead of bidirectional cross-attention, or a classifier/loss option. The `shuffled_report` experiment trains normally and applies a one-to-one cross-class report derangement only on the test split.

The proposed pipeline is fixed before interpreting ablations. If an ablation performs better on a metric, report the result directly as a limitation or trade-off of the proposed component rather than relabeling that ablation as the proposed model after seeing test results.

### 5.6. Streamlit Research Demo

The release demo automatically loads the locked seed-42 package under
`demo/model/`. Verify the checkpoint, resolved config, and OOD feature archives,
then launch the local interface:

```bash
python tools/verify_demo_artifacts.py
streamlit run demo/streamlit_app.py
```

The app displays preprocessing, class probabilities, maximum-softmax
confidence, the OOD score and threshold, nearest CTCH reference images, and
image/text Integrated Gradients. It is a research demonstration and not
a medical device.

See [`HuongDanSuDungDemo.txt`](HuongDanSuDungDemo.txt) for the required artifact
layout, checksum verification, optional `XBONE_DEMO_ARTIFACT_ROOT` override, and
the command used to repackage a newly trained canonical checkpoint.
