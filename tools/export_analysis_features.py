"""Công cụ trích xuất và lưu trữ đặc trưng (Features Export) cho XBone-Net và các mô hình cơ sở.

Hợp nhất toàn bộ quy trình trích xuất đặc trưng cho mọi mô hình và mọi kịch bản:
- Kịch bản ID: ctch_train, ctch_val, ctch_test
- Kịch bản OOD: ctch_ood, btxrd_test, report_mismatch_cross_class, report_mismatch_same_class
- Hỗ trợ: XBone-Net letterbox, các ablation, full fine-tuning, PEFT LoRA và zero-shot CLIP/BiomedCLIP.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

from src.datasets.analysis import CTCHOODDataset, ReportMismatchDataset
from src.datasets.btxrd import BTXRD_CLASS_NAMES, BTXRDDataset
from src.datasets.ctch import CTCHDataset
from src.utils.analysis import (
    ANALYSIS_METADATA_KEYS,
    AnalysisDataCollator,
    MetadataDataset,
    SOURCE_EXPERIMENT,
    SOURCE_SEEDS,
    load_evaluated_classification_model,
    load_feature_archive,
    save_feature_archive,
)
from src.utils.prompts import generate_clip_class_prompts

SCENARIOS = (
    "ctch_train",
    "ctch_val",
    "ctch_test",
    "ctch_ood",
    "btxrd_test",
    "report_mismatch_cross_class",
    "report_mismatch_same_class",
)


def _plain(value: Any) -> Any:
    """Chuyển đổi OmegaConf config sang kiểu dữ liệu Python nguyên bản.

    Parameters
    ----------
    value : Any
        Giá trị cấu hình đầu vào (OmegaConf DictConfig / ListConfig hoặc kiểu nguyên bản).

    Returns
    -------
    Any
        Dữ liệu kiểu Python chuẩn (dict, list, int, str,...).
    """
    return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value


def _to_device(value: Any, device: torch.device) -> Any:
    """Chuyển tensor lên thiết bị tính toán đích nếu là Tensor.

    Parameters
    ----------
    value : Any
        Dữ liệu hoặc Tensor đầu vào.
    device : torch.device
        Thiết bị đích (CUDA / CPU).

    Returns
    -------
    Any
        Tensor đã chuyển thiết bị hoặc giá trị gốc.
    """
    return value.to(device) if isinstance(value, torch.Tensor) else value


def _ctch_dataset(loaded: Any, split: str) -> CTCHDataset:
    """Khởi tạo CTCH dataset từ checkpoint configuration.

    Parameters
    ----------
    loaded : Any
        Đối tượng mô hình đã nạp checkpoint.
    split : str
        Phân chia dữ liệu ('train', 'val', hoặc 'test').

    Returns
    -------
    CTCHDataset
        Đối tượng dataset CTCH đã được cấu hình.
    """
    params = dict(_plain(loaded.cfg.dataset.params))
    return CTCHDataset(
        split=split,
        transform=loaded.model.backbone.preprocess,
        tokenizer=getattr(
            loaded.model.backbone,
            "tokenizer_obj",
            getattr(loaded.model.backbone, "tokenizer", None),
        ),
        **params,
    )


def _btxrd_coverage(dataset: BTXRDDataset) -> dict[str, Any]:
    """Kiểm tra tính toàn vẹn của tệp ảnh và bệnh sử BTXRD.

    Parameters
    ----------
    dataset : BTXRDDataset
        Tập dữ liệu BTXRD cần kiểm tra.

    Returns
    -------
    dict[str, Any]
        Thống kê số lượng hàng, số ảnh hợp lệ và số ảnh bị thiếu.
    """
    image_root = Path(dataset.img_dir)
    image_ids = list(dataset.df["image_id"])
    missing_images = [img_id for img_id in image_ids if not (image_root / img_id).is_file()]

    return {
        "rows": len(image_ids),
        "resolved_images": len(image_ids) - len(missing_images),
        "missing_images": len(missing_images),
        "dataset": "BTXRD",
    }


def build_scenario_dataset(
    scenario: str,
    loaded: Any,
    allow_incomplete_ood: bool = True,
    mismatch_seed: int = 42,
) -> tuple[torch.utils.data.Dataset, dict[str, Any]]:
    """Xây dựng dataset theo kịch bản đánh giá tương ứng.

    Parameters
    ----------
    scenario : str
        Tên kịch bản ('ctch_train', 'ctch_val', 'ctch_test', 'ctch_ood', 'btxrd_test', 'report_mismatch_*').
    loaded : Any
        Đối tượng mô hình đã nạp checkpoint.
    allow_incomplete_ood : bool, optional
        Cho phép bỏ qua các ca thiếu ảnh trong tập OOD ngoại vi, mặc định True.
    mismatch_seed : int, optional
        Hạt giống ngẫu nhiên phục vụ tráo đổi bệnh sử, mặc định 42.

    Returns
    -------
    tuple[torch.utils.data.Dataset, dict[str, Any]]
        Cặp (dataset, metadata) phục vụ trích xuất đặc trưng.

    Raises
    ------
    ValueError
        Khi tên kịch bản không hợp lệ.
    """
    if scenario in {"ctch_train", "ctch_val", "ctch_test"}:
        split = scenario.removeprefix("ctch_")
        return MetadataDataset(_ctch_dataset(loaded, split), scenario), {}

    if scenario == "ctch_ood":
        params = loaded.cfg.dataset.params
        preprocess = dict(_plain(getattr(params, "preprocess", {})))
        dataset = CTCHOODDataset(
            img_dir=str(params.img_dir),
            xray_report_dir=str(params.xray_report_dir),
            clinical_report_dir=str(params.clinical_report_dir),
            csv_manifest_path=str(ROOT / "data" / "CTCH" / "ctch-ood.csv"),
            transform=loaded.model.backbone.preprocess,
            tokenizer=getattr(
                loaded.model.backbone,
                "tokenizer_obj",
                getattr(loaded.model.backbone, "tokenizer", None),
            ),
            preprocess=preprocess,
            # The OOD pipeline fuses each image with its clinical report. X-ray
            # reports are not part of the current 26-case semantic-OOD cohort.
            required_report_types=("clinical",),
            allow_missing=allow_incomplete_ood,
        )
        return dataset, {"coverage": getattr(dataset, "coverage", {})}

    if scenario == "btxrd_test":
        preprocess = dict(
            _plain(getattr(loaded.cfg.dataset.params, "preprocess", {}))
        )
        dataset = BTXRDDataset(
            img_dir=str(ROOT / "data" / "BTXRD" / "images"),
            report_dir=str(ROOT / "data" / "BTXRD" / "reports"),
            clinical_subdir="clinical",
            csv_split_path=str(ROOT / "data" / "BTXRD" / "btxrd-split.csv"),
            csv_labels_path=str(ROOT / "data" / "BTXRD" / "btxrd-labels.csv"),
            classes=list(BTXRD_CLASS_NAMES),
            task_type="multiclass",
            split="test",
            transform=loaded.model.backbone.preprocess,
            tokenizer=getattr(
                loaded.model.backbone,
                "tokenizer_obj",
                getattr(loaded.model.backbone, "tokenizer", None),
            ),
            preprocess=preprocess,
        )
        coverage = _btxrd_coverage(dataset)
        return MetadataDataset(dataset, scenario), {
            "primary_feature": "fused_embeddings",
            "coverage": coverage,
            "ood_dataset": "BTXRD",
            "ood_split": "test",
            "ood_class_names": list(BTXRD_CLASS_NAMES),
        }

    if scenario.startswith("report_mismatch_"):
        mode = scenario.removeprefix("report_mismatch_")
        dataset = ReportMismatchDataset(
            _ctch_dataset(loaded, "test"), mode=mode, seed=mismatch_seed
        )
        return dataset, {"mismatch_seed": int(mismatch_seed), "mode": mode}

    raise ValueError(f"Kịch bản không hợp lệ: {scenario!r}")


def collect_fused_feature_batches(
    loaded: Any,
    loader: torch.utils.data.DataLoader,
    report_type: str = "clinical",
) -> dict[str, np.ndarray]:
    """Trích xuất đặc trưng nhúng và logits cho mô hình huấn luyện phân lớp.

    Parameters
    ----------
    loaded : Any
        Đối tượng mô hình đã nạp checkpoint.
    loader : torch.utils.data.DataLoader
        Bộ nạp dữ liệu.
    report_type : str, optional
        Loại văn bản sử dụng ('clinical' hoặc 'xray'), mặc định 'clinical'.

    Returns
    -------
    dict[str, np.ndarray]
        Từ điển mảng NumPy chứa 'fused_embeddings', 'logits', 'labels', 'predictions', v.v.

    Raises
    ------
    RuntimeError
        Khi tập dữ liệu nạp vào rỗng.
    """
    model = loaded.model
    device = loaded.device
    tensor_parts: dict[str, list[np.ndarray]] = {}
    metadata_parts: dict[str, list[str]] = {key: [] for key in ANALYSIS_METADATA_KEYS}
    labels: list[np.ndarray] = []
    prefix = "xray" if report_type == "xray" else "clinical"
    model.eval()

    with torch.no_grad():
        for batch in loader:
            images = _to_device(batch["pixel_values"], device)
            input_ids = _to_device(batch.get(f"{prefix}_input_ids", batch.get("input_ids")), device)
            attention_mask = _to_device(batch.get(f"{prefix}_attention_mask", batch.get("attention_mask")), device)

            encoded = model._encode_modalities(
                images,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            image_features, text_features, full_image_padding, image_patch_padding, text_token_padding = encoded

            fused = model._fuse_modalities(
                image_features,
                text_features,
                full_image_padding,
                image_patch_padding,
                text_token_padding,
            )
            logits = model.head(fused)
            outputs = {
                "fused_embeddings_raw": fused,
                "fused_embeddings": F.normalize(fused, dim=-1),
                "logits": logits,
                "probabilities": torch.softmax(logits, dim=-1),
            }
            if image_features is not None:
                image_global = image_features[:, 0] if image_features.ndim == 3 else image_features
                outputs["visual_global_embeddings"] = F.normalize(image_global, dim=-1)
            if text_features is not None:
                text_global = text_features[:, 0] if text_features.ndim == 3 else text_features
                outputs["text_global_embeddings"] = F.normalize(text_global, dim=-1)

            for key, value in outputs.items():
                tensor_parts.setdefault(key, []).append(value.detach().cpu().numpy())
            label_values = batch["labels"]
            labels.append(label_values.detach().cpu().numpy())
            for key in ANALYSIS_METADATA_KEYS:
                values = batch.get(key, [""] * len(label_values))
                metadata_parts[key].extend(str(value) for value in values)

    if not labels:
        raise RuntimeError("Tập dữ liệu trích xuất đặc trưng rỗng.")

    arrays = {key: np.concatenate(parts, axis=0) for key, parts in tensor_parts.items()}
    arrays["labels"] = np.concatenate(labels, axis=0)
    for key, values in metadata_parts.items():
        arrays[key] = np.asarray(values, dtype=str)
    arrays["predictions"] = arrays["logits"].argmax(axis=1).astype(np.int64)
    return arrays


def collect_zeroshot_feature_batches(
    loaded: Any,
    loader: torch.utils.data.DataLoader,
    prompt_classes: list[str] | None = None,
    temperature: float = 100.0,
) -> dict[str, np.ndarray]:
    """Trích xuất đặc trưng cho mô hình zero-shot CLIP/BiomedCLIP.

    Parameters
    ----------
    loaded : Any
        Đối tượng mô hình zero-shot đã nạp.
    loader : torch.utils.data.DataLoader
        Bộ nạp dữ liệu.
    prompt_classes : list[str] | None, optional
        Danh sách tên lớp tiếng Anh để sinh prompt câu chẩn đoán, mặc định None.
    temperature : float, optional
        Hệ số nhiệt độ nhân với cosine similarity, mặc định 100.0.

    Returns
    -------
    dict[str, np.ndarray]
        Từ điển mảng NumPy chứa logits, probabilities, labels, predictions.
    """
    model = loaded.model
    device = loaded.device
    if prompt_classes is None:
        prompt_classes = list(getattr(loaded, "classes", []))
    prompts = generate_clip_class_prompts(prompt_classes)

    text_tokens = loaded.tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )
    input_ids = text_tokens["input_ids"].to(device)
    attention_mask = text_tokens.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    model.eval()
    with torch.no_grad():
        text_features = model.backbone.encode_text(input_ids, attention_mask)
        if text_features.ndim == 3:
            text_features = text_features[:, 0]
        text_features = F.normalize(text_features, dim=-1)

    image_parts: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    metadata_parts: dict[str, list[str]] = {key: [] for key in ANALYSIS_METADATA_KEYS}

    with torch.no_grad():
        for batch in loader:
            images = _to_device(batch["pixel_values"], device)
            image_feat = model.backbone.encode_image(images)
            if image_feat.ndim == 3:
                image_feat = image_feat[:, 0]
            image_feat = F.normalize(image_feat, dim=-1)
            image_parts.append(image_feat.detach().cpu().numpy())

            label_values = batch["labels"]
            labels.append(label_values.detach().cpu().numpy())
            for key in ANALYSIS_METADATA_KEYS:
                values = batch.get(key, [""] * len(label_values))
                metadata_parts[key].extend(str(value) for value in values)

    visual_embeddings = np.concatenate(image_parts, axis=0)
    logits = visual_embeddings @ text_features.cpu().numpy().T * temperature
    probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs = probs / probs.sum(axis=-1, keepdims=True)

    arrays = {
        "fused_embeddings_raw": visual_embeddings,
        "fused_embeddings": visual_embeddings,
        "visual_global_embeddings": visual_embeddings,
        "logits": logits,
        "probabilities": probs,
        "labels": np.concatenate(labels, axis=0),
        "predictions": logits.argmax(axis=1).astype(np.int64),
    }
    for key, values in metadata_parts.items():
        arrays[key] = np.asarray(values, dtype=str)
    return arrays


# Aliases for backward compatibility
_collect_fused_feature_batches = collect_fused_feature_batches
_collect_zeroshot_feature_batches = collect_zeroshot_feature_batches


def export_features_for_experiment(
    experiment: str,
    seed: int,
    scenarios: list[str] | tuple[str, ...] = SCENARIOS,
    *,
    device: torch.device,
    batch_size: int = 16,
    num_workers: int = 0,
    force_recompute: bool = False,
) -> dict[str, dict[str, np.ndarray]]:
    """Trích xuất và lưu trữ đặc trưng cho một thí nghiệm cụ thể.

    Parameters
    ----------
    experiment : str
        Định danh thí nghiệm.
    seed : int
        Hạt giống ngẫu nhiên.
    scenarios : list[str] | tuple[str, ...], optional
        Danh sách các kịch bản cần trích xuất đặc trưng.
    device : torch.device
        Thiết bị thực thi.
    batch_size : int, optional
        Kích thước batch, mặc định 16.
    num_workers : int, optional
        Số luồng nạp dữ liệu, mặc định 0.
    force_recompute : bool, optional
        Buộc tính toán lại đặc trưng, mặc định False.

    Returns
    -------
    dict[str, dict[str, np.ndarray]]
        Từ điển chứa dữ liệu đặc trưng theo từng kịch bản.
    """
    save_dir = (
        ROOT / "results" / experiment / f"seed_{seed}" / "analysis" / "features"
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    loaded = None
    feature_dict: dict[str, dict[str, np.ndarray]] = {}

    for sc in scenarios:
        archive_path = save_dir / f"{sc}.npz"
        if not force_recompute and archive_path.is_file():
            try:
                arrays, provenance = load_feature_archive(
                    archive_path,
                    expected_source_experiment=experiment,
                )
                if int(provenance.get("seed", -1)) == int(seed):
                    feature_dict[sc] = arrays
                    print(f"  [Đã có sẵn] {sc}.npz ({experiment}, seed={seed})")
                    continue
            except Exception:
                pass

        if loaded is None:
            loaded = load_evaluated_classification_model(experiment, seed, device=device)

        print(f"  [Trích xuất] {sc} ({experiment}, seed={seed})...")
        dataset, meta = build_scenario_dataset(sc, loaded, allow_incomplete_ood=False)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=AnalysisDataCollator(
                loaded.model.backbone.tokenizer_obj
            ),
        )

        if "zeroshot" in experiment.lower():
            arrays = collect_zeroshot_feature_batches(loaded, loader)
        else:
            arrays = collect_fused_feature_batches(loaded, loader)

        save_feature_archive(
            archive_path,
            arrays,
            provenance={
                **loaded.provenance,
                "source_experiment": experiment,
                "experiment": experiment,
                "seed": seed,
                "scenario": sc,
                "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                **meta,
            },
        )
        feature_dict[sc] = arrays

    return feature_dict


def parse_args() -> argparse.Namespace:
    """Đọc và phân tích tham số dòng lệnh cho công cụ trích xuất đặc trưng.

    Returns
    -------
    argparse.Namespace
        Không gian tên chứa các đối số dòng lệnh.
    """
    parser = argparse.ArgumentParser(description="Export features for XBone-Net analysis")
    parser.add_argument(
        "--experiments",
        "--experiment",
        nargs="+",
        default=[SOURCE_EXPERIMENT],
        help="Danh sách thí nghiệm cần trích xuất đặc trưng",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--force-recompute", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Điểm vào chính khi chạy từ terminal."""
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")

    print("=" * 80)
    print("  XBONE-NET: CÔNG CỤ TRÍCH XUẤT ĐẶC TRƯNG PHÂN TÍCH (FEATURE EXPORT)")
    print(f"  • Thiết bị: {device}")
    print(f"  • Thí nghiệm: {args.experiments}")
    print(f"  • Hạt giống: {args.seeds}")
    print("=" * 80)

    for exp in args.experiments:
        print(f"\n>>> Xử lý thí nghiệm: {exp}")
        for seed in args.seeds:
            try:
                export_features_for_experiment(
                    exp,
                    seed,
                    args.scenarios,
                    device=device,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    force_recompute=args.force_recompute,
                )
            except Exception as e:
                print(f"  [LỖI] Không thể xử lý {exp} seed={seed}: {e}")


if __name__ == "__main__":
    main()
