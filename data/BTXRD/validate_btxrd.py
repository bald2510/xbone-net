"""Kiểm tra tính toàn vẹn của bộ dữ liệu BTXRD đã chuẩn bị.

Notes
-----
Chương trình chỉ đọc dữ liệu, không tự động sửa tệp. Mã thoát bằng ``0`` khi
không có lỗi toàn vẹn và bằng ``1`` khi phát hiện ít nhất một lỗi. Audit
disease-label similarity được báo riêng dưới dạng cảnh báo vì tín hiệu thống kê
không đồng nghĩa với nội dung chẩn đoán đúng hoặc rò rỉ nhãn có ý nghĩa thực tế.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


EXPECTED_SIZE = 3_746
EXPECTED_SPLITS = {"train": 2_621, "validate": 375, "test": 750}
CLASS_COLUMNS = (
    "normal",
    "osteochondroma",
    "osteosarcoma",
    "multiple osteochondromas",
    "simple bone cyst",
    "other bt",
    "giant cell tumor",
    "synovial osteochondroma",
    "other mt",
    "osteofibroma",
)
LOCATION_COLUMNS = (
    "hand",
    "ulna",
    "radius",
    "humerus",
    "foot",
    "tibia",
    "fibula",
    "femur",
    "hip bone",
    "ankle-joint",
    "knee-joint",
    "hip-joint",
    "wrist-joint",
    "elbow-joint",
    "shoulder-joint",
    "upper limb",
    "lower limb",
    "pelvis",
)
PROJECTION_COLUMNS = ("frontal", "lateral", "oblique")
REPORT_SECTIONS = (
    "Patient:",
    "Reason for admission:",
    "Disease history:",
    "Physical examination:",
    "Personal medical history:",
    "Radiographic evaluation:",
)
DIRECT_LABEL_TERMS = (
    "osteochondroma",
    "osteosarcoma",
    "multiple osteochondromas",
    "simple bone cyst",
    "giant cell tumor",
    "synovial osteochondroma",
    "osteofibroma",
)
DISEASE_LABEL_PROMPTS = {
    "normal": "normal bone without a bone tumor",
    "osteochondroma": "osteochondroma of bone",
    "osteosarcoma": "osteosarcoma of bone",
    "multiple osteochondromas": "multiple osteochondromas of bone",
    "simple bone cyst": "simple bone cyst",
    "other bt": "other benign bone tumor",
    "giant cell tumor": "giant cell tumor of bone",
    "synovial osteochondroma": "synovial osteochondromatosis",
    "other mt": "other malignant bone tumor",
    "osteofibroma": "osteofibroma of bone",
}
BIOMEDCLIP_MODEL_NAME = (
    "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
)


def parse_args() -> argparse.Namespace:
    """Đọc tham số dòng lệnh.

    Returns
    -------
    argparse.Namespace
        Các tham số đã được phân tích.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Thư mục BTXRD chứa manifest, ảnh và văn bản.",
    )
    parser.add_argument(
        "--skip-image-decode",
        action="store_true",
        help="Chỉ kiểm tra liên kết ảnh, không giải mã từng ảnh.",
    )
    parser.add_argument(
        "--skip-report-replay",
        action="store_true",
        help="Không phát lại bộ sinh văn bản lâm sàng với seed 42.",
    )
    parser.add_argument(
        "--skip-disease-label-similarity",
        action="store_true",
        help="Không chạy audit semantic similarity giữa văn bản và mười nhãn bệnh.",
    )
    parser.add_argument(
        "--similarity-device",
        default="auto",
        help="Thiết bị cho BioMedCLIP: auto, cpu, cuda hoặc cuda:N.",
    )
    parser.add_argument(
        "--similarity-batch-size",
        type=int,
        default=64,
        help="Kích thước batch khi mã hóa văn bản bằng BioMedCLIP.",
    )
    parser.add_argument(
        "--similarity-permutations",
        type=int,
        default=1_000,
        help="Số lần hoán vị dùng để kiểm định tín hiệu nhãn.",
    )
    parser.add_argument(
        "--similarity-alpha",
        type=float,
        default=0.05,
        help="Mức ý nghĩa để cảnh báo rò rỉ nhãn bệnh cụ thể.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Tùy chọn lưu báo cáo kiểm tra dưới dạng JSON.",
    )
    return parser.parse_args()


def add_check(
    checks: list[dict[str, Any]],
    name: str,
    passed: bool,
    detail: str,
    *,
    warning: bool = False,
    metrics: dict[str, Any] | None = None,
) -> None:
    """Ghi nhận kết quả của một phép kiểm tra.

    Parameters
    ----------
    checks : list[dict[str, Any]]
        Danh sách kết quả đang được xây dựng.
    name : str
        Tên ngắn của phép kiểm tra.
    passed : bool
        Trạng thái đạt hoặc không đạt.
    detail : str
        Mô tả định lượng hoặc nguyên nhân lỗi.
    warning : bool, optional
        Nếu đúng, kết quả không đạt chỉ được xem là cảnh báo.
    metrics : dict[str, Any], optional
        Các số liệu có cấu trúc để lưu trong báo cáo JSON.
    """
    result = {
        "name": name,
        "passed": bool(passed),
        "warning": bool(warning),
        "detail": detail,
    }
    if metrics is not None:
        result["metrics"] = metrics
    checks.append(result)


def require_paths(data_dir: Path, checks: list[dict[str, Any]]) -> dict[str, Path]:
    """Kiểm tra và trả về các đường dẫn đầu vào bắt buộc.

    Parameters
    ----------
    data_dir : pathlib.Path
        Thư mục gốc của BTXRD.
    checks : list[dict[str, Any]]
        Danh sách nhận kết quả kiểm tra.

    Returns
    -------
    dict[str, pathlib.Path]
        Ánh xạ tên tài nguyên sang đường dẫn tuyệt đối.

    Raises
    ------
    FileNotFoundError
        Khi thiếu ít nhất một tài nguyên bắt buộc.
    """
    paths = {
        "dataset": data_dir / "dataset.csv",
        "labels": data_dir / "btxrd-labels.csv",
        "splits": data_dir / "btxrd-split.csv",
        "images": data_dir / "images",
        "reports": data_dir / "reports",
        "generator": data_dir / "generate_clinical_reports.py",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    add_check(
        checks,
        "Tài nguyên bắt buộc",
        not missing,
        "Đủ tất cả tài nguyên." if not missing else f"Thiếu: {missing}",
    )
    if missing:
        raise FileNotFoundError("BTXRD thiếu tài nguyên bắt buộc.")
    return paths


def load_manifests(paths: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Nạp ba manifest chính của BTXRD.

    Parameters
    ----------
    paths : dict[str, pathlib.Path]
        Các đường dẫn đầu vào đã được xác nhận.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]
        Siêu dữ liệu, nhãn chuẩn hóa và ánh xạ tập chia.
    """
    return (
        pd.read_csv(paths["dataset"], encoding="utf-8-sig"),
        pd.read_csv(paths["labels"], encoding="utf-8-sig"),
        pd.read_csv(paths["splits"], encoding="utf-8-sig"),
    )


def validate_manifests(
    dataset: pd.DataFrame,
    labels: pd.DataFrame,
    splits: pd.DataFrame,
    checks: list[dict[str, Any]],
) -> None:
    """Kiểm tra khóa định danh, nhãn và phép chia dữ liệu.

    Parameters
    ----------
    dataset : pandas.DataFrame
        Siêu dữ liệu gốc.
    labels : pandas.DataFrame
        Manifest nhãn mười lớp.
    splits : pandas.DataFrame
        Manifest train, validate và test.
    checks : list[dict[str, Any]]
        Danh sách nhận kết quả kiểm tra.
    """
    required_dataset = {
        "image_id",
        "age",
        "gender",
        "tumor",
        "malignant",
        *LOCATION_COLUMNS,
        *PROJECTION_COLUMNS,
        *CLASS_COLUMNS[1:],
    }
    missing_columns = sorted(required_dataset.difference(dataset.columns))
    add_check(
        checks,
        "Lược đồ dataset.csv",
        not missing_columns,
        "Đủ các cột bắt buộc." if not missing_columns else f"Thiếu cột: {missing_columns}",
    )
    if missing_columns:
        return

    frames = {"dataset.csv": dataset, "btxrd-labels.csv": labels, "btxrd-split.csv": splits}
    for name, frame in frames.items():
        valid_key = "image_id" in frame and frame["image_id"].notna().all()
        unique_key = valid_key and frame["image_id"].astype(str).is_unique
        add_check(
            checks,
            f"Định danh duy nhất trong {name}",
            unique_key,
            f"{frame['image_id'].nunique() if valid_key else 0}/{len(frame)} định danh duy nhất.",
        )

    dataset_ids = set(dataset["image_id"].astype(str))
    label_ids = set(labels["image_id"].astype(str)) if "image_id" in labels else set()
    split_ids = set(splits["image_id"].astype(str)) if "image_id" in splits else set()
    add_check(
        checks,
        "Quần thể giữa ba manifest",
        dataset_ids == label_ids == split_ids and len(dataset_ids) == EXPECTED_SIZE,
        (
            f"dataset={len(dataset_ids)}, labels={len(label_ids)}, splits={len(split_ids)}, "
            f"kỳ vọng={EXPECTED_SIZE}."
        ),
    )

    if "split" in splits:
        split_counts = splits["split"].value_counts().to_dict()
        valid_splits = set(split_counts).issubset(EXPECTED_SPLITS)
        add_check(
            checks,
            "Quy mô tập chia cố định",
            valid_splits and split_counts == EXPECTED_SPLITS,
            f"Quan sát={split_counts}, kỳ vọng={EXPECTED_SPLITS}.",
        )
    else:
        add_check(checks, "Quy mô tập chia cố định", False, "Thiếu cột split.")

    missing_label_columns = [column for column in CLASS_COLUMNS if column not in labels]
    add_check(
        checks,
        "Lược đồ nhãn mười lớp",
        not missing_label_columns,
        "Đủ mười cột nhãn." if not missing_label_columns else f"Thiếu: {missing_label_columns}",
    )
    if missing_label_columns or dataset_ids != label_ids:
        return

    aligned_labels = labels.set_index("image_id").loc[dataset["image_id"].astype(str)]
    raw_numeric_labels = aligned_labels.loc[:, CLASS_COLUMNS].apply(
        pd.to_numeric,
        errors="coerce",
    )
    numeric_labels = raw_numeric_labels.copy()
    hierarchical_rows = numeric_labels[
        ["multiple osteochondromas", "synovial osteochondroma"]
    ].eq(1).any(axis=1)
    numeric_labels.loc[hierarchical_rows, "osteochondroma"] = 0
    binary = numeric_labels.isin((0, 1)).all().all()
    one_hot = binary and numeric_labels.sum(axis=1).eq(1).all()
    add_check(
        checks,
        "Tính một-nóng của nhãn",
        one_hot,
        (
            f"{int(numeric_labels.sum(axis=1).eq(1).sum())}/{len(numeric_labels)} "
            "hàng có đúng một lớp sau chuẩn hóa quan hệ cha-con."
        ),
    )

    expected = dataset.loc[:, CLASS_COLUMNS[1:]].apply(pd.to_numeric, errors="coerce").copy()
    expected.insert(0, "normal", pd.to_numeric(dataset["tumor"], errors="coerce").eq(0).astype(int))
    child_active = expected[["multiple osteochondromas", "synovial osteochondroma"]].eq(1).any(axis=1)
    expected.loc[child_active, "osteochondroma"] = 0
    expected.index = dataset["image_id"].astype(str)
    mismatched = int((expected.loc[:, CLASS_COLUMNS] != numeric_labels).any(axis=1).sum())
    add_check(
        checks,
        "Nhãn khớp siêu dữ liệu nguồn",
        mismatched == 0,
        f"Số hàng không khớp sau xử lý quan hệ cha-con: {mismatched}.",
    )

    missing_age = int(dataset["age"].isna().sum())
    missing_gender = int(dataset["gender"].isna().sum())
    missing_location = int(dataset.loc[:, LOCATION_COLUMNS].fillna(0).eq(1).sum(axis=1).eq(0).sum())
    missing_projection = int(dataset.loc[:, PROJECTION_COLUMNS].fillna(0).eq(1).sum(axis=1).eq(0).sum())
    missing_metadata = missing_age + missing_gender + missing_location + missing_projection
    add_check(
        checks,
        "Siêu dữ liệu dùng để sinh văn bản",
        missing_metadata == 0,
        (
            f"Thiếu tuổi={missing_age}, giới tính={missing_gender}, "
            f"vị trí={missing_location}, tư thế={missing_projection}."
        ),
    )


def validate_images(
    dataset: pd.DataFrame,
    image_dir: Path,
    checks: list[dict[str, Any]],
    *,
    decode: bool,
) -> None:
    """Kiểm tra liên kết và khả năng giải mã ảnh BTXRD.

    Parameters
    ----------
    dataset : pandas.DataFrame
        Siêu dữ liệu chứa tên tệp ảnh.
    image_dir : pathlib.Path
        Thư mục ảnh vật lý.
    checks : list[dict[str, Any]]
        Danh sách nhận kết quả kiểm tra.
    decode : bool
        Có giải mã từng ảnh hay không.
    """
    image_ids = dataset["image_id"].astype(str).tolist()
    missing = [image_id for image_id in image_ids if not (image_dir / image_id).is_file()]
    add_check(
        checks,
        "Liên kết ảnh",
        not missing,
        f"Tìm thấy {len(image_ids) - len(missing)}/{len(image_ids)} ảnh; thiếu {len(missing)}.",
    )
    if not decode or missing:
        return

    failures: list[str] = []
    for image_id in image_ids:
        try:
            with Image.open(image_dir / image_id) as image:
                image.verify()
        except (OSError, ValueError) as error:
            failures.append(f"{image_id}: {error}")
    add_check(
        checks,
        "Giải mã ảnh",
        not failures,
        f"Giải mã được {len(image_ids) - len(failures)}/{len(image_ids)} ảnh.",
    )


def load_generator(path: Path) -> Any:
    """Nạp mô-đun sinh văn bản BTXRD mà không thực thi hàm ``main``.

    Parameters
    ----------
    path : pathlib.Path
        Đường dẫn đến bộ sinh văn bản.

    Returns
    -------
    Any
        Mô-đun Python đã được nạp.

    Raises
    ------
    ImportError
        Khi không thể tạo đặc tả nạp mô-đun.
    """
    specification = importlib.util.spec_from_file_location("btxrd_report_generator", path)
    if specification is None or specification.loader is None:
        raise ImportError(f"Không thể nạp bộ sinh: {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def validate_reports(
    dataset: pd.DataFrame,
    report_dir: Path,
    generator_path: Path,
    checks: list[dict[str, Any]],
    *,
    replay: bool,
) -> None:
    """Kiểm tra ánh xạ, cấu trúc và khả năng tái lập văn bản BTXRD.

    Parameters
    ----------
    dataset : pandas.DataFrame
        Siêu dữ liệu theo đúng thứ tự sinh báo cáo.
    report_dir : pathlib.Path
        Thư mục văn bản lâm sàng.
    generator_path : pathlib.Path
        Tệp Python chứa hàm sinh báo cáo.
    checks : list[dict[str, Any]]
        Danh sách nhận kết quả kiểm tra.
    replay : bool
        Có phát lại bộ sinh với seed 42 hay không.
    """
    expected_names = {f"{Path(value).stem}.txt" for value in dataset["image_id"].astype(str)}
    actual_names = {path.name for path in report_dir.glob("*.txt")}
    missing = sorted(expected_names - actual_names)
    extra = sorted(actual_names - expected_names)
    add_check(
        checks,
        "Liên kết văn bản",
        not missing and not extra,
        f"Khớp={len(expected_names) - len(missing)}/{len(expected_names)}, thiếu={len(missing)}, dư={len(extra)}.",
    )
    if missing:
        return

    texts: dict[str, str] = {}
    empty: list[str] = []
    malformed: list[str] = []
    leakage: list[str] = []
    word_lengths: list[int] = []
    for name in sorted(expected_names):
        text = (report_dir / name).read_text(encoding="utf-8")
        texts[name] = text
        if not text.strip():
            empty.append(name)
        if any(section not in text for section in REPORT_SECTIONS):
            malformed.append(name)
        lowered = text.casefold()
        if any(term in lowered for term in DIRECT_LABEL_TERMS):
            leakage.append(name)
        word_lengths.append(len(text.split()))

    add_check(checks, "Văn bản khác rỗng", not empty, f"Khác rỗng={len(texts) - len(empty)}/{len(texts)}.")
    add_check(
        checks,
        "Sáu trường cấu trúc",
        not malformed,
        f"Đủ cấu trúc={len(texts) - len(malformed)}/{len(texts)}.",
    )
    add_check(
        checks,
        "Không chèn trực tiếp tên lớp",
        not leakage,
        f"Phát hiện tên bệnh đích trong {len(leakage)}/{len(texts)} văn bản.",
    )

    duplicate_count = len(texts) - len(set(texts.values()))
    add_check(
        checks,
        "Tính duy nhất của văn bản",
        duplicate_count == 0,
        f"Duy nhất={len(set(texts.values()))}/{len(texts)}, trùng={duplicate_count}.",
        warning=True,
    )
    add_check(
        checks,
        "Độ dài văn bản",
        bool(word_lengths),
        (
            f"Trung bình={statistics.mean(word_lengths):.2f}, "
            f"độ lệch chuẩn={statistics.pstdev(word_lengths):.2f}, "
            f"trung vị={statistics.median(word_lengths):.0f}, "
            f"miền={min(word_lengths)}-{max(word_lengths)} từ."
        ),
    )

    if not replay:
        return
    generator = load_generator(generator_path)
    generator.random.seed(42)
    mismatches: list[str] = []
    for _, row in dataset.iterrows():
        name = f"{Path(str(row['image_id'])).stem}.txt"
        regenerated = generator.generate_report_v2(row)
        if regenerated != texts[name]:
            mismatches.append(name)
    add_check(
        checks,
        "Phát lại bộ sinh với seed 42",
        not mismatches,
        f"Khớp từng ký tự={len(dataset) - len(mismatches)}/{len(dataset)}.",
    )


def resolve_similarity_device(value: str) -> torch.device:
    """Xác định thiết bị dùng cho phép đo disease-label similarity.

    Parameters
    ----------
    value : str
        Chuỗi thiết bị ``auto``, ``cpu``, ``cuda`` hoặc ``cuda:N``.

    Returns
    -------
    torch.device
        Thiết bị PyTorch đã được xác nhận.

    Raises
    ------
    RuntimeError
        Khi yêu cầu CUDA nhưng CUDA không khả dụng.
    """
    normalized = str(value).strip().casefold()
    if normalized == "auto":
        normalized = "cuda" if torch.cuda.is_available() else "cpu"
    if normalized.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA được yêu cầu nhưng không khả dụng.")
    return torch.device(normalized)


def normalized_label_indices(labels: pd.DataFrame, image_ids: pd.Series) -> np.ndarray:
    """Chuyển manifest nhãn BTXRD thành chỉ số lớp sau chuẩn hóa cha-con.

    Parameters
    ----------
    labels : pandas.DataFrame
        Manifest nhãn mười lớp.
    image_ids : pandas.Series
        Thứ tự định danh cần căn chỉnh.

    Returns
    -------
    numpy.ndarray
        Chỉ số lớp từ 0 đến 9 theo ``CLASS_COLUMNS``.

    Raises
    ------
    ValueError
        Khi nhãn không phải nhị phân hoặc không có đúng một lớp.
    """
    aligned = labels.set_index("image_id").loc[image_ids.astype(str), CLASS_COLUMNS]
    numeric = aligned.apply(pd.to_numeric, errors="coerce").copy()
    child_active = numeric[
        ["multiple osteochondromas", "synovial osteochondroma"]
    ].eq(1).any(axis=1)
    numeric.loc[child_active, "osteochondroma"] = 0
    if not numeric.isin((0, 1)).all().all() or not numeric.sum(axis=1).eq(1).all():
        raise ValueError("Không thể tạo chỉ số lớp từ manifest nhãn BTXRD.")
    return numeric.to_numpy(dtype=np.int64).argmax(axis=1)


def pathology_group_indices(dataset: pd.DataFrame) -> np.ndarray:
    """Suy ra nhóm không u, u lành và u ác từ metadata nguồn.

    Parameters
    ----------
    dataset : pandas.DataFrame
        Siêu dữ liệu BTXRD theo đúng thứ tự mẫu.

    Returns
    -------
    numpy.ndarray
        Mã nhóm 0 cho không u, 1 cho u không ác tính và 2 cho u ác tính.
    """
    tumor = pd.to_numeric(dataset["tumor"], errors="coerce").fillna(0).eq(1)
    malignant = pd.to_numeric(dataset["malignant"], errors="coerce").fillna(0).eq(1)
    groups = np.zeros(len(dataset), dtype=np.int64)
    groups[tumor.to_numpy()] = 1
    groups[(tumor & malignant).to_numpy()] = 2
    return groups


def tokenize_biomedclip(tokenizer: Any, texts: list[str], device: torch.device) -> torch.Tensor:
    """Token hóa một batch văn bản theo giao diện BioMedCLIP.

    Parameters
    ----------
    tokenizer : Any
        Tokenizer được OpenCLIP cung cấp cho BioMedCLIP.
    texts : list[str]
        Batch văn bản cần mã hóa.
    device : torch.device
        Thiết bị nhận tensor token.

    Returns
    -------
    torch.Tensor
        Tensor chỉ số token trên thiết bị đã chọn.
    """
    tokenized = tokenizer(texts)
    if isinstance(tokenized, Mapping):
        tokenized = tokenized["input_ids"]
    if not isinstance(tokenized, torch.Tensor):
        tokenized = torch.as_tensor(tokenized, dtype=torch.long)
    return tokenized.to(device)


def encode_biomedclip_texts(
    model: torch.nn.Module,
    tokenizer: Any,
    texts: list[str],
    device: torch.device,
    *,
    batch_size: int,
) -> np.ndarray:
    """Mã hóa và chuẩn hóa L2 danh sách văn bản bằng BioMedCLIP.

    Parameters
    ----------
    model : torch.nn.Module
        Mô hình BioMedCLIP nền tảng ở chế độ đánh giá.
    tokenizer : Any
        Tokenizer đi kèm mô hình.
    texts : list[str]
        Danh sách văn bản đầu vào.
    device : torch.device
        Thiết bị suy luận.
    batch_size : int
        Số văn bản trong mỗi batch.

    Returns
    -------
    numpy.ndarray
        Ma trận embedding chuẩn hóa có dạng ``[N, D]``.
    """
    parts: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            input_ids = tokenize_biomedclip(tokenizer, batch, device)
            embeddings = model.encode_text(input_ids)
            embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
            parts.append(embeddings.detach().cpu().numpy())
    return np.concatenate(parts, axis=0)


def permutation_similarity_test(
    similarities: np.ndarray,
    labels: np.ndarray,
    *,
    permutations: int,
    seed: int,
    groups: np.ndarray | None = None,
) -> dict[str, float]:
    """Kiểm định một phía cho cosine của nhãn thật bằng phép hoán vị.

    Parameters
    ----------
    similarities : numpy.ndarray
        Ma trận cosine ``[N, C]`` giữa report và prompt lớp.
    labels : numpy.ndarray
        Chỉ số lớp thật của từng report.
    permutations : int
        Số lần hoán vị nhãn.
    seed : int
        Seed của bộ sinh hoán vị.
    groups : numpy.ndarray, optional
        Nếu có, nhãn chỉ được hoán vị trong từng nhóm bệnh lý thô.

    Returns
    -------
    dict[str, float]
        Statistic quan sát, trung bình null, effect và p-value một phía.
    """
    rows = np.arange(len(labels))
    observed = float(similarities[rows, labels].mean())
    rng = np.random.default_rng(seed)
    null_statistics = np.empty(permutations, dtype=np.float64)
    group_values = np.unique(groups) if groups is not None else np.asarray([])
    for index in range(permutations):
        if groups is None:
            permuted = rng.permutation(labels)
        else:
            permuted = labels.copy()
            for group in group_values:
                positions = np.flatnonzero(groups == group)
                permuted[positions] = rng.permutation(labels[positions])
        null_statistics[index] = similarities[rows, permuted].mean()
    p_value = float(
        (1 + np.count_nonzero(null_statistics >= observed))
        / (permutations + 1)
    )
    null_mean = float(null_statistics.mean())
    return {
        "observed_mean_true_label_cosine": observed,
        "null_mean_true_label_cosine": null_mean,
        "effect_over_null": observed - null_mean,
        "null_standard_deviation": float(null_statistics.std(ddof=1)),
        "p_value_one_sided": p_value,
        "permutations": int(permutations),
    }


def validate_disease_label_similarity(
    dataset: pd.DataFrame,
    labels: pd.DataFrame,
    report_dir: Path,
    checks: list[dict[str, Any]],
    *,
    device_name: str,
    batch_size: int,
    permutations: int,
    alpha: float,
) -> None:
    """Audit rò rỉ nhãn bệnh bằng similarity với prompt của mười lớp.

    Phép hoán vị toàn cục đo toàn bộ liên hệ giữa văn bản và nhãn. Phép hoán vị
    có điều kiện trong ba nhóm không u, u lành và u ác kiểm tra tín hiệu bệnh cụ
    thể vượt quá tín hiệu nhóm vốn được đưa chủ động vào luật sinh văn bản.

    Parameters
    ----------
    dataset : pandas.DataFrame
        Siêu dữ liệu BTXRD theo đúng thứ tự sinh báo cáo.
    labels : pandas.DataFrame
        Manifest nhãn mười lớp.
    report_dir : pathlib.Path
        Thư mục chứa văn bản lâm sàng tổng hợp.
    checks : list[dict[str, Any]]
        Danh sách nhận kết quả kiểm tra.
    device_name : str
        Thiết bị dùng để chạy BioMedCLIP.
    batch_size : int
        Kích thước batch mã hóa văn bản.
    permutations : int
        Số lần hoán vị cho mỗi phép kiểm định.
    alpha : float
        Mức ý nghĩa dùng để phát cảnh báo.

    Raises
    ------
    ValueError
        Khi tham số kiểm định không hợp lệ hoặc thiếu prompt lớp.
    """
    if batch_size < 1:
        raise ValueError("--similarity-batch-size phải lớn hơn 0.")
    if permutations < 1:
        raise ValueError("--similarity-permutations phải lớn hơn 0.")
    if not 0.0 < alpha < 1.0:
        raise ValueError("--similarity-alpha phải nằm trong khoảng (0, 1).")
    missing_prompts = sorted(set(CLASS_COLUMNS).difference(DISEASE_LABEL_PROMPTS))
    if missing_prompts:
        raise ValueError(f"Thiếu disease-label prompt: {missing_prompts}")

    image_ids = dataset["image_id"].astype(str)
    report_paths = [report_dir / f"{Path(value).stem}.txt" for value in image_ids]
    reports = [path.read_text(encoding="utf-8").strip() for path in report_paths]
    label_indices = normalized_label_indices(labels, image_ids)
    pathology_groups = pathology_group_indices(dataset)
    prompt_texts = [
        f"Clinical history associated with {DISEASE_LABEL_PROMPTS[name]}."
        for name in CLASS_COLUMNS
    ]

    from open_clip import create_model_and_transforms, get_tokenizer

    device = resolve_similarity_device(device_name)
    model, _, _ = create_model_and_transforms(BIOMEDCLIP_MODEL_NAME)
    model = model.to(device).eval()
    tokenizer = get_tokenizer(BIOMEDCLIP_MODEL_NAME)
    report_embeddings = encode_biomedclip_texts(
        model,
        tokenizer,
        reports,
        device,
        batch_size=batch_size,
    )
    prompt_embeddings = encode_biomedclip_texts(
        model,
        tokenizer,
        prompt_texts,
        device,
        batch_size=len(prompt_texts),
    )
    similarities = report_embeddings @ prompt_embeddings.T
    predictions = similarities.argmax(axis=1)
    rows = np.arange(len(label_indices))
    true_similarities = similarities[rows, label_indices]
    other_similarities = similarities.copy()
    other_similarities[rows, label_indices] = -np.inf
    margins = true_similarities - other_similarities.max(axis=1)
    one_hot = np.eye(len(CLASS_COLUMNS), dtype=np.int64)[label_indices]

    global_test = permutation_similarity_test(
        similarities,
        label_indices,
        permutations=permutations,
        seed=42,
    )
    conditional_test = permutation_similarity_test(
        similarities,
        label_indices,
        permutations=permutations,
        seed=43,
        groups=pathology_groups,
    )
    per_class: dict[str, dict[str, float | int]] = {}
    for class_index, class_name in enumerate(CLASS_COLUMNS):
        positive = label_indices == class_index
        per_class[class_name] = {
            "count": int(positive.sum()),
            "retrieval_recall": float((predictions[positive] == class_index).mean()),
            "mean_positive_cosine": float(similarities[positive, class_index].mean()),
            "mean_negative_cosine": float(similarities[~positive, class_index].mean()),
            "auroc": float(roc_auc_score(positive.astype(np.int64), similarities[:, class_index])),
        }

    metrics: dict[str, Any] = {
        "encoder": BIOMEDCLIP_MODEL_NAME,
        "prompt_template": "Clinical history associated with {disease_label}.",
        "sample_count": len(reports),
        "class_count": len(CLASS_COLUMNS),
        "nominal_top1_chance": 1.0 / len(CLASS_COLUMNS),
        "nominal_macro_auroc_chance": 0.5,
        "top1_accuracy": float((predictions == label_indices).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(label_indices, predictions)),
        "macro_auroc": float(
            roc_auc_score(one_hot, similarities, average="macro")
        ),
        "mean_true_label_cosine": float(true_similarities.mean()),
        "mean_true_vs_best_other_margin": float(margins.mean()),
        "global_permutation_test": global_test,
        "within_pathology_group_permutation_test": conditional_test,
        "pathology_groups": {
            "0": "non-tumor",
            "1": "tumor without malignant flag",
            "2": "malignant tumor",
        },
        "per_class": per_class,
        "warning_rule": (
            "one-sided within-pathology-group permutation p-value <= alpha "
            "and effect_over_null > 0"
        ),
        "alpha": float(alpha),
    }
    conditional_p = conditional_test["p_value_one_sided"]
    exact_label_signal = bool(
        conditional_p <= alpha and conditional_test["effect_over_null"] > 0.0
    )
    add_check(
        checks,
        "Disease-label leakage similarity",
        not exact_label_signal,
        (
            f"BioMedCLIP Top-1={metrics['top1_accuracy']:.4f}, "
            f"Balanced Accuracy={metrics['balanced_accuracy']:.4f}, "
            f"Macro-AUROC={metrics['macro_auroc']:.4f}, "
            f"margin thật-tốt nhất khác={metrics['mean_true_vs_best_other_margin']:+.4f}; "
            f"hoán vị toàn cục: Δcos={global_test['effect_over_null']:+.6f}, "
            f"p={global_test['p_value_one_sided']:.4g}; trong nhóm bệnh lý: "
            f"Δcos={conditional_test['effect_over_null']:+.6f}, p={conditional_p:.4g}."
        ),
        warning=True,
        metrics=metrics,
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


def print_report(checks: list[dict[str, Any]]) -> None:
    """In báo cáo kiểm tra ra màn hình.

    Parameters
    ----------
    checks : list[dict[str, Any]]
        Các kết quả cần hiển thị.
    """
    print("\n=== KIỂM TRA TÍNH TOÀN VẸN BTXRD ===")
    for check in checks:
        if check["passed"]:
            status = "ĐẠT"
        elif check["warning"]:
            status = "CẢNH BÁO"
        else:
            status = "LỖI"
        print(f"[{status:8}] {check['name']}: {check['detail']}")


def main() -> int:
    """Thực thi toàn bộ quy trình kiểm tra BTXRD.

    Returns
    -------
    int
        Mã thoát 0 nếu dữ liệu đạt, ngược lại là 1.
    """
    args = parse_args()
    data_dir = args.data_dir.resolve()
    checks: list[dict[str, Any]] = []
    try:
        paths = require_paths(data_dir, checks)
        dataset, labels, splits = load_manifests(paths)
        validate_manifests(dataset, labels, splits, checks)
        validate_images(
            dataset,
            paths["images"],
            checks,
            decode=not args.skip_image_decode,
        )
        validate_reports(
            dataset,
            paths["reports"],
            paths["generator"],
            checks,
            replay=not args.skip_report_replay,
        )
        if not args.skip_disease_label_similarity:
            validate_disease_label_similarity(
                dataset,
                labels,
                paths["reports"],
                checks,
                device_name=args.similarity_device,
                batch_size=args.similarity_batch_size,
                permutations=args.similarity_permutations,
                alpha=args.similarity_alpha,
            )
    except Exception as error:  # Bảo đảm lỗi bất ngờ vẫn tạo báo cáo và mã thoát thất bại.
        add_check(checks, "Thực thi chương trình", False, f"{type(error).__name__}: {error}")

    print_report(checks)
    errors = [check for check in checks if not check["passed"] and not check["warning"]]
    warnings = [check for check in checks if not check["passed"] and check["warning"]]
    summary = {
        "dataset": "BTXRD",
        "data_dir": str(data_dir),
        "passed": not errors,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "checks": checks,
    }
    if args.json_output is not None:
        output = args.json_output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nĐã lưu báo cáo JSON: {output}")
    print(f"\nKẾT LUẬN: {'ĐẠT' if not errors else 'KHÔNG ĐẠT'} ({len(errors)} lỗi, {len(warnings)} cảnh báo)")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
