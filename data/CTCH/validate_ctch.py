"""Kiểm tra tính toàn vẹn của bộ dữ liệu CTCH đã chuẩn bị.

Notes
-----
Chương trình chỉ đọc manifest, ảnh và văn bản đã chọn. Các tệp vật lý nằm ngoài
manifest được báo dưới dạng cảnh báo vì bộ nạp dữ liệu không sử dụng chúng.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image


EXPECTED_ID_SIZE = 3_249
EXPECTED_OOD_SIZE = 44
EXPECTED_SPLITS = {"train": 2_265, "validate": 315, "test": 669}
REPORT_LAYOUT = {
    "clinical": ("Reason for admission:", "Disease history:", "Personal medical history:"),
    "clinical_vi": ("Reason for admission:", "Disease history:", "Personal medical history:"),
}
OPTIONAL_REPORT_FIELDS = {
    "clinical": {"Personal medical history:"},
    "clinical_vi": {"Personal medical history:"},
}


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
        help="Thư mục CTCH chứa manifest, ảnh và báo cáo.",
    )
    parser.add_argument(
        "--skip-image-decode",
        action="store_true",
        help="Chỉ kiểm tra liên kết ảnh, không giải mã từng ảnh.",
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
    """
    checks.append(
        {
            "name": name,
            "passed": bool(passed),
            "warning": bool(warning),
            "detail": detail,
        }
    )


def require_paths(data_dir: Path, checks: list[dict[str, Any]]) -> dict[str, Path]:
    """Kiểm tra và trả về các đường dẫn đầu vào bắt buộc.

    Parameters
    ----------
    data_dir : pathlib.Path
        Thư mục gốc của CTCH.
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
        "labels": data_dir / "ctch-labels.csv",
        "splits": data_dir / "ctch-split.csv",
        "ood": data_dir / "ctch-ood.csv",
        "class_names": data_dir / "labels.txt",
        "images": data_dir / "images",
        "reports": data_dir / "reports",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    add_check(
        checks,
        "Tài nguyên bắt buộc",
        not missing,
        "Đủ tất cả tài nguyên." if not missing else f"Thiếu: {missing}",
    )
    if missing:
        raise FileNotFoundError("CTCH thiếu tài nguyên bắt buộc.")
    return paths


def normalize_identifier(value: Any) -> str:
    """Chuẩn hóa mã bệnh nhân để so sánh giữa các manifest.

    Parameters
    ----------
    value : Any
        Giá trị định danh cần chuẩn hóa.

    Returns
    -------
    str
        Định danh dạng chuỗi; giá trị thiếu được đổi thành chuỗi rỗng.
    """
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def load_class_names(path: Path) -> list[str]:
    """Nạp danh sách lớp CTCH theo đúng thứ tự mã lớp.

    Parameters
    ----------
    path : pathlib.Path
        Tệp ``labels.txt``.

    Returns
    -------
    list[str]
        Danh sách tên lớp khác rỗng.
    """
    return [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def validate_manifests(
    labels: pd.DataFrame,
    splits: pd.DataFrame,
    ood: pd.DataFrame,
    class_names: list[str],
    checks: list[dict[str, Any]],
) -> tuple[set[str], set[str]]:
    """Kiểm tra định danh, nhãn, tập chia và giao nhau bệnh nhân.

    Parameters
    ----------
    labels : pandas.DataFrame
        Manifest nhãn trong phân phối.
    splits : pandas.DataFrame
        Manifest tập chia theo bệnh nhân.
    ood : pandas.DataFrame
        Manifest ngoài không gian nhãn.
    class_names : list[str]
        Danh sách lớp chuẩn theo thứ tự.
    checks : list[dict[str, Any]]
        Danh sách nhận kết quả kiểm tra.

    Returns
    -------
    tuple[set[str], set[str]]
        Hai tập định danh ID và OOD.
    """
    required_labels = {"image_id", "class_id", "mapped_class", *class_names}
    required_splits = {"image_id", "split", "patient_key"}
    missing_labels = sorted(required_labels.difference(labels.columns))
    missing_splits = sorted(required_splits.difference(splits.columns))
    add_check(
        checks,
        "Lược đồ manifest ID",
        not missing_labels and not missing_splits and len(class_names) == 22,
        (
            f"Số lớp={len(class_names)}, thiếu ở labels={missing_labels}, "
            f"thiếu ở splits={missing_splits}."
        ),
    )
    if missing_labels or missing_splits or "image_id" not in ood:
        return set(), set()

    id_label_ids = set(labels["image_id"].astype(str))
    id_split_ids = set(splits["image_id"].astype(str))
    ood_ids = set(ood["image_id"].astype(str))
    unique_labels = labels["image_id"].astype(str).is_unique
    unique_splits = splits["image_id"].astype(str).is_unique
    unique_ood = ood["image_id"].astype(str).is_unique
    add_check(
        checks,
        "Định danh ID duy nhất",
        unique_labels and unique_splits,
        f"labels={len(id_label_ids)}/{len(labels)}, splits={len(id_split_ids)}/{len(splits)}.",
    )
    add_check(
        checks,
        "Định danh OOD duy nhất",
        unique_ood,
        f"OOD={len(ood_ids)}/{len(ood)}.",
    )
    add_check(
        checks,
        "Quần thể giữa labels và splits",
        id_label_ids == id_split_ids and len(id_label_ids) == EXPECTED_ID_SIZE,
        f"labels={len(id_label_ids)}, splits={len(id_split_ids)}, kỳ vọng={EXPECTED_ID_SIZE}.",
    )
    add_check(
        checks,
        "Quần thể OOD cố định",
        len(ood_ids) == EXPECTED_OOD_SIZE,
        f"Quan sát={len(ood_ids)}, kỳ vọng={EXPECTED_OOD_SIZE}.",
    )
    overlap = id_label_ids & ood_ids
    add_check(checks, "Ảnh không giao nhau giữa ID và OOD", not overlap, f"Giao nhau={len(overlap)} ảnh.")

    split_counts = splits["split"].value_counts().to_dict()
    valid_split_names = set(split_counts).issubset(EXPECTED_SPLITS)
    add_check(
        checks,
        "Quy mô tập chia cố định",
        valid_split_names and split_counts == EXPECTED_SPLITS,
        f"Quan sát={split_counts}, kỳ vọng={EXPECTED_SPLITS}.",
    )

    aligned = labels.set_index("image_id").loc[splits["image_id"].astype(str)]
    one_hot = aligned.loc[:, class_names].apply(pd.to_numeric, errors="coerce")
    binary = one_hot.isin((0, 1)).all().all()
    exactly_one = binary and one_hot.sum(axis=1).eq(1).all()
    add_check(
        checks,
        "Nhãn một-nóng 22 lớp",
        exactly_one,
        f"Đúng một lớp={int(one_hot.sum(axis=1).eq(1).sum())}/{len(one_hot)}.",
    )

    class_ids = pd.to_numeric(aligned["class_id"], errors="coerce")
    inferred_ids = one_hot.to_numpy().argmax(axis=1)
    valid_ids = class_ids.notna().all() and class_ids.between(0, len(class_names) - 1).all()
    id_matches = valid_ids and np.array_equal(class_ids.to_numpy(dtype=int), inferred_ids)
    expected_names = np.asarray(class_names, dtype=object)[inferred_ids]
    name_matches = np.array_equal(aligned["mapped_class"].astype(str).to_numpy(), expected_names)
    add_check(
        checks,
        "Nhất quán class_id, tên lớp và một-nóng",
        id_matches and name_matches,
        f"class_id khớp={id_matches}, mapped_class khớp={name_matches}.",
    )

    split_patients = {
        split_name: {
            normalize_identifier(value)
            for value in splits.loc[splits["split"] == split_name, "patient_key"]
            if normalize_identifier(value)
        }
        for split_name in EXPECTED_SPLITS
    }
    train_val = split_patients["train"] & split_patients["validate"]
    train_test = split_patients["train"] & split_patients["test"]
    val_test = split_patients["validate"] & split_patients["test"]
    add_check(
        checks,
        "Bệnh nhân không giao nhau giữa các tập",
        not train_val and not train_test and not val_test,
        f"train/validate={len(train_val)}, train/test={len(train_test)}, validate/test={len(val_test)}.",
    )

    patient_column = next(
        (column for column in ("Mã bệnh nhân", "patient_key", "patient_id") if column in ood),
        None,
    )
    if patient_column is None:
        add_check(checks, "Bệnh nhân ID/OOD không giao nhau", False, "Không tìm thấy cột mã bệnh nhân OOD.")
    else:
        id_patients = set().union(*split_patients.values())
        ood_patients = {
            normalize_identifier(value)
            for value in ood[patient_column]
            if normalize_identifier(value)
        }
        patient_overlap = id_patients & ood_patients
        add_check(
            checks,
            "Bệnh nhân ID/OOD không giao nhau",
            not patient_overlap,
            f"Giao nhau={len(patient_overlap)} bệnh nhân.",
        )
    return id_label_ids, ood_ids


def validate_images(
    selected_ids: set[str],
    image_dir: Path,
    checks: list[dict[str, Any]],
    *,
    decode: bool,
) -> None:
    """Kiểm tra liên kết và khả năng giải mã ảnh CTCH được chọn.

    Parameters
    ----------
    selected_ids : set[str]
        Tập định danh ID và OOD được manifest lựa chọn.
    image_dir : pathlib.Path
        Thư mục ảnh vật lý.
    checks : list[dict[str, Any]]
        Danh sách nhận kết quả kiểm tra.
    decode : bool
        Có giải mã từng ảnh hay không.
    """
    missing = sorted(image_id for image_id in selected_ids if not (image_dir / image_id).is_file())
    add_check(
        checks,
        "Liên kết ảnh được chọn",
        not missing,
        f"Tìm thấy={len(selected_ids) - len(missing)}/{len(selected_ids)}, thiếu={len(missing)}.",
    )
    physical_images = {path.name for path in image_dir.iterdir() if path.is_file()}
    extra = physical_images - selected_ids
    add_check(
        checks,
        "Ảnh ngoài manifest",
        not extra,
        f"Có {len(extra)} ảnh vật lý không được manifest lựa chọn.",
        warning=True,
    )
    if not decode or missing:
        return

    failures: list[str] = []
    for image_id in sorted(selected_ids):
        try:
            with Image.open(image_dir / image_id) as image:
                image.verify()
        except (OSError, ValueError) as error:
            failures.append(f"{image_id}: {error}")
    add_check(
        checks,
        "Giải mã ảnh được chọn",
        not failures,
        f"Giải mã được={len(selected_ids) - len(failures)}/{len(selected_ids)}.",
    )


def field_value(text: str, prefix: str) -> str | None:
    """Lấy nội dung của một trường văn bản theo tiền tố dòng.

    Parameters
    ----------
    text : str
        Toàn bộ nội dung báo cáo.
    prefix : str
        Tiền tố định danh trường.

    Returns
    -------
    str | None
        Nội dung đã loại khoảng trắng hoặc ``None`` nếu thiếu trường.
    """
    for line in text.splitlines():
        if line.startswith(prefix):
            return line[len(prefix):].strip()
    return None


def validate_reports(
    selected_ids: set[str],
    report_root: Path,
    checks: list[dict[str, Any]],
) -> None:
    """Kiểm tra bốn nguồn văn bản được liên kết với CTCH.

    Parameters
    ----------
    selected_ids : set[str]
        Tập định danh ID và OOD được manifest lựa chọn.
    report_root : pathlib.Path
        Thư mục gốc chứa bốn thư mục báo cáo.
    checks : list[dict[str, Any]]
        Danh sách nhận kết quả kiểm tra.
    """
    expected_names = {f"{Path(image_id).stem}.txt" for image_id in selected_ids}
    total_expected = len(expected_names) * len(REPORT_LAYOUT)
    total_found = 0
    total_nonempty = 0
    structural_errors: list[str] = []
    missing_core_fields: list[str] = []
    optional_missing: dict[str, int] = {
        f"{source}:{prefix}": 0
        for source, prefixes in OPTIONAL_REPORT_FIELDS.items()
        for prefix in prefixes
    }
    personal_history_present = 0
    extra_counts: dict[str, int] = {}

    for source, prefixes in REPORT_LAYOUT.items():
        directory = report_root / source
        if not directory.is_dir():
            structural_errors.append(f"Thiếu thư mục {source}")
            extra_counts[source] = 0
            continue
        actual_names = {path.name for path in directory.glob("*.txt")}
        missing = expected_names - actual_names
        total_found += len(expected_names) - len(missing)
        extra_counts[source] = len(actual_names - expected_names)
        for name in sorted(expected_names - missing):
            text = (directory / name).read_text(encoding="utf-8")
            if text.strip():
                total_nonempty += 1
            values = {prefix: field_value(text, prefix) for prefix in prefixes}
            required_prefixes = set(prefixes) - OPTIONAL_REPORT_FIELDS[source]
            if any(values[prefix] is None for prefix in required_prefixes):
                structural_errors.append(f"{source}/{name}")
            for prefix in prefixes:
                if prefix in OPTIONAL_REPORT_FIELDS[source]:
                    if values[prefix] is None:
                        optional_missing[f"{source}:{prefix}"] += 1
                if prefix == "Personal medical history:":
                    if values[prefix]:
                        personal_history_present += 1
                elif prefix in required_prefixes and not values[prefix]:
                    missing_core_fields.append(f"{source}/{name}:{prefix}")

    add_check(
        checks,
        "Liên kết bốn nguồn văn bản",
        total_found == total_expected,
        f"Tìm thấy={total_found}/{total_expected} tệp.",
    )
    add_check(
        checks,
        "Nội dung văn bản khác rỗng",
        total_nonempty == total_expected,
        f"Khác rỗng={total_nonempty}/{total_expected} tệp.",
    )
    add_check(
        checks,
        "Cấu trúc trường bắt buộc",
        not structural_errors,
        f"Số tệp thiếu tiền tố bắt buộc={len(structural_errors)}.",
    )
    add_check(
        checks,
        "Các trường cốt lõi khác rỗng",
        not missing_core_fields,
        f"Số trường lý do/diễn tiến/khảo sát ảnh chính bị rỗng={len(missing_core_fields)}.",
    )
    optional_missing_total = sum(optional_missing.values())
    add_check(
        checks,
        "Các trường nguồn tùy chọn",
        optional_missing_total == 0,
        f"Số trường tùy chọn không có trong nguồn={optional_missing}.",
        warning=True,
    )
    expected_clinical_files = len(expected_names) * 2
    add_check(
        checks,
        "Độ phủ tiền sử cá nhân",
        personal_history_present == expected_clinical_files,
        (
            f"Có nội dung={personal_history_present}/{expected_clinical_files} trường "
            "trên hai ngôn ngữ; trường thiếu được giữ nguyên theo dữ liệu nguồn."
        ),
        warning=True,
    )
    extra_total = sum(extra_counts.values())
    add_check(
        checks,
        "Văn bản ngoài manifest",
        extra_total == 0,
        f"Số tệp dư theo thư mục={extra_counts}.",
        warning=True,
    )


def print_report(checks: list[dict[str, Any]]) -> None:
    """In báo cáo kiểm tra ra màn hình.

    Parameters
    ----------
    checks : list[dict[str, Any]]
        Các kết quả cần hiển thị.
    """
    print("\n=== KIỂM TRA TÍNH TOÀN VẸN CTCH ===")
    for check in checks:
        if check["passed"]:
            status = "ĐẠT"
        elif check["warning"]:
            status = "CẢNH BÁO"
        else:
            status = "LỖI"
        print(f"[{status:8}] {check['name']}: {check['detail']}")


def main() -> int:
    """Thực thi toàn bộ quy trình kiểm tra CTCH.

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
        labels = pd.read_csv(paths["labels"], encoding="utf-8-sig")
        splits = pd.read_csv(paths["splits"], encoding="utf-8-sig")
        ood = pd.read_csv(paths["ood"], encoding="utf-8-sig")
        class_names = load_class_names(paths["class_names"])
        id_ids, ood_ids = validate_manifests(labels, splits, ood, class_names, checks)
        selected_ids = id_ids | ood_ids
        if selected_ids:
            validate_images(
                selected_ids,
                paths["images"],
                checks,
                decode=not args.skip_image_decode,
            )
            validate_reports(selected_ids, paths["reports"], checks)
    except Exception as error:  # Bảo đảm lỗi bất ngờ vẫn tạo báo cáo và mã thoát thất bại.
        add_check(checks, "Thực thi chương trình", False, f"{type(error).__name__}: {error}")

    print_report(checks)
    errors = [check for check in checks if not check["passed"] and not check["warning"]]
    warnings = [check for check in checks if not check["passed"] and check["warning"]]
    summary = {
        "dataset": "CTCH",
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
