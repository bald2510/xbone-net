"""Tạo hình minh họa các ca được hợp nhất đa phương thức dự đoán đúng.

Notes
-----
Mô-đun này thuộc cơ sở mã nguồn nghiên cứu XBone-Net và giữ các quy ước dùng
chung của dự án. Ba chế độ suy luận dùng cùng một checkpoint và cùng classifier;
chỉ đầu vào của bước hợp nhất được giới hạn lần lượt ở ảnh, bệnh sử hoặc cả hai.
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps


matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.analysis import (
    SOURCE_EXPERIMENT,
    analysis_root,
    load_feature_archive,
    load_locked_proposed_model,
)
from src.utils.online_inference import _build_online_inputs


@dataclass(frozen=True)
class ModalityPrediction:
    """Lưu nhãn và softmax cao nhất của một chế độ suy luận."""

    class_index: int
    class_label: str
    probability: float


@dataclass(frozen=True)
class RescueCase:
    """Lưu một ca mà hợp nhất đa phương thức sửa hai dự đoán đơn phương thức."""

    image_id: str
    true_index: int
    true_label: str
    image_only: ModalityPrediction
    text_only: ModalityPrediction
    multimodal: ModalityPrediction
    inference_report: str
    output_figure: str


def parse_args() -> argparse.Namespace:
    """Phân tích tham số dòng lệnh của công cụ trực quan hóa.

    Returns
    -------
    argparse.Namespace
        Các tham số đã được phân tích.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Chọn các mẫu CTCH-test mà image-only và text-only sai nhưng "
            "XBone-Net đa phương thức đúng, sau đó tạo hình so sánh."
        )
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument(
        "--max-scan",
        type=int,
        default=200,
        help="Số mẫu multimodal đúng tối đa được suy luận lại để tìm ca phù hợp.",
    )
    parser.add_argument(
        "--allow-repeated-labels",
        action="store_true",
        help="Cho phép ba hình có cùng nhãn thật.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "results"
            / "visualization"
            / "modality_rescue_cases"
        ),
    )
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    """Chuyển lựa chọn thiết bị thành đối tượng PyTorch.

    Parameters
    ----------
    value : str
        Tên thiết bị, ví dụ ``auto``, ``cpu`` hoặc ``cuda``.

    Returns
    -------
    torch.device
        Thiết bị dùng để suy luận.

    Raises
    ------
    RuntimeError
        Khi yêu cầu CUDA nhưng CUDA không khả dụng.
    """
    normalized = str(value).strip().casefold()
    if normalized == "auto":
        normalized = "cuda" if torch.cuda.is_available() else "cpu"
    if normalized.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(normalized)


def load_class_labels() -> list[str]:
    """Tải danh sách nhãn CTCH theo đúng chỉ số lớp.

    Returns
    -------
    list[str]
        Danh sách tên lớp CTCH.

    Raises
    ------
    ValueError
        Khi ánh xạ nhãn không liên tục hoặc bị trùng chỉ số.
    """
    table = pd.read_csv(PROJECT_ROOT / "data" / "CTCH" / "ctch-labels.csv")
    mapping = (
        table[["class_id", "mapped_class"]]
        .drop_duplicates()
        .sort_values("class_id")
    )
    expected = list(range(len(mapping)))
    observed = mapping["class_id"].astype(int).tolist()
    if observed != expected:
        raise ValueError(f"CTCH class IDs are not contiguous: {observed}")
    return mapping["mapped_class"].astype(str).tolist()


def validate_archive(
    arrays: dict[str, np.ndarray],
    provenance: dict,
    *,
    seed: int,
    checkpoint_sha256: str,
) -> None:
    """Kiểm tra feature archive trước khi dùng để chọn ứng viên.

    Parameters
    ----------
    arrays : dict[str, np.ndarray]
        Các mảng đặc trưng và dự đoán đã xuất.
    provenance : dict
        Thông tin nguồn của feature archive.
    seed : int
        Seed của checkpoint đang sử dụng.
    checkpoint_sha256 : str
        SHA-256 của checkpoint đã tải.

    Raises
    ------
    ValueError
        Khi archive thiếu trường hoặc không khớp checkpoint, seed hay kịch bản.
    """
    required = {"image_id", "labels", "logits", "predictions"}
    missing = sorted(required.difference(arrays))
    if missing:
        raise ValueError(f"CTCH test archive is missing arrays: {missing}")
    if int(provenance.get("seed", -1)) != int(seed):
        raise ValueError("Feature archive seed differs from the loaded checkpoint.")
    if provenance.get("scenario") != "ctch_test":
        raise ValueError("Feature archive must represent the CTCH test split.")
    if provenance.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError(
            "Feature archive checkpoint SHA-256 differs from the loaded model."
        )
    lengths = {len(np.asarray(arrays[key])) for key in required}
    if len(lengths) != 1:
        raise ValueError(f"Feature archive arrays have different lengths: {lengths}")


def candidate_order(arrays: dict[str, np.ndarray]) -> np.ndarray:
    """Xếp ứng viên multimodal đúng theo độ chắc chắn giảm dần.

    Parameters
    ----------
    arrays : dict[str, np.ndarray]
        Feature archive của CTCH test.

    Returns
    -------
    np.ndarray
        Chỉ số các mẫu multimodal đúng theo thứ tự kiểm tra.
    """
    labels = np.asarray(arrays["labels"], dtype=np.int64)
    predictions = np.asarray(arrays["predictions"], dtype=np.int64)
    logits = np.asarray(arrays["logits"], dtype=np.float64)
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    correct = np.flatnonzero(predictions == labels)
    true_probabilities = probabilities[correct, labels[correct]]
    return correct[np.argsort(-true_probabilities, kind="stable")]


def infer_three_modalities(
    loaded,
    image: Image.Image,
    clinical_text: str,
    class_labels: list[str],
) -> tuple[ModalityPrediction, ModalityPrediction, ModalityPrediction]:
    """Suy luận ba chế độ bằng cùng backbone, head và checkpoint.

    Parameters
    ----------
    loaded : object
        Mô hình phân tích đã tải cùng cấu hình và thiết bị.
    image : PIL.Image.Image
        Ảnh X-quang đầu vào.
    clinical_text : str
        Bệnh sử tiếng Anh thực sự được đưa vào text encoder.
    class_labels : list[str]
        Danh sách tên lớp theo chỉ số.

    Returns
    -------
    tuple[ModalityPrediction, ModalityPrediction, ModalityPrediction]
        Dự đoán image-only, text-only và image+text.
    """
    inputs = _build_online_inputs(loaded, image, clinical_text)
    model = loaded.model
    with torch.inference_mode():
        encoded = model._encode_modalities(
            inputs["pixel_values"],
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            tile_values=inputs["tile_values"],
            tile_mask=inputs["tile_mask"],
            tile_boxes=inputs["tile_boxes"],
        )
        (
            image_tokens,
            text_tokens,
            full_image_padding,
            image_padding,
            text_padding,
        ) = encoded
        image_embedding = model._fuse_modalities(
            image_tokens,
            None,
            full_image_padding,
            image_padding,
            None,
        )
        text_embedding = model._fuse_modalities(
            None,
            text_tokens,
            None,
            None,
            text_padding,
        )
        multimodal_embedding = model._fuse_modalities(*encoded)
        logits_by_mode = (
            model.head(image_embedding),
            model.head(text_embedding),
            model.head(multimodal_embedding),
        )

    predictions: list[ModalityPrediction] = []
    for logits in logits_by_mode:
        probabilities = torch.softmax(logits, dim=-1)[0]
        class_index = int(probabilities.argmax().item())
        predictions.append(
            ModalityPrediction(
                class_index=class_index,
                class_label=class_labels[class_index],
                probability=float(probabilities[class_index].item()),
            )
        )
    return tuple(predictions)


def report_path_for(image_id: str) -> Path:
    """Xác định bệnh sử tiếng Anh tương ứng với một ảnh CTCH.

    Parameters
    ----------
    image_id : str
        Tên tệp ảnh CTCH.

    Returns
    -------
    pathlib.Path
        Đường dẫn bệnh sử dùng trong suy luận.
    """
    return (
        PROJECT_ROOT
        / "data"
        / "CTCH"
        / "reports"
        / "clinical"
        / Path(image_id).with_suffix(".txt").name
    )


def select_rescue_cases(
    loaded,
    arrays: dict[str, np.ndarray],
    class_labels: list[str],
    *,
    count: int,
    max_scan: int,
    require_distinct_labels: bool,
) -> tuple[list[RescueCase], int]:
    """Chọn các ca mà chỉ multimodal dự đoán đúng.

    Parameters
    ----------
    loaded : object
        Mô hình phân tích đã tải.
    arrays : dict[str, np.ndarray]
        Feature archive CTCH test dùng để lập thứ tự ứng viên.
    class_labels : list[str]
        Danh sách tên lớp CTCH.
    count : int
        Số ca cần chọn.
    max_scan : int
        Số ứng viên tối đa được suy luận lại.
    require_distinct_labels : bool
        Có ưu tiên mỗi hình thuộc một nhãn thật khác nhau hay không.

    Returns
    -------
    tuple[list[RescueCase], int]
        Các ca đã chọn và số ứng viên đã suy luận lại.

    Raises
    ------
    RuntimeError
        Khi không tìm đủ số ca thỏa điều kiện.
    """
    order = candidate_order(arrays)
    labels = np.asarray(arrays["labels"], dtype=np.int64)
    image_ids = np.asarray(arrays["image_id"]).astype(str)
    archive_predictions = np.asarray(arrays["predictions"], dtype=np.int64)
    selected: list[RescueCase] = []
    repeated_label_backups: list[RescueCase] = []
    selected_labels: set[int] = set()
    scanned = 0

    for index in order[:max_scan]:
        scanned += 1
        image_id = str(image_ids[index])
        true_index = int(labels[index])
        image_path = PROJECT_ROOT / "data" / "CTCH" / "images" / image_id
        inference_report_path = report_path_for(image_id)
        image = Image.open(image_path).convert("RGB")
        clinical_text = inference_report_path.read_text(encoding="utf-8").strip()
        image_only, text_only, multimodal = infer_three_modalities(
            loaded,
            image,
            clinical_text,
            class_labels,
        )

        # Archive chỉ lập thứ tự; điều kiện chọn luôn được kiểm chứng lại từ ảnh
        # và bệnh sử gốc bằng checkpoint hiện tại.
        if multimodal.class_index != int(archive_predictions[index]):
            raise RuntimeError(
                f"Recomputed multimodal prediction differs from archive: {image_id}."
            )
        if not (
            image_only.class_index != true_index
            and text_only.class_index != true_index
            and multimodal.class_index == true_index
        ):
            continue

        case = RescueCase(
            image_id=image_id,
            true_index=true_index,
            true_label=class_labels[true_index],
            image_only=image_only,
            text_only=text_only,
            multimodal=multimodal,
            inference_report=str(inference_report_path.resolve()),
            output_figure="",
        )
        if require_distinct_labels and true_index in selected_labels:
            repeated_label_backups.append(case)
            continue
        selected.append(case)
        selected_labels.add(true_index)
        if len(selected) >= count:
            break

    if len(selected) < count:
        selected.extend(repeated_label_backups[: count - len(selected)])
    if len(selected) < count:
        raise RuntimeError(
            f"Only found {len(selected)}/{count} qualifying cases after scanning "
            f"{scanned} multimodal-correct CTCH-test samples. Increase --max-scan."
        )
    return selected[:count], scanned


def format_report(text: str, *, width: int = 38, max_lines: int = 15) -> str:
    """Định dạng bệnh sử thành đoạn ngắn vừa khung hình.

    Parameters
    ----------
    text : str
        Bệnh sử cần hiển thị.
    width : int, optional
        Số ký tự mục tiêu trên mỗi dòng.
    max_lines : int, optional
        Số dòng tối đa được giữ lại.

    Returns
    -------
    str
        Bệnh sử đã xuống dòng và rút gọn.
    """
    normalized = " ".join(str(text).split())
    lines = textwrap.wrap(normalized, width=width)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip(" .") + "…"
    return "\n".join(lines)


def draw_prediction_box(
    axis,
    prediction: ModalityPrediction,
    *,
    correct: bool,
) -> None:
    """Vẽ nhãn dự đoán và trạng thái đúng/sai trên một nhánh.

    Parameters
    ----------
    axis : matplotlib.axes.Axes
        Trục chứa nhánh trực quan.
    prediction : ModalityPrediction
        Dự đoán cần hiển thị.
    correct : bool
        Dự đoán có khớp nhãn thật hay không.
    """
    color = "#18794e" if correct else "#b42318"
    facecolor = "#e8f5ee" if correct else "#fff0ee"
    axis.annotate(
        "",
        xy=(0.5, 0.235),
        xytext=(0.5, 0.34),
        xycoords="axes fraction",
        arrowprops={"arrowstyle": "-|>", "color": "#46505a", "lw": 2.0},
    )
    axis.text(
        0.5,
        0.15,
        f"{prediction.class_label}",
        transform=axis.transAxes,
        ha="center",
        va="center",
        fontsize=11,
        weight="bold",
        color=color,
        bbox={
            "boxstyle": "round,pad=0.65",
            "facecolor": facecolor,
            "edgecolor": color,
            "linewidth": 1.8,
        },
    )


def render_case_figure(
    case: RescueCase,
    clinical_text: str,
    output: Path,
    *,
    seed: int,
) -> Path:
    """Tạo một hình gồm ba nhánh image-only, text-only và multimodal.

    Parameters
    ----------
    case : RescueCase
        Ca cần trực quan hóa.
    clinical_text : str
        Bệnh sử tiếng Anh dùng khi suy luận.
    output : pathlib.Path
        Đường dẫn PNG đầu ra.
    seed : int
        Seed checkpoint dùng để tạo dự đoán.

    Returns
    -------
    pathlib.Path
        Đường dẫn hình đã ghi.
    """
    image_path = PROJECT_ROOT / "data" / "CTCH" / "images" / case.image_id
    image = ImageOps.autocontrast(Image.open(image_path).convert("L"))
    report = format_report(clinical_text)
    figure, axes = plt.subplots(1, 3, figsize=(16.5, 7.2), facecolor="white")
    titles = ["(a) Chỉ ảnh", "(b) Chỉ bệnh sử", "(c) Ảnh + bệnh sử"]
    for axis, title in zip(axes, titles):
        axis.set_axis_off()
        axis.set_title(title, fontsize=14, weight="bold", pad=12)

    image_axis = axes[0].inset_axes([0.08, 0.36, 0.84, 0.58])
    image_axis.imshow(image, cmap="gray")
    image_axis.set_axis_off()
    image_axis.set_title("Ảnh X-quang", fontsize=10)
    draw_prediction_box(axes[0], case.image_only, correct=False)

    axes[1].text(
        0.5,
        0.65,
        report,
        transform=axes[1].transAxes,
        ha="center",
        va="center",
        fontsize=10,
        linespacing=1.35,
        bbox={
            "boxstyle": "round,pad=0.8",
            "facecolor": "#f5f7fa",
            "edgecolor": "#7a8694",
        },
    )
    draw_prediction_box(axes[1], case.text_only, correct=False)

    multimodal_image_axis = axes[2].inset_axes([0.03, 0.46, 0.43, 0.45])
    multimodal_image_axis.imshow(image, cmap="gray")
    multimodal_image_axis.set_axis_off()
    axes[2].text(
        0.51,
        0.69,
        "+",
        transform=axes[2].transAxes,
        ha="center",
        va="center",
        fontsize=23,
        weight="bold",
        color="#46505a",
    )
    axes[2].text(
        0.76,
        0.69,
        format_report(clinical_text, width=23, max_lines=11),
        transform=axes[2].transAxes,
        ha="center",
        va="center",
        fontsize=8.5,
        linespacing=1.25,
        bbox={
            "boxstyle": "round,pad=0.55",
            "facecolor": "#f5f7fa",
            "edgecolor": "#7a8694",
        },
    )
    draw_prediction_box(axes[2], case.multimodal, correct=True)

    figure.suptitle(
        f"Nhãn thật: {case.true_label}",
        fontsize=16,
        weight="bold",
        y=0.985,
    )
    figure.subplots_adjust(left=0.025, right=0.975, top=0.86, bottom=0.08, wspace=0.08)
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output


def serializable_case(case: RescueCase) -> dict:
    """Chuyển ca được chọn thành ánh xạ có thể ghi JSON.

    Parameters
    ----------
    case : RescueCase
        Ca cần chuyển đổi.

    Returns
    -------
    dict
        Ánh xạ chỉ chứa kiểu dữ liệu tuần tự hóa được.
    """
    return asdict(case)


def main() -> None:
    """Chọn ba ca, tạo ba hình và ghi tệp kiểm toán kết quả.

    Raises
    ------
    ValueError
        Khi tham số số lượng hoặc giới hạn quét không hợp lệ.
    """
    args = parse_args()
    if args.count < 1:
        raise ValueError("--count must be at least 1.")
    if args.max_scan < args.count:
        raise ValueError("--max-scan must be greater than or equal to --count.")

    device = resolve_device(args.device)
    loaded = load_locked_proposed_model(args.seed, device=device)
    archive_path = analysis_root(args.seed) / "features" / "ctch_test.npz"
    arrays, provenance = load_feature_archive(
        archive_path,
        expected_source_experiment=SOURCE_EXPERIMENT,
    )
    validate_archive(
        arrays,
        provenance,
        seed=args.seed,
        checkpoint_sha256=loaded.checkpoint_sha256,
    )
    class_labels = load_class_labels()
    cases, scanned = select_rescue_cases(
        loaded,
        arrays,
        class_labels,
        count=args.count,
        max_scan=args.max_scan,
        require_distinct_labels=not args.allow_repeated_labels,
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered_cases: list[RescueCase] = []
    for rank, case in enumerate(cases, start=1):
        report_path = Path(case.inference_report)
        clinical_text = report_path.read_text(encoding="utf-8").strip()
        output = output_dir / f"case_{rank:02d}_{Path(case.image_id).stem}.png"
        render_case_figure(case, clinical_text, output, seed=args.seed)
        rendered_cases.append(
            RescueCase(
                **{
                    **asdict(case),
                    "image_only": case.image_only,
                    "text_only": case.text_only,
                    "multimodal": case.multimodal,
                    "output_figure": str(output.resolve()),
                }
            )
        )

    manifest_rows = []
    for case in rendered_cases:
        manifest_rows.append(
            {
                "image_id": case.image_id,
                "true_index": case.true_index,
                "true_label": case.true_label,
                "image_only_prediction": case.image_only.class_label,
                "image_only_softmax": case.image_only.probability,
                "text_only_prediction": case.text_only.class_label,
                "text_only_softmax": case.text_only.probability,
                "multimodal_prediction": case.multimodal.class_label,
                "multimodal_softmax": case.multimodal.probability,
                "inference_report": case.inference_report,
                "output_figure": case.output_figure,
            }
        )
    manifest_path = output_dir / "selected_cases.csv"
    pd.DataFrame(manifest_rows).to_csv(
        manifest_path,
        index=False,
        encoding="utf-8-sig",
    )
    summary = {
        "selection_rule": (
            "image-only prediction is wrong AND text-only prediction is wrong "
            "AND image+text prediction is correct"
        ),
        "selection_scope": "CTCH test split",
        "intervention": "same checkpoint and classifier; modality restricted at inference",
        "ranking": "multimodal true-class softmax descending; distinct true labels preferred",
        "seed": int(args.seed),
        "checkpoint": str(loaded.checkpoint),
        "checkpoint_sha256": loaded.checkpoint_sha256,
        "feature_archive": str(archive_path.resolve()),
        "scanned_multimodal_correct_candidates": scanned,
        "selected_count": len(rendered_cases),
        "cases": [serializable_case(case) for case in rendered_cases],
        "interpretation_note": (
            "Selected cases are illustrative examples and must not be used as "
            "an estimate of population-level modality benefit."
        ),
    }
    summary_path = output_dir / "selection_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Selected {len(rendered_cases)} cases after scanning {scanned} candidates.")
    for case in rendered_cases:
        print(case.output_figure)
    print(manifest_path)
    print(summary_path)


if __name__ == "__main__":
    main()
