"""Generate the CTCH image-only versus multimodal comparison artifacts.

The table aggregates the aligned test predictions across seeds 42, 123, and
456.  The qualitative figure uses seed 42 and deterministically selects a
case that the image-only model gets wrong and the full model gets right.  To
avoid an example whose clinical input directly states the target pathology,
reports containing the corresponding pathology keyword (for example,
``gãy`` or ``viêm khớp``) are excluded; the remaining case with the highest
full-model probability for the true class is shown.
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "results"
DATA_ROOT = REPO_ROOT / "data" / "CTCH"
IMAGE_ONLY_ROOT = (
    RESULTS_ROOT / "ctch" / "ablation_study" / "modality" / "image_only"
)
FULL_MODEL_ROOT = RESULTS_ROOT / "ctch" / "proposed" / "ours_xbone_net"
DEFAULT_TABLE = (
    REPO_ROOT
    / "docs"
    / "report"
    / "tables"
    / "chapter4"
    / "table_modality_classification.tex"
)
DEFAULT_FIGURE = (
    REPO_ROOT
    / "docs"
    / "report"
    / "images"
    / "chapter4"
    / "modality_corrected_case.png"
)
SEEDS = (42, 123, 456)


def _load_archive(root: Path, seed: int) -> dict[str, np.ndarray]:
    source = root / f"seed_{seed}" / "analysis" / "features" / "ctch_test.npz"
    if not source.is_file():
        raise FileNotFoundError(f"Missing aligned prediction archive: {source}")
    with np.load(source, allow_pickle=False) as archive:
        required = {"probabilities", "labels", "predictions", "image_id"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"Missing keys in {source}: {sorted(missing)}")
        return {key: np.asarray(archive[key]) for key in required}


def _validate_alignment(
    image_only: dict[str, np.ndarray], full_model: dict[str, np.ndarray]
) -> None:
    for key in ("image_id", "labels"):
        if not np.array_equal(image_only[key], full_model[key]):
            raise ValueError(f"Image-only and full-model archives differ in {key}.")


def _metrics(archive: dict[str, np.ndarray]) -> dict[str, float]:
    labels = archive["labels"]
    predictions = archive["predictions"]
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "f1_macro": float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        ),
    }


def _format_mean_std(values: list[float]) -> str:
    array = np.asarray(values, dtype=np.float64)
    return f"{array.mean():.4f}\\pm{array.std(ddof=1):.4f}"


def generate_table(output: Path) -> Path:
    metric_order = (
        "accuracy",
        "balanced_accuracy",
        "f1_macro",
    )
    records: dict[str, dict[str, list[float]]] = {
        "image_only": {metric: [] for metric in metric_order},
        "full": {metric: [] for metric in metric_order},
    }
    for seed in SEEDS:
        image_only = _load_archive(IMAGE_ONLY_ROOT, seed)
        full_model = _load_archive(FULL_MODEL_ROOT, seed)
        _validate_alignment(image_only, full_model)
        for config, archive in (("image_only", image_only), ("full", full_model)):
            values = _metrics(archive)
            for metric in metric_order:
                records[config][metric].append(values[metric])

    rows: list[str] = []
    for config, display_name in (
        ("image_only", "XBone-Net (chỉ ảnh)"),
        ("full", "XBone-Net (đầy đủ)"),
    ):
        formatted = [
            f"${_format_mean_std(records[config][metric])}$"
            for metric in metric_order
        ]
        if config == "full":
            display_name = rf"\cellcolor{{gray!30}}{display_name}"
            formatted = [rf"\cellcolor{{gray!30}}$\mathbf{{{value[1:-1]}}}$" for value in formatted]
        rows.append(" & ".join([display_name, *formatted]) + " \\\\")

    latex = "\n".join(
        [
            r"\begin{table}[H]",
            r"\centering",
            r"\small",
            r"\resizebox{\textwidth}{!}{%",
            r"\begin{tabular}{|l|c|c|c|}",
            r"\hline",
            (
                r"\textbf{Cấu hình} & \textbf{Acc $\uparrow$} & "
                r"\textbf{BAcc $\uparrow$} & \textbf{Macro-F1 $\uparrow$} \\ \hline"
            ),
            rows[0] + r" \hline",
            rows[1] + r" \hline",
            r"\end{tabular}%",
            r"}",
            (
                r"\caption{So sánh kết quả phân lớp CTCH giữa XBone-Net chỉ ảnh "
                r"và XBone-Net đầy đủ phương thức trên 669 mẫu kiểm thử; số liệu "
                r"là trung bình $\pm$ độ lệch chuẩn qua ba hạt giống}"
            ),
            r"\label{tab:modality_classification}",
            r"\end{table}",
            "",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(latex, encoding="utf-8")
    return output


def _class_names() -> list[str]:
    return [
        line.strip()
        for line in (DATA_ROOT / "labels.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _clinical_report(image_id: str) -> tuple[Path, str]:
    path = DATA_ROOT / "reports" / "clinical_vi" / f"{Path(image_id).stem}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"Missing clinical report: {path}")
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition(":")
        if separator:
            values[key.strip()] = value.strip()
    history = values.get("Disease history", "")
    relevant_history = [
        part.strip(" .")
        for part in history.split(";")
        if any(
            keyword in part.casefold()
            for keyword in ("té", "đau", "chấn thương", "tai nạn")
        )
    ]
    if relevant_history:
        history = "; ".join(relevant_history[:2]) + "."
    report = "\n".join(
        (
            f"Lý do vào viện: {values.get('Reason for admission', '')}",
            f"Diễn tiến: {history}",
            f"Tiền sử: {values.get('Personal medical history', '')}",
        )
    )
    wrapped = "\n".join(textwrap.fill(line, width=36) for line in report.splitlines())
    return path, wrapped


def _choose_case(
    image_only: dict[str, np.ndarray],
    full_model: dict[str, np.ndarray],
    class_names: list[str],
) -> int:
    labels = image_only["labels"]
    corrected = np.flatnonzero(
        (image_only["predictions"] != labels)
        & (full_model["predictions"] == labels)
    )
    candidates: list[int] = []
    for index in corrected:
        image_id = str(image_only["image_id"][index])
        image_path = DATA_ROOT / "images" / image_id
        report_path = (
            DATA_ROOT / "reports" / "clinical_vi" / f"{Path(image_id).stem}.txt"
        )
        if not image_path.is_file() or not report_path.is_file():
            continue
        report = report_path.read_text(encoding="utf-8").casefold()
        true_label = class_names[int(labels[index])].casefold()
        pathology_keyword = next(
            (
                keyword
                for keyword in ("gãy", "viêm khớp", "cắt cụt", "bình thường")
                if keyword in true_label
            ),
            true_label,
        )
        if pathology_keyword not in report:
            candidates.append(int(index))
    if not candidates:
        raise ValueError("No corrected case satisfies the deterministic selection rule.")
    return min(
        candidates,
        key=lambda index: (
            -float(full_model["probabilities"][index, labels[index]]),
            str(full_model["image_id"][index]),
        ),
    )


def generate_figure(output: Path) -> tuple[Path, dict[str, Any]]:
    image_only = _load_archive(IMAGE_ONLY_ROOT, seed=42)
    full_model = _load_archive(FULL_MODEL_ROOT, seed=42)
    _validate_alignment(image_only, full_model)
    class_names = _class_names()
    index = _choose_case(image_only, full_model, class_names)

    image_id = str(image_only["image_id"][index])
    true_index = int(image_only["labels"][index])
    image_prediction = int(image_only["predictions"][index])
    full_prediction = int(full_model["predictions"][index])
    _, report = _clinical_report(image_id)

    image = Image.open(DATA_ROOT / "images" / image_id).convert("L")
    true_label = class_names[true_index]
    image_label = class_names[image_prediction]
    full_label = class_names[full_prediction]
    image_confidence = float(image_only["probabilities"][index, image_prediction])
    full_confidence = float(full_model["probabilities"][index, full_prediction])
    figure = plt.figure(figsize=(10.8, 6.2), constrained_layout=True)
    outer = figure.add_gridspec(1, 3, width_ratios=(1.0, 0.28, 1.0))
    left = outer[0, 0].subgridspec(4, 1, height_ratios=(0.10, 0.54, 0.23, 0.13))
    right = outer[0, 2].subgridspec(4, 1, height_ratios=(0.10, 0.54, 0.23, 0.13))

    left_title = figure.add_subplot(left[0, 0])
    left_image = figure.add_subplot(left[1, 0])
    left_info = figure.add_subplot(left[2, 0])
    left_prediction = figure.add_subplot(left[3, 0])
    center_label = figure.add_subplot(outer[0, 1])
    right_title = figure.add_subplot(right[0, 0])
    right_image = figure.add_subplot(right[1, 0])
    right_report = figure.add_subplot(right[2, 0])
    right_prediction = figure.add_subplot(right[3, 0])

    for axis in (
        left_title,
        left_info,
        left_prediction,
        center_label,
        right_title,
        right_report,
        right_prediction,
    ):
        axis.axis("off")

    left_title.text(
        0.5, 0.5, "(a) XBone-Net chỉ ảnh", ha="center", va="center",
        fontsize=12, fontweight="bold"
    )
    left_image.imshow(image, cmap="gray")
    left_image.axis("off")
    left_info.text(
        0.5,
        0.5,
        "Đầu vào: chỉ ảnh X-quang\nKhông sử dụng bệnh sử lâm sàng",
        ha="center",
        va="center",
        fontsize=9.5,
        color="#444444",
        bbox={
            "boxstyle": "round,pad=0.55",
            "facecolor": "#E0E0E0",
            "edgecolor": "#888888",
            "linewidth": 1.4,
        },
    )
    left_prediction.text(
        0.5,
        0.5,
        f"Dự đoán: {image_label}\nXác suất: {image_confidence:.3f} - SAI",
        ha="center",
        va="center",
        fontsize=10.5,
        color="#B71C1C",
        fontweight="bold",
        bbox={
            "boxstyle": "round,pad=0.65",
            "facecolor": "#FDECEC",
            "edgecolor": "#C62828",
            "linewidth": 2.2,
        },
    )
    center_label.text(
        0.5,
        0.52,
        f"NHÃN ĐÚNG\n{true_label}",
        ha="center",
        va="center",
        fontsize=10.5,
        fontweight="bold",
        linespacing=1.35,
        bbox={
            "boxstyle": "round,pad=0.65",
            "facecolor": "#D9D9D9",
            "edgecolor": "#6E6E6E",
            "linewidth": 1.8,
        },
    )

    right_title.text(
        0.5, 0.5, "(b) XBone-Net ảnh + bệnh sử", ha="center", va="center",
        fontsize=12, fontweight="bold"
    )
    right_image.imshow(image, cmap="gray")
    right_image.axis("off")
    right_report.text(
        0.5,
        0.5,
        report,
        ha="center",
        va="center",
        fontsize=8.8,
        linespacing=1.25,
        bbox={
            "boxstyle": "round,pad=0.55",
            "facecolor": "#E0E0E0",
            "edgecolor": "#888888",
            "linewidth": 1.4,
        },
    )
    right_prediction.text(
        0.5,
        0.5,
        f"Dự đoán: {full_label}\nXác suất: {full_confidence:.3f} - ĐÚNG",
        ha="center",
        va="center",
        fontsize=10.5,
        color="#1B5E20",
        fontweight="bold",
        bbox={
            "boxstyle": "round,pad=0.65",
            "facecolor": "#E8F5E9",
            "edgecolor": "#2E7D32",
            "linewidth": 2.2,
        },
    )

    figure.suptitle(
        "Ví dụ định tính về vai trò bổ sung của bệnh sử lâm sàng",
        fontsize=13,
        fontweight="bold",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    metadata = {
        "seed": 42,
        "image_id": image_id,
        "true_label": true_label,
        "image_only_prediction": image_label,
        "image_only_confidence": image_confidence,
        "full_prediction": full_label,
        "full_confidence": full_confidence,
        "corrected_cases": int(
            (
                (image_only["predictions"] != image_only["labels"])
                & (full_model["predictions"] == full_model["labels"])
            ).sum()
        ),
        "regressed_cases": int(
            (
                (image_only["predictions"] == image_only["labels"])
                & (full_model["predictions"] != full_model["labels"])
            ).sum()
        ),
    }
    return output, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table-output", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--figure-output", type=Path, default=DEFAULT_FIGURE)
    args = parser.parse_args()
    table = generate_table(args.table_output.resolve())
    figure, metadata = generate_figure(args.figure_output.resolve())
    print(f"Table saved to: {table}")
    print(f"Figure saved to: {figure}")
    for key, value in metadata.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
