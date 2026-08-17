"""Sinh forest plot thống kê dùng trong Chương 4 của báo cáo.

Mô-đun chuẩn hóa mọi chênh lệch theo quy ước XBone-Net trừ mô hình đối
chiếu. Thiết lập toàn bộ dữ liệu và ablation sử dụng kết quả bootstrap và
kiểm định hoán vị ghép cặp đã lưu. Thiết lập ít mẫu sử dụng khoảng Student-t
và kiểm định t ghép cặp trên ba hạt giống chung.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from scipy.stats import ttest_1samp


matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / "results"
DEFAULT_OUTPUT_DIR = ROOT / "docs" / "report" / "images" / "appendix"
SEEDS = (42, 123, 456)
METRICS = (
    "accuracy",
    "balanced_accuracy",
    "f1_macro",
    "auroc_macro",
    "auprc_macro",
)
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced Accuracy",
    "f1_macro": "Macro-F1",
    "auroc_macro": "Macro-AUROC",
    "auprc_macro": "Macro-AUPRC",
}
BASELINE_ORDER = (
    "lora_biomedclip",
    "lora_pubmedclip",
    "fft_biomedclip",
    "fft_pubmedclip",
    "fft_clip",
    "fft_medclip",
    "fft_resnet50",
    "fft_densenet",
)
BASELINE_LABELS = {
    "lora_biomedclip": "LoRA-BiomedCLIP",
    "lora_pubmedclip": "LoRA-PubMedCLIP",
    "fft_biomedclip": "BiomedCLIP",
    "fft_pubmedclip": "PubMedCLIP",
    "fft_clip": "CLIP",
    "fft_medclip": "MedCLIP",
    "fft_resnet50": "ResNet-50",
    "fft_densenet": "DenseNet-121",
}
FEW_SHOT_BASELINES = (
    ("lora_biomedclip", "LoRA-BiomedCLIP"),
    ("lora_pubmedclip", "LoRA-PubMedCLIP"),
)
PALETTE = {
    "xbone": "#1F77B4",
    "comparison": "#FF7F0E",
    "uncertain": "#7F7F7F",
}


def _format_p_value(value: float) -> str:
    """Định dạng p-value ngắn gọn trên biểu đồ.

    Parameters
    ----------
    value : float
        P-value thô cần trình bày.

    Returns
    -------
    str
        Chuỗi hiển thị của p-value.
    """
    if value < 0.001:
        return "p<0.001"
    return f"p={value:.3f}"


def _load_csvs(paths: Sequence[Path]) -> pd.DataFrame:
    """Nạp và ghép các tệp CSV thống kê.

    Parameters
    ----------
    paths : Sequence[Path]
        Các tệp CSV đầu vào.

    Returns
    -------
    pandas.DataFrame
        Bảng thống kê đã ghép.
    """
    frames = [pd.read_csv(path) for path in paths]
    return pd.concat(frames, ignore_index=True)


def build_full_shot_statistics() -> pd.DataFrame:
    """Chuẩn hóa thống kê full-shot theo chiều XBone-Net trừ baseline.

    Returns
    -------
    pandas.DataFrame
        Bảng chênh lệch, CI95% và p-value của năm độ đo.
    """
    files = {
        "BTXRD": (
            RESULTS_ROOT / "summary/classification/statistics_btxrd/paired_bootstrap_results.csv",
            RESULTS_ROOT / "summary/classification/statistics_btxrd_auc/paired_bootstrap_results.csv",
        ),
        "CTCH": (
            RESULTS_ROOT / "summary/classification/statistics_ctch/paired_bootstrap_results.csv",
            RESULTS_ROOT / "summary/classification/statistics_ctch_auc/paired_bootstrap_results.csv",
        ),
    }
    rows: list[dict[str, Any]] = []
    for dataset, paths in files.items():
        frame = _load_csvs(paths)
        for baseline_config in BASELINE_ORDER:
            for metric in METRICS:
                selected = frame[
                    (frame["baseline_config"].astype(str) == baseline_config)
                    & (frame["metric"].astype(str) == metric)
                ]
                if len(selected) != 1:
                    raise ValueError(
                        f"Thiếu thống kê {dataset}/{baseline_config}/{metric}."
                    )
                row = selected.iloc[0]
                rows.append(
                    {
                        "group": dataset,
                        "comparison_config": baseline_config,
                        "comparison": BASELINE_LABELS[baseline_config],
                        "metric": metric,
                        "delta_mean": -float(row["delta_mean"]),
                        "ci_low": -float(row["delta_ci_high"]),
                        "ci_high": -float(row["delta_ci_low"]),
                        "p_raw": float(row["p_raw"]),
                        "n_seeds": int(row["n_seeds"]),
                        "method": (
                            "bootstrap phân tầng theo bệnh nhân và seed; "
                            "kiểm định hoán vị ghép cặp"
                        ),
                        "delta_definition": "XBone-Net minus baseline",
                    }
                )
    return pd.DataFrame(rows)


def _read_metric(path: Path, metric: str) -> float:
    """Đọc một độ đo từ tệp metrics.json.

    Parameters
    ----------
    path : Path
        Đường dẫn đến tệp metrics.json.
    metric : str
        Tên độ đo cần đọc.

    Returns
    -------
    float
        Giá trị độ đo.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    return float(payload["metrics"][metric])


def build_few_shot_statistics() -> pd.DataFrame:
    """Tính thống kê ghép cặp theo seed cho thiết lập ít mẫu.

    Returns
    -------
    pandas.DataFrame
        Bảng chênh lệch, khoảng Student-t 95% và p-value thô.
    """
    rows: list[dict[str, Any]] = []
    for dataset in ("btxrd", "ctch"):
        dataset_label = dataset.upper()
        for shot in (1, 10, 20):
            reference_root = (
                RESULTS_ROOT
                / dataset
                / "few_shot"
                / f"{shot}_shot"
                / "ours_xbone_net"
            )
            for baseline_config, baseline_label in FEW_SHOT_BASELINES:
                baseline_root = (
                    RESULTS_ROOT
                    / dataset
                    / "few_shot"
                    / f"{shot}_shot"
                    / baseline_config
                )
                for metric in METRICS:
                    differences = np.asarray(
                        [
                            _read_metric(
                                reference_root / f"seed_{seed}" / "metrics.json",
                                metric,
                            )
                            - _read_metric(
                                baseline_root / f"seed_{seed}" / "metrics.json",
                                metric,
                            )
                            for seed in SEEDS
                        ],
                        dtype=np.float64,
                    )
                    mean = float(differences.mean())
                    standard_error = float(
                        differences.std(ddof=1) / math.sqrt(len(differences))
                    )
                    critical = float(
                        student_t.ppf(0.975, df=len(differences) - 1)
                    )
                    p_value = float(
                        ttest_1samp(differences, popmean=0.0).pvalue
                    )
                    rows.append(
                        {
                            "group": f"{dataset_label}, {shot} mẫu",
                            "dataset": dataset_label,
                            "shot": shot,
                            "comparison_config": baseline_config,
                            "comparison": baseline_label,
                            "metric": metric,
                            "delta_mean": mean,
                            "ci_low": mean - critical * standard_error,
                            "ci_high": mean + critical * standard_error,
                            "p_raw": p_value,
                            "n_seeds": len(differences),
                            "method": (
                                "khoảng Student-t và kiểm định t ghép cặp "
                                "trên seed"
                            ),
                            "delta_definition": "XBone-Net minus baseline",
                        }
                    )
    return pd.DataFrame(rows)


def build_ablation_statistics() -> pd.DataFrame:
    """Chuẩn hóa thống kê ablation theo chiều XBone-Net trừ biến thể.

    Returns
    -------
    pandas.DataFrame
        Bảng chênh lệch, CI95% và p-value của năm độ đo.
    """
    paths = (
        RESULTS_ROOT / "summary/ablation/statistics/paired_bootstrap_results.csv",
        RESULTS_ROOT / "summary/ablation/statistics_auc/paired_bootstrap_results.csv",
    )
    frame = _load_csvs(paths)
    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        rows.append(
            {
                "group": "CTCH",
                "comparison_config": str(row["variant_experiment"]),
                "comparison": str(row["variant"]),
                "metric": str(row["metric"]),
                "delta_mean": -float(row["delta_mean"]),
                "ci_low": -float(row["delta_ci_high"]),
                "ci_high": -float(row["delta_ci_low"]),
                "p_raw": float(row["p_raw"]),
                "n_seeds": int(row["n_seeds"]),
                "method": (
                    "bootstrap phân tầng theo bệnh nhân và seed; "
                    "kiểm định hoán vị ghép cặp"
                ),
                "delta_definition": "XBone-Net minus ablation variant",
            }
        )
    result = pd.DataFrame(rows)
    missing = set(METRICS) - set(result["metric"].astype(str))
    if missing:
        raise ValueError(f"Thiếu độ đo ablation: {sorted(missing)}")
    return result


def _row_labels(frame: pd.DataFrame, experiment: str) -> list[str]:
    """Tạo nhãn hàng gọn cho từng nhóm thực nghiệm.

    Parameters
    ----------
    frame : pandas.DataFrame
        Dữ liệu của một độ đo.
    experiment : str
        Loại thực nghiệm: full-shot, few-shot hoặc ablation.

    Returns
    -------
    list[str]
        Danh sách nhãn trục tung.
    """
    if experiment == "ablation":
        return frame["comparison"].astype(str).tolist()
    return [
        f"{group} — {comparison}"
        for group, comparison in zip(frame["group"], frame["comparison"])
    ]


def plot_forest(
    frame: pd.DataFrame,
    *,
    metric: str,
    experiment: str,
    output: Path,
    dpi: int,
) -> Path:
    """Vẽ forest plot không có title và nhãn trục hoành.

    Parameters
    ----------
    frame : pandas.DataFrame
        Bảng thống kê đã chuẩn hóa.
    metric : str
        Độ đo cần vẽ.
    experiment : str
        Nhóm thực nghiệm dùng để tạo nhãn hàng.
    output : Path
        Tệp ảnh đầu ra.
    dpi : int
        Độ phân giải ảnh.

    Returns
    -------
    Path
        Đường dẫn ảnh đã sinh.
    """
    subset = frame[frame["metric"].astype(str) == metric].copy()
    if subset.empty:
        raise ValueError(f"Không có dữ liệu cho {experiment}/{metric}.")

    means = subset["delta_mean"].astype(float).to_numpy()
    lowers = subset["ci_low"].astype(float).to_numpy()
    uppers = subset["ci_high"].astype(float).to_numpy()
    p_values = subset["p_raw"].astype(float).to_numpy()
    colors = [
        PALETTE["xbone"]
        if p_value < 0.05 and lower > 0.0
        else PALETTE["comparison"]
        if p_value < 0.05 and upper < 0.0
        else PALETTE["uncertain"]
        for lower, upper, p_value in zip(lowers, uppers, p_values)
    ]
    positions = np.arange(len(subset), dtype=float)
    height = max(4.0, 0.43 * len(subset) + 1.5)
    figure, axis = plt.subplots(figsize=(10.4, height))
    for position, mean, lower, upper, color in zip(
        positions, means, lowers, uppers, colors
    ):
        axis.errorbar(
            mean,
            position,
            xerr=np.asarray([[mean - lower], [upper - mean]]),
            fmt="o",
            color=color,
            ecolor=color,
            markersize=5.8,
            elinewidth=1.7,
            capsize=3.5,
            zorder=3,
        )
    axis.axvline(0.0, color="#222222", linestyle="--", linewidth=1.0)
    axis.set_yticks(positions, labels=_row_labels(subset, experiment))
    axis.invert_yaxis()
    axis.grid(axis="x", color="#D9D9D9", linewidth=0.75, alpha=0.85)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#666666")
    axis.spines["bottom"].set_color("#666666")
    axis.tick_params(axis="both", labelsize=8.5)

    finite_limits = np.concatenate((lowers, uppers, np.asarray([0.0])))
    span = max(float(np.nanmax(finite_limits) - np.nanmin(finite_limits)), 0.02)
    left = float(np.nanmin(finite_limits) - 0.10 * span)
    right = float(np.nanmax(finite_limits) + 0.32 * span)
    axis.set_xlim(left, right)
    for position, upper, p_value in zip(positions, uppers, p_values):
        axis.annotate(
            _format_p_value(p_value),
            xy=(upper, position),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            fontsize=7.7,
            color="#333333",
        )

    from matplotlib.lines import Line2D

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            color=PALETTE["xbone"],
            label="XBone-Net cao hơn: CI không chứa 0 và p<0,05",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            color=PALETTE["comparison"],
            label="Đối chiếu cao hơn: CI không chứa 0 và p<0,05",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            color=PALETTE["uncertain"],
            label="Chưa có bằng chứng khác biệt nhất quán",
        ),
    ]
    figure.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=1 if len(subset) <= 6 else 3,
        frameon=False,
        fontsize=8.0,
    )
    bottom = 0.20 if len(subset) <= 6 else 0.12
    figure.tight_layout(rect=(0.0, bottom, 1.0, 1.0))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return output


def generate_all(output_dir: Path, *, dpi: int = 300) -> list[Path]:
    """Sinh toàn bộ forest plot và bảng CSV thống kê đi kèm.

    Parameters
    ----------
    output_dir : Path
        Thư mục nhận ảnh và CSV.
    dpi : int, optional
        Độ phân giải ảnh.

    Returns
    -------
    list[Path]
        Danh sách tệp đã sinh.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    experiment_frames = {
        "full_shot": build_full_shot_statistics(),
        "few_shot": build_few_shot_statistics(),
        "ablation": build_ablation_statistics(),
    }
    outputs: list[Path] = []
    for experiment, frame in experiment_frames.items():
        csv_path = output_dir / f"{experiment}_paired_statistics.csv"
        frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
        outputs.append(csv_path)
        for metric in METRICS:
            image_path = output_dir / f"{experiment}_forest_{metric}.png"
            outputs.append(
                plot_forest(
                    frame,
                    metric=metric,
                    experiment=experiment,
                    output=image_path,
                    dpi=dpi,
                )
            )
    return outputs


def main() -> None:
    """Thực thi điểm vào dòng lệnh của mô-đun."""
    parser = argparse.ArgumentParser(
        description=(
            "Sinh forest plot CI95% và p-value cho full-shot, few-shot và ablation."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    outputs = generate_all(args.output_dir.resolve(), dpi=args.dpi)
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
