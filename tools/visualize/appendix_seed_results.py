"""Tạo bảng hoặc hình trực quan bằng công cụ appendix seed results.

Notes
-----
Mô-đun này thuộc cơ sở mã nguồn nghiên cứu XBone-Net và giữ các quy ước dùng chung của dự án.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
SEEDS = (42, 123, 456)
CLASSIFICATION_METRICS = (
    ("accuracy", "Accuracy"),
    ("balanced_accuracy", "BAcc"),
    ("f1_macro", "Macro-F1"),
    ("auroc_macro", "Macro-AUROC"),
    ("auprc_macro", "Macro-AP"),
)
ABLATION_METRICS = (*CLASSIFICATION_METRICS, ("ece_15", "ECE"))
OOD_METRICS = (
    ("auroc_ood", "AUROC-OOD"),
    ("aupr_out", "AUPR-Out"),
    ("fpr_at_95tpr", "\\makecell{FPR@\\\\95\\%TPR}"),
)
FULL_SHOT_MODELS = (
    ("BiomedCLIP", "baselines/full_finetuned/fft_BiomedCLIP"),
    ("CLIP", "baselines/full_finetuned/fft_clip"),
    ("DenseNet-121", "baselines/full_finetuned/fft_densenet"),
    ("MedCLIP", "baselines/full_finetuned/fft_medclip"),
    ("PubMedCLIP", "baselines/full_finetuned/fft_pubmedclip"),
    ("ResNet-50", "baselines/full_finetuned/fft_resnet50"),
    ("LoRA-BiomedCLIP", "baselines/peft_finetuned/lora_BiomedCLIP"),
    ("LoRA-PubMedCLIP", "baselines/peft_finetuned/lora_pubmedclip"),
    ("XBone-Net", "proposed/ours_xbone_net"),
)
FEW_SHOT_MODELS = (
    ("LoRA-BiomedCLIP", "lora_BiomedCLIP"),
    ("LoRA-PubMedCLIP", "lora_pubmedclip"),
    ("XBone-Net", "ours_xbone_net"),
)
ABLATION_VARIANTS = (
    ("XBone-Net", "proposed/ours_xbone_net"),
    (
        "Không dùng ảnh độ phân giải cao",
        "ablation_study/architecture/preprocess/xbone_nohighres",
    ),
    (
        "Gộp trung bình",
        "ablation_study/architecture/preprocess/xbone_mean_pooling",
    ),
    ("Chỉ pha 2", "ablation_study/architecture/phase/phase2_only"),
    ("Nối đặc trưng", "ablation_study/architecture/fusion/concat"),
    ("Đầu tuyến tính", "ablation_study/architecture/classifier/linear"),
)
OOD_SCENARIOS = (
    ("OOD ngữ nghĩa", "semantic_ood"),
    ("BTXRD", "domain_ood_btxrd"),
)
OOD_METHODS = (
    ("Cosine theo tâm lớp", "cosine_centroids"),
    ("Mahalanobis theo tâm lớp", "mahalanobis_centroid"),
    ("kNN", "knn"),
    ("Entropy", "entropy"),
)
REPORT_PALETTE = {
    "blue": "#1F77B4",
    "orange": "#FF7F0E",
    "green": "#2CA02C",
    "gray": "#7F7F7F",
}


def resolve_path(path: str | Path) -> Path:
    """Xác định đường dẫn tuyệt đối của tài nguyên.

    Parameters
    ----------
    path : str | Path
        Đường dẫn tài nguyên được sử dụng.

    Returns
    -------
    Path
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    value = Path(path)
    return (value if value.is_absolute() else ROOT / value).resolve()


def load_json(path: Path) -> dict[str, Any]:
    """Tải json cho bước xử lý hiện tại.

    Parameters
    ----------
    path : Path
        Đường dẫn tài nguyên được sử dụng.

    Returns
    -------
    dict[str, Any]
        Kết quả được tạo bởi bước xử lý của hàm.

    Raises
    ------
    FileNotFoundError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    ValueError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Missing result file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def load_classification_metrics(path: Path, expected_seed: int) -> dict[str, float]:
    """Tải classification các độ đo cho bước xử lý hiện tại.

    Parameters
    ----------
    path : Path
        Đường dẫn tài nguyên được sử dụng.
    expected_seed : int
        Hạt giống phục vụ khả năng tái lập.

    Returns
    -------
    dict[str, float]
        Kết quả được tạo bởi bước xử lý của hàm.

    Raises
    ------
    KeyError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    ValueError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    """
    payload = load_json(path)
    recorded_seed = int(payload.get("seed", expected_seed))
    if recorded_seed != expected_seed:
        raise ValueError(
            f"Seed mismatch in {path}: expected {expected_seed}, found {recorded_seed}"
        )
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"Missing classification metrics in: {path}")
    required = {key for key, _ in ABLATION_METRICS}
    missing = sorted(required.difference(metrics))
    if missing:
        raise KeyError(f"Missing metrics in {path}: {', '.join(missing)}")
    return {key: float(metrics[key]) for key in required}


def load_ood_metrics(
    path: Path,
    expected_seed: int,
) -> tuple[dict[str, dict[str, float]], int, int, str]:
    """Tải ood các độ đo cho bước xử lý hiện tại.

    Parameters
    ----------
    path : Path
        Đường dẫn tài nguyên được sử dụng.
    expected_seed : int
        Hạt giống phục vụ khả năng tái lập.

    Returns
    -------
    tuple[dict[str, dict[str, float]], int, int, str]
        Kết quả được tạo bởi bước xử lý của hàm.

    Raises
    ------
    KeyError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    ValueError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    """
    payload = load_json(path)
    recorded_seed = int(payload.get("seed", expected_seed))
    if recorded_seed != expected_seed:
        raise ValueError(
            f"Seed mismatch in {path}: expected {expected_seed}, found {recorded_seed}"
        )
    results = payload.get("results")
    counts = payload.get("counts")
    if not isinstance(results, dict) or not isinstance(counts, dict):
        raise ValueError(f"Missing OOD results or counts in: {path}")
    methods: dict[str, dict[str, float]] = {}
    for _, method_key in OOD_METHODS:
        method_payload = results.get(method_key)
        if not isinstance(method_payload, dict):
            raise KeyError(f"Missing OOD method '{method_key}' in: {path}")
        methods[method_key] = {
            metric_key: float(method_payload[metric_key])
            for metric_key, _ in OOD_METRICS
        }
    return (
        methods,
        int(counts["id_test"]),
        int(counts["ood_test"]),
        str(payload.get("analysis_status", "")),
    )


def number(value: float) -> str:
    """Thực hiện bước number trong quy trình hiện tại.

    Parameters
    ----------
    value : float
        Giá trị ``value`` được sử dụng trong phép xử lý.

    Returns
    -------
    str
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    return f"{value:.4f}".replace(".", "{,}")


def latex_escape(text: str) -> str:
    """Thực hiện bước latex escape trong quy trình hiện tại.

    Parameters
    ----------
    text : str
        Văn bản hoặc biểu diễn văn bản đầu vào.

    Returns
    -------
    str
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in text)


def longtable(
    rows: Iterable[list[str]],
    *,
    column_spec: str,
    headers: list[str],
    caption: str,
    label: str | None,
    shaded_rows: set[int] | None = None,
    continued: bool = False,
    include_in_list: bool = True,
    placement: str = "H",
) -> str:
    """Thực hiện bước longtable trong quy trình hiện tại.

    Parameters
    ----------
    rows : Iterable[list[str]]
        Giá trị ``rows`` được sử dụng trong phép xử lý.
    column_spec : str
        Giá trị ``column_spec`` được sử dụng trong phép xử lý.
    headers : list[str]
        Giá trị ``headers`` được sử dụng trong phép xử lý.
    caption : str
        Giá trị ``caption`` được sử dụng trong phép xử lý.
    label : str | None
        Nhãn hoặc chỉ số lớp liên quan.
    shaded_rows : set[int] | None, optional
        Giá trị ``shaded_rows`` được sử dụng trong phép xử lý.
    continued : bool, optional
        Giá trị ``continued`` được sử dụng trong phép xử lý.
    include_in_list : bool, optional
        Giá trị ``include_in_list`` được sử dụng trong phép xử lý.
    placement : str, optional
        Giá trị ``placement`` được sử dụng trong phép xử lý.

    Returns
    -------
    str
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    shaded_rows = shaded_rows or set()
    lines = [rf"\begin{{table}}[{placement}]"]
    if continued:
        lines.append(r"\ContinuedFloat")
    lines.extend(
        [
        r"\centering",
        r"\footnotesize",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\toprule",
        " & ".join(rf"\textbf{{{header}}}" for header in headers) + r" \\",
        r"\midrule",
        ]
    )
    for index, row in enumerate(rows):
        prefix = r"\rowcolor{gray!12}" if index in shaded_rows else ""
        lines.append(prefix + " & ".join(row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}"])
    caption_command = r"\caption" if include_in_list else r"\caption[]"
    lines.append(rf"{caption_command}{{{caption}}}")
    if label:
        lines.append(rf"\label{{{label}}}")
    lines.extend([r"\end{table}", ""])
    return "\n".join(lines)


def multipage_longtable(
    rows: Iterable[list[str]],
    *,
    column_spec: str,
    headers: list[str],
    caption: str,
    label: str,
    shaded_rows: set[int] | None = None,
) -> str:
    """Thực hiện bước multipage longtable trong quy trình hiện tại.

    Parameters
    ----------
    rows : Iterable[list[str]]
        Giá trị ``rows`` được sử dụng trong phép xử lý.
    column_spec : str
        Giá trị ``column_spec`` được sử dụng trong phép xử lý.
    headers : list[str]
        Giá trị ``headers`` được sử dụng trong phép xử lý.
    caption : str
        Giá trị ``caption`` được sử dụng trong phép xử lý.
    label : str
        Nhãn hoặc chỉ số lớp liên quan.
    shaded_rows : set[int] | None, optional
        Giá trị ``shaded_rows`` được sử dụng trong phép xử lý.

    Returns
    -------
    str
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    shaded_rows = shaded_rows or set()
    column_count = len(headers)
    header_row = " & ".join(
        rf"\textbf{{{header}}}" for header in headers
    ) + r" \\"
    lines = [
        r"\begingroup",
        r"\footnotesize",
        r"\setlength{\LTpre}{0pt}",
        r"\setlength{\LTpost}{0pt}",
        r"\setlength{\LTcapwidth}{\textwidth}",
        r"\setlength{\tabcolsep}{5pt}",
        r"\renewcommand{\arraystretch}{1.08}",
        rf"\begin{{longtable}}{{{column_spec}}}",
        rf"\caption{{{caption}}}\label{{{label}}}\\",
        r"\toprule",
        header_row,
        r"\midrule",
        r"\endfirsthead",
        rf"\multicolumn{{{column_count}}}{{l}}{{\textit{{Bảng~\ref{{{label}}} (tiếp theo)}}}}\\",
        r"\toprule",
        header_row,
        r"\midrule",
        r"\endhead",
        r"\midrule",
        rf"\multicolumn{{{column_count}}}{{r}}{{\textit{{Còn tiếp ở trang sau}}}}\\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    for index, row in enumerate(rows):
        prefix = r"\rowcolor{gray!12}" if index in shaded_rows else ""
        lines.append(prefix + " & ".join(row) + r" \\")
    lines.extend([r"\end{longtable}", r"\endgroup", ""])
    return "\n".join(lines)


def collect_full_shot(results_root: Path) -> dict[str, list[dict[str, Any]]]:
    """Thu thập full shot cho bước xử lý hiện tại.

    Parameters
    ----------
    results_root : Path
        Đường dẫn tài nguyên được sử dụng.

    Returns
    -------
    dict[str, list[dict[str, Any]]]
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    collected: dict[str, list[dict[str, Any]]] = {}
    for dataset in ("btxrd", "ctch"):
        rows: list[dict[str, Any]] = []
        for model_label, experiment_path in FULL_SHOT_MODELS:
            for seed in SEEDS:
                path = (
                    results_root
                    / dataset
                    / experiment_path
                    / f"seed_{seed}"
                    / "metrics.json"
                )
                rows.append(
                    {
                        "model": model_label,
                        "seed": seed,
                        **load_classification_metrics(path, seed),
                    }
                )
        collected[dataset] = rows
    return collected


def collect_few_shot(results_root: Path) -> dict[str, list[dict[str, Any]]]:
    """Thu thập few shot cho bước xử lý hiện tại.

    Parameters
    ----------
    results_root : Path
        Đường dẫn tài nguyên được sử dụng.

    Returns
    -------
    dict[str, list[dict[str, Any]]]
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    collected: dict[str, list[dict[str, Any]]] = {}
    for dataset in ("btxrd", "ctch"):
        rows: list[dict[str, Any]] = []
        for shot in (1, 10, 20):
            for model_label, model_path in FEW_SHOT_MODELS:
                for seed in SEEDS:
                    path = (
                        results_root
                        / dataset
                        / "few_shot"
                        / f"{shot}_shot"
                        / model_path
                        / f"seed_{seed}"
                        / "metrics.json"
                    )
                    rows.append(
                        {
                            "shot": shot,
                            "model": model_label,
                            "seed": seed,
                            **load_classification_metrics(path, seed),
                        }
                    )
        collected[dataset] = rows
    return collected


def collect_ablation(results_root: Path) -> list[dict[str, Any]]:
    """Thu thập ablation cho bước xử lý hiện tại.

    Parameters
    ----------
    results_root : Path
        Đường dẫn tài nguyên được sử dụng.

    Returns
    -------
    list[dict[str, Any]]
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    rows: list[dict[str, Any]] = []
    for variant_label, experiment_path in ABLATION_VARIANTS:
        for seed in SEEDS:
            path = (
                results_root
                / "ctch"
                / experiment_path
                / f"seed_{seed}"
                / "metrics.json"
            )
            rows.append(
                {
                    "variant": variant_label,
                    "seed": seed,
                    **load_classification_metrics(path, seed),
                }
            )
    return rows


def collect_ood(results_root: Path) -> list[dict[str, Any]]:
    """Thu thập ood cho bước xử lý hiện tại.

    Parameters
    ----------
    results_root : Path
        Đường dẫn tài nguyên được sử dụng.

    Returns
    -------
    list[dict[str, Any]]
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    rows: list[dict[str, Any]] = []
    for scenario_label, scenario_key in OOD_SCENARIOS:
        for seed in SEEDS:
            path = (
                results_root
                / "ctch"
                / "proposed"
                / "ours_xbone_net"
                / f"seed_{seed}"
                / "analysis"
                / "ood"
                / scenario_key
                / "ood_metrics.json"
            )
            methods, id_count, ood_count, status = load_ood_metrics(path, seed)
            for method_label, method_key in OOD_METHODS:
                rows.append(
                    {
                        "scenario": scenario_label,
                        "scenario_key": scenario_key,
                        "method": method_label,
                        "method_key": method_key,
                        "seed": seed,
                        "id_count": id_count,
                        "ood_count": ood_count,
                        "status": status,
                        **methods[method_key],
                    }
                )
    return rows


def write_full_shot_tables(
    output: Path,
    collected: dict[str, list[dict[str, Any]]],
) -> Path:
    """Ghi full shot tables cho bước xử lý hiện tại.

    Parameters
    ----------
    output : Path
        Vị trí hoặc cấu trúc nhận kết quả.
    collected : dict[str, list[dict[str, Any]]]
        Giá trị ``collected`` được sử dụng trong phép xử lý.

    Returns
    -------
    Path
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    tables: list[str] = []
    headers = [
        "Mô hình",
        "Hạt giống",
        "Accuracy",
        "BAcc",
        "Macro-F1",
        "Macro-AUROC",
        "Macro-AP",
    ]
    for dataset, display in (("btxrd", "BTXRD"), ("ctch", "CTCH")):
        rows = collected[dataset]
        table_rows = [
            [
                latex_escape(str(row["model"])),
                str(row["seed"]),
                *(number(float(row[key])) for key, _ in CLASSIFICATION_METRICS),
            ]
            for row in rows
        ]
        shaded = {
            index
            for index, row in enumerate(rows)
            if row["model"] == "XBone-Net"
        }
        table_renderer = multipage_longtable if dataset == "btxrd" else longtable
        tables.append(
            table_renderer(
                table_rows,
                column_spec=(
                    r"@{}L{2.85cm}C{1.05cm}*{5}{C{1.55cm}}@{}"
                    if dataset == "btxrd"
                    else r"L{2.85cm}C{1.05cm}*{5}{C{1.55cm}}"
                ),
                headers=headers,
                caption=(
                    "Kết quả khi sử dụng toàn bộ dữ liệu huấn luyện của từng hạt giống trên "
                    f"{display}"
                ),
                label=f"tab:appendix-full-shot-seeds-{dataset}",
                shaded_rows=shaded,
            )
        )
    output.write_text("\n".join(tables), encoding="utf-8")
    return output


def write_few_shot_tables(
    output: Path,
    collected: dict[str, list[dict[str, Any]]],
) -> Path:
    """Ghi few shot tables cho bước xử lý hiện tại.

    Parameters
    ----------
    output : Path
        Vị trí hoặc cấu trúc nhận kết quả.
    collected : dict[str, list[dict[str, Any]]]
        Giá trị ``collected`` được sử dụng trong phép xử lý.

    Returns
    -------
    Path
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    tables: list[str] = []
    headers = [
        "Thiết lập",
        "Mô hình",
        "Hạt giống",
        "Accuracy",
        "BAcc",
        "Macro-F1",
        "Macro-AUROC",
        "Macro-AP",
    ]
    for dataset, display in (("btxrd", "BTXRD"), ("ctch", "CTCH")):
        rows = collected[dataset]
        for part_index, shot in enumerate((1, 10, 20)):
            tables.append(r"\clearpage")
            part_rows = [row for row in rows if int(row["shot"]) == shot]
            table_rows = [
                [
                    f"{row['shot']} mẫu",
                    latex_escape(str(row["model"])),
                    str(row["seed"]),
                    *(
                        number(float(row[key]))
                        for key, _ in CLASSIFICATION_METRICS
                    ),
                ]
                for row in part_rows
            ]
            shaded = {
                index
                for index, row in enumerate(part_rows)
                if row["model"] == "XBone-Net"
            }
            tables.append(
                longtable(
                    table_rows,
                    column_spec=(
                        r"C{1.0cm}L{2.35cm}C{1.0cm}*{5}{C{1.45cm}}"
                    ),
                    headers=headers,
                    caption=(
                        "Kết quả mẫu học hạn chế của từng hạt giống trên "
                        f"{display} ở thiết lập {shot} mẫu; hàng XBone-Net "
                        "được tô xám để dễ đối chiếu."
                    ),
                    label=(
                        f"tab:appendix-few-shot-seeds-{dataset}"
                        if part_index == 0
                        else None
                    ),
                    shaded_rows=shaded,
                    continued=part_index > 0,
                    include_in_list=part_index == 0,
                    placement="p",
                )
            )
    output.write_text("\n".join(tables), encoding="utf-8")
    return output


def write_ablation_table(output: Path, rows: list[dict[str, Any]]) -> Path:
    """Ghi ablation table cho bước xử lý hiện tại.

    Parameters
    ----------
    output : Path
        Vị trí hoặc cấu trúc nhận kết quả.
    rows : list[dict[str, Any]]
        Giá trị ``rows`` được sử dụng trong phép xử lý.

    Returns
    -------
    Path
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    variant_groups = (
        {"XBone-Net", "Không dùng ảnh độ phân giải cao", "Gộp trung bình"},
        {"Chỉ pha 2", "Nối đặc trưng", "Đầu tuyến tính"},
    )
    tables: list[str] = []
    for part_index, variants in enumerate(variant_groups):
        part_rows = [row for row in rows if row["variant"] in variants]
        table_rows = [
            [
                latex_escape(str(row["variant"])),
                str(row["seed"]),
                *(number(float(row[key])) for key, _ in ABLATION_METRICS),
            ]
            for row in part_rows
        ]
        shaded = {
            index
            for index, row in enumerate(part_rows)
            if row["variant"] == "XBone-Net"
        }
        tables.append(
            longtable(
                table_rows,
                column_spec=r"L{2.55cm}C{1.0cm}*{6}{C{1.45cm}}",
                headers=[
                    "Cấu hình",
                    "Hạt giống",
                    "Accuracy",
                    "BAcc",
                    "Macro-F1",
                    "Macro-AUROC",
                    "Macro-AP",
                    "ECE",
                ],
                caption=(
                    "Kết quả của từng hạt giống trong nghiên cứu loại bỏ "
                    f"từng thành phần trên CTCH (phần {part_index + 1}); "
                    "cấu hình đầy đủ được tô xám."
                ),
                label=(
                    "tab:appendix-ablation-seeds"
                    if part_index == 0
                    else None
                ),
                shaded_rows=shaded,
                continued=part_index > 0,
                include_in_list=part_index == 0,
            )
        )
    output.write_text("\n".join(tables), encoding="utf-8")
    return output


def write_ood_table(output: Path, rows: list[dict[str, Any]]) -> Path:
    """Ghi ood table cho bước xử lý hiện tại.

    Parameters
    ----------
    output : Path
        Vị trí hoặc cấu trúc nhận kết quả.
    rows : list[dict[str, Any]]
        Giá trị ``rows`` được sử dụng trong phép xử lý.

    Returns
    -------
    Path
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    tables: list[str] = []
    for part_index, (scenario, scenario_key) in enumerate(OOD_SCENARIOS):
        part_rows = [row for row in rows if row["scenario_key"] == scenario_key]
        table_rows = [
            [
                latex_escape(str(row["scenario"])),
                latex_escape(str(row["method"])),
                str(row["seed"]),
                str(row["id_count"]),
                str(row["ood_count"]),
                *(number(float(row[key])) for key, _ in OOD_METRICS),
            ]
            for row in part_rows
        ]
        shaded = {
            index
            for index, row in enumerate(part_rows)
            if row["method_key"] == "mahalanobis_centroid"
        }
        tables.append(
            longtable(
                table_rows,
                column_spec=(
                    r"L{1.70cm}L{2.50cm}C{1.0cm}C{1.0cm}C{1.0cm}"
                    r"*{3}{C{1.60cm}}"
                ),
                headers=[
                    "Kịch bản",
                    "Phương pháp",
                    "Hạt giống",
                    "$N_{ID}$",
                    "$N_{OOD}$",
                    "AUROC-OOD",
                    "AUPR-Out",
                    "\\makecell{FPR@\\\\95\\%TPR}",
                ],
                caption=(
                    "Kết quả OOD hậu xử lý của từng hạt giống trên kịch bản "
                    f"{scenario}; các hàng Mahalanobis theo tâm lớp được tô xám."
                ),
                label="tab:appendix-ood-seeds" if part_index == 0 else None,
                shaded_rows=shaded,
                continued=part_index > 0,
                include_in_list=part_index == 0,
            )
        )
    output.write_text("\n".join(tables), encoding="utf-8")
    return output


def pyplot() -> Any:
    """Thực hiện bước pyplot trong quy trình hiện tại.

    Returns
    -------
    Any
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
        }
    )
    return plt


def plot_full_shot_seed_f1(
    output: Path,
    collected: dict[str, list[dict[str, Any]]],
    dpi: int,
) -> Path:
    """Vẽ full shot seed f1 cho bước xử lý hiện tại.

    Parameters
    ----------
    output : Path
        Vị trí hoặc cấu trúc nhận kết quả.
    collected : dict[str, list[dict[str, Any]]]
        Giá trị ``collected`` được sử dụng trong phép xử lý.
    dpi : int
        Giá trị ``dpi`` được sử dụng trong phép xử lý.

    Returns
    -------
    Path
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    plt = pyplot()
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharex=True)
    models = (
        ("LoRA-BiomedCLIP", REPORT_PALETTE["orange"], "o"),
        ("LoRA-PubMedCLIP", REPORT_PALETTE["green"], "s"),
        ("XBone-Net", REPORT_PALETTE["blue"], "^"),
    )
    for axis, (dataset, display) in zip(
        axes,
        (("btxrd", "BTXRD"), ("ctch", "CTCH")),
    ):
        rows = collected[dataset]
        seed_positions = list(range(len(SEEDS)))
        for model, color, marker in models:
            selected = [row for row in rows if row["model"] == model]
            selected.sort(key=lambda row: int(row["seed"]))
            axis.plot(
                seed_positions,
                [row["f1_macro"] for row in selected],
                marker=marker,
                markersize=6,
                linewidth=2,
                color=color,
                label=model,
            )
        axis.set_title(f"Macro-F1 khi sử dụng toàn bộ dữ liệu trên {display}")
        axis.set_xlabel("Hạt giống ngẫu nhiên")
        axis.set_xticks(seed_positions, [str(seed) for seed in SEEDS])
        axis.set_ylabel("Macro-F1")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )
    figure.tight_layout(rect=(0, 0.10, 1, 1))
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return output


def plot_ood_seed_metrics(
    output: Path,
    rows: list[dict[str, Any]],
    dpi: int,
) -> Path:
    """Vẽ ood seed các độ đo cho bước xử lý hiện tại.

    Parameters
    ----------
    output : Path
        Vị trí hoặc cấu trúc nhận kết quả.
    rows : list[dict[str, Any]]
        Giá trị ``rows`` được sử dụng trong phép xử lý.
    dpi : int
        Giá trị ``dpi`` được sử dụng trong phép xử lý.

    Returns
    -------
    Path
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    plt = pyplot()
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.4), sharex=True)
    metric_specs = (
        ("auroc_ood", "AUROC-OOD"),
        ("aupr_out", "AUPR-Out"),
        ("fpr_at_95tpr", "FPR@95%TPR"),
    )
    scenario_specs = (
        ("semantic_ood", "OOD ngữ nghĩa", REPORT_PALETTE["blue"], "o"),
        ("domain_ood_btxrd", "BTXRD", REPORT_PALETTE["orange"], "s"),
    )
    for axis, (metric_key, metric_label) in zip(axes, metric_specs):
        seed_positions = list(range(len(SEEDS)))
        for scenario_key, scenario_label, color, marker in scenario_specs:
            selected = [
                row
                for row in rows
                if row["scenario_key"] == scenario_key
                and row["method_key"] == "mahalanobis_centroid"
            ]
            selected.sort(key=lambda row: int(row["seed"]))
            axis.plot(
                seed_positions,
                [row[metric_key] for row in selected],
                marker=marker,
                markersize=6,
                linewidth=2,
                color=color,
                label=scenario_label,
            )
        axis.set_title(metric_label)
        axis.set_xlabel("Hạt giống ngẫu nhiên")
        axis.set_xticks(seed_positions, [str(seed) for seed in SEEDS])
        axis.set_ylim(0.0, 1.02)
        axis.set_ylabel("Giá trị")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )
    figure.tight_layout(rect=(0, 0.11, 1, 1))
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return output


def build_outputs(results_root: Path, output_dir: Path, dpi: int) -> list[Path]:
    """Xây dựng outputs cho bước xử lý hiện tại.

    Parameters
    ----------
    results_root : Path
        Đường dẫn tài nguyên được sử dụng.
    output_dir : Path
        Đường dẫn tài nguyên được sử dụng.
    dpi : int
        Giá trị ``dpi`` được sử dụng trong phép xử lý.

    Returns
    -------
    list[Path]
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    full_shot = collect_full_shot(results_root)
    few_shot = collect_few_shot(results_root)
    ablation = collect_ablation(results_root)
    ood = collect_ood(results_root)
    outputs = [
        write_full_shot_tables(
            output_dir / "table_full_shot_per_seed.tex",
            full_shot,
        ),
        write_few_shot_tables(
            output_dir / "table_few_shot_per_seed.tex",
            few_shot,
        ),
        write_ablation_table(
            output_dir / "table_ablation_per_seed.tex",
            ablation,
        ),
        write_ood_table(output_dir / "table_ood_per_seed.tex", ood),
        plot_full_shot_seed_f1(
            output_dir / "full_shot_macro_f1_by_seed.png",
            full_shot,
            dpi,
        ),
        plot_ood_seed_metrics(
            output_dir / "ood_mahalanobis_by_seed.png",
            ood,
            dpi,
        ),
    ]
    manifest = {
        "seeds": list(SEEDS),
        "full_shot_rows": sum(len(rows) for rows in full_shot.values()),
        "few_shot_rows": sum(len(rows) for rows in few_shot.values()),
        "ablation_rows": len(ablation),
        "ood_rows": len(ood),
        "ood_counts": {
            scenario_key: {
                str(row["seed"]): {
                    "id_test": row["id_count"],
                    "ood_test": row["ood_count"],
                    "analysis_status": row["status"],
                }
                for row in ood
                if row["scenario_key"] == scenario_key
                and row["method_key"] == "mahalanobis_centroid"
            }
            for _, scenario_key in OOD_SCENARIOS
        },
        "outputs": [path.name for path in outputs],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    outputs.append(manifest_path)
    return outputs


def main() -> None:
    """Thực thi điểm vào chính của mô-đun."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/summary/appendix_seed_results"),
    )
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    outputs = build_outputs(
        resolve_path(args.results_root),
        resolve_path(args.output_dir),
        args.dpi,
    )
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
