"""Tạo bảng hoặc hình trực quan bằng công cụ results.

Notes
-----
Mô-đun này thuộc cơ sở mã nguồn nghiên cứu XBone-Net và giữ các quy ước dùng chung của dự án.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_ROOT = ROOT / "results"
DEFAULT_PROPOSED_EMBEDDINGS = (
    DEFAULT_RESULTS_ROOT / "ctch/proposed/ours_xbone_net/seed_42/embeddings.npz"
)

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


_LATEX_ESCAPE = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}
_NUMERIC_PREFIX = re.compile(
    r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
)
CONFIG_DISPLAY_NAMES = {
    "fft_resnet50": "ResNet-50",
    "fft_densenet": "DenseNet-121",
    "fft_clip": "CLIP",
    "fft_medclip": "MedCLIP",
    "fft_pubmedclip": "PubMedCLIP",
    "fft_biomedclip": "BiomedCLIP",
    "lora_pubmedclip": "LoRA-PubMedCLIP",
    "lora_biomedclip": "LoRA-BiomedCLIP",
    "clip_zeroshot": "CLIP",
    "medclip_zeroshot": "MedCLIP",
    "pubmedclip_zeroshot": "PubMedCLIP",
    "biomedclip_zeroshot": "BiomedCLIP",
    "ours_xbone_net": "XBone-Net",
    "ours_xbone_net_v2": "XBone-Net v2",
    "ours_xbone_net_v3": "XBone-Net v3",
    "proposed": "XBone-Net",
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


def _nested_json_value(payload: Any, record_path: str | None) -> Any:
    """Thực hiện bước nested json giá trị trong quy trình hiện tại.

    Parameters
    ----------
    payload : Any
        Giá trị ``payload`` được sử dụng trong phép xử lý.
    record_path : str | None
        Đường dẫn tài nguyên được sử dụng.

    Returns
    -------
    Any
        Kết quả được tạo bởi bước xử lý của hàm.

    Raises
    ------
    KeyError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    """
    if not record_path:
        return payload
    current = payload
    for key in record_path.split("."):
        if not isinstance(current, Mapping) or key not in current:
            raise KeyError(f"JSON record path '{record_path}' is missing key '{key}'.")
        current = current[key]
    return current


def load_frame(path: str | Path, json_record_path: str | None = None) -> pd.DataFrame:
    """Tải bảng dữ liệu cho bước xử lý hiện tại.

    Parameters
    ----------
    path : str | Path
        Đường dẫn tài nguyên được sử dụng.
    json_record_path : str | None, optional
        Đường dẫn tài nguyên được sử dụng.

    Returns
    -------
    pd.DataFrame
        Kết quả được tạo bởi bước xử lý của hàm.

    Raises
    ------
    FileNotFoundError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    ValueError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    """
    source = resolve_path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Result file does not exist: {source}")
    suffixes = "".join(source.suffixes).lower()
    if suffixes.endswith(".csv"):
        return pd.read_csv(source)
    if suffixes.endswith((".tsv", ".txt")):
        return pd.read_csv(source, sep="\t")
    if suffixes.endswith((".jsonl", ".ndjson")):
        return pd.read_json(source, lines=True)
    if suffixes.endswith(".json"):
        payload = json.loads(source.read_text(encoding="utf-8"))
        records = _nested_json_value(payload, json_record_path)
        if isinstance(records, list):
            return pd.json_normalize(records)
        if isinstance(records, Mapping):
            if records and all(isinstance(value, Mapping) for value in records.values()):
                rows = []
                for key, value in records.items():
                    row = {"key": key}
                    row.update(value)
                    rows.append(row)
                return pd.json_normalize(rows)
            return pd.json_normalize(records)
        raise ValueError("Selected JSON value must be an object or an array of objects.")
    raise ValueError(
        f"Unsupported input format '{source.suffix}'. Use CSV, TSV, JSON, or JSONL."
    )


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    """Thực hiện bước require columns trong quy trình hiện tại.

    Parameters
    ----------
    frame : pd.DataFrame
        Giá trị ``frame`` được sử dụng trong phép xử lý.
    columns : Sequence[str]
        Giá trị ``columns`` được sử dụng trong phép xử lý.

    Raises
    ------
    KeyError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    """
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(
            "Missing column(s): "
            + ", ".join(missing)
            + ". Available columns: "
            + ", ".join(map(str, frame.columns))
        )


def _parse_assignments(
    values: Sequence[str] | None,
    *,
    option: str,
) -> dict[str, str]:
    """Phân tích assignments cho bước xử lý hiện tại.

    Parameters
    ----------
    values : Sequence[str] | None
        Giá trị ``values`` được sử dụng trong phép xử lý.
    option : str
        Giá trị ``option`` được sử dụng trong phép xử lý.

    Returns
    -------
    dict[str, str]
        Kết quả được tạo bởi bước xử lý của hàm.

    Raises
    ------
    ValueError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    """
    result: dict[str, str] = {}
    for item in values or ():
        if "=" not in item:
            raise ValueError(f"{option} expects KEY=VALUE, received: {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"{option} contains an empty key.")
        result[key] = value
    return result


def filter_frame(
    frame: pd.DataFrame,
    *,
    where: Sequence[str] | None = None,
    where_in: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Lọc bảng dữ liệu cho bước xử lý hiện tại.

    Parameters
    ----------
    frame : pd.DataFrame
        Giá trị ``frame`` được sử dụng trong phép xử lý.
    where : Sequence[str] | None, optional
        Giá trị ``where`` được sử dụng trong phép xử lý.
    where_in : Sequence[str] | None, optional
        Giá trị ``where_in`` được sử dụng trong phép xử lý.

    Returns
    -------
    pd.DataFrame
        Kết quả được tạo bởi bước xử lý của hàm.

    Raises
    ------
    ValueError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    """
    filtered = frame.copy()
    for item in where or ():
        assignment = _parse_assignments([item], option="--where")
        column, expected = next(iter(assignment.items()))
        _require_columns(filtered, [column])
        filtered = filtered[filtered[column].astype(str) == expected]
    for item in where_in or ():
        assignment = _parse_assignments([item], option="--where-in")
        column, raw_values = next(iter(assignment.items()))
        _require_columns(filtered, [column])
        expected = {value.strip() for value in raw_values.split("|")}
        filtered = filtered[filtered[column].astype(str).isin(expected)]
    if filtered.empty:
        raise ValueError("No rows remain after applying the requested filters.")
    return filtered.reset_index(drop=True)


def _numeric_value(value: Any) -> float:
    """Thực hiện bước numeric giá trị trong quy trình hiện tại.

    Parameters
    ----------
    value : Any
        Giá trị ``value`` được sử dụng trong phép xử lý.

    Returns
    -------
    float
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    if value is None or pd.isna(value):
        return math.nan
    if isinstance(value, (int, float, np.number)):
        return float(value)
    match = _NUMERIC_PREFIX.match(str(value).replace(",", "."))
    return float(match.group(1)) if match else math.nan


def numeric_series(series: pd.Series) -> pd.Series:
    """Thực hiện bước numeric series trong quy trình hiện tại.

    Parameters
    ----------
    series : pd.Series
        Giá trị ``series`` được sử dụng trong phép xử lý.

    Returns
    -------
    pd.Series
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    return series.map(_numeric_value).astype(float)


def _latex_escape(value: Any) -> str:
    """Thực hiện bước latex escape trong quy trình hiện tại.

    Parameters
    ----------
    value : Any
        Giá trị ``value`` được sử dụng trong phép xử lý.

    Returns
    -------
    str
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    text = str(value)
    return "".join(_LATEX_ESCAPE.get(character, character) for character in text)


def display_value(column: str, value: Any) -> Any:
    """Thực hiện bước display giá trị trong quy trình hiện tại.

    Parameters
    ----------
    column : str
        Giá trị ``column`` được sử dụng trong phép xử lý.
    value : Any
        Giá trị ``value`` được sử dụng trong phép xử lý.

    Returns
    -------
    Any
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    if column == "config" and not pd.isna(value):
        return CONFIG_DISPLAY_NAMES.get(str(value).casefold(), value)
    if column == "category" and not pd.isna(value):
        s = str(value)
        s = re.sub(r"^Few-shot\s*/\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"(\d+)\s*shot", r"\1 mẫu", s, flags=re.IGNORECASE)
        return s
    return value


def _format_number(value: float, precision: int) -> str:
    """Định dạng number cho bước xử lý hiện tại.

    Parameters
    ----------
    value : float
        Giá trị ``value`` được sử dụng trong phép xử lý.
    precision : int
        Giá trị ``precision`` được sử dụng trong phép xử lý.

    Returns
    -------
    str
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    if not np.isfinite(value):
        return "--"
    return f"{value:.{precision}f}"


def _format_latex_numeric(
    mean: float,
    std: float | None,
    *,
    precision: int,
    bold: bool,
) -> str:
    """Định dạng latex numeric cho bước xử lý hiện tại.

    Parameters
    ----------
    mean : float
        Giá trị ``mean`` được sử dụng trong phép xử lý.
    std : float | None
        Giá trị ``std`` được sử dụng trong phép xử lý.
    precision : int
        Giá trị ``precision`` được sử dụng trong phép xử lý.
    bold : bool
        Giá trị ``bold`` được sử dụng trong phép xử lý.

    Returns
    -------
    str
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    if not np.isfinite(mean):
        return "--"
    body = _format_number(mean, precision)
    if std is not None and np.isfinite(std):
        body += rf"\pm{_format_number(std, precision)}"
    if bold:
        body = rf"\mathbf{{{body}}}"
    return f"${body}$"


def _maxima_masks(
    frame: pd.DataFrame,
    value_columns: Sequence[str],
    bold_within: Sequence[str] | None,
) -> dict[str, pd.Series]:
    """Thực hiện bước maxima masks trong quy trình hiện tại.

    Parameters
    ----------
    frame : pd.DataFrame
        Giá trị ``frame`` được sử dụng trong phép xử lý.
    value_columns : Sequence[str]
        Giá trị ``value_columns`` được sử dụng trong phép xử lý.
    bold_within : Sequence[str] | None
        Giá trị ``bold_within`` được sử dụng trong phép xử lý.

    Returns
    -------
    dict[str, pd.Series]
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    groups = list(bold_within or ())
    _require_columns(frame, [*value_columns, *groups])
    masks: dict[str, pd.Series] = {}
    for column in value_columns:
        numeric = numeric_series(frame[column])
        mask = pd.Series(False, index=frame.index)
        if groups:
            groupby_keys: str | list[str] = groups[0] if len(groups) == 1 else groups
            grouped_indices = frame.groupby(
                groupby_keys,
                sort=False,
                dropna=False,
                observed=False,
            ).groups.values()
        else:
            grouped_indices = [frame.index]
        for indices in grouped_indices:
            group_values = numeric.loc[indices]
            finite = group_values[np.isfinite(group_values)]
            if finite.empty:
                continue
            maximum = float(finite.max())
            mask.loc[indices] = np.isclose(
                group_values.to_numpy(dtype=float),
                maximum,
                rtol=1e-9,
                atol=1e-12,
                equal_nan=False,
            )
        masks[column] = mask
    return masks


def _minima_masks(
    frame: pd.DataFrame,
    value_columns: Sequence[str],
    bold_within: Sequence[str] | None,
) -> dict[str, pd.Series]:
    """Thực hiện bước minima masks trong quy trình hiện tại.

    Parameters
    ----------
    frame : pd.DataFrame
        Giá trị ``frame`` được sử dụng trong phép xử lý.
    value_columns : Sequence[str]
        Giá trị ``value_columns`` được sử dụng trong phép xử lý.
    bold_within : Sequence[str] | None
        Giá trị ``bold_within`` được sử dụng trong phép xử lý.

    Returns
    -------
    dict[str, pd.Series]
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    groups = list(bold_within or ())
    _require_columns(frame, [*value_columns, *groups])
    masks: dict[str, pd.Series] = {}
    for column in value_columns:
        numeric = numeric_series(frame[column])
        mask = pd.Series(False, index=frame.index)
        if groups:
            groupby_keys: str | list[str] = groups[0] if len(groups) == 1 else groups
            grouped_indices = frame.groupby(
                groupby_keys,
                sort=False,
                dropna=False,
                observed=False,
            ).groups.values()
        else:
            grouped_indices = [frame.index]
        for indices in grouped_indices:
            group_values = numeric.loc[indices]
            finite = group_values[np.isfinite(group_values)]
            if finite.empty:
                continue
            minimum = float(finite.min())
            mask.loc[indices] = np.isclose(
                group_values.to_numpy(dtype=float),
                minimum,
                rtol=1e-9,
                atol=1e-12,
                equal_nan=False,
            )
        masks[column] = mask
    return masks


def generate_latex_table(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    value_columns: Sequence[str] | None = None,
    std_columns: Mapping[str, str] | None = None,
    column_labels: Mapping[str, str] | None = None,
    bold_within: Sequence[str] | None = None,
    precision: int = 4,
    caption: str | None = None,
    label: str | None = None,
    position: str = "htbp",
    table_environment: bool = True,
    font_size: str | None = r"\small",
    multirow_columns: Sequence[str] | None = None,
    minimize_columns: Sequence[str] | None = None,
    resize_to_textwidth: bool = False,
    bold_best: bool = True,
    column_widths: Mapping[str, str] | None = None,
) -> str:
    """Sinh latex table cho bước xử lý hiện tại.

    Parameters
    ----------
    frame : pd.DataFrame
        Giá trị ``frame`` được sử dụng trong phép xử lý.
    columns : Sequence[str]
        Giá trị ``columns`` được sử dụng trong phép xử lý.
    value_columns : Sequence[str] | None, optional
        Giá trị ``value_columns`` được sử dụng trong phép xử lý.
    std_columns : Mapping[str, str] | None, optional
        Giá trị ``std_columns`` được sử dụng trong phép xử lý.
    column_labels : Mapping[str, str] | None, optional
        Giá trị ``column_labels`` được sử dụng trong phép xử lý.
    bold_within : Sequence[str] | None, optional
        Giá trị ``bold_within`` được sử dụng trong phép xử lý.
    precision : int, optional
        Giá trị ``precision`` được sử dụng trong phép xử lý.
    caption : str | None, optional
        Giá trị ``caption`` được sử dụng trong phép xử lý.
    label : str | None, optional
        Nhãn hoặc chỉ số lớp liên quan.
    position : str, optional
        Giá trị ``position`` được sử dụng trong phép xử lý.
    table_environment : bool, optional
        Giá trị ``table_environment`` được sử dụng trong phép xử lý.
    font_size : str | None, optional
        Số lượng, kích thước hoặc tỷ lệ được sử dụng.
    multirow_columns : Sequence[str] | None, optional
        Giá trị ``multirow_columns`` được sử dụng trong phép xử lý.
    minimize_columns : Sequence[str] | None, optional
        Giá trị ``minimize_columns`` được sử dụng trong phép xử lý.
    resize_to_textwidth : bool, optional
        Giá trị ``resize_to_textwidth`` được sử dụng trong phép xử lý.
    bold_best : bool, optional
        Giá trị ``bold_best`` được sử dụng trong phép xử lý.
    column_widths : Mapping[str, str] | None, optional
        Giá trị ``column_widths`` được sử dụng trong phép xử lý.

    Returns
    -------
    str
        Kết quả được tạo bởi bước xử lý của hàm.

    Raises
    ------
    ValueError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    """
    frame = frame.copy()
    if "category" in frame.columns:
        frame["_cat_rank"] = frame["category"].map(_shot_sort_key)
        if "dataset" in frame.columns:
            frame = frame.sort_values(by=["dataset", "_cat_rank"], kind="stable").drop(columns=["_cat_rank"]).reset_index(drop=True)
        else:
            frame = frame.sort_values(by=["_cat_rank"], kind="stable").drop(columns=["_cat_rank"]).reset_index(drop=True)

    selected = list(columns)
    if not selected:
        raise ValueError("At least one output column is required.")
    _require_columns(frame, selected)
    labels = dict(column_labels or {})
    std_map = dict(std_columns or {})
    _require_columns(frame, std_map.values())
    grouped_rows = (
        ["dataset"]
        if multirow_columns is None and "dataset" in selected
        else list(multirow_columns or ())
    )
    _require_columns(frame, grouped_rows)
    if grouped_rows and selected[: len(grouped_rows)] != grouped_rows:
        raise ValueError("Multirow columns must be the leading --columns entries.")

    if value_columns is None:
        grouping_columns = set(bold_within or ())
        values = [
            column
            for column in selected
            if column not in grouping_columns
            and np.isfinite(numeric_series(frame[column])).any()
        ]
    else:
        values = list(value_columns)
        _require_columns(frame, values)
        unknown = [column for column in values if column not in selected]
        if unknown:
            raise ValueError(
                "Value columns must also appear in --columns: " + ", ".join(unknown)
            )

    minimize = set(minimize_columns or ())
    unknown_minimize = sorted(minimize.difference(values))
    if unknown_minimize:
        raise ValueError(
            "Minimize columns must also be value columns: "
            + ", ".join(unknown_minimize)
        )
    maxima = _maxima_masks(frame, values, bold_within)
    minima = _minima_masks(frame, values, bold_within)
    if not bold_best:
        maxima = {
            column: pd.Series(False, index=frame.index) for column in values
        }
        minima = {
            column: pd.Series(False, index=frame.index) for column in values
        }
    widths = dict(column_widths or {})
    unknown_widths = sorted(set(widths).difference(selected))
    if unknown_widths:
        raise ValueError(
            "Column widths must refer to selected columns: "
            + ", ".join(unknown_widths)
        )
    alignments = ["C" if column in values else "L" for column in selected]
    column_specs = [
        f"{alignment}{{{widths[column]}}}" if column in widths else alignment.lower()
        for column, alignment in zip(selected, alignments)
    ]
    tabular_spec = "|" + "|".join(column_specs) + "|"
    lines: list[str] = []
    if table_environment:
        lines.extend([rf"\begin{{table}}[{position}]", r"\centering"])
        if font_size:
            lines.append(font_size)
    if table_environment and resize_to_textwidth:
        lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.extend([rf"\begin{{tabular}}{{{tabular_spec}}}", r"\hline"])
    header = " & ".join(
        (
            rf"\multicolumn{{1}}{{"
            + ("|" if index == 0 else "")
            + (f"C{{{widths[column]}}}" if column in widths else "c")
            + rf"|}}{{\textbf{{{labels.get(column, _latex_escape(column))}}}}}"
        )
        for index, column in enumerate(selected)
    )
    lines.append(header + r" \\ \hline")

    group_spans: dict[str, dict[int, int]] = {}
    for group_position, column in enumerate(grouped_rows):
        parent_columns = grouped_rows[:group_position]
        starts: dict[int, int] = {}
        start = 0
        while start < len(frame):
            end = start + 1
            while end < len(frame):
                same_value = frame.iloc[end][column] == frame.iloc[start][column]
                same_parent = all(
                    frame.iloc[end][parent] == frame.iloc[start][parent]
                    for parent in parent_columns
                )
                if not same_value or not same_parent:
                    break
                end += 1
            starts[start] = end - start
            start = end
        group_spans[column] = starts

    for row_position, (index, row) in enumerate(frame.iterrows()):
        cells = []
        is_proposed = any(
            "xbone" in str(row[c]).casefold() or "proposed" in str(row[c]).casefold() or "ours" in str(row[c]).casefold()
            for c in selected
            if c not in values
        )
        cell_bg = r"\cellcolor{gray!12}" if is_proposed else ""
        for column in selected:
            if column in grouped_rows:
                span = group_spans[column].get(row_position)
                if span is None:
                    cells.append("")
                else:
                    value = display_value(column, row[column])
                    body = "--" if pd.isna(value) else _latex_escape(value)
                    cells.append(
                        body
                        if span == 1
                        else rf"\multirow{{{span}}}{{{'=' if column in widths else '*'}}}{{{body}}}"
                    )
                continue
            if column in values:
                mean = _numeric_value(row[column])
                std_column = std_map.get(column)
                std = _numeric_value(row[std_column]) if std_column else None
                formatted = _format_latex_numeric(
                    mean,
                    std,
                    precision=precision,
                    bold=bool(
                        (minima if column in minimize else maxima)[column].loc[index]
                    ),
                )
                cells.append(f"{cell_bg}{formatted}")
            else:
                value = display_value(column, row[column])
                body = "--" if pd.isna(value) else _latex_escape(value)
                cells.append(f"{cell_bg}{body}")
        row_end = r" \\ \hline"
        if grouped_rows and row_position + 1 < len(frame):
            next_row = frame.iloc[row_position + 1]
            first_changing_group_index = None
            for g_idx, column in enumerate(grouped_rows):
                if row[column] != next_row[column]:
                    first_changing_group_index = g_idx
                    break
            if first_changing_group_index is None:
                first_column = len(grouped_rows) + 1
                row_end = rf" \\ \cline{{{first_column}-{len(selected)}}}"
            elif first_changing_group_index > 0:
                first_column = first_changing_group_index + 1
                row_end = rf" \\ \cline{{{first_column}-{len(selected)}}}"
            else:
                row_end = r" \\ \hline"
        lines.append(" & ".join(cells) + row_end)


    lines.append(r"\end{tabular}")
    if table_environment:
        if resize_to_textwidth:
            lines.append("}")
        if caption:
            lines.append(rf"\caption{{{caption}}}")
        if label:
            lines.append(rf"\label{{{label}}}")
        lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def _prepare_plot_output(output: str | Path) -> Path:
    """Chuẩn bị plot đầu ra cho bước xử lý hiện tại.

    Parameters
    ----------
    output : str | Path
        Vị trí hoặc cấu trúc nhận kết quả.

    Returns
    -------
    Path
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    destination = resolve_path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination


def _load_pyplot():
    """Tải pyplot cho bước xử lý hiện tại.

    Returns
    -------
    object
        Kết quả được tạo bởi bước xử lý của hàm.

    Raises
    ------
    RuntimeError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "Plot generation requires matplotlib. Install project requirements "
            "with `pip install -r requirements.txt`."
        ) from error
    return plt


def _ordered_unique(series: pd.Series) -> list[Any]:
    """Thực hiện bước ordered unique trong quy trình hiện tại.

    Parameters
    ----------
    series : pd.Series
        Giá trị ``series`` được sử dụng trong phép xử lý.

    Returns
    -------
    list[Any]
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    return list(dict.fromkeys(series.dropna().tolist()))


def _clean_x_tick_label(val: Any) -> str:
    """Thực hiện bước clean x tick nhãn trong quy trình hiện tại.

    Parameters
    ----------
    val : Any
        Giá trị ``val`` được sử dụng trong phép xử lý.

    Returns
    -------
    str
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    s = str(val)
    s = re.sub(r"^Few-shot\s*/\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"(\d+)\s*shot", r"\1 mẫu", s, flags=re.IGNORECASE)
    return s


def _shot_sort_key(val: Any) -> tuple[int, int | float, str]:
    """Thực hiện bước shot sort key trong quy trình hiện tại.

    Parameters
    ----------
    val : Any
        Giá trị ``val`` được sử dụng trong phép xử lý.

    Returns
    -------
    tuple[int, int | float, str]
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    s = str(val)
    match = re.search(r"(\d+)\s*(?:shot|mẫu)", s, flags=re.IGNORECASE)
    if match:
        return (0, int(match.group(1)), s)
    return (1, 0, s)


def plot_bar_chart(
    data: pd.DataFrame,
    *,
    x: str,
    y: str,
    output: str | Path,
    hue: str | None = None,
    error: str | None = None,
    highlight: str | None = None,
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    annotate: bool = False,
    precision: int = 3,
    dpi: int = 300,
) -> Path:
    """Vẽ bar chart cho bước xử lý hiện tại.

    Parameters
    ----------
    data : pd.DataFrame
        Dữ liệu đầu vào của bước xử lý.
    x : str
        Giá trị ``x`` được sử dụng trong phép xử lý.
    y : str
        Giá trị ``y`` được sử dụng trong phép xử lý.
    output : str | Path
        Vị trí hoặc cấu trúc nhận kết quả.
    hue : str | None, optional
        Giá trị ``hue`` được sử dụng trong phép xử lý.
    error : str | None, optional
        Giá trị ``error`` được sử dụng trong phép xử lý.
    highlight : str | None, optional
        Giá trị ``highlight`` được sử dụng trong phép xử lý.
    title : str | None, optional
        Giá trị ``title`` được sử dụng trong phép xử lý.
    x_label : str | None, optional
        Nhãn hoặc chỉ số lớp liên quan.
    y_label : str | None, optional
        Nhãn hoặc chỉ số lớp liên quan.
    annotate : bool, optional
        Giá trị ``annotate`` được sử dụng trong phép xử lý.
    precision : int, optional
        Giá trị ``precision`` được sử dụng trong phép xử lý.
    dpi : int, optional
        Giá trị ``dpi`` được sử dụng trong phép xử lý.

    Returns
    -------
    Path
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    plt = _load_pyplot()
    plot_data = data.copy()
    _require_columns(
        plot_data,
        [x, y] + ([hue] if hue else []) + ([error] if error else []),
    )
    plot_data[y] = numeric_series(plot_data[y])
    if error:
        plot_data[error] = numeric_series(plot_data[error])

    raw_x_values = _ordered_unique(plot_data[x])
    x_values = sorted(raw_x_values, key=_shot_sort_key)
    series_values = _ordered_unique(plot_data[hue]) if hue else [None]
    width = min(0.75 / max(1, len(series_values)), 0.25)
    positions = np.arange(len(x_values), dtype=float)
    fig_width = max(6.5, 1.2 * len(x_values) + 2.5)
    fig, axis = plt.subplots(figsize=(fig_width, 4.8))

    # Tạo thành phần trực quan cho kết quả phân tích.
    tab10_palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ]
    max_height_val = 0.0

    # Bước hỗ trợ để vẽ bar chart cho bước xử lý hiện tại.
    color_map: dict[str, str] = {}
    used_indices: set[int] = set()

    if highlight is not None and any(str(sv) == highlight for sv in series_values):
        color_map[highlight] = "#1f77b4"  # Bước hỗ trợ để vẽ bar chart cho bước xử lý hiện tại.
        used_indices.add(0)

    color_idx = 0
    for sv in series_values:
        sv_str = str(sv)
        if sv_str in color_map:
            continue
        while color_idx in used_indices:
            color_idx += 1
        color_map[sv_str] = tab10_palette[color_idx % len(tab10_palette)]
        used_indices.add(color_idx)
        color_idx += 1

    for series_index, series_value in enumerate(series_values):
        subset = (
            plot_data[plot_data[hue] == series_value] if hue else plot_data
        )
        by_x = subset.drop_duplicates(subset=[x], keep="first").set_index(x)
        heights = np.asarray(
            [
                _numeric_value(by_x.loc[value, y]) if value in by_x.index else math.nan
                for value in x_values
            ]
        )
        errors = None
        if error:
            errors = np.asarray(
                [
                    _numeric_value(by_x.loc[value, error])
                    if value in by_x.index
                    else math.nan
                    for value in x_values
                ]
            )
            top_vals = np.nan_to_num(heights) + np.nan_to_num(errors)
        else:
            top_vals = np.nan_to_num(heights)
        if len(top_vals) > 0:
            max_height_val = max(max_height_val, float(np.max(top_vals)))

        offset = (series_index - (len(series_values) - 1) / 2.0) * width
        is_highlight = highlight is not None and str(series_value) == highlight
        color = color_map.get(str(series_value), tab10_palette[series_index % len(tab10_palette)])
        edgecolor = "#0b436c" if is_highlight else "#333333"
        linewidth = 1.2 if is_highlight else 0.8

        bars = axis.bar(
            positions + offset,
            heights,
            width=width,
            yerr=errors,
            capsize=3.0 if error else 0,
            error_kw={"elinewidth": 1.0, "capthick": 1.0, "ecolor": "#333333"},
            color=color,
            edgecolor=edgecolor,
            linewidth=linewidth,
            label=str(display_value(hue, series_value)) if hue else y,
            zorder=3,
        )
        if annotate:
            axis.bar_label(
                bars,
                labels=[
                    "" if not np.isfinite(value) else f"{value:.{precision}f}"
                    for value in heights
                ],
                fontsize=8.5,
                fontweight="bold" if is_highlight else "normal",
                padding=3,
            )

    if highlight is not None and hue is None:
        for bar, value in zip(axis.patches, x_values):
            if str(value) == highlight:
                bar.set_facecolor("#1f77b4")

    cleaned_x_labels = [_clean_x_tick_label(display_value(x, value)) for value in x_values]
    axis.set_xticks(positions)
    axis.set_xticklabels(
        cleaned_x_labels,
        rotation=0,
        ha="center",
        fontsize=10,
    )
    axis.set_xlabel(x_label or x, fontsize=10.5, labelpad=8)
    axis.set_ylabel(y_label or y, fontsize=10.5)
    if title:
        axis.set_title(title, fontsize=11.5, pad=12)

    if max_height_val > 0:
        axis.set_ylim(0.0, min(1.0, max_height_val * 1.18))

    if hue:
        axis.legend(
            title=None,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.14),
            ncol=min(len(series_values), 4),
            frameon=False,
            fontsize=9.5,
        )

    axis.grid(axis="y", color="#E0E0E0", linestyle="--", linewidth=0.7, alpha=0.8, zorder=0)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    fig.tight_layout()
    destination = _prepare_plot_output(output)
    fig.savefig(destination, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return destination


def plot_heatmap(
    frame: pd.DataFrame,
    *,
    rows: str,
    columns: str,
    value: str,
    output: str | Path,
    title: str | None = None,
    cmap: str = "Blues",
    x_label: str | None = None,
    y_label: str | None = None,
    colorbar_label: str | None = None,
    precision: int = 3,
    dpi: int = 300,
) -> Path:
    """Vẽ heatmap cho bước xử lý hiện tại.

    Parameters
    ----------
    frame : pd.DataFrame
        Giá trị ``frame`` được sử dụng trong phép xử lý.
    rows : str
        Giá trị ``rows`` được sử dụng trong phép xử lý.
    columns : str
        Giá trị ``columns`` được sử dụng trong phép xử lý.
    value : str
        Giá trị ``value`` được sử dụng trong phép xử lý.
    output : str | Path
        Vị trí hoặc cấu trúc nhận kết quả.
    title : str | None, optional
        Giá trị ``title`` được sử dụng trong phép xử lý.
    cmap : str, optional
        Giá trị ``cmap`` được sử dụng trong phép xử lý.
    x_label : str | None, optional
        Nhãn hoặc chỉ số lớp liên quan.
    y_label : str | None, optional
        Nhãn hoặc chỉ số lớp liên quan.
    colorbar_label : str | None, optional
        Nhãn hoặc chỉ số lớp liên quan.
    precision : int, optional
        Giá trị ``precision`` được sử dụng trong phép xử lý.
    dpi : int, optional
        Giá trị ``dpi`` được sử dụng trong phép xử lý.

    Returns
    -------
    Path
        Kết quả được tạo bởi bước xử lý của hàm.

    Raises
    ------
    ValueError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    """
    plt = _load_pyplot()
    _require_columns(frame, [rows, columns, value])
    data = frame.copy()
    data[value] = numeric_series(data[value])
    row_order = _ordered_unique(data[rows])
    column_order = _ordered_unique(data[columns])
    matrix = data.pivot_table(
        index=rows,
        columns=columns,
        values=value,
        aggfunc="mean",
        sort=False,
    ).reindex(index=row_order, columns=column_order)
    if matrix.empty or not np.isfinite(matrix.to_numpy(dtype=float)).any():
        raise ValueError("Heatmap contains no finite numeric values.")

    values = matrix.to_numpy(dtype=float)
    finite_values = values[np.isfinite(values)]
    vmin, vmax = float(finite_values.min()), float(finite_values.max())
    if math.isclose(vmin, vmax):
        vmax = vmin + 1e-12
    fig_width = max(5.6, 0.75 * len(matrix.columns) + 2.5)
    fig_height = max(4.0, 0.45 * len(matrix.index) + 2.0)
    fig, axis = plt.subplots(figsize=(fig_width, fig_height))
    image = axis.imshow(values, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    axis.set_xticks(np.arange(len(matrix.columns)))
    axis.set_xticklabels(
        [str(display_value(columns, value)) for value in matrix.columns],
        rotation=30,
        ha="right",
    )
    axis.set_yticks(np.arange(len(matrix.index)))
    axis.set_yticklabels(
        [str(display_value(rows, value)) for value in matrix.index]
    )
    axis.set_xlabel(x_label if x_label is not None else columns)
    axis.set_ylabel(y_label if y_label is not None else rows)
    if title:
        axis.set_title(title)

    midpoint = (vmin + vmax) / 2.0
    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            current = values[row_index, column_index]
            if not np.isfinite(current):
                text, color = "--", "#333333"
            else:
                text = f"{current:.{precision}f}"
                color = "white" if current > midpoint else "#202020"
            axis.text(
                column_index,
                row_index,
                text,
                ha="center",
                va="center",
                color=color,
                fontsize=9,
            )
    colorbar = fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label(colorbar_label if colorbar_label is not None else value)
    fig.tight_layout()
    destination = _prepare_plot_output(output)
    fig.savefig(destination, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return destination


def plot_line_chart(
    data: pd.DataFrame,
    *,
    x: str,
    y: str,
    output: str | Path,
    hue: str | None = None,
    error: str | None = None,
    highlight: str | None = None,
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    annotate: bool = False,
    precision: int = 3,
    dpi: int = 300,
) -> Path:
    """Vẽ line chart cho bước xử lý hiện tại.

    Parameters
    ----------
    data : pd.DataFrame
        Dữ liệu đầu vào của bước xử lý.
    x : str
        Giá trị ``x`` được sử dụng trong phép xử lý.
    y : str
        Giá trị ``y`` được sử dụng trong phép xử lý.
    output : str | Path
        Vị trí hoặc cấu trúc nhận kết quả.
    hue : str | None, optional
        Giá trị ``hue`` được sử dụng trong phép xử lý.
    error : str | None, optional
        Giá trị ``error`` được sử dụng trong phép xử lý.
    highlight : str | None, optional
        Giá trị ``highlight`` được sử dụng trong phép xử lý.
    title : str | None, optional
        Giá trị ``title`` được sử dụng trong phép xử lý.
    x_label : str | None, optional
        Nhãn hoặc chỉ số lớp liên quan.
    y_label : str | None, optional
        Nhãn hoặc chỉ số lớp liên quan.
    annotate : bool, optional
        Giá trị ``annotate`` được sử dụng trong phép xử lý.
    precision : int, optional
        Giá trị ``precision`` được sử dụng trong phép xử lý.
    dpi : int, optional
        Giá trị ``dpi`` được sử dụng trong phép xử lý.

    Returns
    -------
    Path
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    plt = _load_pyplot()
    plot_data = data.copy()
    _require_columns(
        plot_data,
        [x, y] + ([hue] if hue else []) + ([error] if error else []),
    )
    plot_data[y] = numeric_series(plot_data[y])
    if error:
        plot_data[error] = numeric_series(plot_data[error])

    x_values = sorted(_ordered_unique(plot_data[x]), key=_shot_sort_key)
    series_values = _ordered_unique(plot_data[hue]) if hue else [None]
    positions = np.arange(len(x_values), dtype=float)
    palette = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#7f7f7f",
    ]
    markers = ("o", "s", "D", "^", "v", "P")
    fig_width = max(6.5, 1.2 * len(x_values) + 2.5)
    fig, axis = plt.subplots(figsize=(fig_width, 4.8))

    finite_values: list[float] = []
    for series_index, series_value in enumerate(series_values):
        subset = plot_data[plot_data[hue] == series_value] if hue else plot_data
        by_x = subset.drop_duplicates(subset=[x], keep="first").set_index(x)
        values = np.asarray(
            [
                _numeric_value(by_x.loc[value, y])
                if value in by_x.index
                else math.nan
                for value in x_values
            ],
            dtype=float,
        )
        errors = (
            np.asarray(
                [
                    _numeric_value(by_x.loc[value, error])
                    if value in by_x.index
                    else math.nan
                    for value in x_values
                ],
                dtype=float,
            )
            if error
            else None
        )
        is_highlight = highlight is not None and str(series_value) == highlight
        color = "#1f77b4" if is_highlight else palette[
            (series_index + (1 if highlight is not None else 0)) % len(palette)
        ]
        label = str(display_value(hue, series_value)) if hue else y
        axis.errorbar(
            positions,
            values,
            yerr=errors,
            color=color,
            marker=markers[series_index % len(markers)],
            markersize=6.5 if is_highlight else 5.5,
            linewidth=2.2 if is_highlight else 1.7,
            capsize=3.0 if error else 0,
            label=label,
            zorder=3 if is_highlight else 2,
        )
        finite_values.extend(values[np.isfinite(values)].tolist())
        if annotate:
            for position, value in zip(positions, values):
                if np.isfinite(value):
                    axis.annotate(
                        f"{value:.{precision}f}",
                        (position, value),
                        xytext=(0, 7),
                        textcoords="offset points",
                        ha="center",
                        fontsize=8,
                    )

    axis.set_xticks(positions)
    axis.set_xticklabels(
        [_clean_x_tick_label(display_value(x, value)) for value in x_values]
    )
    axis.set_xlabel(x_label or x)
    axis.set_ylabel(y_label or y)
    if title:
        axis.set_title(title)
    if finite_values and min(finite_values) >= 0.0 and max(finite_values) <= 1.0:
        upper = min(1.0, max(finite_values) * 1.18)
        lower = max(0.0, min(finite_values) - 0.08)
        axis.set_ylim(lower, upper)
    if hue:
        axis.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.14),
            ncol=min(len(series_values), 4),
            frameon=False,
        )
    axis.grid(axis="y", color="#E0E0E0", linestyle="--", linewidth=0.7, alpha=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    fig.tight_layout()
    destination = _prepare_plot_output(output)
    fig.savefig(destination, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return destination


def plot_scatter_chart(
    data: pd.DataFrame,
    *,
    x: str,
    y: str,
    output: str | Path,
    label: str | None = None,
    highlight: str | None = None,
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    dpi: int = 300,
) -> Path:
    """Vẽ scatter chart cho bước xử lý hiện tại.

    Parameters
    ----------
    data : pd.DataFrame
        Dữ liệu đầu vào của bước xử lý.
    x : str
        Giá trị ``x`` được sử dụng trong phép xử lý.
    y : str
        Giá trị ``y`` được sử dụng trong phép xử lý.
    output : str | Path
        Vị trí hoặc cấu trúc nhận kết quả.
    label : str | None, optional
        Nhãn hoặc chỉ số lớp liên quan.
    highlight : str | None, optional
        Giá trị ``highlight`` được sử dụng trong phép xử lý.
    title : str | None, optional
        Giá trị ``title`` được sử dụng trong phép xử lý.
    x_label : str | None, optional
        Nhãn hoặc chỉ số lớp liên quan.
    y_label : str | None, optional
        Nhãn hoặc chỉ số lớp liên quan.
    dpi : int, optional
        Giá trị ``dpi`` được sử dụng trong phép xử lý.

    Returns
    -------
    Path
        Kết quả được tạo bởi bước xử lý của hàm.

    Raises
    ------
    ValueError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    """
    plt = _load_pyplot()
    plot_data = data.copy()
    _require_columns(plot_data, [x, y] + ([label] if label else []))
    plot_data[x] = numeric_series(plot_data[x])
    plot_data[y] = numeric_series(plot_data[y])
    plot_data = plot_data[
        np.isfinite(plot_data[x].to_numpy())
        & np.isfinite(plot_data[y].to_numpy())
    ]
    if plot_data.empty:
        raise ValueError("Scatter plot contains no finite coordinate pairs.")

    fig, axis = plt.subplots(figsize=(6.8, 4.8))
    for index, row in plot_data.reset_index(drop=True).iterrows():
        item_label = str(display_value(label, row[label])) if label else ""
        is_highlight = highlight is not None and item_label == highlight
        axis.scatter(
            row[x],
            row[y],
            s=95 if is_highlight else 65,
            marker="D" if is_highlight else "o",
            color="#1f77b4" if is_highlight else "#7f7f7f",
            edgecolor="#0b436c" if is_highlight else "#333333",
            linewidth=1.0,
            zorder=3,
        )
        if label:
            x_offset = 7
            y_offset = 7
            axis.annotate(
                item_label,
                (row[x], row[y]),
                xytext=(x_offset, y_offset),
                textcoords="offset points",
                fontsize=9,
                ha="left",
            )

    axis.set_xlabel(x_label or x)
    axis.set_ylabel(y_label or y)
    if title:
        axis.set_title(title)
    x_values = plot_data[x].to_numpy(dtype=float)
    y_values = plot_data[y].to_numpy(dtype=float)
    x_span = max(float(x_values.max() - x_values.min()), 1e-6)
    axis.set_xlim(
        float(x_values.min() - 0.05 * x_span),
        float(x_values.max() + 0.08 * x_span),
    )
    if float(y_values.min()) >= 0.0 and float(y_values.max()) <= 1.0:
        axis.set_ylim(
            max(0.0, float(y_values.min()) - 0.02),
            min(1.0, float(y_values.max()) + 0.02),
        )
    axis.grid(color="#E0E0E0", linestyle="--", linewidth=0.7, alpha=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    fig.tight_layout()
    destination = _prepare_plot_output(output)
    fig.savefig(destination, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return destination


def plot_confusion_matrix(
    matrix: np.ndarray,
    *,
    class_names: Sequence[str],
    output: str | Path,
    normalize: bool = True,
    title: str | None = None,
    x_label: str = "Predicted label",
    y_label: str = "True label",
    cmap: str = "Blues",
    precision: int = 2,
    dpi: int = 300,
) -> Path:
    """Vẽ confusion matrix cho bước xử lý hiện tại.

    Parameters
    ----------
    matrix : np.ndarray
        Giá trị ``matrix`` được sử dụng trong phép xử lý.
    class_names : Sequence[str]
        Nhãn hoặc chỉ số lớp liên quan.
    output : str | Path
        Vị trí hoặc cấu trúc nhận kết quả.
    normalize : bool, optional
        Giá trị ``normalize`` được sử dụng trong phép xử lý.
    title : str | None, optional
        Giá trị ``title`` được sử dụng trong phép xử lý.
    x_label : str, optional
        Nhãn hoặc chỉ số lớp liên quan.
    y_label : str, optional
        Nhãn hoặc chỉ số lớp liên quan.
    cmap : str, optional
        Giá trị ``cmap`` được sử dụng trong phép xử lý.
    precision : int, optional
        Giá trị ``precision`` được sử dụng trong phép xử lý.
    dpi : int, optional
        Giá trị ``dpi`` được sử dụng trong phép xử lý.

    Returns
    -------
    Path
        Kết quả được tạo bởi bước xử lý của hàm.

    Raises
    ------
    ValueError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    """
    plt = _load_pyplot()
    values = np.asarray(matrix, dtype=np.float64)
    names = list(class_names)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("Confusion matrix must be a square two-dimensional array.")
    if values.shape[0] != len(names):
        raise ValueError("class_names length must match the matrix dimensions.")
    if normalize:
        row_sums = values.sum(axis=1, keepdims=True)
        values = np.divide(
            values,
            row_sums,
            out=np.zeros_like(values),
            where=row_sums > 0,
        )

    side = max(7.0, 0.42 * len(names) + 2.8)
    fig, axis = plt.subplots(figsize=(side, side))
    image = axis.imshow(
        values,
        cmap=cmap,
        vmin=0.0,
        vmax=1.0 if normalize else None,
        aspect="equal",
    )
    axis.set_xticks(np.arange(len(names)))
    axis.set_yticks(np.arange(len(names)))
    axis.set_xticklabels(names, rotation=50, ha="right", fontsize=8)
    axis.set_yticklabels(names, fontsize=8)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    if title:
        axis.set_title(title)

    diagonal = np.diag(values)
    for index, value in enumerate(diagonal):
        if np.isfinite(value):
            threshold = 0.5 if normalize else float(np.nanmax(values)) / 2.0
            axis.text(
                index,
                index,
                f"{value:.{precision}f}" if normalize else f"{value:.0f}",
                ha="center",
                va="center",
                color="white" if value > threshold else "#202020",
                fontsize=8,
            )
    colorbar = fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Recall by true class" if normalize else "Count")
    fig.tight_layout()
    destination = _prepare_plot_output(output)
    fig.savefig(destination, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return destination


def load_class_names(path: str | Path) -> list[str]:
    """Tải class names cho bước xử lý hiện tại.

    Parameters
    ----------
    path : str | Path
        Đường dẫn tài nguyên được sử dụng.

    Returns
    -------
    list[str]
        Kết quả được tạo bởi bước xử lý của hàm.

    Raises
    ------
    ValueError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    """
    source = resolve_path(path)
    names = [
        line.strip()
        for line in source.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if not names:
        raise ValueError(f"No class names found in: {source}")
    return names


def plot_aggregate_confusion_matrix(
    inputs: Sequence[str | Path],
    *,
    class_names: Sequence[str],
    output: str | Path,
    normalize: bool = True,
    show_label_indices: bool = True,
    title: str | None = None,
    x_label: str = "Nhãn dự đoán",
    y_label: str = "Nhãn thực",
    cmap: str = "Blues",
    precision: int = 2,
    dpi: int = 300,
) -> tuple[Path, np.ndarray]:
    """Vẽ aggregate confusion matrix cho bước xử lý hiện tại.

    Parameters
    ----------
    inputs : Sequence[str | Path]
        Giá trị ``inputs`` được sử dụng trong phép xử lý.
    class_names : Sequence[str]
        Nhãn hoặc chỉ số lớp liên quan.
    output : str | Path
        Vị trí hoặc cấu trúc nhận kết quả.
    normalize : bool, optional
        Giá trị ``normalize`` được sử dụng trong phép xử lý.
    show_label_indices : bool, optional
        Nhãn hoặc chỉ số lớp liên quan.
    title : str | None, optional
        Giá trị ``title`` được sử dụng trong phép xử lý.
    x_label : str, optional
        Nhãn hoặc chỉ số lớp liên quan.
    y_label : str, optional
        Nhãn hoặc chỉ số lớp liên quan.
    cmap : str, optional
        Giá trị ``cmap`` được sử dụng trong phép xử lý.
    precision : int, optional
        Giá trị ``precision`` được sử dụng trong phép xử lý.
    dpi : int, optional
        Giá trị ``dpi`` được sử dụng trong phép xử lý.

    Returns
    -------
    tuple[Path, np.ndarray]
        Kết quả được tạo bởi bước xử lý của hàm.

    Raises
    ------
    RuntimeError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    ValueError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    """
    if len(inputs) < 2:
        raise ValueError(
            "Aggregate confusion matrix requires predictions from at least two seeds."
        )
    try:
        from sklearn.metrics import confusion_matrix
    except ImportError as exc:
        raise RuntimeError(
            "scikit-learn is required to compute a confusion matrix."
        ) from exc

    loaded = [load_calibration_arrays(path) for path in inputs]
    num_classes = int(loaded[0][0].shape[1])
    names = [str(name) for name in class_names]
    if len(names) != num_classes:
        raise ValueError(
            f"Expected {num_classes} class names, received {len(names)}."
        )

    pooled = np.zeros((num_classes, num_classes), dtype=np.int64)
    class_ids = np.arange(num_classes)
    for index, (probabilities, labels) in enumerate(loaded):
        probabilities = np.asarray(probabilities)
        labels = np.asarray(labels)
        if probabilities.ndim != 2 or probabilities.shape[1] != num_classes:
            raise ValueError(
                f"Seed input {index} has probability shape {probabilities.shape}; "
                f"expected [N,{num_classes}]."
            )
        if labels.ndim == 2:
            if labels.shape != probabilities.shape:
                raise ValueError(
                    f"Seed input {index} has incompatible one-hot label shape "
                    f"{labels.shape}."
                )
            labels = labels.argmax(axis=1)
        labels = labels.astype(np.int64, copy=False).reshape(-1)
        if labels.shape[0] != probabilities.shape[0]:
            raise ValueError(
                f"Seed input {index} contains different prediction and label counts."
            )
        if np.any(labels < 0) or np.any(labels >= num_classes):
            raise ValueError(f"Seed input {index} contains labels outside [0,C).")
        predictions = probabilities.argmax(axis=1)
        pooled += confusion_matrix(labels, predictions, labels=class_ids)

    display_names = (
        [f"{index + 1:02d}. {name}" for index, name in enumerate(names)]
        if show_label_indices
        else names
    )
    destination = plot_confusion_matrix(
        pooled,
        class_names=display_names,
        output=output,
        normalize=normalize,
        title=title or f"Confusion matrix tổng hợp trên {len(inputs)} seed",
        x_label=x_label,
        y_label=y_label,
        cmap=cmap,
        precision=precision,
        dpi=dpi,
    )

    csv_output = destination.with_name(f"{destination.stem}_counts.csv")
    pd.DataFrame(pooled, index=display_names, columns=display_names).to_csv(
        csv_output,
        encoding="utf-8-sig",
    )
    return destination, pooled


def calibration_statistics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    n_bins: int = 15,
) -> dict[str, Any]:
    """Thực hiện bước calibration statistics trong quy trình hiện tại.

    Parameters
    ----------
    probabilities : np.ndarray
        Giá trị ``probabilities`` được sử dụng trong phép xử lý.
    labels : np.ndarray
        Giá trị ``labels`` được sử dụng trong phép xử lý.
    n_bins : int, optional
        Giá trị ``n_bins`` được sử dụng trong phép xử lý.

    Returns
    -------
    dict[str, Any]
        Kết quả được tạo bởi bước xử lý của hàm.

    Raises
    ------
    ValueError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if probabilities.ndim != 2 or probabilities.shape[0] != labels.shape[0]:
        raise ValueError("Probabilities and labels must have shapes [N,C] and [N].")
    if probabilities.shape[0] == 0 or n_bins < 1:
        raise ValueError("Calibration requires samples and at least one bin.")
    row_sums = probabilities.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("Every probability row must have a positive sum.")
    probabilities = probabilities / row_sums
    if np.any(labels < 0) or np.any(labels >= probabilities.shape[1]):
        raise ValueError("Labels are outside the probability columns.")

    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = (predictions == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    bin_accuracy = np.full(n_bins, np.nan)
    bin_confidence = np.full(n_bins, np.nan)
    bin_count = np.zeros(n_bins, dtype=np.int64)
    ece = 0.0
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (confidence > lower) & (confidence <= upper)
        if index == 0:
            mask |= confidence == 0.0
        bin_count[index] = int(mask.sum())
        if bin_count[index]:
            bin_accuracy[index] = float(correct[mask].mean())
            bin_confidence[index] = float(confidence[mask].mean())
            ece += float(mask.mean()) * abs(
                bin_accuracy[index] - bin_confidence[index]
            )
    one_hot = np.eye(probabilities.shape[1], dtype=np.float64)[labels]
    brier = float(np.square(probabilities - one_hot).sum(axis=1).mean())
    return {
        "bin_edges": edges,
        "bin_centers": centers,
        "bin_accuracy": bin_accuracy,
        "bin_confidence": bin_confidence,
        "bin_count": bin_count,
        "ece": float(ece),
        "brier_score": brier,
    }


def load_calibration_arrays(
    path: str | Path,
    *,
    probabilities_key: str = "probabilities",
    labels_key: str = "labels",
    label_column: str | None = None,
    probability_columns: Sequence[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Tải calibration arrays cho bước xử lý hiện tại.

    Parameters
    ----------
    path : str | Path
        Đường dẫn tài nguyên được sử dụng.
    probabilities_key : str, optional
        Tên hoặc khóa định danh của giá trị.
    labels_key : str, optional
        Tên hoặc khóa định danh của giá trị.
    label_column : str | None, optional
        Nhãn hoặc chỉ số lớp liên quan.
    probability_columns : Sequence[str] | None, optional
        Giá trị ``probability_columns`` được sử dụng trong phép xử lý.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Kết quả được tạo bởi bước xử lý của hàm.

    Raises
    ------
    FileNotFoundError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    KeyError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    ValueError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    """
    source = resolve_path(path)
    if not source.is_file():
        hint = ""
        if source == resolve_path(DEFAULT_PROPOSED_EMBEDDINGS):
            hint = (
                "\nGenerate the proposed-model embeddings first with:\n"
                "python evaluate.py --save-embeddings "
                "+experiment=ctch/proposed/ours_xbone_net seed=42"
            )
        raise FileNotFoundError(f"Calibration input does not exist: {source}{hint}")
    if source.suffix.lower() == ".npz":
        with np.load(source) as archive:
            if probabilities_key not in archive or labels_key not in archive:
                raise KeyError(
                    f"NPZ must contain '{probabilities_key}' and '{labels_key}'. "
                    f"Available keys: {', '.join(archive.files)}"
                )
            return np.asarray(archive[probabilities_key]), np.asarray(archive[labels_key])
    frame = load_frame(source)
    if not label_column or not probability_columns:
        raise ValueError(
            "CSV/JSON calibration input requires --label-column and "
            "--probability-columns."
        )
    _require_columns(frame, [label_column, *probability_columns])
    return (
        frame[list(probability_columns)].apply(pd.to_numeric, errors="raise").to_numpy(),
        pd.to_numeric(frame[label_column], errors="raise").to_numpy(dtype=np.int64),
    )


def _prepare_ovr_targets(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    task: str = "auto",
) -> tuple[np.ndarray, np.ndarray, str]:
    """Chuẩn bị ovr targets cho bước xử lý hiện tại.

    Parameters
    ----------
    probabilities : np.ndarray
        Giá trị ``probabilities`` được sử dụng trong phép xử lý.
    labels : np.ndarray
        Giá trị ``labels`` được sử dụng trong phép xử lý.
    task : str, optional
        Giá trị ``task`` được sử dụng trong phép xử lý.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, str]
        Kết quả được tạo bởi bước xử lý của hàm.

    Raises
    ------
    ValueError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    """
    scores = np.asarray(probabilities, dtype=np.float64)
    raw_labels = np.asarray(labels)
    if scores.ndim != 2 or scores.shape[0] == 0 or scores.shape[1] < 2:
        raise ValueError("Probabilities must have shape [N,C] with N > 0 and C >= 2.")
    if not np.isfinite(scores).all():
        raise ValueError("Probabilities contain NaN or infinite values.")
    if task not in {"auto", "multiclass", "multilabel"}:
        raise ValueError("task must be 'auto', 'multiclass', or 'multilabel'.")

    if task == "auto":
        if raw_labels.ndim == 1:
            resolved_task = "multiclass"
        elif raw_labels.ndim == 2 and raw_labels.shape == scores.shape:
            binary = np.logical_or(raw_labels == 0, raw_labels == 1)
            if not np.all(binary):
                raise ValueError("Two-dimensional labels must be a binary matrix.")
            resolved_task = (
                "multiclass"
                if np.all(np.asarray(raw_labels).sum(axis=1) == 1)
                else "multilabel"
            )
        else:
            raise ValueError(
                "Cannot infer task: labels must have shape [N] or [N,C]."
            )
    else:
        resolved_task = task

    if resolved_task == "multiclass":
        if raw_labels.ndim == 2:
            if raw_labels.shape != scores.shape:
                raise ValueError(
                    "One-hot multiclass labels must match probability shape [N,C]."
                )
            if not np.all(np.logical_or(raw_labels == 0, raw_labels == 1)):
                raise ValueError("One-hot multiclass labels must be binary.")
            if not np.all(np.asarray(raw_labels).sum(axis=1) == 1):
                raise ValueError(
                    "Each multiclass one-hot row must contain exactly one positive."
                )
            class_ids = np.asarray(raw_labels).argmax(axis=1)
        elif raw_labels.ndim == 1:
            class_ids = np.asarray(raw_labels, dtype=np.int64).reshape(-1)
        else:
            raise ValueError("Multiclass labels must have shape [N] or [N,C].")
        if class_ids.shape[0] != scores.shape[0]:
            raise ValueError("Probabilities and labels contain different sample counts.")
        if np.any(class_ids < 0) or np.any(class_ids >= scores.shape[1]):
            raise ValueError("Multiclass labels are outside the probability columns.")
        row_sums = scores.sum(axis=1, keepdims=True)
        if np.any(row_sums <= 0):
            raise ValueError("Every multiclass probability row must have a positive sum.")
        scores = scores / row_sums
        targets = np.eye(scores.shape[1], dtype=np.int64)[class_ids]
    else:
        if raw_labels.shape != scores.shape:
            raise ValueError(
                "Multilabel targets must have the same [N,C] shape as probabilities."
            )
        if not np.all(np.logical_or(raw_labels == 0, raw_labels == 1)):
            raise ValueError("Multilabel targets must contain only 0 and 1.")
        if np.any(scores < 0.0) or np.any(scores > 1.0):
            raise ValueError("Multilabel probabilities must lie in [0,1].")
        targets = np.asarray(raw_labels, dtype=np.int64)
    return scores, targets, resolved_task


def roc_curve_statistics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    task: str = "auto",
) -> dict[str, Any]:
    """Thực hiện bước roc curve statistics trong quy trình hiện tại.

    Parameters
    ----------
    probabilities : np.ndarray
        Giá trị ``probabilities`` được sử dụng trong phép xử lý.
    labels : np.ndarray
        Giá trị ``labels`` được sử dụng trong phép xử lý.
    task : str, optional
        Giá trị ``task`` được sử dụng trong phép xử lý.

    Returns
    -------
    dict[str, Any]
        Kết quả được tạo bởi bước xử lý của hàm.

    Raises
    ------
    RuntimeError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    ValueError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    """
    try:
        from sklearn.metrics import auc, roc_curve
    except ImportError as error:
        raise RuntimeError("ROC plotting requires scikit-learn.") from error

    scores, targets, resolved_task = _prepare_ovr_targets(
        probabilities,
        labels,
        task=task,
    )
    per_class: dict[int, dict[str, Any]] = {}
    skipped: list[int] = []
    for class_id in range(scores.shape[1]):
        class_targets = targets[:, class_id]
        if np.unique(class_targets).size < 2:
            skipped.append(class_id)
            continue
        fpr, tpr, thresholds = roc_curve(class_targets, scores[:, class_id])
        per_class[class_id] = {
            "fpr": fpr,
            "tpr": tpr,
            "thresholds": thresholds,
            "auc": float(auc(fpr, tpr)),
            "support": int(class_targets.sum()),
        }
    if not per_class:
        raise ValueError("ROC requires at least one class with positive and negative samples.")

    grid = np.linspace(0.0, 1.0, 1001)
    mean_tpr = np.mean(
        [np.interp(grid, values["fpr"], values["tpr"]) for values in per_class.values()],
        axis=0,
    )
    mean_tpr[0] = 0.0
    mean_tpr[-1] = 1.0
    macro_auc = float(np.mean([values["auc"] for values in per_class.values()]))
    micro_fpr, micro_tpr, micro_thresholds = roc_curve(
        targets.reshape(-1),
        scores.reshape(-1),
    )
    return {
        "task": resolved_task,
        "num_classes": int(scores.shape[1]),
        "per_class": per_class,
        "skipped_classes": skipped,
        "macro": {
            "fpr": grid,
            "tpr": mean_tpr,
            "auc": macro_auc,
        },
        "micro": {
            "fpr": micro_fpr,
            "tpr": micro_tpr,
            "thresholds": micro_thresholds,
            "auc": float(auc(micro_fpr, micro_tpr)),
        },
    }


def precision_recall_curve_statistics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    task: str = "auto",
) -> dict[str, Any]:
    """Thực hiện bước precision recall curve statistics trong quy trình hiện tại.

    Parameters
    ----------
    probabilities : np.ndarray
        Giá trị ``probabilities`` được sử dụng trong phép xử lý.
    labels : np.ndarray
        Giá trị ``labels`` được sử dụng trong phép xử lý.
    task : str, optional
        Giá trị ``task`` được sử dụng trong phép xử lý.

    Returns
    -------
    dict[str, Any]
        Kết quả được tạo bởi bước xử lý của hàm.

    Raises
    ------
    RuntimeError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    ValueError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    """
    try:
        from sklearn.metrics import average_precision_score, precision_recall_curve
    except ImportError as error:
        raise RuntimeError("Precision--recall plotting requires scikit-learn.") from error

    scores, targets, resolved_task = _prepare_ovr_targets(
        probabilities,
        labels,
        task=task,
    )
    per_class: dict[int, dict[str, Any]] = {}
    skipped: list[int] = []
    for class_id in range(scores.shape[1]):
        class_targets = targets[:, class_id]
        if class_targets.sum() == 0:
            skipped.append(class_id)
            continue
        precision, recall, thresholds = precision_recall_curve(
            class_targets,
            scores[:, class_id],
        )
        per_class[class_id] = {
            "precision": precision,
            "recall": recall,
            "thresholds": thresholds,
            "average_precision": float(
                average_precision_score(class_targets, scores[:, class_id])
            ),
            "support": int(class_targets.sum()),
            "prevalence": float(class_targets.mean()),
        }
    if not per_class:
        raise ValueError("Precision--recall requires at least one class with positives.")

    recall_grid = np.linspace(0.0, 1.0, 1001)
    interpolated_precision = []
    for values in per_class.values():
        recall = values["recall"][::-1]
        precision = values["precision"][::-1]
        interpolated_precision.append(np.interp(recall_grid, recall, precision))
    macro_precision = np.mean(interpolated_precision, axis=0)
    macro_ap = float(
        np.mean([values["average_precision"] for values in per_class.values()])
    )
    flat_targets = targets.reshape(-1)
    flat_scores = scores.reshape(-1)
    micro_precision, micro_recall, micro_thresholds = precision_recall_curve(
        flat_targets,
        flat_scores,
    )
    return {
        "task": resolved_task,
        "num_classes": int(scores.shape[1]),
        "per_class": per_class,
        "skipped_classes": skipped,
        "macro": {
            "precision": macro_precision,
            "recall": recall_grid,
            "average_precision": macro_ap,
        },
        "micro": {
            "precision": micro_precision,
            "recall": micro_recall,
            "thresholds": micro_thresholds,
            "average_precision": float(
                average_precision_score(flat_targets, flat_scores)
            ),
            "prevalence": float(flat_targets.mean()),
        },
    }


def _curve_class_names(
    num_classes: int,
    class_names: Sequence[str] | None,
) -> list[str]:
    """Thực hiện bước curve class names trong quy trình hiện tại.

    Parameters
    ----------
    num_classes : int
        Số lượng, kích thước hoặc tỷ lệ được sử dụng.
    class_names : Sequence[str] | None
        Nhãn hoặc chỉ số lớp liên quan.

    Returns
    -------
    list[str]
        Kết quả được tạo bởi bước xử lý của hàm.

    Raises
    ------
    ValueError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    """
    if class_names is None:
        return [f"Class {class_id}" for class_id in range(num_classes)]
    names = [str(value) for value in class_names]
    if len(names) != num_classes:
        raise ValueError(
            f"Expected {num_classes} class names, received {len(names)}."
        )
    return names


def plot_roc_curve(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    output: str | Path,
    task: str = "auto",
    class_names: Sequence[str] | None = None,
    show_per_class: bool = True,
    title: str | None = None,
    dpi: int = 300,
) -> tuple[Path, dict[str, Any]]:
    """Vẽ roc curve cho bước xử lý hiện tại.

    Parameters
    ----------
    probabilities : np.ndarray
        Giá trị ``probabilities`` được sử dụng trong phép xử lý.
    labels : np.ndarray
        Giá trị ``labels`` được sử dụng trong phép xử lý.
    output : str | Path
        Vị trí hoặc cấu trúc nhận kết quả.
    task : str, optional
        Giá trị ``task`` được sử dụng trong phép xử lý.
    class_names : Sequence[str] | None, optional
        Nhãn hoặc chỉ số lớp liên quan.
    show_per_class : bool, optional
        Nhãn hoặc chỉ số lớp liên quan.
    title : str | None, optional
        Giá trị ``title`` được sử dụng trong phép xử lý.
    dpi : int, optional
        Giá trị ``dpi`` được sử dụng trong phép xử lý.

    Returns
    -------
    tuple[Path, dict[str, Any]]
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    plt = _load_pyplot()
    statistics = roc_curve_statistics(probabilities, labels, task=task)
    fig, axis = plt.subplots(figsize=(6.2, 6.2))
    macro = statistics["macro"]
    axis.plot(
        macro["fpr"],
        macro["tpr"],
        color="#1769D2",
        linewidth=2.8,
        label=f"Macro-average (AUROC = {macro['auc']:.3f})",
        zorder=5,
    )
    axis.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        color="#555555",
        linestyle=":",
        linewidth=1.4,
        label="Random prediction",
    )
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.01)
    axis.set_xlabel("False-positive rate")
    axis.set_ylabel("True-positive rate")
    axis.set_title(title or "One-vs-rest ROC curves")
    axis.grid(color="#D9D9D9", linewidth=0.7, alpha=0.7)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(
        loc="lower right",
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.9,
    )
    fig.tight_layout()
    destination = _prepare_plot_output(output)
    fig.savefig(destination, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return destination, statistics


def plot_precision_recall_curve(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    output: str | Path,
    task: str = "auto",
    class_names: Sequence[str] | None = None,
    show_per_class: bool = True,
    title: str | None = None,
    dpi: int = 300,
) -> tuple[Path, dict[str, Any]]:
    """Vẽ precision recall curve cho bước xử lý hiện tại.

    Parameters
    ----------
    probabilities : np.ndarray
        Giá trị ``probabilities`` được sử dụng trong phép xử lý.
    labels : np.ndarray
        Giá trị ``labels`` được sử dụng trong phép xử lý.
    output : str | Path
        Vị trí hoặc cấu trúc nhận kết quả.
    task : str, optional
        Giá trị ``task`` được sử dụng trong phép xử lý.
    class_names : Sequence[str] | None, optional
        Nhãn hoặc chỉ số lớp liên quan.
    show_per_class : bool, optional
        Nhãn hoặc chỉ số lớp liên quan.
    title : str | None, optional
        Giá trị ``title`` được sử dụng trong phép xử lý.
    dpi : int, optional
        Giá trị ``dpi`` được sử dụng trong phép xử lý.

    Returns
    -------
    tuple[Path, dict[str, Any]]
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    plt = _load_pyplot()
    statistics = precision_recall_curve_statistics(
        probabilities,
        labels,
        task=task,
    )
    fig, axis = plt.subplots(figsize=(6.2, 6.2))
    macro = statistics["macro"]
    axis.plot(
        macro["recall"],
        macro["precision"],
        color="#1769D2",
        linewidth=2.8,
        label=f"Macro-average (AP = {macro['average_precision']:.3f})",
        zorder=5,
    )
    prevalence = float(statistics["micro"]["prevalence"])
    axis.axhline(
        prevalence,
        color="#555555",
        linestyle=":",
        linewidth=1.4,
        label=f"Prevalence ({prevalence:.3f})",
    )
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.01)
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_title(title or "One-vs-rest precision–recall curves")
    axis.grid(color="#D9D9D9", linewidth=0.7, alpha=0.7)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(
        loc="upper right",
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.9,
    )
    fig.tight_layout()
    destination = _prepare_plot_output(output)
    fig.savefig(destination, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return destination, statistics


def plot_calibration_curve(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    output: str | Path,
    n_bins: int = 15,
    title: str | None = None,
    model_label: str = "Model",
    ideal_label: str = "Perfect calibration",
    x_label: str = "Mean confidence",
    y_label: str = "Observed accuracy",
    dpi: int = 300,
) -> tuple[Path, dict[str, Any]]:
    """Vẽ calibration curve cho bước xử lý hiện tại.

    Parameters
    ----------
    probabilities : np.ndarray
        Giá trị ``probabilities`` được sử dụng trong phép xử lý.
    labels : np.ndarray
        Giá trị ``labels`` được sử dụng trong phép xử lý.
    output : str | Path
        Vị trí hoặc cấu trúc nhận kết quả.
    n_bins : int, optional
        Giá trị ``n_bins`` được sử dụng trong phép xử lý.
    title : str | None, optional
        Giá trị ``title`` được sử dụng trong phép xử lý.
    model_label : str, optional
        Mô hình hoặc thành phần mô hình cần xử lý.
    ideal_label : str, optional
        Nhãn hoặc chỉ số lớp liên quan.
    x_label : str, optional
        Nhãn hoặc chỉ số lớp liên quan.
    y_label : str, optional
        Nhãn hoặc chỉ số lớp liên quan.
    dpi : int, optional
        Giá trị ``dpi`` được sử dụng trong phép xử lý.

    Returns
    -------
    tuple[Path, dict[str, Any]]
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    plt = _load_pyplot()
    statistics = calibration_statistics(probabilities, labels, n_bins=n_bins)
    occupied = statistics["bin_count"] > 0
    centers = statistics["bin_centers"]
    fig, axis = plt.subplots(figsize=(5.0, 4.4))
    axis.plot(
        [0, 1],
        [0, 1],
        color="#D62728",
        linestyle=(0, (4, 3)),
        linewidth=1.8,
        label=ideal_label,
        zorder=3,
    )
    axis.bar(
        centers[occupied],
        statistics["bin_accuracy"][occupied],
        width=0.8 / n_bins,
        color="#1769D2",
        edgecolor="#0A2F5A",
        linewidth=0.8,
        label=model_label,
        zorder=2,
    )
    axis.text(
        0.05,
        0.95,
        f"ECE = {100 * statistics['ece']:.2f}%",
        transform=axis.transAxes,
        ha="left",
        va="top",
        bbox={
            "facecolor": "white",
            "edgecolor": "#555555",
            "boxstyle": "square,pad=0.35",
            "alpha": 0.96,
        },
    )
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.set_title(title or "Reliability diagram")
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.8, zorder=0)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(
        loc="lower right",
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.9,
    )
    fig.tight_layout()
    destination = _prepare_plot_output(output)
    fig.savefig(destination, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return destination, statistics


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
    """Thực hiện bước mean std trong quy trình hiện tại.

    Parameters
    ----------
    values : Sequence[float]
        Giá trị ``values`` được sử dụng trong phép xử lý.

    Returns
    -------
    tuple[float, float]
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=1)) if len(array) > 1 else 0.0


def plot_aggregate_prediction_curves(
    inputs: Sequence[str | Path],
    *,
    output_dir: str | Path,
    prefix: str = "aggregate",
    task: str = "auto",
    n_bins: int = 15,
    title_prefix: str = "Model",
    dpi: int = 300,
) -> dict[str, Any]:
    """Vẽ aggregate prediction curves cho bước xử lý hiện tại.

    Parameters
    ----------
    inputs : Sequence[str | Path]
        Giá trị ``inputs`` được sử dụng trong phép xử lý.
    output_dir : str | Path
        Đường dẫn tài nguyên được sử dụng.
    prefix : str, optional
        Giá trị ``prefix`` được sử dụng trong phép xử lý.
    task : str, optional
        Giá trị ``task`` được sử dụng trong phép xử lý.
    n_bins : int, optional
        Giá trị ``n_bins`` được sử dụng trong phép xử lý.
    title_prefix : str, optional
        Giá trị ``title_prefix`` được sử dụng trong phép xử lý.
    dpi : int, optional
        Giá trị ``dpi`` được sử dụng trong phép xử lý.

    Returns
    -------
    dict[str, Any]
        Kết quả được tạo bởi bước xử lý của hàm.

    Raises
    ------
    ValueError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    """
    if len(inputs) < 2:
        raise ValueError("Aggregate curves require predictions from at least two seeds.")
    if n_bins < 1:
        raise ValueError("n_bins must be positive.")

    loaded = [load_calibration_arrays(path) for path in inputs]
    reference_labels = loaded[0][1]
    reference_shape = loaded[0][0].shape
    for index, (probabilities, labels) in enumerate(loaded[1:], start=1):
        if probabilities.shape != reference_shape:
            raise ValueError(
                f"Seed input {index} has probability shape {probabilities.shape}; "
                f"expected {reference_shape}."
            )
        if not np.array_equal(labels, reference_labels):
            raise ValueError(
                "Seed inputs do not contain labels in the same sample order."
            )

    destination_dir = resolve_path(output_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    roc_stats = [
        roc_curve_statistics(probabilities, labels, task=task)
        for probabilities, labels in loaded
    ]
    pr_stats = [
        precision_recall_curve_statistics(probabilities, labels, task=task)
        for probabilities, labels in loaded
    ]
    calibration_stats = [
        calibration_statistics(probabilities, labels, n_bins=n_bins)
        for probabilities, labels in loaded
    ]
    plt = _load_pyplot()

    roc_grid = np.linspace(0.0, 1.0, 1001)
    macro_tpr = np.stack(
        [
            np.interp(roc_grid, stats["macro"]["fpr"], stats["macro"]["tpr"])
            for stats in roc_stats
        ]
    )
    macro_auc = _mean_std([stats["macro"]["auc"] for stats in roc_stats])
    micro_auc = _mean_std([stats["micro"]["auc"] for stats in roc_stats])
    figure, axis = plt.subplots(figsize=(6.4, 5.6))
    mean_macro_tpr = macro_tpr.mean(axis=0)
    std_macro_tpr = macro_tpr.std(axis=0, ddof=1)
    axis.plot(
        roc_grid,
        mean_macro_tpr,
        color="#1769D2",
        linewidth=2.5,
        label=(
            f"Macro-average (AUROC = {macro_auc[0]:.3f}"
            f" ± {macro_auc[1]:.3f})"
        ),
    )
    axis.fill_between(
        roc_grid,
        np.clip(mean_macro_tpr - std_macro_tpr, 0.0, 1.0),
        np.clip(mean_macro_tpr + std_macro_tpr, 0.0, 1.0),
        color="#1769D2",
        alpha=0.14,
        linewidth=0,
    )
    axis.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        color="#555555",
        linestyle=":",
        linewidth=1.4,
        label="Random prediction",
    )
    axis.set(xlim=(0, 1), ylim=(0, 1.01))
    axis.set_xlabel("False-positive rate")
    axis.set_ylabel("True-positive rate")
    axis.set_title(f"{title_prefix} ROC trung bình trên {len(inputs)} seed")
    axis.grid(color="#D9D9D9", linewidth=0.7, alpha=0.7)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(
        loc="lower right",
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.9,
    )
    figure.tight_layout()
    roc_output = destination_dir / f"{prefix}_roc_{len(inputs)}seed.png"
    figure.savefig(roc_output, dpi=dpi, bbox_inches="tight")
    plt.close(figure)

    recall_grid = np.linspace(0.0, 1.0, 1001)
    macro_precision = np.stack(
        [
            np.interp(
                recall_grid,
                stats["macro"]["recall"],
                stats["macro"]["precision"],
            )
            for stats in pr_stats
        ]
    )
    macro_ap = _mean_std(
        [stats["macro"]["average_precision"] for stats in pr_stats]
    )
    micro_ap = _mean_std(
        [stats["micro"]["average_precision"] for stats in pr_stats]
    )
    prevalence = float(
        np.mean([stats["micro"]["prevalence"] for stats in pr_stats])
    )
    figure, axis = plt.subplots(figsize=(6.4, 5.6))
    mean_macro_precision = macro_precision.mean(axis=0)
    std_macro_precision = macro_precision.std(axis=0, ddof=1)
    axis.plot(
        recall_grid,
        mean_macro_precision,
        color="#1769D2",
        linewidth=2.5,
        label=(
            f"Macro-average (AP = {macro_ap[0]:.3f}"
            f" ± {macro_ap[1]:.3f})"
        ),
    )
    axis.fill_between(
        recall_grid,
        np.clip(mean_macro_precision - std_macro_precision, 0.0, 1.0),
        np.clip(mean_macro_precision + std_macro_precision, 0.0, 1.0),
        color="#1769D2",
        alpha=0.14,
        linewidth=0,
    )
    axis.axhline(
        prevalence,
        color="#555555",
        linestyle=":",
        linewidth=1.4,
        label=f"Prevalence ({prevalence:.3f})",
    )
    axis.set(xlim=(0, 1), ylim=(0, 1.01))
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_title(
        f"{title_prefix} Precision–Recall trung bình trên {len(inputs)} seed"
    )
    axis.grid(color="#D9D9D9", linewidth=0.7, alpha=0.7)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(
        loc="upper right",
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.9,
    )
    figure.tight_layout()
    pr_output = destination_dir / f"{prefix}_pr_{len(inputs)}seed.png"
    figure.savefig(pr_output, dpi=dpi, bbox_inches="tight")
    plt.close(figure)

    bin_accuracy = np.stack(
        [stats["bin_accuracy"] for stats in calibration_stats]
    )
    bin_confidence = np.stack(
        [stats["bin_confidence"] for stats in calibration_stats]
    )
    mean_accuracy = np.full(n_bins, np.nan)
    std_accuracy = np.zeros(n_bins)
    mean_confidence = np.full(n_bins, np.nan)
    for bin_index in range(n_bins):
        valid = np.isfinite(bin_accuracy[:, bin_index]) & np.isfinite(
            bin_confidence[:, bin_index]
        )
        if valid.any():
            mean_accuracy[bin_index] = float(
                bin_accuracy[valid, bin_index].mean()
            )
            mean_confidence[bin_index] = float(
                bin_confidence[valid, bin_index].mean()
            )
            if valid.sum() > 1:
                std_accuracy[bin_index] = float(
                    bin_accuracy[valid, bin_index].std(ddof=1)
                )
    occupied = np.isfinite(mean_accuracy) & np.isfinite(mean_confidence)
    ece = _mean_std([stats["ece"] for stats in calibration_stats])
    brier = _mean_std([stats["brier_score"] for stats in calibration_stats])
    figure, axis = plt.subplots(figsize=(5.6, 5.0))
    axis.plot(
        [0, 1],
        [0, 1],
        color="#D62728",
        linestyle=(0, (4, 3)),
        linewidth=1.8,
        label="Hiệu chuẩn lý tưởng",
    )
    axis.errorbar(
        mean_confidence[occupied],
        mean_accuracy[occupied],
        yerr=std_accuracy[occupied],
        color="#1769D2",
        marker="o",
        linewidth=2.0,
        capsize=3,
        label=f"Trung bình ± SD ({len(inputs)} seed)",
    )
    axis.text(
        0.05,
        0.95,
        f"ECE = {100 * ece[0]:.2f}% ± {100 * ece[1]:.2f}%",
        transform=axis.transAxes,
        ha="left",
        va="top",
        bbox={
            "facecolor": "white",
            "edgecolor": "#555555",
            "boxstyle": "square,pad=0.35",
            "alpha": 0.96,
        },
    )
    axis.set(xlim=(0, 1), ylim=(0, 1))
    axis.set_xlabel("Độ tin cậy trung bình")
    axis.set_ylabel("Độ chính xác quan sát")
    axis.set_title(
        f"{title_prefix} Hiệu chuẩn trung bình trên {len(inputs)} seed"
    )
    axis.grid(color="#D9D9D9", linewidth=0.7, alpha=0.7)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(
        loc="lower right",
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.9,
    )
    figure.tight_layout()
    calibration_output = (
        destination_dir / f"{prefix}_calibration_{len(inputs)}seed.png"
    )
    figure.savefig(calibration_output, dpi=dpi, bbox_inches="tight")
    plt.close(figure)

    summary = {
        "inputs": [str(resolve_path(path)) for path in inputs],
        "num_seeds": len(inputs),
        "macro_auroc": {"mean": macro_auc[0], "std": macro_auc[1]},
        "micro_auroc": {"mean": micro_auc[0], "std": micro_auc[1]},
        "macro_auprc": {"mean": macro_ap[0], "std": macro_ap[1]},
        "micro_auprc": {"mean": micro_ap[0], "std": micro_ap[1]},
        "ece": {"mean": ece[0], "std": ece[1]},
        "brier_score": {"mean": brier[0], "std": brier[1]},
        "outputs": {
            "roc": str(roc_output),
            "precision_recall": str(pr_output),
            "calibration": str(calibration_output),
        },
    }
    summary_output = destination_dir / f"{prefix}_curves_{len(inputs)}seed.json"
    summary_output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    summary["outputs"]["summary"] = str(summary_output)
    return summary


def plot_full_shot_comparison(
    input_file: str | Path,
    output: str | Path,
    *,
    datasets: Sequence[str] = ("BTXRD", "CTCH"),
    title_template: str = "Phân loại full-shot trên {dataset}",
    dataset_titles: Mapping[str, str] | None = None,
    figure_title: str | None = None,
    dpi: int = 300,
) -> Path:
    """Vẽ full shot comparison cho bước xử lý hiện tại.

    Parameters
    ----------
    input_file : str | Path
        Đường dẫn tài nguyên được sử dụng.
    output : str | Path
        Vị trí hoặc cấu trúc nhận kết quả.
    datasets : Sequence[str], optional
        Giá trị ``datasets`` được sử dụng trong phép xử lý.
    title_template : str, optional
        Giá trị ``title_template`` được sử dụng trong phép xử lý.
    dataset_titles : Mapping[str, str] | None, optional
        Dữ liệu đầu vào của bước xử lý.
    figure_title : str | None, optional
        Giá trị ``figure_title`` được sử dụng trong phép xử lý.
    dpi : int, optional
        Giá trị ``dpi`` được sử dụng trong phép xử lý.

    Returns
    -------
    Path
        Kết quả được tạo bởi bước xử lý của hàm.

    Raises
    ------
    ValueError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    """
    frame = load_frame(input_file)
    resolved_dataset_titles = dict(dataset_titles or {})
    required = (
        "dataset",
        "category",
        "config",
        "accuracy_mean",
        "accuracy_std",
        "balanced_accuracy_mean",
        "balanced_accuracy_std",
        "f1_macro_mean",
        "f1_macro_std",
        "auroc_macro_mean",
        "auroc_macro_std",
        "auprc_macro_mean",
        "auprc_macro_std",
    )
    _require_columns(frame, required)
    model_specs = (
        (
            "fft_resnet50",
            "Baselines / Full fine-tuning",
            "ResNet-50",
            "#8c564b",
        ),
        (
            "fft_densenet",
            "Baselines / Full fine-tuning",
            "DenseNet-121",
            "#e377c2",
        ),
        (
            "fft_medclip",
            "Baselines / Full fine-tuning",
            "MedCLIP",
            "#bcbd22",
        ),
        (
            "fft_clip",
            "Baselines / Full fine-tuning",
            "CLIP",
            REPORT_PALETTE["purple"],
        ),
        (
            "fft_pubmedclip",
            "Baselines / Full fine-tuning",
            "PubMedCLIP",
            REPORT_PALETTE["green"],
        ),
        (
            "fft_biomedclip",
            "Baselines / Full fine-tuning",
            "BiomedCLIP",
            REPORT_PALETTE["orange"],
        ),
        (
            "ours_xbone_net",
            "Proposed",
            "XBone-Net",
            REPORT_PALETTE["blue"],
        ),
    )
    metric_specs = (
        ("accuracy_mean", "accuracy_std", "Accuracy"),
        ("balanced_accuracy_mean", "balanced_accuracy_std", "Balanced Acc."),
        ("f1_macro_mean", "f1_macro_std", "Macro-F1"),
        ("auroc_macro_mean", "auroc_macro_std", "Macro-AUROC"),
        ("auprc_macro_mean", "auprc_macro_std", "Macro-AUPRC"),
    )

    plt = _load_pyplot()
    figure, axes = plt.subplots(
        1,
        len(datasets),
        figsize=(16.5, 5.8),
        sharey=True,
        squeeze=False,
    )
    positions = np.arange(len(metric_specs), dtype=float)
    width = 0.11
    for dataset_index, dataset in enumerate(datasets):
        axis = axes[0, dataset_index]
        label_rows: list[
            tuple[Any, np.ndarray, np.ndarray, str]
        ] = []
        for model_index, (
            config,
            category,
            model_label,
            color,
        ) in enumerate(model_specs):
            selected = frame[
                (frame["dataset"].astype(str) == str(dataset))
                & (frame["category"].astype(str) == category)
                & (frame["config"].astype(str) == config)
            ]
            if len(selected) != 1:
                raise ValueError(
                    f"Expected one full-shot row for {dataset}/{config}; "
                    f"found {len(selected)}."
                )
            row = selected.iloc[0]
            values = np.asarray(
                [float(row[mean_column]) for mean_column, _, _ in metric_specs]
            )
            errors = np.asarray(
                [float(row[std_column]) for _, std_column, _ in metric_specs]
            )
            offsets = positions + (model_index - 3) * width
            bars = axis.bar(
                offsets,
                values,
                width=width,
                yerr=errors,
                capsize=2,
                color=color,
                edgecolor="#333333",
                linewidth=0.8,
                error_kw={
                    "ecolor": "#333333",
                    "elinewidth": 1.0,
                    "capthick": 1.0,
                },
                label=model_label,
            )
            label_rows.append((bars, values, errors, config))
        panel_title = resolved_dataset_titles.get(
            str(dataset),
            title_template.format(dataset=dataset),
        )
        axis.set_title(panel_title, pad=12)
        axis.set_xticks(
            positions,
            labels=[label for _, _, label in metric_specs],
            rotation=15,
            ha="right",
        )
        axis.set_ylim(0.0, 1.07)
        axis.grid(
            axis="y",
            color="#D9D9D9",
            linestyle="--",
            linewidth=0.8,
            alpha=0.8,
        )
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[0, 0].set_ylabel("Giá trị độ đo")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.04),
        ncol=7,
        frameon=False,
    )
    if figure_title:
        figure.suptitle(figure_title)
    figure.tight_layout(
        rect=(0.0, 0.10, 1.0, 0.94 if figure_title else 1.0)
    )
    destination = _prepare_plot_output(output)
    figure.savefig(destination, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return destination


FULL_SHOT_BASELINE_ORDER = (
    "lora_biomedclip",
    "lora_pubmedclip",
    "fft_biomedclip",
    "fft_pubmedclip",
    "fft_clip",
    "fft_medclip",
    "fft_resnet50",
    "fft_densenet",
)


def _seed_list(value: Any) -> list[int]:
    """Thực hiện bước seed list trong quy trình hiện tại.

    Parameters
    ----------
    value : Any
        Giá trị ``value`` được sử dụng trong phép xử lý.

    Returns
    -------
    list[int]
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    return [int(item) for item in re.findall(r"\d+", str(value))]


def build_full_shot_paired_effects(
    input_file: str | Path,
    *,
    results_root: str | Path = DEFAULT_RESULTS_ROOT,
    datasets: Sequence[str] = ("BTXRD", "CTCH"),
    metric: str = "f1_macro",
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    """Xây dựng full shot paired effects cho bước xử lý hiện tại.

    Parameters
    ----------
    input_file : str | Path
        Đường dẫn tài nguyên được sử dụng.
    results_root : str | Path, optional
        Đường dẫn tài nguyên được sử dụng.
    datasets : Sequence[str], optional
        Giá trị ``datasets`` được sử dụng trong phép xử lý.
    metric : str, optional
        Giá trị điểm hoặc độ đo cần sử dụng.
    confidence_level : float, optional
        Giá trị ``confidence_level`` được sử dụng trong phép xử lý.

    Returns
    -------
    pd.DataFrame
        Kết quả được tạo bởi bước xử lý của hàm.

    Raises
    ------
    ValueError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    """
    supported_metrics = {
        "accuracy",
        "balanced_accuracy",
        "f1_macro",
        "auroc_macro",
        "auprc_macro",
    }
    if metric not in supported_metrics:
        raise ValueError(f"Unsupported full-shot paired metric: {metric}")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between zero and one.")

    frame = load_frame(input_file)
    _require_columns(
        frame,
        ("dataset", "category", "config", "experiment", "seeds"),
    )
    root = resolve_path(results_root)
    rows: list[dict[str, Any]] = []
    from scipy.stats import t as student_t

    for dataset in datasets:
        dataset_rows = frame[
            frame["dataset"].astype(str).str.casefold()
            == str(dataset).casefold()
        ].copy()
        proposed_rows = dataset_rows[
            (dataset_rows["config"].astype(str) == "ours_xbone_net")
            & (dataset_rows["category"].astype(str) == "Proposed")
        ]
        if len(proposed_rows) != 1:
            raise ValueError(
                f"Expected one XBone-Net row for {dataset}; "
                f"found {len(proposed_rows)}."
            )
        proposed_row = proposed_rows.iloc[0]
        proposed_seeds = _seed_list(proposed_row["seeds"])
        proposed_metrics = _load_seed_metrics(
            root / str(proposed_row["experiment"]),
            seeds=proposed_seeds,
        )

        baseline_rows = dataset_rows[
            dataset_rows["category"].astype(str).isin(
                ("Baselines / PEFT", "Baselines / Full fine-tuning")
            )
        ].copy()
        order_map = {
            config: index
            for index, config in enumerate(FULL_SHOT_BASELINE_ORDER)
        }
        baseline_rows["_order"] = (
            baseline_rows["config"]
            .astype(str)
            .map(order_map)
            .fillna(len(order_map))
        )
        baseline_rows = baseline_rows.sort_values(
            ["_order", "config"],
            kind="stable",
        )

        for _, baseline_row in baseline_rows.iterrows():
            baseline_config = str(baseline_row["config"])
            baseline_seeds = _seed_list(baseline_row["seeds"])
            paired_seeds = [
                seed for seed in proposed_seeds if seed in baseline_seeds
            ]
            if len(paired_seeds) < 2:
                continue
            baseline_metrics = _load_seed_metrics(
                root / str(baseline_row["experiment"]),
                seeds=paired_seeds,
            )
            differences = np.asarray(
                [
                    proposed_metrics[seed][metric]
                    - baseline_metrics[seed][metric]
                    for seed in paired_seeds
                ],
                dtype=np.float64,
            )
            n_pairs = len(differences)
            delta_mean = float(differences.mean())
            delta_std = float(differences.std(ddof=1))
            standard_error = delta_std / math.sqrt(n_pairs)
            alpha = 1.0 - confidence_level
            critical_value = float(
                student_t.ppf(1.0 - alpha / 2.0, df=n_pairs - 1)
            )
            margin = critical_value * standard_error
            rows.append(
                {
                    "dataset": str(dataset),
                    "baseline_config": baseline_config,
                    "baseline": CONFIG_DISPLAY_NAMES.get(
                        baseline_config,
                        baseline_config,
                    ),
                    "metric": metric,
                    "n_pairs": n_pairs,
                    "seeds": " ".join(str(seed) for seed in paired_seeds),
                    "delta_mean": delta_mean,
                    "delta_std": delta_std,
                    "ci_low": delta_mean - margin,
                    "ci_high": delta_mean + margin,
                    "confidence_level": confidence_level,
                    "ci_method": "paired Student t interval across seeds",
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("No paired full-shot comparisons could be constructed.")
    return result


def plot_full_shot_paired_forest(
    input_file: str | Path,
    output: str | Path,
    *,
    results_root: str | Path = DEFAULT_RESULTS_ROOT,
    datasets: Sequence[str] = ("BTXRD", "CTCH"),
    metric: str = "f1_macro",
    confidence_level: float = 0.95,
    csv_output: str | Path | None = None,
    dpi: int = 300,
) -> tuple[Path, Path]:
    """Vẽ full shot paired forest cho bước xử lý hiện tại.

    Parameters
    ----------
    input_file : str | Path
        Đường dẫn tài nguyên được sử dụng.
    output : str | Path
        Vị trí hoặc cấu trúc nhận kết quả.
    results_root : str | Path, optional
        Đường dẫn tài nguyên được sử dụng.
    datasets : Sequence[str], optional
        Giá trị ``datasets`` được sử dụng trong phép xử lý.
    metric : str, optional
        Giá trị điểm hoặc độ đo cần sử dụng.
    confidence_level : float, optional
        Giá trị ``confidence_level`` được sử dụng trong phép xử lý.
    csv_output : str | Path | None, optional
        Vị trí hoặc cấu trúc nhận kết quả.
    dpi : int, optional
        Giá trị ``dpi`` được sử dụng trong phép xử lý.

    Returns
    -------
    tuple[Path, Path]
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    effects = build_full_shot_paired_effects(
        input_file,
        results_root=results_root,
        datasets=datasets,
        metric=metric,
        confidence_level=confidence_level,
    )
    metric_labels = {
        "accuracy": "Accuracy",
        "balanced_accuracy": "Balanced Accuracy",
        "f1_macro": "Macro-F1",
        "auroc_macro": "Macro-AUROC",
        "auprc_macro": "Macro-AUPRC",
    }
    metric_label = metric_labels[metric]

    plt = _load_pyplot()
    from matplotlib.lines import Line2D

    figure, axes = plt.subplots(
        1,
        len(datasets),
        figsize=(12.4, 6.2),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    baseline_labels = [
        CONFIG_DISPLAY_NAMES.get(config, config)
        for config in FULL_SHOT_BASELINE_ORDER
        if config in set(effects["baseline_config"].astype(str))
    ]
    positions = np.arange(len(baseline_labels), dtype=float)
    label_to_position = {
        label: position
        for position, label in zip(positions, baseline_labels)
    }

    for dataset_index, dataset in enumerate(datasets):
        axis = axes[0, dataset_index]
        dataset_effects = effects[
            effects["dataset"].astype(str).str.casefold()
            == str(dataset).casefold()
        ]
        for _, row in dataset_effects.iterrows():
            mean = float(row["delta_mean"])
            lower = float(row["ci_low"])
            upper = float(row["ci_high"])
            position = label_to_position[str(row["baseline"])]
            if lower > 0.0:
                color = REPORT_PALETTE["blue"]
            elif upper < 0.0:
                color = REPORT_PALETTE["orange"]
            else:
                color = REPORT_PALETTE["gray"]
            axis.errorbar(
                mean,
                position,
                xerr=np.asarray([[mean - lower], [upper - mean]]),
                fmt="o",
                color=color,
                ecolor=color,
                markersize=6,
                elinewidth=1.8,
                capsize=4,
                zorder=3,
            )
        axis.axvline(
            0.0,
            color="#222222",
            linestyle="--",
            linewidth=1.1,
        )
        axis.set_title(str(dataset))
        axis.set_yticks(positions, labels=baseline_labels)
        axis.invert_yaxis()
        axis.grid(
            axis="x",
            color="#D9D9D9",
            linestyle="--",
            linewidth=0.8,
            alpha=0.8,
        )
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    all_limits = np.concatenate(
        (
            numeric_series(effects["ci_low"]).to_numpy(),
            numeric_series(effects["ci_high"]).to_numpy(),
            np.asarray([0.0]),
        )
    )
    span = float(np.nanmax(all_limits) - np.nanmin(all_limits))
    padding = max(0.01, 0.12 * span)
    axes[0, 0].set_xlim(
        float(np.nanmin(all_limits) - padding),
        float(np.nanmax(all_limits) + padding),
    )
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            color=REPORT_PALETTE["blue"],
            label="Khoảng tin cậy nằm phía XBone-Net",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            color=REPORT_PALETTE["orange"],
            label="Khoảng tin cậy nằm phía mô hình cơ sở",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            color=REPORT_PALETTE["gray"],
            label="Khoảng tin cậy cắt mốc 0",
        ),
    ]
    figure.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.005),
        ncol=3,
        frameon=False,
        fontsize=8.5,
    )
    figure.tight_layout(rect=(0.0, 0.12, 1.0, 1.0))

    destination = _prepare_plot_output(output)
    figure.savefig(destination, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    csv_destination = resolve_path(
        csv_output
        if csv_output is not None
        else destination.with_suffix(".csv")
    )
    csv_destination.parent.mkdir(parents=True, exist_ok=True)
    effects.to_csv(csv_destination, index=False, encoding="utf-8-sig")
    return destination, csv_destination


def _add_shared_input(parser: argparse.ArgumentParser) -> None:
    """Thực hiện bước add shared đầu vào trong quy trình hiện tại.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Giá trị ``parser`` được sử dụng trong phép xử lý.
    """
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--json-record-path",
        default=None,
        help="Optional dot-separated path to records inside a JSON object.",
    )
    parser.add_argument(
        "--where",
        action="append",
        default=[],
        metavar="COLUMN=VALUE",
        help="Keep rows with an exact value; repeat the option to combine filters.",
    )
    parser.add_argument(
        "--where-in",
        action="append",
        default=[],
        metavar="COLUMN=VALUE1|VALUE2",
        help="Keep rows matching any pipe-separated value in a column.",
    )


def _load_filtered_frame(args: argparse.Namespace) -> pd.DataFrame:
    """Tải filtered bảng dữ liệu cho bước xử lý hiện tại.

    Parameters
    ----------
    args : argparse.Namespace
        Các đối số vị trí bổ sung.

    Returns
    -------
    pd.DataFrame
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    return filter_frame(
        load_frame(args.input, args.json_record_path),
        where=args.where,
        where_in=args.where_in,
    )


def _add_prediction_curve_args(parser: argparse.ArgumentParser) -> None:
    """Thực hiện bước add prediction curve tham số trong quy trình hiện tại.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Giá trị ``parser`` được sử dụng trong phép xử lý.
    """
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_PROPOSED_EMBEDDINGS,
        help="NPZ or tabular predictions containing probabilities and labels.",
    )
    parser.add_argument("--probabilities-key", default="probabilities")
    parser.add_argument("--labels-key", default="labels")
    parser.add_argument("--label-column", default=None)
    parser.add_argument("--probability-columns", nargs="+", default=None)
    parser.add_argument(
        "--task",
        choices=("auto", "multiclass", "multilabel"),
        default="auto",
    )
    parser.add_argument(
        "--class-names",
        nargs="+",
        default=None,
        help="Display names in probability-column order.",
    )
    parser.add_argument(
        "--no-per-class",
        action="store_true",
        help="Deprecated compatibility option; ROC/PR plots are macro-average only.",
    )
    parser.add_argument("--title", default=None)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--output", type=Path, required=True)


def _build_parser() -> argparse.ArgumentParser:
    """Xây dựng parser cho bước xử lý hiện tại.

    Returns
    -------
    argparse.ArgumentParser
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    latex = subparsers.add_parser("latex", help="Generate LaTeX table code.")
    _add_shared_input(latex)
    latex.add_argument("--columns", nargs="+", required=True)
    latex.add_argument(
        "--value-columns",
        nargs="+",
        default=None,
        help="Numeric columns to format and maximize; inferred when omitted.",
    )
    latex.add_argument(
        "--std-map",
        action="append",
        default=[],
        metavar="MEAN=STD",
        help="Pair a displayed mean column with its standard-deviation column.",
    )
    latex.add_argument(
        "--column-label",
        action="append",
        default=[],
        metavar="COLUMN=LATEX",
        help="Override a column heading; LaTeX commands are preserved.",
    )
    latex.add_argument(
        "--bold-within",
        nargs="+",
        default=None,
        help="Restart best-value selection within these grouping columns.",
    )
    latex.add_argument(
        "--minimize-columns",
        nargs="+",
        default=None,
        help="Value columns where smaller values are better.",
    )
    latex.add_argument("--precision", type=int, default=4)
    latex.add_argument("--caption", default=None)
    latex.add_argument("--label", default=None)
    latex.add_argument("--position", default="htbp")
    latex.add_argument("--output", type=Path, default=None)
    latex.add_argument("--tabular-only", action="store_true")
    latex.add_argument(
        "--resize-to-textwidth",
        action="store_true",
        help="Wrap the tabular in a resizebox constrained to text width.",
    )
    latex.add_argument(
        "--no-bold-best",
        action="store_true",
        help="Format numeric values without bolding column-wise best values.",
    )
    latex.add_argument(
        "--multirow",
        nargs="+",
        default=None,
        metavar="COLUMN",
        help="Merge contiguous values in leading columns; defaults to dataset.",
    )
    latex.add_argument(
        "--no-multirow",
        action="store_true",
        help="Repeat dataset values instead of generating LaTeX multirow cells.",
    )

    bar = subparsers.add_parser("bar", help="Generate a grouped bar chart.")
    _add_shared_input(bar)
    bar.add_argument("--x", required=True)
    bar.add_argument("--y", required=True)
    bar.add_argument("--hue", default=None)
    bar.add_argument("--error", default=None)
    bar.add_argument("--highlight", default=None)
    bar.add_argument("--title", default=None)
    bar.add_argument("--x-label", default=None)
    bar.add_argument("--y-label", default=None)
    bar.add_argument("--annotate", action="store_true")
    bar.add_argument("--precision", type=int, default=3)
    bar.add_argument("--dpi", type=int, default=300)
    bar.add_argument("--output", type=Path, required=True)

    line = subparsers.add_parser(
        "line",
        help="Generate a grouped line chart with optional error bars.",
    )
    _add_shared_input(line)
    line.add_argument("--x", required=True)
    line.add_argument("--y", required=True)
    line.add_argument("--hue", default=None)
    line.add_argument("--error", default=None)
    line.add_argument("--highlight", default=None)
    line.add_argument("--title", default=None)
    line.add_argument("--x-label", default=None)
    line.add_argument("--y-label", default=None)
    line.add_argument("--annotate", action="store_true")
    line.add_argument("--precision", type=int, default=3)
    line.add_argument("--dpi", type=int, default=300)
    line.add_argument("--output", type=Path, required=True)

    scatter = subparsers.add_parser(
        "scatter",
        help="Generate an annotated scatter plot for trade-off analyses.",
    )
    _add_shared_input(scatter)
    scatter.add_argument("--x", required=True)
    scatter.add_argument("--y", required=True)
    scatter.add_argument("--label", default=None)
    scatter.add_argument("--highlight", default=None)
    scatter.add_argument("--title", default=None)
    scatter.add_argument("--x-label", default=None)
    scatter.add_argument("--y-label", default=None)
    scatter.add_argument("--dpi", type=int, default=300)
    scatter.add_argument("--output", type=Path, required=True)

    heatmap = subparsers.add_parser("heatmap", help="Generate an annotated heatmap.")
    _add_shared_input(heatmap)
    heatmap.add_argument("--rows", required=True)
    heatmap.add_argument("--cols", required=True)
    heatmap.add_argument("--value", required=True)
    heatmap.add_argument("--title", default=None)
    heatmap.add_argument("--cmap", default="Blues")
    heatmap.add_argument("--x-label", default=None)
    heatmap.add_argument("--y-label", default=None)
    heatmap.add_argument("--colorbar-label", default=None)
    heatmap.add_argument("--precision", type=int, default=3)
    heatmap.add_argument("--dpi", type=int, default=300)
    heatmap.add_argument("--output", type=Path, required=True)

    calibration = subparsers.add_parser(
        "calibration",
        help="Generate a reliability diagram with ECE annotation.",
    )
    calibration.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_PROPOSED_EMBEDDINGS,
        help=(
            "NPZ or tabular predictions; defaults to the canonical proposed "
            "XBone-Net seed-42 embeddings."
        ),
    )
    calibration.add_argument("--probabilities-key", default="probabilities")
    calibration.add_argument("--labels-key", default="labels")
    calibration.add_argument("--label-column", default=None)
    calibration.add_argument("--probability-columns", nargs="+", default=None)
    calibration.add_argument("--n-bins", type=int, default=15)
    calibration.add_argument("--title", default=None)
    calibration.add_argument("--model-label", default="Model")
    calibration.add_argument("--ideal-label", default="Perfect calibration")
    calibration.add_argument("--x-label", default="Mean confidence")
    calibration.add_argument("--y-label", default="Observed accuracy")
    calibration.add_argument("--dpi", type=int, default=300)
    calibration.add_argument("--output", type=Path, required=True)

    roc = subparsers.add_parser(
        "roc",
        help="Generate a macro-average one-vs-rest ROC curve.",
    )
    _add_prediction_curve_args(roc)

    precision_recall = subparsers.add_parser(
        "pr",
        help="Generate a macro-average one-vs-rest precision--recall curve.",
    )
    _add_prediction_curve_args(precision_recall)

    aggregate_curves = subparsers.add_parser(
        "aggregate-curves",
        help="Generate mean ROC, PR, and calibration curves across seeds.",
    )
    aggregate_curves.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        required=True,
        help="Prediction NPZ files in seed order; each must contain probabilities and labels.",
    )
    aggregate_curves.add_argument("--output-dir", type=Path, required=True)
    aggregate_curves.add_argument("--prefix", default="aggregate")
    aggregate_curves.add_argument(
        "--task",
        choices=("auto", "multiclass", "multilabel"),
        default="auto",
    )
    aggregate_curves.add_argument("--n-bins", type=int, default=15)
    aggregate_curves.add_argument("--title-prefix", default="Model")
    aggregate_curves.add_argument("--dpi", type=int, default=300)

    full_shot_comparison = subparsers.add_parser(
        "full-shot-comparison",
        help=(
            "Compare XBone-Net with LoRA-BiomedCLIP and LoRA-PubMedCLIP "
            "using the latest three-seed full-shot summary."
        ),
    )
    full_shot_comparison.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_RESULTS_ROOT / "summary" / "run_all_table.csv",
    )
    full_shot_comparison.add_argument(
        "--datasets",
        nargs="+",
        default=["BTXRD", "CTCH"],
    )
    full_shot_comparison.add_argument(
        "--title-template",
        default="Phân loại full-shot trên {dataset}",
        help=(
            "Panel-title template; use {dataset} for the dataset name. "
            "Ignored for datasets overridden by --dataset-title."
        ),
    )
    full_shot_comparison.add_argument(
        "--dataset-title",
        action="append",
        default=[],
        metavar="DATASET=TITLE",
        help="Override one panel title; may be specified multiple times.",
    )
    full_shot_comparison.add_argument(
        "--figure-title",
        default=None,
        help="Optional title centered above the complete multi-panel figure.",
    )
    full_shot_comparison.add_argument("--dpi", type=int, default=300)
    full_shot_comparison.add_argument("--output", type=Path, required=True)

    full_shot_paired_forest = subparsers.add_parser(
        "full-shot-paired-forest",
        help=(
            "Plot paired seed-level confidence intervals for XBone-Net "
            "minus every trained full-shot baseline."
        ),
    )
    full_shot_paired_forest.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_RESULTS_ROOT / "summary" / "run_all_table.csv",
    )
    full_shot_paired_forest.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
    )
    full_shot_paired_forest.add_argument(
        "--datasets",
        nargs="+",
        default=["BTXRD", "CTCH"],
    )
    full_shot_paired_forest.add_argument(
        "--metric",
        choices=(
            "accuracy",
            "balanced_accuracy",
            "f1_macro",
            "auroc_macro",
            "auprc_macro",
        ),
        default="f1_macro",
    )
    full_shot_paired_forest.add_argument(
        "--confidence-level",
        type=float,
        default=0.95,
    )
    full_shot_paired_forest.add_argument("--dpi", type=int, default=300)
    full_shot_paired_forest.add_argument(
        "--csv-output",
        type=Path,
        default=None,
    )
    full_shot_paired_forest.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    aggregate_confusion = subparsers.add_parser(
        "aggregate-confusion",
        help="Pool and plot a multiclass confusion matrix across seeds.",
    )
    aggregate_confusion.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        required=True,
        help="Prediction NPZ files; each must contain probabilities and labels.",
    )
    aggregate_confusion.add_argument(
        "--class-names-file",
        type=Path,
        required=True,
        help="UTF-8 text file with one class name per line in class-ID order.",
    )
    aggregate_confusion.add_argument("--title", default=None)
    aggregate_confusion.add_argument("--x-label", default="Nhãn dự đoán")
    aggregate_confusion.add_argument("--y-label", default="Nhãn thực")
    aggregate_confusion.add_argument("--cmap", default="Blues")
    aggregate_confusion.add_argument("--precision", type=int, default=2)
    aggregate_confusion.add_argument("--dpi", type=int, default=300)
    aggregate_confusion.add_argument(
        "--no-normalize",
        action="store_true",
        help="Display pooled counts instead of row-normalized proportions.",
    )
    aggregate_confusion.add_argument(
        "--no-label-indices",
        action="store_true",
        help="Do not prefix class labels with their one-based display index.",
    )
    aggregate_confusion.add_argument("--output", type=Path, required=True)

    eff = subparsers.add_parser(
        "efficiency",
        help="Generate unified LaTeX summary tables from efficiency.json benchmark results.",
    )
    eff.add_argument(
        "--full-shot-output",
        type=Path,
        default=DEFAULT_RESULTS_ROOT / "summary" / "classification" / "table_efficiency_full_shot.tex",
    )
    eff.add_argument(
        "--few-shot-output",
        type=Path,
        default=DEFAULT_RESULTS_ROOT / "summary" / "classification" / "table_efficiency_few_shot.tex",
    )

    ood_cmd = subparsers.add_parser(
        "ood",
        help="Generate a LaTeX table from aggregated post-processing OOD results.",
    )
    ood_cmd.add_argument(
        "--input",
        type=Path,
        default=(
            DEFAULT_RESULTS_ROOT
            / "ctch/proposed/ours_xbone_net/analysis/aggregated_results.json"
        ),
    )
    ood_cmd.add_argument(
        "--scenarios",
        nargs="+",
        default=["semantic_ood", "domain_ood_btxrd"],
    )
    ood_cmd.add_argument(
        "--methods",
        nargs="+",
        default=[
            "cosine_centroids",
            "mahalanobis_centroid",
            "knn",
            "entropy",
        ],
    )
    ood_cmd.add_argument(
        "--csv-output",
        type=Path,
        default=DEFAULT_RESULTS_ROOT / "summary" / "ood" / "ood_2scenarios.csv",
    )
    ood_cmd.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RESULTS_ROOT / "summary" / "ood" / "table_ood_2scenarios.tex",
    )

    ood_metrics = subparsers.add_parser(
        "ood-metrics",
        help=(
            "Plot AUROC-OOD, AUPR-Out, and FPR@95%TPR for one OOD scoring "
            "method across selected scenarios."
        ),
    )
    ood_metrics.add_argument(
        "--input",
        type=Path,
        default=(
            DEFAULT_RESULTS_ROOT
            / "ctch/proposed/ours_xbone_net/analysis/aggregated_results.json"
        ),
    )
    ood_metrics.add_argument(
        "--scenarios",
        nargs="+",
        default=["semantic_ood", "domain_ood_btxrd"],
    )
    ood_metrics.add_argument(
        "--method",
        default="mahalanobis_centroid",
        choices=["cosine_centroids", "mahalanobis_centroid", "knn", "entropy"],
    )
    ood_metrics.add_argument("--title", default=None)
    ood_metrics.add_argument("--dpi", type=int, default=300)
    ood_metrics.add_argument(
        "--output",
        type=Path,
        default=(
            DEFAULT_RESULTS_ROOT
            / "summary/ood/mahalanobis_semantic_btxrd_metrics.png"
        ),
    )

    ood_heatmaps = subparsers.add_parser(
        "ood-heatmaps",
        help="Generate one heatmap per core OOD metric from aggregated results.",
    )
    ood_heatmaps.add_argument(
        "--input",
        type=Path,
        default=(
            DEFAULT_RESULTS_ROOT
            / "ctch/proposed/ours_xbone_net/analysis/aggregated_results.json"
        ),
    )
    ood_heatmaps.add_argument(
        "--scenarios",
        nargs="+",
        default=["semantic_ood", "domain_ood_btxrd"],
    )
    ood_heatmaps.add_argument(
        "--methods",
        nargs="+",
        default=[
            "cosine_centroids",
            "mahalanobis_centroid",
            "knn",
            "entropy",
        ],
    )
    ood_heatmaps.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_RESULTS_ROOT / "summary" / "ood" / "heatmaps",
    )
    ood_heatmaps.add_argument("--precision", type=int, default=3)
    ood_heatmaps.add_argument("--dpi", type=int, default=300)

    explainability_report = subparsers.add_parser(
        "explainability-report",
        help=(
            "Generate three-seed LaTeX tables and figures for attention, "
            "attribution, and representation analysis."
        ),
    )
    explainability_report.add_argument(
        "--aggregate-input",
        type=Path,
        default=(
            DEFAULT_RESULTS_ROOT
            / "ctch/proposed/ours_xbone_net/analysis/aggregated_results.json"
        ),
    )
    explainability_report.add_argument(
        "--experiment-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT / "ctch/proposed/ours_xbone_net",
    )
    explainability_report.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 123, 456],
    )
    explainability_report.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_RESULTS_ROOT / "summary" / "explainability",
    )
    explainability_report.add_argument("--dpi", type=int, default=300)

    ablation_report = subparsers.add_parser(
        "ablation-report",
        help=(
            "Generate an ordered CTCH ablation LaTeX table and paired "
            "Macro-F1 delta chart."
        ),
    )
    ablation_report.add_argument(
        "--ablation-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT / "ctch/ablation_study",
    )
    ablation_report.add_argument(
        "--reference-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT / "ctch/proposed/ours_xbone_net",
    )
    ablation_report.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 123, 456],
    )
    ablation_report.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_RESULTS_ROOT / "summary" / "ablation",
    )
    ablation_report.add_argument("--precision", type=int, default=3)
    ablation_report.add_argument("--dpi", type=int, default=300)

    leave_one_out = subparsers.add_parser(
        "ablation-leave-one-out",
        help=(
            "Generate the CTCH leave-one-component-out classification table "
            "from run_all_table.csv."
        ),
    )
    leave_one_out.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_RESULTS_ROOT / "summary" / "run_all_table.csv",
    )
    leave_one_out.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "docs"
            / "report"
            / "tables"
            / "chapter4"
            / "table_ablation_leave_one_out_classification.tex"
        ),
    )
    leave_one_out.add_argument("--precision", type=int, default=4)

    ablation_statistics = subparsers.add_parser(
        "ablation-statistics",
        help=(
            "Run paired crossed bootstrap and patient-level permutation tests "
            "for XBone-Net versus the leave-one-out variants."
        ),
    )
    ablation_statistics.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
    )
    ablation_statistics.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 123, 456],
    )
    ablation_statistics.add_argument(
        "--metrics",
        nargs="+",
        choices=(
            "accuracy",
            "balanced_accuracy",
            "f1_macro",
            "auroc_macro",
            "auprc_macro",
            "ece_15",
            "brier_score",
        ),
        default=["f1_macro", "balanced_accuracy", "accuracy"],
        help=(
            "Metrics included in the confirmatory analysis. Macro-F1 is the "
            "recommended primary endpoint."
        ),
    )
    ablation_statistics.add_argument("--n-bootstrap", type=int, default=10_000)
    ablation_statistics.add_argument("--n-permutations", type=int, default=10_000)
    ablation_statistics.add_argument(
        "--test-method",
        choices=("permutation", "bootstrap"),
        default="permutation",
        help=(
            "Use the crossed patient/seed permutation test, or a centered "
            "paired bootstrap test. The bootstrap test is much faster for "
            "Macro-AUROC and Macro-AUPRC."
        ),
    )
    ablation_statistics.add_argument("--alpha", type=float, default=0.05)
    ablation_statistics.add_argument("--random-seed", type=int, default=2026)
    ablation_statistics.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_RESULTS_ROOT / "summary" / "ablation" / "statistics",
    )
    ablation_statistics.add_argument("--dpi", type=int, default=300)

    ablation_forest = subparsers.add_parser(
        "ablation-forest",
        help="Draw a forest plot from paired leave-one-out statistics.",
    )
    ablation_forest.add_argument(
        "--input",
        type=Path,
        default=(
            DEFAULT_RESULTS_ROOT
            / "summary/ablation/statistics/paired_bootstrap_results.csv"
        ),
    )
    ablation_forest.add_argument(
        "--metric",
        choices=(
            "accuracy",
            "balanced_accuracy",
            "f1_macro",
            "auroc_macro",
            "auprc_macro",
            "ece_15",
        ),
        default="f1_macro",
    )
    ablation_forest.add_argument("--output", type=Path, required=True)
    ablation_forest.add_argument("--dpi", type=int, default=300)
    return parser


def _load_efficiency_json(exp_dir: str, seed: int = 42) -> dict | None:
    """Tải efficiency json cho bước xử lý hiện tại.

    Parameters
    ----------
    exp_dir : str
        Đường dẫn tài nguyên được sử dụng.
    seed : int, optional
        Hạt giống phục vụ khả năng tái lập.

    Returns
    -------
    dict | None
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    json_file = DEFAULT_RESULTS_ROOT / exp_dir / f"seed_{seed}" / "efficiency.json"
    if not json_file.is_file():
        return None
    with json_file.open(encoding="utf-8") as f:
        return json.load(f)


def generate_full_shot_efficiency_table(output_file: Path) -> None:
    """Sinh full shot efficiency table cho bước xử lý hiện tại.

    Parameters
    ----------
    output_file : Path
        Đường dẫn tài nguyên được sử dụng.
    """
    targets = [
        ("BTXRD", "btxrd/baselines/peft_finetuned/lora_biomedclip", "LoRA-BiomedCLIP"),
        ("BTXRD", "btxrd/baselines/peft_finetuned/lora_pubmedclip", "LoRA-PubMedCLIP"),
        ("BTXRD", "btxrd/proposed/ours_xbone_net", "XBone-Net"),
        ("CTCH", "ctch/baselines/peft_finetuned/lora_biomedclip", "LoRA-BiomedCLIP"),
        ("CTCH", "ctch/baselines/peft_finetuned/lora_pubmedclip", "LoRA-PubMedCLIP"),
        ("CTCH", "ctch/proposed/ours_xbone_net", "XBone-Net"),
    ]

    raw_rows = []
    for ds, exp, name in targets:
        data = _load_efficiency_json(exp)
        if not data:
            print(f"[WARN] Missing efficiency.json for {exp}")
            continue
        params_total = data["parameters"]["total"]
        params_trainable = data["parameters"]["trainable"]
        pct = data["parameters"].get("trainable_pct", (params_trainable / params_total) * 100)
        flops = data.get("supported_gflops_per_sample")
        gflops = f"{flops['mean']:.3f}" if flops and "mean" in flops else "--"
        latency = data.get("latency_ms_per_sample", {})
        lat_mean = f"{latency['mean']:.2f}" if "mean" in latency else "--"
        throughput = data.get("throughput_samples_per_second", {})
        tp = f"{throughput['mean']:.1f}" if "mean" in throughput else "--"
        memory = data.get("cuda_memory", {})
        peak_memory = (
            f"{memory['peak_allocated_mb']:.1f}"
            if "peak_allocated_mb" in memory
            else "--"
        )

        is_proposed = "XBone-Net" in name or "ours" in exp
        raw_rows.append({
            "dataset": ds,
            "model": name,
            "total_params": f"{params_total:,}",
            "trainable_params": f"{params_trainable:,}",
            "trainable_pct": f"{pct:.2f}\\%",
            "gflops": gflops,
            "latency_mean": lat_mean,
            "throughput": tp,
            "peak_memory": peak_memory,
            "is_proposed": is_proposed,
        })

    if not raw_rows:
        print("[ERROR] No full-shot efficiency data found. Please run efficiency.py benchmark first.")
        return

    # Thiết lập trạng thái và thống kê các tham số mô hình.
    lines_params = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\small",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{|C{1.50cm}|L{3.15cm}|C{2.75cm}|C{3.05cm}|C{1.55cm}|}",
        r"\hline",
        r"\multicolumn{1}{|C{1.50cm}|}{\textbf{Dữ liệu}} & \multicolumn{1}{C{3.15cm}|}{\textbf{Mô hình}} & \multicolumn{1}{C{2.75cm}|}{\textbf{Tổng tham số}} & \multicolumn{1}{C{3.05cm}|}{\textbf{Tham số huấn luyện}} & \multicolumn{1}{C{1.55cm}|}{\textbf{Tỷ lệ (\%)}} \\ \hline",
    ]

    ds_groups = {}
    for r in raw_rows:
        ds_groups.setdefault(r["dataset"], []).append(r)

    for ds, items in ds_groups.items():
        span = len(items)
        for i, item in enumerate(items):
            ds_prefix = (
                f"\\multirow[c]{{{span}}}{{=}}{{\\centering {ds}}}"
                if i == 0
                else ""
            )
            cell_bg = r"\cellcolor{gray!12}" if item["is_proposed"] else ""
            row_end = r" \\ \hline" if i == span - 1 else r" \\ \cline{2-5}"
            lines_params.append(
                f"{ds_prefix} & {cell_bg}{item['model']} & {cell_bg}{item['total_params']} & {cell_bg}{item['trainable_params']} & {cell_bg}{item['trainable_pct']}{row_end}"
            )

    lines_params.extend([
        r"\end{tabular}%",
        r"}",
        r"\caption{Đánh giá số lượng tham số của các mô hình khi sử dụng toàn bộ dữ liệu}",
        r"\label{tab:efficiency_full_shot_params}",
        r"\end{table}",
    ])

    # Thiết lập giá trị trung gian cho bước xử lý tiếp theo.
    lines_compute = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\small",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{|C{1.55cm}|L{2.55cm}|C{1.90cm}|C{1.85cm}|C{2.15cm}|C{2.05cm}|}",
        r"\hline",
        r"\multirow[c]{2}{=}{\centering\textbf{Dữ liệu}} & \multirow[c]{2}{=}{\centering\textbf{Mô hình}} & \multicolumn{1}{C{1.90cm}|}{\textbf{GFLOPs}} & \multicolumn{1}{C{1.85cm}|}{\textbf{Thông lượng}} & \multicolumn{1}{C{2.15cm}|}{\textbf{Độ trễ trung bình}} & \multicolumn{1}{C{2.05cm}|}{\textbf{Bộ nhớ GPU cực đại}} \\",
        r" & & \multicolumn{1}{C{1.90cm}|}{\textbf{\makecell{(GFLOP/\\mẫu)}}} & \multicolumn{1}{C{1.85cm}|}{\textbf{(mẫu/s)}} & \multicolumn{1}{C{2.15cm}|}{\textbf{(ms/mẫu)}} & \multicolumn{1}{C{2.05cm}|}{\textbf{(MiB)}} \\ \hline",
    ]

    for ds, items in ds_groups.items():
        span = len(items)
        for i, item in enumerate(items):
            ds_prefix = (
                f"\\multirow[c]{{{span}}}{{=}}{{\\centering {ds}}}"
                if i == 0
                else ""
            )
            cell_bg = r"\cellcolor{gray!12}" if item["is_proposed"] else ""
            row_end = r" \\ \hline" if i == span - 1 else r" \\ \cline{2-6}"
            lines_compute.append(
                f"{ds_prefix} & {cell_bg}{item['model']} & {cell_bg}{item['gflops']} & {cell_bg}{item['throughput']} & {cell_bg}{item['latency_mean']} & {cell_bg}{item['peak_memory']}{row_end}"
            )

    lines_compute.extend([
        r"\end{tabular}%",
        r"}",
        r"\caption{Đánh giá chi phí tính toán và tốc độ suy luận của các mô hình khi sử dụng toàn bộ dữ liệu}",
        r"\label{tab:efficiency_full_shot_compute}",
        r"\end{table}",
    ])

    content = "\n".join(lines_params) + "\n\n" + "\n".join(lines_compute) + "\n"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(content, encoding="utf-8")
    print(f"Full-shot efficiency tables (split into 2) saved to: {output_file}")


def generate_few_shot_efficiency_table(output_file: Path) -> None:
    """Sinh few shot efficiency table cho bước xử lý hiện tại.

    Parameters
    ----------
    output_file : Path
        Đường dẫn tài nguyên được sử dụng.
    """
    shots = ["1", "10", "20"]
    models = [
        ("lora_biomedclip", "LoRA-BiomedCLIP"),
        ("lora_pubmedclip", "LoRA-PubMedCLIP"),
        ("ours_xbone_net", "XBone-Net"),
    ]
    datasets = ["btxrd", "ctch"]

    raw_rows = []
    for ds_code in datasets:
        ds_display = ds_code.upper()
        for shot in shots:
            cat_display = f"{shot} mẫu"
            for m_code, m_display in models:
                exp = f"{ds_code}/few_shot/{shot}_shot/{m_code}"
                data = _load_efficiency_json(exp)
                if not data:
                    print(f"[WARN] Missing efficiency.json for {exp}")
                    continue
                params_total = data["parameters"]["total"]
                params_trainable = data["parameters"]["trainable"]
                pct = data["parameters"].get("trainable_pct", (params_trainable / params_total) * 100)
                flops = data.get("supported_gflops_per_sample")
                gflops = f"{flops['mean']:.3f}" if flops and "mean" in flops else "--"
                latency = data.get("latency_ms_per_sample", {})
                lat_mean = (
                    f"{latency['mean']:.2f}" if "mean" in latency else "--"
                )
                throughput = data.get("throughput_samples_per_second", {})
                tp = (
                    f"{throughput['mean']:.1f}"
                    if "mean" in throughput
                    else "--"
                )
                memory = data.get("cuda_memory", {})
                peak_memory = (
                    f"{memory['peak_allocated_mb']:.1f}"
                    if "peak_allocated_mb" in memory
                    else "--"
                )

                is_proposed = "XBone-Net" in m_display or "ours" in m_code

                raw_rows.append({
                    "dataset": ds_display,
                    "category": cat_display,
                    "model": m_display,
                    "total_params": f"{params_total:,}",
                    "trainable_params": f"{params_trainable:,}",
                    "trainable_pct": f"{pct:.2f}\\%",
                    "gflops": gflops,
                    "latency_mean": lat_mean,
                    "throughput": tp,
                    "peak_memory": peak_memory,
                    "is_proposed": is_proposed,
                })

    if not raw_rows:
        print("[ERROR] No few-shot efficiency data found. Please run efficiency.py benchmark first.")
        return

    # Thiết lập trạng thái và thống kê các tham số mô hình.
    lines_params = [
        r"\begin{table}[H]",
        r"\centering",
        r"\footnotesize",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{|C{1.30cm}|C{1.35cm}|L{2.50cm}|C{2.40cm}|C{2.65cm}|C{1.20cm}|}",
        r"\hline",
        r"\multicolumn{1}{|C{1.30cm}|}{\textbf{Dữ liệu}} & \multicolumn{1}{C{1.35cm}|}{\textbf{Kịch bản}} & \multicolumn{1}{C{2.50cm}|}{\textbf{Mô hình}} & \multicolumn{1}{C{2.40cm}|}{\textbf{Tổng tham số}} & \multicolumn{1}{C{2.65cm}|}{\textbf{Tham số huấn luyện}} & \multicolumn{1}{C{1.20cm}|}{\textbf{Tỷ lệ (\%)}} \\ \hline",
    ]

    ds_groups = {}
    for r in raw_rows:
        ds_groups.setdefault(r["dataset"], []).append(r)

    for ds, items in ds_groups.items():
        ds_span = len(items)
        cat_groups = {}
        for item in items:
            cat_groups.setdefault(item["category"], []).append(item)

        item_counter = 0
        for cat_idx, (cat, cat_items) in enumerate(cat_groups.items()):
            cat_span = len(cat_items)
            for cat_item_idx, item in enumerate(cat_items):
                ds_prefix = (
                    f"\\multirow[c]{{{ds_span}}}{{=}}{{\\centering {ds}}}"
                    if item_counter == 0
                    else ""
                )
                cat_prefix = (
                    f"\\multirow[c]{{{cat_span}}}{{=}}{{\\centering {cat}}}"
                    if cat_item_idx == 0
                    else ""
                )
                cell_bg = r"\cellcolor{gray!12}" if item["is_proposed"] else ""

                is_last_item_of_dataset = (item_counter == ds_span - 1)
                is_last_item_of_cat = (cat_item_idx == cat_span - 1)

                if is_last_item_of_dataset:
                    row_end = r" \\ \hline"
                elif is_last_item_of_cat:
                    row_end = r" \\ \cline{2-6}"
                else:
                    row_end = r" \\ \cline{3-6}"

                lines_params.append(
                    f"{ds_prefix} & {cat_prefix} & {cell_bg}{item['model']} & {cell_bg}{item['total_params']} & {cell_bg}{item['trainable_params']} & {cell_bg}{item['trainable_pct']}{row_end}"
                )
                item_counter += 1

    lines_params.extend([
        r"\end{tabular}%",
        r"}",
        r"\caption{Đánh giá số lượng tham số của các mô hình trong thiết lập mẫu học hạn chế}",
        r"\label{tab:efficiency_few_shot_params}",
        r"\end{table}",
    ])

    # Thiết lập giá trị trung gian cho bước xử lý tiếp theo.
    lines_compute = [
        r"\begin{table}[H]",
        r"\centering",
        r"\footnotesize",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{|C{1.35cm}|C{1.25cm}|L{2.20cm}|C{1.75cm}|C{1.45cm}|C{1.75cm}|C{1.65cm}|}",
        r"\hline",
        r"\multirow[c]{2}{=}{\centering\textbf{Dữ liệu}} & \multirow[c]{2}{=}{\centering\textbf{Kịch bản}} & \multirow[c]{2}{=}{\centering\textbf{Mô hình}} & \multicolumn{1}{C{1.75cm}|}{\textbf{GFLOPs}} & \multicolumn{1}{C{1.45cm}|}{\textbf{Thông lượng}} & \multicolumn{1}{C{1.75cm}|}{\textbf{Độ trễ trung bình}} & \multicolumn{1}{C{1.65cm}|}{\textbf{Bộ nhớ GPU cực đại}} \\",
        r" & & & \multicolumn{1}{C{1.75cm}|}{\textbf{\makecell{(GFLOP/\\mẫu)}}} & \multicolumn{1}{C{1.45cm}|}{\textbf{(mẫu/s)}} & \multicolumn{1}{C{1.75cm}|}{\textbf{(ms/mẫu)}} & \multicolumn{1}{C{1.65cm}|}{\textbf{(MiB)}} \\ \hline",
    ]

    for ds, items in ds_groups.items():
        ds_span = len(items)
        cat_groups = {}
        for item in items:
            cat_groups.setdefault(item["category"], []).append(item)

        item_counter = 0
        for cat_idx, (cat, cat_items) in enumerate(cat_groups.items()):
            cat_span = len(cat_items)
            for cat_item_idx, item in enumerate(cat_items):
                ds_prefix = (
                    f"\\multirow[c]{{{ds_span}}}{{=}}{{\\centering {ds}}}"
                    if item_counter == 0
                    else ""
                )
                cat_prefix = (
                    f"\\multirow[c]{{{cat_span}}}{{=}}{{\\centering {cat}}}"
                    if cat_item_idx == 0
                    else ""
                )
                cell_bg = r"\cellcolor{gray!12}" if item["is_proposed"] else ""

                is_last_item_of_dataset = (item_counter == ds_span - 1)
                is_last_item_of_cat = (cat_item_idx == cat_span - 1)

                if is_last_item_of_dataset:
                    row_end = r" \\ \hline"
                elif is_last_item_of_cat:
                    row_end = r" \\ \cline{2-7}"
                else:
                    row_end = r" \\ \cline{3-7}"

                lines_compute.append(
                    f"{ds_prefix} & {cat_prefix} & {cell_bg}{item['model']} & {cell_bg}{item['gflops']} & {cell_bg}{item['throughput']} & {cell_bg}{item['latency_mean']} & {cell_bg}{item['peak_memory']}{row_end}"
                )
                item_counter += 1


    lines_compute.extend([
        r"\end{tabular}%",
        r"}",
        r"\caption{Đánh giá chi phí tính toán và tốc độ suy luận của các mô hình trong thiết lập mẫu học hạn chế}",
        r"\label{tab:efficiency_few_shot_compute}",
        r"\end{table}",
    ])

    content = "\n".join(lines_params) + "\n\n" + "\n".join(lines_compute) + "\n"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(content, encoding="utf-8")
    print(f"Few-shot efficiency tables (split into 2) saved to: {output_file}")


def _generate_ood_table_legacy(output_file: Path) -> None:
    """Sinh ood table legacy cho bước xử lý hiện tại.

    Parameters
    ----------
    output_file : Path
        Đường dẫn tài nguyên được sử dụng.
    """
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{|l|l|c|c|c|}",
        r"\hline",
        r"\multicolumn{1}{|c|}{\textbf{Kịch bản}} & \multicolumn{1}{c|}{\textbf{Phương pháp}} & \multicolumn{1}{c|}{\textbf{AUROC-OOD $\uparrow$}} & \multicolumn{1}{c|}{\textbf{AUPR-Out $\uparrow$}} & \multicolumn{1}{c|}{\textbf{FPR@95\%TPR $\downarrow$}} \\ \hline",
        r"\multirow{6}{*}{Semantic OOD}",
        r"& \textbf{Mahalanobis} & $\mathbf{0{.}6070\pm0{.}0123}$ & $\mathbf{0{.}5892\pm0{.}0145}$ & $\mathbf{0{.}8939\pm0{.}0572}$ \\ \cline{2-5}",
        r"& kNN         & $0{.}6046\pm0{.}0199$ & $0{.}5810\pm0{.}0182$ & $0{.}9015\pm0{.}0262$ \\ \cline{2-5}",
        r"& MSP         & $0{.}5672\pm0{.}0040$ & $0{.}5420\pm0{.}0085$ & $0{.}9242\pm0{.}0131$ \\ \cline{2-5}",
        r"& Entropy     & $0{.}5719\pm0{.}0072$ & $0{.}5488\pm0{.}0090$ & $0{.}9394\pm0{.}0347$ \\ \cline{2-5}",
        r"& Energy      & $0{.}5962\pm0{.}0232$ & $0{.}5735\pm0{.}0210$ & $0{.}9242\pm0{.}0131$ \\ \cline{2-5}",
        r"& Max-logit   & $0{.}5968\pm0{.}0235$ & $0{.}5740\pm0{.}0215$ & $0{.}9242\pm0{.}0131$ \\ \hline",
        r"\multirow{2}{*}{Cross-dataset OOD--BTXRD}",
        r"& \textbf{Mahalanobis} & $\mathbf{0{.}8429\pm0{.}0208}$ & $\mathbf{0{.}8240\pm0{.}0225}$ & $\mathbf{0{.}6533\pm0{.}0335}$ \\ \cline{2-5}",
        r"& kNN         & $0{.}7843\pm0{.}0140$ & $0{.}7610\pm0{.}0152$ & $0{.}7831\pm0{.}0104$ \\ \hline",
        r"\multirow{6}{*}{Bệnh sử không tương hợp khác lớp}",
        r"& Mahalanobis & $0{.}7106\pm0{.}0091$ & $0{.}6912\pm0{.}0105$ & $0{.}8032\pm0{.}0238$ \\ \cline{2-5}",
        r"& \textbf{kNN} & $\mathbf{0{.}7412\pm0{.}0119}$ & $\mathbf{0{.}7250\pm0{.}0130}$ & $\mathbf{0{.}7519\pm0{.}0312}$ \\ \cline{2-5}",
        r"& MSP         & $0{.}6350\pm0{.}0059$ & $0{.}6120\pm0{.}0070$ & $0{.}8974\pm0{.}0195$ \\ \cline{2-5}",
        r"& Entropy     & $0{.}6575\pm0{.}0129$ & $0{.}6340\pm0{.}0140$ & $0{.}8640\pm0{.}0220$ \\ \cline{2-5}",
        r"& Energy      & $0{.}7293\pm0{.}0101$ & $0{.}7105\pm0{.}0110$ & $0{.}8191\pm0{.}0113$ \\ \cline{2-5}",
        r"& Max-logit   & $0{.}7253\pm0{.}0122$ & $0{.}7080\pm0{.}0132$ & $0{.}8231\pm0{.}0242$ \\ \hline",
        r"\end{tabular}",
        r"\caption{Kết quả OOD hậu xử lý của XBone-Net trên CTCH; chữ đậm biểu thị kết quả tốt nhất trong từng kịch bản chính}",
        r"\label{tab:ood_canonical_results}",
        r"\end{table}",
    ]
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"OOD summary table saved to: {output_file}")


OOD_SCENARIO_NAMES = {
    "semantic_ood": "OOD ngữ nghĩa",
    "domain_ood_btxrd": "BTXRD",
}

OOD_METHOD_NAMES = {
    "cosine_centroids": "Cosine theo tâm lớp",
    "mahalanobis_centroid": "Mahalanobis theo tâm lớp",
    "knn": "kNN",
    "entropy": "Entropy",
}

OOD_METRICS = {
    "auroc_ood": {
        "column": "auroc_ood_mean",
        "std_column": "auroc_ood_std",
        "label": r"AUROC-OOD $\uparrow$",
        "title": "AUROC-OOD trung bình trên 3 hạt giống",
        "colorbar": "AUROC-OOD",
        "cmap": "Blues",
        "filename": "ood_auroc_heatmap.png",
    },
    "aupr_out": {
        "column": "aupr_out_mean",
        "std_column": "aupr_out_std",
        "label": r"AUPR-Out $\uparrow$",
        "title": "AUPR-Out trung bình trên 3 hạt giống",
        "colorbar": "AUPR-Out",
        "cmap": "Blues",
        "filename": "ood_aupr_out_heatmap.png",
    },
    "fpr_at_95tpr": {
        "column": "fpr_at_95tpr_mean",
        "std_column": "fpr_at_95tpr_std",
        "label": r"FPR@95\%TPR $\downarrow$",
        "title": "FPR@95%TPR trung bình trên 3 hạt giống",
        "colorbar": "FPR@95%TPR",
        "cmap": "Reds",
        "filename": "ood_fpr95_heatmap.png",
    },
}


EXPLANATION_METRIC_NAMES = {
    "attention_entropy": "Attention Entropy chuẩn hóa",
    "deletion_auc": "Deletion AUC",
    "insertion_auc": "Insertion AUC",
}

REPORT_PALETTE = {
    "blue": "#1F77B4",
    "orange": "#FF7F0E",
    "green": "#2CA02C",
    "red": "#D62728",
    "purple": "#9467BD",
    "gray": "#7F7F7F",
    "cyan": "#17BECF",
}

ABLATION_VARIANTS = (
    {
        "group": "Đầu vào",
        "path": "modality/image_only",
        "label": "Chỉ ảnh",
    },
    {
        "group": "Đầu vào",
        "path": "modality/text_only",
        "label": "Chỉ bệnh sử",
    },
    {
        "group": "Đầu vào",
        "path": "modality/shuffled_report",
        "label": "Bệnh sử xáo trộn",
    },
    {
        "group": "Đầu vào",
        "path": "architecture/preprocess/direct_resize_current",
        "label": "Resize trực tiếp, không letterbox",
    },
    {
        "group": "Đầu vào",
        "path": "architecture/preprocess/xbone_letterbox",
        "label": "Letterbox",
    },
    {
        "group": "Đầu vào",
        "path": "architecture/preprocess/xbone_mean_pooling",
        "label": "Mean pooling",
    },
    {
        "group": "Đầu vào",
        "path": "architecture/preprocess/xbone_reduced_local_tokens",
        "label": "Giảm số token ảnh cục bộ",
    },
    {
        "group": "Huấn luyện",
        "path": "architecture/phase/phase1_merged",
        "label": "Huấn luyện gộp hai pha",
    },
    {
        "group": "Huấn luyện",
        "path": "architecture/phase/phase2_only",
        "label": "Chỉ huấn luyện pha 2",
    },
    {
        "group": "Huấn luyện",
        "path": "modality/phase1_xray_phase2_clinical",
        "label": "Bổ sung bệnh sử ở pha 2",
    },
    {
        "group": "Dung hợp",
        "path": "architecture/fusion/concat",
        "label": "Nối đặc trưng",
    },
    {
        "group": "Dung hợp",
        "path": "architecture/fusion/image_to_text",
        "label": "Ảnh truy vấn văn bản",
    },
    {
        "group": "Dung hợp",
        "path": "architecture/fusion/text_to_image",
        "label": "Văn bản truy vấn ảnh",
    },
    {
        "group": "Bộ phân loại",
        "path": "architecture/classifier/linear",
        "label": "Đầu tuyến tính",
    },
    {
        "group": "Bộ phân loại",
        "path": "architecture/classifier/no_class_weight",
        "label": "Không trọng số lớp",
    },
    {
        "group": "Bộ phân loại",
        "path": "architecture/classifier/no_class_bias",
        "label": "Không bias theo lớp",
    },
)

ABLATION_GROUP_COLORS = {
    "Đầu vào": REPORT_PALETTE["blue"],
    "Tinh chỉnh": REPORT_PALETTE["orange"],
    "Huấn luyện": REPORT_PALETTE["green"],
    "Dung hợp": REPORT_PALETTE["red"],
    "Bộ phân loại": REPORT_PALETTE["purple"],
}

ABLATION_METRICS = {
    "balanced_accuracy": r"Balanced Acc. $\uparrow$",
    "f1_macro": r"Macro-F1 $\uparrow$",
    "auroc_macro": r"Macro-AUROC $\uparrow$",
    "auprc_macro": r"Macro-AUPRC $\uparrow$",
}

LEAVE_ONE_OUT_CONFIGS = (
    {
        "experiment": "ctch/proposed/ours_xbone_net",
        "label": "XBone-Net",
        "components": (True, True, True, True, True),
    },
    {
        "experiment": (
            "ctch/ablation_study/architecture/preprocess/direct_resize_current"
        ),
        "label": "Không dùng letterbox",
        "components": (False, None, True, True, True),
    },
    {
        "experiment": (
            "ctch/ablation_study/architecture/preprocess/xbone_mean_pooling"
        ),
        "label": "Gộp trung bình",
        "components": (True, False, True, True, True),
    },
    {
        "experiment": "ctch/ablation_study/architecture/phase/phase2_only",
        "label": "Chỉ pha 2",
        "components": (True, True, False, True, True),
    },
    {
        "experiment": "ctch/ablation_study/architecture/fusion/concat",
        "label": "Nối đặc trưng",
        "components": (True, True, True, False, True),
    },
    {
        "experiment": "ctch/ablation_study/architecture/classifier/linear",
        "label": "Đầu tuyến tính",
        "components": (True, True, True, True, False),
    },
)

LEAVE_ONE_OUT_COMPONENT_LABELS = (
    "Nhánh ảnh độ phân giải cao",
    "Gộp chú ý cục bộ",
    "Huấn luyện pha 1",
    "Chú ý chéo hai chiều",
    "Tâm lớp thực nghiệm",
)

LEAVE_ONE_OUT_METRICS = (
    ("Accuracy", "accuracy_mean", "accuracy_std", False),
    (
        "Balanced Accuracy",
        "balanced_accuracy_mean",
        "balanced_accuracy_std",
        False,
    ),
    ("Macro-F1", "f1_macro_mean", "f1_macro_std", False),
    ("Macro-AUROC", "auroc_macro_mean", "auroc_macro_std", False),
    ("Macro-AUPRC", "auprc_macro_mean", "auprc_macro_std", False),
    ("ECE", "ece_15_mean", "ece_15_std", True),
)


def _leave_one_out_number(value: float, precision: int) -> str:
    """Thực hiện bước leave one out number trong quy trình hiện tại.

    Parameters
    ----------
    value : float
        Giá trị ``value`` được sử dụng trong phép xử lý.
    precision : int
        Giá trị ``precision`` được sử dụng trong phép xử lý.

    Returns
    -------
    str
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    return f"{float(value):.{precision}f}".replace(".", "{.}")


def generate_ablation_leave_one_out_table(
    input_file: str | Path,
    output_file: str | Path,
    *,
    precision: int = 4,
) -> Path:
    """Sinh ablation leave one out table cho bước xử lý hiện tại.

    Parameters
    ----------
    input_file : str | Path
        Đường dẫn tài nguyên được sử dụng.
    output_file : str | Path
        Đường dẫn tài nguyên được sử dụng.
    precision : int, optional
        Giá trị ``precision`` được sử dụng trong phép xử lý.

    Returns
    -------
    Path
        Kết quả được tạo bởi bước xử lý của hàm.

    Raises
    ------
    ValueError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    """
    frame = load_frame(input_file)
    required = {
        "experiment",
        "n_seeds",
        *(
            column
            for _, mean_column, std_column, _ in LEAVE_ONE_OUT_METRICS
            for column in (mean_column, std_column)
        ),
    }
    _require_columns(frame, sorted(required))

    indexed = frame.set_index("experiment", drop=False)
    ordered_rows: list[pd.Series] = []
    for specification in LEAVE_ONE_OUT_CONFIGS:
        experiment = str(specification["experiment"])
        if experiment not in indexed.index:
            raise ValueError(
                "Missing leave-one-out experiment in consolidated summary: "
                f"{experiment}"
            )
        selected = indexed.loc[experiment]
        if isinstance(selected, pd.DataFrame):
            raise ValueError(
                "Expected one consolidated row for leave-one-out experiment, "
                f"found {len(selected)}: {experiment}"
            )
        ordered_rows.append(selected)

    seed_counts = {
        int(_numeric_value(row["n_seeds"]))
        for row in ordered_rows
        if np.isfinite(_numeric_value(row["n_seeds"]))
    }
    if len(seed_counts) != 1:
        raise ValueError(
            "Leave-one-out experiments must use one common number of seeds; "
            f"received {sorted(seed_counts)}."
        )
    seed_count = next(iter(seed_counts))

    best_indices: dict[str, set[int]] = {}
    for _, mean_column, _, minimize in LEAVE_ONE_OUT_METRICS:
        values = np.asarray(
            [_numeric_value(row[mean_column]) for row in ordered_rows],
            dtype=np.float64,
        )
        if not np.isfinite(values).all():
            raise ValueError(
                f"Metric '{mean_column}' contains missing or invalid values."
            )
        optimum = float(np.min(values) if minimize else np.max(values))
        best_indices[mean_column] = {
            index
            for index, value in enumerate(values)
            if np.isclose(value, optimum, rtol=1e-9, atol=1e-12)
        }

    column_definition = (
        r"L{3.20cm}>{\columncolor{gray!12}}C{3.15cm}*{5}{C{3.15cm}}"
    )
    lines = [
        r"\begin{landscape}",
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{column_definition}}}",
        r"\toprule",
        r"\multicolumn{1}{C{3.20cm}}{\textbf{Thành phần hoặc độ đo}} &",
    ]
    header_cells = [
        (
            rf"\multicolumn{{1}}{{>{{\columncolor{{gray!12}}}}C{{3.15cm}}}}"
            rf"{{\textbf{{{_latex_escape(specification['label'])}}}}}"
            if index == 0
            else rf"\multicolumn{{1}}{{C{{3.15cm}}}}{{\textbf{{{_latex_escape(specification['label'])}}}}}"
        )
        for index, specification in enumerate(LEAVE_ONE_OUT_CONFIGS)
    ]
    lines.append(" &\n".join(header_cells) + r" \\")
    lines.append(r"\midrule")

    for component_index, component_label in enumerate(
        LEAVE_ONE_OUT_COMPONENT_LABELS
    ):
        cells = [_latex_escape(component_label)]
        for specification in LEAVE_ONE_OUT_CONFIGS:
            state = specification["components"][component_index]
            cells.append(
                r"$\checkmark$" if state is True else "--" if state is None else ""
            )
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\midrule")
    for metric_label, mean_column, std_column, minimize in LEAVE_ONE_OUT_METRICS:
        direction = r"\downarrow" if minimize else r"\uparrow"
        cells = [rf"{metric_label} ${direction}$"]
        for index, row in enumerate(ordered_rows):
            mean = _numeric_value(row[mean_column])
            std = _numeric_value(row[std_column])
            if not np.isfinite(mean) or not np.isfinite(std):
                raise ValueError(
                    f"Invalid mean/std for '{mean_column}' in "
                    f"{row['experiment']}."
                )
            body = (
                f"{_leave_one_out_number(mean, precision)}"
                rf"\pm{_leave_one_out_number(std, precision)}"
            )
            if index in best_indices[mean_column]:
                body = rf"\mathbf{{{body}}}"
            cells.append(rf"${body}$")
        lines.append(" &\n".join(cells) + r" \\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            (
                r"\caption{Nghiên cứu loại bỏ từng thành phần của XBone-Net "
                r"trên CTCH. Kết quả được trình bày "
                rf"dưới dạng trung bình $\pm$ độ lệch chuẩn trên {seed_count} hạt "
                r"giống; chữ đậm biểu thị kết quả tốt nhất theo từng độ đo.}"
            ),
            r"\label{tab:ablation_leave_one_out_classification}",
            r"\end{table}",
            r"\end{landscape}",
            "",
        ]
    )

    destination = resolve_path(output_file)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines), encoding="utf-8")
    return destination


PAIRED_STATISTIC_METRICS = {
    "accuracy": {
        "label": "Accuracy",
        "latex": r"Accuracy $\uparrow$",
        "higher_is_better": True,
    },
    "balanced_accuracy": {
        "label": "Balanced Accuracy",
        "latex": r"Balanced Accuracy $\uparrow$",
        "higher_is_better": True,
    },
    "f1_macro": {
        "label": "Macro-F1",
        "latex": r"Macro-F1 $\uparrow$",
        "higher_is_better": True,
    },
    "auroc_macro": {
        "label": "Macro-AUROC",
        "latex": r"Macro-AUROC $\uparrow$",
        "higher_is_better": True,
    },
    "auprc_macro": {
        "label": "Macro-AUPRC",
        "latex": r"Macro-AUPRC $\uparrow$",
        "higher_is_better": True,
    },
    "ece_15": {
        "label": "ECE",
        "latex": r"ECE $\downarrow$",
        "higher_is_better": False,
    },
    "brier_score": {
        "label": "Brier Score",
        "latex": r"Brier Score $\downarrow$",
        "higher_is_better": False,
    },
}


def _classification_metric_values(
    probabilities: np.ndarray,
    labels: np.ndarray,
    metrics: Sequence[str],
) -> dict[str, float]:
    """Thực hiện bước classification độ đo các giá trị trong quy trình hiện tại.

    Parameters
    ----------
    probabilities : np.ndarray
        Giá trị ``probabilities`` được sử dụng trong phép xử lý.
    labels : np.ndarray
        Giá trị ``labels`` được sử dụng trong phép xử lý.
    metrics : Sequence[str]
        Giá trị ``metrics`` được sử dụng trong phép xử lý.

    Returns
    -------
    dict[str, float]
        Kết quả được tạo bởi bước xử lý của hàm.

    Raises
    ------
    ValueError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if probabilities.ndim != 2 or len(probabilities) != len(labels):
        raise ValueError("Expected probabilities [N,C] and labels [N].")
    row_sums = probabilities.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("Probability rows must have positive sums.")
    probabilities = probabilities / row_sums
    predictions = probabilities.argmax(axis=1)
    class_order = np.arange(probabilities.shape[1])
    requested = set(metrics)
    values: dict[str, float] = {}

    if requested.intersection(("accuracy", "balanced_accuracy", "f1_macro")):
        num_classes = probabilities.shape[1]
        matrix = np.bincount(
            labels * num_classes + predictions,
            minlength=num_classes * num_classes,
        ).reshape(num_classes, num_classes)
        true_positive = np.diag(matrix).astype(np.float64)
        support = matrix.sum(axis=1).astype(np.float64)
        predicted = matrix.sum(axis=0).astype(np.float64)
        if "accuracy" in requested:
            values["accuracy"] = float(true_positive.sum() / matrix.sum())
        if "balanced_accuracy" in requested:
            recalls = np.divide(
                true_positive,
                support,
                out=np.zeros_like(true_positive),
                where=support > 0,
            )
            values["balanced_accuracy"] = float(recalls.mean())
        if "f1_macro" in requested:
            denominator = support + predicted
            per_class_f1 = np.divide(
                2.0 * true_positive,
                denominator,
                out=np.zeros_like(true_positive),
                where=denominator > 0,
            )
            values["f1_macro"] = float(per_class_f1.mean())

    if "auroc_macro" in requested or "auprc_macro" in requested:
        aurocs: list[float] = []
        auprcs: list[float] = []
        for class_id in class_order:
            binary_labels = (labels == class_id).astype(np.int64)
            if np.unique(binary_labels).size < 2:
                continue
            if "auroc_macro" in requested:
                aurocs.append(
                    float(
                        roc_auc_score(
                            binary_labels,
                            probabilities[:, class_id],
                        )
                    )
                )
            if "auprc_macro" in requested:
                auprcs.append(
                    float(
                        average_precision_score(
                            binary_labels,
                            probabilities[:, class_id],
                        )
                    )
                )
        if "auroc_macro" in requested:
            values["auroc_macro"] = (
                float(np.mean(aurocs)) if aurocs else float("nan")
            )
        if "auprc_macro" in requested:
            values["auprc_macro"] = (
                float(np.mean(auprcs)) if auprcs else float("nan")
            )

    if "ece_15" in requested:
        confidence = probabilities.max(axis=1)
        correct = (predictions == labels).astype(np.float64)
        edges = np.linspace(0.0, 1.0, 16)
        ece = 0.0
        for index in range(15):
            lower, upper = edges[index], edges[index + 1]
            mask = (confidence > lower) & (confidence <= upper)
            if index == 0:
                mask |= confidence == 0.0
            if np.any(mask):
                ece += float(mask.mean()) * abs(
                    float(correct[mask].mean())
                    - float(confidence[mask].mean())
                )
        values["ece_15"] = float(ece)
    if "brier_score" in requested:
        true_class_probability = probabilities[
            np.arange(len(labels)),
            labels,
        ]
        per_case_brier = (
            np.square(probabilities).sum(axis=1)
            - 2.0 * true_class_probability
            + 1.0
        )
        values["brier_score"] = float(per_case_brier.mean())
    return values


def _prepare_rank_metric_cache(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> list[dict[str, np.ndarray]]:
    """Chuẩn bị rank độ đo cache cho bước xử lý hiện tại.

    Parameters
    ----------
    probabilities : np.ndarray
        Giá trị ``probabilities`` được sử dụng trong phép xử lý.
    labels : np.ndarray
        Giá trị ``labels`` được sử dụng trong phép xử lý.

    Returns
    -------
    list[dict[str, np.ndarray]]
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    row_sums = probabilities.sum(axis=1, keepdims=True)
    probabilities = probabilities / np.clip(row_sums, 1e-12, None)
    cache: list[dict[str, np.ndarray]] = []
    for class_id in range(probabilities.shape[1]):
        scores = probabilities[:, class_id]
        order = np.argsort(-scores, kind="mergesort")
        sorted_scores = scores[order]
        group_end = np.empty(len(order), dtype=bool)
        group_end[:-1] = sorted_scores[:-1] != sorted_scores[1:]
        group_end[-1] = True
        cache.append(
            {
                "order": order,
                "group_end": np.flatnonzero(group_end),
                "positive": (labels[order] == class_id).astype(np.float64),
            }
        )
    return cache


def _weighted_rank_metrics(
    cache: Sequence[Mapping[str, np.ndarray]],
    sample_weight: np.ndarray,
    *,
    need_auroc: bool,
    need_auprc: bool,
) -> tuple[float, float]:
    """Thực hiện bước weighted rank các độ đo trong quy trình hiện tại.

    Parameters
    ----------
    cache : Sequence[Mapping[str, np.ndarray]]
        Giá trị ``cache`` được sử dụng trong phép xử lý.
    sample_weight : np.ndarray
        Giá trị ``sample_weight`` được sử dụng trong phép xử lý.
    need_auroc : bool
        Giá trị ``need_auroc`` được sử dụng trong phép xử lý.
    need_auprc : bool
        Giá trị ``need_auprc`` được sử dụng trong phép xử lý.

    Returns
    -------
    tuple[float, float]
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    aurocs: list[float] = []
    auprcs: list[float] = []
    weights = np.asarray(sample_weight, dtype=np.float64)
    for class_cache in cache:
        order = np.asarray(class_cache["order"], dtype=np.int64)
        ends = np.asarray(class_cache["group_end"], dtype=np.int64)
        positive = np.asarray(class_cache["positive"], dtype=np.float64)
        ordered_weight = weights[order]
        positive_weight = ordered_weight * positive
        negative_weight = ordered_weight * (1.0 - positive)
        total_positive = float(positive_weight.sum())
        total_negative = float(negative_weight.sum())
        if total_positive <= 0.0 or total_negative <= 0.0:
            continue
        cumulative_positive = np.cumsum(positive_weight)[ends]
        cumulative_negative = np.cumsum(negative_weight)[ends]
        if need_auroc:
            true_positive_rate = np.concatenate(
                ([0.0], cumulative_positive / total_positive)
            )
            false_positive_rate = np.concatenate(
                ([0.0], cumulative_negative / total_negative)
            )
            aurocs.append(float(np.trapezoid(true_positive_rate, false_positive_rate)))
        if need_auprc:
            previous_positive = np.concatenate(
                ([0.0], cumulative_positive[:-1])
            )
            recall_increment = (
                cumulative_positive - previous_positive
            ) / total_positive
            precision = cumulative_positive / np.clip(
                cumulative_positive + cumulative_negative,
                1e-12,
                None,
            )
            auprcs.append(float(np.sum(recall_increment * precision)))
    return (
        float(np.mean(aurocs)) if aurocs else float("nan"),
        float(np.mean(auprcs)) if auprcs else float("nan"),
    )


def _classification_metric_values_weighted(
    probabilities: np.ndarray,
    labels: np.ndarray,
    metrics: Sequence[str],
    sample_weight: np.ndarray,
    *,
    rank_cache: Sequence[Mapping[str, np.ndarray]] | None = None,
) -> dict[str, float]:
    """Thực hiện bước classification độ đo các giá trị weighted trong quy trình hiện tại.

    Parameters
    ----------
    probabilities : np.ndarray
        Giá trị ``probabilities`` được sử dụng trong phép xử lý.
    labels : np.ndarray
        Giá trị ``labels`` được sử dụng trong phép xử lý.
    metrics : Sequence[str]
        Giá trị ``metrics`` được sử dụng trong phép xử lý.
    sample_weight : np.ndarray
        Giá trị ``sample_weight`` được sử dụng trong phép xử lý.
    rank_cache : Sequence[Mapping[str, np.ndarray]] | None, optional
        Giá trị ``rank_cache`` được sử dụng trong phép xử lý.

    Returns
    -------
    dict[str, float]
        Kết quả được tạo bởi bước xử lý của hàm.

    Raises
    ------
    ValueError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    weights = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
    if len(weights) != len(labels):
        raise ValueError("Sample weights must match the number of labels.")
    row_sums = probabilities.sum(axis=1, keepdims=True)
    probabilities = probabilities / np.clip(row_sums, 1e-12, None)
    predictions = probabilities.argmax(axis=1)
    requested = set(metrics)
    values: dict[str, float] = {}
    num_classes = probabilities.shape[1]

    if requested.intersection(("accuracy", "balanced_accuracy", "f1_macro")):
        matrix = np.bincount(
            labels * num_classes + predictions,
            weights=weights,
            minlength=num_classes * num_classes,
        ).reshape(num_classes, num_classes)
        true_positive = np.diag(matrix)
        support = matrix.sum(axis=1)
        predicted = matrix.sum(axis=0)
        if "accuracy" in requested:
            values["accuracy"] = float(true_positive.sum() / matrix.sum())
        if "balanced_accuracy" in requested:
            values["balanced_accuracy"] = float(
                np.divide(
                    true_positive,
                    support,
                    out=np.zeros_like(true_positive),
                    where=support > 0,
                ).mean()
            )
        if "f1_macro" in requested:
            denominator = support + predicted
            values["f1_macro"] = float(
                np.divide(
                    2.0 * true_positive,
                    denominator,
                    out=np.zeros_like(true_positive),
                    where=denominator > 0,
                ).mean()
            )

    need_auroc = "auroc_macro" in requested
    need_auprc = "auprc_macro" in requested
    if need_auroc or need_auprc:
        if rank_cache is None:
            rank_cache = _prepare_rank_metric_cache(probabilities, labels)
        auroc, auprc = _weighted_rank_metrics(
            rank_cache,
            weights,
            need_auroc=need_auroc,
            need_auprc=need_auprc,
        )
        if need_auroc:
            values["auroc_macro"] = auroc
        if need_auprc:
            values["auprc_macro"] = auprc

    if "ece_15" in requested:
        confidence = probabilities.max(axis=1)
        correct = (predictions == labels).astype(np.float64)
        total_weight = float(weights.sum())
        edges = np.linspace(0.0, 1.0, 16)
        ece = 0.0
        for index in range(15):
            lower, upper = edges[index], edges[index + 1]
            mask = (confidence > lower) & (confidence <= upper)
            if index == 0:
                mask |= confidence == 0.0
            bin_weight = float(weights[mask].sum())
            if bin_weight > 0.0:
                bin_accuracy = float(np.average(correct[mask], weights=weights[mask]))
                bin_confidence = float(
                    np.average(confidence[mask], weights=weights[mask])
                )
                ece += (bin_weight / total_weight) * abs(
                    bin_accuracy - bin_confidence
                )
        values["ece_15"] = float(ece)
    if "brier_score" in requested:
        true_class_probability = probabilities[
            np.arange(len(labels)),
            labels,
        ]
        per_case_brier = (
            np.square(probabilities).sum(axis=1)
            - 2.0 * true_class_probability
            + 1.0
        )
        values["brier_score"] = float(
            np.average(per_case_brier, weights=weights)
        )
    return values


def _load_aligned_prediction_archive(
    source: Path,
    *,
    expected_image_ids: np.ndarray | None = None,
    expected_labels: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Tải aligned prediction archive cho bước xử lý hiện tại.

    Parameters
    ----------
    source : Path
        Dữ liệu nguồn của phép xử lý.
    expected_image_ids : np.ndarray | None, optional
        Ảnh hoặc biểu diễn ảnh đầu vào.
    expected_labels : np.ndarray | None, optional
        Giá trị ``expected_labels`` được sử dụng trong phép xử lý.

    Returns
    -------
    dict[str, np.ndarray]
        Kết quả được tạo bởi bước xử lý của hàm.

    Raises
    ------
    FileNotFoundError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    KeyError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    RuntimeError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    ValueError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    """
    if not source.is_file():
        raise FileNotFoundError(f"Missing prediction archive: {source}")
    required = ("probabilities", "labels", "image_id", "patient_id")
    with np.load(source, allow_pickle=False) as archive:
        missing = [key for key in required if key not in archive]
        if missing:
            raise KeyError(f"Missing keys in {source}: {missing}")
        data = {
            key: np.asarray(archive[key]).copy()
            for key in required
        }

    image_ids = data["image_id"].astype(str)
    if len(np.unique(image_ids)) != len(image_ids):
        raise ValueError(f"Image IDs are not unique in {source}.")

    if expected_image_ids is not None:
        expected = np.asarray(expected_image_ids).astype(str)
        lookup = {image_id: index for index, image_id in enumerate(image_ids)}
        missing_ids = [image_id for image_id in expected if image_id not in lookup]
        if missing_ids:
            raise ValueError(
                f"{source} is missing {len(missing_ids)} paired test cases."
            )
        order = np.asarray([lookup[image_id] for image_id in expected], dtype=int)
        data = {key: value[order] for key, value in data.items()}
        image_ids = data["image_id"].astype(str)
        if not np.array_equal(image_ids, expected):
            raise RuntimeError(f"Failed to align prediction archive: {source}")

    data["labels"] = np.asarray(data["labels"], dtype=np.int64).reshape(-1)
    data["probabilities"] = np.asarray(data["probabilities"], dtype=np.float64)
    data["image_id"] = image_ids
    data["patient_id"] = data["patient_id"].astype(str)
    if expected_labels is not None and not np.array_equal(
        data["labels"],
        np.asarray(expected_labels, dtype=np.int64).reshape(-1),
    ):
        raise ValueError(f"Ground-truth labels differ in paired archive: {source}")
    return data


def _leave_one_out_prediction_roots(
    results_root: str | Path,
) -> list[tuple[str, str, Path]]:
    """Thực hiện bước leave one out prediction roots trong quy trình hiện tại.

    Parameters
    ----------
    results_root : str | Path
        Đường dẫn tài nguyên được sử dụng.

    Returns
    -------
    list[tuple[str, str, Path]]
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    base = resolve_path(results_root)
    roots: list[tuple[str, str, Path]] = []
    for specification in LEAVE_ONE_OUT_CONFIGS:
        experiment = str(specification["experiment"])
        roots.append(
            (
                experiment,
                str(specification["label"]),
                base / experiment,
            )
        )
    return roots


def _load_leave_one_out_predictions(
    results_root: str | Path,
    *,
    seeds: Sequence[int],
) -> tuple[
    list[tuple[str, str, Path]],
    dict[str, dict[int, np.ndarray]],
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Tải leave one out predictions cho bước xử lý hiện tại.

    Parameters
    ----------
    results_root : str | Path
        Đường dẫn tài nguyên được sử dụng.
    seeds : Sequence[int]
        Giá trị ``seeds`` được sử dụng trong phép xử lý.

    Returns
    -------
    tuple[list[tuple[str, str, Path]], dict[str, dict[int, np.ndarray]], np.ndarray, np.ndarray, np.ndarray]
        Kết quả được tạo bởi bước xử lý của hàm.

    Raises
    ------
    ValueError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    """
    roots = _leave_one_out_prediction_roots(results_root)
    if not roots:
        raise ValueError("No leave-one-out configurations were defined.")
    seed_order = [int(seed) for seed in seeds]
    probabilities: dict[str, dict[int, np.ndarray]] = {}
    common_image_ids: np.ndarray | None = None
    common_labels: np.ndarray | None = None
    common_patient_ids: np.ndarray | None = None

    for experiment, _, root in roots:
        probabilities[experiment] = {}
        for seed in seed_order:
            source = root / f"seed_{seed}" / "analysis/features/ctch_test.npz"
            archive = _load_aligned_prediction_archive(
                source,
                expected_image_ids=common_image_ids,
                expected_labels=common_labels,
            )
            if common_image_ids is None:
                common_image_ids = archive["image_id"]
                common_labels = archive["labels"]
                common_patient_ids = archive["patient_id"]
            elif not np.array_equal(
                archive["patient_id"].astype(str),
                np.asarray(common_patient_ids).astype(str),
            ):
                raise ValueError(f"Patient IDs differ in paired archive: {source}")
            probabilities[experiment][seed] = archive["probabilities"]

    assert common_image_ids is not None
    assert common_labels is not None
    assert common_patient_ids is not None
    return (
        roots,
        probabilities,
        common_labels,
        common_image_ids,
        common_patient_ids,
    )


def _stratified_patient_groups(
    labels: np.ndarray,
    patient_ids: np.ndarray,
) -> tuple[list[np.ndarray], dict[int, np.ndarray]]:
    """Thực hiện bước stratified patient groups trong quy trình hiện tại.

    Parameters
    ----------
    labels : np.ndarray
        Giá trị ``labels`` được sử dụng trong phép xử lý.
    patient_ids : np.ndarray
        Giá trị ``patient_ids`` được sử dụng trong phép xử lý.

    Returns
    -------
    tuple[list[np.ndarray], dict[int, np.ndarray]]
        Kết quả được tạo bởi bước xử lý của hàm.

    Raises
    ------
    ValueError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    """
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    patient_ids = np.asarray(patient_ids).astype(str)
    unique_patients, inverse = np.unique(patient_ids, return_inverse=True)
    clusters = [np.flatnonzero(inverse == index) for index in range(len(unique_patients))]
    class_groups: dict[int, list[int]] = {}
    for group_index, indices in enumerate(clusters):
        group_labels = np.unique(labels[indices])
        if len(group_labels) != 1:
            raise ValueError(
                "Stratified patient bootstrap requires one ground-truth class "
                "per patient."
            )
        class_groups.setdefault(int(group_labels[0]), []).append(group_index)
    return clusters, {
        class_id: np.asarray(indices, dtype=np.int64)
        for class_id, indices in sorted(class_groups.items())
    }


def _sample_stratified_patient_indices(
    rng: np.random.Generator,
    clusters: Sequence[np.ndarray],
    class_groups: Mapping[int, np.ndarray],
) -> np.ndarray:
    """Lấy mẫu stratified patient indices cho bước xử lý hiện tại.

    Parameters
    ----------
    rng : np.random.Generator
        Giá trị ``rng`` được sử dụng trong phép xử lý.
    clusters : Sequence[np.ndarray]
        Giá trị ``clusters`` được sử dụng trong phép xử lý.
    class_groups : Mapping[int, np.ndarray]
        Nhãn hoặc chỉ số lớp liên quan.

    Returns
    -------
    np.ndarray
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    selected: list[np.ndarray] = []
    for group_indices in class_groups.values():
        sampled_groups = rng.choice(
            group_indices,
            size=len(group_indices),
            replace=True,
        )
        selected.extend(clusters[int(group)] for group in sampled_groups)
    return np.concatenate(selected)


def _percentile_interval(
    values: np.ndarray,
    *,
    alpha: float,
) -> tuple[float, float]:
    """Thực hiện bước percentile interval trong quy trình hiện tại.

    Parameters
    ----------
    values : np.ndarray
        Giá trị ``values`` được sử dụng trong phép xử lý.
    alpha : float
        Giá trị ``alpha`` được sử dụng trong phép xử lý.

    Returns
    -------
    tuple[float, float]
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    lower = 100.0 * alpha / 2.0
    upper = 100.0 * (1.0 - alpha / 2.0)
    return (
        float(np.percentile(values, lower)),
        float(np.percentile(values, upper)),
    )


def _paired_variant_statistics(
    reference: Mapping[int, np.ndarray],
    variant: Mapping[int, np.ndarray],
    *,
    labels: np.ndarray,
    patient_ids: np.ndarray,
    seeds: Sequence[int],
    metrics: Sequence[str],
    n_bootstrap: int,
    n_permutations: int,
    alpha: float,
    random_seed: int,
    test_method: str,
) -> list[dict[str, float]]:
    """Thực hiện bước paired variant statistics trong quy trình hiện tại.

    Parameters
    ----------
    reference : Mapping[int, np.ndarray]
        Giá trị ``reference`` được sử dụng trong phép xử lý.
    variant : Mapping[int, np.ndarray]
        Giá trị ``variant`` được sử dụng trong phép xử lý.
    labels : np.ndarray
        Giá trị ``labels`` được sử dụng trong phép xử lý.
    patient_ids : np.ndarray
        Giá trị ``patient_ids`` được sử dụng trong phép xử lý.
    seeds : Sequence[int]
        Giá trị ``seeds`` được sử dụng trong phép xử lý.
    metrics : Sequence[str]
        Giá trị ``metrics`` được sử dụng trong phép xử lý.
    n_bootstrap : int
        Giá trị ``n_bootstrap`` được sử dụng trong phép xử lý.
    n_permutations : int
        Giá trị ``n_permutations`` được sử dụng trong phép xử lý.
    alpha : float
        Giá trị ``alpha`` được sử dụng trong phép xử lý.
    random_seed : int
        Hạt giống phục vụ khả năng tái lập.
    test_method : str
        Phương pháp hoặc chế độ xử lý được chọn.

    Returns
    -------
    list[dict[str, float]]
        Kết quả được tạo bởi bước xử lý của hàm.

    Raises
    ------
    ValueError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    """
    seed_order = [int(seed) for seed in seeds]
    clusters, class_groups = _stratified_patient_groups(labels, patient_ids)
    _, patient_inverse = np.unique(
        np.asarray(patient_ids).astype(str),
        return_inverse=True,
    )
    point_reference = {
        seed: _classification_metric_values(
            reference[seed],
            labels,
            metrics,
        )
        for seed in seed_order
    }
    point_variant = {
        seed: _classification_metric_values(
            variant[seed],
            labels,
            metrics,
        )
        for seed in seed_order
    }
    point_ref_mean = {
        metric: float(np.mean([point_reference[seed][metric] for seed in seed_order]))
        for metric in metrics
    }
    point_var_mean = {
        metric: float(np.mean([point_variant[seed][metric] for seed in seed_order]))
        for metric in metrics
    }
    needs_rank_cache = bool(
        set(metrics).intersection(("auroc_macro", "auprc_macro"))
    )
    reference_rank_cache = {
        seed: (
            _prepare_rank_metric_cache(reference[seed], labels)
            if needs_rank_cache
            else None
        )
        for seed in seed_order
    }
    variant_rank_cache = {
        seed: (
            _prepare_rank_metric_cache(variant[seed], labels)
            if needs_rank_cache
            else None
        )
        for seed in seed_order
    }

    bootstrap_reference = {
        metric: np.empty(n_bootstrap, dtype=np.float64)
        for metric in metrics
    }
    bootstrap_variant = {
        metric: np.empty(n_bootstrap, dtype=np.float64)
        for metric in metrics
    }
    bootstrap_delta = {
        metric: np.empty(n_bootstrap, dtype=np.float64)
        for metric in metrics
    }
    rng = np.random.default_rng(random_seed)
    for iteration in range(n_bootstrap):
        indices = _sample_stratified_patient_indices(
            rng,
            clusters,
            class_groups,
        )
        sample_weight = np.bincount(
            indices,
            minlength=len(labels),
        ).astype(np.float64)
        sampled_seed_indices = rng.integers(
            0,
            len(seed_order),
            size=len(seed_order),
        )
        selected_seeds, counts = np.unique(
            sampled_seed_indices,
            return_counts=True,
        )
        ref_sum = {metric: 0.0 for metric in metrics}
        var_sum = {metric: 0.0 for metric in metrics}
        for seed_index, count in zip(selected_seeds, counts):
            seed = seed_order[int(seed_index)]
            ref_values = _classification_metric_values_weighted(
                reference[seed],
                labels,
                metrics,
                sample_weight,
                rank_cache=reference_rank_cache[seed],
            )
            var_values = _classification_metric_values_weighted(
                variant[seed],
                labels,
                metrics,
                sample_weight,
                rank_cache=variant_rank_cache[seed],
            )
            for metric in metrics:
                ref_sum[metric] += int(count) * ref_values[metric]
                var_sum[metric] += int(count) * var_values[metric]
        for metric in metrics:
            ref_value = ref_sum[metric] / len(seed_order)
            var_value = var_sum[metric] / len(seed_order)
            bootstrap_reference[metric][iteration] = ref_value
            bootstrap_variant[metric][iteration] = var_value
            bootstrap_delta[metric][iteration] = var_value - ref_value

    # Bước hỗ trợ để thực hiện xử lý ``paired_variant_statistics`` trong quy trình hiện tại.
    # Bước hỗ trợ để thực hiện xử lý ``paired_variant_statistics`` trong quy trình hiện tại.
    # Bước hỗ trợ để thực hiện xử lý ``paired_variant_statistics`` trong quy trình hiện tại.
    permutation_delta: dict[str, np.ndarray] = {}
    if test_method == "permutation":
        permutation_delta = {
            metric: np.empty(n_permutations, dtype=np.float64)
            for metric in metrics
        }
        rng = np.random.default_rng(random_seed + 1)
        for iteration in range(n_permutations):
            group_swap = rng.integers(0, 2, size=len(clusters)).astype(bool)
            swap = group_swap[patient_inverse]
            sampled_seed_indices = rng.integers(
                0,
                len(seed_order),
                size=len(seed_order),
            )
            selected_seeds, counts = np.unique(
                sampled_seed_indices,
                return_counts=True,
            )
            delta_sum = {metric: 0.0 for metric in metrics}
            for seed_index, count in zip(selected_seeds, counts):
                seed = seed_order[int(seed_index)]
                ref_probabilities = np.where(
                    swap[:, None],
                    variant[seed],
                    reference[seed],
                )
                var_probabilities = np.where(
                    swap[:, None],
                    reference[seed],
                    variant[seed],
                )
                ref_values = _classification_metric_values(
                    ref_probabilities,
                    labels,
                    metrics,
                )
                var_values = _classification_metric_values(
                    var_probabilities,
                    labels,
                    metrics,
                )
                for metric in metrics:
                    delta_sum[metric] += int(count) * (
                        var_values[metric] - ref_values[metric]
                    )
            for metric in metrics:
                permutation_delta[metric][iteration] = (
                    delta_sum[metric] / len(seed_order)
                )

    rows: list[dict[str, float]] = []
    for metric in metrics:
        reference_ci = _percentile_interval(
            bootstrap_reference[metric],
            alpha=alpha,
        )
        variant_ci = _percentile_interval(
            bootstrap_variant[metric],
            alpha=alpha,
        )
        delta_ci = _percentile_interval(
            bootstrap_delta[metric],
            alpha=alpha,
        )
        observed_delta = point_var_mean[metric] - point_ref_mean[metric]
        if test_method == "permutation":
            null_distribution = permutation_delta[metric]
            denominator = n_permutations + 1.0
        elif test_method == "bootstrap":
            null_distribution = bootstrap_delta[metric] - observed_delta
            denominator = n_bootstrap + 1.0
        else:
            raise ValueError(f"Unsupported test method: {test_method}")
        p_raw = (
            1.0
            + float(
                np.count_nonzero(
                    np.abs(null_distribution)
                    >= abs(observed_delta)
                )
            )
        ) / denominator
        higher_is_better = bool(
            PAIRED_STATISTIC_METRICS[metric]["higher_is_better"]
        )
        probability_better = float(
            np.mean(
                bootstrap_delta[metric] > 0
                if higher_is_better
                else bootstrap_delta[metric] < 0
            )
        )
        rows.append(
            {
                "metric": metric,
                "reference_mean": point_ref_mean[metric],
                "reference_ci_low": reference_ci[0],
                "reference_ci_high": reference_ci[1],
                "variant_mean": point_var_mean[metric],
                "variant_ci_low": variant_ci[0],
                "variant_ci_high": variant_ci[1],
                "delta_mean": observed_delta,
                "delta_ci_low": delta_ci[0],
                "delta_ci_high": delta_ci[1],
                "p_raw": p_raw,
                "test_method": test_method,
                "probability_variant_better": probability_better,
            }
        )
    return rows


def _format_p_value(value: float) -> str:
    """Định dạng p giá trị cho bước xử lý hiện tại.

    Parameters
    ----------
    value : float
        Giá trị ``value`` được sử dụng trong phép xử lý.

    Returns
    -------
    str
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    if value < 0.001:
        return r"$<0{.}001$"
    return f"${value:.3f}$".replace(".", "{.}")


def generate_paired_statistics_latex(
    frame: pd.DataFrame,
    output: str | Path,
    *,
    precision: int = 4,
) -> Path:
    """Sinh paired statistics latex cho bước xử lý hiện tại.

    Parameters
    ----------
    frame : pd.DataFrame
        Giá trị ``frame`` được sử dụng trong phép xử lý.
    output : str | Path
        Vị trí hoặc cấu trúc nhận kết quả.
    precision : int, optional
        Giá trị ``precision`` được sử dụng trong phép xử lý.

    Returns
    -------
    Path
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    required = (
        "variant",
        "metric",
        "reference_mean",
        "reference_ci_low",
        "reference_ci_high",
        "variant_mean",
        "variant_ci_low",
        "variant_ci_high",
        "delta_mean",
        "delta_ci_low",
        "delta_ci_high",
        "p_raw",
    )
    _require_columns(frame, required)

    def estimate(mean: float, lower: float, upper: float) -> str:
        """Ước lượng kết quả cho bước xử lý hiện tại.

        Parameters
        ----------
        mean : float
            Giá trị ``mean`` được sử dụng trong phép xử lý.
        lower : float
            Giá trị ``lower`` được sử dụng trong phép xử lý.
        upper : float
            Giá trị ``upper`` được sử dụng trong phép xử lý.

        Returns
        -------
        str
            Kết quả được tạo bởi bước xử lý của hàm.
        """
        return (
            f"${mean:.{precision}f}$ "
            f"$[{lower:.{precision}f};{upper:.{precision}f}]$"
        ).replace(".", "{.}")

    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\renewcommand{\arraystretch}{1.12}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        (
            r"\multicolumn{1}{c}{\textbf{Biến thể}} & "
            r"\multicolumn{1}{c}{\textbf{Độ đo}} & "
            r"\multicolumn{1}{c}{\textbf{XBone-Net [KTC 95\%]}} & "
            r"\multicolumn{1}{c}{\textbf{Biến thể [KTC 95\%]}} & "
            r"\multicolumn{1}{c}{\textbf{$\Delta$ [KTC 95\%]}} & "
            r"\multicolumn{1}{c}{\textbf{$p$}} \\"
        ),
        r"\midrule",
    ]
    variants = frame["variant"].drop_duplicates().tolist()
    for variant_index, variant in enumerate(variants):
        subset = frame[frame["variant"] == variant]
        for row_index, (_, row) in enumerate(subset.iterrows()):
            variant_cell = (
                rf"\multirow{{{len(subset)}}}{{*}}{{{_latex_escape(variant)}}}"
                if row_index == 0
                else ""
            )
            metric_label = str(
                PAIRED_STATISTIC_METRICS[str(row["metric"])]["latex"]
            )
            delta_text = estimate(
                float(row["delta_mean"]),
                float(row["delta_ci_low"]),
                float(row["delta_ci_high"]),
            )
            if float(row["p_raw"]) < 0.05:
                delta_text = rf"\textbf{{{delta_text}}}"
            lines.append(
                " & ".join(
                    (
                        variant_cell,
                        metric_label,
                        estimate(
                            float(row["reference_mean"]),
                            float(row["reference_ci_low"]),
                            float(row["reference_ci_high"]),
                        ),
                        estimate(
                            float(row["variant_mean"]),
                            float(row["variant_ci_low"]),
                            float(row["variant_ci_high"]),
                        ),
                        delta_text,
                        _format_p_value(float(row["p_raw"])),
                    )
                )
                + r" \\"
            )
        if variant_index < len(variants) - 1:
            lines.append(r"\midrule")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            (
                r"\caption{So sánh thống kê ghép cặp giữa XBone-Net và các "
                r"biến thể leave-one-out. Giá trị trong ngoặc vuông là khoảng "
                r"tin cậy bootstrap 95\%; $\Delta$ được tính bằng biến thể trừ "
                r"XBone-Net. Giá trị $p$ thô thu được từ phép kiểm định ghép "
                r"cặp và chưa được hiệu chỉnh cho nhiều phép so sánh.}"
            ),
            r"\label{tab:ablation_paired_statistics}",
            r"\end{table}",
            "",
        ]
    )
    destination = resolve_path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines), encoding="utf-8")
    return destination


def generate_paired_metric_latex(
    frame: pd.DataFrame,
    output: str | Path,
    *,
    metric: str,
    precision: int = 4,
) -> Path:
    """Sinh paired độ đo latex cho bước xử lý hiện tại.

    Parameters
    ----------
    frame : pd.DataFrame
        Giá trị ``frame`` được sử dụng trong phép xử lý.
    output : str | Path
        Vị trí hoặc cấu trúc nhận kết quả.
    metric : str
        Giá trị điểm hoặc độ đo cần sử dụng.
    precision : int, optional
        Giá trị ``precision`` được sử dụng trong phép xử lý.

    Returns
    -------
    Path
        Kết quả được tạo bởi bước xử lý của hàm.

    Raises
    ------
    ValueError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    """
    if metric not in PAIRED_STATISTIC_METRICS:
        raise ValueError(f"Unsupported table metric: {metric}")
    subset = frame[frame["metric"].astype(str) == metric].copy()
    if subset.empty:
        raise ValueError(f"No paired statistics are available for '{metric}'.")

    def estimate(mean: float, lower: float, upper: float) -> str:
        """Ước lượng kết quả cho bước xử lý hiện tại.

        Parameters
        ----------
        mean : float
            Giá trị ``mean`` được sử dụng trong phép xử lý.
        lower : float
            Giá trị ``lower`` được sử dụng trong phép xử lý.
        upper : float
            Giá trị ``upper`` được sử dụng trong phép xử lý.

        Returns
        -------
        str
            Kết quả được tạo bởi bước xử lý của hàm.
        """
        return (
            f"${mean:.{precision}f}$ "
            f"$[{lower:.{precision}f};{upper:.{precision}f}]$"
        ).replace(".", "{.}")

    label = str(PAIRED_STATISTIC_METRICS[metric]["latex"])
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.12}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        (
            r"\multicolumn{1}{c}{\textbf{Biến thể}} & "
            r"\multicolumn{1}{c}{\textbf{XBone-Net [KTC 95\%]}} & "
            r"\multicolumn{1}{c}{\textbf{Biến thể [KTC 95\%]}} & "
            r"\multicolumn{1}{c}{\textbf{$\Delta$ [KTC 95\%]}} & "
            r"\multicolumn{1}{c}{\textbf{$p$}} \\"
        ),
        r"\midrule",
    ]
    for _, row in subset.iterrows():
        delta = estimate(
            float(row["delta_mean"]),
            float(row["delta_ci_low"]),
            float(row["delta_ci_high"]),
        )
        if float(row["p_raw"]) < 0.05:
            delta = rf"\textbf{{{delta}}}"
        lines.append(
            " & ".join(
                (
                    _latex_escape(row["variant"]),
                    estimate(
                        float(row["reference_mean"]),
                        float(row["reference_ci_low"]),
                        float(row["reference_ci_high"]),
                    ),
                    estimate(
                        float(row["variant_mean"]),
                        float(row["variant_ci_low"]),
                        float(row["variant_ci_high"]),
                    ),
                    delta,
                    _format_p_value(float(row["p_raw"])),
                )
            )
            + r" \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            (
                rf"\caption{{So sánh ghép cặp {label} giữa XBone-Net và các "
                r"biến thể leave-one-out trên CTCH. Khoảng tin cậy 95\% được "
                r"ước lượng theo từng phép so sánh (pointwise) bằng bootstrap "
                r"phân tầng theo bệnh nhân và seed; "
                r"$\Delta$ được tính bằng biến thể trừ XBone-Net. Giá trị $p$ "
                r"là p-value thô và chưa được hiệu chỉnh cho nhiều phép so sánh.}"
            ),
            rf"\label{{tab:ablation_paired_{metric}}}",
            r"\end{table}",
            "",
        ]
    )
    destination = resolve_path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines), encoding="utf-8")
    return destination


def plot_ablation_forest(
    frame: pd.DataFrame,
    *,
    metric: str,
    output: str | Path,
    dpi: int = 300,
) -> Path:
    """Vẽ ablation forest cho bước xử lý hiện tại.

    Parameters
    ----------
    frame : pd.DataFrame
        Giá trị ``frame`` được sử dụng trong phép xử lý.
    metric : str
        Giá trị điểm hoặc độ đo cần sử dụng.
    output : str | Path
        Vị trí hoặc cấu trúc nhận kết quả.
    dpi : int, optional
        Giá trị ``dpi`` được sử dụng trong phép xử lý.

    Returns
    -------
    Path
        Kết quả được tạo bởi bước xử lý của hàm.

    Raises
    ------
    ValueError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    """
    if metric not in PAIRED_STATISTIC_METRICS:
        raise ValueError(f"Unsupported forest metric: {metric}")
    required = (
        "variant",
        "metric",
        "delta_mean",
        "delta_ci_low",
        "delta_ci_high",
        "p_raw",
    )
    _require_columns(frame, required)
    plot_data = frame[frame["metric"].astype(str) == metric].copy()
    if plot_data.empty:
        raise ValueError(f"No rows are available for metric '{metric}'.")

    plt = _load_pyplot()
    from matplotlib.lines import Line2D

    plot_data = plot_data.reset_index(drop=True)
    means = numeric_series(plot_data["delta_mean"]).to_numpy()
    lowers = numeric_series(plot_data["delta_ci_low"]).to_numpy()
    uppers = numeric_series(plot_data["delta_ci_high"]).to_numpy()
    significant = numeric_series(plot_data["p_raw"]).to_numpy() < 0.05
    higher_is_better = bool(
        PAIRED_STATISTIC_METRICS[metric]["higher_is_better"]
    )
    variant_better = means > 0 if higher_is_better else means < 0
    colors = [
        (
            REPORT_PALETTE["orange"]
            if better
            else REPORT_PALETTE["blue"]
        )
        if is_significant
        else REPORT_PALETTE["gray"]
        for better, is_significant in zip(variant_better, significant)
    ]
    positions = np.arange(len(plot_data))
    figure, axis = plt.subplots(figsize=(9.2, max(4.2, 0.65 * len(plot_data) + 1.7)))
    for position, mean, lower, upper, color in zip(
        positions,
        means,
        lowers,
        uppers,
        colors,
    ):
        axis.errorbar(
            mean,
            position,
            xerr=np.asarray([[mean - lower], [upper - mean]]),
            fmt="o",
            color=color,
            ecolor=color,
            markersize=6,
            elinewidth=1.8,
            capsize=4,
            zorder=3,
        )
    axis.axvline(0.0, color="#222222", linestyle="--", linewidth=1.1)
    variant_names = {
        "Direct resize": "Không dùng letterbox",
        "Mean pooling": "Gộp trung bình",
        "Concat": "Nối đặc trưng",
        "Linear head": "Đầu tuyến tính",
    }
    axis.set_yticks(
        positions,
        labels=[
            variant_names.get(value, value)
            for value in plot_data["variant"].astype(str)
        ],
    )
    axis.invert_yaxis()
    axis.grid(axis="x", color="#D9D9D9", linewidth=0.8, alpha=0.8)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    for position, upper, p_value in zip(
        positions,
        uppers,
        numeric_series(plot_data["p_raw"]).to_numpy(),
    ):
        axis.annotate(
            f"p={p_value:.3f}" if p_value >= 0.001 else "p<0.001",
            xy=(upper, position),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color="#333333",
        )

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            color=REPORT_PALETTE["blue"],
            label="XBone-Net tốt hơn, có ý nghĩa",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            color=REPORT_PALETTE["orange"],
            label="Biến thể tốt hơn, có ý nghĩa",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            color=REPORT_PALETTE["gray"],
            label=r"Chưa có ý nghĩa ($p\geq0{,}05$)",
        ),
    ]
    axis.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=1,
        frameon=False,
        fontsize=8.5,
    )
    extent = np.concatenate((lowers, uppers, [0.0]))
    span = float(np.nanmax(extent) - np.nanmin(extent))
    padding = max(0.005, 0.18 * span)
    axis.set_xlim(float(np.nanmin(extent) - padding), float(np.nanmax(extent) + padding))

    destination = _prepare_plot_output(output)
    figure.tight_layout()
    figure.savefig(destination, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return destination


def run_leave_one_out_statistical_analysis(
    results_root: str | Path,
    output_dir: str | Path,
    *,
    seeds: Sequence[int],
    metrics: Sequence[str],
    n_bootstrap: int = 10_000,
    n_permutations: int = 10_000,
    alpha: float = 0.05,
    random_seed: int = 2026,
    test_method: str = "permutation",
    dpi: int = 300,
) -> dict[str, Path]:
    """Thực hiện leave one out statistical analysis cho bước xử lý hiện tại.

    Parameters
    ----------
    results_root : str | Path
        Đường dẫn tài nguyên được sử dụng.
    output_dir : str | Path
        Đường dẫn tài nguyên được sử dụng.
    seeds : Sequence[int]
        Giá trị ``seeds`` được sử dụng trong phép xử lý.
    metrics : Sequence[str]
        Giá trị ``metrics`` được sử dụng trong phép xử lý.
    n_bootstrap : int, optional
        Giá trị ``n_bootstrap`` được sử dụng trong phép xử lý.
    n_permutations : int, optional
        Giá trị ``n_permutations`` được sử dụng trong phép xử lý.
    alpha : float, optional
        Giá trị ``alpha`` được sử dụng trong phép xử lý.
    random_seed : int, optional
        Hạt giống phục vụ khả năng tái lập.
    test_method : str, optional
        Phương pháp hoặc chế độ xử lý được chọn.
    dpi : int, optional
        Giá trị ``dpi`` được sử dụng trong phép xử lý.

    Returns
    -------
    dict[str, Path]
        Kết quả được tạo bởi bước xử lý của hàm.

    Raises
    ------
    ValueError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    """
    if not seeds:
        raise ValueError("At least one training seed is required.")
    if n_bootstrap < 1:
        raise ValueError("Bootstrap count must be positive.")
    if test_method == "permutation" and n_permutations < 1:
        raise ValueError("Permutation count must be positive.")
    if test_method not in {"permutation", "bootstrap"}:
        raise ValueError(f"Unsupported test method: {test_method}")
    if not 0.0 < alpha < 1.0:
        raise ValueError("Alpha must lie strictly between zero and one.")
    unknown = [metric for metric in metrics if metric not in PAIRED_STATISTIC_METRICS]
    if unknown:
        raise ValueError(f"Unsupported metrics: {unknown}")

    (
        roots,
        probabilities,
        labels,
        image_ids,
        patient_ids,
    ) = _load_leave_one_out_predictions(results_root, seeds=seeds)
    reference_experiment, reference_label, _ = roots[0]
    rows: list[dict[str, Any]] = []
    for variant_index, (experiment, label, _) in enumerate(roots[1:]):
        print(
            f"[{variant_index + 1}/{len(roots) - 1}] "
            f"Paired analysis: {label}"
        )
        variant_rows = _paired_variant_statistics(
            probabilities[reference_experiment],
            probabilities[experiment],
            labels=labels,
            patient_ids=patient_ids,
            seeds=seeds,
            metrics=metrics,
            n_bootstrap=n_bootstrap,
            n_permutations=n_permutations,
            alpha=alpha,
            random_seed=random_seed,
            test_method=test_method,
        )
        for row in variant_rows:
            row.update(
                {
                    "reference_experiment": reference_experiment,
                    "reference": reference_label,
                    "variant_experiment": experiment,
                    "variant": label,
                    "n_cases": len(labels),
                    "n_patients": len(np.unique(patient_ids)),
                    "n_seeds": len(seeds),
                    "seeds": "|".join(map(str, seeds)),
                    "n_bootstrap": n_bootstrap,
                    "n_permutations": n_permutations,
                    "test_method": test_method,
                    "alpha": alpha,
                }
            )
            rows.append(row)

    frame = pd.DataFrame(rows)
    frame["significant_raw"] = frame["p_raw"] < alpha
    frame["ci_excludes_zero"] = (
        (frame["delta_ci_low"] > 0.0)
        | (frame["delta_ci_high"] < 0.0)
    )

    destination_dir = resolve_path(output_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    csv_output = destination_dir / "paired_bootstrap_results.csv"
    frame.to_csv(csv_output, index=False, encoding="utf-8-sig")
    latex_output = generate_paired_statistics_latex(
        frame,
        destination_dir / "table_paired_statistics.tex",
    )
    protocol_output = destination_dir / "statistical_protocol.json"
    protocol = {
        "analysis": "paired_leave_one_out_classification",
        "reference": reference_experiment,
        "variants": [experiment for experiment, _, _ in roots[1:]],
        "metrics": list(metrics),
        "primary_metric": "f1_macro" if "f1_macro" in metrics else metrics[0],
        "seeds": list(map(int, seeds)),
        "n_cases": int(len(image_ids)),
        "n_patients": int(len(np.unique(patient_ids))),
        "bootstrap": {
            "type": "paired crossed percentile bootstrap",
            "patient_sampling": "class-stratified cluster resampling",
            "seed_sampling": "training seeds resampled with replacement",
            "same_resample_for_models": True,
            "n_resamples": int(n_bootstrap),
            "confidence_level": float(1.0 - alpha),
        },
        "test": {
            "type": (
                "two-sided crossed seed/patient permutation"
                if test_method == "permutation"
                else "two-sided centered paired bootstrap"
            ),
            "same_patient_swap_across_seeds": (
                True if test_method == "permutation" else None
            ),
            "seed_sampling": "training seeds resampled with replacement",
            "n_permutations": (
                int(n_permutations) if test_method == "permutation" else 0
            ),
            "multiplicity_correction": "none; report pointwise raw p-values",
            "alpha": float(alpha),
        },
        "random_seed": int(random_seed),
    }
    protocol_output.write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    outputs = {
        "csv": csv_output,
        "latex": latex_output,
        "protocol": protocol_output,
    }
    for metric in metrics:
        metric_table = generate_paired_metric_latex(
            frame,
            destination_dir / f"table_{metric}_statistics.tex",
            metric=metric,
        )
        outputs[f"table_{metric}"] = metric_table
        forest_output = plot_ablation_forest(
            frame,
            metric=metric,
            output=destination_dir / f"forest_{metric}.png",
            dpi=dpi,
        )
        outputs[f"forest_{metric}"] = forest_output
    return outputs


def _load_seed_metrics(
    experiment_root: str | Path,
    *,
    seeds: Sequence[int],
) -> dict[int, dict[str, float]]:
    """Tải seed các độ đo cho bước xử lý hiện tại.

    Parameters
    ----------
    experiment_root : str | Path
        Đường dẫn tài nguyên được sử dụng.
    seeds : Sequence[int]
        Giá trị ``seeds`` được sử dụng trong phép xử lý.

    Returns
    -------
    dict[int, dict[str, float]]
        Kết quả được tạo bởi bước xử lý của hàm.

    Raises
    ------
    FileNotFoundError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    KeyError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    ValueError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    """
    root = resolve_path(experiment_root)
    values: dict[int, dict[str, float]] = {}
    for seed in seeds:
        source = root / f"seed_{int(seed)}" / "metrics.json"
        if not source.is_file():
            raise FileNotFoundError(f"Missing seed metrics: {source}")
        payload = json.loads(source.read_text(encoding="utf-8"))
        metrics = payload.get("metrics")
        if not isinstance(metrics, Mapping):
            raise KeyError(f"Missing 'metrics' object in {source}")
        selected: dict[str, float] = {}
        for metric in ABLATION_METRICS:
            value = _numeric_value(metrics.get(metric))
            if not np.isfinite(value):
                raise ValueError(f"Metric '{metric}' is missing or invalid in {source}")
            selected[metric] = value
        values[int(seed)] = selected
    return values


def load_ablation_summary_frame(
    ablation_root: str | Path,
    reference_root: str | Path,
    *,
    seeds: Sequence[int] = (42, 123, 456),
) -> pd.DataFrame:
    """Tải ablation summary bảng dữ liệu cho bước xử lý hiện tại.

    Parameters
    ----------
    ablation_root : str | Path
        Đường dẫn tài nguyên được sử dụng.
    reference_root : str | Path
        Đường dẫn tài nguyên được sử dụng.
    seeds : Sequence[int], optional
        Giá trị ``seeds`` được sử dụng trong phép xử lý.

    Returns
    -------
    pd.DataFrame
        Kết quả được tạo bởi bước xử lý của hàm.

    Raises
    ------
    ValueError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    """
    seed_order = [int(seed) for seed in seeds]
    if not seed_order:
        raise ValueError("At least one seed is required.")
    reference_seed_metrics = _load_seed_metrics(reference_root, seeds=seed_order)
    ablation_base = resolve_path(ablation_root)

    specifications = (
        {
            "group": "Tham chiếu",
            "path": "proposed/ours_xbone_net",
            "label": "XBone-Net",
            "root": resolve_path(reference_root),
        },
        *(
            {
                **variant,
                "root": ablation_base / str(variant["path"]),
            }
            for variant in ABLATION_VARIANTS
        ),
    )

    rows: list[dict[str, Any]] = []
    for specification in specifications:
        seed_metrics = (
            reference_seed_metrics
            if specification["group"] == "Tham chiếu"
            else _load_seed_metrics(specification["root"], seeds=seed_order)
        )
        row: dict[str, Any] = {
            "group": specification["group"],
            "variant": specification["label"],
            "config_path": specification["path"],
            "n_seeds": len(seed_order),
        }
        for metric in ABLATION_METRICS:
            metric_values = np.asarray(
                [seed_metrics[seed][metric] for seed in seed_order],
                dtype=np.float64,
            )
            row[f"{metric}_mean"] = float(metric_values.mean())
            row[f"{metric}_std"] = (
                float(metric_values.std(ddof=1)) if len(metric_values) > 1 else 0.0
            )

        paired_deltas = np.asarray(
            [
                seed_metrics[seed]["f1_macro"]
                - reference_seed_metrics[seed]["f1_macro"]
                for seed in seed_order
            ],
            dtype=np.float64,
        )
        row["f1_macro_delta_mean"] = float(paired_deltas.mean())
        row["f1_macro_delta_std"] = (
            float(paired_deltas.std(ddof=1)) if len(paired_deltas) > 1 else 0.0
        )
        rows.append(row)
    return pd.DataFrame(rows)


def plot_ablation_macro_f1_delta(
    frame: pd.DataFrame,
    *,
    output: str | Path,
    dpi: int = 300,
) -> Path:
    """Vẽ ablation macro f1 delta cho bước xử lý hiện tại.

    Parameters
    ----------
    frame : pd.DataFrame
        Giá trị ``frame`` được sử dụng trong phép xử lý.
    output : str | Path
        Vị trí hoặc cấu trúc nhận kết quả.
    dpi : int, optional
        Giá trị ``dpi`` được sử dụng trong phép xử lý.

    Returns
    -------
    Path
        Kết quả được tạo bởi bước xử lý của hàm.

    Raises
    ------
    ValueError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    """
    plt = _load_pyplot()
    from matplotlib.patches import Patch

    required = (
        "group",
        "variant",
        "f1_macro_delta_mean",
        "f1_macro_delta_std",
    )
    _require_columns(frame, required)
    plot_data = frame[frame["group"] != "Tham chiếu"].reset_index(drop=True)
    if plot_data.empty:
        raise ValueError("No ablation variants are available for plotting.")

    positions = np.arange(len(plot_data))
    colors = [
        ABLATION_GROUP_COLORS.get(str(group), REPORT_PALETTE["gray"])
        for group in plot_data["group"]
    ]
    means = numeric_series(plot_data["f1_macro_delta_mean"]).to_numpy()
    errors = numeric_series(plot_data["f1_macro_delta_std"]).to_numpy()

    figure_height = max(7.0, 0.34 * len(plot_data) + 1.5)
    figure, axis = plt.subplots(figsize=(11.5, figure_height))
    axis.barh(
        positions,
        means,
        xerr=errors,
        color=colors,
        alpha=0.88,
        edgecolor="none",
        error_kw={"ecolor": "#222222", "elinewidth": 1.2, "capsize": 3},
    )
    axis.set_yticks(positions, labels=plot_data["variant"].astype(str), fontsize=8.5)
    axis.invert_yaxis()
    axis.axvline(0.0, color="#333333", linewidth=1.0)
    axis.grid(axis="x", color="#D9D9D9", linewidth=0.8, alpha=0.75)
    axis.set_axisbelow(True)
    axis.set_xlabel(r"Chênh lệch Macro-F1 so với XBone-Net ($\Delta$)")
    axis.set_title("Ảnh hưởng của các biến thể thành phần trên CTCH", fontsize=12)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    groups = plot_data["group"].astype(str).tolist()
    for index in range(1, len(groups)):
        if groups[index] != groups[index - 1]:
            axis.axhline(index - 0.5, color="#BDBDBD", linewidth=0.8)

    legend_groups = list(dict.fromkeys(groups))
    handles = [
        Patch(
            facecolor=ABLATION_GROUP_COLORS[group],
            edgecolor="none",
            label=group,
        )
        for group in legend_groups
    ]
    axis.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.09),
        ncol=min(5, len(handles)),
        frameon=False,
        fontsize=8.5,
    )

    finite_extent = np.concatenate((means - errors, means + errors, [0.0]))
    lower = float(np.nanmin(finite_extent))
    upper = float(np.nanmax(finite_extent))
    padding = max(0.015, (upper - lower) * 0.08)
    axis.set_xlim(lower - padding, upper + padding)

    destination = _prepare_plot_output(output)
    figure.tight_layout()
    figure.savefig(destination, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return destination


def generate_ablation_report(
    ablation_root: str | Path,
    reference_root: str | Path,
    output_dir: str | Path,
    *,
    seeds: Sequence[int] = (42, 123, 456),
    precision: int = 3,
    dpi: int = 300,
) -> dict[str, Path]:
    """Sinh ablation báo cáo cho bước xử lý hiện tại.

    Parameters
    ----------
    ablation_root : str | Path
        Đường dẫn tài nguyên được sử dụng.
    reference_root : str | Path
        Đường dẫn tài nguyên được sử dụng.
    output_dir : str | Path
        Đường dẫn tài nguyên được sử dụng.
    seeds : Sequence[int], optional
        Giá trị ``seeds`` được sử dụng trong phép xử lý.
    precision : int, optional
        Giá trị ``precision`` được sử dụng trong phép xử lý.
    dpi : int, optional
        Giá trị ``dpi`` được sử dụng trong phép xử lý.

    Returns
    -------
    dict[str, Path]
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    destination_dir = resolve_path(output_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    frame = load_ablation_summary_frame(
        ablation_root,
        reference_root,
        seeds=seeds,
    )

    csv_output = destination_dir / "ablation_summary.csv"
    frame.to_csv(csv_output, index=False, encoding="utf-8-sig")

    columns = ["group", "variant"]
    value_columns: list[str] = []
    std_columns: dict[str, str] = {}
    column_labels = {
        "group": "Nhóm thành phần",
        "variant": "Biến thể",
    }
    for metric, label in ABLATION_METRICS.items():
        mean_column = f"{metric}_mean"
        columns.append(mean_column)
        value_columns.append(mean_column)
        std_columns[mean_column] = f"{metric}_std"
        column_labels[mean_column] = label

    latex = generate_latex_table(
        frame,
        columns,
        value_columns=value_columns,
        std_columns=std_columns,
        column_labels=column_labels,
        precision=precision,
        caption=(
            "Kết quả đánh giá thành phần trên CTCH, được báo cáo dưới dạng "
            f"trung bình $\\pm$ độ lệch chuẩn qua {len(seeds)} seed. "
            "XBone-Net là cấu "
            "hình tham chiếu."
        ),
        label="tab:ctch_ablation_variants",
        font_size=r"\footnotesize",
        multirow_columns=["group"],
        resize_to_textwidth=True,
    )
    table_output = destination_dir / "table_ablation_variants.tex"
    table_output.write_text(latex, encoding="utf-8")

    figure_output = plot_ablation_macro_f1_delta(
        frame,
        output=destination_dir / "ablation_macro_f1_delta.png",
        dpi=dpi,
    )
    return {
        "csv": csv_output,
        "table": table_output,
        "figure": figure_output,
    }


REPRESENTATION_NAMES = {
    "fused_embeddings": "Dung hợp (fused)",
    "visual_global_embeddings": "Ảnh toàn cục",
    "visual_patch_summary_embeddings": "Tóm tắt patch ảnh",
    "text_global_embeddings": "Văn bản toàn cục",
    "image_from_text_embeddings": "Image-from-text",
    "text_from_image_embeddings": "Text-from-image",
}

REPRESENTATION_METRICS = {
    "silhouette_cosine": {
        "column": "silhouette_mean",
        "std_column": "silhouette_std",
        "label": r"Silhouette $\uparrow$",
    },
    "davies_bouldin": {
        "column": "davies_bouldin_mean",
        "std_column": "davies_bouldin_std",
        "label": r"Davies--Bouldin $\downarrow$",
    },
    "kmeans_nmi": {
        "column": "nmi_mean",
        "std_column": "nmi_std",
        "label": r"NMI $\uparrow$",
    },
    "nearest_train_centroid_balanced_accuracy": {
        "column": "centroid_bacc_mean",
        "std_column": "centroid_bacc_std",
        "label": r"Centroid BAcc $\uparrow$",
    },
}


def _sample_mean_std(values: Sequence[float]) -> tuple[float, float]:
    """Lấy mẫu mean std cho bước xử lý hiện tại.

    Parameters
    ----------
    values : Sequence[float]
        Giá trị ``values`` được sử dụng trong phép xử lý.

    Returns
    -------
    tuple[float, float]
        Kết quả được tạo bởi bước xử lý của hàm.

    Raises
    ------
    ValueError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("Cannot aggregate an empty or non-finite value sequence.")
    return (
        float(array.mean()),
        float(array.std(ddof=1)) if array.size > 1 else 0.0,
    )


def load_explanation_mechanism_frame(
    experiment_root: str | Path,
    *,
    seeds: Sequence[int],
) -> pd.DataFrame:
    """Tải explanation mechanism bảng dữ liệu cho bước xử lý hiện tại.

    Parameters
    ----------
    experiment_root : str | Path
        Đường dẫn tài nguyên được sử dụng.
    seeds : Sequence[int]
        Giá trị ``seeds`` được sử dụng trong phép xử lý.

    Returns
    -------
    pd.DataFrame
        Kết quả được tạo bởi bước xử lý của hàm.

    Raises
    ------
    FileNotFoundError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    KeyError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    ValueError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    """
    root = resolve_path(experiment_root)
    per_seed: list[dict[str, float]] = []
    for seed in seeds:
        seed_root = root / f"seed_{int(seed)}" / "analysis"
        feature_path = seed_root / "features" / "ctch_test.npz"
        sample_path = seed_root / "explainability" / "samples.json"
        if not feature_path.is_file():
            raise FileNotFoundError(feature_path)
        if not sample_path.is_file():
            raise FileNotFoundError(sample_path)

        required = (
            "attention_text_entropy_normalized",
            "attention_visual_entropy_normalized",
        )
        attention_feature_path = feature_path
        with np.load(attention_feature_path, allow_pickle=False) as archive:
            missing = [key for key in required if key not in archive]
        if missing:
            comparison_path = (
                root.parent
                / f"comparison_seed_{int(seed)}"
                / "v3"
                / "features"
                / "ctch_test.npz"
            )
            if comparison_path.is_file():
                with np.load(comparison_path, allow_pickle=False) as archive:
                    comparison_missing = [
                        key for key in required if key not in archive
                    ]
                if not comparison_missing:
                    attention_feature_path = comparison_path
                    missing = []
        if missing:
            raise KeyError(
                f"Missing attention arrays in {feature_path}: {missing}"
            )

        with np.load(attention_feature_path, allow_pickle=False) as archive:
            text_entropy = float(
                np.asarray(archive["attention_text_entropy_normalized"]).mean()
            )
            visual_entropy = float(
                np.asarray(archive["attention_visual_entropy_normalized"]).mean()
            )

        records = json.loads(sample_path.read_text(encoding="utf-8"))
        if not isinstance(records, list) or not records:
            raise ValueError(f"No explainability sample records in: {sample_path}")

        def aggregate_curve_auc(
            record_key: str,
            curve_key: str,
        ) -> np.ndarray:
            """Tổng hợp curve auc cho bước xử lý hiện tại.

            Parameters
            ----------
            record_key : str
                Tên hoặc khóa định danh của giá trị.
            curve_key : str
                Tên hoặc khóa định danh của giá trị.

            Returns
            -------
            np.ndarray
                Kết quả được tạo bởi bước xử lý của hàm.

            Raises
            ------
            KeyError
                Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
            ValueError
                Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
            """
            values = []
            for record in records:
                curves = record.get(record_key)
                if not isinstance(curves, Mapping):
                    raise KeyError(
                        f"Missing {record_key} in explainability record."
                    )
                fractions = np.asarray(curves["fractions"], dtype=np.float64)
                probabilities = np.asarray(curves[curve_key], dtype=np.float64)
                if fractions.shape != probabilities.shape:
                    raise ValueError(
                        f"Invalid {record_key}.{curve_key} curve."
                    )
                values.append(float(np.trapezoid(probabilities, fractions)))
            return np.asarray(values, dtype=np.float64)

        visual_deletion_auc = aggregate_curve_auc(
            "curves",
            "delete_most_relevant",
        )
        visual_insertion_auc = aggregate_curve_auc(
            "curves",
            "insert_most_relevant",
        )
        text_deletion_auc = aggregate_curve_auc(
            "text_curves",
            "delete_most_relevant",
        )
        text_insertion_auc = aggregate_curve_auc(
            "text_curves",
            "insert_most_relevant",
        )
        per_seed.append(
            {
                "seed": float(seed),
                "attention_entropy_visual": visual_entropy,
                "attention_entropy_text": text_entropy,
                "deletion_auc_visual": float(visual_deletion_auc.mean()),
                "deletion_auc_text": float(text_deletion_auc.mean()),
                "insertion_auc_visual": float(visual_insertion_auc.mean()),
                "insertion_auc_text": float(text_insertion_auc.mean()),
            }
        )

    seed_frame = pd.DataFrame(per_seed)
    definitions = (
        ("attention_entropy", "Token ảnh", "attention_entropy_visual"),
        ("attention_entropy", "Token bệnh sử", "attention_entropy_text"),
        ("deletion_auc", "Token ảnh", "deletion_auc_visual"),
        ("deletion_auc", "Token bệnh sử", "deletion_auc_text"),
        ("insertion_auc", "Token ảnh", "insertion_auc_visual"),
        ("insertion_auc", "Token bệnh sử", "insertion_auc_text"),
    )
    rows = []
    for metric_key, component, column in definitions:
        mean, std = _sample_mean_std(seed_frame[column].tolist())
        rows.append(
            {
                "metric_key": metric_key,
                "metric": EXPLANATION_METRIC_NAMES[metric_key],
                "component": component,
                "mean": mean,
                "std": std,
                "num_seeds": len(seeds),
            }
        )
    return pd.DataFrame(rows)


def load_explanation_distribution_data(
    experiment_root: str | Path,
    *,
    seeds: Sequence[int],
) -> dict[str, np.ndarray]:
    """Tải explanation distribution data cho bước xử lý hiện tại.

    Parameters
    ----------
    experiment_root : str | Path
        Đường dẫn tài nguyên được sử dụng.
    seeds : Sequence[int]
        Giá trị ``seeds`` được sử dụng trong phép xử lý.

    Returns
    -------
    dict[str, np.ndarray]
        Kết quả được tạo bởi bước xử lý của hàm.

    Raises
    ------
    FileNotFoundError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    KeyError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    ValueError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    """
    root = resolve_path(experiment_root)
    pooled: dict[str, list[np.ndarray]] = {
        "attention_entropy_visual": [],
        "attention_entropy_text": [],
    }
    archive_keys = {
        "attention_entropy_visual": "attention_visual_entropy_normalized",
        "attention_entropy_text": "attention_text_entropy_normalized",
    }
    for seed in seeds:
        feature_path = (
            root
            / f"seed_{int(seed)}"
            / "analysis"
            / "features"
            / "ctch_test.npz"
        )
        if not feature_path.is_file():
            raise FileNotFoundError(feature_path)
        attention_feature_path = feature_path
        with np.load(attention_feature_path, allow_pickle=False) as archive:
            missing = [
                archive_key
                for archive_key in archive_keys.values()
                if archive_key not in archive
            ]
        if missing:
            comparison_path = (
                root.parent
                / f"comparison_seed_{int(seed)}"
                / "v3"
                / "features"
                / "ctch_test.npz"
            )
            if comparison_path.is_file():
                with np.load(comparison_path, allow_pickle=False) as archive:
                    comparison_missing = [
                        key for key in archive_keys.values() if key not in archive
                    ]
                if not comparison_missing:
                    attention_feature_path = comparison_path
                    missing = []
        if missing:
            raise KeyError(
                f"Missing attention arrays in {feature_path}: {missing}"
            )

        with np.load(attention_feature_path, allow_pickle=False) as archive:
            for output_key, archive_key in archive_keys.items():
                values = np.asarray(archive[archive_key], dtype=np.float64).reshape(-1)
                values = values[np.isfinite(values)]
                if values.size == 0:
                    raise ValueError(
                        f"Attention array '{archive_key}' is empty in {feature_path}."
                    )
                pooled[output_key].append(values)
    return {
        key: np.concatenate(seed_values)
        for key, seed_values in pooled.items()
    }


def load_representation_frame(
    aggregate_input: str | Path,
) -> pd.DataFrame:
    """Tải representation bảng dữ liệu cho bước xử lý hiện tại.

    Parameters
    ----------
    aggregate_input : str | Path
        Dữ liệu nguồn của phép xử lý.

    Returns
    -------
    pd.DataFrame
        Kết quả được tạo bởi bước xử lý của hàm.

    Raises
    ------
    KeyError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    ValueError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    """
    source = resolve_path(aggregate_input)
    payload = json.loads(source.read_text(encoding="utf-8"))
    representation = (
        payload.get("explainability", {}).get("representation_quality")
    )
    if not isinstance(representation, Mapping):
        raise ValueError(
            f"Missing explainability.representation_quality in: {source}"
        )

    rows: list[dict[str, Any]] = []
    for space_key, display_name in REPRESENTATION_NAMES.items():
        row: dict[str, Any] = {
            "space_key": space_key,
            "space": display_name,
        }
        for metric_key, metadata in REPRESENTATION_METRICS.items():
            result = representation.get(f"{space_key}.{metric_key}")
            if not isinstance(result, Mapping):
                raise KeyError(
                    f"Missing representation metric "
                    f"'{space_key}.{metric_key}' in {source}."
                )
            row[str(metadata["column"])] = float(result["mean"])
            row[str(metadata["std_column"])] = float(result["std"])
        rows.append(row)
    return pd.DataFrame(rows)


def plot_explanation_mechanism_summary(
    frame: pd.DataFrame,
    *,
    distributions: Mapping[str, np.ndarray],
    output: str | Path,
    dpi: int = 300,
) -> Path:
    """Vẽ explanation mechanism summary cho bước xử lý hiện tại.

    Parameters
    ----------
    frame : pd.DataFrame
        Giá trị ``frame`` được sử dụng trong phép xử lý.
    distributions : Mapping[str, np.ndarray]
        Giá trị ``distributions`` được sử dụng trong phép xử lý.
    output : str | Path
        Vị trí hoặc cấu trúc nhận kết quả.
    dpi : int, optional
        Giá trị ``dpi`` được sử dụng trong phép xử lý.

    Returns
    -------
    Path
        Kết quả được tạo bởi bước xử lý của hàm.

    Raises
    ------
    KeyError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    """
    plt = _load_pyplot()
    _require_columns(frame, ["metric_key", "component", "mean", "std"])
    required_distributions = (
        "attention_entropy_visual",
        "attention_entropy_text",
    )
    missing = [key for key in required_distributions if key not in distributions]
    if missing:
        raise KeyError(f"Missing explanation distributions: {missing}")

    figure, axes = plt.subplots(1, 2, figsize=(10.6, 4.2), squeeze=False)
    blue = REPORT_PALETTE["blue"]
    orange = REPORT_PALETTE["orange"]

    entropy_axis = axes[0, 0]
    entropy_values = [
        np.asarray(distributions["attention_entropy_visual"], dtype=float),
        np.asarray(distributions["attention_entropy_text"], dtype=float),
    ]
    violin = entropy_axis.violinplot(
        entropy_values,
        positions=[0, 1],
        widths=0.78,
        showmeans=False,
        showmedians=False,
        showextrema=True,
    )
    for body, color in zip(violin["bodies"], (blue, orange)):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.78)
    for part in ("cbars", "cmins", "cmaxes"):
        violin[part].set_color(blue)
        violin[part].set_linewidth(1.4)
    entropy_axis.set_xticks([0, 1])
    entropy_axis.set_xticklabels(["Token ảnh", "Token bệnh sử"])
    entropy_axis.set_ylim(0.0, 1.02)
    entropy_axis.set_ylabel("Attention entropy chuẩn hóa")
    entropy_axis.set_title("(a) Mức phân tán attention")

    auc_axis = axes[0, 1]
    modalities = ("Token ảnh", "Token bệnh sử")
    positions = np.arange(len(modalities), dtype=float)
    width = 0.34
    deletion = (
        frame[frame["metric_key"] == "deletion_auc"]
        .set_index("component")
        .loc[list(modalities)]
    )
    insertion = (
        frame[frame["metric_key"] == "insertion_auc"]
        .set_index("component")
        .loc[list(modalities)]
    )
    auc_axis.bar(
        positions - width / 2,
        deletion["mean"],
        yerr=deletion["std"],
        width=width,
        capsize=4,
        color=blue,
        label="Deletion AUC ↓",
    )
    auc_axis.bar(
        positions + width / 2,
        insertion["mean"],
        yerr=insertion["std"],
        width=width,
        capsize=4,
        color=orange,
        label="Insertion AUC ↑",
    )
    auc_axis.set_xticks(positions)
    auc_axis.set_xticklabels(modalities)
    auc_axis.set_ylim(0.0, 1.0)
    auc_axis.set_ylabel("Diện tích dưới đường cong")
    auc_axis.set_title("(b) Độ trung thực của attribution")
    auc_axis.legend(frameon=False, loc="best")

    for axis in axes.flat:
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.7)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    figure.tight_layout()
    destination = _prepare_plot_output(output)
    figure.savefig(destination, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return destination


def load_attention_intervention_summary(
    experiment_root: str | Path,
    *,
    seeds: Sequence[int],
) -> dict[str, Any]:
    """Tải attention intervention summary cho bước xử lý hiện tại.

    Parameters
    ----------
    experiment_root : str | Path
        Đường dẫn tài nguyên được sử dụng.
    seeds : Sequence[int]
        Giá trị ``seeds`` được sử dụng trong phép xử lý.

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
    root = resolve_path(experiment_root)
    modalities = {
        "visual": "curves",
        "text": "text_curves",
    }
    strategies = (
        "delete_most_relevant",
        "delete_random",
        "delete_least_relevant",
    )
    summaries: dict[str, Any] = {}
    for modality, record_key in modalities.items():
        seed_means: dict[str, list[np.ndarray]] = {
            strategy: [] for strategy in strategies
        }
        fractions_reference: np.ndarray | None = None
        for seed in seeds:
            sample_path = (
                root
                / f"seed_{int(seed)}"
                / "analysis"
                / "explainability"
                / "samples.json"
            )
            if not sample_path.is_file():
                raise FileNotFoundError(sample_path)
            records = json.loads(sample_path.read_text(encoding="utf-8"))
            if not isinstance(records, list) or not records:
                raise ValueError(f"No explainability sample records in: {sample_path}")

            per_strategy: dict[str, list[np.ndarray]] = {
                strategy: [] for strategy in strategies
            }
            for record in records:
                curves = record.get(record_key)
                if not isinstance(curves, Mapping):
                    continue
                fractions = np.asarray(curves.get("fractions"), dtype=np.float64)
                if fractions_reference is None:
                    fractions_reference = fractions
                elif (
                    fractions.shape != fractions_reference.shape
                    or not np.allclose(fractions, fractions_reference)
                ):
                    raise ValueError(
                        f"Inconsistent intervention fractions in {sample_path}."
                    )
                for strategy in strategies:
                    values = np.asarray(curves.get(strategy), dtype=np.float64)
                    if values.shape != fractions.shape:
                        raise ValueError(
                            f"Invalid '{record_key}.{strategy}' in {sample_path}."
                        )
                    per_strategy[strategy].append(values)

            for strategy in strategies:
                if not per_strategy[strategy]:
                    raise ValueError(
                        f"No '{record_key}.{strategy}' curves in {sample_path}."
                    )
                seed_means[strategy].append(
                    np.stack(per_strategy[strategy], axis=0).mean(axis=0)
                )

        if fractions_reference is None:
            raise ValueError(f"No intervention curves found under: {root}")
        summaries[modality] = {
            "fractions": fractions_reference,
            "strategies": {
                strategy: {
                    "mean": np.stack(values, axis=0).mean(axis=0),
                    "std": np.stack(values, axis=0).std(
                        axis=0,
                        ddof=1 if len(values) > 1 else 0,
                    ),
                }
                for strategy, values in seed_means.items()
            },
        }
    return summaries


def plot_attention_intervention_curves(
    summary: Mapping[str, Any],
    *,
    output: str | Path,
    dpi: int = 300,
) -> Path:
    """Vẽ attention intervention curves cho bước xử lý hiện tại.

    Parameters
    ----------
    summary : Mapping[str, Any]
        Giá trị ``summary`` được sử dụng trong phép xử lý.
    output : str | Path
        Vị trí hoặc cấu trúc nhận kết quả.
    dpi : int, optional
        Giá trị ``dpi`` được sử dụng trong phép xử lý.

    Returns
    -------
    Path
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    plt = _load_pyplot()
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), sharey=True)
    strategy_specs = (
        (
            "delete_most_relevant",
            "Quan trọng nhất",
            REPORT_PALETTE["blue"],
            "o",
        ),
        ("delete_random", "Ngẫu nhiên", REPORT_PALETTE["orange"], "s"),
        (
            "delete_least_relevant",
            "Ít quan trọng nhất",
            REPORT_PALETTE["green"],
            "^",
        ),
    )
    modality_specs = (
        ("visual", "Token ảnh"),
        ("text", "Token bệnh sử"),
    )
    extrema: list[float] = [0.0]
    for modality, _ in modality_specs:
        for strategy, _, _, _ in strategy_specs:
            result = summary[modality]["strategies"][strategy]
            extrema.extend(
                np.asarray(result["mean"]) - np.asarray(result["std"])
            )
            extrema.extend(
                np.asarray(result["mean"]) + np.asarray(result["std"])
            )
    lower = min(extrema)
    upper = max(extrema)
    padding = max(0.0015, 0.08 * max(upper - lower, 0.01))

    legend_handles = []
    for axis, (modality, title) in zip(axes, modality_specs):
        fractions = np.asarray(summary[modality]["fractions"]) * 100.0
        for strategy, label, color, marker in strategy_specs:
            result = summary[modality]["strategies"][strategy]
            mean = np.asarray(result["mean"], dtype=np.float64)
            std = np.asarray(result["std"], dtype=np.float64)
            line = axis.plot(
                fractions,
                mean,
                color=color,
                marker=marker,
                linewidth=1.8,
                markersize=5,
                label=label,
            )[0]
            axis.fill_between(
                fractions,
                mean - std,
                mean + std,
                color=color,
                alpha=0.14,
                linewidth=0.8,
            )
            if len(legend_handles) < len(strategy_specs):
                legend_handles.append(line)
        axis.set_title(title)
        axis.set_xlabel("Tỷ lệ token bị thay thế (%)")
        axis.set_ylim(lower - padding, upper + padding)
        axis.grid(color="#D9D9D9", linewidth=0.7, alpha=0.7)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[0].set_ylabel("Xác suất lớp mục tiêu")
    figure.legend(
        legend_handles,
        [spec[1] for spec in strategy_specs],
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    figure.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    destination = _prepare_plot_output(output)
    figure.savefig(destination, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return destination


def load_explainability_cases(
    experiment_root: str | Path,
    *,
    seed: int,
) -> list[dict[str, Any]]:
    """Tải explainability cases cho bước xử lý hiện tại.

    Parameters
    ----------
    experiment_root : str | Path
        Đường dẫn tài nguyên được sử dụng.
    seed : int
        Hạt giống phục vụ khả năng tái lập.

    Returns
    -------
    list[dict[str, Any]]
        Kết quả được tạo bởi bước xử lý của hàm.

    Raises
    ------
    FileNotFoundError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    ValueError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    """
    root = resolve_path(experiment_root)
    sample_path = (
        root
        / f"seed_{int(seed)}"
        / "analysis"
        / "explainability"
        / "samples.json"
    )
    if not sample_path.is_file():
        raise FileNotFoundError(sample_path)
    records = json.loads(sample_path.read_text(encoding="utf-8"))
    visualization_root = sample_path.parent / "visualizations"
    candidates = []
    for record in records:
        visualization = Path(str(record.get("visualization", "")))
        if not visualization.is_file():
            visualization = visualization_root / f"{Path(record['image_id']).stem}.png"
        if visualization.is_file():
            candidate = dict(record)
            candidate["_visualization_path"] = visualization
            candidates.append(candidate)
    selected = []
    for is_correct in (True, False):
        matches = [
            record
            for record in candidates
            if bool(record.get("correct")) is is_correct
        ]
        if not matches:
            raise ValueError(
                f"No visualized {'correct' if is_correct else 'incorrect'} "
                f"case found for seed {seed}."
            )
        selected.append(
            max(matches, key=lambda record: float(record.get("confidence", 0.0)))
        )
    return selected


def plot_explainability_cases(
    cases: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    output: str | Path,
    dpi: int = 300,
) -> Path:
    """Vẽ explainability cases cho bước xử lý hiện tại.

    Parameters
    ----------
    cases : Sequence[Mapping[str, Any]]
        Giá trị ``cases`` được sử dụng trong phép xử lý.
    seed : int
        Hạt giống phục vụ khả năng tái lập.
    output : str | Path
        Vị trí hoặc cấu trúc nhận kết quả.
    dpi : int, optional
        Giá trị ``dpi`` được sử dụng trong phép xử lý.

    Returns
    -------
    Path
        Kết quả được tạo bởi bước xử lý của hàm.

    Raises
    ------
    ValueError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    """
    if len(cases) != 2:
        raise ValueError("Exactly one correct and one incorrect case are required.")
    plt = _load_pyplot()
    figure = plt.figure(figsize=(14.0, 7.2))
    outer = figure.add_gridspec(
        2,
        2,
        left=0.04,
        right=0.985,
        bottom=0.13,
        top=0.82,
        height_ratios=(1.05, 1.0),
        hspace=0.30,
        wspace=0.12,
    )
    strategy_specs = (
        (
            "delete_most_relevant",
            "Xóa token quan trọng nhất",
            REPORT_PALETTE["blue"],
            "o",
        ),
        (
            "delete_random",
            "Xóa token ngẫu nhiên",
            REPORT_PALETTE["orange"],
            "s",
        ),
        (
            "insert_most_relevant",
            "Chèn token quan trọng nhất",
            REPORT_PALETTE["green"],
            "^",
        ),
    )
    case_titles = (
        "Dự đoán đúng, độ tin cậy cao",
        "Dự đoán sai, độ tin cậy cao",
    )
    legend_handles = []
    for column, (record, case_title) in enumerate(zip(cases, case_titles)):
        image_array = plt.imread(record["_visualization_path"])
        height = image_array.shape[0]
        width = image_array.shape[1]
        top = outer[0, column].subgridspec(1, 2, wspace=0.04)
        panel_crops = (
            image_array[
                int(0.07 * height) : int(0.51 * height),
                int(0.13 * width) : int(0.39 * width),
            ],
            image_array[
                int(0.07 * height) : int(0.51 * height),
                int(0.63 * width) : int(0.91 * width),
            ],
        )
        for panel_column, crop in enumerate(panel_crops):
            top_axis = figure.add_subplot(top[0, panel_column])
            top_axis.imshow(crop)
            top_axis.axis("off")
        center_x = 0.26 if column == 0 else 0.76
        figure.text(
            center_x,
            0.915,
            case_title,
            ha="center",
            va="center",
            fontsize=14,
        )
        figure.text(
            center_x,
            0.875,
            (
                f"{record['image_id']} | GT={record['ground_truth']} "
                f"Pred={record['prediction']} "
                f"({100.0 * float(record['confidence']):.1f}%)"
            ),
            ha="center",
            va="center",
            fontsize=9,
        )

        lower = outer[1, column].subgridspec(1, 2, wspace=0.24)
        for local_column, (record_key, title, xlabel) in enumerate(
            (
                ("curves", "Độ trung thực token ảnh", "Token ảnh bị can thiệp (%)"),
                (
                    "text_curves",
                    "Độ trung thực token bệnh sử",
                    "Token bệnh sử bị can thiệp (%)",
                ),
            )
        ):
            axis = figure.add_subplot(lower[0, local_column])
            curves = record[record_key]
            fractions = np.asarray(curves["fractions"], dtype=np.float64) * 100.0
            for strategy, label, color, marker in strategy_specs:
                line = axis.plot(
                    fractions,
                    curves[strategy],
                    color=color,
                    marker=marker,
                    linewidth=1.5,
                    markersize=4,
                    label=label,
                )[0]
                if len(legend_handles) < len(strategy_specs):
                    legend_handles.append(line)
            axis.set_ylim(0.0, 1.02)
            axis.set_xlabel(xlabel)
            axis.set_ylabel("Xác suất lớp mục tiêu")
            if record_key == "text_curves":
                tokens = ", ".join(
                    item["token"]
                    for item in record.get("top_clinical_tokens", [])[:5]
                )
                axis.set_title(f"{title}\nToken nổi bật: {tokens}", fontsize=9)
            else:
                axis.set_title(title, fontsize=9)
            axis.grid(color="#D9D9D9", linewidth=0.7, alpha=0.7)
            axis.set_axisbelow(True)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
    figure.legend(
        legend_handles,
        [spec[1] for spec in strategy_specs],
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.015),
    )
    figure.suptitle(f"Hai trường hợp định tính tại seed {seed}", y=0.99)
    destination = _prepare_plot_output(output)
    figure.savefig(destination, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return destination


def plot_representation_geometry_summary(
    frame: pd.DataFrame,
    *,
    output: str | Path,
    dpi: int = 300,
) -> Path:
    """Vẽ representation geometry summary cho bước xử lý hiện tại.

    Parameters
    ----------
    frame : pd.DataFrame
        Giá trị ``frame`` được sử dụng trong phép xử lý.
    output : str | Path
        Vị trí hoặc cấu trúc nhận kết quả.
    dpi : int, optional
        Giá trị ``dpi`` được sử dụng trong phép xử lý.

    Returns
    -------
    Path
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    plt = _load_pyplot()
    panel_specs = (
        ("silhouette_mean", "silhouette_std", "Silhouette ↑"),
        ("davies_bouldin_mean", "davies_bouldin_std", "Davies–Bouldin ↓"),
        ("nmi_mean", "nmi_std", "NMI ↑"),
        ("centroid_bacc_mean", "centroid_bacc_std", "Centroid Balanced Accuracy ↑"),
    )
    _require_columns(
        frame,
        [
            "space",
            *[column for mean, std, _ in panel_specs for column in (mean, std)],
        ],
    )
    figure, axes = plt.subplots(2, 2, figsize=(12.0, 8.4), squeeze=False)
    positions = np.arange(len(frame))
    for axis, (mean_column, std_column, title) in zip(axes.flat, panel_specs):
        values = frame[mean_column].to_numpy(dtype=float)
        errors = frame[std_column].to_numpy(dtype=float)
        for position, value, error, key in zip(
            positions,
            values,
            errors,
            frame["space_key"],
        ):
            color = "#1769D2" if key == "fused_embeddings" else "#8C8C8C"
            axis.errorbar(
                value,
                position,
                xerr=error,
                fmt="o",
                markersize=8 if key == "fused_embeddings" else 6,
                capsize=4,
                color=color,
                linewidth=2.0 if key == "fused_embeddings" else 1.5,
            )
        axis.set_yticks(positions)
        axis.set_yticklabels(frame["space"])
        axis.invert_yaxis()
        axis.set_title(title)
        axis.grid(axis="x", color="#D9D9D9", linewidth=0.7, alpha=0.75)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        low = float(np.min(values - errors))
        high = float(np.max(values + errors))
        padding = max(0.03 * max(abs(low), abs(high), 1.0), 0.12 * (high - low))
        axis.set_xlim(low - padding, high + 2.2 * padding)
        if low < 0.0 < high:
            axis.axvline(0.0, color="#555555", linewidth=0.8)
        for position, value, error in zip(positions, values, errors):
            axis.annotate(
                f"{value:.3f}",
                (value + error, position),
                xytext=(5, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=8.5,
            )
    figure.tight_layout()
    destination = _prepare_plot_output(output)
    figure.savefig(destination, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return destination


def generate_explainability_report(
    aggregate_input: str | Path,
    experiment_root: str | Path,
    output_dir: str | Path,
    *,
    seeds: Sequence[int],
    dpi: int = 300,
) -> dict[str, Path]:
    """Sinh explainability báo cáo cho bước xử lý hiện tại.

    Parameters
    ----------
    aggregate_input : str | Path
        Dữ liệu nguồn của phép xử lý.
    experiment_root : str | Path
        Đường dẫn tài nguyên được sử dụng.
    output_dir : str | Path
        Đường dẫn tài nguyên được sử dụng.
    seeds : Sequence[int]
        Giá trị ``seeds`` được sử dụng trong phép xử lý.
    dpi : int, optional
        Giá trị ``dpi`` được sử dụng trong phép xử lý.

    Returns
    -------
    dict[str, Path]
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    explanation = load_explanation_mechanism_frame(
        experiment_root,
        seeds=seeds,
    )
    distributions = load_explanation_distribution_data(
        experiment_root,
        seeds=seeds,
    )
    intervention_summary = load_attention_intervention_summary(
        experiment_root,
        seeds=seeds,
    )
    case_seed = int(seeds[0])
    cases = load_explainability_cases(
        experiment_root,
        seed=case_seed,
    )
    representation = load_representation_frame(aggregate_input)
    destination_dir = resolve_path(output_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)

    explanation_csv = destination_dir / "explanation_mechanism_summary.csv"
    representation_csv = destination_dir / "representation_summary.csv"
    explanation.to_csv(explanation_csv, index=False, encoding="utf-8-sig")
    representation.to_csv(representation_csv, index=False, encoding="utf-8-sig")

    explanation_latex = generate_latex_table(
        explanation,
        ["metric", "component", "mean"],
        value_columns=["mean"],
        std_columns={"mean": "std"},
        column_labels={
            "metric": "Độ đo",
            "component": "Hướng hoặc thành phần",
            "mean": r"Trung bình $\pm$ độ lệch chuẩn",
        },
        precision=4,
        caption=(
            "Các độ đo attention và attribution đã được sử dụng trong nghiên cứu "
            "trước trên XBone-Net và CTCH; "
            "độ lệch chuẩn được tính giữa ba hạt giống."
        ),
        label="tab:explainability_mechanism_results",
        position="htbp",
        multirow_columns=["metric"],
        font_size=r"\footnotesize",
        resize_to_textwidth=False,
        bold_best=False,
    )
    explanation_table = destination_dir / "table_explanation_mechanism.tex"
    explanation_table.write_text(explanation_latex, encoding="utf-8")

    representation_metric_columns = [
        str(metadata["column"])
        for metadata in REPRESENTATION_METRICS.values()
    ]
    representation_latex = generate_latex_table(
        representation,
        ["space", *representation_metric_columns],
        value_columns=representation_metric_columns,
        std_columns={
            str(metadata["column"]): str(metadata["std_column"])
            for metadata in REPRESENTATION_METRICS.values()
        },
        column_labels={
            "space": "Không gian biểu diễn",
            **{
                str(metadata["column"]): str(metadata["label"])
                for metadata in REPRESENTATION_METRICS.values()
            },
        },
        minimize_columns=["davies_bouldin_mean"],
        precision=4,
        caption=(
            "Hình học của sáu không gian biểu diễn XBone-Net trên CTCH; "
            "chữ đậm biểu thị giá trị tốt nhất theo từng cột."
        ),
        label="tab:representation_geometry_results",
        position="htbp",
        multirow_columns=[],
        resize_to_textwidth=True,
    )
    representation_table = destination_dir / "table_representation_geometry.tex"
    representation_table.write_text(representation_latex, encoding="utf-8")

    mechanism_figure = plot_explanation_mechanism_summary(
        explanation,
        distributions=distributions,
        output=destination_dir / "explanation_mechanism_summary.png",
        dpi=dpi,
    )
    representation_figure = plot_representation_geometry_summary(
        representation,
        output=destination_dir / "representation_geometry_summary.png",
        dpi=dpi,
    )
    intervention_figure = plot_attention_intervention_curves(
        intervention_summary,
        output=destination_dir / "attention_intervention_curves.png",
        dpi=dpi,
    )
    cases_figure = plot_explainability_cases(
        cases,
        seed=case_seed,
        output=destination_dir / "attention_cases.png",
        dpi=dpi,
    )
    return {
        "explanation_table": explanation_table,
        "representation_table": representation_table,
        "mechanism_figure": mechanism_figure,
        "representation_figure": representation_figure,
        "intervention_figure": intervention_figure,
        "cases_figure": cases_figure,
        "explanation_csv": explanation_csv,
        "representation_csv": representation_csv,
    }


def load_aggregated_ood_frame(
    input_file: str | Path,
    *,
    scenarios: Sequence[str],
    methods: Sequence[str],
) -> pd.DataFrame:
    """Tải aggregated ood bảng dữ liệu cho bước xử lý hiện tại.

    Parameters
    ----------
    input_file : str | Path
        Đường dẫn tài nguyên được sử dụng.
    scenarios : Sequence[str]
        Giá trị ``scenarios`` được sử dụng trong phép xử lý.
    methods : Sequence[str]
        Giá trị ``methods`` được sử dụng trong phép xử lý.

    Returns
    -------
    pd.DataFrame
        Kết quả được tạo bởi bước xử lý của hàm.

    Raises
    ------
    KeyError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    ValueError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    """
    source = resolve_path(input_file)
    payload = json.loads(source.read_text(encoding="utf-8"))
    ood_payload = payload.get("ood")
    if not isinstance(ood_payload, Mapping):
        raise ValueError(f"Missing 'ood' object in aggregate result: {source}")

    rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        scenario_payload = ood_payload.get(scenario)
        if not isinstance(scenario_payload, Mapping):
            raise KeyError(f"OOD scenario '{scenario}' is missing from: {source}")
        aggregated = scenario_payload.get("aggregated")
        if not isinstance(aggregated, Mapping):
            raise KeyError(
                f"OOD scenario '{scenario}' has no aggregated metric object."
            )
        seeds = scenario_payload.get("seeds", [])
        for method in methods:
            row: dict[str, Any] = {
                "scenario": OOD_SCENARIO_NAMES.get(scenario, scenario),
                "scenario_key": scenario,
                "method": OOD_METHOD_NAMES.get(method, method),
                "method_key": method,
                "num_seeds": len(seeds),
            }
            for metric, metadata in OOD_METRICS.items():
                metric_payload = aggregated.get(f"{method}.{metric}")
                if not isinstance(metric_payload, Mapping):
                    raise KeyError(
                        f"Missing aggregate metric '{method}.{metric}' "
                        f"for scenario '{scenario}'."
                    )
                row[str(metadata["column"])] = float(metric_payload["mean"])
                row[str(metadata["std_column"])] = float(metric_payload["std"])
            rows.append(row)
    return pd.DataFrame(rows)


def generate_ood_table(
    input_file: str | Path,
    output_file: str | Path,
    *,
    scenarios: Sequence[str],
    methods: Sequence[str],
    csv_output: str | Path | None = None,
) -> Path:
    """Sinh ood table cho bước xử lý hiện tại.

    Parameters
    ----------
    input_file : str | Path
        Đường dẫn tài nguyên được sử dụng.
    output_file : str | Path
        Đường dẫn tài nguyên được sử dụng.
    scenarios : Sequence[str]
        Giá trị ``scenarios`` được sử dụng trong phép xử lý.
    methods : Sequence[str]
        Giá trị ``methods`` được sử dụng trong phép xử lý.
    csv_output : str | Path | None, optional
        Vị trí hoặc cấu trúc nhận kết quả.

    Returns
    -------
    Path
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    frame = load_aggregated_ood_frame(
        input_file,
        scenarios=scenarios,
        methods=methods,
    )
    metric_columns = [str(value["column"]) for value in OOD_METRICS.values()]
    std_columns = {
        str(value["column"]): str(value["std_column"])
        for value in OOD_METRICS.values()
    }
    column_labels = {
        "scenario": "Kịch bản",
        "method": "Phương pháp",
        **{
            str(value["column"]): str(value["label"])
            for value in OOD_METRICS.values()
        },
    }
    latex = generate_latex_table(
        frame,
        ["scenario", "method", *metric_columns],
        value_columns=metric_columns,
        std_columns=std_columns,
        column_labels=column_labels,
        bold_within=["scenario"],
        minimize_columns=["fpr_at_95tpr_mean"],
        precision=4,
        caption=(
            "Kết quả phát hiện OOD hậu xử lý của XBone-Net trên các kịch bản "
            "được xét; các giá trị được trình bày dưới dạng trung bình "
            "$\\pm$ độ lệch chuẩn trên ba hạt giống."
        ),
        label=(
            "tab:ood_semantic_btxrd"
            if list(scenarios) == ["semantic_ood", "domain_ood_btxrd"]
            else "tab:ood_selected_scenarios"
        ),
        position="H",
        font_size=r"\footnotesize",
        multirow_columns=["scenario"],
        resize_to_textwidth=True,
        column_widths={
            "scenario": "2.20cm",
            "method": "3.30cm",
            "auroc_ood_mean": "2.70cm",
            "aupr_out_mean": "2.70cm",
            "fpr_at_95tpr_mean": "2.80cm",
        },
    )
    destination = _prepare_plot_output(output_file)
    destination.write_text(latex, encoding="utf-8")
    if csv_output is not None:
        csv_destination = _prepare_plot_output(csv_output)
        frame.to_csv(csv_destination, index=False, encoding="utf-8-sig")
    return destination


def plot_ood_metric_summary(
    input_file: str | Path,
    output: str | Path,
    *,
    scenarios: Sequence[str],
    method: str = "mahalanobis_centroid",
    title: str | None = None,
    dpi: int = 300,
) -> Path:
    """Vẽ ood độ đo summary cho bước xử lý hiện tại.

    Parameters
    ----------
    input_file : str | Path
        Đường dẫn tài nguyên được sử dụng.
    output : str | Path
        Vị trí hoặc cấu trúc nhận kết quả.
    scenarios : Sequence[str]
        Giá trị ``scenarios`` được sử dụng trong phép xử lý.
    method : str, optional
        Phương pháp hoặc chế độ xử lý được chọn.
    title : str | None, optional
        Giá trị ``title`` được sử dụng trong phép xử lý.
    dpi : int, optional
        Giá trị ``dpi`` được sử dụng trong phép xử lý.

    Returns
    -------
    Path
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    plt = _load_pyplot()

    frame = load_aggregated_ood_frame(
        input_file,
        scenarios=scenarios,
        methods=[method],
    )
    metric_specs = (
        ("auroc_ood_mean", "auroc_ood_std", "AUROC-OOD", True),
        ("aupr_out_mean", "aupr_out_std", "AUPR-Out", True),
        ("fpr_at_95tpr_mean", "fpr_at_95tpr_std", "FPR@95%TPR", False),
    )
    scenario_colors = [
        REPORT_PALETTE["blue"],
        REPORT_PALETTE["orange"],
        REPORT_PALETTE["green"],
        REPORT_PALETTE["purple"],
    ]
    x = np.arange(len(frame), dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.1), sharey=True)
    for axis, (mean_col, std_col, metric_label, higher_is_better) in zip(
        axes,
        metric_specs,
    ):
        values = frame[mean_col].to_numpy(dtype=float)
        errors = frame[std_col].to_numpy(dtype=float)
        bars = axis.bar(
            x,
            values,
            yerr=errors,
            capsize=4,
            color=scenario_colors[: len(frame)],
            edgecolor="white",
            linewidth=0.8,
        )
        axis.set_xticks(x)
        axis.set_xticklabels(frame["scenario"].astype(str), rotation=0)
        axis.set_ylim(0.0, 1.05)
        axis.set_title(
            f"{metric_label} ({'cao hơn tốt hơn' if higher_is_better else 'thấp hơn tốt hơn'})"
        )
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
        for bar, value, error in zip(bars, values, errors):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                min(1.02, value + error + 0.025),
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    axes[0].set_ylabel("Giá trị trung bình trên ba hạt giống")
    method_name = OOD_METHOD_NAMES.get(method, method)
    fig.suptitle(title or f"Hiệu năng OOD của {method_name}", y=1.01)
    fig.tight_layout()
    destination = _prepare_plot_output(output)
    fig.savefig(destination, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return destination


def generate_ood_heatmaps(
    input_file: str | Path,
    output_dir: str | Path,
    *,
    scenarios: Sequence[str],
    methods: Sequence[str],
    precision: int = 3,
    dpi: int = 300,
) -> dict[str, Path]:
    """Sinh ood heatmaps cho bước xử lý hiện tại.

    Parameters
    ----------
    input_file : str | Path
        Đường dẫn tài nguyên được sử dụng.
    output_dir : str | Path
        Đường dẫn tài nguyên được sử dụng.
    scenarios : Sequence[str]
        Giá trị ``scenarios`` được sử dụng trong phép xử lý.
    methods : Sequence[str]
        Giá trị ``methods`` được sử dụng trong phép xử lý.
    precision : int, optional
        Giá trị ``precision`` được sử dụng trong phép xử lý.
    dpi : int, optional
        Giá trị ``dpi`` được sử dụng trong phép xử lý.

    Returns
    -------
    dict[str, Path]
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    frame = load_aggregated_ood_frame(
        input_file,
        scenarios=scenarios,
        methods=methods,
    )
    destination_dir = resolve_path(output_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    for metric, metadata in OOD_METRICS.items():
        outputs[metric] = plot_heatmap(
            frame,
            rows="method",
            columns="scenario",
            value=str(metadata["column"]),
            output=destination_dir / str(metadata["filename"]),
            title=str(metadata["title"]),
            cmap=str(metadata["cmap"]),
            x_label="Kịch bản OOD",
            y_label="Phương pháp tính điểm",
            colorbar_label=str(metadata["colorbar"]),
            precision=precision,
            dpi=dpi,
        )
    return outputs


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Kiểm tra tính hợp lệ của tham số cho bước xử lý hiện tại.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Giá trị ``parser`` được sử dụng trong phép xử lý.
    args : argparse.Namespace
        Các đối số vị trí bổ sung.
    """
    if hasattr(args, "precision") and args.precision < 0:
        parser.error("--precision cannot be negative.")
    if hasattr(args, "dpi") and args.dpi < 72:
        parser.error("--dpi must be at least 72.")
    if getattr(args, "n_bins", 1) < 1:
        parser.error("--n-bins must be positive.")
    if getattr(args, "n_bootstrap", 1) < 1:
        parser.error("--n-bootstrap must be positive.")
    if getattr(args, "n_permutations", 1) < 1:
        parser.error("--n-permutations must be positive.")
    alpha = getattr(args, "alpha", 0.05)
    if not 0.0 < alpha < 1.0:
        parser.error("--alpha must lie strictly between zero and one.")


def main() -> None:
    """Thực thi điểm vào chính của mô-đun."""
    parser = _build_parser()
    args = parser.parse_args()
    _validate_args(parser, args)

    if args.command == "efficiency":
        generate_full_shot_efficiency_table(args.full_shot_output)
        generate_few_shot_efficiency_table(args.few_shot_output)
        return

    if args.command == "ood":
        output = generate_ood_table(
            args.input,
            args.output,
            scenarios=args.scenarios,
            methods=args.methods,
            csv_output=args.csv_output,
        )
        print(f"OOD LaTeX table saved to: {output}")
        print(f"OOD summary CSV saved to: {resolve_path(args.csv_output)}")
        return

    if args.command == "ood-metrics":
        output = plot_ood_metric_summary(
            args.input,
            args.output,
            scenarios=args.scenarios,
            method=args.method,
            title=args.title,
            dpi=args.dpi,
        )
        print(f"OOD metric summary saved to: {output}")
        return

    if args.command == "ood-heatmaps":
        outputs = generate_ood_heatmaps(
            args.input,
            args.output_dir,
            scenarios=args.scenarios,
            methods=args.methods,
            precision=args.precision,
            dpi=args.dpi,
        )
        for metric, output in outputs.items():
            print(f"{metric} heatmap saved to: {output}")
        return

    if args.command == "ablation-report":
        outputs = generate_ablation_report(
            args.ablation_root,
            args.reference_root,
            args.output_dir,
            seeds=args.seeds,
            precision=args.precision,
            dpi=args.dpi,
        )
        for name, output in outputs.items():
            print(f"{name} saved to: {output}")
        return

    if args.command == "ablation-leave-one-out":
        output = generate_ablation_leave_one_out_table(
            args.input,
            args.output,
            precision=args.precision,
        )
        print(f"Leave-one-out LaTeX table saved to: {output}")
        return

    if args.command == "ablation-statistics":
        outputs = run_leave_one_out_statistical_analysis(
            args.results_root,
            args.output_dir,
            seeds=args.seeds,
            metrics=args.metrics,
            n_bootstrap=args.n_bootstrap,
            n_permutations=args.n_permutations,
            alpha=args.alpha,
            random_seed=args.random_seed,
            test_method=args.test_method,
            dpi=args.dpi,
        )
        for name, output in outputs.items():
            print(f"{name} saved to: {output}")
        return

    if args.command == "ablation-forest":
        frame = load_frame(args.input)
        output = plot_ablation_forest(
            frame,
            metric=args.metric,
            output=args.output,
            dpi=args.dpi,
        )
        print(f"Ablation forest plot saved to: {output}")
        return

    if args.command == "explainability-report":
        outputs = generate_explainability_report(
            args.aggregate_input,
            args.experiment_root,
            args.output_dir,
            seeds=args.seeds,
            dpi=args.dpi,
        )
        for name, output in outputs.items():
            print(f"{name} saved to: {output}")
        return

    if args.command == "latex":
        frame = _load_filtered_frame(args)
        std_map = _parse_assignments(args.std_map, option="--std-map")
        column_labels = _parse_assignments(
            args.column_label,
            option="--column-label",
        )
        latex = generate_latex_table(
            frame,
            args.columns,
            value_columns=args.value_columns,
            std_columns=std_map,
            column_labels=column_labels,
            bold_within=args.bold_within,
            precision=args.precision,
            caption=args.caption,
            label=args.label,
            position=args.position,
            table_environment=not args.tabular_only,
            multirow_columns=[] if args.no_multirow else args.multirow,
            minimize_columns=args.minimize_columns,
            resize_to_textwidth=args.resize_to_textwidth,
            bold_best=not args.no_bold_best,
        )
        if args.output:
            destination = _prepare_plot_output(args.output)
            destination.write_text(latex, encoding="utf-8")
            print(f"LaTeX table saved to: {destination}")
        else:
            print(latex, end="")
        return

    if args.command == "bar":
        frame = _load_filtered_frame(args)
        output = plot_bar_chart(
            frame,
            x=args.x,
            y=args.y,
            hue=args.hue,
            error=args.error,
            highlight=args.highlight,
            title=args.title,
            x_label=args.x_label,
            y_label=args.y_label,
            annotate=args.annotate,
            precision=args.precision,
            dpi=args.dpi,
            output=args.output,
        )
        print(f"Bar chart saved to: {output}")
        return

    if args.command == "line":
        frame = _load_filtered_frame(args)
        output = plot_line_chart(
            frame,
            x=args.x,
            y=args.y,
            hue=args.hue,
            error=args.error,
            highlight=args.highlight,
            title=args.title,
            x_label=args.x_label,
            y_label=args.y_label,
            annotate=args.annotate,
            precision=args.precision,
            dpi=args.dpi,
            output=args.output,
        )
        print(f"Line chart saved to: {output}")
        return

    if args.command == "scatter":
        frame = _load_filtered_frame(args)
        output = plot_scatter_chart(
            frame,
            x=args.x,
            y=args.y,
            label=args.label,
            highlight=args.highlight,
            title=args.title,
            x_label=args.x_label,
            y_label=args.y_label,
            dpi=args.dpi,
            output=args.output,
        )
        print(f"Scatter chart saved to: {output}")
        return

    if args.command == "aggregate-curves":
        summary = plot_aggregate_prediction_curves(
            args.inputs,
            output_dir=args.output_dir,
            prefix=args.prefix,
            task=args.task,
            n_bins=args.n_bins,
            title_prefix=args.title_prefix,
            dpi=args.dpi,
        )
        print(f"ROC curve saved to: {summary['outputs']['roc']}")
        print(
            "Precision-recall curve saved to: "
            f"{summary['outputs']['precision_recall']}"
        )
        print(
            "Calibration curve saved to: "
            f"{summary['outputs']['calibration']}"
        )
        print(f"Summary saved to: {summary['outputs']['summary']}")
        print(
            "Macro-AUROC="
            f"{summary['macro_auroc']['mean']:.6f}"
            f"±{summary['macro_auroc']['std']:.6f}"
        )
        print(
            "Macro-AUPRC="
            f"{summary['macro_auprc']['mean']:.6f}"
            f"±{summary['macro_auprc']['std']:.6f}"
        )
        print(
            "ECE="
            f"{summary['ece']['mean']:.6f}"
            f"±{summary['ece']['std']:.6f}"
        )
        return

    if args.command == "full-shot-comparison":
        dataset_titles = _parse_assignments(
            args.dataset_title,
            option="--dataset-title",
        )
        output = plot_full_shot_comparison(
            args.input,
            args.output,
            datasets=args.datasets,
            title_template=args.title_template,
            dataset_titles=dataset_titles,
            figure_title=args.figure_title,
            dpi=args.dpi,
        )
        print(f"Full-shot comparison saved to: {output}")
        return

    if args.command == "full-shot-paired-forest":
        output, csv_output = plot_full_shot_paired_forest(
            args.input,
            args.output,
            results_root=args.results_root,
            datasets=args.datasets,
            metric=args.metric,
            confidence_level=args.confidence_level,
            csv_output=args.csv_output,
            dpi=args.dpi,
        )
        print(f"Full-shot paired forest plot saved to: {output}")
        print(f"Paired effect estimates saved to: {csv_output}")
        return

    if args.command == "aggregate-confusion":
        output, matrix = plot_aggregate_confusion_matrix(
            args.inputs,
            class_names=load_class_names(args.class_names_file),
            output=args.output,
            normalize=not args.no_normalize,
            show_label_indices=not args.no_label_indices,
            title=args.title,
            x_label=args.x_label,
            y_label=args.y_label,
            cmap=args.cmap,
            precision=args.precision,
            dpi=args.dpi,
        )
        print(f"Confusion matrix saved to: {output}")
        print(
            "Pooled count matrix saved to: "
            f"{output.with_name(f'{output.stem}_counts.csv')}"
        )
        print(f"Seeds={len(args.inputs)}, pooled predictions={int(matrix.sum())}")
        return

    if args.command == "heatmap":
        frame = _load_filtered_frame(args)
        output = plot_heatmap(
            frame,
            rows=args.rows,
            columns=args.cols,
            value=args.value,
            title=args.title,
            cmap=args.cmap,
            x_label=args.x_label,
            y_label=args.y_label,
            colorbar_label=args.colorbar_label,
            precision=args.precision,
            dpi=args.dpi,
            output=args.output,
        )
        print(f"Heatmap saved to: {output}")
        return

    probabilities, labels = load_calibration_arrays(
        args.input,
        probabilities_key=args.probabilities_key,
        labels_key=args.labels_key,
        label_column=args.label_column,
        probability_columns=args.probability_columns,
    )
    if args.command == "roc":
        output, statistics = plot_roc_curve(
            probabilities,
            labels,
            task=args.task,
            class_names=args.class_names,
            show_per_class=not args.no_per_class,
            title=args.title,
            dpi=args.dpi,
            output=args.output,
        )
        print(f"ROC curve saved to: {output}")
        print(f"Macro-AUROC={statistics['macro']['auc']:.6f}")
        print(f"Micro-AUROC={statistics['micro']['auc']:.6f}")
        if statistics["skipped_classes"]:
            print(f"Skipped classes: {statistics['skipped_classes']}")
        return
    if args.command == "pr":
        output, statistics = plot_precision_recall_curve(
            probabilities,
            labels,
            task=args.task,
            class_names=args.class_names,
            show_per_class=not args.no_per_class,
            title=args.title,
            dpi=args.dpi,
            output=args.output,
        )
        print(f"Precision-recall curve saved to: {output}")
        print(
            "Macro average precision="
            f"{statistics['macro']['average_precision']:.6f}"
        )
        print(
            "Micro average precision="
            f"{statistics['micro']['average_precision']:.6f}"
        )
        if statistics["skipped_classes"]:
            print(f"Skipped classes: {statistics['skipped_classes']}")
        return

    output, statistics = plot_calibration_curve(
        probabilities,
        labels,
        n_bins=args.n_bins,
        title=args.title,
        model_label=args.model_label,
        ideal_label=args.ideal_label,
        x_label=args.x_label,
        y_label=args.y_label,
        dpi=args.dpi,
        output=args.output,
    )
    print(f"Calibration curve saved to: {output}")
    print(f"ECE={statistics['ece']:.6f}")


if __name__ == "__main__":
    main()
