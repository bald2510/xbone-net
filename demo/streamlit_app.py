"""Cung cấp giao diện Streamlit cho suy luận và phân tích XBone-Net.

Notes
-----
Mô-đun này thuộc cơ sở mã nguồn nghiên cứu XBone-Net và giữ các quy ước dùng chung của dự án.
"""

from __future__ import annotations

import hashlib
import io
import os
import sys
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd
from matplotlib import colors
from matplotlib.colors import LinearSegmentedColormap
from omegaconf import OmegaConf
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ưu tiên bộ checkpoint và feature archive đã đóng gói cùng demo. Biến môi
# trường vẫn cho phép người triển khai chủ động trỏ sang một bộ tài nguyên khác.
DEMO_ARTIFACT_ROOT = Path(__file__).resolve().parent / "artifacts"
os.environ.setdefault("XBONE_DEMO_ARTIFACT_ROOT", str(DEMO_ARTIFACT_ROOT))

import streamlit as st

from src.datasets.high_resolution import build_sparse_focal_views
from src.utils.online_inference import (
    OOD_METHODS,
    OnlineInferenceEngine,
    render_global_ig_overlay,
    render_local_ig_overlay,
)


LOGO_PATH = PROJECT_ROOT / "docs" / "report" / "images" / "logo-khtn.png"
CTCH_IMAGE_ROOT = PROJECT_ROOT / "data" / "CTCH" / "images"
ANALYSIS_STATE_KEY = "xbonenet_analysis"

METHOD_LABELS = {
    "mahalanobis_centroid": "Mahalanobis–centroid",
    "cosine_centroids": "Cosine–centroid",
    "knn": "Cosine kNN (k=5)",
    "entropy": "Entropy dự đoán",
}

TILE_ROLE_LABELS = {
    "coverage": "Bao phủ",
    "focal": "Tiêu điểm",
    "fallback": "Dự phòng",
}

TEXT_ATTRIBUTION_CMAP = LinearSegmentedColormap.from_list(
    "xbonenet_text_ig",
    ["#1D4ED8", "#FFFFFF", "#EA580C"],
)

APP_CSS = """
<style>
:root { color-scheme: light; }
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background: #FFFFFF;
    color: #0F172A;
}
[data-testid="stSidebar"] {
    background: #F8FAFC;
    border-right: 1px solid #CBD5E1;
}
[data-testid="stHeader"] {
    background: #FFFFFF;
    color: #0F172A;
}
[data-testid="stVerticalBlockBorderWrapper"] {
    background: #FFFFFF !important;
    border-color: #CBD5E1 !important;
    border-radius: 14px;
}
[data-testid="stExpander"],
[data-testid="stExpander"] details,
[data-testid="stExpander"] summary,
[data-testid="stExpander"] [data-testid="stExpanderDetails"],
[data-testid="stExpander"] [data-testid="stVerticalBlock"] {
    background: #FFFFFF !important;
    color: #0F172A !important;
}
[data-testid="stExpander"] {
    border: 1px solid #CBD5E1 !important;
    border-radius: 10px !important;
    overflow: hidden;
}
[data-testid="stExpander"] summary:hover {
    background: #F8FAFC !important;
}
[data-testid="stExpander"] svg {
    fill: #0F172A !important;
}
[data-testid="stCodeBlock"],
[data-testid="stCodeBlock"] pre,
[data-testid="stCodeBlock"] code {
    background: #F8FAFC !important;
    color: #0F172A !important;
}
[data-testid="stToolbar"] button,
[data-testid="stToolbar"] button p,
[data-testid="stToolbar"] button span {
    color: #0F172A !important;
}
[data-testid="stToolbar"] svg {
    fill: #0F172A !important;
}
.stApp h1, .stApp h2, .stApp h3, .stApp h4,
.stApp p, .stApp label, .stApp li, .stApp span,
[data-testid="stMetricLabel"], [data-testid="stMetricValue"] {
    color: #0F172A;
}
.stApp [data-testid="stCaptionContainer"] p {
    color: #475569;
}
.stApp input, .stApp textarea {
    background: #FFFFFF !important;
    color: #0F172A !important;
    border-color: #94A3B8 !important;
}
.stApp [data-baseweb="select"] > div {
    background: #FFFFFF;
    color: #0F172A;
    border-color: #94A3B8;
}
.stApp [data-testid="stFileUploaderDropzone"] {
    background: #F8FAFC;
    border: 1px dashed #64748B;
}
.stApp [data-testid="stFileUploaderDropzone"] *,
.stApp [data-testid="stFileUploaderDropzoneInstructions"] * {
    color: #0F172A !important;
}
.stApp [data-testid="stFileUploaderDropzone"] button {
    background: #FFFFFF;
    border: 1px solid #64748B;
    color: #0F172A;
}
.stApp [data-testid="stFileChip"] {
    background: #F1F5F9;
    border: 1px solid #94A3B8;
}
.stApp [data-testid="stFileChip"] *,
.stApp [data-testid="stFileChipName"] {
    color: #0F172A !important;
}
.stApp .stButton > button[kind="primary"],
.stApp .stFormSubmitButton > button[kind="primary"] {
    background: #07549A;
    border-color: #07549A;
    color: #FFFFFF;
    font-weight: 700;
}
.stApp .stButton > button[kind="primary"] p,
.stApp .stFormSubmitButton > button[kind="primary"] p {
    color: #FFFFFF;
}
.result-card {
    background: #F8FAFC;
    border: 1px solid #CBD5E1;
    border-radius: 12px;
    padding: 0.9rem 1rem;
    margin: 0.25rem 0 0.75rem 0;
}
.probability-table-wrap {
    max-height: 510px;
    overflow: auto;
    background: #FFFFFF;
    border: 1px solid #CBD5E1;
    border-radius: 10px;
}
.probability-table {
    width: 100%;
    border-collapse: collapse;
    background: #FFFFFF;
    color: #0F172A;
}
.probability-table th {
    position: sticky;
    top: 0;
    z-index: 1;
    background: #FFFFFF;
    color: #0F172A;
    border-bottom: 2px solid #94A3B8;
    padding: 0.7rem 0.8rem;
    text-align: left;
}
.probability-table td {
    background: #FFFFFF;
    color: #0F172A;
    border-bottom: 1px solid #E2E8F0;
    padding: 0.6rem 0.8rem;
}
.probability-table tbody tr:hover td {
    background: #F8FAFC;
}
.section-heading {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin-bottom: 0.85rem;
}
.section-number {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 2.1rem;
    height: 2.1rem;
    border-radius: 999px;
    background: #07549A;
    color: #FFFFFF !important;
    font-weight: 800;
}
.section-title {
    color: #0F172A;
    font-size: 1.55rem;
    font-weight: 800;
    line-height: 1.2;
}
.section-description {
    color: #475569;
    font-size: 0.92rem;
    margin-top: 0.15rem;
}
.app-kicker {
    color: #07549A;
    font-size: 0.82rem;
    font-weight: 800;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.app-title {
    color: #0F172A;
    font-size: clamp(1.9rem, 3.2vw, 3rem);
    font-weight: 800;
    line-height: 1.05;
    margin: 0.15rem 0 0.35rem 0;
}
.app-subtitle {
    color: #475569;
    font-size: 1rem;
    margin: 0;
}
</style>
"""


st.set_page_config(
    page_title="XBone-Net Demo",
    page_icon="🩻",
    layout="wide",
)


@st.cache_resource(show_spinner=False, max_entries=3)
def load_engine(seed: int, device: str) -> OnlineInferenceEngine:
    """Tải engine cho bước xử lý hiện tại.

    Parameters
    ----------
    seed : int
        Hạt giống phục vụ khả năng tái lập.
    device : str
        Thiết bị thực thi phép tính.

    Returns
    -------
    OnlineInferenceEngine
        Kết quả được tạo bởi bước xử lý của hàm.
    """

    return OnlineInferenceEngine(seed=seed, device=device)


def inject_light_theme() -> None:
    """Thực hiện bước inject light theme trong quy trình hiện tại."""

    st.markdown(APP_CSS, unsafe_allow_html=True)


def show_section_heading(number: int, title: str, description: str) -> None:
    """Thực hiện bước show section heading trong quy trình hiện tại.

    Parameters
    ----------
    number : int
        Giá trị ``number`` được sử dụng trong phép xử lý.
    title : str
        Giá trị ``title`` được sử dụng trong phép xử lý.
    description : str
        Giá trị ``description`` được sử dụng trong phép xử lý.
    """

    st.markdown(
        "<div class='section-heading'>"
        f"<span class='section-number'>{int(number):02d}</span>"
        "<div>"
        f"<div class='section-title'>{escape(title)}</div>"
        f"<div class='section-description'>{escape(description)}</div>"
        "</div></div>",
        unsafe_allow_html=True,
    )


def show_header() -> None:
    """Thực hiện bước show header trong quy trình hiện tại."""

    logo_column, title_column = st.columns([1, 8])
    with logo_column:
        if LOGO_PATH.is_file():
            st.image(str(LOGO_PATH), width=96)
    with title_column:
        st.markdown(
            """
            <div class="app-kicker">Trường Đại học Khoa học Tự nhiên, ĐHQG-HCM</div>
            <div class="app-title">XBone-Net · Online inference</div>
            <p class="app-subtitle">
                Phân loại X-quang xương đa phương thức, cảnh báo OOD và giải thích
                Integrated Gradients cho ảnh cùng bệnh sử.
            </p>
            """,
            unsafe_allow_html=True,
        )


def bounded_image(image: Image.Image, max_width: int, max_height: int) -> Image.Image:
    """Thực hiện bước bounded ảnh trong quy trình hiện tại.

    Parameters
    ----------
    image : Image.Image
        Ảnh hoặc biểu diễn ảnh đầu vào.
    max_width : int
        Giá trị ``max_width`` được sử dụng trong phép xử lý.
    max_height : int
        Giá trị ``max_height`` được sử dụng trong phép xử lý.

    Returns
    -------
    Image.Image
        Kết quả được tạo bởi bước xử lý của hàm.
    """

    preview = image.convert("RGB").copy()
    preview.thumbnail((int(max_width), int(max_height)), Image.Resampling.LANCZOS)
    return preview


def uploaded_image(uploaded: Any) -> tuple[Image.Image | None, bytes | None]:
    """Thực hiện bước uploaded ảnh trong quy trình hiện tại.

    Parameters
    ----------
    uploaded : Any
        Giá trị ``uploaded`` được sử dụng trong phép xử lý.

    Returns
    -------
    tuple[Image.Image | None, bytes | None]
        Kết quả được tạo bởi bước xử lý của hàm.
    """

    if uploaded is None:
        return None, None
    payload = uploaded.getvalue()
    try:
        return Image.open(io.BytesIO(payload)).convert("RGB"), payload
    except Exception as error:
        st.error(f"Không thể đọc ảnh: {error}")
        return None, payload


def analysis_signature(
    image_bytes: bytes,
    clinical_text: str,
    *,
    seed: int,
    device: str,
    method: str,
    ig_steps: int,
    similar_image_count: int,
) -> str:
    """Thực hiện bước analysis signature trong quy trình hiện tại.

    Parameters
    ----------
    image_bytes : bytes
        Ảnh hoặc biểu diễn ảnh đầu vào.
    clinical_text : str
        Văn bản hoặc biểu diễn văn bản đầu vào.
    seed : int
        Hạt giống phục vụ khả năng tái lập.
    device : str
        Thiết bị thực thi phép tính.
    method : str
        Phương pháp hoặc chế độ xử lý được chọn.
    ig_steps : int
        Giá trị ``ig_steps`` được sử dụng trong phép xử lý.
    similar_image_count : int
        Số lượng, kích thước hoặc tỷ lệ được sử dụng.

    Returns
    -------
    str
        Kết quả được tạo bởi bước xử lý của hàm.
    """

    digest = hashlib.sha256()
    digest.update(image_bytes)
    settings = "\n".join(
        [
            clinical_text.strip(),
            str(seed),
            str(device),
            str(method),
            str(ig_steps),
            str(similar_image_count),
        ]
    )
    digest.update(settings.encode("utf-8"))
    return digest.hexdigest()


def preprocessing_views(engine: OnlineInferenceEngine, image: Image.Image):
    """Thực hiện bước preprocessing views trong quy trình hiện tại.

    Parameters
    ----------
    engine : OnlineInferenceEngine
        Giá trị ``engine`` được sử dụng trong phép xử lý.
    image : Image.Image
        Ảnh hoặc biểu diễn ảnh đầu vào.

    Returns
    -------
    object
        Kết quả được tạo bởi bước xử lý của hàm.

    Raises
    ------
    RuntimeError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    """

    high_res_cfg = OmegaConf.to_container(
        engine.loaded.cfg.dataset.params.high_res,
        resolve=True,
    )
    if not isinstance(high_res_cfg, dict):
        raise RuntimeError("Cấu hình high-resolution của checkpoint không hợp lệ.")
    return build_sparse_focal_views(image.convert("RGB"), high_res_cfg)


def source_with_boxes(image: Image.Image, views) -> Image.Image:
    """Thực hiện bước source with boxes trong quy trình hiện tại.

    Parameters
    ----------
    image : Image.Image
        Ảnh hoặc biểu diễn ảnh đầu vào.
    views : object
        Giá trị ``views`` được sử dụng trong phép xử lý.

    Returns
    -------
    Image.Image
        Kết quả được tạo bởi bước xử lý của hàm.
    """

    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    line_width = max(2, round(min(image.size) / 180))
    draw.rectangle(views.foreground_box, outline="#F59E0B", width=line_width)
    for index, box in enumerate(views.tile_boxes, start=1):
        draw.rectangle(box, outline="#DC2626", width=line_width)
        draw.text((box[0] + line_width, box[1] + line_width), str(index), fill="#DC2626")
    return annotated


def show_preprocessing(image: Image.Image, views) -> None:
    """Thực hiện bước show preprocessing trong quy trình hiện tại.

    Parameters
    ----------
    image : Image.Image
        Ảnh hoặc biểu diễn ảnh đầu vào.
    views : object
        Giá trị ``views`` được sử dụng trong phép xử lý.
    """

    with st.expander("Xem các bước preprocessing", expanded=False):
        st.caption(
            "Các ảnh dưới đây được tạo bằng đúng cấu hình high-resolution của "
            "checkpoint đang suy luận. Khung vàng là vùng tiền cảnh; khung đỏ là "
            "bốn vùng cục bộ được chọn."
        )
        source_column, box_column, global_column = st.columns(3)
        source_column.image(
            bounded_image(image, 360, 280),
            caption=f"1. Ảnh gốc · {image.width}×{image.height} px",
        )
        box_column.image(
            bounded_image(source_with_boxes(image, views), 360, 280),
            caption="2. Cắt tiền cảnh và chọn local tile",
        )
        global_column.image(
            views.global_image,
            caption=f"3. Global view · {views.global_image.width}×{views.global_image.height} px",
        )

        st.markdown("**4. Bốn local tile đưa vào mô hình**")
        tile_columns = st.columns(len(views.tiles))
        tile_rows = []
        for index, (column, tile, role, box) in enumerate(
            zip(tile_columns, views.tiles, views.tile_roles, views.tile_boxes),
            start=1,
        ):
            role_label = TILE_ROLE_LABELS.get(str(role), str(role))
            column.image(tile, caption=f"Tile {index} · {role_label}")
            tile_rows.append(
                {
                    "Tile": index,
                    "Vai trò": role_label,
                    "Hộp nguồn (x1, y1, x2, y2)": str(tuple(int(v) for v in box)),
                }
            )
        rows = []
        for tile in tile_rows:
            rows.append(
                "<tr>"
                f"<td>{int(tile['Tile'])}</td>"
                f"<td>{escape(str(tile['Vai trò']))}</td>"
                f"<td>{escape(str(tile['Hộp nguồn (x1, y1, x2, y2)']))}</td>"
                "</tr>"
            )
        st.markdown(
            "<div class='probability-table-wrap'>"
            "<table class='probability-table'>"
            "<thead><tr><th>Tile</th><th>Vai trò</th>"
            "<th>Hộp nguồn (x1, y1, x2, y2)</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>",
            unsafe_allow_html=True,
        )


def show_reproducibility(result, engine: OnlineInferenceEngine) -> None:
    """Thực hiện bước show reproducibility trong quy trình hiện tại.

    Parameters
    ----------
    result : object
        Giá trị ``result`` được sử dụng trong phép xử lý.
    engine : OnlineInferenceEngine
        Giá trị ``engine`` được sử dụng trong phép xử lý.
    """

    with st.expander("Thông tin tái lập"):
        lines = [
            f"seed={result.seed}",
            f"device={engine.device}",
            f"ood_method={result.ood_method}",
            f"confidence_calibrated={result.confidence_is_calibrated}",
            f"temperature={result.calibration_temperature:.8f}",
            f"checkpoint={result.checkpoint}",
        ]
        if engine.confidence_calibration is not None:
            lines.extend(
                [
                    (
                        "confidence_calibration_id_count="
                        f"{engine.confidence_calibration.validation_count}"
                    ),
                    (
                        "confidence_calibration_nll_before="
                        f"{engine.confidence_calibration.nll_before:.8f}"
                    ),
                    (
                        "confidence_calibration_nll_after="
                        f"{engine.confidence_calibration.nll_after:.8f}"
                    ),
                ]
            )
        if engine.ood is not None:
            lines.extend(
                [
                    f"target_id_fpr={engine.ood.target_id_fpr}",
                    (
                        "calibration_id_count="
                        f"{engine.ood.calibration_counts[result.ood_method]}"
                    ),
                ]
            )
        st.code("\n".join(lines), language="text")


def text_ig_markup(tokens: list[str], scores) -> str:
    """Thực hiện bước văn bản ig markup trong quy trình hiện tại.

    Parameters
    ----------
    tokens : list[str]
        Giá trị ``tokens`` được sử dụng trong phép xử lý.
    scores : object
        Giá trị ``scores`` được sử dụng trong phép xử lý.

    Returns
    -------
    str
        Kết quả được tạo bởi bước xử lý của hàm.
    """

    spans = []
    for token, score in zip(tokens, scores):
        normalized = max(-1.0, min(1.0, float(score)))
        rgba = TEXT_ATTRIBUTION_CMAP(0.5 + 0.5 * normalized)
        background = colors.to_hex(rgba, keep_alpha=False)
        spans.append(
            "<span style='display:inline-block;margin:2px;padding:4px 6px;"
            f"border:1px solid #CBD5E1;border-radius:5px;background:{background};"
            "color:#0F172A' "
            f"title='IG={float(score):.3f}'>{escape(token)}</span>"
        )
    return "<div style='line-height:2.2'>" + " ".join(spans) + "</div>"


def probability_frame(result) -> pd.DataFrame:
    """Thực hiện bước probability bảng dữ liệu trong quy trình hiện tại.

    Parameters
    ----------
    result : object
        Giá trị ``result`` được sử dụng trong phép xử lý.

    Returns
    -------
    pd.DataFrame
        Kết quả được tạo bởi bước xử lý của hàm.
    """

    order = result.probabilities.argsort()[::-1]
    return pd.DataFrame(
        {
            "Hạng": range(1, len(order) + 1),
            "Lớp": [result.class_labels[index] for index in order],
            "Xác suất": [float(result.probabilities[index]) for index in order],
            "Tỷ lệ": [
                f"{100.0 * float(result.probabilities[index]):.2f}%"
                for index in order
            ],
        }
    )


def probability_table_markup(table: pd.DataFrame) -> str:
    """Thực hiện bước probability table markup trong quy trình hiện tại.

    Parameters
    ----------
    table : pd.DataFrame
        Giá trị ``table`` được sử dụng trong phép xử lý.

    Returns
    -------
    str
        Kết quả được tạo bởi bước xử lý của hàm.
    """

    rows = []
    for row in table.itertuples(index=False, name=None):
        rank, class_label, probability, percentage = row
        rows.append(
            "<tr>"
            f"<td>{int(rank)}</td>"
            f"<td>{escape(str(class_label))}</td>"
            f"<td>{float(probability):.6f}</td>"
            f"<td>{escape(str(percentage))}</td>"
            "</tr>"
        )
    body = "".join(rows)
    return (
        "<div class='probability-table-wrap'>"
        "<table class='probability-table'>"
        "<thead><tr><th>Hạng</th><th>Lớp</th><th>Xác suất</th><th>Tỷ lệ</th></tr></thead>"
        f"<tbody>{body}</tbody></table></div>"
    )


def resolve_ctch_image(image_id: str) -> Path | None:
    """Xác định ctch ảnh cho bước xử lý hiện tại.

    Parameters
    ----------
    image_id : str
        Ảnh hoặc biểu diễn ảnh đầu vào.

    Returns
    -------
    Path | None
        Kết quả được tạo bởi bước xử lý của hàm.
    """

    direct = CTCH_IMAGE_ROOT / str(image_id)
    if direct.is_file():
        return direct
    stem = Path(str(image_id)).stem
    for extension in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
        candidate = CTCH_IMAGE_ROOT / f"{stem}{extension}"
        if candidate.is_file():
            return candidate
    return None


def show_similar_images(result, *, show_labels: bool = True) -> None:
    """Thực hiện bước show similar các ảnh trong quy trình hiện tại.

    Parameters
    ----------
    result : object
        Giá trị ``result`` được sử dụng trong phép xử lý.
    show_labels : bool, optional
        Có hiển thị nhãn thật của các ảnh tham chiếu hay không.
    """

    st.subheader("Các mẫu CTCH-train gần ảnh đầu vào nhất")
    st.caption(
        "Độ giống là cosine similarity giữa global visual embedding của ảnh "
        "đầu vào và CTCH-train. Phép tìm kiếm này không sử dụng bệnh sử đầu vào "
        "và không phải là một phép gán nhãn cho ảnh đang kiểm tra."
    )
    if not result.similar_images:
        st.info("Feature archive không có dữ liệu tham chiếu để tìm ảnh tương tự.")
        return

    for start in range(0, len(result.similar_images), 4):
        references = result.similar_images[start:start + 4]
        columns = st.columns(4)
        for column, reference in zip(columns, references):
            path = resolve_ctch_image(reference.image_id)
            with column:
                if path is None:
                    st.warning(f"Không tìm thấy {reference.image_id}")
                    continue
                try:
                    image = Image.open(path).convert("RGB")
                    st.image(bounded_image(image, 240, 180))
                    if show_labels:
                        st.markdown(f"**{escape(reference.class_label)}**")
                    st.caption(
                        f"{reference.image_id} · cosine={reference.cosine_similarity:.4f}"
                    )
                except Exception as error:
                    st.warning(f"Không đọc được {reference.image_id}: {error}")


def show_classification_and_ood(result) -> None:
    """Thực hiện bước show classification and ood trong quy trình hiện tại.

    Parameters
    ----------
    result : object
        Giá trị ``result`` được sử dụng trong phép xử lý.
    """

    if result.is_ood:
        st.error(
            "Cảnh báo ngoài phân phối (OOD): mẫu không đủ tương đồng với dữ liệu "
            "CTCH-ID theo ngưỡng đã khóa. Hệ thống không gán nhãn bệnh, không "
            "hiển thị confidence hoặc bảng xác suất cho mẫu này."
        )
        return

    st.success(
        "Mẫu được chấp nhận là trong phân phối CTCH (ID) theo ngưỡng đã khóa."
    )
    gap = float(result.ood_score - result.ood_threshold)

    metric_1, metric_2, metric_3, metric_4 = st.columns(4)
    metric_1.metric("Lớp dự đoán", result.predicted_label)
    metric_2.metric(
        "Confidence hiệu chỉnh (MSP)",
        f"{100.0 * float(result.confidence):.2f}%",
    )
    metric_3.metric(
        "OOD score / ngưỡng",
        f"{result.ood_score:.4f} / {result.ood_threshold:.4f}",
    )
    metric_4.metric("Khoảng cách tới ngưỡng", f"{gap:+.4f}")

    st.markdown(
        "<div class='result-card'><strong>Cách tính confidence:</strong> "
        "maximum softmax probability sau temperature scaling trên CTCH "
        f"validation-ID (T={result.calibration_temperature:.4f}). Confidence là "
        "ước lượng đã hiệu chỉnh cho một dự đoán cụ thể, không phải accuracy và "
        "không bảo đảm xác suất đúng tuyệt đối.</div>",
        unsafe_allow_html=True,
    )

    st.subheader("Xếp hạng 22 lớp CTCH theo softmax gốc")
    st.caption(
        "Bảng này dùng softmax thông thường để báo cáo và xếp hạng lớp. "
        "Temperature scaling chỉ được áp dụng cho điểm confidence ở phía trên."
    )
    table = probability_frame(result)
    st.markdown(
        probability_table_markup(table),
        unsafe_allow_html=True,
    )


def show_integrated_gradients(image: Image.Image, result) -> None:
    """Thực hiện bước show integrated gradients trong quy trình hiện tại.

    Parameters
    ----------
    image : Image.Image
        Ảnh hoặc biểu diễn ảnh đầu vào.
    result : object
        Giá trị ``result`` được sử dụng trong phép xử lý.
    """

    st.subheader("Integrated Gradients")
    if result.is_ood:
        st.info(
            "Integrated Gradients theo lớp không được tạo cho mẫu bị cảnh báo "
            "OOD, đúng với quy tắc trong online_inference.py."
        )
        return

    st.markdown("#### Integrated Gradients cho ảnh")
    original_column, global_column, local_column = st.columns(3)
    original_column.image(
        bounded_image(image, 360, 300),
        caption="Ảnh đầu vào",
    )
    if result.global_ig_map is not None:
        global_overlay = render_global_ig_overlay(image, result)
        global_column.image(
            bounded_image(global_overlay, 360, 300),
            caption="Global IG · chiếu về tọa độ ảnh nguồn",
        )
    else:
        global_column.info("Không có global IG cho mẫu này.")

    if result.local_ig_scores is not None:
        local_overlay = render_local_ig_overlay(image, result)
        local_column.image(
            bounded_image(local_overlay, 360, 300),
            caption="Local IG · các token sparse-focal",
        )
    else:
        local_column.info("Không có local IG cho mẫu này.")

    st.caption(
        "Màu nóng biểu thị vùng có attribution dương lớn hơn cho lớp dự đoán. "
        "Bản đồ này giải thích phản ứng của mô hình, không chứng minh vị trí tổn "
        "thương, quan hệ nhân quả hoặc tính đúng đắn lâm sàng."
    )

    st.markdown("#### Integrated Gradients cho văn bản")
    if result.text_tokens is None or result.text_ig_scores is None:
        st.info("Không có attribution token văn bản cho mẫu này.")
        return
    st.markdown(
        text_ig_markup(result.text_tokens, result.text_ig_scores),
        unsafe_allow_html=True,
    )
    st.caption(
        "Màu cam biểu thị đóng góp ủng hộ và màu xanh biểu thị đóng góp phản đối "
        "logit của lớp dự đoán; nhánh ảnh được giữ cố định."
    )


def main() -> None:
    """Thực thi điểm vào chính của mô-đun."""

    inject_light_theme()
    show_header()
    st.divider()

    with st.sidebar:
        if LOGO_PATH.is_file():
            st.image(str(LOGO_PATH), width=72)
        st.subheader("Thiết lập suy luận")
        seed = st.selectbox(
            "Checkpoint seed",
            [42],
            index=0,
            help="Bản demo chấm điểm được khóa với checkpoint seed 42 đã đóng gói.",
        )
        device = st.selectbox(
            "Thiết bị",
            ["auto", "cuda", "cpu"],
            index=0,
            help="Auto ưu tiên CUDA khi khả dụng.",
        )
        method = st.selectbox(
            "Phương pháp OOD",
            list(OOD_METHODS),
            index=list(OOD_METHODS).index("mahalanobis_centroid"),
            format_func=lambda value: METHOD_LABELS[value],
        )
        ig_steps = st.slider(
            "Số bước Integrated Gradients",
            min_value=8,
            max_value=32,
            value=16,
            step=4,
            help="Giá trị lớn hơn cho xấp xỉ mượt hơn nhưng suy luận lâu hơn.",
        )
        similar_image_count = st.slider(
            "Số ảnh CTCH tương tự",
            min_value=1,
            max_value=8,
            value=4,
            step=1,
        )
        st.info(
            "Ngưỡng OOD là phân vị 95% của CTCH validation-ID và được khóa "
            "riêng cho từng seed/phương pháp. Điểm lớn hơn biểu thị bằng chứng "
            "OOD mạnh hơn."
        )

    with st.container(border=True):
        show_section_heading(
            1,
            "Đầu vào",
            "Chọn ảnh X-quang và nhập bệnh sử dùng cho suy luận đa phương thức.",
        )
        input_column, text_column = st.columns([1, 1.4])
        with input_column:
            uploaded = st.file_uploader(
                "Ảnh X-quang",
                type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
            )
        with text_column:
            clinical_text = st.text_area(
                "Bệnh sử hoặc lời khai lâm sàng",
                height=150,
                placeholder=(
                    "Ví dụ: Người bệnh đau và sưng cổ tay sau té ngã, hạn chế vận "
                    "động, chưa ghi nhận tiền sử phẫu thuật tại vị trí tổn thương."
                ),
            )

        image, image_bytes = uploaded_image(uploaded)
        if image is not None:
            preview_column, _ = st.columns([1, 2])
            preview_column.image(
                bounded_image(image, 420, 300),
                caption=f"Ảnh đã chọn · {image.width}×{image.height} px",
            )

        run = st.button(
            "Phân tích",
            type="primary",
            use_container_width=True,
            disabled=image is None,
        )

    current_signature = None
    if image_bytes is not None:
        current_signature = analysis_signature(
            image_bytes,
            clinical_text,
            seed=int(seed),
            device=str(device),
            method=str(method),
            ig_steps=int(ig_steps),
            similar_image_count=int(similar_image_count),
        )

    if run:
        if image is None or image_bytes is None:
            st.error("Cần chọn một ảnh X-quang hợp lệ.")
            return
        if not clinical_text.strip():
            st.error(
                "Cần nhập bệnh sử vì checkpoint đa phương thức yêu cầu cả ảnh "
                "và văn bản."
            )
            return
        try:
            with st.spinner("Đang tải checkpoint và detector CTCH-ID..."):
                engine = load_engine(int(seed), str(device))
            with st.spinner("Đang tiền xử lý sparse-focal và suy luận..."):
                views = preprocessing_views(engine, image)
                result = engine.predict(
                    image,
                    clinical_text,
                    ood_method=str(method),
                    ig_steps=int(ig_steps),
                    compute_global_ig=True,
                    similar_image_count=int(similar_image_count),
                )
            st.session_state[ANALYSIS_STATE_KEY] = {
                "signature": current_signature,
                "image": image,
                "views": views,
                "result": result,
                "engine": engine,
            }
        except Exception as error:
            st.exception(error)
            return

    analysis = st.session_state.get(ANALYSIS_STATE_KEY)
    with st.container(border=True):
        show_section_heading(
            2,
            "Đầu ra",
            "Mẫu ID có nhãn và confidence hiệu chỉnh; mẫu OOD chỉ có cảnh báo và các mẫu gần nhất.",
        )
        if analysis is None:
            st.info("Kết quả sẽ xuất hiện tại đây sau khi nhấn Phân tích.")
        elif analysis["signature"] != current_signature:
            st.info(
                "Đầu vào hoặc cấu hình đã thay đổi. Nhấn Phân tích để cập nhật kết quả."
            )
        else:
            result = analysis["result"]
            result_image = analysis["image"]
            engine = analysis["engine"]
            show_preprocessing(result_image, analysis["views"])
            show_classification_and_ood(result)
            if result.is_ood:
                show_similar_images(result, show_labels=False)
            else:
                show_similar_images(result)
                show_integrated_gradients(result_image, result)
                show_reproducibility(result, engine)

            st.warning(
                "Demo chỉ phục vụ nghiên cứu và minh họa luận văn; không phải thiết bị "
                "y tế và không thay thế đánh giá của bác sĩ. Không tải dữ liệu định danh "
                "người bệnh lên máy chủ công khai."
            )


if __name__ == "__main__":
    main()
