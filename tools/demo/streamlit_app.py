"""Small Streamlit front end for XBone-Net online inference."""

from __future__ import annotations

import io
import sys
from html import escape
from pathlib import Path

import pandas as pd
from matplotlib import colors
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src.utils.online_inference import (
    OOD_METHODS,
    OnlineInferenceEngine,
    render_local_ig_overlay,
)


METHOD_LABELS = {
    "mahalanobis_centroid": "Mahalanobis–centroid",
    "cosine_centroids": "Cosine–centroid",
    "knn": "Cosine kNN (k=5)",
    "entropy": "Entropy dự đoán",
}

TEXT_ATTRIBUTION_CMAP = LinearSegmentedColormap.from_list(
    "xbonenet_text_ig",
    ["#1f77b4", "#ffffff", "#ff7f0e"],
)


st.set_page_config(
    page_title="XBone-Net Demo",
    page_icon="🩻",
    layout="wide",
)


@st.cache_resource(show_spinner=False, max_entries=1)
def load_engine(seed: int, device: str) -> OnlineInferenceEngine:
    return OnlineInferenceEngine(seed=seed, device=device)


def show_reproducibility(result, engine) -> None:
    with st.expander("Thông tin tái lập"):
        st.code(
            "\n".join(
                [
                    f"seed={result.seed}",
                    f"device={engine.device}",
                    f"ood_method={result.ood_method}",
                    f"checkpoint={result.checkpoint}",
                    f"target_id_fpr={engine.ood.target_id_fpr}",
                    (
                        "calibration_id_count="
                        f"{engine.ood.calibration_counts[result.ood_method]}"
                    ),
                ]
            ),
            language="text",
        )


def text_ig_markup(tokens: list[str], scores) -> str:
    spans = []
    for token, score in zip(tokens, scores):
        normalized = max(-1.0, min(1.0, float(score)))
        rgba = TEXT_ATTRIBUTION_CMAP(0.5 + 0.5 * normalized)
        background = colors.to_hex(rgba, keep_alpha=False)
        foreground = "#111111"
        spans.append(
            "<span style='display:inline-block;margin:2px;padding:3px 5px;"
            f"border-radius:4px;background:{background};color:{foreground}' "
            f"title='IG={float(score):.3f}'>{escape(token)}</span>"
        )
    return "<div style='line-height:2.1'>" + " ".join(spans) + "</div>"


def main() -> None:
    st.title("XBone-Net — Online inference")
    st.caption(
        "Demo nghiên cứu: ảnh X-quang và bệnh sử → cảnh báo OOD, "
        "phân loại CTCH 22 lớp và Integrated Gradients trên local token."
    )

    with st.sidebar:
        st.subheader("Thiết lập")
        seed = st.selectbox("Checkpoint seed", [42, 123, 456], index=0)
        device = st.selectbox(
            "Thiết bị",
            ["auto", "cuda", "cpu"],
            index=0,
            help="Auto ưu tiên CUDA khi khả dụng.",
        )
        method = st.selectbox(
            "OOD score",
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
        st.info(
            "Ngưỡng OOD là phân vị 95% của CTCH validation-ID và được "
            "khóa riêng cho từng seed/phương pháp."
        )

    left, right = st.columns([1, 1])
    with left:
        uploaded = st.file_uploader(
            "Ảnh X-quang",
            type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
        )
    with right:
        clinical_text = st.text_area(
            "Bệnh sử hoặc lời khai lâm sàng",
            height=180,
            placeholder=(
                "Ví dụ: Người bệnh đau và sưng cổ tay sau té ngã, hạn chế vận "
                "động, chưa ghi nhận tiền sử phẫu thuật tại vị trí tổn thương."
            ),
        )

    image = None
    if uploaded is not None:
        try:
            image = Image.open(io.BytesIO(uploaded.getvalue())).convert("RGB")
            st.image(image, caption=f"Ảnh đầu vào — {image.width}×{image.height} px")
        except Exception as error:
            st.error(f"Không thể đọc ảnh: {error}")

    run = st.button(
        "Phân tích",
        type="primary",
        use_container_width=True,
        disabled=image is None,
    )
    if not run:
        return
    if not clinical_text.strip():
        st.error("Cần nhập bệnh sử vì checkpoint đa phương thức yêu cầu cả ảnh và văn bản.")
        return

    try:
        with st.spinner("Đang tải checkpoint và hiệu chỉnh detector từ CTCH-ID..."):
            engine = load_engine(int(seed), str(device))
        with st.spinner("Đang suy luận và tính Integrated Gradients..."):
            result = engine.predict(
                image,
                clinical_text,
                ood_method=method,
                ig_steps=int(ig_steps),
            )
    except Exception as error:
        st.exception(error)
        return

    st.divider()
    if result.is_ood:
        st.error(
            "Phát hiện mẫu ngoài phân phối (OOD). Hệ thống không đưa ra lớp "
            "dự đoán hoặc Integrated Gradients theo lớp cho mẫu này."
        )
        metric_1, metric_2, metric_3 = st.columns(3)
        metric_1.metric("OOD score", f"{result.ood_score:.4f}")
        metric_2.metric("Ngưỡng", f"{result.ood_threshold:.4f}")
        metric_3.metric(
            "Khoảng cách tới ngưỡng",
            f"{result.ood_score - result.ood_threshold:+.4f}",
        )
        show_reproducibility(result, engine)
        st.warning(
            "Demo chỉ phục vụ nghiên cứu và minh họa luận văn; không phải thiết "
            "bị y tế và không thay thế đánh giá của bác sĩ."
        )
        return

    st.success(
        "Mẫu được chấp nhận là trong phân phối CTCH (ID) theo ngưỡng đã khóa."
    )
    if (
        result.probabilities is None
        or result.predicted_label is None
        or result.predicted_index is None
    ):
        st.error("Kết quả ID không chứa đầu ra phân loại hợp lệ.")
        return

    metric_1, metric_2, metric_3, metric_4 = st.columns(4)
    metric_1.metric("OOD score", f"{result.ood_score:.4f}")
    metric_2.metric("Ngưỡng", f"{result.ood_threshold:.4f}")
    metric_3.metric(
        "Khoảng cách tới ngưỡng",
        f"{result.ood_score - result.ood_threshold:+.4f}",
    )
    metric_4.metric(
        "Độ tin cậy lớp cao nhất",
        f"{100.0 * float(result.probabilities.max()):.1f}%",
    )

    st.subheader("Kết quả phân loại")
    st.markdown(f"**Lớp dự đoán:** {result.predicted_label}")
    order = result.probabilities.argsort()[::-1]
    probability_table = pd.DataFrame(
        {
            "Hạng": range(1, len(order) + 1),
            "Lớp": [result.class_labels[index] for index in order],
            "Xác suất": [float(result.probabilities[index]) for index in order],
        }
    )
    st.dataframe(
        probability_table.style.format({"Xác suất": "{:.4f}"}),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Integrated Gradients của nhánh ảnh cục bộ")
    original_column, overlay_column = st.columns(2)
    original_column.image(image, caption="Ảnh đầu vào", use_container_width=True)
    overlay = render_local_ig_overlay(image, result)
    overlay_column.image(
        overlay,
        caption=(
            "Attribution cho lớp dự đoán — màu nóng biểu thị local token "
            "có đóng góp lớn hơn"
        ),
        use_container_width=True,
    )
    st.caption(
        "Heatmap được ánh xạ từ Integrated Gradients của các token không gian "
        "2×2 trong bốn local tile. Đây là giải thích ở mức token vùng, không "
        "phải bản đồ saliency mức pixel và không thể hiện quan hệ nhân quả."
    )

    st.subheader("Integrated Gradients của bệnh sử")
    if result.text_tokens is None or result.text_ig_scores is None:
        st.info("Không có attribution token văn bản cho mẫu này.")
    else:
        st.markdown(
            text_ig_markup(result.text_tokens, result.text_ig_scores),
            unsafe_allow_html=True,
        )
        st.caption(
            "Màu cam biểu thị đóng góp ủng hộ và màu xanh biểu thị đóng góp phản "
            "đối logit của lớp dự đoán. Attribution đi từ embedding từ qua text "
            "encoder, cross-attention, fusion MLP và classifier; nhánh ảnh được "
            "giữ cố định."
        )

    show_reproducibility(result, engine)

    st.warning(
        "Demo chỉ phục vụ nghiên cứu và minh họa luận văn; không phải thiết bị "
        "y tế và không thay thế đánh giá của bác sĩ."
    )


if __name__ == "__main__":
    main()
