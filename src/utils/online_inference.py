"""Suy luận trực tuyến cho XBone-Net dùng ảnh letterbox và linear head.

Mô-đun nạp đúng checkpoint đã đánh giá, hiệu chỉnh confidence/OOD từ tập
validation-ID và tạo Integrated Gradients trên ảnh toàn cục cùng văn bản. Ảnh
đầu vào chỉ đi qua một phép letterbox trước backbone; không tạo thêm nhánh ảnh
hay đặc trưng phụ trợ.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import colormaps
from omegaconf import OmegaConf
from PIL import Image

from src.datasets.preprocessing import prepare_image
from src.utils.analysis import (
    SOURCE_EXPERIMENT,
    analysis_root,
    load_evaluated_classification_model,
    load_feature_archive,
    load_locked_proposed_model,
    load_proposed_experiment_model,
)
from src.utils.explainability import (
    curve_auc,
    global_image_perturbation_curves,
    integrated_gradients_global_image,
    integrated_gradients_text,
    text_input_perturbation_curves,
)
from src.utils.ood import OODDetector, calibrate_ood_threshold
from src.utils.trainer import resolve_pad_token_id


OOD_METHODS = (
    "cosine_centroids",
    "mahalanobis_centroid",
    "knn",
    "entropy",
)


@dataclass(frozen=True)
class OODCalibration:
    """Các detector và ngưỡng OOD được khớp trên validation-ID."""

    normalized_detector: OODDetector
    raw_detector: OODDetector
    reference_image_ids: np.ndarray
    reference_visual_embeddings: np.ndarray
    thresholds: dict[str, float]
    calibration_counts: dict[str, int]
    target_id_fpr: float


@dataclass(frozen=True)
class ConfidenceCalibration:
    """Lưu tham số temperature scaling của confidence phân loại.

    Notes
    -----
    Temperature được khớp trên logits và nhãn của CTCH validation-ID thuộc
    đúng checkpoint đang suy luận. Phép biến đổi không làm thay đổi lớp có
    logit lớn nhất, mà chỉ hiệu chỉnh độ lớn xác suất.
    """

    temperature: float
    validation_count: int
    nll_before: float
    nll_after: float


@dataclass(frozen=True)
class SimilarImageReference:
    """Một ảnh huấn luyện gần nhất theo cosine similarity."""

    image_id: str
    class_index: int
    class_label: str
    cosine_similarity: float


@dataclass
class OnlineInferenceResult:
    """Kết quả phân loại, cảnh báo OOD và IG của một mẫu đầu vào."""

    predicted_index: int | None
    predicted_label: str | None
    probabilities: np.ndarray
    confidence: float | None
    confidence_is_calibrated: bool
    calibration_temperature: float
    class_labels: list[str]
    ood_method: str
    ood_score: float
    ood_threshold: float
    is_ood: bool
    similar_images: list[SimilarImageReference]
    global_ig_map: np.ndarray | None
    text_tokens: list[str] | None
    text_ig_scores: np.ndarray | None
    contribution_drops: dict[str, float] | None
    global_faithfulness: dict[str, Any] | None
    text_faithfulness: dict[str, Any] | None
    seed: int
    checkpoint: str


def _labels(values: np.ndarray) -> np.ndarray:
    """Trích xuất danh sách nhãn theo đúng thứ tự lớp.

    Parameters
    ----------
    values : np.ndarray
        Giá trị ``values`` được sử dụng trong phép xử lý.

    Returns
    -------
    np.ndarray
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    labels = np.asarray(values)
    if labels.ndim == 2:
        labels = labels.argmax(axis=1)
    return labels.astype(np.int64).reshape(-1)


def multiclass_nll(
    logits: np.ndarray,
    labels: np.ndarray,
    temperature: float,
) -> float:
    """Tính negative log-likelihood đa lớp sau temperature scaling.

    Parameters
    ----------
    logits : numpy.ndarray
        Ma trận logits có kích thước ``[N, C]``.
    labels : numpy.ndarray
        Nhãn nguyên có kích thước ``[N]``.
    temperature : float
        Nhiệt độ dương dùng để chia logits.

    Returns
    -------
    float
        Negative log-likelihood trung bình.

    Raises
    ------
    ValueError
        Khi logits, nhãn hoặc temperature không hợp lệ.
    """

    scores = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64).reshape(-1)
    if scores.ndim != 2 or len(scores) != len(targets) or not len(targets):
        raise ValueError("Expected non-empty logits [N,C] and labels [N].")
    if np.any(targets < 0) or np.any(targets >= scores.shape[1]):
        raise ValueError("Calibration labels are outside the logits class range.")
    if not np.isfinite(temperature) or float(temperature) <= 0.0:
        raise ValueError("Temperature must be a finite positive value.")

    scaled = scores / float(temperature)
    row_max = scaled.max(axis=1, keepdims=True)
    log_partition = row_max[:, 0] + np.log(
        np.exp(scaled - row_max).sum(axis=1)
    )
    true_logits = scaled[np.arange(len(targets)), targets]
    return float(np.mean(log_partition - true_logits))


def fit_temperature_scaling(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    minimum: float = 0.01,
    maximum: float = 100.0,
    iterations: int = 96,
) -> ConfidenceCalibration:
    """Khớp một temperature bằng tối thiểu hóa NLL trên tập xác thực.

    Parameters
    ----------
    logits : numpy.ndarray
        Ma trận logits xác thực có kích thước ``[N, C]``.
    labels : numpy.ndarray
        Nhãn nguyên hoặc nhãn một-nóng của tập xác thực.
    minimum : float, optional
        Cận dưới dương của miền tìm kiếm temperature.
    maximum : float, optional
        Cận trên của miền tìm kiếm temperature.
    iterations : int, optional
        Số vòng lặp golden-section trên thang log-temperature.

    Returns
    -------
    ConfidenceCalibration
        Tham số hiệu chỉnh và NLL trước/sau hiệu chỉnh.

    Raises
    ------
    ValueError
        Khi miền tìm kiếm hoặc số vòng lặp không hợp lệ.
    """

    scores = np.asarray(logits, dtype=np.float64)
    targets = _labels(np.asarray(labels))
    if minimum <= 0.0 or maximum <= minimum:
        raise ValueError("Temperature bounds must satisfy 0 < minimum < maximum.")
    if int(iterations) < 8:
        raise ValueError("Temperature fitting requires at least eight iterations.")

    lower = float(np.log(minimum))
    upper = float(np.log(maximum))
    golden_ratio = (np.sqrt(5.0) - 1.0) / 2.0
    left = upper - golden_ratio * (upper - lower)
    right = lower + golden_ratio * (upper - lower)
    left_loss = multiclass_nll(scores, targets, np.exp(left))
    right_loss = multiclass_nll(scores, targets, np.exp(right))

    for _ in range(int(iterations)):
        if left_loss <= right_loss:
            upper = right
            right = left
            right_loss = left_loss
            left = upper - golden_ratio * (upper - lower)
            left_loss = multiclass_nll(scores, targets, np.exp(left))
        else:
            lower = left
            left = right
            left_loss = right_loss
            right = lower + golden_ratio * (upper - lower)
            right_loss = multiclass_nll(scores, targets, np.exp(right))

    temperature = float(np.exp((lower + upper) / 2.0))
    return ConfidenceCalibration(
        temperature=temperature,
        validation_count=int(len(targets)),
        nll_before=multiclass_nll(scores, targets, 1.0),
        nll_after=multiclass_nll(scores, targets, temperature),
    )


def calibrated_softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Chuyển logits thành xác suất bằng temperature scaling và softmax.

    Parameters
    ----------
    logits : numpy.ndarray
        Vector ``[C]`` hoặc ma trận ``[N, C]`` logits.
    temperature : float
        Temperature dương đã được khớp trên tập xác thực.

    Returns
    -------
    numpy.ndarray
        Xác suất có cùng số chiều với đầu vào và tổng mỗi hàng bằng một.

    Raises
    ------
    ValueError
        Khi logits hoặc temperature không hợp lệ.
    """

    scores = np.asarray(logits, dtype=np.float64)
    was_vector = scores.ndim == 1
    if was_vector:
        scores = scores[None, :]
    if scores.ndim != 2 or scores.shape[1] < 2:
        raise ValueError("Expected logits with shape [C] or [N,C].")
    if not np.isfinite(temperature) or float(temperature) <= 0.0:
        raise ValueError("Temperature must be a finite positive value.")

    scaled = scores / float(temperature)
    scaled -= scaled.max(axis=1, keepdims=True)
    exponentiated = np.exp(scaled)
    probabilities = exponentiated / exponentiated.sum(axis=1, keepdims=True)
    return probabilities[0] if was_vector else probabilities


def fit_locked_confidence_calibration(loaded) -> ConfidenceCalibration:
    """Khớp temperature scaling từ CTCH validation-ID đã khóa.

    Parameters
    ----------
    loaded : object
        Mô hình và provenance của checkpoint đang suy luận.

    Returns
    -------
    ConfidenceCalibration
        Temperature và thống kê NLL của đúng checkpoint.

    Raises
    ------
    RuntimeError
        Khi feature archive không thuộc checkpoint đang tải.
    """

    feature_root = analysis_root(loaded.provenance["seed"]) / "features"
    validation, provenance = load_feature_archive(
        feature_root / "ctch_val.npz",
        expected_source_experiment=SOURCE_EXPERIMENT,
    )
    archive_hash = str(provenance.get("checkpoint_sha256", ""))
    if archive_hash != loaded.checkpoint_sha256:
        raise RuntimeError(
            "CTCH validation feature archive was not produced by the loaded "
            "checkpoint. Re-export it before calibrated online inference."
        )
    return fit_temperature_scaling(validation["logits"], validation["labels"])


def _score_method(
    method: str,
    normalized_detector: OODDetector,
    raw_detector: OODDetector,
    *,
    fused_embedding: np.ndarray,
    fused_embedding_raw: np.ndarray,
    logits: np.ndarray,
    knn_k: int,
) -> np.ndarray:
    """Tính điểm OOD bằng phương pháp được lựa chọn.

    Parameters
    ----------
    method : str
        Phương pháp hoặc chế độ xử lý được chọn.
    normalized_detector : OODDetector
        Giá trị ``normalized_detector`` được sử dụng trong phép xử lý.
    raw_detector : OODDetector
        Giá trị ``raw_detector`` được sử dụng trong phép xử lý.
    fused_embedding : np.ndarray
        Biểu diễn đặc trưng cần xử lý.
    fused_embedding_raw : np.ndarray
        Biểu diễn đặc trưng cần xử lý.
    logits : np.ndarray
        Giá trị ``logits`` được sử dụng trong phép xử lý.
    knn_k : int
        Giá trị ``knn_k`` được sử dụng trong phép xử lý.

    Returns
    -------
    np.ndarray
        Kết quả được tạo bởi bước xử lý của hàm.

    Raises
    ------
    ValueError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    """
    if method == "cosine_centroids":
        return normalized_detector.score_cosine_centroids(fused_embedding)
    if method == "mahalanobis_centroid":
        return raw_detector.score_mahalanobis_centroid(fused_embedding_raw)
    if method == "knn":
        return normalized_detector.score_knn(fused_embedding, k=knn_k)
    if method == "entropy":
        return normalized_detector.score_entropy(logits)
    raise ValueError(f"Unsupported OOD method: {method}")


def _calibration_scores(
    method: str,
    normalized_detector: OODDetector,
    raw_detector: OODDetector,
    arrays: Mapping[str, np.ndarray],
    knn_k: int,
) -> np.ndarray:
    """Thực hiện bước calibration các điểm trong quy trình hiện tại.

    Parameters
    ----------
    method : str
        Phương pháp hoặc chế độ xử lý được chọn.
    normalized_detector : OODDetector
        Giá trị ``normalized_detector`` được sử dụng trong phép xử lý.
    raw_detector : OODDetector
        Giá trị ``raw_detector`` được sử dụng trong phép xử lý.
    arrays : Mapping[str, np.ndarray]
        Giá trị ``arrays`` được sử dụng trong phép xử lý.
    knn_k : int
        Giá trị ``knn_k`` được sử dụng trong phép xử lý.

    Returns
    -------
    np.ndarray
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    return _score_method(
        method,
        normalized_detector,
        raw_detector,
        fused_embedding=np.asarray(arrays["fused_embeddings"]),
        fused_embedding_raw=np.asarray(arrays["fused_embeddings_raw"]),
        logits=np.asarray(arrays["logits"]),
        knn_k=knn_k,
    )


def fit_locked_ood_calibration(
    loaded,
    *,
    target_id_fpr: float = 0.05,
    knn_k: int = 5,
) -> OODCalibration:
    """Khớp locked ood calibration cho bước xử lý hiện tại.

    Parameters
    ----------
    loaded : object
        Giá trị ``loaded`` được sử dụng trong phép xử lý.
    target_id_fpr : float, optional
        Nhãn hoặc chỉ số lớp liên quan.
    knn_k : int, optional
        Giá trị ``knn_k`` được sử dụng trong phép xử lý.

    Returns
    -------
    OODCalibration
        Kết quả được tạo bởi bước xử lý của hàm.

    Raises
    ------
    RuntimeError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    """

    feature_root = analysis_root(loaded.provenance["seed"]) / "features"
    train, train_provenance = load_feature_archive(
        feature_root / "ctch_train.npz",
        expected_source_experiment=SOURCE_EXPERIMENT,
    )
    validation, validation_provenance = load_feature_archive(
        feature_root / "ctch_val.npz",
        expected_source_experiment=SOURCE_EXPERIMENT,
    )

    for split_name, provenance in (
        ("CTCH train", train_provenance),
        ("CTCH validation", validation_provenance),
    ):
        archive_hash = str(provenance.get("checkpoint_sha256", ""))
        if archive_hash != loaded.checkpoint_sha256:
            raise RuntimeError(
                f"{split_name} feature archive was not produced by the loaded "
                "checkpoint. Re-export the feature archives before online OOD "
                "inference."
            )

    train_labels = _labels(train["labels"])
    reference_image_ids = np.asarray(train.get("image_id", []), dtype=str).reshape(-1)
    reference_visual_embeddings = np.asarray(
        train.get("visual_global_embeddings", []),
        dtype=np.float64,
    )
    if len(reference_image_ids) != len(train_labels):
        raise RuntimeError(
            "CTCH train feature archive does not contain one image_id per "
            "reference embedding."
        )
    if (
        reference_visual_embeddings.ndim != 2
        or len(reference_visual_embeddings) != len(train_labels)
    ):
        raise RuntimeError(
            "CTCH train feature archive does not contain one global visual "
            "embedding per reference image."
        )
    normalized_detector = OODDetector().fit(
        np.asarray(train["fused_embeddings"]),
        train_labels,
    )
    raw_detector = OODDetector().fit(
        np.asarray(train["fused_embeddings_raw"]),
        train_labels,
    )

    thresholds: dict[str, float] = {}
    counts: dict[str, int] = {}
    for method in OOD_METHODS:
        scores = _calibration_scores(
            method,
            normalized_detector,
            raw_detector,
            validation,
            knn_k,
        )
        thresholds[method] = calibrate_ood_threshold(
            scores,
            target_id_fpr=target_id_fpr,
        )
        counts[method] = int(len(scores))

    return OODCalibration(
        normalized_detector=normalized_detector,
        raw_detector=raw_detector,
        reference_image_ids=reference_image_ids,
        reference_visual_embeddings=reference_visual_embeddings,
        thresholds=thresholds,
        calibration_counts=counts,
        target_id_fpr=float(target_id_fpr),
    )


def _tokenize_text(tokenizer, text: str, device: torch.device):
    """Thực hiện bước tokenize văn bản trong quy trình hiện tại.

    Parameters
    ----------
    tokenizer : object
        Giá trị ``tokenizer`` được sử dụng trong phép xử lý.
    text : str
        Văn bản hoặc biểu diễn văn bản đầu vào.
    device : torch.device
        Thiết bị thực thi phép tính.

    Returns
    -------
    object
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    tokenized = tokenizer([text])
    attention_mask = None
    if isinstance(tokenized, Mapping):
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized.get("attention_mask")
    else:
        input_ids = tokenized
    if not isinstance(input_ids, torch.Tensor):
        input_ids = torch.as_tensor(input_ids, dtype=torch.long)
    input_ids = input_ids.to(device)
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    if attention_mask is None:
        pad_id = resolve_pad_token_id(tokenizer)
        attention_mask = (input_ids != pad_id).long()
    else:
        attention_mask = torch.as_tensor(attention_mask, dtype=torch.long, device=device)
        if attention_mask.ndim == 1:
            attention_mask = attention_mask.unsqueeze(0)
    return input_ids, attention_mask


def _build_online_inputs(loaded, image: Image.Image, clinical_text: str):
    """Chuẩn hóa một ảnh bằng letterbox và token hóa mô tả lâm sàng.

    Parameters
    ----------
    loaded : object
        Giá trị ``loaded`` được sử dụng trong phép xử lý.
    image : Image.Image
        Ảnh hoặc biểu diễn ảnh đầu vào.
    clinical_text : str
        Văn bản hoặc biểu diễn văn bản đầu vào.

    Returns
    -------
    object
        Kết quả được tạo bởi bước xử lý của hàm.

    """
    cfg = loaded.cfg
    preprocess_cfg = (
        OmegaConf.to_container(cfg.dataset.params.preprocess, resolve=True)
        if hasattr(cfg.dataset.params, "preprocess")
        and cfg.dataset.params.preprocess is not None
        else {}
    )
    pixel_values = prepare_image(
        image.convert("RGB"),
        loaded.model.backbone.preprocess,
        preprocess_cfg,
    ).unsqueeze(0).to(loaded.device)

    input_ids, attention_mask = _tokenize_text(
        loaded.model.backbone.tokenizer_obj,
        clinical_text,
        loaded.device,
    )
    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "foreground_box": (0, 0, image.width, image.height),
    }


def _forward_and_cache(loaded, inputs: Mapping[str, Any]):
    """Thực hiện bước forward and cache trong quy trình hiện tại.

    Parameters
    ----------
    loaded : object
        Giá trị ``loaded`` được sử dụng trong phép xử lý.
    inputs : Mapping[str, Any]
        Giá trị ``inputs`` được sử dụng trong phép xử lý.

    Returns
    -------
    object
        Kết quả được tạo bởi bước xử lý của hàm.

    Raises
    ------
    RuntimeError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    """
    model = loaded.model
    with torch.no_grad():
        image_tokens, text_tokens = model.backbone(
            inputs["pixel_values"],
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        full_image_padding = getattr(
            model.backbone,
            "last_image_key_padding_mask",
            None,
        )
        image_padding = (
            full_image_padding[:, 1:]
            if full_image_padding is not None
            else None
        )
        text_padding = inputs["attention_mask"][:, 1:] == 0
        fused_output = model.fusion(
            image_tokens,
            text_tokens,
            img_key_padding_mask=image_padding,
            txt_key_padding_mask=text_padding,
            return_attn=True,
        )
        if isinstance(fused_output, tuple):
            fused, _ = fused_output
        else:
            fused = fused_output
        logits = model.head(fused) if hasattr(model, "head") else model.classifier(fused)

        global_fg = inputs.get("foreground_box", (0, 0, 224, 224))
        global_feat = getattr(model.backbone, "last_global_feature", image_tokens[:, 0]).detach()

    cached = {
        "global_feature": global_feat,
        "visual_global_embedding": image_tokens[:, 0].detach(),
        "image_tokens": image_tokens.detach(),
        "text_tokens": text_tokens.detach(),
        "text_attention_mask": inputs["attention_mask"],
        "text_input_ids": inputs["input_ids"],
        "global_pixel_values": inputs["pixel_values"].detach(),
        "global_foreground_box": global_fg,
        "text_pad_token_id": resolve_pad_token_id(
            model.backbone.tokenizer_obj
        ),
    }
    tokenizer = getattr(
        model.backbone.tokenizer_obj,
        "tokenizer",
        model.backbone.tokenizer_obj,
    )
    special_ids = {
        int(value)
        for value in (getattr(tokenizer, "all_special_ids", None) or [])
    }
    cached["text_special_token_mask"] = torch.zeros_like(
        inputs["input_ids"],
        dtype=torch.bool,
    )
    for special_id in special_ids:
        cached["text_special_token_mask"] |= (
            inputs["input_ids"] == special_id
        )
    return fused.detach(), logits.detach(), cached


def _decoded_text_attribution(
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    scores: np.ndarray,
) -> tuple[list[str], np.ndarray]:
    """Thực hiện bước decoded văn bản attribution trong quy trình hiện tại.

    Parameters
    ----------
    tokenizer : object
        Giá trị ``tokenizer`` được sử dụng trong phép xử lý.
    input_ids : torch.Tensor
        Dữ liệu nguồn của phép xử lý.
    attention_mask : torch.Tensor
        Giá trị ``attention_mask`` được sử dụng trong phép xử lý.
    scores : np.ndarray
        Giá trị ``scores`` được sử dụng trong phép xử lý.

    Returns
    -------
    tuple[list[str], np.ndarray]
        Kết quả được tạo bởi bước xử lý của hàm.

    Raises
    ------
    RuntimeError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    """

    inner = getattr(tokenizer, "tokenizer", tokenizer)
    converter = getattr(inner, "convert_ids_to_tokens", None)
    if not callable(converter):
        raise RuntimeError("Tokenizer does not expose convert_ids_to_tokens().")
    special_tokens = set(getattr(inner, "all_special_tokens", []) or [])
    ids = [int(value) for value in input_ids[0].detach().cpu().tolist()]
    valid = [
        bool(value) for value in attention_mask[0].detach().cpu().tolist()
    ]
    raw_tokens = [str(value) for value in converter(ids)]

    words: list[str] = []
    word_scores: list[float] = []
    for token, is_valid, score in zip(raw_tokens, valid, scores):
        if not is_valid or token in special_tokens:
            continue
        cleaned = token.replace("▁", "").strip()
        if not cleaned:
            continue
        if cleaned.startswith("##") and words:
            words[-1] += cleaned[2:]
            word_scores[-1] += float(score)
        else:
            words.append(cleaned.removeprefix("##"))
            word_scores.append(float(score))

    values = np.asarray(word_scores, dtype=np.float32)
    if values.size:
        scale = float(np.max(np.abs(values), initial=0.0))
        if scale > 0:
            values = values / scale
    return words, values


class OnlineInferenceEngine:
    """Điều phối suy luận trực tuyến bằng lớp ``OnlineInferenceEngine``.

    Notes
    -----
    Lớp này đóng gói trạng thái và hành vi để các thành phần khác có thể tái sử dụng nhất quán.
    """

    def __init__(
        self,
        seed: int = 42,
        device: str | torch.device | None = None,
        *,
        experiment_name: str = SOURCE_EXPERIMENT,
        enable_ood: bool = True,
        enable_confidence_calibration: bool = True,
        target_id_fpr: float = 0.05,
        knn_k: int = 5,
    ) -> None:
        """Thực hiện bước init trong quy trình hiện tại.

        Parameters
        ----------
        seed : int, optional
            Hạt giống phục vụ khả năng tái lập.
        device : str | torch.device | None, optional
            Thiết bị thực thi phép tính.
        experiment_name : str, optional
            Tên hoặc khóa định danh của giá trị.
        enable_ood : bool, optional
            Giá trị ``enable_ood`` được sử dụng trong phép xử lý.
        enable_confidence_calibration : bool, optional
            Có khớp temperature scaling trên CTCH validation-ID hay không.
        target_id_fpr : float, optional
            Nhãn hoặc chỉ số lớp liên quan.
        knn_k : int, optional
            Giá trị ``knn_k`` được sử dụng trong phép xử lý.
        """
        self.seed = int(seed)
        if device is None or str(device).lower() == "auto":
            resolved_device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            resolved_device = torch.device(device)
        self.experiment_name = str(experiment_name).strip("/")
        if self.experiment_name == SOURCE_EXPERIMENT:
            self.loaded = load_locked_proposed_model(
                self.seed,
                device=resolved_device,
                strict_fingerprint=True,
            )
        else:
            try:
                self.loaded = load_proposed_experiment_model(
                    self.experiment_name,
                    self.seed,
                    device=resolved_device,
                )
            except (ValueError, FileNotFoundError):
                self.loaded = load_evaluated_classification_model(
                    self.experiment_name,
                    self.seed,
                    device=resolved_device,
                )
        for parameter in self.loaded.model.parameters():
            parameter.requires_grad_(False)
        self.loaded.model.eval()
        if hasattr(self.loaded.model.backbone, "explain_mode"):
            self.loaded.model.backbone.explain_mode = True
        self.knn_k = int(knn_k)
        self.confidence_calibration = (
            fit_locked_confidence_calibration(self.loaded)
            if enable_confidence_calibration
            and self.experiment_name == SOURCE_EXPERIMENT
            else None
        )
        self.ood = (
            fit_locked_ood_calibration(
                self.loaded,
                target_id_fpr=target_id_fpr,
                knn_k=self.knn_k,
            )
            if enable_ood
            else None
        )
        self.class_labels = [
            str(value) for value in self.loaded.cfg.dataset.params.classes
        ]

    @property
    def device(self) -> torch.device:
        """Thực hiện bước thiết bị trong quy trình hiện tại.

        Returns
        -------
        torch.device
            Kết quả được tạo bởi bước xử lý của hàm.
        """
        return self.loaded.device

    def _nearest_training_images(
        self,
        global_feature: torch.Tensor,
        count: int,
    ) -> list[SimilarImageReference]:
        """Thực hiện bước nearest training các ảnh trong quy trình hiện tại.

        Parameters
        ----------
        global_feature : torch.Tensor
            Biểu diễn đặc trưng cần xử lý.
        count : int
            Số lượng, kích thước hoặc tỷ lệ được sử dụng.

        Returns
        -------
        list[SimilarImageReference]
            Kết quả được tạo bởi bước xử lý của hàm.

        Raises
        ------
        RuntimeError
            Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
        """

        if self.ood is None or int(count) <= 0:
            return []
        references = np.asarray(
            self.ood.reference_visual_embeddings,
            dtype=np.float64,
        )
        if references.ndim != 2 or references.shape[0] == 0:
            return []
        reference_norms = np.linalg.norm(references, axis=1, keepdims=True)
        references = references / np.maximum(reference_norms, 1e-12)

        query = global_feature.detach().float()
        if query.ndim == 1:
            query = query.unsqueeze(0)
        query_array = F.normalize(query, dim=-1)[0].cpu().numpy().astype(np.float64)
        if query_array.shape[0] != references.shape[1]:
            raise RuntimeError(
                "Online and CTCH-train global visual embedding dimensions differ."
            )

        similarities = references @ query_array
        limit = min(max(0, int(count)), len(similarities))
        order = np.argsort(-similarities, kind="stable")[:limit]
        labels = self.ood.normalized_detector.ref_labels
        if labels is None or len(labels) != len(references):
            raise RuntimeError("CTCH-train reference labels are unavailable.")

        nearest: list[SimilarImageReference] = []
        for index in order:
            class_index = int(labels[index])
            class_label = (
                self.class_labels[class_index]
                if 0 <= class_index < len(self.class_labels)
                else f"Class {class_index}"
            )
            nearest.append(
                SimilarImageReference(
                    image_id=str(self.ood.reference_image_ids[index]),
                    class_index=class_index,
                    class_label=class_label,
                    cosine_similarity=float(similarities[index]),
                )
            )
        return nearest

    def predict(
        self,
        image: Image.Image,
        clinical_text: str,
        *,
        ood_method: str = "mahalanobis_centroid",
        ig_steps: int = 16,
        compute_faithfulness: bool = False,
        compute_global_ig: bool = False,
        similar_image_count: int = 4,
    ) -> OnlineInferenceResult:
        """Dự đoán kết quả cho bước xử lý hiện tại.

        Parameters
        ----------
        image : Image.Image
            Ảnh hoặc biểu diễn ảnh đầu vào.
        clinical_text : str
            Văn bản hoặc biểu diễn văn bản đầu vào.
        ood_method : str, optional
            Phương pháp hoặc chế độ xử lý được chọn.
        ig_steps : int, optional
            Giá trị ``ig_steps`` được sử dụng trong phép xử lý.
        compute_faithfulness : bool, optional
            Giá trị ``compute_faithfulness`` được sử dụng trong phép xử lý.
        compute_global_ig : bool, optional
            Giá trị ``compute_global_ig`` được sử dụng trong phép xử lý.
        similar_image_count : int, optional
            Số lượng, kích thước hoặc tỷ lệ được sử dụng.

        Returns
        -------
        OnlineInferenceResult
            Kết quả được tạo bởi bước xử lý của hàm.

        Raises
        ------
        ValueError
            Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
        """
        clinical_text = str(clinical_text).strip()
        if not clinical_text:
            raise ValueError("Clinical text must not be empty.")
        if ood_method not in OOD_METHODS:
            raise ValueError(f"Unsupported OOD method: {ood_method}")
        if int(ig_steps) < 2:
            raise ValueError("Integrated Gradients requires at least two steps.")

        inputs = _build_online_inputs(
            self.loaded,
            image.convert("RGB"),
            clinical_text,
        )
        fused_raw, logits, cached = _forward_and_cache(self.loaded, inputs)
        raw_probabilities = torch.softmax(logits, dim=-1)[0]
        raw_probabilities_array = raw_probabilities.detach().cpu().numpy()
        logits_array = logits.detach().cpu().numpy()
        if self.confidence_calibration is None:
            calibrated_probabilities = raw_probabilities_array
            calibration_temperature = 1.0
            confidence_is_calibrated = False
        else:
            calibration_temperature = self.confidence_calibration.temperature
            calibrated_probabilities = calibrated_softmax(
                logits_array,
                calibration_temperature,
            )[0]
            confidence_is_calibrated = True
        # Bảng xếp hạng dùng softmax gốc; temperature scaling chỉ hiệu chỉnh
        # confidence của lớp đứng đầu và không làm thay đổi thứ tự lớp.
        predicted_index = int(raw_probabilities_array.argmax())
        confidence = float(calibrated_probabilities[predicted_index])
        similar_images = self._nearest_training_images(
            cached["visual_global_embedding"],
            int(similar_image_count),
        )

        fused_embedding_raw = fused_raw.cpu().numpy()
        fused_embedding = F.normalize(fused_raw, dim=-1).cpu().numpy()
        if self.ood is None:
            score = float("nan")
            threshold = float("nan")
            is_ood = False
        else:
            score = float(
                _score_method(
                    ood_method,
                    self.ood.normalized_detector,
                    self.ood.raw_detector,
                    fused_embedding=fused_embedding,
                    fused_embedding_raw=fused_embedding_raw,
                    logits=logits_array,
                    knn_k=self.knn_k,
                )[0]
            )
            threshold = float(self.ood.thresholds[ood_method])
            is_ood = bool(score > threshold)

        # Không công bố nhãn hoặc confidence khi mẫu bị phát hiện là OOD.
        if is_ood:
            return OnlineInferenceResult(
                predicted_index=None,
                predicted_label=None,
                probabilities=raw_probabilities_array,
                confidence=None,
                confidence_is_calibrated=confidence_is_calibrated,
                calibration_temperature=calibration_temperature,
                class_labels=self.class_labels,
                ood_method=ood_method,
                ood_score=score,
                ood_threshold=threshold,
                is_ood=True,
                similar_images=similar_images,
                global_ig_map=None,
                text_tokens=None,
                text_ig_scores=None,
                contribution_drops=None,
                global_faithfulness=None,
                text_faithfulness=None,
                seed=self.seed,
                checkpoint=str(self.loaded.checkpoint),
            )

        self.loaded.model.zero_grad(set_to_none=True)
        text_ig_relevance, text_ig_attribution = integrated_gradients_text(
            self.loaded.model,
            cached,
            predicted_index,
            steps=int(ig_steps),
        )
        text_ig_scores = text_ig_attribution.sum(dim=-1)
        global_ig_map = None
        global_relevance = None
        if compute_global_ig:
            global_relevance, _ = integrated_gradients_global_image(
                self.loaded.model,
                cached,
                inputs["pixel_values"],
                predicted_index,
                steps=int(ig_steps),
            )
            global_ig_map = project_global_ig_to_source(
                global_relevance.detach().cpu().numpy(),
                cached["global_foreground_box"],
                image.size,
                patch_grid=_global_patch_grid(self.loaded.model),
            )
        global_faithfulness = None
        text_faithfulness = None
        contribution_drops = None
        if compute_faithfulness:
            fractions = np.linspace(0.0, 1.0, 11, dtype=np.float64)
            global_curves = None
            if global_relevance is not None:
                global_curves = global_image_perturbation_curves(
                    self.loaded.model,
                    cached,
                    inputs["pixel_values"],
                    global_relevance,
                    predicted_index,
                    fractions=fractions,
                    random_trials=1,
                    seed=self.seed + 2000,
                )
            text_curves = text_input_perturbation_curves(
                self.loaded.model,
                cached,
                text_ig_relevance,
                predicted_index,
                fractions=fractions,
                random_trials=1,
                seed=self.seed + 3000,
            )
            if global_curves is not None:
                global_faithfulness = {
                    "fractions": global_curves["fractions"],
                    "deletion": global_curves["delete_most_relevant"],
                    "random_deletion": global_curves["delete_random"],
                    "insertion": global_curves["insert_most_relevant"],
                    "deletion_auc": curve_auc(
                        global_curves["fractions"],
                        global_curves["delete_most_relevant"],
                    ),
                    "insertion_auc": curve_auc(
                        global_curves["fractions"],
                        global_curves["insert_most_relevant"],
                    ),
                    "random_deletion_auc": curve_auc(
                        global_curves["fractions"],
                        global_curves["delete_random"],
                    ),
                }
                global_faithfulness["random_minus_targeted_deletion_auc"] = (
                    global_faithfulness["random_deletion_auc"]
                    - global_faithfulness["deletion_auc"]
                )
            text_faithfulness = {
                "fractions": text_curves["fractions"],
                "deletion": text_curves["delete_most_relevant"],
                "random_deletion": text_curves["delete_random"],
                "insertion": text_curves["insert_most_relevant"],
                "deletion_auc": curve_auc(
                    text_curves["fractions"],
                    text_curves["delete_most_relevant"],
                ),
                "insertion_auc": curve_auc(
                    text_curves["fractions"],
                    text_curves["insert_most_relevant"],
                ),
                "random_deletion_auc": curve_auc(
                    text_curves["fractions"],
                    text_curves["delete_random"],
                ),
            }
            text_faithfulness["random_minus_targeted_deletion_auc"] = (
                text_faithfulness["random_deletion_auc"]
                - text_faithfulness["deletion_auc"]
            )
            if global_faithfulness is not None:
                full_probability = float(raw_probabilities[predicted_index])
                contribution_drops = {
                    "global_visual": full_probability
                    - float(global_faithfulness["deletion"][-1]),
                    "clinical_text": full_probability
                    - float(text_faithfulness["deletion"][-1]),
                }
        decoded_tokens, decoded_text_scores = _decoded_text_attribution(
            self.loaded.model.backbone.tokenizer_obj,
            inputs["input_ids"],
            inputs["attention_mask"],
            text_ig_scores.detach().cpu().numpy(),
        )

        return OnlineInferenceResult(
            predicted_index=predicted_index,
            predicted_label=self.class_labels[predicted_index],
            probabilities=raw_probabilities_array,
            confidence=confidence,
            confidence_is_calibrated=confidence_is_calibrated,
            calibration_temperature=calibration_temperature,
            class_labels=self.class_labels,
            ood_method=ood_method,
            ood_score=score,
            ood_threshold=threshold,
            is_ood=False,
            similar_images=similar_images,
            global_ig_map=global_ig_map,
            text_tokens=decoded_tokens,
            text_ig_scores=decoded_text_scores,
            contribution_drops=contribution_drops,
            global_faithfulness=global_faithfulness,
            text_faithfulness=text_faithfulness,
            seed=self.seed,
            checkpoint=str(self.loaded.checkpoint),
        )


def _global_patch_grid(model) -> tuple[int, int]:
    """Thực hiện bước global patch grid trong quy trình hiện tại.

    Parameters
    ----------
    model : object
        Mô hình hoặc thành phần mô hình cần xử lý.

    Returns
    -------
    tuple[int, int]
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    trunk = getattr(
        getattr(model.backbone.model, "visual", None),
        "trunk",
        None,
    )
    patch_embed = getattr(trunk, "patch_embed", None)
    grid_size = getattr(patch_embed, "grid_size", (14, 14))
    if isinstance(grid_size, int):
        return int(grid_size), int(grid_size)
    return int(grid_size[0]), int(grid_size[1])


def project_global_ig_to_source(
    heatmap: np.ndarray,
    foreground_box: tuple[int, int, int, int],
    image_size: tuple[int, int],
    *,
    patch_grid: tuple[int, int] | None = None,
) -> np.ndarray:
    """Thực hiện bước project global ig to source trong quy trình hiện tại.

    Parameters
    ----------
    heatmap : np.ndarray
        Giá trị ``heatmap`` được sử dụng trong phép xử lý.
    foreground_box : tuple[int, int, int, int]
        Giá trị ``foreground_box`` được sử dụng trong phép xử lý.
    image_size : tuple[int, int]
        Số lượng, kích thước hoặc tỷ lệ được sử dụng.
    patch_grid : tuple[int, int] | None, optional
        Giá trị ``patch_grid`` được sử dụng trong phép xử lý.

    Returns
    -------
    np.ndarray
        Kết quả được tạo bởi bước xử lý của hàm.

    Raises
    ------
    ValueError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    """

    values = np.maximum(np.asarray(heatmap, dtype=np.float32), 0.0)
    if values.ndim != 2:
        raise ValueError("Global IG heatmap must be two-dimensional.")
    canvas_height, canvas_width = values.shape
    if patch_grid is not None:
        grid_height, grid_width = patch_grid
        values = np.asarray(
            Image.fromarray(values, mode="F")
            .resize(
                (grid_width, grid_height),
                Image.Resampling.BOX,
            )
            .resize(
                (canvas_width, canvas_height),
                Image.Resampling.BICUBIC,
            ),
            dtype=np.float32,
        )
        values = np.maximum(values, 0.0)
    left, top, right, bottom = (int(value) for value in foreground_box)
    foreground_width = max(1, right - left)
    foreground_height = max(1, bottom - top)
    scale = min(
        canvas_width / foreground_width,
        canvas_height / foreground_height,
    )
    resized_width = max(
        1,
        min(canvas_width, round(foreground_width * scale)),
    )
    resized_height = max(
        1,
        min(canvas_height, round(foreground_height * scale)),
    )
    x_offset = (canvas_width - resized_width) // 2
    y_offset = (canvas_height - resized_height) // 2
    content = values[
        y_offset:y_offset + resized_height,
        x_offset:x_offset + resized_width,
    ]
    content_image = Image.fromarray(content, mode="F").resize(
        (foreground_width, foreground_height),
        Image.Resampling.BICUBIC,
    )
    source_width, source_height = image_size
    source = np.zeros((source_height, source_width), dtype=np.float32)
    clipped_left = max(0, left)
    clipped_top = max(0, top)
    clipped_right = min(source_width, right)
    clipped_bottom = min(source_height, bottom)
    restored = np.asarray(content_image, dtype=np.float32)
    source[
        clipped_top:clipped_bottom,
        clipped_left:clipped_right,
    ] = restored[
        clipped_top - top:clipped_bottom - top,
        clipped_left - left:clipped_right - left,
    ]
    positive = source[source > 0]
    if positive.size:
        low, high = np.percentile(positive, [1.0, 99.0])
        source = np.clip(
            (source - float(low)) / max(float(high - low), 1e-8),
            0.0,
            1.0,
        )
    return source


def _render_heatmap_overlay(
    image: Image.Image,
    heatmap: np.ndarray,
    *,
    alpha: float,
) -> Image.Image:
    """Kết xuất heatmap overlay cho bước xử lý hiện tại.

    Parameters
    ----------
    image : Image.Image
        Ảnh hoặc biểu diễn ảnh đầu vào.
    heatmap : np.ndarray
        Giá trị ``heatmap`` được sử dụng trong phép xử lý.
    alpha : float
        Giá trị ``alpha`` được sử dụng trong phép xử lý.

    Returns
    -------
    Image.Image
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    resized = image.convert("RGB").resize(
        (heatmap.shape[1], heatmap.shape[0]),
        Image.Resampling.LANCZOS,
    )
    base = np.asarray(resized, dtype=np.float32) / 255.0
    color = colormaps["turbo"](heatmap)[..., :3].astype(np.float32)
    opacity = (float(alpha) * np.power(heatmap, 0.7))[..., None]
    overlay = np.clip(base * (1.0 - opacity) + color * opacity, 0.0, 1.0)
    return Image.fromarray(np.uint8(overlay * 255.0), mode="RGB")


def render_global_ig_overlay(
    image: Image.Image,
    result: OnlineInferenceResult,
    *,
    alpha: float = 0.58,
) -> Image.Image:
    """Kết xuất global ig overlay cho bước xử lý hiện tại.

    Parameters
    ----------
    image : Image.Image
        Ảnh hoặc biểu diễn ảnh đầu vào.
    result : OnlineInferenceResult
        Giá trị ``result`` được sử dụng trong phép xử lý.
    alpha : float, optional
        Giá trị ``alpha`` được sử dụng trong phép xử lý.

    Returns
    -------
    Image.Image
        Kết quả được tạo bởi bước xử lý của hàm.

    Raises
    ------
    ValueError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    """

    if result.global_ig_map is None:
        raise ValueError("Global-image Integrated Gradients is unavailable.")
    return _render_heatmap_overlay(
        image,
        result.global_ig_map,
        alpha=alpha,
    )
