"""Cung cấp tiện ích ood cho huấn luyện, đánh giá và phân tích XBone-Net.

Notes
-----
Mô-đun này thuộc cơ sở mã nguồn nghiên cứu XBone-Net và giữ các quy ước dùng chung của dự án.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


OOD_PROTOCOL_VERSION = 4
MAHALANOBIS_SCORE_DEFINITION = (
    "minimum_class_conditional_mahalanobis_distance_to_empirical_centroid"
)
COSINE_CENTROIDS_SCORE_DEFINITION = (
    "one_minus_maximum_cosine_similarity_to_empirical_centroid"
)


def _embedding_matrix(array: np.ndarray, name: str = "embeddings") -> np.ndarray:
    """Kiểm tra và chuẩn hóa ma trận biểu diễn đặc trưng.

    Parameters
    ----------
    array : np.ndarray
        Giá trị ``array`` được sử dụng trong phép xử lý.
    name : str, optional
        Tên hoặc khóa định danh của giá trị.

    Returns
    -------
    np.ndarray
        Kết quả được tạo bởi bước xử lý của hàm.

    Raises
    ------
    ValueError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    """
    values = np.asarray(array, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"Expected a 2-D {name} matrix, got {values.shape}.")
    if values.shape[1] < 1:
        raise ValueError(f"{name} must contain at least one feature.")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains NaN or infinity.")
    return values


def _l2_normalize(array: np.ndarray) -> np.ndarray:
    """Chuẩn hóa các vectơ theo chuẩn L2.

    Parameters
    ----------
    array : np.ndarray
        Giá trị ``array`` được sử dụng trong phép xử lý.

    Returns
    -------
    np.ndarray
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    array = _embedding_matrix(array)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(norms, 1e-12)


def _finite_scores(scores: np.ndarray, name: str) -> np.ndarray:
    """Kiểm tra và chuẩn hóa dãy điểm hữu hạn.

    Parameters
    ----------
    scores : np.ndarray
        Giá trị ``scores`` được sử dụng trong phép xử lý.
    name : str
        Tên hoặc khóa định danh của giá trị.

    Returns
    -------
    np.ndarray
        Kết quả được tạo bởi bước xử lý của hàm.

    Raises
    ------
    ValueError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    """
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains NaN or infinity.")
    return values


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Thực hiện bước softmax trong quy trình hiện tại.

    Parameters
    ----------
    logits : np.ndarray
        Giá trị ``logits`` được sử dụng trong phép xử lý.

    Returns
    -------
    np.ndarray
        Kết quả được tạo bởi bước xử lý của hàm.

    Raises
    ------
    ValueError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    """
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2 or logits.shape[1] < 2:
        raise ValueError("logits must have shape [N,C] with C >= 2.")
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.maximum(exponentials.sum(axis=1, keepdims=True), 1e-12)


class OODDetector:
    """Phát hiện dữ liệu ngoài phân phối bằng lớp ``OODDetector``.

    Notes
    -----
    Lớp này đóng gói trạng thái và hành vi để các thành phần khác có thể tái sử dụng nhất quán.
    """

    def __init__(self):
        """Thực hiện bước init trong quy trình hiện tại."""
        self.class_means: dict[int, np.ndarray] = {}
        self.shared_cov_inv: Optional[np.ndarray] = None
        self.ref_embeddings: Optional[np.ndarray] = None
        self.ref_labels: Optional[np.ndarray] = None
        self._fitted = False

    def fit(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        prototypes: Optional[np.ndarray] = None,
    ):
        # Tính điểm và độ đo phát hiện dữ liệu ngoài phân phối.
        # Bước hỗ trợ để khớp kết quả cho bước xử lý hiện tại.
        # Bước hỗ trợ để khớp kết quả cho bước xử lý hiện tại.
        # Bước hỗ trợ để khớp kết quả cho bước xử lý hiện tại.
        # Thiết lập giá trị trung gian cho bước xử lý tiếp theo.
        """Khớp kết quả cho bước xử lý hiện tại.

        Parameters
        ----------
        embeddings : np.ndarray
            Giá trị ``embeddings`` được sử dụng trong phép xử lý.
        labels : np.ndarray
            Giá trị ``labels`` được sử dụng trong phép xử lý.
        prototypes : Optional[np.ndarray]
            Giá trị ``prototypes`` được sử dụng trong phép xử lý.

        Returns
        -------
        object
            Kết quả được tạo bởi bước xử lý của hàm.

        Raises
        ------
        ValueError
            Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
        """
        embeddings = _embedding_matrix(embeddings)
        labels = np.asarray(labels)
        if labels.ndim == 2:
            labels = np.argmax(labels, axis=1)
        labels = labels.astype(np.int64).reshape(-1)
        if len(labels) != len(embeddings):
            raise ValueError("Embedding and label counts differ.")
        if len(embeddings) < 2:
            raise ValueError("At least two reference samples are required.")
        if np.any(labels < 0):
            raise ValueError("Reference labels must be non-negative class IDs.")

        self.ref_embeddings = _l2_normalize(embeddings)
        self.ref_labels = labels
        unique_classes = np.unique(labels)
        if unique_classes.size < 1:
            raise ValueError("Reference labels contain no classes.")

        prototype_centers = None
        if prototypes is not None:
            prototype_centers = _embedding_matrix(prototypes, "prototype")
            if prototype_centers.shape[1] != embeddings.shape[1]:
                raise ValueError(
                    "Prototype and reference embedding dimensions differ."
                )

        self.class_means = {}
        centered_parts = []
        for class_id in unique_classes:
            class_embeddings = embeddings[labels == class_id]
            if (
                prototype_centers is not None
                and 0 <= class_id < len(prototype_centers)
            ):
                center = prototype_centers[class_id]
            else:
                center = class_embeddings.mean(axis=0)
            self.class_means[int(class_id)] = center
            centered_parts.append(class_embeddings - center)

        # Tính hoặc cập nhật các tâm lớp trong không gian biểu diễn.
        # Chuẩn bị dữ liệu và chiến lược lấy mẫu tương ứng.
        if prototype_centers is not None:
            for class_id, center in enumerate(prototype_centers):
                self.class_means.setdefault(int(class_id), center)

        centered = np.vstack(centered_parts)
        covariance = LedoitWolf().fit(centered).covariance_
        self.shared_cov_inv = np.linalg.pinv(covariance, hermitian=True)
        self._fitted = True
        return self

    def score_mahalanobis_centroid(
        self, test_embeddings: np.ndarray
    ) -> np.ndarray:
        """Tính điểm mahalanobis tâm lớp cho bước xử lý hiện tại.

        Parameters
        ----------
        test_embeddings : np.ndarray
            Giá trị ``test_embeddings`` được sử dụng trong phép xử lý.

        Returns
        -------
        np.ndarray
            Kết quả được tạo bởi bước xử lý của hàm.

        Raises
        ------
        RuntimeError
            Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
        ValueError
            Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
        """
        if not self._fitted or self.shared_cov_inv is None:
            raise RuntimeError("Call fit() before Mahalanobis scoring.")
        test_embeddings = _embedding_matrix(test_embeddings, "test embedding")
        if test_embeddings.shape[1] != self.shared_cov_inv.shape[0]:
            raise ValueError(
                "Test and reference embedding dimensions differ."
            )
        squared_scores = np.full(
            test_embeddings.shape[0], np.inf, dtype=np.float64
        )
        for center in self.class_means.values():
            diff = test_embeddings - center
            squared_distances = np.einsum(
                "ni,ij,nj->n", diff, self.shared_cov_inv, diff
            )
            squared_scores = np.minimum(squared_scores, squared_distances)
        # Tính và tổng hợp các độ đo đánh giá cần thiết.
        # do sai số làm tròn số thực. Việc chặn tại 0 giữ nguyên tính chất
        # Bước hỗ trợ để tính điểm mahalanobis centroid cho bước xử lý hiện tại.
        return np.sqrt(np.maximum(squared_scores, 0.0))

    def score_mahalanobis(self, test_embeddings: np.ndarray) -> np.ndarray:
        """Tính điểm mahalanobis cho bước xử lý hiện tại.

        Parameters
        ----------
        test_embeddings : np.ndarray
            Giá trị ``test_embeddings`` được sử dụng trong phép xử lý.

        Returns
        -------
        np.ndarray
            Kết quả được tạo bởi bước xử lý của hàm.
        """
        return self.score_mahalanobis_centroid(test_embeddings)

    def score_cosine_centroids(
        self, test_embeddings: np.ndarray
    ) -> np.ndarray:
        """Tính điểm cosine các tâm lớp cho bước xử lý hiện tại.

        Parameters
        ----------
        test_embeddings : np.ndarray
            Giá trị ``test_embeddings`` được sử dụng trong phép xử lý.

        Returns
        -------
        np.ndarray
            Kết quả được tạo bởi bước xử lý của hàm.

        Raises
        ------
        RuntimeError
            Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
        ValueError
            Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
        """
        if not self._fitted or not self.class_means:
            raise RuntimeError("Call fit() before cosine-centroid scoring.")
        test_embeddings = _l2_normalize(test_embeddings)
        centers = _l2_normalize(np.stack(list(self.class_means.values()), axis=0))
        if test_embeddings.shape[1] != centers.shape[1]:
            raise ValueError(
                "Test and empirical-centroid embedding dimensions differ."
            )
        return 1.0 - (test_embeddings @ centers.T).max(axis=1)

    def score_knn(
        self,
        test_embeddings: np.ndarray,
        k: int = 10,
        reduction: str = "kth",
        metric: str = "euclidean",
        exclude_self: bool = False,
    ) -> np.ndarray:
        """Tính điểm kNN OOD theo phương pháp của Sun et al. (ICML 2022 - KNN-OOD).

        Parameters
        ----------
        test_embeddings : np.ndarray
            Ma trận embedding của các mẫu kiểm thử.
        k : int, optional
            Số lân cận thứ k (mặc định k=10).
        reduction : str, optional
            Quy tắc xác định điểm số OOD:
            - "kth" (mặc định theo Sun et al.): Khoảng cách tới lân cận thứ k r_k.
            - "mean": Khoảng cách trung bình tới k lân cận gần nhất.
        metric : str, optional
            Hàm khoảng cách:
            - "euclidean" (mặc định): Khoảng cách Euclid trên các vector chuẩn hóa L2.
            - "cosine": 1 - cosine similarity.
        exclude_self : bool, optional
            Loại bỏ phần tử đường chéo khi embedding kiểm thử trùng embedding tham chiếu.

        Returns
        -------
        np.ndarray
            Dãy điểm số OOD cho từng mẫu kiểm thử.

        Raises
        ------
        RuntimeError
            Khi bộ dò chưa được huấn luyện bằng fit().
        ValueError
            Khi dữ liệu hoặc cấu hình đầu vào không hợp lệ.
        """
        if not self._fitted or self.ref_embeddings is None:
            raise RuntimeError("Call fit() before k-NN scoring.")
        test_embeddings = _l2_normalize(test_embeddings)
        sim = test_embeddings @ self.ref_embeddings.T

        if metric == "euclidean":
            # Khoảng cách Euclid trên các vector chuẩn hóa L2: ||u - v|| = sqrt(2 - 2 * cos(u, v))
            distances = np.sqrt(np.maximum(2.0 * (1.0 - sim), 0.0))
        elif metric == "cosine":
            distances = 1.0 - sim
        else:
            raise ValueError(f"Unknown metric '{metric}'. Choose 'euclidean' or 'cosine'.")

        if exclude_self and distances.shape[0] == distances.shape[1]:
            np.fill_diagonal(distances, np.inf)

        excluded = int(exclude_self and distances.shape[0] == distances.shape[1])
        n_available = distances.shape[1] - excluded
        if n_available < 1:
            raise ValueError("No reference neighbor is available for k-NN scoring.")
        k = max(1, min(int(k), n_available))

        if reduction == "kth":
            # Theo Sun et al. (ICML 2022): Khoảng cách tới lân cận thứ k r_k(x)
            return np.partition(distances, kth=k - 1, axis=1)[:, k - 1]
        elif reduction == "mean":
            nearest = np.partition(distances, kth=k - 1, axis=1)[:, :k]
            return nearest.mean(axis=1)
        else:
            raise ValueError(f"Unknown reduction '{reduction}'. Choose 'kth' or 'mean'.")

    @staticmethod
    def score_energy(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """Tính điểm energy cho bước xử lý hiện tại.

        Parameters
        ----------
        logits : np.ndarray
            Giá trị ``logits`` được sử dụng trong phép xử lý.
        temperature : float, optional
            Giá trị ``temperature`` được sử dụng trong phép xử lý.

        Returns
        -------
        np.ndarray
            Kết quả được tạo bởi bước xử lý của hàm.

        Raises
        ------
        ValueError
            Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
        """
        logits = np.asarray(logits, dtype=np.float64)
        if logits.ndim != 2:
            raise ValueError("logits must have shape [N,C].")
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        scaled = logits / temperature
        row_max = scaled.max(axis=1, keepdims=True)
        logsumexp = row_max[:, 0] + np.log(np.exp(scaled - row_max).sum(axis=1))
        return -temperature * logsumexp

    @staticmethod
    def score_msp(logits: np.ndarray) -> np.ndarray:
        """Tính điểm msp cho bước xử lý hiện tại.

        Parameters
        ----------
        logits : np.ndarray
            Giá trị ``logits`` được sử dụng trong phép xử lý.

        Returns
        -------
        np.ndarray
            Kết quả được tạo bởi bước xử lý của hàm.
        """
        return 1.0 - _softmax(logits).max(axis=1)

    @staticmethod
    def score_entropy(logits: np.ndarray) -> np.ndarray:
        """Tính điểm entropy cho bước xử lý hiện tại.

        Parameters
        ----------
        logits : np.ndarray
            Giá trị ``logits`` được sử dụng trong phép xử lý.

        Returns
        -------
        np.ndarray
            Kết quả được tạo bởi bước xử lý của hàm.
        """
        probabilities = _softmax(logits)
        entropy = -(probabilities * np.log(np.maximum(probabilities, 1e-12))).sum(axis=1)
        return entropy / np.log(probabilities.shape[1])

    @staticmethod
    def score_max_logit(logits: np.ndarray) -> np.ndarray:
        """Tính điểm max logit cho bước xử lý hiện tại.

        Parameters
        ----------
        logits : np.ndarray
            Giá trị ``logits`` được sử dụng trong phép xử lý.

        Returns
        -------
        np.ndarray
            Kết quả được tạo bởi bước xử lý của hàm.

        Raises
        ------
        ValueError
            Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
        """
        logits = np.asarray(logits, dtype=np.float64)
        if logits.ndim != 2:
            raise ValueError("logits must have shape [N,C].")
        return -logits.max(axis=1)

    @staticmethod
    def score_text_anchor(
        test_img_embeddings: np.ndarray,
        anchor_text_embeddings: np.ndarray,
    ) -> np.ndarray:
        """Tính điểm văn bản anchor cho bước xử lý hiện tại.

        Parameters
        ----------
        test_img_embeddings : np.ndarray
            Ảnh hoặc biểu diễn ảnh đầu vào.
        anchor_text_embeddings : np.ndarray
            Văn bản hoặc biểu diễn văn bản đầu vào.

        Returns
        -------
        np.ndarray
            Kết quả được tạo bởi bước xử lý của hàm.
        """
        test_img_embeddings = _l2_normalize(test_img_embeddings)
        anchor_text_embeddings = _l2_normalize(anchor_text_embeddings)
        return 1.0 - (test_img_embeddings @ anchor_text_embeddings.T).max(axis=1)

    def score(
        self,
        test_embeddings: np.ndarray,
        method: str = "mahalanobis_centroid",
        **kwargs,
    ) -> np.ndarray:
        """Tính điểm kết quả cho bước xử lý hiện tại.

        Parameters
        ----------
        test_embeddings : np.ndarray
            Giá trị ``test_embeddings`` được sử dụng trong phép xử lý.
        method : str, optional
            Phương pháp hoặc chế độ xử lý được chọn.
        **kwargs : dict
            Các đối số từ khóa bổ sung.

        Returns
        -------
        np.ndarray
            Kết quả được tạo bởi bước xử lý của hàm.

        Raises
        ------
        ValueError
            Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
        """
        if method == "mahalanobis_centroid":
            return self.score_mahalanobis_centroid(test_embeddings)
        if method == "cosine_centroids":
            return self.score_cosine_centroids(test_embeddings)
        if method == "knn":
            return self.score_knn(
                test_embeddings,
                k=kwargs.get("k", 10),
                reduction=kwargs.get("reduction", "kth"),
                metric=kwargs.get("metric", "euclidean"),
                exclude_self=kwargs.get("exclude_self", False),
            )
        if method == "entropy":
            return self.score_entropy(kwargs["logits"])
        if method == "multimodal_ensemble":
            ensemble_detector = kwargs.get("ensemble_detector")
            if ensemble_detector is not None:
                return ensemble_detector.score(
                    test_embeddings,
                    text_embeddings=kwargs.get("text_embeddings"),
                    logits=kwargs.get("logits"),
                )
            raise ValueError("ensemble_detector must be provided for method 'multimodal_ensemble'.")
        raise ValueError(
            "Unknown OOD method: "
            f"{method}. Choose cosine_centroids, mahalanobis_centroid, "
            "knn, entropy, or multimodal_ensemble."
        )


class MultimodalEnsembleOODDetector:
    """Bộ phát hiện Multi-Modal Modality-Decoupled Ensemble OOD Detection.

    Nguyên lý hoạt động:
    1. Modality-Decoupled (Tách rời phương thức):
       Tuyệt đối không sử dụng fused_embedding (vốn có thể bị mạng chú ý chéo ép khớp vào
       không gian phân loại nội tại làm lu mờ tín hiệu OOD đơn lẻ). Thay vào đó, mô-đun
       sử dụng trực tiếp hai biểu diễn đơn phương thức đã chuẩn hóa L2:
       - Biểu diễn ảnh toàn cục (visual_global_embeddings)
       - Biểu diễn văn bản toàn cục (text_global_embeddings)

    2. Phương pháp kNN đơn phương thức:
       - Điểm bất thường thị giác: s_vis = score_knn(z_img, k=k)
       - Điểm bất thường văn bản: s_txt = score_knn(z_txt, k=k)

    3. Chuẩn hóa Z-score trên tập Validation ID:
       Tính trung bình mu_val và độ lệch chuẩn sigma_val trên tập xác thực ID (val_data):
       - z_vis = (s_vis - mu_vis_val) / (sigma_vis_val + eps)
       - z_txt = (s_txt - mu_txt_val) / (sigma_txt_val + eps)

    4. Hợp nhất bằng toán tử Max (Max-pooling Fusion):
       s_OOD = max(z_vis, z_txt)
       Nguyên tắc: Chỉ cần một trong hai phương thức (ảnh hoặc văn bản) có dấu hiệu
       bất thường vượt trội so với phân phối ID thì toàn bộ mẫu sẽ được đánh giá là OOD.
    """

    def __init__(
        self,
        knn_k: int = 10,
        knn_reduction: str = "kth",
        knn_metric: str = "euclidean",
    ):
        """Khởi tạo bộ phát hiện MultimodalEnsembleOODDetector."""
        self.knn_k = knn_k
        self.knn_reduction = knn_reduction
        self.knn_metric = knn_metric
        self.visual_detector = OODDetector()
        self.text_detector = OODDetector()
        self.scalers: dict[str, tuple[float, float]] = {}
        self._fitted = False

    def fit(
        self,
        visual_embeddings: np.ndarray,
        labels: np.ndarray,
        text_embeddings: Optional[np.ndarray] = None,
        logits: Optional[np.ndarray] = None,
        val_visual_embeddings: Optional[np.ndarray] = None,
        val_text_embeddings: Optional[np.ndarray] = None,
        val_logits: Optional[np.ndarray] = None,
    ):
        """Huấn luyện các bộ dò kNN đơn phương thức và chuẩn hóa z-score trên tập validation ID."""
        visual_embeddings = _embedding_matrix(visual_embeddings, "visual_embeddings")
        labels = np.asarray(labels, dtype=np.int64).reshape(-1)

        # 1. Huấn luyện bộ dò kNN trên tập huấn luyện tham chiếu ID
        self.visual_detector.fit(visual_embeddings, labels)

        if text_embeddings is not None and len(text_embeddings) == len(labels):
            text_embeddings = _embedding_matrix(text_embeddings, "text_embeddings")
            self.text_detector.fit(text_embeddings, labels)

        # 2. Cân chỉnh z-score scalers trên tập ID validation (hoặc train fallback)
        val_vis = val_visual_embeddings if val_visual_embeddings is not None else visual_embeddings
        val_txt = val_text_embeddings if val_text_embeddings is not None else text_embeddings

        # kNN score trên validation ảnh
        s_vis_val = self.visual_detector.score_knn(
            val_vis,
            k=self.knn_k,
            reduction=self.knn_reduction,
            metric=self.knn_metric,
        )
        self.scalers["vis"] = (
            float(np.mean(s_vis_val)),
            float(np.std(s_vis_val) + 1e-8),
        )

        # kNN score trên validation văn bản
        if val_txt is not None:
            s_txt_val = self.text_detector.score_knn(
                val_txt,
                k=self.knn_k,
                reduction=self.knn_reduction,
                metric=self.knn_metric,
            )
            self.scalers["txt"] = (
                float(np.mean(s_txt_val)),
                float(np.std(s_txt_val) + 1e-8),
            )

        self._fitted = True
        return self

    def score(
        self,
        visual_embeddings: np.ndarray,
        text_embeddings: Optional[np.ndarray] = None,
        logits: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Tính điểm OOD bằng kNN tách rời phương thức, chuẩn hóa Z-score và hợp nhất bằng toán tử Max."""
        if not self._fitted:
            raise RuntimeError(
                "MultimodalEnsembleOODDetector must be fitted before scoring."
            )

        visual_embeddings = _embedding_matrix(visual_embeddings, "visual_embeddings")
        n_samples = len(visual_embeddings)

        # 1. Điểm kNN ảnh và chuẩn hóa z-score
        s_vis = self.visual_detector.score_knn(
            visual_embeddings,
            k=self.knn_k,
            reduction=self.knn_reduction,
            metric=self.knn_metric,
        )
        mean_vis, std_vis = self.scalers["vis"]
        z_vis = (s_vis - mean_vis) / std_vis

        # 2. Điểm kNN văn bản và chuẩn hóa z-score
        if text_embeddings is not None and "txt" in self.scalers:
            text_embeddings = _embedding_matrix(text_embeddings, "text_embeddings")
            s_txt = self.text_detector.score_knn(
                text_embeddings,
                k=self.knn_k,
                reduction=self.knn_reduction,
                metric=self.knn_metric,
            )
            mean_txt, std_txt = self.scalers["txt"]
            z_txt = (s_txt - mean_txt) / std_txt

            # 3. Hợp nhất bằng toán tử Max (Max-pooling Fusion)
            ensemble_score = np.maximum(z_vis, z_txt)
        else:
            ensemble_score = z_vis

        return ensemble_score


MultimodalDecoupledEnsembleOODDetector = MultimodalEnsembleOODDetector


def calibrate_ood_threshold(
    calibration_id_scores: np.ndarray,
    target_id_fpr: float = 0.05,
) -> float:
    """Thực hiện bước calibrate ood threshold trong quy trình hiện tại.

    Parameters
    ----------
    calibration_id_scores : np.ndarray
        Giá trị ``calibration_id_scores`` được sử dụng trong phép xử lý.
    target_id_fpr : float, optional
        Nhãn hoặc chỉ số lớp liên quan.

    Returns
    -------
    float
        Kết quả được tạo bởi bước xử lý của hàm.

    Raises
    ------
    ValueError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    """
    scores = _finite_scores(calibration_id_scores, "calibration_id_scores")
    if not 0.0 < target_id_fpr < 1.0:
        raise ValueError("target_id_fpr must be between 0 and 1.")
    quantile = 1.0 - float(target_id_fpr)
    try:
        return float(np.quantile(scores, quantile, method="higher"))
    except TypeError:  # Tương thích với NumPy phiên bản nhỏ hơn 1.22
        return float(np.quantile(scores, quantile, interpolation="higher"))


def evaluate_ood(
    id_scores: np.ndarray,
    ood_scores: np.ndarray,
) -> dict:
    """Đánh giá ood cho bước xử lý hiện tại.

    Parameters
    ----------
    id_scores : np.ndarray
        Giá trị ``id_scores`` được sử dụng trong phép xử lý.
    ood_scores : np.ndarray
        Giá trị ``ood_scores`` được sử dụng trong phép xử lý.

    Returns
    -------
    dict
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    id_scores = _finite_scores(id_scores, "id_scores")
    ood_scores = _finite_scores(ood_scores, "ood_scores")

    labels = np.concatenate([
        np.zeros(id_scores.size, dtype=np.int64),
        np.ones(ood_scores.size, dtype=np.int64),
    ])
    scores = np.concatenate([id_scores, ood_scores])
    auroc_ood = float(roc_auc_score(labels, scores))
    aupr_out = float(average_precision_score(labels, scores))

    # Tính điểm và độ đo phát hiện dữ liệu ngoài phân phối.
    fpr_id, tpr_id, _ = roc_curve(1 - labels, -scores, pos_label=1)
    eligible = np.flatnonzero(tpr_id >= 0.95)
    if eligible.size:
        best = eligible[np.argmin(fpr_id[eligible])]
        fpr_at_95 = float(fpr_id[best])
    else:
        fpr_at_95 = 1.0

    return {
        "auroc_ood": auroc_ood,
        "aupr_out": aupr_out,
        "fpr_at_95tpr": fpr_at_95,
    }


def bootstrap_ood_metrics(
    id_scores: np.ndarray,
    ood_scores: np.ndarray,
    n_bootstrap: int = 2_000,
    seed: int = 42,
    alpha: float = 0.05,
    paired: bool = False,
    id_groups: Optional[np.ndarray] = None,
    ood_groups: Optional[np.ndarray] = None,
) -> dict[str, list[float]]:
    """Ước lượng bootstrap cho ood các độ đo cho bước xử lý hiện tại.

    Parameters
    ----------
    id_scores : np.ndarray
        Giá trị ``id_scores`` được sử dụng trong phép xử lý.
    ood_scores : np.ndarray
        Giá trị ``ood_scores`` được sử dụng trong phép xử lý.
    n_bootstrap : int, optional
        Giá trị ``n_bootstrap`` được sử dụng trong phép xử lý.
    seed : int, optional
        Hạt giống phục vụ khả năng tái lập.
    alpha : float, optional
        Giá trị ``alpha`` được sử dụng trong phép xử lý.
    paired : bool, optional
        Giá trị ``paired`` được sử dụng trong phép xử lý.
    id_groups : Optional[np.ndarray]
        Giá trị ``id_groups`` được sử dụng trong phép xử lý.
    ood_groups : Optional[np.ndarray]
        Giá trị ``ood_groups`` được sử dụng trong phép xử lý.

    Returns
    -------
    dict[str, list[float]]
        Kết quả được tạo bởi bước xử lý của hàm.

    Raises
    ------
    ValueError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    """
    id_scores = _finite_scores(id_scores, "id_scores")
    ood_scores = _finite_scores(ood_scores, "ood_scores")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive.")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1.")
    if paired and len(id_scores) != len(ood_scores):
        raise ValueError("Paired bootstrap requires equal ID and OOD sample counts.")

    def validate_groups(groups, expected, name):
        """Kiểm tra tính hợp lệ của groups cho bước xử lý hiện tại.

        Parameters
        ----------
        groups : object
            Giá trị ``groups`` được sử dụng trong phép xử lý.
        expected : object
            Giá trị ``expected`` được sử dụng trong phép xử lý.
        name : object
            Tên hoặc khóa định danh của giá trị.

        Returns
        -------
        object
            Kết quả được tạo bởi bước xử lý của hàm.

        Raises
        ------
        ValueError
            Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
        """
        if groups is None:
            return None
        values = np.asarray(groups).astype(str).reshape(-1)
        if len(values) != expected:
            raise ValueError(f"{name} must contain one group per score.")
        if np.any(values == ""):
            return None
        return values

    id_groups = validate_groups(id_groups, len(id_scores), "id_groups")
    ood_groups = validate_groups(ood_groups, len(ood_scores), "ood_groups")
    if paired and (id_groups is not None or ood_groups is not None):
        raise ValueError("Paired and clustered bootstrap modes cannot be combined.")

    rng = np.random.default_rng(seed)

    def draw_indices(length: int, groups: Optional[np.ndarray]) -> np.ndarray:
        """Vẽ indices cho bước xử lý hiện tại.

        Parameters
        ----------
        length : int
            Số lượng, kích thước hoặc tỷ lệ được sử dụng.
        groups : Optional[np.ndarray]
            Giá trị ``groups`` được sử dụng trong phép xử lý.

        Returns
        -------
        np.ndarray
            Kết quả được tạo bởi bước xử lý của hàm.
        """
        if groups is None:
            return rng.integers(0, length, size=length)
        unique = np.unique(groups)
        sampled_groups = rng.choice(unique, size=len(unique), replace=True)
        return np.concatenate(
            [np.flatnonzero(groups == group) for group in sampled_groups]
        )

    samples: dict[str, list[float]] = {}
    for _ in range(int(n_bootstrap)):
        id_indices = draw_indices(len(id_scores), id_groups)
        ood_indices = (
            id_indices
            if paired
            else draw_indices(len(ood_scores), ood_groups)
        )
        metrics = evaluate_ood(
            id_scores[id_indices],
            ood_scores[ood_indices],
        )
        for key, value in metrics.items():
            if isinstance(value, (float, int)) and np.isfinite(value):
                samples.setdefault(key, []).append(float(value))

    lower = 100.0 * alpha / 2.0
    upper = 100.0 * (1.0 - alpha / 2.0)
    return {
        key: [float(np.percentile(values, lower)), float(np.percentile(values, upper))]
        for key, values in samples.items()
        if values
    }
