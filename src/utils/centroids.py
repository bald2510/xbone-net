"""Cung cấp tiện ích centroids cho huấn luyện, đánh giá và phân tích XBone-Net.

Notes
-----
Mô-đun này thuộc cơ sở mã nguồn nghiên cứu XBone-Net và giữ các quy ước dùng chung của dự án.
"""

from __future__ import annotations

from typing import Optional

import torch
from transformers import TrainerCallback


def _select_phase2_text(batch: dict, use_text: bool, report_type: str):
    """Chọn phase2 văn bản cho bước xử lý hiện tại.

    Parameters
    ----------
    batch : dict
        Batch dữ liệu đầu vào.
    use_text : bool
        Văn bản hoặc biểu diễn văn bản đầu vào.
    report_type : str
        Văn bản hoặc biểu diễn văn bản đầu vào.

    Returns
    -------
    object
        Kết quả được tạo bởi bước xử lý của hàm.

    Raises
    ------
    NotImplementedError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    ValueError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    """
    if not use_text:
        return None, None
    if report_type in ("both", "xray_clinical"):
        raise NotImplementedError(
            "Empirical centroid estimation supports one Phase-2 report stream at "
            "a time. Use p2_report_type='xray' or 'clinical'."
        )
    if report_type not in ("xray", "clinical"):
        raise ValueError(
            f"Unknown p2_report_type='{report_type}'. Use xray or clinical."
        )
    prefix = "xray" if report_type == "xray" else "clinical"
    return batch[f"{prefix}_input_ids"], batch.get(f"{prefix}_attention_mask")


@torch.no_grad()
def compute_empirical_centroids(
    model,
    data_loader,
    device: torch.device,
    use_text: bool = True,
    report_type: str = "clinical",
) -> torch.Tensor:
    """Tính empirical các tâm lớp cho bước xử lý hiện tại.

    Parameters
    ----------
    model : object
        Mô hình hoặc thành phần mô hình cần xử lý.
    data_loader : object
        Dữ liệu đầu vào của bước xử lý.
    device : torch.device
        Thiết bị thực thi phép tính.
    use_text : bool, optional
        Văn bản hoặc biểu diễn văn bản đầu vào.
    report_type : str, optional
        Văn bản hoặc biểu diễn văn bản đầu vào.

    Returns
    -------
    torch.Tensor
        Kết quả được tạo bởi bước xử lý của hàm.

    Raises
    ------
    TypeError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    ValueError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    """
    head = getattr(model, "head", None)
    if head is None or not hasattr(head, "set_centroids"):
        raise TypeError(
            "compute_empirical_centroids requires an EmpiricalCentroidHead."
        )
    if not hasattr(model, "encode_fused"):
        raise TypeError("Model must expose encode_fused() for centroid estimation.")

    num_classes = int(head.num_classes)
    feature_dim = int(head.feature_dim)
    sums = torch.zeros(num_classes, feature_dim, dtype=torch.float32, device=device)
    counts = torch.zeros(num_classes, dtype=torch.long, device=device)

    was_training = model.training
    model.eval()
    try:
        for batch in data_loader:
            if not isinstance(batch, dict):
                raise TypeError(
                    "Centroid estimation expects dictionary-format batches from "
                    "BioMedCLIPDataCollator."
                )

            images = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            labels = labels.argmax(dim=-1) if labels.ndim > 1 else labels
            labels = labels.long().view(-1)

            if labels.numel() == 0:
                continue
            if torch.any(labels < 0) or torch.any(labels >= num_classes):
                raise ValueError(
                    f"Training labels must be in [0, {num_classes - 1}]."
                )

            input_ids, attention_mask = _select_phase2_text(
                batch, use_text=use_text, report_type=report_type
            )
            if input_ids is not None:
                input_ids = input_ids.to(device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            features = model.encode_fused(
                images,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            if tuple(features.shape) != (labels.numel(), feature_dim):
                raise ValueError(
                    "Unexpected fused feature shape during centroid estimation: "
                    f"{tuple(features.shape)}; expected {(labels.numel(), feature_dim)}."
                )
            sums.index_add_(0, labels, features.detach().float())
            counts.index_add_(0, labels, torch.ones_like(labels, dtype=torch.long))

        missing = torch.nonzero(counts == 0, as_tuple=False).flatten().tolist()
        if missing:
            raise ValueError(
                "Cannot build empirical centroids because the training subset "
                f"contains no samples for classes: {missing}"
            )
        centroids = sums / counts.unsqueeze(1).to(sums.dtype)
        head.set_centroids(centroids, counts)
    finally:
        model.train(was_training)

    return counts.detach().cpu()


class EmpiricalCentroidUpdateCallback(TrainerCallback):
    """Xử lý sự kiện trong quá trình huấn luyện bằng lớp ``EmpiricalCentroidUpdateCallback``.

    Notes
    -----
    Lớp này đóng gói trạng thái và hành vi để các thành phần khác có thể tái sử dụng nhất quán.
    """

    def __init__(
        self,
        model,
        data_loader,
        device: torch.device,
        use_text: bool = True,
        report_type: str = "clinical",
        interval_epochs: int = 1,
    ) -> None:
        """Thực hiện bước init trong quy trình hiện tại.

        Parameters
        ----------
        model : object
            Mô hình hoặc thành phần mô hình cần xử lý.
        data_loader : object
            Dữ liệu đầu vào của bước xử lý.
        device : torch.device
            Thiết bị thực thi phép tính.
        use_text : bool, optional
            Văn bản hoặc biểu diễn văn bản đầu vào.
        report_type : str, optional
            Văn bản hoặc biểu diễn văn bản đầu vào.
        interval_epochs : int, optional
            Giá trị ``interval_epochs`` được sử dụng trong phép xử lý.

        Raises
        ------
        ValueError
            Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
        """
        if interval_epochs < 1:
            raise ValueError("interval_epochs must be >= 1.")
        self.model = model
        self.data_loader = data_loader
        self.device = device
        self.use_text = bool(use_text)
        self.report_type = str(report_type)
        self.interval_epochs = int(interval_epochs)

    def on_epoch_end(self, args, state, control, **kwargs):
        """Thực hiện bước on epoch end trong quy trình hiện tại.

        Parameters
        ----------
        args : object
            Các đối số vị trí bổ sung.
        state : object
            Giá trị ``state`` được sử dụng trong phép xử lý.
        control : object
            Giá trị ``control`` được sử dụng trong phép xử lý.
        **kwargs : dict
            Các đối số từ khóa bổ sung.

        Returns
        -------
        object
            Kết quả được tạo bởi bước xử lý của hàm.
        """
        epoch = int(round(float(state.epoch or 0.0)))
        if epoch > 0 and epoch % self.interval_epochs == 0:
            counts = compute_empirical_centroids(
                self.model,
                self.data_loader,
                self.device,
                use_text=self.use_text,
                report_type=self.report_type,
            )
            print(
                f"  [Centroids] Refreshed after epoch {epoch}; "
                f"class counts={counts.tolist()}"
            )
        return control
