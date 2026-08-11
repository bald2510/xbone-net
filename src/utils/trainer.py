"""Cung cấp tiện ích trainer cho huấn luyện, đánh giá và phân tích XBone-Net.

Notes
-----
Mô-đun này thuộc cơ sở mã nguồn nghiên cứu XBone-Net và giữ các quy ước dùng chung của dự án.
"""

from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.utils.data import Sampler, Subset
from transformers import Trainer
import torchvision.transforms.functional as F_t


# ============================================================
# Tiện ích đệm chuỗi cho bộ tách từ
# ============================================================

def resolve_pad_token_id(tokenizer) -> int:
    """Xác định pad token id cho bước xử lý hiện tại.

    Parameters
    ----------
    tokenizer : object
        Giá trị ``tokenizer`` được sử dụng trong phép xử lý.

    Returns
    -------
    int
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    # Xử lý bộ mã hóa nền tảng theo giao diện tương ứng.
    hf_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    pad_token_id = getattr(hf_tokenizer, "pad_token_id", None)
    if pad_token_id is not None:
        return pad_token_id
    # Chuẩn hóa chuỗi token và mặt nạ đệm cho batch.
    return getattr(hf_tokenizer, "eos_token_id", 0) or 0


def _sampler_class_ids(dataset) -> torch.Tensor:
    """Thực hiện bước sampler class ids trong quy trình hiện tại.

    Parameters
    ----------
    dataset : object
        Dữ liệu đầu vào của bước xử lý.

    Returns
    -------
    torch.Tensor
        Kết quả được tạo bởi bước xử lý của hàm.

    Raises
    ------
    AttributeError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    """
    if isinstance(dataset, Subset):
        parent = _sampler_class_ids(dataset.dataset)
        indices = torch.as_tensor(dataset.indices, dtype=torch.long)
        return parent[indices]
    if hasattr(dataset, "df") and "class_id" in dataset.df.columns:
        return torch.as_tensor(
            dataset.df["class_id"].to_numpy(), dtype=torch.long
        )
    for attribute in ("labels", "targets"):
        if hasattr(dataset, attribute):
            labels = torch.as_tensor(getattr(dataset, attribute))
            if labels.ndim > 1:
                labels = labels.argmax(dim=-1)
            return labels.long().reshape(-1)
    raise AttributeError(
        "Class-aware sampling requires df['class_id'], labels, targets, or a Subset."
    )


class ClassAwareSampler(Sampler[int]):
    """Lấy mẫu dữ liệu bằng lớp ``ClassAwareSampler``.

    Notes
    -----
    Lớp này đóng gói trạng thái và hành vi để các thành phần khác có thể tái sử dụng nhất quán.
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        samples_per_class: int = 2,
        seed: int = 42,
    ) -> None:
        """Thực hiện bước init trong quy trình hiện tại.

        Parameters
        ----------
        dataset : object
            Dữ liệu đầu vào của bước xử lý.
        batch_size : int
            Số lượng, kích thước hoặc tỷ lệ được sử dụng.
        samples_per_class : int, optional
            Nhãn hoặc chỉ số lớp liên quan.
        seed : int, optional
            Hạt giống phục vụ khả năng tái lập.

        Raises
        ------
        ValueError
            Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
        """
        if batch_size < 2:
            raise ValueError("Class-aware sampling requires batch_size >= 2.")
        if samples_per_class < 2 or samples_per_class > batch_size:
            raise ValueError(
                "samples_per_class must be between 2 and batch_size."
            )
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.samples_per_class = int(samples_per_class)
        self.seed = int(seed)
        self.epoch = 0
        labels = _sampler_class_ids(dataset)
        if labels.numel() != len(dataset):
            raise ValueError("Extracted labels do not match the dataset length.")
        self.class_to_indices = {
            int(class_id): torch.nonzero(
                labels == class_id, as_tuple=False
            ).flatten()
            for class_id in torch.unique(labels, sorted=True).tolist()
        }
        if len(self.class_to_indices) < 2:
            raise ValueError("Class-aware sampling requires at least two classes.")
        self.classes = torch.tensor(
            sorted(self.class_to_indices), dtype=torch.long
        )

    def __len__(self) -> int:
        """Thực hiện bước len trong quy trình hiện tại.

        Returns
        -------
        int
            Kết quả được tạo bởi bước xử lý của hàm.
        """
        return len(self.dataset)

    def set_epoch(self, epoch: int) -> None:
        """Thiết lập epoch cho bước xử lý hiện tại.

        Parameters
        ----------
        epoch : int
            Giá trị ``epoch`` được sử dụng trong phép xử lý.
        """
        self.epoch = int(epoch)

    def __iter__(self):
        """Thực hiện bước iter trong quy trình hiện tại.

        Returns
        -------
        iterator
            Kết quả được tạo bởi bước xử lý của hàm.
        """
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        sequence: list[int] = []
        remaining_total = len(self.dataset)

        while remaining_total > 0:
            current_batch = min(self.batch_size, remaining_total)
            pair_groups = current_batch // self.samples_per_class
            selected: list[int] = []

            if pair_groups:
                replace_classes = pair_groups > len(self.classes)
                if replace_classes:
                    class_positions = torch.randint(
                        len(self.classes),
                        (pair_groups,),
                        generator=generator,
                    )
                else:
                    class_positions = torch.randperm(
                        len(self.classes), generator=generator
                    )[:pair_groups]

                for position in class_positions.tolist():
                    class_id = int(self.classes[position])
                    candidates = self.class_to_indices[class_id]
                    if len(candidates) >= self.samples_per_class:
                        chosen = candidates[
                            torch.randperm(len(candidates), generator=generator)[
                                : self.samples_per_class
                            ]
                        ]
                    else:
                        chosen = candidates[
                            torch.randint(
                                len(candidates),
                                (self.samples_per_class,),
                                generator=generator,
                            )
                        ]
                    selected.extend(int(index) for index in chosen.tolist())

            while len(selected) < current_batch:
                selected.append(
                    int(torch.randint(len(self.dataset), (1,), generator=generator))
                )
            order = torch.randperm(len(selected), generator=generator).tolist()
            sequence.extend(selected[index] for index in order)
            remaining_total -= current_batch

        return iter(sequence)


# ============================================================
# Bộ gộp dữ liệu cho hai loại báo cáo
# ============================================================

@dataclass
class BioMedCLIPDataCollator:
    """Đóng gói hành vi của thành phần ``BioMedCLIPDataCollator``.

    Notes
    -----
    Lớp này đóng gói trạng thái và hành vi để các thành phần khác có thể tái sử dụng nhất quán.
    """

    pad_token_id: int = 0

    def _pad_ids_with_mask(self, ids_list):
        """Thực hiện bước pad ids with mask trong quy trình hiện tại.

        Parameters
        ----------
        ids_list : object
            Giá trị ``ids_list`` được sử dụng trong phép xử lý.

        Returns
        -------
        object
            Kết quả được tạo bởi bước xử lý của hàm.

        Raises
        ------
        ValueError
            Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
        """
        import torch

        if not ids_list:
            raise ValueError("ids_list must not be empty.")

        if any(ids is None for ids in ids_list):
            raise ValueError(
                "ids_list contains None. Check dataset text tokenization."
            )

        max_len = max(ids.shape[0] for ids in ids_list)
        padded_list = []

        for ids in ids_list:
            if ids.ndim != 1:
                ids = ids.view(-1)

            pad_len = max_len - ids.shape[0]

            if pad_len > 0:
                padding = torch.full(
                    (pad_len,),
                    fill_value=self.pad_token_id,
                    dtype=ids.dtype,
                    device=ids.device,
                )
                ids = torch.cat([ids, padding], dim=0)

            padded_list.append(ids)

        padded_ids = torch.stack(padded_list, dim=0)

        # Chuẩn hóa chuỗi token và mặt nạ đệm cho batch.
        attention_mask = (
            padded_ids != self.pad_token_id
        ).to(dtype=torch.long)

        return padded_ids, attention_mask


    def _pad_optional_ids_with_mask(self, ids_list):
        """Thực hiện bước pad optional ids with mask trong quy trình hiện tại.

        Parameters
        ----------
        ids_list : object
            Giá trị ``ids_list`` được sử dụng trong phép xử lý.

        Returns
        -------
        object
            Kết quả được tạo bởi bước xử lý của hàm.

        Raises
        ------
        ValueError
            Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
        """
        if all(ids is None for ids in ids_list):
            batch_size = len(ids_list)
            ids = torch.full(
                (batch_size, 1), self.pad_token_id, dtype=torch.long
            )
            mask = torch.zeros((batch_size, 1), dtype=torch.long)
            return ids, mask
        if any(ids is None for ids in ids_list):
            raise ValueError(
                "A text field is missing for only part of a batch. Ensure dataset "
                "samples use a consistent schema."
            )
        return self._pad_ids_with_mask(ids_list)

    def _pad_tiles_with_mask(self, tiles_list):
        """Thực hiện bước pad tiles with mask trong quy trình hiện tại.

        Parameters
        ----------
        tiles_list : object
            Giá trị ``tiles_list`` được sử dụng trong phép xử lý.

        Returns
        -------
        object
            Kết quả được tạo bởi bước xử lý của hàm.
        """
        import torch
        max_tiles = max(tiles.shape[0] for tiles in tiles_list)
        
        padded_tiles = []
        tile_masks = []
        
        for tiles in tiles_list:
            n = tiles.shape[0]
            pad_n = max_tiles - n
            
            mask = torch.cat([
                torch.ones(n, dtype=torch.long),
                torch.zeros(pad_n, dtype=torch.long),
            ])
            
            if pad_n > 0:
                pad = torch.zeros((pad_n, *tiles.shape[1:]), dtype=tiles.dtype, device=tiles.device)
                tiles = torch.cat([tiles, pad], dim=0)
                
            padded_tiles.append(tiles)
            tile_masks.append(mask)
            
        return torch.stack(padded_tiles), torch.stack(tile_masks)

    def _pad_tile_metadata(self, features: list, max_tiles: int):
        """Thực hiện bước pad tile metadata trong quy trình hiện tại.

        Parameters
        ----------
        features : list
            Giá trị ``features`` được sử dụng trong phép xử lý.
        max_tiles : int
            Giá trị ``max_tiles`` được sử dụng trong phép xử lý.

        Returns
        -------
        object
            Kết quả được tạo bởi bước xử lý của hàm.

        Raises
        ------
        ValueError
            Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
        """
        padded_boxes = []
        for feature in features:
            tile_count = feature["tile_values"].shape[0]
            boxes = feature.get("tile_boxes")
            if boxes is None:
                boxes = torch.zeros((tile_count, 4), dtype=torch.float32)
            if boxes.shape != (tile_count, 4):
                raise ValueError(
                    f"tile_boxes must be {(tile_count, 4)}, got {tuple(boxes.shape)}"
                )
            pad_count = max_tiles - tile_count
            if pad_count:
                boxes = torch.cat(
                    [boxes, torch.zeros((pad_count, 4), dtype=boxes.dtype)], dim=0
                )
            padded_boxes.append(boxes)
        return torch.stack(padded_boxes)

    def __call__(self, features: list) -> dict:
        """Thực hiện bước call trong quy trình hiện tại.

        Parameters
        ----------
        features : list
            Giá trị ``features`` được sử dụng trong phép xử lý.

        Returns
        -------
        dict
            Kết quả được tạo bởi bước xử lý của hàm.
        """
        if not features:
            return {}

        first = features[0]
        if isinstance(first, tuple):
            if len(first) == 4:
                images = torch.stack([f[0] if isinstance(f[0], torch.Tensor) else F_t.to_tensor(f[0]) for f in features])
                xray_ids, xray_mask = self._pad_ids_with_mask([f[1] for f in features])
                clinical_ids, clinical_mask = self._pad_ids_with_mask([f[2] for f in features])
                labels = torch.stack([f[3] if isinstance(f[3], torch.Tensor) else torch.tensor(f[3]) for f in features])
                return {
                    "pixel_values": images,
                    "xray_input_ids": xray_ids,
                    "xray_attention_mask": xray_mask,
                    "clinical_input_ids": clinical_ids,
                    "clinical_attention_mask": clinical_mask,
                    "labels": labels,
                }
            elif len(first) == 3:
                images = torch.stack([f[0] for f in features])
                input_ids, input_mask = self._pad_ids_with_mask([f[1] for f in features])
                labels = torch.stack([f[2] if isinstance(f[2], torch.Tensor) else torch.tensor(f[2]) for f in features])
                return {
                    "pixel_values": images,
                    "xray_input_ids": input_ids,
                    "xray_attention_mask": input_mask,
                    "clinical_input_ids": input_ids,
                    "clinical_attention_mask": input_mask,
                    "labels": labels,
                }

        # Xử lý dự phòng cho dữ liệu dạng từ điển
        pixel_values = torch.stack([f["pixel_values"] for f in features])
        labels = torch.stack([f["labels"] if isinstance(f["labels"], torch.Tensor) else torch.tensor(f["labels"]) for f in features])
        
        xray_ids_list = [f.get("xray_input_ids", f.get("input_ids")) for f in features]
        clinical_ids_list = [f.get("clinical_input_ids", f.get("input_ids")) for f in features]
        
        xray_ids, xray_mask = self._pad_optional_ids_with_mask(xray_ids_list)
        clinical_ids, clinical_mask = self._pad_optional_ids_with_mask(clinical_ids_list)

        
        batch = {
            "pixel_values": pixel_values,
            "xray_input_ids": xray_ids,
            "xray_attention_mask": xray_mask,
            "clinical_input_ids": clinical_ids,
            "clinical_attention_mask": clinical_mask,
            "labels": labels,
        }
        
        # Hỗ trợ chia vùng ảnh độ phân giải cao
        if "tile_values" in features[0]:
            tile_values_list = [f["tile_values"] for f in features]
            padded_tiles, tile_masks = self._pad_tiles_with_mask(tile_values_list)
            batch["tile_values"] = padded_tiles
            batch["tile_mask"] = tile_masks
            tile_boxes = self._pad_tile_metadata(
                features, padded_tiles.shape[1]
            )
            batch["tile_boxes"] = tile_boxes
            
        return batch


# ============================================================
# Bộ điều hợp Trainer theo từng pha
# ============================================================

class SFTrainer(Trainer):
    """Điều phối quá trình huấn luyện bằng lớp ``SFTrainer``.

    Notes
    -----
    Lớp này đóng gói trạng thái và hành vi để các thành phần khác có thể tái sử dụng nhất quán.
    """

    def __init__(self, phase, loss_fn, use_text_in_p2=True,
                 p1_report_type="xray", p2_report_type="clinical",
                 class_aware_sampling=None,
                 *args, **kwargs):
        """Thực hiện bước init trong quy trình hiện tại.

        Parameters
        ----------
        phase : object
            Giá trị ``phase`` được sử dụng trong phép xử lý.
        loss_fn : object
            Giá trị ``loss_fn`` được sử dụng trong phép xử lý.
        use_text_in_p2 : object, optional
            Văn bản hoặc biểu diễn văn bản đầu vào.
        p1_report_type : object, optional
            Văn bản hoặc biểu diễn văn bản đầu vào.
        p2_report_type : object, optional
            Văn bản hoặc biểu diễn văn bản đầu vào.
        class_aware_sampling : object, optional
            Nhãn hoặc chỉ số lớp liên quan.
        *args : tuple
            Các đối số vị trí bổ sung.
        **kwargs : dict
            Các đối số từ khóa bổ sung.
        """
        self.phase = phase
        self.class_aware_sampling = dict(class_aware_sampling or {})
        super().__init__(*args, **kwargs)
        self.loss_fn = loss_fn
        self.use_text_in_p2 = use_text_in_p2
        self.p1_report_type = p1_report_type
        self.p2_report_type = p2_report_type

    def _get_train_sampler(self, train_dataset=None):
        """Lấy train sampler cho bước xử lý hiện tại.

        Parameters
        ----------
        train_dataset : object, optional
            Dữ liệu đầu vào của bước xử lý.

        Returns
        -------
        object
            Kết quả được tạo bởi bước xử lý của hàm.
        """
        dataset = train_dataset if train_dataset is not None else self.train_dataset
        enabled = bool(self.class_aware_sampling.get("enabled", False))
        if self.phase == "phase1" and enabled and dataset is not None:
            return ClassAwareSampler(
                dataset=dataset,
                batch_size=int(self.args.train_batch_size),
                samples_per_class=int(
                    self.class_aware_sampling.get("samples_per_class", 2)
                ),
                seed=int(self.class_aware_sampling.get("seed", self.args.seed)),
            )
        return super()._get_train_sampler(train_dataset)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Tính loss cho bước xử lý hiện tại.

        Parameters
        ----------
        model : object
            Mô hình hoặc thành phần mô hình cần xử lý.
        inputs : object
            Giá trị ``inputs`` được sử dụng trong phép xử lý.
        return_outputs : object, optional
            Giá trị ``return_outputs`` được sử dụng trong phép xử lý.
        **kwargs : dict
            Các đối số từ khóa bổ sung.

        Returns
        -------
        object
            Kết quả được tạo bởi bước xử lý của hàm.

        Raises
        ------
        NotImplementedError
            Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
        """
        images = inputs["pixel_values"]
        labels = inputs["labels"]
        labels_for_loss = labels.argmax(dim=-1) if labels.ndim > 1 else labels
        
        tile_values = inputs.get("tile_values")
        tile_mask = inputs.get("tile_mask")
        tile_boxes = inputs.get("tile_boxes")

        if self.phase == "phase1":
            # Thiết lập và thực thi pha 1 căn chỉnh ảnh-văn bản.
            if self.p1_report_type in ("both", "xray_clinical"):
                raise NotImplementedError(
                    "Simultaneous token-level X-ray and clinical report alignment is "
                    "not implemented. Configure p1_report_type='xray' or 'clinical'."
                )

            is_xray = self.p1_report_type == "xray"
            text_ids = inputs["xray_input_ids"] if is_xray else inputs["clinical_input_ids"]
            text_mask = inputs.get(
                "xray_attention_mask" if is_xray else "clinical_attention_mask"
            )
            image_features, text_features = model.backbone(
                images,
                text_ids,
                attention_mask=text_mask,
                tile_values=tile_values,
                tile_mask=tile_mask,
                tile_boxes=tile_boxes,
            )

            # Chuẩn bị và xử lý đầu vào hoặc đặc trưng hình ảnh.
            # Xử lý bộ mã hóa nền tảng theo giao diện tương ứng.
            # Xử lý bộ mã hóa nền tảng theo giao diện tương ứng.
            if image_features.dim() == 3:
                contrastive_pooler = getattr(
                    model.backbone,
                    "pool_contrastive_image_features",
                    None,
                )
                if callable(contrastive_pooler):
                    image_features = contrastive_pooler(image_features)
                else:
                    image_padding_mask = getattr(
                        model.backbone, "last_image_key_padding_mask", None
                    )
                    if image_padding_mask is None:
                        image_features = image_features.mean(dim=1)
                    else:
                        valid = (~image_padding_mask).unsqueeze(-1).to(
                            image_features.dtype
                        )
                        image_features = (image_features * valid).sum(
                            dim=1
                        ) / valid.sum(dim=1).clamp_min(1.0)
                
            loss = self.loss_fn(image_features, text_features, labels_for_loss)
            outputs = {"image_features": image_features, "text_features": text_features}
        else:
            # Thiết lập và thực thi pha 2 phân lớp đa phương thức.
            if self.use_text_in_p2 and self.p2_report_type in ("both", "xray_clinical"):
                raise NotImplementedError(
                    "Simultaneous token-level X-ray and clinical report fusion is "
                    "not implemented. Configure p2_report_type='xray' or 'clinical'."
                )
            elif self.use_text_in_p2:
                is_xray = self.p2_report_type == "xray"
                text_ids = inputs["xray_input_ids"] if is_xray else inputs["clinical_input_ids"]
                text_mask = inputs.get("xray_attention_mask" if is_xray else "clinical_attention_mask")
            else:
                # Xử lý bộ mã hóa nền tảng theo giao diện tương ứng.
                text_ids = None
                text_mask = None

            if self.phase == "phase3":
                outputs = model.forward_drl(
                    images,
                    text_ids,
                    attention_mask=text_mask,
                    tile_values=tile_values,
                    tile_mask=tile_mask,
                    tile_boxes=tile_boxes,
                )
                logits = outputs["auxiliary_logits"]
            else:
                outputs = model(
                    images,
                    text_ids,
                    attention_mask=text_mask,
                    tile_values=tile_values,
                    tile_mask=tile_mask,
                    tile_boxes=tile_boxes,
                )
                logits = outputs[0] if isinstance(outputs, tuple) else outputs
            loss = self.loss_fn(logits, labels_for_loss)

        return (loss, outputs) if return_outputs else loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: dict,
        prediction_loss_only: bool,
        ignore_keys: list | None = None,
    ):
        """Thực hiện bước prediction step trong quy trình hiện tại.

        Parameters
        ----------
        model : nn.Module
            Mô hình hoặc thành phần mô hình cần xử lý.
        inputs : dict
            Giá trị ``inputs`` được sử dụng trong phép xử lý.
        prediction_loss_only : bool
            Giá trị ``prediction_loss_only`` được sử dụng trong phép xử lý.
        ignore_keys : list | None, optional
            Giá trị ``ignore_keys`` được sử dụng trong phép xử lý.

        Returns
        -------
        object
            Kết quả được tạo bởi bước xử lý của hàm.
        """
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)

        logits = None
        if not prediction_loss_only and self.phase in {"phase2", "phase3"}:
            if self.phase == "phase3":
                logits = outputs["auxiliary_logits"]
            else:
                logits = outputs[0] if isinstance(outputs, tuple) else outputs

        labels = inputs.get("labels")
        return (loss, logits, labels)
