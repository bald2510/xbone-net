"""Cung cấp bộ mã hóa nền tảng medclip foundation cho XBone-Net.

Notes
-----
Mô-đun này thuộc cơ sở mã nguồn nghiên cứu XBone-Net và giữ các quy ước dùng chung của dự án.
"""

import sys
import torch
import torch.nn as nn
from torchvision import transforms
import transformers
from transformers import CLIPImageProcessor
import transformers.processing_utils

# Xử lý bộ mã hóa nền tảng theo giao diện tương ứng.
if 'feature_extractor' in transformers.processing_utils.MODALITY_TO_BASE_CLASS_MAPPING:
    transformers.processing_utils.MODALITY_TO_BASE_CLASS_MAPPING['feature_extractor'] = (
        'FeatureExtractionMixin', 'ImageProcessingMixin'
    )

original_clip_init = CLIPImageProcessor.__init__
def wrapped_clip_init(self, *args, **kwargs):
    """Thực hiện bước wrapped clip init trong quy trình hiện tại.

    Parameters
    ----------
    *args : tuple
        Các đối số vị trí bổ sung.
    **kwargs : dict
        Các đối số từ khóa bổ sung.

    Returns
    -------
    object
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    arg_names = [
        "do_resize", "size", "resample", "do_center_crop", 
        "crop_size", "do_normalize", "image_mean", "image_std", 
        "do_convert_rgb"
    ]
    new_kwargs = dict(kwargs)
    for name, val in zip(arg_names, args):
        new_kwargs[name] = val
    return original_clip_init(self, **new_kwargs)

CLIPImageProcessor.__init__ = wrapped_clip_init
sys.modules['transformers'].CLIPFeatureExtractor = CLIPImageProcessor
transformers.CLIPFeatureExtractor = CLIPImageProcessor

try:
    from medclip import MedCLIPModel, MedCLIPVisionModel, MedCLIPProcessor
except ImportError:
    raise ImportError("MedCLIP is not installed. Install via `pip install medclip`.")


# ============================================================
# Xử lý bộ mã hóa nền tảng theo giao diện tương ứng.
# ============================================================

class MedCLIPTokenizerWrapper:
    """Đóng gói hành vi của thành phần ``MedCLIPTokenizerWrapper``.

    Notes
    -----
    Lớp này đóng gói trạng thái và hành vi để các thành phần khác có thể tái sử dụng nhất quán.
    """

    def __init__(self, processor: MedCLIPProcessor):
        """Thực hiện bước init trong quy trình hiện tại.

        Parameters
        ----------
        processor : MedCLIPProcessor
            Giá trị ``processor`` được sử dụng trong phép xử lý.
        """
        self.processor = processor

    def __call__(self, texts, **kwargs):
        """Thực hiện bước call trong quy trình hiện tại.

        Parameters
        ----------
        texts : object
            Giá trị ``texts`` được sử dụng trong phép xử lý.
        **kwargs : dict
            Các đối số từ khóa bổ sung.

        Returns
        -------
        object
            Kết quả được tạo bởi bước xử lý của hàm.
        """
        if isinstance(texts, str):
            texts = [texts]
        encoded = self.processor(
            text=texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=256,
        )
        return encoded["input_ids"]


class MedCLIPTextModule(nn.Module):
    """Đóng gói hành vi của thành phần ``MedCLIPTextModule``.

    Notes
    -----
    Lớp này đóng gói trạng thái và hành vi để các thành phần khác có thể tái sử dụng nhất quán.
    """

    def __init__(self, medclip_model: "MedCLIPModel"):
        """Thực hiện bước init trong quy trình hiện tại.

        Parameters
        ----------
        medclip_model : 'MedCLIPModel'
            Mô hình hoặc thành phần mô hình cần xử lý.
        """
        super().__init__()
        self.transformer = getattr(medclip_model, "bert_model", nn.Identity())

    def forward(self, x):
        """Thực hiện lượt lan truyền xuôi của mô hình.

        Parameters
        ----------
        x : object
            Giá trị ``x`` được sử dụng trong phép xử lý.

        Returns
        -------
        object
            Kết quả được tạo bởi bước xử lý của hàm.
        """
        return x


class MedCLIPModelWrapper:
    """Đóng gói hành vi của thành phần ``MedCLIPModelWrapper``.

    Notes
    -----
    Lớp này đóng gói trạng thái và hành vi để các thành phần khác có thể tái sử dụng nhất quán.
    """

    def __init__(self, medclip_model: "MedCLIPModel"):
        """Thực hiện bước init trong quy trình hiện tại.

        Parameters
        ----------
        medclip_model : 'MedCLIPModel'
            Mô hình hoặc thành phần mô hình cần xử lý.
        """
        self._medclip = medclip_model
        self.visual: nn.Module = medclip_model.vision_model
        self.text = MedCLIPTextModule(medclip_model)

    def set_grad_checkpointing(self, enable: bool = True):
        """Thiết lập grad checkpointing cho bước xử lý hiện tại.

        Parameters
        ----------
        enable : bool, optional
            Giá trị ``enable`` được sử dụng trong phép xử lý.
        """
        if hasattr(self.visual, "set_grad_checkpointing"):
            self.visual.set_grad_checkpointing(enable)
        elif hasattr(self.visual, "gradient_checkpointing_enable"):
            if enable:
                self.visual.gradient_checkpointing_enable()
            else:
                self.visual.gradient_checkpointing_disable()

    def encode_image(self, pixel_values):
        """Mã hóa ảnh cho bước xử lý hiện tại.

        Parameters
        ----------
        pixel_values : object
            Giá trị ``pixel_values`` được sử dụng trong phép xử lý.

        Returns
        -------
        object
            Kết quả được tạo bởi bước xử lý của hàm.
        """
        return self._medclip.encode_image(pixel_values)

    def encode_text(self, input_ids, attention_mask=None):
        """Mã hóa văn bản cho bước xử lý hiện tại.

        Parameters
        ----------
        input_ids : object
            Dữ liệu nguồn của phép xử lý.
        attention_mask : object, optional
            Giá trị ``attention_mask`` được sử dụng trong phép xử lý.

        Returns
        -------
        object
            Kết quả được tạo bởi bước xử lý của hàm.
        """
        if attention_mask is None:
            attention_mask = (input_ids != 0).long()
        return self._medclip.encode_text(input_ids, attention_mask)

    def parameters(self):
        """Thực hiện bước parameters trong quy trình hiện tại.

        Returns
        -------
        object
            Kết quả được tạo bởi bước xử lý của hàm.
        """
        return self._medclip.parameters()


# ============================================================
# Xử lý bộ mã hóa nền tảng theo giao diện tương ứng.
# ============================================================

class MedCLIPFoundation(nn.Module):
    """Bao bọc mô hình nền tảng bằng lớp ``MedCLIPFoundation``.

    Notes
    -----
    Lớp này đóng gói trạng thái và hành vi để các thành phần khác có thể tái sử dụng nhất quán.
    """

    EMBED_DIM = 512

    def __init__(self, freeze_base: bool = True):
        """Thực hiện bước init trong quy trình hiện tại.

        Parameters
        ----------
        freeze_base : bool, optional
            Giá trị ``freeze_base`` được sử dụng trong phép xử lý.
        """
        super().__init__()

        medclip_model = MedCLIPModel(vision_cls=MedCLIPVisionModel)
        
        # Thiết lập giá trị trung gian cho bước xử lý tiếp theo.
        original_load_state_dict = medclip_model.load_state_dict
        def custom_load_state_dict(state_dict, strict=True):
            """Thực hiện bước custom load state dict trong quy trình hiện tại.

            Parameters
            ----------
            state_dict : object
                Giá trị ``state_dict`` được sử dụng trong phép xử lý.
            strict : object, optional
                Giá trị ``strict`` được sử dụng trong phép xử lý.

            Returns
            -------
            object
                Kết quả được tạo bởi bước xử lý của hàm.
            """
            keys_to_remove = ["text_model.model.embeddings.position_ids"]
            for key in keys_to_remove:
                if key in state_dict:
                    del state_dict[key]
            return original_load_state_dict(state_dict, strict=False)
            
        medclip_model.load_state_dict = custom_load_state_dict
        medclip_model.from_pretrained()

        # Thiết lập trạng thái và thống kê các tham số mô hình.
        for param in medclip_model.parameters():
            if param.data is not None:
                param.data = param.data.contiguous()

        self._medclip = medclip_model
        self.model = MedCLIPModelWrapper(medclip_model)

        # Chuẩn hóa chuỗi token và mặt nạ đệm cho batch.
        self.preprocess = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        processor = MedCLIPProcessor()
        self.tokenizer = MedCLIPTokenizerWrapper(processor)

        # Thiết lập trạng thái và thống kê các tham số mô hình.
        if freeze_base:
            for param in self._medclip.parameters():
                param.requires_grad = False

    def forward(self, images, input_ids, attention_mask=None, **kwargs):
        """Thực hiện lượt lan truyền xuôi của mô hình.

        Parameters
        ----------
        images : object
            Giá trị ``images`` được sử dụng trong phép xử lý.
        input_ids : object
            Dữ liệu nguồn của phép xử lý.
        attention_mask : object, optional
            Giá trị ``attention_mask`` được sử dụng trong phép xử lý.
        **kwargs : dict
            Các đối số từ khóa bổ sung.

        Returns
        -------
        object
            Kết quả được tạo bởi bước xử lý của hàm.
        """
        # Thu thập và xử lý biểu diễn đặc trưng của mô hình.
        image_features = self._medclip.encode_image(images)
        if attention_mask is None:
            attention_mask = (input_ids != 0).long()
        text_features = self._medclip.encode_text(input_ids, attention_mask)

        # Thiết lập giá trị trung gian cho bước xử lý tiếp theo.
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return image_features, text_features


