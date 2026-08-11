"""Cung cấp bộ mã hóa nền tảng openclip foundation cho XBone-Net.

Notes
-----
Mô-đun này thuộc cơ sở mã nguồn nghiên cứu XBone-Net và giữ các quy ước dùng chung của dự án.
"""

import torch.nn as nn
from open_clip import create_model_and_transforms, get_tokenizer
from transformers import CLIPModel


# ============================================================
# Chuẩn hóa chuỗi token và mặt nạ đệm cho batch.
# ============================================================

class PubMedCLIPTokenizerWrapper:
    """Đóng gói hành vi của thành phần ``PubMedCLIPTokenizerWrapper``.

    Notes
    -----
    Lớp này đóng gói trạng thái và hành vi để các thành phần khác có thể tái sử dụng nhất quán.
    """

    def __init__(self, tokenizer_obj):
        """Thực hiện bước init trong quy trình hiện tại.

        Parameters
        ----------
        tokenizer_obj : object
            Giá trị ``tokenizer_obj`` được sử dụng trong phép xử lý.
        """
        self.tokenizer_obj = tokenizer_obj

    def __call__(self, texts):
        """Thực hiện bước call trong quy trình hiện tại.

        Parameters
        ----------
        texts : object
            Giá trị ``texts`` được sử dụng trong phép xử lý.

        Returns
        -------
        object
            Kết quả được tạo bởi bước xử lý của hàm.
        """
        return self.tokenizer_obj(
            texts, padding="max_length", max_length=77, truncation=True, return_tensors="pt"
        )["input_ids"]


class PubMedCLIPModel(CLIPModel):
    """Đóng gói hành vi của thành phần ``PubMedCLIPModel``.

    Notes
    -----
    Lớp này đóng gói trạng thái và hành vi để các thành phần khác có thể tái sử dụng nhất quán.
    """

    @property
    def visual(self):
        """Thực hiện bước visual trong quy trình hiện tại.

        Returns
        -------
        object
            Kết quả được tạo bởi bước xử lý của hàm.
        """
        return self.vision_model
    
    @visual.setter
    def visual(self, value):
        """Thực hiện bước visual trong quy trình hiện tại.

        Parameters
        ----------
        value : object
            Giá trị ``value`` được sử dụng trong phép xử lý.
        """
        self.vision_model = value

    def __setattr__(self, name, value):
        """Thực hiện bước setattr trong quy trình hiện tại.

        Parameters
        ----------
        name : object
            Tên hoặc khóa định danh của giá trị.
        value : object
            Giá trị ``value`` được sử dụng trong phép xử lý.
        """
        if name == 'visual':
            super().__setattr__('vision_model', value)
        else:
            super().__setattr__(name, value)


# ============================================================
# Thiết lập thành phần dùng chung cho quy trình xử lý của mô-đun.
# ============================================================

# Xử lý bộ mã hóa nền tảng theo giao diện tương ứng.
OPENCLIP_MODEL_REGISTRY = {
    "clip": {
        "model_name": "ViT-B-16",
        "pretrained": "openai",
        "description": "CLIP ViT-B/16 (Radford et al., 2021)",
    },
    "pubmedclip": {
        "model_name": "hf-hub:flaviagiammarino/pubmed-clip-vit-base-patch32",
        "pretrained": None,
        "description": "PubMedCLIP ViT-B/32 (Eslami et al., 2023)",
    },
    "biomedclip": {
        "model_name": "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        "pretrained": None,
        "description": "BiomedCLIP ViT-B/16 (Zhang et al., 2023)",
    },
}


# ============================================================
# Xử lý bộ mã hóa nền tảng theo giao diện tương ứng.
# ============================================================

class OpenCLIPFoundation(nn.Module):
    """Bao bọc mô hình nền tảng bằng lớp ``OpenCLIPFoundation``.

    Notes
    -----
    Lớp này đóng gói trạng thái và hành vi để các thành phần khác có thể tái sử dụng nhất quán.
    """

    def __init__(self, model_key: str = "clip", freeze_base: bool = True):
        """Thực hiện bước init trong quy trình hiện tại.

        Parameters
        ----------
        model_key : str, optional
            Mô hình hoặc thành phần mô hình cần xử lý.
        freeze_base : bool, optional
            Giá trị ``freeze_base`` được sử dụng trong phép xử lý.

        Raises
        ------
        ValueError
            Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
        """
        super().__init__()

        if model_key not in OPENCLIP_MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model_key '{model_key}'. "
                f"Choose from: {list(OPENCLIP_MODEL_REGISTRY.keys())}"
            )

        entry = OPENCLIP_MODEL_REGISTRY[model_key]
        model_name = entry["model_name"]
        pretrained = entry["pretrained"]

        # Xử lý bộ mã hóa nền tảng theo giao diện tương ứng.
        if model_key == "pubmedclip":
            from transformers import CLIPTokenizer
            from torchvision import transforms
            
            print("[OpenCLIP] Loading PubMedCLIP via HuggingFace Transformers")
            self.model = PubMedCLIPModel.from_pretrained("flaviagiammarino/pubmed-clip-vit-base-patch32")
            
            # Thiết lập trạng thái và thống kê các tham số mô hình.
            for name, module in self.model.named_modules():
                for param_name, param in list(module.named_parameters(recurse=False)):
                    if not param.is_contiguous():
                        new_param = nn.Parameter(param.contiguous(), requires_grad=param.requires_grad)
                        setattr(module, param_name, new_param)
            
            tokenizer_obj = CLIPTokenizer.from_pretrained("flaviagiammarino/pubmed-clip-vit-base-patch32")
            self.tokenizer = PubMedCLIPTokenizerWrapper(tokenizer_obj)
            
            self.preprocess = transforms.Compose([
                transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711)
                ),
            ])
            
            self.model.encode_image = lambda images: self.model.get_image_features(images).pooler_output
            self.model.encode_text = lambda input_ids: self.model.get_text_features(input_ids).pooler_output
        else:
            print(f"[OpenCLIP] Loading {entry['description']}")

            if pretrained:
                self.model, _, self.preprocess = create_model_and_transforms(
                    model_name, pretrained=pretrained
                )
            else:
                self.model, _, self.preprocess = create_model_and_transforms(model_name)

            self.tokenizer = get_tokenizer(model_name)

        # Xử lý bộ mã hóa nền tảng theo giao diện tương ứng.
        if freeze_base:
            for param in self.model.parameters():
                param.requires_grad = False

    @property
    def tokenizer_obj(self):
        """Thực hiện bước tokenizer obj trong quy trình hiện tại.

        Returns
        -------
        object
            Kết quả được tạo bởi bước xử lý của hàm.
        """
        return self.tokenizer

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
        image_features = self.model.encode_image(images)
        # Xử lý bộ mã hóa nền tảng theo giao diện tương ứng.
        # Thiết lập giá trị trung gian cho bước xử lý tiếp theo.
        text_features = self.model.encode_text(input_ids)

        # Thiết lập giá trị trung gian cho bước xử lý tiếp theo.
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return image_features, text_features


