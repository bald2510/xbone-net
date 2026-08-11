"""Cung cấp bộ mã hóa nền tảng resnet50 cho XBone-Net.

Notes
-----
Mô-đun này thuộc cơ sở mã nguồn nghiên cứu XBone-Net và giữ các quy ước dùng chung của dự án.
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms


# ============================================================
# Xử lý bộ mã hóa nền tảng theo giao diện tương ứng.
# ============================================================

class ResNet50Backbone(nn.Module):
    """Đóng gói hành vi của thành phần ``ResNet50Backbone``.

    Notes
    -----
    Lớp này đóng gói trạng thái và hành vi để các thành phần khác có thể tái sử dụng nhất quán.
    """

    EMBED_DIM = 512

    def __init__(
        self,
        pretrained: str = "imagenet",
        embed_dim: int = 512,
        freeze_base: bool = False,
    ):
        """Thực hiện bước init trong quy trình hiện tại.

        Parameters
        ----------
        pretrained : str, optional
            Giá trị ``pretrained`` được sử dụng trong phép xử lý.
        embed_dim : int, optional
            Số lượng, kích thước hoặc tỷ lệ được sử dụng.
        freeze_base : bool, optional
            Giá trị ``freeze_base`` được sử dụng trong phép xử lý.
        """
        super().__init__()
        self.pretrained_type = pretrained

        # Kiểm tra điều kiện trước khi thực hiện nhánh xử lý tương ứng.
        if pretrained == "imagenet" or pretrained == "medical":
            weights = models.ResNet50_Weights.IMAGENET1K_V2
            self.resnet = models.resnet50(weights=weights)
            self.preprocess = weights.transforms()
            if pretrained == "medical":
                print("[ResNet50] Using ImageNet weights as medical baseline.")
        else:
            self.resnet = models.resnet50(weights=None)
            self.preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])

        # Thiết lập mô-đun dung hợp và đầu phân lớp theo cấu hình.
        resnet_feature_dim = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()

        self.projection = nn.Sequential(
            nn.Linear(resnet_feature_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        # Xử lý bộ mã hóa nền tảng theo giao diện tương ứng.
        if freeze_base:
            for param in self.resnet.parameters():
                param.requires_grad = False

        self.tokenizer = DummyTokenizer()

    def forward(self, images, input_ids=None, **kwargs):
        """Thực hiện lượt lan truyền xuôi của mô hình.

        Parameters
        ----------
        images : object
            Giá trị ``images`` được sử dụng trong phép xử lý.
        input_ids : object, optional
            Dữ liệu nguồn của phép xử lý.
        **kwargs : dict
            Các đối số từ khóa bổ sung.

        Returns
        -------
        object
            Kết quả được tạo bởi bước xử lý của hàm.
        """
        feats = self.resnet(images)
        projected = self.projection(feats)
        image_features = projected / projected.norm(dim=-1, keepdim=True)

        return image_features, None


# ============================================================
# Chuẩn hóa chuỗi token và mặt nạ đệm cho batch.
# ============================================================

class DummyTokenizer:
    """Đóng gói hành vi của thành phần ``DummyTokenizer``.

    Notes
    -----
    Lớp này đóng gói trạng thái và hành vi để các thành phần khác có thể tái sử dụng nhất quán.
    """

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
        return torch.zeros(len(texts), 1, dtype=torch.long)


class DummyTextModule(nn.Module):
    """Đóng gói hành vi của thành phần ``DummyTextModule``.

    Notes
    -----
    Lớp này đóng gói trạng thái và hành vi để các thành phần khác có thể tái sử dụng nhất quán.
    """

    def __init__(self):
        """Thực hiện bước init trong quy trình hiện tại."""
        super().__init__()
        self.transformer = nn.Identity()

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


# ============================================================
# Thiết lập thành phần dùng chung cho quy trình xử lý của mô-đun.
# ============================================================

class ResNet50Foundation(nn.Module):
    """Bao bọc mô hình nền tảng bằng lớp ``ResNet50Foundation``.

    Notes
    -----
    Lớp này đóng gói trạng thái và hành vi để các thành phần khác có thể tái sử dụng nhất quán.
    """

    def __init__(
        self,
        pretrained: str = "imagenet",
        embed_dim: int = 512,
        freeze_base: bool = False,
    ):
        """Thực hiện bước init trong quy trình hiện tại.

        Parameters
        ----------
        pretrained : str, optional
            Giá trị ``pretrained`` được sử dụng trong phép xử lý.
        embed_dim : int, optional
            Số lượng, kích thước hoặc tỷ lệ được sử dụng.
        freeze_base : bool, optional
            Giá trị ``freeze_base`` được sử dụng trong phép xử lý.
        """
        super().__init__()
        self._backbone = ResNet50Backbone(
            pretrained=pretrained,
            embed_dim=embed_dim,
            freeze_base=freeze_base,
        )

        self.model = ResNet50ModelWrapper(self._backbone)
        self.preprocess = self._backbone.preprocess
        self.tokenizer = self._backbone.tokenizer

    def forward(self, images, input_ids=None, **kwargs):
        """Thực hiện lượt lan truyền xuôi của mô hình.

        Parameters
        ----------
        images : object
            Giá trị ``images`` được sử dụng trong phép xử lý.
        input_ids : object, optional
            Dữ liệu nguồn của phép xử lý.
        **kwargs : dict
            Các đối số từ khóa bổ sung.

        Returns
        -------
        object
            Kết quả được tạo bởi bước xử lý của hàm.
        """
        return self._backbone(images, input_ids)


# ============================================================
# Thiết lập thành phần dùng chung cho quy trình xử lý của mô-đun.
# ============================================================

class ResNet50ModelWrapper:
    """Đóng gói hành vi của thành phần ``ResNet50ModelWrapper``.

    Notes
    -----
    Lớp này đóng gói trạng thái và hành vi để các thành phần khác có thể tái sử dụng nhất quán.
    """

    def __init__(self, backbone: ResNet50Backbone):
        """Thực hiện bước init trong quy trình hiện tại.

        Parameters
        ----------
        backbone : ResNet50Backbone
            Mô hình hoặc thành phần mô hình cần xử lý.
        """
        self.visual = nn.Sequential(
            backbone.resnet,
            backbone.projection,
        )
        self.text = DummyTextModule()

    def encode_image(self, images):
        """Mã hóa ảnh cho bước xử lý hiện tại.

        Parameters
        ----------
        images : object
            Giá trị ``images`` được sử dụng trong phép xử lý.

        Returns
        -------
        object
            Kết quả được tạo bởi bước xử lý của hàm.
        """
        feats = self.visual(images)
        return feats / feats.norm(dim=-1, keepdim=True)

    def encode_text(self, input_ids):
        """Mã hóa văn bản cho bước xử lý hiện tại.

        Parameters
        ----------
        input_ids : object
            Dữ liệu nguồn của phép xử lý.

        Returns
        -------
        object
            Kết quả được tạo bởi bước xử lý của hàm.
        """
        batch_size = input_ids.shape[0] if isinstance(input_ids, torch.Tensor) else 1
        return torch.zeros(batch_size, 512, device=input_ids.device if isinstance(input_ids, torch.Tensor) else "cpu")


