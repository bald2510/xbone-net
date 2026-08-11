"""Cung cấp bộ mã hóa nền tảng densenet cho XBone-Net.

Notes
-----
Mô-đun này thuộc cơ sở mã nguồn nghiên cứu XBone-Net và giữ các quy ước dùng chung của dự án.
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms
from .resnet50 import DummyTokenizer, DummyTextModule


# ============================================================
# Xử lý bộ mã hóa nền tảng theo giao diện tương ứng.
# ============================================================

class DenseNetBackbone(nn.Module):
    """Đóng gói hành vi của thành phần ``DenseNetBackbone``.

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

        # Xử lý bộ mã hóa nền tảng theo giao diện tương ứng.
        if pretrained == "imagenet" or pretrained == "medical":
            weights = models.DenseNet121_Weights.IMAGENET1K_V1
            self.densenet = models.densenet121(weights=weights)
            self.preprocess = weights.transforms()
            if pretrained == "medical":
                print("[DenseNet121] Using ImageNet weights as medical baseline.")
        else:
            self.densenet = models.densenet121(weights=None)
            self.preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])

        # Thiết lập mô-đun dung hợp và đầu phân lớp theo cấu hình.
        densenet_feature_dim = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Identity()

        self.projection = nn.Sequential(
            nn.Linear(densenet_feature_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        # Xử lý bộ mã hóa nền tảng theo giao diện tương ứng.
        if freeze_base:
            for param in self.densenet.parameters():
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
        feats = self.densenet(images)
        projected = self.projection(feats)
        image_features = projected / projected.norm(dim=-1, keepdim=True)

        return image_features, None


# ============================================================
# Thiết lập thành phần dùng chung cho quy trình xử lý của mô-đun.
# ============================================================

class DenseNetFoundation(nn.Module):
    """Bao bọc mô hình nền tảng bằng lớp ``DenseNetFoundation``.

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
        self._backbone = DenseNetBackbone(
            pretrained=pretrained,
            embed_dim=embed_dim,
            freeze_base=freeze_base,
        )

        self.model = DenseNetModelWrapper(self._backbone)
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

class DenseNetModelWrapper:
    """Đóng gói hành vi của thành phần ``DenseNetModelWrapper``.

    Notes
    -----
    Lớp này đóng gói trạng thái và hành vi để các thành phần khác có thể tái sử dụng nhất quán.
    """

    def __init__(self, backbone: DenseNetBackbone):
        """Thực hiện bước init trong quy trình hiện tại.

        Parameters
        ----------
        backbone : DenseNetBackbone
            Mô hình hoặc thành phần mô hình cần xử lý.
        """
        self.visual = nn.Sequential(
            backbone.densenet,
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


