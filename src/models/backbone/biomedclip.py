"""Cung cấp bộ mã hóa nền tảng biomedclip cho XBone-Net.

Notes
-----
Mô-đun này thuộc cơ sở mã nguồn nghiên cứu XBone-Net và giữ các quy ước dùng chung của dự án.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from open_clip import create_model_and_transforms, get_tokenizer


class _ResamplerLayer(nn.Module):
    """Đóng gói hành vi của thành phần ``_ResamplerLayer``.

    Notes
    -----
    Lớp này đóng gói trạng thái và hành vi để các thành phần khác có thể tái sử dụng nhất quán.
    """
    def __init__(self, dim: int, num_heads: int, dropout: float):
        """Thực hiện bước init trong quy trình hiện tại.

        Parameters
        ----------
        dim : int
            Số lượng, kích thước hoặc tỷ lệ được sử dụng.
        num_heads : int
            Số lượng, kích thước hoặc tỷ lệ được sử dụng.
        dropout : float
            Giá trị ``dropout`` được sử dụng trong phép xử lý.
        """
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(
            dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(
        self,
        queries: torch.Tensor,
        local_tokens: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Thực hiện lượt lan truyền xuôi của mô hình.

        Parameters
        ----------
        queries : torch.Tensor
            Giá trị ``queries`` được sử dụng trong phép xử lý.
        local_tokens : torch.Tensor
            Chuỗi token hoặc biểu diễn token đầu vào.
        key_padding_mask : torch.Tensor
            Tên hoặc khóa định danh của giá trị.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Kết quả được tạo bởi bước xử lý của hàm.
        """
        attended, weights = self.cross_attention(
            queries,
            local_tokens,
            local_tokens,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        queries = self.norm1(queries + attended)
        queries = self.norm2(queries + self.ffn(queries))
        return queries, weights


class GlobalGuidedAttentionPool(nn.Module):
    """Đóng gói hành vi của thành phần ``GlobalGuidedAttentionPool``.

    Notes
    -----
    Lớp này đóng gói trạng thái và hành vi để các thành phần khác có thể tái sử dụng nhất quán.
    """

    def __init__(self, dim: int = 512, hidden_dim: int = 128) -> None:
        """Thực hiện bước init trong quy trình hiện tại.

        Parameters
        ----------
        dim : int, optional
            Số lượng, kích thước hoặc tỷ lệ được sử dụng.
        hidden_dim : int, optional
            Số lượng, kích thước hoặc tỷ lệ được sử dụng.

        Raises
        ------
        ValueError
            Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
        """
        super().__init__()
        if dim < 1 or hidden_dim < 1:
            raise ValueError("dim and hidden_dim must be positive.")
        self.local_projection = nn.Linear(dim, hidden_dim, bias=False)
        self.global_projection = nn.Linear(dim, hidden_dim, bias=False)
        self.score = nn.Linear(hidden_dim, 1, bias=False)
        nn.init.zeros_(self.score.weight)
        self.last_attention: Optional[torch.Tensor] = None

    def forward(
        self,
        global_feature: torch.Tensor,
        local_features: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Thực hiện lượt lan truyền xuôi của mô hình.

        Parameters
        ----------
        global_feature : torch.Tensor
            Biểu diễn đặc trưng cần xử lý.
        local_features : torch.Tensor
            Giá trị ``local_features`` được sử dụng trong phép xử lý.
        valid_mask : torch.Tensor
            Giá trị ``valid_mask`` được sử dụng trong phép xử lý.

        Returns
        -------
        torch.Tensor
            Kết quả được tạo bởi bước xử lý của hàm.

        Raises
        ------
        ValueError
            Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
        """
        if global_feature.ndim != 2 or local_features.ndim != 3:
            raise ValueError(
                "Attention pooling expects global [B,D] and local [B,N,D] features."
            )
        if global_feature.shape != (
            local_features.size(0),
            local_features.size(2),
        ):
            raise ValueError("Global and local feature dimensions are incompatible.")
        if valid_mask.shape != local_features.shape[:2]:
            raise ValueError("valid_mask must have shape [B,N].")
        valid_mask = valid_mask.to(device=local_features.device, dtype=torch.bool)
        if torch.any(valid_mask.sum(dim=1) == 0):
            raise ValueError("Each sample requires at least one valid local token.")

        local_hidden = self.local_projection(local_features)
        global_hidden = self.global_projection(global_feature).unsqueeze(1)
        scores = self.score(torch.tanh(local_hidden + global_hidden)).squeeze(-1)
        scores = scores.masked_fill(~valid_mask, torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=1)
        self.last_attention = attention.detach()
        return torch.sum(attention.unsqueeze(-1) * local_features, dim=1)


class SpatialTokenResampler(nn.Module):
    """Đóng gói hành vi của thành phần ``SpatialTokenResampler``.

    Notes
    -----
    Lớp này đóng gói trạng thái và hành vi để các thành phần khác có thể tái sử dụng nhất quán.
    """

    def __init__(
        self,
        dim: int = 512,
        num_tokens: int = 24,
        num_heads: int = 8,
        depth: int = 2,
        dropout: float = 0.1,
        use_spatial_coordinates: bool = True,
        aggregation: str = "learned_queries",
        spatial_coordinate_scale_init: float = 0.1,
    ):
        """Thực hiện bước init trong quy trình hiện tại.

        Parameters
        ----------
        dim : int, optional
            Số lượng, kích thước hoặc tỷ lệ được sử dụng.
        num_tokens : int, optional
            Số lượng, kích thước hoặc tỷ lệ được sử dụng.
        num_heads : int, optional
            Số lượng, kích thước hoặc tỷ lệ được sử dụng.
        depth : int, optional
            Giá trị ``depth`` được sử dụng trong phép xử lý.
        dropout : float, optional
            Giá trị ``dropout`` được sử dụng trong phép xử lý.
        use_spatial_coordinates : bool, optional
            Giá trị ``use_spatial_coordinates`` được sử dụng trong phép xử lý.
        aggregation : str, optional
            Giá trị ``aggregation`` được sử dụng trong phép xử lý.
        spatial_coordinate_scale_init : float, optional
            Giá trị ``spatial_coordinate_scale_init`` được sử dụng trong phép xử lý.

        Raises
        ------
        ValueError
            Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
        """
        super().__init__()
        if num_tokens < 1 or depth < 1:
            raise ValueError("num_tokens and depth must be positive.")
        if aggregation not in {"learned_queries", "mean_pool", "passthrough"}:
            raise ValueError(
                "aggregation must be 'learned_queries', 'mean_pool', or "
                "'passthrough'."
            )
        if spatial_coordinate_scale_init < 0:
            raise ValueError("spatial_coordinate_scale_init must be non-negative.")
        self.num_tokens = int(num_tokens)
        self.use_spatial_coordinates = bool(use_spatial_coordinates)
        self.aggregation = aggregation
        if aggregation == "learned_queries":
            self.queries = nn.Parameter(torch.empty(1, self.num_tokens, dim))
            nn.init.trunc_normal_(self.queries, std=0.02)
        else:
            self.register_parameter("queries", None)
        spatial_dim = max(dim // 4, 32)
        self.spatial_projection = nn.Sequential(
            nn.Linear(4, spatial_dim),
            nn.GELU(),
            nn.Linear(spatial_dim, dim),
        )
        # Thu thập và xử lý biểu diễn đặc trưng của mô hình.
        # Xử lý bộ mã hóa nền tảng theo giao diện tương ứng.
        # Bước hỗ trợ để khởi tạo trạng thái.
        # Thiết lập giá trị trung gian cho bước xử lý tiếp theo.
        self.spatial_gate = nn.Parameter(
            torch.tensor(float(spatial_coordinate_scale_init))
        )
        self.layers = nn.ModuleList(
            [_ResamplerLayer(dim, num_heads, dropout) for _ in range(depth)]
            if aggregation == "learned_queries"
            else []
        )
        self.last_attention: Optional[torch.Tensor] = None
        self.last_local_boxes: Optional[torch.Tensor] = None
        self.capture_explanations = False
        self.last_attention_layers_raw: list[torch.Tensor] = []
        self.last_output_mask: Optional[torch.Tensor] = None

    def forward(
        self,
        global_feature: torch.Tensor,
        local_tokens: torch.Tensor,
        local_mask: torch.Tensor,
        local_boxes: torch.Tensor,
    ) -> torch.Tensor:
        """Thực hiện lượt lan truyền xuôi của mô hình.

        Parameters
        ----------
        global_feature : torch.Tensor
            Biểu diễn đặc trưng cần xử lý.
        local_tokens : torch.Tensor
            Chuỗi token hoặc biểu diễn token đầu vào.
        local_mask : torch.Tensor
            Giá trị ``local_mask`` được sử dụng trong phép xử lý.
        local_boxes : torch.Tensor
            Giá trị ``local_boxes`` được sử dụng trong phép xử lý.

        Returns
        -------
        torch.Tensor
            Kết quả được tạo bởi bước xử lý của hàm.

        Raises
        ------
        ValueError
            Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
        """
        if local_tokens.ndim != 3:
            raise ValueError("local_tokens must have shape [B,N,D].")
        if local_mask.shape != local_tokens.shape[:2]:
            raise ValueError("local_mask must have shape [B,N].")
        if local_boxes.shape != (*local_tokens.shape[:2], 4):
            raise ValueError("local_boxes must have shape [B,N,4].")
        if torch.any(local_mask.sum(dim=1) == 0):
            raise ValueError("Each image must contain at least one local token.")

        local_mask = local_mask.to(device=local_tokens.device, dtype=torch.bool)
        local_boxes = local_boxes.to(device=local_tokens.device, dtype=local_tokens.dtype)
        positioned_tokens = local_tokens
        if self.use_spatial_coordinates:
            spatial_tokens = F.normalize(
                self.spatial_projection(local_boxes), dim=-1, eps=1e-6
            )
            positioned_tokens = F.normalize(
                F.normalize(positioned_tokens, dim=-1, eps=1e-6)
                + torch.tanh(self.spatial_gate) * spatial_tokens,
                dim=-1,
                eps=1e-6,
            )
        positioned_tokens = positioned_tokens * local_mask.unsqueeze(-1).to(
            positioned_tokens.dtype
        )

        if self.aggregation == "passthrough":
            if local_tokens.size(1) > self.num_tokens:
                raise ValueError(
                    "passthrough received more local tokens than its configured "
                    f"budget: {local_tokens.size(1)} > {self.num_tokens}."
                )
            identity = torch.diag_embed(local_mask.to(positioned_tokens.dtype))
            self.last_attention = identity.detach()
            self.last_local_boxes = local_boxes.detach()
            self.last_attention_layers_raw = []
            self.last_output_mask = local_mask.detach()
            return torch.cat(
                [global_feature.unsqueeze(1), positioned_tokens], dim=1
            )

        if self.aggregation == "mean_pool":
            valid = local_mask.unsqueeze(-1).to(positioned_tokens.dtype)
            pooled = (positioned_tokens * valid).sum(dim=1) / valid.sum(
                dim=1
            ).clamp_min(1.0)
            normalized_weights = local_mask.to(positioned_tokens.dtype)
            normalized_weights = normalized_weights / normalized_weights.sum(
                dim=1, keepdim=True
            ).clamp_min(1.0)
            self.last_attention = normalized_weights.unsqueeze(1).detach()
            self.last_local_boxes = local_boxes.detach()
            self.last_attention_layers_raw = []
            self.last_output_mask = torch.ones(
                local_tokens.size(0),
                1,
                dtype=torch.bool,
                device=local_tokens.device,
            )
            return torch.cat(
                [global_feature.unsqueeze(1), pooled.unsqueeze(1)], dim=1
            )

        queries = self.queries.expand(local_tokens.size(0), -1, -1)
        queries = queries + global_feature.unsqueeze(1)

        weights = None
        raw_weights = []
        for layer in self.layers:
            queries, weights = layer(queries, positioned_tokens, ~local_mask)
            if self.capture_explanations:
                raw_weights.append(weights)

        self.last_attention = weights.mean(dim=1).detach()
        self.last_local_boxes = local_boxes.detach()
        self.last_attention_layers_raw = raw_weights
        self.last_output_mask = torch.ones(
            local_tokens.size(0),
            self.num_tokens,
            dtype=torch.bool,
            device=local_tokens.device,
        )
        return torch.cat([global_feature.unsqueeze(1), queries], dim=1)


class BiomedCLIPFoundation(nn.Module):
    """Bao bọc mô hình nền tảng bằng lớp ``BiomedCLIPFoundation``.

    Notes
    -----
    Lớp này đóng gói trạng thái và hành vi để các thành phần khác có thể tái sử dụng nhất quán.
    """

    def __init__(self, freeze_base: bool = True, **kwargs):
        """Thực hiện bước init trong quy trình hiện tại.

        Parameters
        ----------
        freeze_base : bool, optional
            Giá trị ``freeze_base`` được sử dụng trong phép xử lý.
        **kwargs : dict
            Các đối số từ khóa bổ sung.

        Raises
        ------
        ValueError
            Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
        """
        super().__init__()
        model_name = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        self.model, _, self.preprocess = create_model_and_transforms(model_name)
        self.tokenizer = get_tokenizer(model_name)

        self.num_visual_tokens = int(kwargs.get("num_visual_tokens", 0))
        self.local_pool_grid = int(kwargs.get("local_pool_grid", 2))
        single_view_pool_shape = kwargs.get("single_view_pool_shape", None)
        if single_view_pool_shape is None:
            self.single_view_pool_shape = None
        else:
            if len(single_view_pool_shape) != 2:
                raise ValueError("single_view_pool_shape must contain [rows, columns].")
            self.single_view_pool_shape = tuple(
                int(value) for value in single_view_pool_shape
            )
            if any(value < 1 for value in self.single_view_pool_shape):
                raise ValueError("single_view_pool_shape values must be positive.")
        self.include_local_cls_token = bool(
            kwargs.get("include_local_cls_token", False)
        )
        self.contrastive_local_weight = float(
            kwargs.get("contrastive_local_weight", 0.25)
        )
        self.contrastive_pooling = str(
            kwargs.get("contrastive_pooling", "mean")
        ).lower()
        if self.contrastive_pooling not in {"mean", "attention"}:
            raise ValueError(
                "contrastive_pooling must be either 'mean' or 'attention'."
            )
        if self.contrastive_pooling == "attention":
            self.contrastive_local_pooler = GlobalGuidedAttentionPool(
                dim=512,
                hidden_dim=int(
                    kwargs.get("contrastive_attention_hidden_dim", 128)
                ),
            )
        else:
            self.contrastive_local_pooler = None
        self.tile_encode_chunk_size = int(kwargs.get("tile_encode_chunk_size", 32))
        self.local_tile_grad_enabled = bool(
            kwargs.get("local_tile_grad_enabled", True)
        )
        self.local_tile_gradient_checkpointing = bool(
            kwargs.get("local_tile_gradient_checkpointing", True)
        )
        if self.local_pool_grid < 1:
            raise ValueError("local_pool_grid must be positive.")
        if self.contrastive_local_weight < 0:
            raise ValueError("contrastive_local_weight must be non-negative.")
        if self.tile_encode_chunk_size < 1:
            raise ValueError("tile_encode_chunk_size must be positive.")
        if self.single_view_pool_shape is not None:
            pooled_tokens = math.prod(self.single_view_pool_shape)
            if self.num_visual_tokens != pooled_tokens:
                raise ValueError(
                    "single_view_pool_shape must produce exactly "
                    f"num_visual_tokens ({pooled_tokens} != "
                    f"{self.num_visual_tokens})."
                )
        if self.num_visual_tokens > 0:
            resampler_cfg = kwargs.get("resampler_cfg", {}) or {}
            self.visual_resampler = SpatialTokenResampler(
                dim=512,
                num_tokens=self.num_visual_tokens,
                **resampler_cfg,
            )

        self.return_local = False
        self.last_image_key_padding_mask: Optional[torch.Tensor] = None
        self.last_local_token_boxes: Optional[torch.Tensor] = None
        self.explain_mode = False
        self.last_global_feature: Optional[torch.Tensor] = None
        self.last_local_tokens: Optional[torch.Tensor] = None
        self.last_local_mask: Optional[torch.Tensor] = None

        if freeze_base:
            for parameter in self.model.parameters():
                parameter.requires_grad = False

    @property
    def tokenizer_obj(self):
        """Thực hiện bước tokenizer obj trong quy trình hiện tại.

        Returns
        -------
        object
            Kết quả được tạo bởi bước xử lý của hàm.
        """
        return self.tokenizer

    def _encode_local_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Mã hóa local văn bản cho bước xử lý hiện tại.

        Parameters
        ----------
        input_ids : torch.Tensor
            Dữ liệu nguồn của phép xử lý.
        attention_mask : Optional[torch.Tensor]
            Giá trị ``attention_mask`` được sử dụng trong phép xử lý.

        Returns
        -------
        torch.Tensor
            Kết quả được tạo bởi bước xử lý của hàm.

        Raises
        ------
        RuntimeError
            Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
        """
        text_module = getattr(self.model, "text", getattr(self.model, "text_model", None))
        if text_module is None:
            raise RuntimeError("BiomedCLIP text module was not found.")
        transformer = getattr(text_module, "transformer", text_module)
        output = (
            transformer(input_ids, attention_mask=attention_mask)
            if attention_mask is not None
            else transformer(input_ids)
        )
        hidden = output[0] if isinstance(output, (tuple, list)) else output.last_hidden_state
        projection = getattr(text_module, "proj", None)
        if projection is None:
            raise RuntimeError("BiomedCLIP text projection was not found.")
        return projection(hidden)

    @staticmethod
    def _pool_spatial_patch_tokens(
        patch_tokens: torch.Tensor,
        pool_shape: tuple[int, int],
    ) -> torch.Tensor:
        """Thực hiện bước pool spatial patch các token trong quy trình hiện tại.

        Parameters
        ----------
        patch_tokens : torch.Tensor
            Giá trị ``patch_tokens`` được sử dụng trong phép xử lý.
        pool_shape : tuple[int, int]
            Giá trị ``pool_shape`` được sử dụng trong phép xử lý.

        Returns
        -------
        torch.Tensor
            Kết quả được tạo bởi bước xử lý của hàm.

        Raises
        ------
        RuntimeError
            Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
        ValueError
            Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
        """
        if patch_tokens.ndim != 3 or patch_tokens.size(1) < 2:
            raise ValueError(
                "BiomedCLIP visual tokens must have shape [B,1+P,D]."
            )
        spatial_tokens = patch_tokens[:, 1:, :]
        side = int(math.sqrt(spatial_tokens.size(1)))
        if side * side != spatial_tokens.size(1):
            raise RuntimeError("BiomedCLIP patch tokens do not form a square grid.")
        grid = spatial_tokens.transpose(1, 2).reshape(
            spatial_tokens.size(0), spatial_tokens.size(2), side, side
        )
        pooled = F.adaptive_avg_pool2d(
            grid, pool_shape
        ).flatten(2).transpose(1, 2)
        return pooled

    def _pool_patch_tokens(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """Thực hiện bước pool patch các token trong quy trình hiện tại.

        Parameters
        ----------
        patch_tokens : torch.Tensor
            Giá trị ``patch_tokens`` được sử dụng trong phép xử lý.

        Returns
        -------
        torch.Tensor
            Kết quả được tạo bởi bước xử lý của hàm.
        """
        cls_token = patch_tokens[:, :1, :]
        pooled = self._pool_spatial_patch_tokens(
            patch_tokens,
            (self.local_pool_grid, self.local_pool_grid),
        )
        if self.include_local_cls_token:
            return torch.cat([cls_token, pooled], dim=1)
        return pooled

    def _encode_local_tile_chunk(self, tile_chunk: torch.Tensor) -> torch.Tensor:
        """Mã hóa local tile chunk cho bước xử lý hiện tại.

        Parameters
        ----------
        tile_chunk : torch.Tensor
            Giá trị ``tile_chunk`` được sử dụng trong phép xử lý.

        Returns
        -------
        torch.Tensor
            Kết quả được tạo bởi bước xử lý của hàm.
        """
        patch_features = self.model.visual.trunk.forward_features(tile_chunk)
        projected = self.model.visual.head(patch_features)
        return self._pool_patch_tokens(projected)

    def _encode_valid_local_tiles(
        self,
        valid_tiles: torch.Tensor,
        track_gradients: bool,
    ) -> torch.Tensor:
        """Mã hóa valid local tiles cho bước xử lý hiện tại.

        Parameters
        ----------
        valid_tiles : torch.Tensor
            Giá trị ``valid_tiles`` được sử dụng trong phép xử lý.
        track_gradients : bool
            Giá trị ``track_gradients`` được sử dụng trong phép xử lý.

        Returns
        -------
        torch.Tensor
            Kết quả được tạo bởi bước xử lý của hàm.

        Raises
        ------
        ValueError
            Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
        """
        if valid_tiles.size(0) == 0:
            raise ValueError("At least one valid high-resolution tile is required.")

        encoded_chunks = []
        for start in range(0, valid_tiles.size(0), self.tile_encode_chunk_size):
            tile_chunk = valid_tiles[start:start + self.tile_encode_chunk_size]
            if track_gradients:
                if self.local_tile_gradient_checkpointing:
                    encoded = checkpoint(
                        self._encode_local_tile_chunk,
                        tile_chunk,
                        use_reentrant=False,
                    )
                else:
                    encoded = self._encode_local_tile_chunk(tile_chunk)
            else:
                with torch.no_grad():
                    encoded = self._encode_local_tile_chunk(tile_chunk)
            encoded_chunks.append(encoded)
        return torch.cat(encoded_chunks, dim=0)

    def _should_track_local_tile_gradients(self) -> bool:
        """Thực hiện bước should track local tile gradients trong quy trình hiện tại.

        Returns
        -------
        bool
            Kết quả được tạo bởi bước xử lý của hàm.
        """
        if not self.local_tile_grad_enabled or not self.training:
            return False
        if not torch.is_grad_enabled():
            return False
        return any(
            parameter.requires_grad
            for parameter in self.model.visual.parameters()
        )

    def _expand_local_boxes(self, tile_boxes: torch.Tensor) -> torch.Tensor:
        """Thực hiện bước expand local boxes trong quy trình hiện tại.

        Parameters
        ----------
        tile_boxes : torch.Tensor
            Giá trị ``tile_boxes`` được sử dụng trong phép xử lý.

        Returns
        -------
        torch.Tensor
            Kết quả được tạo bởi bước xử lý của hàm.
        """
        grid = self.local_pool_grid
        left, top, right, bottom = tile_boxes.unbind(dim=-1)
        width = right - left
        height = bottom - top
        boxes = []
        if self.include_local_cls_token:
            boxes.append(tile_boxes)
        for row in range(grid):
            for column in range(grid):
                boxes.append(
                    torch.stack(
                        [
                            left + width * column / grid,
                            top + height * row / grid,
                            left + width * (column + 1) / grid,
                            top + height * (row + 1) / grid,
                        ],
                        dim=-1,
                    )
                )
        return torch.stack(boxes, dim=2).flatten(1, 2)

    @staticmethod
    def _grid_boxes(
        batch_size: int,
        pool_shape: tuple[int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Thực hiện bước grid boxes trong quy trình hiện tại.

        Parameters
        ----------
        batch_size : int
            Số lượng, kích thước hoặc tỷ lệ được sử dụng.
        pool_shape : tuple[int, int]
            Giá trị ``pool_shape`` được sử dụng trong phép xử lý.
        device : torch.device
            Thiết bị thực thi phép tính.
        dtype : torch.dtype
            Giá trị ``dtype`` được sử dụng trong phép xử lý.

        Returns
        -------
        torch.Tensor
            Kết quả được tạo bởi bước xử lý của hàm.
        """
        rows, columns = pool_shape
        boxes = []
        for row in range(rows):
            for column in range(columns):
                boxes.append(
                    [
                        column / columns,
                        row / rows,
                        (column + 1) / columns,
                        (row + 1) / rows,
                    ]
                )
        return torch.tensor(boxes, device=device, dtype=dtype).unsqueeze(0).expand(
            batch_size, -1, -1
        )

    def _encode_single_view_image(self, images: torch.Tensor) -> torch.Tensor:
        """Mã hóa single view ảnh cho bước xử lý hiện tại.

        Parameters
        ----------
        images : torch.Tensor
            Giá trị ``images`` được sử dụng trong phép xử lý.

        Returns
        -------
        torch.Tensor
            Kết quả được tạo bởi bước xử lý của hàm.

        Raises
        ------
        RuntimeError
            Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
        """
        if self.single_view_pool_shape is None:
            raise RuntimeError("single_view_pool_shape is not configured.")
        if not hasattr(self, "visual_resampler"):
            raise RuntimeError("Matched single-view tokens require a visual resampler.")

        patch_features = self.model.visual.trunk.forward_features(images)
        projected = self.model.visual.head(patch_features)
        local_tokens = self._pool_spatial_patch_tokens(
            projected, self.single_view_pool_shape
        )
        global_feature = F.normalize(projected[:, 0], dim=-1, eps=1e-6)
        local_tokens = F.normalize(local_tokens, dim=-1, eps=1e-6)
        local_mask = torch.ones(
            local_tokens.shape[:2],
            dtype=torch.bool,
            device=local_tokens.device,
        )
        local_boxes = self._grid_boxes(
            local_tokens.size(0),
            self.single_view_pool_shape,
            local_tokens.device,
            local_tokens.dtype,
        )

        if self.explain_mode:
            local_tokens = local_tokens.detach().requires_grad_(True)
        self.last_global_feature = global_feature.detach()
        self.last_local_tokens = (
            local_tokens if self.explain_mode else local_tokens.detach()
        )
        self.last_local_mask = local_mask.detach()
        self.visual_resampler.capture_explanations = self.explain_mode

        image_tokens = self.visual_resampler(
            global_feature,
            local_tokens,
            local_mask,
            local_boxes,
        )
        self.last_local_token_boxes = local_boxes.detach()
        output_local_mask = getattr(
            self.visual_resampler, "last_output_mask", local_mask
        )
        self.last_image_key_padding_mask = torch.cat(
            [
                torch.zeros(
                    local_tokens.size(0),
                    1,
                    dtype=torch.bool,
                    device=local_tokens.device,
                ),
                ~output_local_mask.to(device=local_tokens.device, dtype=torch.bool),
            ],
            dim=1,
        )
        return image_tokens

    def _encode_high_res_image(
        self,
        images: torch.Tensor,
        tile_values: torch.Tensor,
        tile_mask: Optional[torch.Tensor],
        tile_boxes: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Mã hóa high res ảnh cho bước xử lý hiện tại.

        Parameters
        ----------
        images : torch.Tensor
            Giá trị ``images`` được sử dụng trong phép xử lý.
        tile_values : torch.Tensor
            Giá trị ``tile_values`` được sử dụng trong phép xử lý.
        tile_mask : Optional[torch.Tensor]
            Giá trị ``tile_mask`` được sử dụng trong phép xử lý.
        tile_boxes : Optional[torch.Tensor]
            Giá trị ``tile_boxes`` được sử dụng trong phép xử lý.

        Returns
        -------
        torch.Tensor
            Kết quả được tạo bởi bước xử lý của hàm.

        Raises
        ------
        RuntimeError
            Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
        """
        if not hasattr(self, "visual_resampler"):
            raise RuntimeError("High-resolution tiles require num_visual_tokens > 0.")

        batch_size, tile_count, channels, height, width = tile_values.shape
        global_feature = F.normalize(self.model.encode_image(images), dim=-1)
        if tile_mask is None:
            tile_mask = torch.ones(
                batch_size, tile_count, dtype=torch.bool, device=tile_values.device
            )
        else:
            tile_mask = tile_mask.to(device=tile_values.device, dtype=torch.bool)
        if tile_boxes is None:
            tile_boxes = torch.zeros(
                batch_size, tile_count, 4, device=tile_values.device
            )

        flat_tiles = tile_values.reshape(-1, channels, height, width)
        valid_flat = tile_mask.flatten()
        valid_tiles = flat_tiles[valid_flat]
        valid_local = self._encode_valid_local_tiles(
            valid_tiles,
            track_gradients=self._should_track_local_tile_gradients(),
        )

        tokens_per_tile = self.local_pool_grid ** 2 + int(
            self.include_local_cls_token
        )
        local_tokens = torch.zeros(
            batch_size * tile_count,
            tokens_per_tile,
            valid_local.size(-1),
            device=valid_local.device,
            dtype=valid_local.dtype,
        )
        local_tokens[valid_flat] = valid_local
        local_tokens = local_tokens.reshape(batch_size, tile_count * tokens_per_tile, -1)
        local_mask = tile_mask.unsqueeze(-1).expand(-1, -1, tokens_per_tile).flatten(1)
        local_boxes = self._expand_local_boxes(tile_boxes.to(local_tokens.device))
        local_tokens = F.normalize(local_tokens, dim=-1)

        if self.explain_mode:
            # Thu thập và xử lý biểu diễn đặc trưng của mô hình.
            # Thu thập và xử lý biểu diễn đặc trưng của mô hình.
            # Chuẩn bị và xử lý đầu vào hoặc đặc trưng hình ảnh.
            local_tokens = local_tokens.detach().requires_grad_(True)
        self.last_global_feature = global_feature.detach()
        self.last_local_tokens = (
            local_tokens if self.explain_mode else local_tokens.detach()
        )
        self.last_local_mask = local_mask.detach()
        self.visual_resampler.capture_explanations = self.explain_mode

        image_tokens = self.visual_resampler(
            global_feature,
            local_tokens,
            local_mask,
            local_boxes,
        )
        self.last_local_token_boxes = local_boxes.detach()
        output_local_mask = getattr(
            self.visual_resampler, "last_output_mask", None
        )
        if (
            output_local_mask is None
            or output_local_mask.shape != (batch_size, image_tokens.size(1) - 1)
        ):
            output_local_mask = torch.ones(
                batch_size,
                image_tokens.size(1) - 1,
                dtype=torch.bool,
                device=image_tokens.device,
            )
        self.last_image_key_padding_mask = torch.cat(
            [
                torch.zeros(
                    batch_size, 1, dtype=torch.bool, device=image_tokens.device
                ),
                ~output_local_mask.to(device=image_tokens.device, dtype=torch.bool),
            ],
            dim=1,
        )
        return image_tokens

    def pool_contrastive_image_features(
        self,
        image_features: torch.Tensor,
    ) -> torch.Tensor:
        """Thực hiện bước pool contrastive ảnh các đặc trưng trong quy trình hiện tại.

        Parameters
        ----------
        image_features : torch.Tensor
            Ảnh hoặc biểu diễn ảnh đầu vào.

        Returns
        -------
        torch.Tensor
            Kết quả được tạo bởi bước xử lý của hàm.

        Raises
        ------
        ValueError
            Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
        """
        if image_features.ndim == 2:
            return F.normalize(image_features, dim=-1)
        if image_features.ndim != 3 or image_features.size(1) < 1:
            raise ValueError(
                "image_features must have shape [B,D] or [B,T,D]."
            )

        global_feature = F.normalize(image_features[:, 0], dim=-1, eps=1e-6)
        if image_features.size(1) == 1 or self.contrastive_local_weight == 0:
            return global_feature

        local_features = image_features[:, 1:]
        padding_mask = self.last_image_key_padding_mask
        if padding_mask is None:
            valid = torch.ones(
                local_features.shape[:2],
                dtype=torch.bool,
                device=local_features.device,
            )
        else:
            if padding_mask.shape != image_features.shape[:2]:
                raise ValueError(
                    "Image padding mask does not match the visual-token sequence."
                )
            valid = ~padding_mask[:, 1:].to(
                device=local_features.device, dtype=torch.bool
            )
        if self.contrastive_local_pooler is not None:
            local_summary = self.contrastive_local_pooler(
                global_feature,
                local_features,
                valid,
            )
        else:
            weights = valid.unsqueeze(-1).to(local_features.dtype)
            local_summary = (local_features * weights).sum(dim=1) / weights.sum(
                dim=1
            ).clamp_min(1.0)
        local_summary = F.normalize(local_summary, dim=-1, eps=1e-6)
        return F.normalize(
            global_feature + self.contrastive_local_weight * local_summary,
            dim=-1,
            eps=1e-6,
        )

    def forward(
        self,
        images: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        tile_values: Optional[torch.Tensor] = None,
        tile_mask: Optional[torch.Tensor] = None,
        tile_boxes: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Thực hiện lượt lan truyền xuôi của mô hình.

        Parameters
        ----------
        images : torch.Tensor
            Giá trị ``images`` được sử dụng trong phép xử lý.
        input_ids : Optional[torch.Tensor]
            Dữ liệu nguồn của phép xử lý.
        attention_mask : Optional[torch.Tensor]
            Giá trị ``attention_mask`` được sử dụng trong phép xử lý.
        tile_values : Optional[torch.Tensor]
            Giá trị ``tile_values`` được sử dụng trong phép xử lý.
        tile_mask : Optional[torch.Tensor]
            Giá trị ``tile_mask`` được sử dụng trong phép xử lý.
        tile_boxes : Optional[torch.Tensor]
            Giá trị ``tile_boxes`` được sử dụng trong phép xử lý.
        **kwargs : dict
            Các đối số từ khóa bổ sung.

        Returns
        -------
        object
            Kết quả được tạo bởi bước xử lý của hàm.
        """
        del kwargs
        return_local = bool(self.return_local)
        self.last_image_key_padding_mask = None

        if input_ids is None:
            text_features = None
        elif return_local:
            text_features = self._encode_local_text(input_ids, attention_mask)
        else:
            text_features = self.model.encode_text(input_ids)

        if images is None:
            image_features = None
        elif tile_values is not None:
            image_features = self._encode_high_res_image(
                images, tile_values, tile_mask, tile_boxes
            )
        elif self.single_view_pool_shape is not None:
            image_features = self._encode_single_view_image(images)
        elif return_local:
            patch_features = self.model.visual.trunk.forward_features(images)
            image_features = self.model.visual.head(patch_features)
        else:
            image_features = self.model.encode_image(images)

        if image_features is not None and image_features.ndim == 2:
            image_features = F.normalize(image_features, dim=-1)
        if text_features is not None and text_features.ndim == 2:
            text_features = F.normalize(text_features, dim=-1)
        return image_features, text_features
