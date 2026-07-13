"""BiomedCLIP backbone with global and high-resolution local patch tokens."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from open_clip import create_model_and_transforms, get_tokenizer


class _ResamplerLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float):
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


class SpatialTokenResampler(nn.Module):
    """Compress all local patch tokens with learned coordinate-aware queries.

    Unlike hard Top-K selection, every valid high-resolution patch can
    contribute through soft attention. The final attention matrix is retained
    so fused attention can later be projected back to source-image coordinates.
    """

    def __init__(
        self,
        dim: int = 512,
        num_tokens: int = 24,
        num_heads: int = 8,
        depth: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        if num_tokens < 1 or depth < 1:
            raise ValueError("num_tokens and depth must be positive.")
        self.num_tokens = int(num_tokens)
        self.queries = nn.Parameter(torch.empty(1, self.num_tokens, dim))
        nn.init.trunc_normal_(self.queries, std=0.02)
        spatial_dim = max(dim // 4, 32)
        self.spatial_projection = nn.Sequential(
            nn.Linear(4, spatial_dim),
            nn.GELU(),
            nn.Linear(spatial_dim, dim),
        )
        self.layers = nn.ModuleList(
            [_ResamplerLayer(dim, num_heads, dropout) for _ in range(depth)]
        )
        self.last_attention: Optional[torch.Tensor] = None
        self.last_local_boxes: Optional[torch.Tensor] = None
        self.capture_explanations = False
        self.last_attention_layers_raw: list[torch.Tensor] = []

    def forward(
        self,
        global_feature: torch.Tensor,
        local_tokens: torch.Tensor,
        local_mask: torch.Tensor,
        local_boxes: torch.Tensor,
    ) -> torch.Tensor:
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
        positioned_tokens = local_tokens + self.spatial_projection(local_boxes)
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
        return torch.cat([global_feature.unsqueeze(1), queries], dim=1)


class BiomedCLIPFoundation(nn.Module):
    """BiomedCLIP global encoder plus frozen local patch encoder."""

    def __init__(self, freeze_base: bool = True, **kwargs):
        super().__init__()
        model_name = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        self.model, _, self.preprocess = create_model_and_transforms(model_name)
        self.tokenizer = get_tokenizer(model_name)

        self.num_visual_tokens = int(kwargs.get("num_visual_tokens", 0))
        self.local_pool_grid = int(kwargs.get("local_pool_grid", 2))
        self.tile_encode_chunk_size = int(kwargs.get("tile_encode_chunk_size", 32))
        if self.local_pool_grid < 1:
            raise ValueError("local_pool_grid must be positive.")
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
        return self.tokenizer

    def _encode_local_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
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

    def _pool_patch_tokens(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """Pool a 14x14 BiomedCLIP patch grid to GxG spatial tokens."""
        patch_tokens = patch_tokens[:, 1:, :]
        side = int(math.sqrt(patch_tokens.size(1)))
        if side * side != patch_tokens.size(1):
            raise RuntimeError("BiomedCLIP patch tokens do not form a square grid.")
        grid = patch_tokens.transpose(1, 2).reshape(
            patch_tokens.size(0), patch_tokens.size(2), side, side
        )
        pooled = F.adaptive_avg_pool2d(
            grid, (self.local_pool_grid, self.local_pool_grid)
        )
        return pooled.flatten(2).transpose(1, 2)

    def _expand_local_boxes(self, tile_boxes: torch.Tensor) -> torch.Tensor:
        """Split each tile box into the same GxG layout as pooled patch tokens."""
        grid = self.local_pool_grid
        left, top, right, bottom = tile_boxes.unbind(dim=-1)
        width = right - left
        height = bottom - top
        boxes = []
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

    def _encode_high_res_image(
        self,
        images: torch.Tensor,
        tile_values: torch.Tensor,
        tile_mask: Optional[torch.Tensor],
        tile_boxes: Optional[torch.Tensor],
    ) -> torch.Tensor:
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
        encoded_chunks = []
        with torch.no_grad():
            for start in range(0, valid_tiles.size(0), self.tile_encode_chunk_size):
                patch_features = self.model.visual.trunk.forward_features(
                    valid_tiles[start:start + self.tile_encode_chunk_size]
                )
                projected = self.model.visual.head(patch_features)
                encoded_chunks.append(self._pool_patch_tokens(projected))
        valid_local = torch.cat(encoded_chunks, dim=0)

        tokens_per_tile = self.local_pool_grid ** 2
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
            # Explanation methods operate on the exact frozen local embeddings
            # consumed during inference. Gradients start at these embeddings, so
            # the large frozen tile encoder does not need to retain its graph.
            local_tokens = local_tokens.detach().requires_grad_(True)
        self.last_global_feature = global_feature.detach()
        self.last_local_tokens = local_tokens
        self.last_local_mask = local_mask.detach()
        self.visual_resampler.capture_explanations = self.explain_mode

        image_tokens = self.visual_resampler(
            global_feature,
            local_tokens,
            local_mask,
            local_boxes,
        )
        self.last_local_token_boxes = local_boxes.detach()
        self.last_image_key_padding_mask = torch.zeros(
            batch_size,
            image_tokens.size(1),
            dtype=torch.bool,
            device=image_tokens.device,
        )
        return image_tokens

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
        del kwargs
        return_local = bool(self.return_local)
        self.last_image_key_padding_mask = None

        if input_ids is None:
            text_features = None
        elif return_local:
            text_features = self._encode_local_text(input_ids, attention_mask)
        else:
            text_features = self.model.encode_text(input_ids)

        if tile_values is not None:
            image_features = self._encode_high_res_image(
                images, tile_values, tile_mask, tile_boxes
            )
        elif return_local:
            patch_features = self.model.visual.trunk.forward_features(images)
            image_features = self.model.visual.head(patch_features)
        else:
            image_features = self.model.encode_image(images)

        if image_features.ndim == 2:
            image_features = F.normalize(image_features, dim=-1)
        if text_features is not None and text_features.ndim == 2:
            text_features = F.normalize(text_features, dim=-1)
        return image_features, text_features
