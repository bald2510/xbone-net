"""Global/local pooling controls with the same token entry path as attention."""
import torch
from torch import nn


class MatchedConcatFusion(nn.Module):
    supports_padding_mask = True

    def __init__(self, img_dim=512, text_dim=512, embed_dim=512,
                 hidden_dim=1879, dropout=0.1, pooling='cls'):
        super().__init__()
        if pooling not in {'cls', 'local_mean'}:
            raise ValueError('pooling must be cls or local_mean.')
        self.pooling = pooling
        self.img_input_norm = nn.LayerNorm(img_dim)
        self.txt_input_norm = nn.LayerNorm(text_dim)
        self.img_proj = nn.Linear(img_dim, embed_dim)
        self.txt_proj = nn.Linear(text_dim, embed_dim)
        self.norm_img_global = nn.LayerNorm(embed_dim)
        self.norm_txt_global = nn.LayerNorm(embed_dim)
        self.norm_img = nn.LayerNorm(embed_dim)
        self.norm_txt = nn.LayerNorm(embed_dim)
        self.norm_fuse = nn.LayerNorm(embed_dim)
        self.fusion_mlp = nn.Sequential(nn.Linear(2*embed_dim, hidden_dim), nn.GELU(),
                                       nn.Dropout(dropout), nn.Linear(hidden_dim, embed_dim))

    def _pool(self, features, mask):
        if features.ndim == 2:
            return features
        if self.pooling == 'cls' or features.shape[1] == 1:
            return features[:, 0]
        local = features[:, 1:]
        if mask is None:
            return local.mean(1)
        if mask.shape != local.shape[:2] or mask.all(1).any():
            raise ValueError('Invalid local token mask for pooling.')
        valid = (~mask).unsqueeze(-1).to(local.dtype)
        return (local*valid).sum(1)/valid.sum(1)

    def forward(self, img_feats, txt_feats, img_key_padding_mask=None,
                txt_key_padding_mask=None, return_attn=False):
        image = self.img_proj(self.img_input_norm(img_feats))
        text = self.txt_proj(self.txt_input_norm(txt_feats))
        image = self.norm_img(self.norm_img_global(self._pool(image, img_key_padding_mask)))
        text = self.norm_txt(self.norm_txt_global(self._pool(text, txt_key_padding_mask)))
        fused = self.norm_fuse(self.fusion_mlp(torch.cat([image,text], dim=-1)))
        return (fused, {}) if return_attn else fused
