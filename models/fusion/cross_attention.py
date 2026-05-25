import torch
import torch.nn as nn

class CrossAttentionFusion(nn.Module):
    def __init__(self, text_dim=512, img_dim=512, num_heads=8, **kwargs):
        super().__init__()
        # Đảm bảo chiều nhúng của ảnh và text giống nhau (BioMedCLIP thường là 512)
        self.embed_dim = img_dim 
        
        # ==========================================
        # TOP BRANCH (Theo bản vẽ)
        # ==========================================
        # Nhận Query là Image, Key/Value là Text
        self.top_cross_attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim, num_heads=num_heads, batch_first=True
        )
        self.top_linear = nn.Linear(self.embed_dim, self.embed_dim)

        # ==========================================
        # BOTTOM BRANCH (Theo bản vẽ)
        # ==========================================
        # Nhận Query là Text, Key/Value là Image
        self.bottom_cross_attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim, num_heads=num_heads, batch_first=True
        )
        self.bottom_linear = nn.Linear(self.embed_dim, self.embed_dim)

        # ==========================================
        # TRANSFORMER BLOCK (Theo bản vẽ)
        # ==========================================
        # Khối Transformer Encoder tiêu chuẩn để hòa trộn 2 luồng
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim, 
            nhead=num_heads, 
            dim_feedforward=self.embed_dim * 4,
            batch_first=True
        )
        self.transformer_block = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def forward(self, img_feats, txt_feats):
        """
        Đầu vào:
        img_feats: [Batch_size, embed_dim]
        txt_feats: [Batch_size, embed_dim]
        """
        # nn.MultiheadAttention yêu cầu đầu vào dạng chuỗi 3D: [Batch, Seq_Len, Dim]
        # Ta unsqueeze để biến vector toàn cục thành chuỗi có độ dài = 1
        if img_feats.dim() == 2:
            img_feats = img_feats.unsqueeze(1) # -> [B, 1, 512]
        if txt_feats.dim() == 2:
            txt_feats = txt_feats.unsqueeze(1) # -> [B, 1, 512]

        # 1. LUỒNG TRÊN (Query: Ảnh | Key/Value: Text)
        top_out, _ = self.top_cross_attn(
            query=img_feats, key=txt_feats, value=txt_feats
        )
        top_out = self.top_linear(top_out) # Đầu ra: [B, 1, 512]

        # 2. LUỒNG DƯỚI (Query: Text | Key/Value: Ảnh)
        bottom_out, _ = self.bottom_cross_attn(
            query=txt_feats, key=img_feats, value=img_feats
        )
        bottom_out = self.bottom_linear(bottom_out) # Đầu ra: [B, 1, 512]

        # 3. GỘP LUỒNG (Vào Transformer Block)
        # Ghép 2 token đại diện của 2 luồng lại thành một chuỗi độ dài 2
        combined_seq = torch.cat([top_out, bottom_out], dim=1) # -> [B, 2, 512]
        
        # Đưa qua Transformer Block để tương tác nội bộ (Self-Attention) giữa 2 token này
        fused_seq = self.transformer_block(combined_seq) # -> [B, 2, 512]

        # 4. TỔNG HỢP (Pooling)
        # Lấy trung bình cộng (Mean Pooling) của 2 token để trả về 1 vector duy nhất 
        # khớp với yêu cầu đầu vào [B, 512] của Prototypical Head
        fused_global = fused_seq.mean(dim=1)
        
        return fused_global