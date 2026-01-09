import torch
import torch.nn as nn
import math

# 导入组件
from .attention import MultiHeadAttention
from .ffn import FeedForward
from .norm import LayerNorm
from .pos import PositionalEncoding

class EncoderLayer(nn.Module):
    def __init__(self, dim, n_heads, hidden_dim, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(dim, n_heads, dropout)
        self.attention_norm = LayerNorm(dim)
        
        self.feed_forward = FeedForward(dim, hidden_dim, dropout)
        self.ffn_norm = LayerNorm(dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Post-LN 结构：子层 -> Dropout -> 残差相加 -> LayerNorm
        
        # 子层 1：自注意力
        _x = x
        x = self.attention(x, x, x, mask) # Q=K=V=x
        x = self.attention_norm(_x + self.dropout(x))
        
        # 子层 2：前馈网络
        _x = x
        x = self.feed_forward(x)
        x = self.ffn_norm(_x + self.dropout(x))
        
        return x

class DecoderLayer(nn.Module):
    def __init__(self, dim, n_heads, hidden_dim, dropout=0.1):
        super().__init__()
        # 1. 带掩码的自注意力
        self.self_attention = MultiHeadAttention(dim, n_heads, dropout)
        self.self_attention_norm = LayerNorm(dim)
        
        # 2. 交叉注意力
        self.cross_attention = MultiHeadAttention(dim, n_heads, dropout)
        self.cross_attention_norm = LayerNorm(dim)
        
        # 3. 前馈网络
        self.feed_forward = FeedForward(dim, hidden_dim, dropout)
        self.ffn_norm = LayerNorm(dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask, tgt_mask):
        # 子层 1：带掩码的自注意力
        _x = x
        x = self.self_attention(x, x, x, tgt_mask)
        x = self.self_attention_norm(_x + self.dropout(x))
        
        # 子层 2：交叉注意力（Q 来自解码器，K/V 来自编码器输出）
        _x = x
        x = self.cross_attention(x, enc_output, enc_output, src_mask)
        x = self.cross_attention_norm(_x + self.dropout(x))
        
        # 子层 3：前馈网络
        _x = x
        x = self.feed_forward(x)
        x = self.ffn_norm(_x + self.dropout(x))
        
        return x

class Transformer(nn.Module):
    def __init__(self, 
                 src_vocab_size, 
                 tgt_vocab_size, 
                 dim=512, 
                 n_heads=8, 
                 n_layers=6, 
                 hidden_dim=2048, 
                 max_seq_len=5000, 
                 dropout=0.1):
        super().__init__()
        
        self.dim = dim
        
        # 嵌入层与位置编码
        self.src_embedding = nn.Embedding(src_vocab_size, dim)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, dim)
        self.pos_encoder = PositionalEncoding(dim, max_seq_len)
        self.dropout = nn.Dropout(dropout)
        
        # 编码器与解码器堆叠
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(dim, n_heads, hidden_dim, dropout) 
            for _ in range(n_layers)
        ])
        
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(dim, n_heads, hidden_dim, dropout) 
            for _ in range(n_layers)
        ])
        
        # 输出头
        self.output = nn.Linear(dim, tgt_vocab_size)
        
        # 初始化参数
        self._init_parameters()

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def generate_mask(self, src, tgt):
        # src_mask: [batch, 1, 1, src_len]
        # pad token 假设为 0
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        
        # tgt_mask: [batch, 1, tgt_len, tgt_len]
        # 结合 pad mask 和 causal mask
        tgt_len = tgt.size(1)
        tgt_pad_mask = (tgt != 0).unsqueeze(1).unsqueeze(2) # [batch, 1, 1, tgt_len]
        tgt_subsequent_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=tgt.device)).bool()
        tgt_mask = tgt_pad_mask & tgt_subsequent_mask.unsqueeze(0)
        
        return src_mask, tgt_mask

    def encode(self, src, src_mask):
        x = self.src_embedding(src) * math.sqrt(self.dim)
        x = self.pos_encoder(x)
        x = self.dropout(x)
        
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x

    def decode(self, tgt, enc_output, src_mask, tgt_mask):
        x = self.tgt_embedding(tgt) * math.sqrt(self.dim)
        x = self.pos_encoder(x)
        x = self.dropout(x)
        
        for layer in self.decoder_layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        return x

    def forward(self, src, tgt):
        # 生成掩码
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        
        # 编码
        enc_output = self.encode(src, src_mask)
        
        # 解码
        dec_output = self.decode(tgt, enc_output, src_mask, tgt_mask)
        
        # 输出
        logits = self.output(dec_output)
        return logits
