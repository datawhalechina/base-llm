import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    """
    多头自注意力机制
    """
    def __init__(self, dim, n_heads, dropout=0.1):
        super().__init__()
        
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        
        # 线性变换层
        self.wq = nn.Linear(dim, dim)
        self.wk = nn.Linear(dim, dim)
        self.wv = nn.Linear(dim, dim)
        self.wo = nn.Linear(dim, dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # 1. 线性投影 [batch_size, seq_len, dim]
        # 这里的 q, k, v 可能来自不同的源（交叉注意力时）
        xq = self.wq(q)
        xk = self.wk(k)
        xv = self.wv(v)
        
        # 2. 拆分多头 [batch_size, seq_len, n_heads, head_dim] -> [batch_size, n_heads, seq_len, head_dim]
        xq = xq.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        xk = xk.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        xv = xv.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        
        # 3. 计算注意力分数
        # scores: [batch_size, n_heads, seq_len_q, seq_len_k]
        scores = torch.matmul(xq, xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        # 4. 归一化和加权
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # context: [batch_size, n_heads, seq_len_q, head_dim]
        context = torch.matmul(attention_weights, xv)
        
        # 5. 合并多头 [batch_size, seq_len_q, dim]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.dim)
        
        # 6. 输出层
        output = self.wo(context)
        return output

if __name__ == "__main__":
    # 准备参数
    batch_size, seq_len, dim = 2, 10, 512
    n_heads = 8
    
    # 初始化模块
    mha = MultiHeadAttention(dim, n_heads)
    
    # 准备输入 (Query, Key, Value 相同)
    x = torch.randn(batch_size, seq_len, dim)
    
    # 前向传播
    output = mha(x, x, x)
    
    # 验证输出
    print("--- MultiHeadAttention Test ---")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
