import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    """
    层归一化 (Layer Normalization)
    公式: y = (x - mean) / sqrt(var + eps) * gamma + beta
    """
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        # 可学习参数 gamma (缩放) 和 beta (偏移)
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        # x: [batch_size, seq_len, dim]
        # 在最后一个维度 (dim) 上计算均值和方差
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        
        # 归一化
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        # 缩放和平移
        return self.gamma * x_norm + self.beta

if __name__ == "__main__":
    # 准备参数
    batch_size, seq_len, dim = 2, 10, 512
    
    # 初始化模块
    ln = LayerNorm(dim)
    
    # 准备输入
    x = torch.randn(batch_size, seq_len, dim)
    
    # 前向传播
    output = ln(x)
    
    # 验证输出
    print("--- LayerNorm Test ---")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
