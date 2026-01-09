import torch
from src.transformer import Transformer

def main():
    # 超参数
    src_vocab_size = 100
    tgt_vocab_size = 100
    dim = 512
    n_heads = 8
    n_layers = 6
    hidden_dim = 2048
    max_seq_len = 50
    dropout = 0.1
    
    # 实例化模型
    model = Transformer(
        src_vocab_size, 
        tgt_vocab_size, 
        dim, 
        n_heads, 
        n_layers, 
        hidden_dim, 
        max_seq_len, 
        dropout
    )
    
    # 模拟输入数据
    batch_size = 2
    src_len = 10
    tgt_len = 12
    
    # 随机生成 src 和 tgt 序列 (假设 pad_token_id=0)
    # 确保没有 pad token 影响简单测试，或者手动插入
    src = torch.randint(1, src_vocab_size, (batch_size, src_len))
    tgt = torch.randint(1, tgt_vocab_size, (batch_size, tgt_len))
    
    # 前向传播
    output = model(src, tgt)
    
    print("Model Architecture:")
    # print(model)
    print("\nTest Input:")
    print(f"Source Shape: {src.shape}")
    print(f"Target Shape: {tgt.shape}")
    
    print("\nModel Output:")
    print(f"Output Shape: {output.shape}") # 预期 [batch_size, tgt_len, tgt_vocab_size]

if __name__ == "__main__":
    main()
