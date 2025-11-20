import math
import torch

def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
    """
    T5 相对位置编码的核心分桶逻辑
    将相对距离（relative_position）映射为一个桶编号（bucket ID）
    """
    relative_buckets = 0
    
    # 1. 处理双向/单向 Attention
    # 如果是双向 Attention (如 Encoder)，正负距离是不同的桶
    if bidirectional:
        num_buckets //= 2
        # 如果距离 > 0 (Key 在 Query 后面)，桶编号加上总数的一半
        relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
        # 取绝对值，统一处理正负距离
        relative_position = torch.abs(relative_position)
    else:
        # 如果是单向 Attention (如 Decoder)，只考虑过去的距离
        relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
    
    # 2. 核心分桶逻辑：近距离精确，远距离模糊
    
    # 前一半的桶（max_exact）用于精确匹配近距离
    max_exact = num_buckets // 2
    is_small = relative_position < max_exact

    # 情况1：距离较小 (is_small 为 True)，直接使用距离作为桶编号
    # 例如距离为 1 -> 桶 1; 距离为 5 -> 桶 5
    
    # 情况2：距离较大 (is_small 为 False)，使用对数公式计算桶编号
    # 使用对数函数 log 把很大的距离压缩到剩下的桶里
    relative_position_if_large = max_exact + (
        torch.log(relative_position.float() / max_exact)
        / math.log(max_distance / max_exact)
        * (num_buckets - max_exact)
    ).to(torch.long)
    
    # 防止越界，最大不超过 num_buckets - 1
    relative_position_if_large = torch.min(
        relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
    )

    # 根据 is_small 的判断，选择使用精确编号还是对数编号
    relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
    return relative_buckets


# 假设有 32 个桶，最大敏感距离为 128
distances = torch.tensor([-10000, -5, -1, 0, 1, 5, 10, 50, 100])
buckets = _relative_position_bucket(distances)
print(f"真实距离: {distances.tolist()}")
print(f"映射桶号: {buckets.tolist()}")