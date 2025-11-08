import torch
import numpy as np

def get_r2_order(n, seed=0.5):
    """
    根据 R₂ 序列（1维 additive recurrence）生成长度 n 的一个低差异排序。
    公式：x_i = (seed + a1 * i) mod 1, 其中 a1 = 1/φ₂，φ₂ ≈ 1.324717957244746，
    i 从 1 开始。
    返回按照 x_i 从小到大排序的原始索引（0-based）。
    """
    phi = 1.6180339887498948482
    a1 = 1.0 / phi
    indices = torch.arange(0, n, dtype=torch.float32)
    # 计算 R₂ 序列值
    r = (seed + a1 * indices) % 1
    # 按照 r 的值排序，返回排序后的（原始）索引（注意这里转换为 0-based）
    sorted_order = torch.argsort(r)
    return sorted_order


def split_indices_into_groups(n, num_groups):
    """
    根据 10 组要求，按照每组大小比例为 (i+1)^(2.2)（i=0,...,num_groups-1）来计算各组大小，
    并返回一个长度为 num_groups 的列表，每个元素为对应组的索引（从 0 到 n-1）。
    """
    # 计算权重
    group_weights = np.array([(i + 1) ** 2.2 for i in range(num_groups)], dtype=np.float32)
    # 归一化后乘以 n
    group_sizes = np.round(n * group_weights / group_weights.sum()).astype(int)
    # 调整以确保总和为 n
    diff = n - group_sizes.sum()
    group_sizes[0] += diff  # 简单将差值加到第一组

    groups = []
    start = 0
    for size in group_sizes:
        groups.append(torch.arange(start, start + size))
        start += size
    return groups, group_sizes


if __name__ == '__main__':
    # 示例：假设 data 的形状为 [B, 256, C]
    B = 4  # batch_size
    C = 10  # 特征数
    data = torch.randn(B, 256, C)

    # 1. 计算 256 个点的低差异排列（所有 batch 共享同一排列）
    order = get_r2_order(256, seed=0.5)  # shape [256]

    # 2. 根据要求将 256 个索引划分为 10 组
    num_groups = 10
    groups_list, group_sizes = split_indices_into_groups(256, num_groups)
    print("各组大小:", group_sizes)  # 输出每组大小

    # 3. 对于每个组，利用排序后的 order 来提取每个 batch 对应的行。
    #    我们希望得到一个列表，每个元素为 shape [B, group_size, C]
    group_tensors = []
    for idx in groups_list:
        # 从排序后的 order 中选取当前组的索引
        group_idx = order[idx]  # 此时 group_idx 的 shape 为 [group_size]
        # 对于每个 batch，从 data 的第二个维度中按照 group_idx 进行索引
        # 方法1：使用 advanced indexing，先将 group_idx 扩展到 batch 维度
        # 这里 data 的形状为 [B, 256, C]，我们希望选取第 1 维对应 group_idx
        group_tensor = data[:, group_idx, :]  # 结果 shape [B, group_size, C]
        group_tensors.append(group_tensor)
    #
    # # 打印各组的形状
    # for i, gt in enumerate(group_tensors):
    #     print(f"Group {i + 1}: shape {gt.shape}")