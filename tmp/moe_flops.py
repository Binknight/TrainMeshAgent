def compute_moe_flops(
    micro_batch_size: int,          # b: 单次数据并行处理的 micro-batch 大小
    seq_len: int,                   # s: 序列长度 (sequence length)
    num_layers: int,                # L: 模型总层数 (total layers)
    hidden_size: int,               # H: 隐藏层维度 (hidden dimension)
    tensor_parallel: int,           # TP: 张量并行度 (Tensor Parallelism size)
    num_moe_layers: int,            # L_moe: MoE 层数 (number of MoE layers)
    expert_ffn_hidden_size: int,    # F_expert: 单个专家 FFN 的隐藏层维度 (expert FFN hidden size)
    expert_parallel: int,           # EP: 专家并行度 (Expert Parallelism size)
    topk: int,                      # K_top: Top-k 激活专家数 (number of activated experts)
    num_shared_expert_layers: int = None  # L_shared: 包含共享专家的层数 (默认等于 num_moe_layers)
) -> float:
    """
    根据 MoE 大模型公式计算单张显卡上的理论 FLOPs (前向/反向传播)。
    对应四项计算：Attention, Dense FFN, Top-k MoE, Shared Expert。
    """

    # 如果没有显式传入共享专家层数，通常与 MoE 层数相同
    if num_shared_expert_layers is None:
        num_shared_expert_layers = num_moe_layers

    # 1. 注意力部分 (Attention) - 所有层
    # 公式: (6 * micro_batch_size * seq_len * num_layers * hidden_size / tensor_parallel) * (4 * hidden_size + 2 * seq_len)
    flops_attention = (6 * micro_batch_size * seq_len * num_layers * hidden_size / tensor_parallel) * (4 * hidden_size + 2 * seq_len)

    # Dense 层数 = 总层数 - MoE 层数
    num_dense_layers = num_layers - num_moe_layers

    # 2. Dense FFN 层 - 非 MoE 层
    # 公式: 6 * micro_batch_size * seq_len * num_dense_layers * 3 * hidden_size * expert_ffn_hidden_size / tensor_parallel
    flops_dense_ffn = (6 * micro_batch_size * seq_len * num_dense_layers * 3 * hidden_size * expert_ffn_hidden_size) / tensor_parallel

    # 3. MoE 专家 FFN (top-k 激活)
    # 公式: (6 * micro_batch_size * seq_len * topk * 3 * hidden_size * expert_ffn_hidden_size * num_moe_layers) / expert_parallel
    flops_moe = (6 * micro_batch_size * seq_len * topk * 3 * hidden_size * expert_ffn_hidden_size * num_moe_layers) / expert_parallel

    # 4. 共享专家 (若存在)
    # 公式: (6 * micro_batch_size * seq_len * 3 * hidden_size * expert_ffn_hidden_size * num_shared_expert_layers) / expert_parallel
    flops_shared = (6 * micro_batch_size * seq_len * 3 * hidden_size * expert_ffn_hidden_size * num_shared_expert_layers) / expert_parallel

    # 总计
    total_flops = flops_attention + flops_dense_ffn + flops_moe + flops_shared

    return total_flops


# ==========================================
# 示例使用 (假设一个典型的大模型配置)
# ==========================================
if __name__ == "__main__":
    # 假设的参数
    config = {
        "micro_batch_size": 4,          # 微批次大小 per GPU
        "seq_len": 2048,                # 序列长度
        "num_layers": 32,               # 总层数
        "hidden_size": 4096,            # 隐藏层维度
        "tensor_parallel": 8,           # 张量并行数
        "num_moe_layers": 16,           # 其中 16 层是 MoE 层
        "expert_ffn_hidden_size": 14336,# 专家 FFN 维度
        "expert_parallel": 8,           # 专家并行数
        "topk": 2,                      # 每个 token 激活 2 个专家
        # num_shared_expert_layers 不传，默认会自动设为 num_moe_layers (16)
    }

    flops = compute_moe_flops(**config)

    # 以 "PFLOPs" (10^15) 为单位打印结果，方便查看
    print(f"单卡理论计算量: {flops / 1e15:.2f} PFLOPs")
