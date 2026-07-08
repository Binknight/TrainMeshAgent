def calculate_moe_hbm(
    num_dense_layers: int,           # L_dense: 密集层 (Dense FFN) 总数
    num_moe_layers: int,             # L_moe: MoE 层总数
    pipeline_parallel: int,          # PP: 流水线并行度 (Pipeline Parallelism size)
    hidden_size: int,                # H: 隐藏层维度 (Hidden size)
    ffn_hidden_size: int,            # F_dense: 密集层 FFN 中间维度
    tensor_parallel: int,            # TP: 张量并行度 (Tensor Parallelism size)
    expert_ffn_hidden_size: int,     # F_expert: MoE 层每个专家的 FFN 中间维度
    num_experts: int,                # E: 每 MoE 层的专家数
    expert_parallel: int,            # EP: 专家并行度 (Expert Parallelism size)
    vocab_size: int,                 # V: 词表大小 (Vocabulary size)
    expert_tensor_parallel: int = 1, # TP_e: 专家内部张量并行度 (默认 1, 即不额外切分)
    bytes_per_param: int = 18        # K: 混合精度训练下每参数占用的字节数 (默认 18, 对应 BF16 参数2B+FP32主参数4B+梯度4B+动量4B+方差4B)
) -> float:
    """
    计算混合精度 (BF16 + Adam) 训练 MoE 大模型单卡所需的 HBM (显存) 大小。

    返回:
    float: 模型所需的总显存大小，单位为字节 (Bytes)。
    """

    # 1. 密集层 (Dense Layers) 显存计算
    # 对应公式第一项: (num_dense_layers / pipeline_parallel) * ((4*hidden_size^2 + 3*hidden_size*ffn_hidden_size) / tensor_parallel + 2*hidden_size)
    term_dense = (num_dense_layers / pipeline_parallel) * (
        (4 * hidden_size**2 + 3 * hidden_size * ffn_hidden_size) / tensor_parallel + 2 * hidden_size
    )

    # 2. MoE 层 (MoE Layers) 显存计算
    # 对应公式第二项: (num_moe_layers / pipeline_parallel) * ((4*hidden_size^2 / tensor_parallel) + (3*num_experts*hidden_size*expert_ffn_hidden_size) / (expert_parallel * expert_tensor_parallel) + (num_experts*hidden_size / tensor_parallel) + 2*hidden_size)
    term_moe = (num_moe_layers / pipeline_parallel) * (
        (4 * hidden_size**2 / tensor_parallel) +
        (3 * num_experts * hidden_size * expert_ffn_hidden_size) / (expert_parallel * expert_tensor_parallel) +
        (num_experts * hidden_size / tensor_parallel) +
        2 * hidden_size
    )

    # 3. 词嵌入层 (Embedding) 显存计算
    # 对应公式第三项: vocab_size * hidden_size / tensor_parallel
    term_embedding = (vocab_size * hidden_size) / tensor_parallel

    # 4. 三项求和后乘 bytes_per_param，得到最终字节数
    total_bytes = (term_dense + term_moe + term_embedding) * bytes_per_param

    return total_bytes


# ==========================================
# 使用示例 (基于常见大模型参数配置)
# ==========================================
if __name__ == "__main__":
    # 假设一个 70B 级别 MoE 模型的配置 (举例)
    params = {
        "num_dense_layers": 20,         # 密集层数
        "num_moe_layers": 60,           # MoE 层数
        "pipeline_parallel": 4,         # 流水线并行度 (分成4个阶段)
        "hidden_size": 8192,            # 隐藏层维度
        "ffn_hidden_size": 28672,       # 密集 FFN 维度
        "tensor_parallel": 8,           # 张量并行度
        "expert_ffn_hidden_size": 28672,# 专家 FFN 维度
        "num_experts": 8,               # 每层专家数量
        "expert_parallel": 4,           # 专家并行度
        "vocab_size": 32000,            # 词表大小
        "expert_tensor_parallel": 1,   # 专家内张量并行度
        "bytes_per_param": 18           # 每参数字节数 (BF16+Adam)
    }

    bytes_required = calculate_moe_hbm(**params)

    # 转换为常见单位 (1 GB = 1024^3 Bytes)
    gb_required = bytes_required / (1024**3)

    print(f"该模型单卡显存需求: {bytes_required:,.0f} Bytes")
    print(f"该模型单卡显存需求: {gb_required:.2f} GB")
