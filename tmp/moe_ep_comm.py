def calculate_moe_ep_traffic(
    topk: float,                    # K_top: Top-k 路由数量，即每个 Token 激活的专家数
    global_batch_size: float,       # B: 全局批量大小 (Global Batch Size)
    seq_len: float,                 # S: 序列长度 (Sequence Length)
    hidden_size: float,             # H: 隐藏层维度 (Hidden Dimension)
    num_layers: float,              # L: 模型总层数 (Number of Layers)。注意：EP 通信仅发生在 MoE 层，
                                    #    若模型中包含 Dense 层，建议传入 num_moe_layers 以得到更精确的估算。
    pipeline_parallel: float        # PP: 流水线并行度 (Pipeline Parallelism Degree)
) -> float:
    """
    计算 MoE（混合专家）模型中的 EP（专家并行）通信流量。

    公式: EP = 8 * topk * global_batch_size * seq_len * hidden_size * num_layers / pipeline_parallel

    常数 8 的含义：BF16 每元素 2 字节 × 2(前向+反向) × 2(dispatch+combine) = 8

    返回：
        float: 计算得到的 EP 通信流量值（单位为字节）。

    异常：
        ValueError: 当 pipeline_parallel 为 0 时抛出，防止除以零错误。
    """
    if pipeline_parallel == 0:
        raise ValueError("流水线并行度 (pipeline_parallel) 不能为 0，请传入有效值。")

    ep_value = (8.0 * topk * global_batch_size * seq_len * hidden_size * num_layers) / pipeline_parallel
    return ep_value


# --- 使用示例 ---
if __name__ == "__main__":
    # 典型参数配置示例（假设值）
    result = calculate_moe_ep_traffic(
        topk=8,
        global_batch_size=32,
        seq_len=2048,
        hidden_size=4096,
        num_layers=32,
        pipeline_parallel=4
    )

    print(f"计算得到的 EP 流量值为: {result:.2f} 字节")

    # 另一组参数
    result_default = calculate_moe_ep_traffic(
        topk=8,
        global_batch_size=16,
        seq_len=1024,
        hidden_size=2048,
        num_layers=24,
        pipeline_parallel=2
    )
    print(f"另一组参数计算结果: {result_default:.2f} 字节")
