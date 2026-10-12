"""
MoE (Mixture of Experts) 模型性能指标估算。

与同目录下 ``__init__.py`` 中的稠密模型估算函数（``_estimate_flops``,
``_estimate_hbm_gb`` 等）对应，提供 MoE 模型的三个理论计算公式：

- ``compute_moe_flops()``          — 单卡 FLOPS (前向+反向)
- ``calculate_moe_hbm()``          — 单卡 HBM 显存占用
- ``calculate_moe_ep_traffic()``   — EP (专家并行) 通信流量

参数命名已与 Megatron-LM / MindSpeed-LLM CLI 参数对齐，配合
``app.models.moe_layer_parser`` 可从脚本配置直接推导 num_moe_layers。

参考公式来源：Megatron-LM MoE training 文档及 DeepSeek/Mixtral/Qwen-MoE 模型实践。
"""

from __future__ import annotations


# ═══════════════════════════════════════════════════════════════════════════
# 1. MoE FLOPS 估算
# ═══════════════════════════════════════════════════════════════════════════


def compute_moe_flops(
    micro_batch_size: int,          # b: 单次数据并行处理的 micro-batch 大小
    seq_len: int,                   # s: 序列长度 (sequence length)
    num_layers: int,                # L: 模型总层数 (total layers)
    hidden_size: int,               # H: 隐藏层维度 (hidden dimension)
    tensor_parallel: int,           # TP: 张量并行度 (Tensor Parallelism size)
    num_moe_layers: int,            # L_moe: MoE 层数
    expert_ffn_hidden_size: int,    # F_expert: 单个专家 FFN 的隐藏层维度
    expert_parallel: int,           # EP: 专家并行度 (Expert Parallelism size)
    topk: int,                      # K_top: Top-k 激活专家数
    num_shared_expert_layers: int = None,  # L_shared: 包含共享专家的层数
    pipeline_parallel: int = 1,     # PP: 流水线并行度 (每卡仅处理 L/PP 层)
) -> float:
    """计算 MoE 大模型单卡理论 FLOPS (前向/反向传播)。

    公式由四项组成，每项除以 PP 得到单卡（单 PP stage）的计算量：

    * **Attention**（所有层）：``(6·b·s·L·H / (TP·PP)) · (4H + 2s)``
    * **Dense FFN**（非 MoE 层）：``6·b·s·(L-M)·3H·F_dense / (TP·PP)``
    * **MoE Expert**（Top-K 激活专家）：``6·b·s·K·3H·F_expert·M / (EP·PP)``
    * **Shared Expert**（若存在）：``6·b·s·3H·F_expert·L_shared / (EP·PP)``
    """
    if num_shared_expert_layers is None:
        num_shared_expert_layers = num_moe_layers

    num_dense_layers = num_layers - num_moe_layers

    # Attention — 所有层，除以 TP 和 PP
    flops_attention = (
        (6 * micro_batch_size * seq_len * num_layers * hidden_size / (tensor_parallel * pipeline_parallel))
        * (4 * hidden_size + 2 * seq_len)
    )

    # Dense FFN — 非 MoE 层，除以 TP 和 PP
    flops_dense_ffn = (
        6 * micro_batch_size * seq_len * num_dense_layers
        * 3 * hidden_size * expert_ffn_hidden_size
    ) / (tensor_parallel * pipeline_parallel)

    # MoE 专家 FFN — Top-K 激活，除以 EP 和 PP
    flops_moe = (
        6 * micro_batch_size * seq_len * topk * 3
        * hidden_size * expert_ffn_hidden_size * num_moe_layers
    ) / (expert_parallel * pipeline_parallel)

    # 共享专家，除以 EP 和 PP
    flops_shared = (
        6 * micro_batch_size * seq_len * 3
        * hidden_size * expert_ffn_hidden_size * num_shared_expert_layers
    ) / (expert_parallel * pipeline_parallel)

    return flops_attention + flops_dense_ffn + flops_moe + flops_shared


# ═══════════════════════════════════════════════════════════════════════════
# 2. MoE HBM 显存估算
# ═══════════════════════════════════════════════════════════════════════════


def calculate_moe_hbm(
    num_dense_layers: int,           # L_dense: Dense FFN 层数
    num_moe_layers: int,             # L_moe: MoE 层数
    pipeline_parallel: int,          # PP: 流水线并行度
    hidden_size: int,                # H: 隐藏层维度
    ffn_hidden_size: int,            # F_dense: Dense FFN 中间维度
    tensor_parallel: int,            # TP: 张量并行度
    expert_ffn_hidden_size: int,     # F_expert: 专家 FFN 中间维度
    num_experts: int,                # E: 每 MoE 层的专家数
    expert_parallel: int,            # EP: 专家并行度
    vocab_size: int,                 # V: 词表大小
    expert_tensor_parallel: int = 1, # TP_e: 专家内部张量并行度
    bytes_per_param: int = 18        # K: 混合精度训练下每参数占用字节数
) -> float:
    """计算混合精度 (BF16 + Adam) 训练 MoE 大模型单卡 HBM 显存需求。

    公式由三项组成：

    * **Dense 层**：``(L_dense/PP) · ((4H² + 3H·F_dense)/TP + 2H)``
    * **MoE 层**：``(L_moe/PP) · (4H²/TP + 3E·H·F_expert/(EP·TP_e) + E·H/TP + 2H)``
    * **Embedding 层**：``V·H/TP``

    返回：
        float: 模型总显存需求，单位为 **字节 (Bytes)**。

    说明：
        ``bytes_per_param`` 默认 18 对应 BF16 训练 (参数2B + FP32主副本4B +
        梯度4B + Adam动量4B + Adam方差4B = 18B/param)。
    """
    # Dense 层
    term_dense = (num_dense_layers / pipeline_parallel) * (
        (4 * hidden_size**2 + 3 * hidden_size * ffn_hidden_size) / tensor_parallel
        + 2 * hidden_size
    )

    # MoE 层 (含 router 参数 E·H/TP)
    term_moe = (num_moe_layers / pipeline_parallel) * (
        (4 * hidden_size**2 / tensor_parallel)
        + (3 * num_experts * hidden_size * expert_ffn_hidden_size)
          / (expert_parallel * expert_tensor_parallel)
        + (num_experts * hidden_size / tensor_parallel)
        + 2 * hidden_size
    )

    # Embedding 层
    term_embedding = (vocab_size * hidden_size) / tensor_parallel

    total_bytes = (term_dense + term_moe + term_embedding) * bytes_per_param
    return total_bytes


# ═══════════════════════════════════════════════════════════════════════════
# 3. MoE EP 通信流量估算
# ═══════════════════════════════════════════════════════════════════════════


def calculate_moe_ep_traffic(
    topk: float,                    # K_top: Top-k 激活专家数
    global_batch_size: float,       # B: 全局批量大小 (注意：非 micro-batch)
    seq_len: float,                 # S: 序列长度
    hidden_size: float,             # H: 隐藏层维度
    num_moe_layers: float,          # L_moe: MoE 层数 (仅 MoE 层产生 EP 通信)
    pipeline_parallel: float        # PP: 流水线并行度
) -> float:
    """计算 MoE 模型中 EP (专家并行) 通信流量。

    公式：
        ``EP = 8 · K · B · S · H · L_moe / PP``

    常数 8 的含义：
        BF16 每元素 2 字节 × 2(前向+反向) × 2(dispatch+combine) = 8

    注意：
        *B* 是 **全局** 批量大小 (``--global-batch-size``)，不是微批次。
        *L_moe* 仅计 MoE 层 — Dense 层不产生 EP 通信。

    返回：
        float: EP 通信流量，单位为 **字节**。

    异常：
        ValueError: 当 pipeline_parallel 为 0 时抛出。
    """
    if pipeline_parallel == 0:
        raise ValueError("pipeline_parallel 不能为 0")

    ep_value = (
        8.0 * topk * global_batch_size * seq_len * hidden_size * num_moe_layers
    ) / pipeline_parallel
    return ep_value
