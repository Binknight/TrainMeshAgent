"""方案 B：由 topology 生成 MindSpeed 风格临时 .sh，供 run.py 解析。"""

from pathlib import Path
from typing import Any, Optional

from mcp_server.schemas.common import SimulationTaskInput

DEFAULT_MICRO_BATCH_SIZE = 4
DEFAULT_VOCAB_SIZE = 18277
DEFAULT_LEARNING_RATE = "1.5e-4"
DEFAULT_OPTIMIZER = "AdamW"


def _cluster_layout(total_nodes: int, npu_per_node: int = 8) -> tuple[int, int]:
    """返回 (NPUS_PER_NODE, NNODES)，保证 NPUS_PER_NODE * NNODES == total_nodes。"""
    if total_nodes <= npu_per_node:
        return total_nodes, 1
    nnodes = (total_nodes + npu_per_node - 1) // npu_per_node
    while total_nodes % nnodes != 0 and npu_per_node > 1:
        npu_per_node -= 1
        nnodes = (total_nodes + npu_per_node - 1) // npu_per_node
    if total_nodes % nnodes != 0:
        return total_nodes, 1
    return total_nodes // nnodes, nnodes


def resolve_batch_sizes(topology: SimulationTaskInput) -> tuple[int, int]:
    """
    解析写入脚本的 micro/global batch。

    - micro-batch-size 默认 4
    - 前端 batch_size 表示等效前（参考组网）的 global-batch-size
    - global-batch-size = batch_size * current_dp / reference_dp
      例如参考 DP8、当前 DP2、batch_size=32 → 32*2/8=8
    """
    micro_batch = topology.micro_batch_size or DEFAULT_MICRO_BATCH_SIZE
    micro_batch = max(1, micro_batch)

    reference_dp = topology.reference_dp_size
    if reference_dp is None:
        if topology.batch_size % micro_batch == 0:
            reference_dp = topology.batch_size // micro_batch
        else:
            reference_dp = topology.dp_size
    reference_dp = max(1, reference_dp)

    global_batch = max(1, topology.batch_size * topology.dp_size // reference_dp)
    return micro_batch, global_batch


def resolve_script_filename(topology_name: str) -> str:
    """根据组网名称生成建议下载文件名（§11）。"""
    if "原始" in topology_name:
        return "pretrain_orig.sh"
    if "等效" in topology_name:
        return "pretrain_equiv.sh"
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in topology_name)
    safe = safe.strip("_") or "pretrain"
    return f"pretrain_{safe}.sh"


def _resolve_vocab_size(simulation_params: Optional[dict[str, Any]]) -> int:
    if not simulation_params:
        return DEFAULT_VOCAB_SIZE
    raw = simulation_params.get("vocab_size")
    if raw is None or str(raw).strip() == "":
        return DEFAULT_VOCAB_SIZE
    return int(raw)


def _overlap_optimizer_args(dp_size: int) -> list[str]:
    """overlap-grad-reduce / overlap-param-gather 要求 DP > 1。

    返回「行」列表（每行自带续行符）而非拼好的字符串：调用方按行拼接，
    避免片段之间漏掉换行导致相邻参数被挤到同一行（曾把 MoE 的
    ``--expert-model-parallel-size`` 粘到 ``--overlap-param-gather`` 上）。
    """
    if dp_size <= 1:
        return []
    return [
        "    --overlap-grad-reduce \\",
        "    --overlap-param-gather \\",
    ]


def resolve_moe_layer_freq(topology: SimulationTaskInput) -> str:
    """MoE 层分布表达式（§11 解析契约与 GPT_ARGS 共用同一取值）。

    - 调用方显式给出 ``moe_layer_freq`` 时原样使用：可表达交替 MoE 层
      （如 ``([0,1]*24)``）这类仅靠层数无法还原的分布
    - 否则由 ``num_moe_layers`` 推导：前 ``L-L_moe`` 层 Dense、其余 MoE，
      与内置模型目录中 ``[0]*3+[1]*58`` 的写法语义一致
    - 全 MoE 时用 ``1``（与 MindSpeed 的 ``-1`` 哨兵等价）

    两种来源都保证与 ``num_layers`` / ``num_moe_layers`` 自洽——旧实现硬编码
    ``MOE_LAYER_FREQ=1``（= 全层 MoE），与同一脚本里的 ``NUM_MOE_LAYERS=58`` 直接矛盾。
    """
    explicit = topology.moe_layer_freq
    if explicit is not None and str(explicit).strip():
        return str(explicit).strip()
    num_layers = topology.num_layers
    num_moe = min(topology.num_moe_layers or 0, num_layers)
    num_dense = max(0, num_layers - num_moe)
    if num_dense <= 0:
        return "1"
    return f"[0]*{num_dense}+[1]*{num_moe}"


def _resolve_grad_accum_steps(
    global_batch: int,
    micro_batch: int,
    dp_size: int,
) -> int:
    denom = micro_batch * dp_size
    if denom <= 0 or global_batch % denom != 0:
        return 1
    steps = global_batch // denom
    return max(1, steps)


def generate_topology_script(
    topology: SimulationTaskInput,
    output_path: Path,
    simulation_params: Optional[dict[str, Any]] = None,
) -> Path:
    """
    根据 topology 写入临时训练脚本。

    run.py 会去掉 torchrun 行，并从 GPT_ARGS 中解析并行与模型参数。
    """
    expected = topology.dp_size * topology.tp_size * topology.pp_size
    if topology.total_nodes != expected:
        raise ValueError(
            f"total_nodes ({topology.total_nodes}) != "
            f"dp*tp*pp ({expected})"
        )

    npus_per_node, nnodes = _cluster_layout(topology.total_nodes)
    micro_batch, global_batch = resolve_batch_sizes(topology)
    ffn_hidden = topology.d_ffn
    vocab_size = _resolve_vocab_size(simulation_params)
    grad_accum = _resolve_grad_accum_steps(global_batch, micro_batch, topology.dp_size)
    max_pos = max(topology.seq_len, 4096)

    is_moe = topology.model_type == "sparse"
    ep = topology.ep or 1
    expert_tp = topology.expert_tensor_parallel_size or 1
    shared_expert_size = topology.shared_expert_intermediate_size
    moe_key_lines: list[str] = []
    moe_arg_lines: list[str] = []
    if is_moe:
        moe_layer_freq = resolve_moe_layer_freq(topology)
        # §11 解析契约：脚本须暴露 MoE 键，供前端三张对比卡解析
        moe_key_lines = [
            f"NUM_EXPERTS={topology.num_experts}",
            f"MOE_ROUTER_TOPK={topology.moe_router_topk}",
            f"NUM_MOE_LAYERS={topology.num_moe_layers}",
            f"MOE_LAYER_FREQ={moe_layer_freq}",
            f"MOE_FFN_HIDDEN_SIZE={topology.moe_ffn_hidden_size}",
            f"HAS_SHARED_EXPERT={'true' if topology.has_shared_expert else 'false'}",
            f"EXPERT_TP={expert_tp}",
        ]
        if shared_expert_size:
            moe_key_lines.append(
                f"SHARED_EXPERT_INTERMEDIATE_SIZE={shared_expert_size}"
            )
        # GPT_ARGS 内参数供 run.py get_arg_value 读取并执行 MoE 语义
        moe_arg_lines = [
            "    --expert-model-parallel-size ${EP} \\",
            f"    --num-experts {topology.num_experts} \\",
            f"    --moe-router-topk {topology.moe_router_topk} \\",
            f"    --moe-layer-freq {moe_layer_freq} \\",
            f"    --moe-ffn-hidden-size {topology.moe_ffn_hidden_size} \\",
            f"    --expert-tensor-parallel-size {expert_tp} \\",
        ]
        if topology.has_shared_expert:
            moe_arg_lines.append("    --n-shared-experts 1 \\")
            if shared_expert_size:
                moe_arg_lines.append(
                    f"    --moe-shared-expert-intermediate-size {shared_expert_size} \\"
                )
    moe_keys = ("\n" + "\n".join(moe_key_lines)) if moe_key_lines else ""

    gpt_arg_lines = [
        "    --use-mcore-models \\",
        "    --tensor-model-parallel-size ${TP} \\",
        "    --pipeline-model-parallel-size ${PP} \\",
        f"    --num-layers {topology.num_layers} \\",
        f"    --hidden-size {topology.hidden_dim} \\",
        f"    --ffn-hidden-size {ffn_hidden} \\",
        f"    --num-attention-heads {topology.num_heads} \\",
        f"    --seq-length {topology.seq_len} \\",
        f"    --max-position-embeddings {max_pos} \\",
        f"    --micro-batch-size {micro_batch} \\",
        f"    --global-batch-size {global_batch} \\",
        "    --make-vocab-size-divisible-by 1 \\",
        "    --bf16 \\",
        *_overlap_optimizer_args(topology.dp_size),
        *moe_arg_lines,
        "    --use-flash-attn \\",
        "    --swiglu \\",
        "    --normalization RMSNorm \\",
        "    --use-fused-rmsnorm \\",
        "    --position-embedding-type rope",
    ]
    gpt_args = "\n".join(gpt_arg_lines)

    content = f"""#!/bin/bash
# Auto-generated by AICM MCP Server
# Topology: {topology.name} | device={topology.device_type}

export CUDA_DEVICE_MAX_CONNECTIONS=1

# --- 组网参数（equivalent-modeling-service 解析契约 §11）---
DEVICE_TYPE={topology.device_type}
DP={topology.dp_size}
TP={topology.tp_size}
PP={topology.pp_size}
EP={ep}

# --- 模型结构参数 ---
NUM_LAYERS={topology.num_layers}
D_MODEL={topology.hidden_dim}
NUM_HEADS={topology.num_heads}
D_FFN={ffn_hidden}
VOCAB_SIZE={vocab_size}{moe_keys}

# --- 训练运行时参数 ---
GLOBAL_BATCH_SIZE={global_batch}
MICRO_BATCH_SIZE={micro_batch}
SEQ_LENGTH={topology.seq_len}
LEARNING_RATE={DEFAULT_LEARNING_RATE}
OPTIMIZER={DEFAULT_OPTIMIZER}
GRAD_ACCUM_STEPS={grad_accum}

NPUS_PER_NODE={npus_per_node}
NNODES={nnodes}
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=6000
WORLD_SIZE=$(($NPUS_PER_NODE * $NNODES))

GPT_ARGS="
{gpt_args}
"

# 占位：run.py 会过滤 torchrun 行
torchrun --nproc_per_node 1 pretrain_gpt.py \\
    $GPT_ARGS \\
    --distributed-backend nccl
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    output_path.chmod(0o755)
    return output_path
