"""
training-mesh-profiler-skill: Fetch simulation results and compute per-card metrics.
"""

from app.mcp.client import MCPClient
from app.models.schemas import (
    CardMetrics,
    DeviceType,
    GuardrailResult,
    SimulationResult,
)
from app.skills.base import BaseSkill, SkillContext, SkillResult

# ── Reference model parameters per device type ──
_MODEL_CONFIG = {
    DeviceType.A2: {"hidden_dim": 4096, "num_layers": 32},
    DeviceType.A3: {"hidden_dim": 6144, "num_layers": 48},
    DeviceType.A5: {"hidden_dim": 8192, "num_layers": 64},
}

# ── Default estimation parameters ──
_SEQ_LEN = 2048
_TOTAL_BATCH = 32
_MICRO_BATCH = 4
_QUANT_COEFF = 1


def _estimate_flops(
    L: int, H: int, S: int, B: int, dff: int, dp: int, tp: int, pp: int
) -> float:
    """FLOPs = (6*B*S*L*H/(DP*PP*TP)) * (4*H + 3*dff + 2*S)"""
    return (6 * B * S * L * H / (dp * pp * tp)) * (4 * H + 3 * dff + 2 * S)


def _estimate_flops_first_last(
    L: int, H: int, S: int, B: int, dff: int, dp: int, tp: int, pp: int, V: int
) -> float:
    """首末PP rank的FLOPs = 中间计算值 + (6 * V * H) / TP * (B / DP) * S

    首/末PP stage 额外包含 embedding / lm_head 的前向+反向计算量。
    """
    middle = _estimate_flops(L, H, S, B, dff, dp, tp, pp)
    return middle + (6 * V * H) / tp * (B / dp) * S


def _estimate_hbm_gb(L: int, H: int, dff: int, tp: int, pp: int) -> float:
    """HBM = K * L/PP * ((4*H^2 + 3*H*dff)/TP + 2*H) / 1e9"""
    return 18 * L / pp * ((4 * H**2 + 3 * H * dff) / tp + 2 * H) / 1e9


def _estimate_hbm_gb_first_last(
    L: int, H: int, dff: int, tp: int, pp: int, V: int
) -> float:
    """首末PP rank的HBM = 中间计算值 + (V * H) / (TP * 1e9)

    首/末PP stage 额外持有 embedding / lm_head 权重，大小为 V*H，
    按 TP 切分后每个 rank 持有 V*H/TP 个元素，转换为 GB。
    """
    middle = _estimate_hbm_gb(L, H, dff, tp, pp)
    return middle + V * H / (tp * 1e9)


_DEFAULT_VOCAB_SIZE = 32000


def _estimate_dp_comm_gb(L: int, H: int, dff: int, dp: int, tp: int, pp: int) -> float:
    """DP comm = 2*(DP-1)/DP * 4 * L/PP * (4*H^2/TP + 3*H*dff/TP) / 1e9"""
    if dp <= 1:
        return 0.0
    return 2 * (dp - 1) / dp * 4 * L / pp * (4 * H**2 / tp + 3 * H * dff / tp) / 1e9


def _estimate_tp_comm_gb(L: int, H: int, S: int, b: int, pp: int) -> float:
    """TP comm = L/PP * 15 * b * S * H / 1e9  -> GB/micro-step"""
    return L / pp * 15 * b * S * H / 1e9


def _estimate_pp_comm_mb(H: int, S: int, b: int) -> float:
    """PP comm = 4*b*S*H / 1e6  → MB/micro-step"""
    return 4 * b * S * H / 1e6


class MeshProfilerSkill(BaseSkill):
    name = "training-mesh-profiler-skill"
    description = (
        "通过MCP获取仿真结果，计算单卡FLOPs、HBM占用(GB)、TP通信(GB/micro)、PP通信(MB/micro)、DP通信(GB/step)等性能指标。"
        "当需要分析组网性能、对比仿真结果、验证等效性时触发。"
    )

    @property
    def tool_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task_id": {
                            "type": "string",
                            "description": "仿真任务 ID (MCP)，不提供则使用估算模式",
                        },
                        "topology_name": {
                            "type": "string",
                            "description": "组网名称",
                        },
                        "device_type": {
                            "type": "string",
                            "description": "设备类型: A2, A3, A5",
                        },
                        "total_nodes": {
                            "type": "integer",
                            "description": "总节点数",
                        },
                        "dp": {"type": "integer", "description": "数据并行度"},
                        "tp": {"type": "integer", "description": "张量并行度"},
                        "pp": {"type": "integer", "description": "流水线并行度"},
                        "seq_len": {
                            "type": "integer",
                            "description": f"序列长度，默认 {_SEQ_LEN}",
                        },
                        "total_batch": {
                            "type": "integer",
                            "description": f"总批次大小，默认 {_TOTAL_BATCH}",
                        },
                        "quant_coeff": {
                            "type": "number",
                            "description": f"量化系数，默认 {_QUANT_COEFF}",
                        },
                        "num_layers": {
                            "type": "integer",
                            "description": "模型层数，默认使用设备类型对应的配置值",
                        },
                        "hidden_dim": {
                            "type": "integer",
                            "description": "隐藏维度，默认使用设备类型对应的配置值",
                        },
                        "d_ffn": {
                            "type": "integer",
                            "description": "FFN 隐藏层维度，默认 14336",
                        },
                        "micro_batch": {
                            "type": "integer",
                            "description": f"微批次大小 b，默认 {_MICRO_BATCH}",
                        },
                        "model_type": {
                            "type": "string",
                            "description": "模型类型: 'dense' 或 'sparse', 默认从 session 推断",
                        },
                        "ep": {
                            "type": "integer",
                            "description": "MoE: 专家并行度 (Expert Parallel)",
                        },
                        "num_experts": {
                            "type": "integer",
                            "description": "MoE: 每层专家数",
                        },
                        "moe_router_topk": {
                            "type": "integer",
                            "description": "MoE: Top-K 激活专家数",
                        },
                        "num_moe_layers": {
                            "type": "integer",
                            "description": "MoE: MoE 层数",
                        },
                        "moe_ffn_hidden_size": {
                            "type": "integer",
                            "description": "MoE: 专家 FFN 隐藏维度",
                        },
                        "has_shared_expert": {
                            "type": "boolean",
                            "description": "MoE: 是否有共享专家",
                        },
                        "expert_tensor_parallel_size": {
                            "type": "integer",
                            "description": "MoE: 专家内部 TP 大小",
                        },
                    },
                    "required": [
                        "topology_name",
                        "device_type",
                        "total_nodes",
                        "dp",
                        "tp",
                        "pp",
                    ],
                },
            },
        }

    def input_guard(self, arguments: dict) -> GuardrailResult:
        errors = []
        if arguments.get("total_nodes", 0) <= 0:
            errors.append("total_nodes 必须 > 0")
        device_type = arguments.get("device_type", "").upper()
        if device_type not in {"A2", "A3", "A5"}:
            errors.append(f"无效设备类型: {device_type}")
        return GuardrailResult(passed=len(errors) == 0, errors=errors)

    def execute(self, arguments: dict, context: SkillContext) -> SkillResult:
        device_type = DeviceType(arguments["device_type"].upper())
        task_id = arguments.get("task_id")
        total_nodes = int(arguments["total_nodes"])
        dp = int(arguments["dp"])
        tp = int(arguments["tp"])
        pp = int(arguments["pp"])

        cards: list[CardMetrics] = []
        mcp: MCPClient | None = context.mcp_client

        if task_id and mcp:
            card_details = mcp.get_card_details(task_id)
            # ── Enrich hbm_model_gb from hbm_detail ──
            hbm_model_map: dict[int, float] = {}
            try:
                for detail in card_details:
                    rank = detail.get("global_rank", 0)
                    hbd = mcp.get_hbm_detail(task_id, rank)
                    w = float(hbd.get("weights_gb", 0))
                    g = float(hbd.get("gradients_gb", 0))
                    o = float(hbd.get("optimizer_gb", 0))
                    model_gb = w + g + o
                    if model_gb > 0:
                        hbm_model_map[rank] = model_gb
            except Exception:
                pass  # fall back to hbm_gb
            for detail in card_details:
                rank = detail.get("global_rank", 0)
                hbm_model_gb = hbm_model_map.get(rank, detail.get("hbm_gb", 0))
                cards.append(
                    CardMetrics(
                        card_id=detail.get("card_id", ""),
                        global_rank=rank,
                        flops_per_card=detail.get("flops_per_card", 0),
                        hbm_gb=detail.get("hbm_gb", 0),
                        hbm_model_gb=hbm_model_gb,
                        tp_comm_gb_per_micro=detail.get("tp_comm_gb_per_micro", 0),
                        pp_comm_mb_per_micro=detail.get("pp_comm_mb_per_micro", 0),
                        dp_comm_gb_per_step=detail.get("dp_comm_gb_per_step", 0),
                    )
                )

        if not cards:
            cfg = _MODEL_CONFIG[device_type]

            # Try to get L/H from training model in session as fallback
            training_model = None
            session = getattr(context, "session", None)
            if session:
                name = arguments.get("topology_name", "")
                if "原始" in name:
                    training_model = getattr(session, "original_training_model", None)
                elif "等效" in name:
                    training_model = getattr(session, "equivalent_training_model", None)

            L = int(
                arguments.get("num_layers")
                or (training_model.config.num_layers if training_model else None)
                or cfg["num_layers"]
            )
            H = int(
                arguments.get("hidden_dim")
                or (training_model.config.d_model if training_model else None)
                or cfg["hidden_dim"]
            )
            S = int(arguments.get("seq_len", _SEQ_LEN))
            B = int(arguments.get("total_batch", _TOTAL_BATCH))
            b_micro = int(arguments.get("micro_batch", _MICRO_BATCH))
            dff_val = int(
                arguments.get("d_ffn")
                or (training_model.config.d_ffn if training_model else None)
                or 14336
            )
            a = float(arguments.get("quant_coeff", _QUANT_COEFF))
            V = int(
                arguments.get("vocab_size")
                or (training_model.config.vocab_size if training_model and hasattr(training_model.config, "vocab_size") else None)
                or _DEFAULT_VOCAB_SIZE
            )
            # ── MoE model detection ──
            model_type = (
                arguments.get("model_type")
                or (training_model.config.model_type if training_model else None)
                or "dense"
            )
            # ── MoE parameters from arguments or session or training_model ──
            ep = arguments.get("ep") or (getattr(session, "original_ep", None) if session else None)
            num_experts = arguments.get("num_experts") or (
                getattr(session, "original_num_experts", None) if session else None
            ) or (training_model.config.num_experts if training_model and training_model.config.model_type == "sparse" else None)
            moe_topk = arguments.get("moe_router_topk") or (
                getattr(session, "original_moe_topk", None) if session else None
            ) or (training_model.config.moe_router_topk if training_model and training_model.config.model_type == "sparse" else None)
            n_moe_layers = arguments.get("num_moe_layers") or (
                getattr(session, "equivalent_num_moe_layers", None) if session and "等效" in arguments.get("topology_name", "") else None
            ) or (
                getattr(session, "original_num_moe_layers", None) if session else None
            ) or (training_model.config.num_moe_layers if training_model and training_model.config.model_type == "sparse" else None)
            moe_fexp = arguments.get("moe_ffn_hidden_size") or (
                getattr(session, "original_moe_ffn_hidden_size", None) if session else None
            ) or (training_model.config.moe_ffn_hidden_size if training_model and training_model.config.model_type == "sparse" else None)
            has_shared = arguments.get("has_shared_expert") or (
                getattr(session, "original_has_shared_expert", None) if session else False
            ) or (training_model.config.has_shared_expert if training_model and training_model.config.model_type == "sparse" else False)
            expert_tp = arguments.get("expert_tensor_parallel_size") or (
                getattr(session, "original_expert_tensor_parallel_size", None) if session else 1
            ) or (training_model.config.expert_tensor_parallel_size if training_model and training_model.config.model_type == "sparse" else 1)

            if model_type == "sparse" and num_experts and n_moe_layers:
                # ── MoE estimation ──
                from .moe_estimator import compute_moe_flops, calculate_moe_hbm

                ep_val = ep or dp
                topk_val = moe_topk or 1
                n_dense = L - n_moe_layers
                fexp = moe_fexp or dff_val
                n_shared = n_moe_layers if has_shared else 0

                flops_mid = compute_moe_flops(
                    micro_batch_size=b_micro, seq_len=S,
                    num_layers=L, hidden_size=H,
                    tensor_parallel=tp, num_moe_layers=n_moe_layers,
                    expert_ffn_hidden_size=fexp,
                    expert_parallel=ep_val, topk=topk_val,
                    num_shared_expert_layers=n_shared,
                    pipeline_parallel=pp,
                )
                hbm_bytes = calculate_moe_hbm(
                    num_dense_layers=n_dense, num_moe_layers=n_moe_layers,
                    pipeline_parallel=pp, hidden_size=H,
                    ffn_hidden_size=dff_val, tensor_parallel=tp,
                    expert_ffn_hidden_size=fexp,
                    num_experts=num_experts, expert_parallel=ep_val,
                    vocab_size=V, expert_tensor_parallel=expert_tp or 1,
                )
                hbm_mid = hbm_bytes / 1e9
                flops_edge = flops_mid
                hbm_edge = hbm_mid
            else:
                flops_mid = _estimate_flops(L, H, S, B, dff_val, dp, tp, pp)
                flops_edge = _estimate_flops_first_last(L, H, S, B, dff_val, dp, tp, pp, V)
                hbm_mid = _estimate_hbm_gb(L, H, dff_val, tp, pp)
                hbm_edge = _estimate_hbm_gb_first_last(L, H, dff_val, tp, pp, V)
            dp_comm = _estimate_dp_comm_gb(L, H, dff_val, dp, tp, pp)
            tp_comm = _estimate_tp_comm_gb(L, H, S, b_micro, pp)
            pp_comm = _estimate_pp_comm_mb(H, S, b_micro)
            for rank in range(total_nodes):
                # 与前端 meshBuildData 一致：global_rank = dp*(tp*pp)+pp*tp+tp
                # 故 pp_idx = (rank // tp) % pp（TP 最低位）；旧用 rank%pp 会与前端 PP 分组错位
                pp_rank = (rank // tp) % pp
                is_edge = pp > 1 and (pp_rank == 0 or pp_rank == pp - 1)
                flops = flops_edge if is_edge else flops_mid
                hbm = hbm_edge if is_edge else hbm_mid
                cards.append(
                    CardMetrics(
                        card_id=f"card_{rank}",
                        global_rank=rank,
                        flops_per_card=flops,
                        hbm_gb=hbm,
                        hbm_model_gb=hbm,
                        tp_comm_gb_per_micro=tp_comm,
                        pp_comm_mb_per_micro=pp_comm,
                        dp_comm_gb_per_step=dp_comm,
                    )
                )

        result = SimulationResult(
            topology_name=arguments["topology_name"],
            device_type=device_type,
            total_nodes=total_nodes,
            cards=cards,
        )

        return SkillResult(success=True, data=result)
