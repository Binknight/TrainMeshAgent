"""
training-mesh-profiler-skill: Fetch simulation results and compute per-card metrics.
"""
from app.skills.base import BaseSkill, SkillContext, SkillResult
from app.models.schemas import CardMetrics, DeviceType, GuardrailResult, SimulationResult
from app.mcp.client import MCPClient

# ── Reference model parameters per device type ──
_MODEL_CONFIG = {
    DeviceType.A2: {"model_params": 7e9, "hidden_dim": 4096, "num_layers": 32},
    DeviceType.A3: {"model_params": 20e9, "hidden_dim": 6144, "num_layers": 48},
    DeviceType.A5: {"model_params": 70e9, "hidden_dim": 8192, "num_layers": 64},
}

# ── Fixed simulation parameters ──
_SEQ_LEN = 4096
_MICRO_BATCH = 1
_BYTES_PER_PARAM = 2  # bf16


def _estimate_flops(device_type: DeviceType, dp: int, tp: int, pp: int) -> float:
    """FLOPs = 6 × model_params × seq_len × micro_batch / (dp × tp × pp)"""
    cfg = _MODEL_CONFIG[device_type]
    return 6 * cfg["model_params"] * _SEQ_LEN * _MICRO_BATCH / (dp * tp * pp)


def _estimate_hbm_gb(device_type: DeviceType, dp: int, tp: int, pp: int) -> float:
    """HBM = (params×2/tp + 2×params×2/dp + hidden×seq×micro×2×layers/pp) / 1e9"""
    cfg = _MODEL_CONFIG[device_type]
    params = cfg["model_params"]
    # ZeRO-2: optimizer states sharded across DP, params+grads kept. TP shards weights.
    param_mem = params * _BYTES_PER_PARAM / tp
    optim_mem = 2 * params * _BYTES_PER_PARAM / dp  # optimizer states
    act_mem = cfg["hidden_dim"] * _SEQ_LEN * _MICRO_BATCH * _BYTES_PER_PARAM * cfg["num_layers"] / pp
    return (param_mem + optim_mem + act_mem) / 1e9


def _estimate_tp_comm_gb(device_type: DeviceType, tp: int) -> float:
    """TP comm = num_layers × 2 × hidden_dim × seq_len × micro_batch × 2 × (tp-1)/tp / 1e9"""
    cfg = _MODEL_CONFIG[device_type]
    if tp <= 1:
        return 0.0
    return cfg["num_layers"] * 2 * cfg["hidden_dim"] * _SEQ_LEN * _MICRO_BATCH * _BYTES_PER_PARAM * (tp - 1) / tp / 1e9


def _estimate_pp_comm_mb(device_type: DeviceType, pp: int) -> float:
    """PP comm = 2 × (pp-1)/pp × hidden_dim × seq_len × micro_batch × 2 / 1e6"""
    cfg = _MODEL_CONFIG[device_type]
    if pp <= 1:
        return 0.0
    return 2 * (pp - 1) / pp * cfg["hidden_dim"] * _SEQ_LEN * _MICRO_BATCH * _BYTES_PER_PARAM / 1e6


def _estimate_dp_comm_gb(device_type: DeviceType, dp: int) -> float:
    """DP comm = 2 × model_params × 2 × (dp-1)/dp / 1e9"""
    cfg = _MODEL_CONFIG[device_type]
    if dp <= 1:
        return 0.0
    return 2 * cfg["model_params"] * _BYTES_PER_PARAM * (dp - 1) / dp / 1e9


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
                    },
                    "required": ["topology_name", "device_type", "total_nodes", "dp", "tp", "pp"],
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
            for detail in card_details:
                cards.append(CardMetrics(
                    card_id=detail.get("card_id", ""),
                    global_rank=detail.get("global_rank", 0),
                    flops_per_card=detail.get("flops_per_card", 0),
                    hbm_gb=detail.get("hbm_gb", 0),
                    tp_comm_gb_per_micro=detail.get("tp_comm_gb_per_micro", 0),
                    pp_comm_mb_per_micro=detail.get("pp_comm_mb_per_micro", 0),
                    dp_comm_gb_per_step=detail.get("dp_comm_gb_per_step", 0),
                ))
        else:
            flops = _estimate_flops(device_type, dp, tp, pp)
            hbm = _estimate_hbm_gb(device_type, dp, tp, pp)
            tp_comm = _estimate_tp_comm_gb(device_type, tp)
            pp_comm = _estimate_pp_comm_mb(device_type, pp)
            dp_comm = _estimate_dp_comm_gb(device_type, dp)
            for rank in range(total_nodes):
                cards.append(CardMetrics(
                    card_id=f"card_{rank}",
                    global_rank=rank,
                    flops_per_card=flops,
                    hbm_gb=hbm,
                    tp_comm_gb_per_micro=tp_comm,
                    pp_comm_mb_per_micro=pp_comm,
                    dp_comm_gb_per_step=dp_comm,
                ))

        result = SimulationResult(
            topology_name=arguments["topology_name"],
            device_type=device_type,
            total_nodes=total_nodes,
            cards=cards,
            aggregate_flops=sum(c.flops_per_card for c in cards),
            aggregate_hbm_gb=sum(c.hbm_gb for c in cards),
            aggregate_tp_comm_gb_per_micro=sum(c.tp_comm_gb_per_micro for c in cards),
            aggregate_pp_comm_mb_per_micro=sum(c.pp_comm_mb_per_micro for c in cards),
            aggregate_dp_comm_gb_per_step=sum(c.dp_comm_gb_per_step for c in cards),
        )

        return SkillResult(success=True, data=result)
