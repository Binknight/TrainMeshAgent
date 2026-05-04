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
_QUANT_COEFF = 1


def _estimate_flops(L: int, H: int, S: int, B: int, dp: int, tp: int, pp: int) -> float:
    """FLOPs = [(72*B*S*H^2 + 12*B*S^2*H) / (DP*TP)] * L/PP"""
    return ((72 * B * S * H**2 + 12 * B * S**2 * H) / (dp * tp)) * L / pp


def _estimate_hbm_gb(
    L: int, H: int, S: int, B: int, dp: int, tp: int, pp: int, a: float = 1
) -> float:
    """HBM = [L*(12*H^2+4H)/(TP*PP) + B*S*H*L/PP + L*(12*H^2+4H)/(DP*TP*PP)] * a / 1e9"""
    param_term = L * (12 * H**2 + 4 * H)
    term1 = param_term / (tp * pp)
    term2 = B * S * H * L / pp
    term3 = param_term / (dp * tp * pp)
    return (term1 + term2 + term3) * a / 1e9


def _estimate_tp_comm_gb(L: int, H: int, dp: int) -> float:
    """TP comm = 2*(DP-1)/DP * 12*L*H^2 / 1e9"""
    if dp <= 1:
        return 0.0
    return 2 * (dp - 1) / dp * 12 * L * H**2 / 1e9


def _estimate_pp_comm_mb(L: int, H: int, S: int, B: int, tp: int) -> float:
    """PP comm = 8*(TP-1)/TP * B*S*H*L / 1e6"""
    if tp <= 1:
        return 0.0
    return 8 * (tp - 1) / tp * B * S * H * L / 1e6


def _estimate_dp_comm_gb(H: int, S: int, B: int) -> float:
    """DP comm = 4*B*S*H / 1e9"""
    return 4 * B * S * H / 1e9


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
            for detail in card_details:
                cards.append(
                    CardMetrics(
                        card_id=detail.get("card_id", ""),
                        global_rank=detail.get("global_rank", 0),
                        flops_per_card=detail.get("flops_per_card", 0),
                        hbm_gb=detail.get("hbm_gb", 0),
                        tp_comm_gb_per_micro=detail.get("tp_comm_gb_per_micro", 0),
                        pp_comm_mb_per_micro=detail.get("pp_comm_mb_per_micro", 0),
                        dp_comm_gb_per_step=detail.get("dp_comm_gb_per_step", 0),
                    )
                )

        if not cards:
            cfg = _MODEL_CONFIG[device_type]
            L = int(arguments.get("num_layers", cfg["num_layers"]))
            H = int(arguments.get("hidden_dim", cfg["hidden_dim"]))
            S = int(arguments.get("seq_len", _SEQ_LEN))
            B = int(arguments.get("total_batch", _TOTAL_BATCH))
            a = float(arguments.get("quant_coeff", _QUANT_COEFF))

            flops = _estimate_flops(L, H, S, B, dp, tp, pp)
            hbm = _estimate_hbm_gb(L, H, S, B, dp, tp, pp, a)
            tp_comm = _estimate_tp_comm_gb(L, H, dp)
            pp_comm = _estimate_pp_comm_mb(L, H, S, B, tp)
            dp_comm = _estimate_dp_comm_gb(H, S, B)
            for rank in range(total_nodes):
                cards.append(
                    CardMetrics(
                        card_id=f"card_{rank}",
                        global_rank=rank,
                        flops_per_card=flops,
                        hbm_gb=hbm,
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
            aggregate_flops=sum(c.flops_per_card for c in cards),
            aggregate_hbm_gb=sum(c.hbm_gb for c in cards),
            aggregate_tp_comm_gb_per_micro=sum(c.tp_comm_gb_per_micro for c in cards),
            aggregate_pp_comm_mb_per_micro=sum(c.pp_comm_mb_per_micro for c in cards),
            aggregate_dp_comm_gb_per_step=sum(c.dp_comm_gb_per_step for c in cards),
        )

        return SkillResult(success=True, data=result)
