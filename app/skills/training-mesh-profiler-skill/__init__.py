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


def _estimate_flops_old(
    L: int, H: int, S: int, B: int, dp: int, tp: int, pp: int
) -> float:
    """FLOPs = [(72*B*S*H^2 + 12*B*S^2*H) / TP] * L/PP"""
    return ((72 * B * S * H**2 + 12 * B * S**2 * H) / tp) * L / pp


def _estimate_flops(
    L: int, H: int, S: int, B: int, dff: int, dp: int, tp: int, pp: int
) -> float:
    """FLOPs = (6*B*S*L*H/(DP*PP*TP)) * (4*H + 3*dff + 2*S)"""
    return (6 * B * S * L * H / (dp * pp * tp)) * (4 * H + 3 * dff + 2 * S)


def _estimate_hbm_gb_old(
    L: int, H: int, S: int, B: int, dp: int, tp: int, pp: int, a: float = 1
) -> float:
    """HBM = [L*(12*H^2+4H)/(TP*PP) + B*S*H*L/PP + L*(12*H^2+4H)/(TP*PP)] * a / 1e9"""
    param_term = L * (12 * H**2 + 4 * H)
    term1 = param_term / (tp * pp)
    term2 = B * S * H * L / pp
    term3 = param_term / (tp * pp)
    return (term1 + term2 + term3) * a / 1e9


def _estimate_hbm_gb(L: int, H: int, dff: int, tp: int, pp: int) -> float:
    """HBM = L/PP * ((4*H^2 + 3*H*dff)/TP + 2*H) / 1e9"""
    return L / pp * ((4 * H**2 + 3 * H * dff) / tp + 2 * H) / 1e9


def _estimate_dp_comm_gb_old(L: int, H: int, tp: int, pp: int) -> float:
    """DP comm = 8*L*(4H^2+3H*4)/(TP*PP) / 1e9"""
    return 8 * L * (4 * H**2 + 3 * H * 4 * H) / (tp * pp) / 1e9


def _estimate_dp_comm_gb_old_lh(L: int, H: int, dp: int) -> float:
    """DP comm = 2*(DP-1)/DP * 12*L*H^2 / 1e9"""
    if dp <= 1:
        return 0.0
    return 2 * (dp - 1) / dp * 12 * L * H**2 / 1e9


def _estimate_dp_comm_gb(L: int, H: int, dff: int, dp: int, tp: int, pp: int) -> float:
    """DP comm = 2*(DP-1)/DP * 4 * L/PP * (4*H^2/TP + 3*H*dff/TP) / 1e9"""
    if dp <= 1:
        return 0.0
    return 2 * (dp - 1) / dp * 4 * L / pp * (4 * H**2 / tp + 3 * H * dff / tp) / 1e9


def _estimate_tp_comm_gb(L: int, H: int, S: int, B: int, pp: int) -> float:
    """TP comm = L/PP * 15 * B * S * H / 1e9  -> GB/micro-step"""
    return L / pp * 15 * B * S * H / 1e9


def _estimate_pp_comm_mb(H: int, S: int, B: int) -> float:
    """PP comm = 4*B*S*H / 1e6  → MB/micro-step"""
    return 4 * B * S * H / 1e6


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
            dff_val = int(
                arguments.get("d_ffn")
                or (training_model.config.d_ffn if training_model else None)
                or 14336
            )
            a = float(arguments.get("quant_coeff", _QUANT_COEFF))

            flops = _estimate_flops(L, H, S, B, dff_val, dp, tp, pp)
            hbm = _estimate_hbm_gb(L, H, dff_val, tp, pp)
            dp_comm = _estimate_dp_comm_gb(L, H, dff_val, dp, tp, pp)
            tp_comm = _estimate_tp_comm_gb(L, H, S, B, pp)
            pp_comm = _estimate_pp_comm_mb(H, S, B)
            for rank in range(total_nodes):
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
