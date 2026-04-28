"""
training-mesh-profiler-skill: Fetch simulation results and compute per-card metrics.
"""
from app.skills.base import BaseSkill, SkillContext, SkillResult
from app.models.schemas import CardMetrics, DeviceType, GuardrailResult, SimulationResult
from app.mcp.client import MCPClient


def _compute_card_intensity(device_type: DeviceType) -> float:
    base = {
        DeviceType.A2: 312.0,
        DeviceType.A3: 624.0,
        DeviceType.A5: 1248.0,
    }
    return base.get(device_type, 100.0)


def _estimate_memory_per_card(device_type: DeviceType, total_nodes: int) -> float:
    base_mem = {
        DeviceType.A2: 40.0,
        DeviceType.A3: 80.0,
        DeviceType.A5: 80.0,
    }
    mem = base_mem.get(device_type, 40.0)
    return round(mem * (1.0 + 1.0 / max(total_nodes, 1)), 2)


def _estimate_communication_per_card(device_type: DeviceType, dp: int, tp: int, pp: int) -> float:
    base_bw = {
        DeviceType.A2: 200.0,
        DeviceType.A3: 400.0,
        DeviceType.A5: 800.0,
    }
    bw = base_bw.get(device_type, 200.0)
    scale = 1.0 + 0.1 * (dp - 1) + 0.2 * (tp - 1) + 0.3 * (pp - 1)
    return round(bw * scale, 2)


class MeshProfilerSkill(BaseSkill):
    name = "training-mesh-profiler-skill"
    description = (
        "通过MCP获取仿真结果，计算单卡计算强度(TFLOPS)、内存占用(GB)、通信流量(GB/s)等性能指标。"
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
                    compute_intensity_tflops=detail.get("compute_intensity_tflops", 0),
                    memory_usage_gb=detail.get("memory_usage_gb", 0),
                    communication_traffic_gbps=detail.get("communication_traffic_gbps", 0),
                    peak_memory_gb=detail.get("peak_memory_gb", 0),
                    communication_volume_gb=detail.get("communication_volume_gb", 0),
                ))
        else:
            for rank in range(total_nodes):
                cards.append(CardMetrics(
                    card_id=f"card_{rank}",
                    global_rank=rank,
                    compute_intensity_tflops=_compute_card_intensity(device_type),
                    memory_usage_gb=_estimate_memory_per_card(device_type, total_nodes),
                    communication_traffic_gbps=_estimate_communication_per_card(device_type, dp, tp, pp),
                    peak_memory_gb=_estimate_memory_per_card(device_type, total_nodes) + 10,
                    communication_volume_gb=_estimate_communication_per_card(device_type, dp, tp, pp) * 0.01,
                ))

        result = SimulationResult(
            topology_name=arguments["topology_name"],
            device_type=device_type,
            total_nodes=total_nodes,
            cards=cards,
            aggregate_compute_intensity_tflops=sum(c.compute_intensity_tflops for c in cards),
            aggregate_memory_usage_gb=sum(c.memory_usage_gb for c in cards),
            aggregate_communication_traffic_gbps=sum(c.communication_traffic_gbps for c in cards),
        )

        return SkillResult(success=True, data=result)
