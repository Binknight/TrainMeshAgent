"""
training-mesh-gen-skill: Generate structured JSON mesh topology from DP/TP/PP params.
"""
from app.skills.base import BaseSkill, SkillContext, SkillResult
from app.models.schemas import DeviceType, GuardrailResult, MeshNode, MeshTopology
from app.agent.guardrails import validate_input_params, validate_topology_output


def _build_communication_groups(dp: int, tp: int, pp: int) -> dict[str, list[list[int]]]:
    groups: dict[str, list[list[int]]] = {"dp": [], "tp": [], "pp": []}
    total = dp * tp * pp

    for tp_r in range(tp):
        for pp_r in range(pp):
            group = []
            for dp_r in range(dp):
                rank = dp_r * tp * pp + tp_r * pp + pp_r
                group.append(rank)
            groups["dp"].append(group)

    for dp_r in range(dp):
        for pp_r in range(pp):
            group = []
            for tp_r in range(tp):
                rank = dp_r * tp * pp + tp_r * pp + pp_r
                group.append(rank)
            groups["tp"].append(group)

    for dp_r in range(dp):
        for tp_r in range(tp):
            group = []
            for pp_r in range(pp):
                rank = dp_r * tp * pp + tp_r * pp + pp_r
                group.append(rank)
            groups["pp"].append(group)

    return groups


class MeshGenSkill(BaseSkill):
    name = "training-mesh-gen-skill"
    description = (
        "根据设备类型(A2/A3/A5)和并行参数(DP/TP/PP)生成结构化JSON组网拓扑图。"
        "当用户需要生成组网、创建网络拓扑、构建训练集群拓扑时触发。"
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
                        "name": {
                            "type": "string",
                            "description": "组网名称，如 '原始组网' 或 '等效组网'",
                        },
                        "device_type": {
                            "type": "string",
                            "description": "设备类型: A2, A3, A5",
                        },
                        "dp": {"type": "integer", "description": "数据并行度"},
                        "tp": {"type": "integer", "description": "张量并行度"},
                        "pp": {"type": "integer", "description": "流水线并行度"},
                    },
                    "required": ["name", "device_type", "dp", "tp", "pp"],
                },
            },
        }

    def input_guard(self, arguments: dict) -> GuardrailResult:
        return validate_input_params(
            device_type=arguments.get("device_type", ""),
            dp=arguments.get("dp", 0),
            tp=arguments.get("tp", 0),
            pp=arguments.get("pp", 0),
        )

    def output_guard(self, result) -> GuardrailResult:
        if not isinstance(result, MeshTopology):
            return GuardrailResult(passed=False, errors=["Output is not a MeshTopology"])
        return validate_topology_output(result)

    def execute(self, arguments: dict, context: SkillContext) -> SkillResult:
        device_type_str = arguments["device_type"].upper()
        dp = int(arguments["dp"])
        tp = int(arguments["tp"])
        pp = int(arguments["pp"])

        total_nodes = dp * tp * pp
        nodes: list[MeshNode] = []

        for global_rank in range(total_nodes):
            dp_rank = global_rank // (tp * pp)
            remainder = global_rank % (tp * pp)
            tp_rank = remainder // pp
            pp_rank = remainder % pp

            neighbors = []
            if dp > 1:
                neighbors.append(f"node_{global_rank}_{dp_rank}_{tp_rank}_{pp_rank}_dp")
            if tp > 1:
                neighbors.append(f"node_{global_rank}_{dp_rank}_{tp_rank}_{pp_rank}_tp")
            if pp > 1:
                neighbors.append(f"node_{global_rank}_{dp_rank}_{tp_rank}_{pp_rank}_pp")

            nodes.append(MeshNode(
                id=f"node_{global_rank}",
                device_type=DeviceType(device_type_str),
                dp_rank=dp_rank,
                tp_rank=tp_rank,
                pp_rank=pp_rank,
                global_rank=global_rank,
                neighbors=neighbors,
            ))

        topology = MeshTopology(
            name=arguments["name"],
            device_type=DeviceType(device_type_str),
            total_nodes=total_nodes,
            dp_size=dp,
            tp_size=tp,
            pp_size=pp,
            nodes=nodes,
            communication_groups=_build_communication_groups(dp, tp, pp),
        )

        return SkillResult(success=True, data=topology)
