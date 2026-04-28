from app.config import config
from app.models.schemas import DeviceType, GuardrailResult, MeshTopology, TopologyParams


def validate_input_params(device_type: str, dp: int, tp: int, pp: int) -> GuardrailResult:
    """Input guardrail: validate device type, DP, TP, PP parameters."""
    errors = []
    warnings = []

    # Validate device type
    if device_type.upper() not in config.VALID_DEVICE_TYPES:
        errors.append(
            f"无效设备类型 '{device_type}'，支持的设备类型: {', '.join(sorted(config.VALID_DEVICE_TYPES))}"
        )

    # Validate DP
    if dp < config.DP_MIN or dp > config.DP_MAX:
        errors.append(f"DP值 {dp} 超出范围 [{config.DP_MIN}, {config.DP_MAX}]")
    elif dp > 512:
        warnings.append(f"DP值 {dp} 较大，仿真可能耗时较长")

    # Validate TP
    if tp < config.TP_MIN or tp > config.TP_MAX:
        errors.append(f"TP值 {tp} 超出范围 [{config.TP_MIN}, {config.TP_MAX}]")
    elif tp not in (1, 2, 4, 8, 16, 32):
        warnings.append(f"TP值 {tp} 非2的幂次，建议使用 (1,2,4,8,16,32)")

    # Validate PP
    if pp < config.PP_MIN or pp > config.PP_MAX:
        errors.append(f"PP值 {pp} 超出范围 [{config.PP_MIN}, {config.PP_MAX}]")

    # Validate total world size
    world_size = dp * tp * pp
    if world_size < 1 or world_size > 100000:
        warnings.append(f"总节点数 {world_size} 异常，请确认参数")
    if world_size > 10000:
        warnings.append(f"总节点数 {world_size} 较大，仿真可能耗时较长")

    return GuardrailResult(
        passed=len(errors) == 0,
        errors=errors,
        warnings=warnings
    )


def validate_topology_output(topology: MeshTopology) -> GuardrailResult:
    """Output guardrail: validate generated JSON topology."""
    errors = []
    warnings = []

    # Check required fields
    if not topology.name:
        errors.append("组网名称为空")
    if topology.total_nodes <= 0:
        errors.append("总节点数 <= 0")
    if not topology.nodes:
        errors.append("节点列表为空")

    # Check node count matches
    if len(topology.nodes) != topology.total_nodes:
        errors.append(
            f"节点列表长度 {len(topology.nodes)} 与 total_nodes {topology.total_nodes} 不一致"
        )

    # Check node count matches dp*tp*pp
    expected_total = topology.dp_size * topology.tp_size * topology.pp_size
    if topology.total_nodes != expected_total:
        errors.append(
            f"节点总数 {topology.total_nodes} 与 dp*tp*pp={expected_total} 不一致"
        )

    # Check communication groups
    for key, expected_dim in [("dp", topology.dp_size), ("tp", topology.tp_size), ("pp", topology.pp_size)]:
        groups = topology.communication_groups.get(key, [])
        if not groups:
            warnings.append(f"通信域 '{key}' 为空")
        else:
            for group in groups:
                if len(group) != expected_dim:
                    warnings.append(
                        f"通信域 '{key}' 分组大小 {len(group)} 与期望 {expected_dim} 不一致"
                    )

    return GuardrailResult(
        passed=len(errors) == 0,
        errors=errors,
        warnings=warnings
    )
