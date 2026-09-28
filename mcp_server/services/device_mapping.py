"""TrainMeshAgent 设备枚举 → AICM ASCEND 设备类型。"""

from typing import Literal, Optional, Union

TopologyDevice = Literal["A2", "A3", "A5"]

# Agent 侧 A2/A3/A5 映射到 run.py / generate_megatron 的 device_type
TOPOLOGY_DEVICE_TO_ASCEND: dict[str, str] = {
    "A2": "ASCEND_910B2",
    "A3": "ASCEND_910B3",
    "A5": "ASCEND_910C",
}

_VALID_ASCEND_DEVICE_TYPES = frozenset(TOPOLOGY_DEVICE_TO_ASCEND.values()) | {
    "ASCEND_910B1",
}

# TrainMeshAgent / spec 中常见的泛称，需结合 topology 解析
_GENERIC_ASCEND_ALIASES = frozenset(
    {
        "ASCEND_910B",
        "ASCEND_910",
        "ASCEND",
    }
)


def resolve_ascend_device_type(
    topology_device: TopologyDevice,
    override: Optional[str] = None,
) -> str:
    """
    解析最终传给 run.py 的 --device_type。

    - simulation_params.device_type 若为合法 ASCEND_910B1/B2/B3/C 则原样使用
    - ASCEND_910B 等泛称按 topology.device_type 查表
    - 否则按 topology.device_type 查表
    """
    topology_default = TOPOLOGY_DEVICE_TO_ASCEND[topology_device]

    if override:
        upper = override.strip().upper()
        if upper in _VALID_ASCEND_DEVICE_TYPES:
            return upper
        if upper in _GENERIC_ASCEND_ALIASES:
            return topology_default
        if upper in TOPOLOGY_DEVICE_TO_ASCEND:
            return TOPOLOGY_DEVICE_TO_ASCEND[upper]
        # 允许直接传 910B3 等简写
        if upper in ("910B1", "910B2", "910B3", "910C"):
            return f"ASCEND_{upper}"
        if upper.startswith("ASCEND_"):
            # 未识别的 ASCEND_* 回退到 topology 映射，避免非法枚举传入 workload
            return topology_default
    return topology_default
