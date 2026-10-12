"""MoE 在 Agent 侧的字段透传与脚本解析契约 —— 不连数据库、不起服务。

对应此前断链的三个环节：

1. **模型配置 → `execute_task` 入参**：`moe_layer_freq` 连字段都没有、
   `shared_expert_intermediate_size` 只存在于模型目录，二者都到不了 MCP。
2. **离线兜底**：共享专家 FFN 维度从仓库内置目录补齐，
   且**不得触发网络请求**（`resolve_model_config` 在 PG 未命中时会走 HF 搜索）。
3. **§11 脚本解析键**：MoE 键过去不在 `_SCRIPT_FIELD_ALIASES` 里，
   即使 MCP 脚本暴露了 EP / Experts / MoE 层分布，三张对比卡也拿不到。

Run: python tests/test_moe_agent_plumbing.py
"""

import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

try:
    from app.models.schemas import (
        DeviceType,
        MeshTopology,
        TrainingModel,
        TrainingModelComputed,
        TrainingModelConfig,
    )
    from app.models.model_catalog import lookup_builtin_model_config
    from app.routes.session import _parse_script_params, _topo_with_model
    from mcp_server.schemas.common import SimulationTaskInput
    from mcp_server.services.script_generator import generate_topology_script
except ImportError as exc:  # pragma: no cover - 依赖缺失时跳过而非报错
    print(f"  [SKIP] 依赖不完整（{exc}）—— 请先执行:")
    print("         pip install -r requirements.txt -r mcp_server/requirements.txt")
    sys.exit(0)

_failures: list[str] = []


def check(label: str, cond: bool, detail: str = "") -> None:
    print(("  [PASS] " if cond else "  [FAIL] ") + label + ("" if cond else f"   {detail}"))
    if not cond:
        _failures.append(label)


def make_topology(name: str = "原始组网", dp: int = 8, tp: int = 16, pp: int = 8) -> MeshTopology:
    return MeshTopology(
        name=name, device_type=DeviceType.A3,
        total_nodes=dp * tp * pp, dp_size=dp, tp_size=tp, pp_size=pp, nodes=[],
    )


def make_model(cfg: TrainingModelConfig, model_name: str | None = None) -> TrainingModel:
    return TrainingModel(
        model_name=model_name,
        config=cfg,
        computed=TrainingModelComputed(d_head=cfg.d_model // cfg.num_heads, total_params_billions="~1.0"),
        layers=[],
    )


MOE_CFG = dict(
    num_layers=64, d_model=4096, num_heads=32, d_ffn=18432, vocab_size=18277,
    model_type="sparse", num_experts=256, moe_router_topk=8, num_moe_layers=58,
    moe_ffn_hidden_size=2048, has_shared_expert=True,
    expert_tensor_parallel_size=1,
)

print("=" * 74)
print("  MoE Agent 侧透传与解析契约")
print("=" * 74)

# ── 1. 离线模型目录查询（含 MoE 表，无网络）──
print("\n[1] 离线模型目录兜底")
hints = lookup_builtin_model_config("DeepSeek-R1")
check(
    "内置目录命中 DeepSeek-R1 并给出共享专家维度",
    bool(hints) and hints.get("shared_expert_intermediate_size") == 2048,
    str(hints),
)
check(
    "支持 org/repo 形式（退回 basename）",
    (lookup_builtin_model_config("deepseek-ai/DeepSeek-R1") or {}).get(
        "shared_expert_intermediate_size"
    ) == 2048,
    "",
)
check("未命中返回 None（不抛错）", lookup_builtin_model_config("some-unknown-model-xyz") is None, "")
check("空串返回 None", lookup_builtin_model_config("") is None, "")

# ── 2. 配置 → execute_task 入参 ──
print("\n[2] TrainingModelConfig -> execute_task 入参")
topo = make_topology()
payload = _topo_with_model(
    topo, make_model(TrainingModelConfig(**MOE_CFG), "DeepSeek-R1"),
    model_name="DeepSeek-R1", ep=8, seq_len=2048, batch_size=32,
)
for key, want in (
    ("model_type", "sparse"), ("num_experts", 256), ("moe_router_topk", 8),
    ("num_moe_layers", 58), ("moe_ffn_hidden_size", 2048), ("has_shared_expert", True),
    ("expert_tensor_parallel_size", 1), ("ep", 8),
):
    check(f"入参含 {key}={want!r}", payload.get(key) == want, str(payload.get(key)))
check(
    "共享专家维度由内置目录补齐（配置里为 None）",
    payload.get("shared_expert_intermediate_size") == 2048,
    str(payload.get("shared_expert_intermediate_size")),
)

with_freq = _topo_with_model(
    topo,
    make_model(
        TrainingModelConfig(**{**MOE_CFG, "moe_layer_freq": "([0,1]*24)"}),
        "Llama-4-Maverick-17B-128E",
    ),
    model_name="Llama-4-Maverick-17B-128E", ep=8,
)
check(
    "moe_layer_freq 透传到 MCP 入参",
    with_freq.get("moe_layer_freq") == "([0,1]*24)",
    str(with_freq.get("moe_layer_freq")),
)

dense_payload = _topo_with_model(
    make_topology(),
    make_model(
        TrainingModelConfig(
            num_layers=64, d_model=4096, num_heads=32, d_ffn=14336, vocab_size=32000
        ),
        "Llama-3.1-8B",
    ),
    model_name="Llama-3.1-8B",
)
# 决定仿真语义的字段必须缺席，MCP 才会按稠密模型处理。
# 注：`expert_tensor_parallel_size` 因 TrainingModelConfig 默认值为 1 而由旧逻辑一并透传，
# MCP 侧仅在 sparse 分支使用它，dense 任务不受影响（既有行为，本次不动）。
check(
    "dense 入参不带决定 MoE 语义的字段",
    not any(
        k in dense_payload
        for k in ("model_type", "num_experts", "moe_router_topk", "num_moe_layers",
                  "moe_ffn_hidden_size", "moe_layer_freq", "shared_expert_intermediate_size")
    )
    and dense_payload.get("has_shared_expert") is not True,
    str(sorted(dense_payload)),
)

# ── 3. §11 脚本解析键（MoE）──
print("\n[3] §11 脚本解析（MCP 脚本 -> 对比卡参数）")
work = REPO / ".tmp" / "moe_agent_plumbing"
shutil.rmtree(work, ignore_errors=True)
work.mkdir(parents=True, exist_ok=True)
script_path = generate_topology_script(
    SimulationTaskInput(
        name="原始组网", device_type="A3",
        dp_size=8, tp_size=16, pp_size=8, total_nodes=1024,
        num_layers=64, hidden_dim=4096, num_heads=32, d_ffn=18432,
        model_type="sparse", ep=8, num_experts=256, moe_router_topk=8,
        num_moe_layers=58, moe_ffn_hidden_size=2048,
        has_shared_expert=True, shared_expert_intermediate_size=2048,
    ),
    work / "topology_generated.sh",
)
script = script_path.read_text(encoding="utf-8")
parsed = _parse_script_params(script)
for group, key, want in (
    ("topology", "ep", "8"),
    ("model", "num_experts", "256"),
    ("model", "moe_router_topk", "8"),
    ("model", "num_moe_layers", "58"),
    ("model", "moe_layer_freq", "[0]*6+[1]*58"),
    ("model", "moe_ffn_hidden_size", "2048"),
    ("model", "has_shared_expert", "true"),
    ("model", "expert_tensor_parallel_size", "1"),
    ("model", "shared_expert_intermediate_size", "2048"),
):
    got = (parsed.get(group) or {}).get(key)
    check(f"解析出 {group}.{key}={want}", got == want, f"got {got!r}")
check(
    "dense 脚本仍解析出组网参数（未回归）",
    (_parse_script_params(
        generate_topology_script(
            SimulationTaskInput(
                name="原始组网", device_type="A3",
                dp_size=8, tp_size=16, pp_size=8, total_nodes=1024,
                num_layers=64, hidden_dim=4096, num_heads=32,
            ),
            work / "dense.sh",
        ).read_text(encoding="utf-8")
    ).get("topology") or {}).get("dp") == "8",
    "",
)

print()
if _failures:
    print(f"  [FAIL] {len(_failures)} 项未通过:")
    for f in _failures:
        print(f"    - {f}")
    sys.exit(1)
print("  [PASS] MoE Agent 侧透传与解析契约全部通过。")
