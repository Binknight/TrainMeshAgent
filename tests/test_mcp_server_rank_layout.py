"""
MCP Server rank-layout regression test.

The simulation-system side (`mcp_server/`) must decompose `global_rank` with the
same ordering the Agent side uses — **TP-DP-PP**:

    global_rank = pp_rank * (tp * dp) + dp_rank * tp + tp_rank

`mcp_server/services/results_reader.py` used to carry the retired TP-PP-DP
formula (`pp_rank = (rank // tp) % pp`) — the very formula commit e739979
removed from the Agent side — so `get_device_detail` reported a different
(dp, tp, pp) triple than the layout the rest of the system assumes.

This test pins the server-side decomposition against `app.rank_layout` as the
single source of truth, and exercises the production path
`get_device_detail -> DeviceDetailOutput`.

Run: python tests/test_mcp_server_rank_layout.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Windows consoles default to a legacy code page; the report below is UTF-8.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from app.rank_layout import decompose, rank_of  # noqa: E402
from mcp_server.schemas.common import SimulationTaskInput  # noqa: E402
from mcp_server.services.results_reader import (  # noqa: E402
    _derive_parallel_ranks,
    get_device_detail,
)
from mcp_server.services.task_store import TaskRecord  # noqa: E402

_failures: list[str] = []


def check(label: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  [PASS] {label}")
    else:
        print(f"  [FAIL] {label} {detail}")
        _failures.append(label)


def old_parallel_ranks(global_rank: int, dp: int, tp: int, pp: int) -> tuple[int, int, int]:
    """The retired TP-PP-DP convention, kept here only to quantify the change."""
    return (
        global_rank // (tp * pp),
        global_rank % tp,
        (global_rank // tp) % pp,
    )


def make_topology(name: str, dp: int, tp: int, pp: int) -> SimulationTaskInput:
    return SimulationTaskInput(
        name=name,
        device_type="A3",
        dp_size=dp,
        tp_size=tp,
        pp_size=pp,
        total_nodes=dp * tp * pp,
        num_layers=64,
        hidden_dim=4096,
        num_heads=32,
    )


# (name, dp, tp, pp)
CASES = [
    ("A3 原始组网  DP=8  TP=16 PP=8", 8, 16, 8),
    ("A3 等效组网  DP=2  TP=16 PP=3", 2, 16, 3),
    ("DP=1 退化", 1, 16, 4),
    ("TP=1 退化", 4, 1, 3),
    ("PP=1 退化", 4, 8, 1),
]

print("=" * 76)
print("  MCP Server rank layout: TP-DP-PP  (global_rank = pp*(tp*dp) + dp*tp + tp)")
print("=" * 76)

for name, dp, tp, pp in CASES:
    total = dp * tp * pp
    topo = make_topology(name, dp, tp, pp)
    print(f"\n{'-' * 70}\n  {name}  共 {total} 卡\n{'-' * 70}")

    # ── 1. Server decomposition agrees with the canonical Agent implementation ──
    mismatch = [
        g
        for g in range(total)
        if _derive_parallel_ranks(g, topo) != decompose(g, dp, tp, pp)
    ]
    check(
        "服务端分解 == app.rank_layout.decompose",
        not mismatch,
        f"不一致 rank: {mismatch[:5]}",
    )

    # ── 2. Round-trip through the canonical composer ──
    roundtrip_ok = all(
        rank_of(
            dp,
            tp,
            dp_rank=_derive_parallel_ranks(g, topo)[0],
            tp_rank=_derive_parallel_ranks(g, topo)[1],
            pp_rank=_derive_parallel_ranks(g, topo)[2],
        )
        == g
        for g in range(total)
    )
    check("rank -> (dp,tp,pp) -> rank 往返一致", roundtrip_ok)

    # ── 3. TP is the fastest-varying dimension ──
    check(
        "tp_rank = rank % tp（TP 最低位）",
        all(_derive_parallel_ranks(g, topo)[1] == g % tp for g in range(total)),
    )

    # ── 4. Quantify the change vs the retired TP-PP-DP layout ──
    old_bad = [
        g
        for g in range(total)
        if old_parallel_ranks(g, dp, tp, pp) != _derive_parallel_ranks(g, topo)
    ]
    print(f"        分解结果与旧布局(TP-PP-DP)不一致的 rank 数: {len(old_bad)}/{total}")
    if dp > 1 and pp > 1:
        check(
            "新旧布局在 (dp,tp,pp) 归属上确实存在差异（本次修复的原因）",
            len(old_bad) > 0,
            f"mismatched={len(old_bad)}",
        )
    else:
        check(
            "dp==1 或 pp==1 时新旧布局退化为一致（无回归）",
            len(old_bad) == 0,
            f"mismatched={len(old_bad)}",
        )

# ── 5. Production path: get_device_detail carries the canonical triple ──
print(f"\n{'=' * 76}\n  get_device_detail 出参一致性（真实生产路径）\n{'=' * 76}")

for name, dp, tp, pp in CASES:
    total = dp * tp * pp
    record = TaskRecord(
        task_id="sim_test_rank_layout",
        topology=make_topology(name, dp, tp, pp),
        simulation_params={},
        results_dir=Path(__file__).resolve().parent / "_nonexistent_results_",
    )

    bad_rank = None
    for g in range(total):
        out = get_device_detail(record, g, 0, is_task_running=False)
        if (out.dp_rank, out.tp_rank, out.pp_rank) != decompose(g, dp, tp, pp):
            bad_rank = g
            break
    check(
        f"{name}: DeviceDetailOutput 的 dp/tp/pp_rank 与契约一致",
        bad_rank is None,
        f"首个不一致 rank={bad_rank}",
    )

print()
if _failures:
    print(f"  [FAIL] {len(_failures)} 项未通过: {_failures}")
    sys.exit(1)
print("  [PASS] MCP Server rank 分解与 TP-DP-PP 契约一致，且出参与 Agent 侧单一真相源相符。")
