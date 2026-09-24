"""
Rank-layout regression test.

The mesh rank numbering must match the simulation system's ordering:
**TP-DP-PP** (TP innermost / fastest-varying, then DP, then PP), i.e.

    global_rank = pp_rank * (tp * dp) + dp_rank * tp + tp_rank

This test pins the layout contract and proves the mesh-gen skill, the node
tables and the per-PP-stage metric grouping all agree with it.

Run: python tests/test_rank_layout.py
"""

import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Windows consoles default to a legacy code page; the report below is UTF-8.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from app.rank_layout import RANK_ORDER, decompose, pp_rank_of, rank_of  # noqa: E402

_failures: list[str] = []


def check(label: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  [PASS] {label}")
    else:
        print(f"  [FAIL] {label} {detail}")
        _failures.append(label)


def old_pp_rank(global_rank: int, tp: int, pp: int) -> int:
    """The retired TP-PP-DP convention, kept here only to quantify the change."""
    return (global_rank // tp) % pp


# (name, dp, tp, pp)
CASES = [
    ("A3 原始组网  DP=8  TP=16 PP=8", 8, 16, 8),
    ("A3 等效组网  DP=2  TP=16 PP=3", 2, 16, 3),
    ("DP=1 退化", 1, 16, 4),
    ("TP=1 退化", 4, 1, 3),
    ("PP=1 退化", 4, 8, 1),
]

print("=" * 76)
print(f"  Rank layout contract: {RANK_ORDER}  (global_rank = pp*(tp*dp) + dp*tp + tp)")
print("=" * 76)

for name, dp, tp, pp in CASES:
    total = dp * tp * pp
    print(f"\n{'-' * 70}\n  {name}  共 {total} 卡\n{'-' * 70}")

    # ── 1. Decomposition is a bijection onto the (dp, tp, pp) index cube ──
    seen: set[tuple[int, int, int]] = set()
    roundtrip_ok = True
    range_ok = True
    for g in range(total):
        d, t, p = decompose(g, dp, tp, pp)
        if not (0 <= d < dp and 0 <= t < tp and 0 <= p < pp):
            range_ok = False
        if rank_of(dp, tp, dp_rank=d, tp_rank=t, pp_rank=p) != g:
            roundtrip_ok = False
        seen.add((d, t, p))
    check("rank -> (dp,tp,pp) 落在合法范围", range_ok)
    check("rank -> 索引 -> rank 往返一致", roundtrip_ok)
    check(f"索引组合覆盖全量 {dp}*{tp}*{pp}", len(seen) == total)

    # ── 2. Dimension strides follow TP fastest -> DP -> PP slowest ──
    stride_ok = True
    for g in range(total):
        d, t, p = decompose(g, dp, tp, pp)
        if tp > 1 and rank_of(dp, tp, dp_rank=d, tp_rank=(t + 1) % tp, pp_rank=p) != g - t + (t + 1) % tp:
            stride_ok = False
    check("TP 为最低位（步长 1，连续）", stride_ok)
    if dp > 1:
        check(
            "DP 步长 = tp",
            rank_of(dp, tp, dp_rank=1, tp_rank=0, pp_rank=0) == tp,
        )
    if pp > 1:
        check(
            "PP 步长 = tp*dp（最外层）",
            rank_of(dp, tp, dp_rank=0, tp_rank=0, pp_rank=1) == tp * dp,
        )

    # ── 3. PP stage of a rank is a contiguous block of tp*dp ranks ──
    block_ok = all(
        pp_rank_of(g, dp, tp) == g // (tp * dp) == min(g // (tp * dp), pp - 1)
        for g in range(total)
    )
    check("PP 段号 = rank // (tp*dp)（首/中/末段连续分块）", block_ok)

    first = [g for g in range(total) if pp_rank_of(g, dp, tp) == 0]
    last = [g for g in range(total) if pp_rank_of(g, dp, tp) == pp - 1]
    check(
        "首 PP 段 = rank 0..tp*dp-1",
        first == list(range(tp * dp)) if pp > 1 else True,
    )
    check(
        "末 PP 段 = 最高一段 rank 区间",
        last == list(range((pp - 1) * tp * dp, total)) if pp > 1 else True,
    )

    # ── 4. Quantify the change vs the retired TP-PP-DP layout ──
    mismatched = [g for g in range(total) if old_pp_rank(g, tp, pp) != pp_rank_of(g, dp, tp)]
    print(f"        PP 段号与旧布局(TP-PP-DP)不一致的 rank 数: {len(mismatched)}/{total}")
    if pp > 1 and dp > 1:
        check(
            "新旧布局在 PP 段归属上确实存在差异（本次校准的原因）",
            len(mismatched) > 0,
            f"mismatched={len(mismatched)}",
        )

# ── 5. Mesh topology nodes agree with the layout ──
print(f"\n{'=' * 76}\n  MeshTopology 节点 / 通信组一致性\n{'=' * 76}")

mesh_gen = importlib.import_module("app.skills.training-mesh-gen-skill")
for name, dp, tp, pp in CASES:
    topo = mesh_gen.MeshGenSkill().execute(
        {"name": "测试组网", "device_type": "A3", "dp": dp, "tp": tp, "pp": pp},
        None,
    ).data

    nodes_ok = all(
        (n.dp_rank, n.tp_rank, n.pp_rank) == decompose(n.global_rank, dp, tp, pp)
        for n in topo.nodes
    )
    check(f"{name}: 节点 dp/tp/pp 与 rank 布局一致", nodes_ok)

    groups = topo.communication_groups
    check(f"{name}: 通信组数量 dp*tp+dp*pp+tp*pp",
          len(groups["dp"]) == tp * pp and len(groups["tp"]) == dp * pp and len(groups["pp"]) == dp * tp)

    def _group_ok(kind: str) -> bool:
        for grp in groups[kind]:
            if len(grp) != {"dp": dp, "tp": tp, "pp": pp}[kind]:
                return False
            if len(set(grp)) != len(grp) or min(grp) < 0 or max(grp) >= dp * tp * pp:
                return False
            idx = [decompose(g, dp, tp, pp) for g in grp]
            for fixed in range(3):
                if fixed != {"dp": 0, "tp": 1, "pp": 2}[kind]:
                    if len({i[fixed] for i in idx}) != 1:
                        return False
            if len({i[{"dp": 0, "tp": 1, "pp": 2}[kind]] for i in idx}) != len(grp):
                return False
        return True

    check(f"{name}: dp/tp/pp 通信组划分与布局一致", all(_group_ok(k) for k in ("dp", "tp", "pp")))

print()
if _failures:
    print(f"  [FAIL] {len(_failures)} 项未通过: {_failures}")
    sys.exit(1)
print("  [PASS] rank 布局(TP-DP-PP)与仿真系统一致，且全部消费方一致。")
