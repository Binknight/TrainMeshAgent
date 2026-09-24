"""
PP-stage equivalence grouping test.

Reproduces the reported symptom: with the retired TP-PP-DP rank layout the
per-PP-stage comparison grouped the wrong ranks, so PP metrics looked
non-equivalent although the two topologies behave identically.

The simulation feeds one ``pp_comm_mb_per_micro`` value per ``global_rank``,
and that value depends only on the pipeline stage the rank belongs to. The
comparison therefore has to reconstruct the *same* stage groups as the
simulation. This test builds "ground truth" cards straight from the canonical
layout and checks that:

  1. the current comparison groups ranks correctly (all stage diffs ≈ 0), and
  2. the retired formula would indeed have reported a >5% PP mismatch.

Run: python tests/test_pp_compare_grouping.py
"""

import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from app.models.schemas import CardMetrics, DeviceType, SimulationResult  # noqa: E402
from app.rank_layout import pp_rank_of  # noqa: E402

_compare = importlib.import_module("app.routes.session")

TOLERANCE = 5.0
_failures: list[str] = []


def check(label: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  [PASS] {label}")
    else:
        print(f"  [FAIL] {label} {detail}")
        _failures.append(label)


def stage_value(stage: int, pp: int) -> float:
    """Ground truth PP communication of a pipeline stage (MB/micro-step).

    Middle stages both send and receive activations (2 boundaries), the first
    and last stages only have one boundary — identical in both topologies.
    """
    if pp <= 1:
        return 12.0
    return 12.0 if stage in (0, pp - 1) else 24.0


def make_cards(label: str, dp: int, tp: int, pp: int) -> SimulationResult:
    cards = []
    for g in range(dp * tp * pp):
        cards.append(CardMetrics(
            card_id=f"card_{g}",
            global_rank=g,
            flops_per_card=1.0,
            hbm_gb=1.0,
            hbm_model_gb=1.0,
            tp_comm_gb_per_micro=1.0,
            pp_comm_mb_per_micro=stage_value(pp_rank_of(g, dp, tp), pp),
            dp_comm_gb_per_step=1.0,
        ))
    return SimulationResult(
        topology_name=label, device_type=DeviceType.A3, total_nodes=dp * tp * pp, cards=cards,
    )


def old_stage_averages(cards: list[CardMetrics], tp: int, pp: int) -> dict[str, float]:
    """Per-stage averages under the retired TP-PP-DP grouping, for comparison."""
    groups: dict[str, list[float]] = {"first": [], "middle": [], "last": []}
    stride = tp * pp
    for c in cards:
        pp_rank = (c.global_rank % stride) // tp
        if pp_rank == 0:
            groups["first"].append(c.pp_comm_mb_per_micro)
        if pp > 1 and pp_rank == pp - 1:
            groups["last"].append(c.pp_comm_mb_per_micro)
        if 0 < pp_rank < pp - 1:
            groups["middle"].append(c.pp_comm_mb_per_micro)
    return {k: (sum(v) / len(v)) if v else 0.0 for k, v in groups.items()}


ORIG = (8, 16, 8)   # DP, TP, PP -> A3 原始组网 1024 卡
EQ = (2, 16, 3)     # DP, TP, PP -> A3 等效组网 96 卡

print("=" * 76)
print(f"  PP 等效性分组校验  原始 DP{ORIG[0]} TP{ORIG[1]} PP{ORIG[2]}"
      f"  vs  等效 DP{EQ[0]} TP{EQ[1]} PP{EQ[2]}")
print("=" * 76)

orig_sim = make_cards("原始组网", *ORIG)
eq_sim = make_cards("等效组网", *EQ)

report = _compare._build_comparison(
    orig_sim, eq_sim, ORIG[0], ORIG[1], ORIG[2], EQ[0], EQ[1], EQ[2],
)
breakdown = report.details["pp_breakdown"]

print("\n  ── 校准后（TP-DP-PP，与仿真系统一致）──")
for stage in ("first", "middle", "last"):
    print(f"    {stage:<7} 原始={breakdown['original'][stage]:8.3f}  "
          f"等效={breakdown['equivalent'][stage]:8.3f}  "
          f"diff={breakdown['diff_pct'][stage]:6.2f}%")

for stage, diff in breakdown["diff_pct"].items():
    check(f"{stage} PP 段通信等效 (diff={diff:.2f}% <= {TOLERANCE}%)", diff <= TOLERANCE)
check("整体等效判定通过", report.is_equivalent)
check("PP 通信差值 ≈ 0", report.pp_comm_diff_pct <= TOLERANCE)

# ── Counter-proof: the retired grouping would have failed ──
print("\n  ── 反证：旧布局(TP-PP-DP)分组 ──")
old_orig = old_stage_averages(orig_sim.cards, ORIG[1], ORIG[2])
old_eq = old_stage_averages(eq_sim.cards, EQ[1], EQ[2])
old_diffs = {}
for stage in ("first", "middle", "last"):
    d = abs(old_orig[stage] - old_eq[stage]) / max(abs(old_orig[stage]), 1e-9) * 100
    old_diffs[stage] = d
    print(f"    {stage:<7} 原始={old_orig[stage]:8.3f}  "
          f"等效={old_eq[stage]:8.3f}  diff={d:6.2f}%")
worst = max(old_diffs.values())
check(
    f"旧分组确实存在 PP 段错配 (max diff={worst:.2f}% > {TOLERANCE}%)",
    worst > TOLERANCE,
)

print()
if _failures:
    print(f"  [FAIL] {len(_failures)} 项未通过: {_failures}")
    sys.exit(1)
print("  [PASS] PP 段分组已与仿真系统对齐，等效性判定恢复正确。")
