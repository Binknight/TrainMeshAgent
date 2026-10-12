"""仿真会话兜底自愈（app.routes.session._fallback_reconcile_simulation）回归测试。

背景：`run_simulation()` 只在「提交时就拿到 card_detail」或「前端收到 WebSocket
complete 后补打一次 run-simulation」时才推进 `session.step`。页面刷新会掐断
WebSocket，于是 MCP 侧任务早已 completed、会话却永远停在 `simulating`，前端一直
显示"仿真验证中"。修复后 REST 读取路径（GET /topology、GET /simulation）会调用
`_fallback_reconcile_simulation()` 做一次兜底判定。

覆盖：
  1. 两个任务 completed → 补齐结果 + 建对比 + step=completed
  2. 仍有任务 running → 不改动任何状态（仍 simulating）
  3. MCP 报 failed → step=failed 且不误建对比
  4. MCP 不可达（unavailable）→ 不改动状态（不能把网络问题当成任务失败）
  5. 无 task_id → 直接返回，不触碰 MCP
  6. step 已是 completed → 零额外开销（不查 MCP）

不触碰网络与数据库：`_build_simulation_result` 被替换为固定卡片数据，
`session_manager.save_session` 被替换为 no-op。

Run: python tests/test_simulation_reconcile.py
"""

import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# 必须在 import app.* 之前设好：app.main 导入时会执行 init_db()，
# 而默认 SQLITE_PATH 是容器路径 /home/data/db/...，在 Windows 上不存在。
_TMP_DB = REPO / ".tmp" / "reconcile_test"
_TMP_DB.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("SQLITE_PATH", str(_TMP_DB / "test.db"))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from app.models.schemas import (  # noqa: E402
    CardMetrics, DeviceType, MeshNode, MeshTopology, SessionState,
    SimulationResult,
)
from app.routes import session as sess_mod  # noqa: E402
from app.routes.session import _fallback_reconcile_simulation  # noqa: E402

_failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" — {detail}" if detail and not ok else ""))
    if not ok:
        _failures.append(label)


def _topology(name: str, dp: int, tp: int, pp: int) -> MeshTopology:
    nodes = [
        MeshNode(id=f"node_{g}", device_type=DeviceType.A3, dp_rank=0, tp_rank=0,
                 pp_rank=0, global_rank=g)
        for g in range(dp * tp * pp)
    ]
    return MeshTopology(
        name=name, device_type=DeviceType.A3,
        dp_size=dp, tp_size=tp, pp_size=pp, total_nodes=dp * tp * pp,
        nodes=nodes,
    )


def _cards(topology: MeshTopology, flops: float, hbm: float) -> list[CardMetrics]:
    return [
        CardMetrics(card_id=f"card_{n.global_rank}", global_rank=n.global_rank,
                    flops_per_card=flops, hbm_gb=hbm, hbm_model_gb=hbm,
                    tp_comm_gb_per_micro=1.0, pp_comm_mb_per_micro=2.0,
                    dp_comm_gb_per_step=3.0, ep_comm_gb_per_step=0.0)
        for n in topology.nodes
    ]


def _make_session(*, step="simulating", orig_task="sim_orig_1", eq_task="sim_eq_1") -> SessionState:
    # 两侧每卡指标完全一致 → 对比必然 is_equivalent=True（DP 不纳入判定）
    # 注意：兜底路径不需要 TrainingModel（它只从 MCP 取卡片指标，不复用拓扑
    # payload），因此这里不构造模型，测试也就不依赖模型 schema 的必填字段。
    orig_topo = _topology("原始组网", 2, 2, 2)
    eq_topo = _topology("等效组网", 1, 2, 2)
    state = SessionState(session_id="testsess")
    state.step = step
    state.original_task_id = orig_task
    state.equivalent_task_id = eq_task
    state.original_topology = orig_topo
    state.equivalent_topology = eq_topo
    state.original_model_name = "TestModel"
    return state


_ORIG_TOPO = _topology("原始组网", 2, 2, 2)
_EQ_TOPO = _topology("等效组网", 1, 2, 2)
_FAKE_RESULTS = {
    "original": SimulationResult(
        topology_name="原始组网", device_type=DeviceType.A3, total_nodes=8,
        cards=_cards(_ORIG_TOPO, flops=100.0, hbm=64.0),
    ),
    "equivalent": SimulationResult(
        topology_name="等效组网", device_type=DeviceType.A3, total_nodes=4,
        cards=_cards(_EQ_TOPO, flops=100.0, hbm=64.0),
    ),
}


def _install_fakes(status_map: dict, calls: dict) -> None:
    """把 MCP 查询与取卡逻辑替换为可控的假实现。"""
    def _fake_status(task_id: str) -> dict:
        calls["status"].append(task_id)
        return status_map.get(task_id, {"status": "unavailable"})

    def _fake_build(topo, training_model, task_id, label, **kwargs):
        calls["build"].append(label)
        return _FAKE_RESULTS[label]

    sess_mod.mcp_client.get_task_status = _fake_status
    sess_mod._build_simulation_result = _fake_build
    sess_mod.session_manager.save_session = lambda s: calls["saved"].append(s.session_id)


def _reset_calls() -> dict:
    return {"status": [], "build": [], "saved": []}


def case_all_completed() -> None:
    print("[1] 两个任务 completed → 补齐结果并推进到 completed")
    calls = _reset_calls()
    _install_fakes({"sim_orig_1": {"status": "completed", "progress": 100},
                    "sim_eq_1": {"status": "completed", "progress": 100}}, calls)
    state = _make_session()
    _fallback_reconcile_simulation(state)

    check("step 推进为 completed", state.step == "completed", state.step)
    check("原始组网结果已补齐", state.original_simulation is not None)
    check("等效组网结果已补齐", state.equivalent_simulation is not None)
    check("对比报告已生成", state.comparison_report is not None)
    check("等效判定为通过",
          bool(state.comparison_report and state.comparison_report.is_equivalent))
    check("确实落库了一次", calls["saved"] == ["testsess"], str(calls["saved"]))
    check("两侧取卡各调一次", sorted(calls["build"]) == ["equivalent", "original"], str(calls["build"]))


def case_still_running() -> None:
    print("[2] 仍有任务 running → 保持 simulating，不做任何落库")
    calls = _reset_calls()
    _install_fakes({"sim_orig_1": {"status": "completed", "progress": 100},
                    "sim_eq_1": {"status": "running", "progress": 40}}, calls)
    state = _make_session()
    _fallback_reconcile_simulation(state)

    check("step 仍为 simulating", state.step == "simulating", state.step)
    check("未生成对比报告", state.comparison_report is None)
    check("未落库", calls["saved"] == [], str(calls["saved"]))
    check("未提前取卡", calls["build"] == [], str(calls["build"]))


def case_failed() -> None:
    print("[3] MCP 报 failed → 会话落到 failed，且不误建对比")
    calls = _reset_calls()
    _install_fakes({"sim_orig_1": {"status": "failed", "message": "process exited 1"},
                    "sim_eq_1": {"status": "completed", "progress": 100}}, calls)
    state = _make_session()
    _fallback_reconcile_simulation(state)

    check("step 变为 failed", state.step == "failed", state.step)
    check("未生成对比报告", state.comparison_report is None)
    check("失败原因写入会话消息",
          any("process exited 1" in m.get("content", "") for m in state.history),
          str(state.history))
    check("失败状态已落库", calls["saved"] == ["testsess"], str(calls["saved"]))


def case_mcp_unreachable() -> None:
    print("[4] MCP 不可达（unavailable）→ 保持现状，不能误判为任务失败")
    calls = _reset_calls()
    _install_fakes({"sim_orig_1": {"error": "Connection refused", "status": "unavailable"},
                    "sim_eq_1": {"error": "Connection refused", "status": "unavailable"}}, calls)
    state = _make_session()
    _fallback_reconcile_simulation(state)

    check("step 仍为 simulating", state.step == "simulating", state.step)
    check("未落库", calls["saved"] == [], str(calls["saved"]))


def case_no_task_id() -> None:
    print("[5] 无 task_id → 不查 MCP，保持现状")
    calls = _reset_calls()
    _install_fakes({}, calls)
    state = _make_session(orig_task=None, eq_task=None)
    _fallback_reconcile_simulation(state)

    check("未调用 MCP 状态查询", calls["status"] == [], str(calls["status"]))
    check("step 仍为 simulating", state.step == "simulating", state.step)


def case_terminal_step() -> None:
    print("[6] step 已是 completed → 零额外开销（不查 MCP）")
    calls = _reset_calls()
    _install_fakes({"sim_orig_1": {"status": "completed"}}, calls)
    state = _make_session(step="completed")
    _fallback_reconcile_simulation(state)

    check("未调用 MCP 状态查询", calls["status"] == [], str(calls["status"]))
    check("未落库", calls["saved"] == [], str(calls["saved"]))


def case_endpoint_wiring() -> None:
    print("[7] REST 读取路径已接入兜底判定，且回传 task_id")
    import inspect
    src = inspect.getsource(sess_mod.get_simulation)
    check("get_simulation 调用兜底判定", "_fallback_reconcile_simulation" in src)
    check("get_simulation 回传 original_task_id", "original_task_id" in src)
    src_topo = inspect.getsource(sess_mod.get_topology)
    check("get_topology 调用兜底判定", "_fallback_reconcile_simulation" in src_topo)
    check("get_topology 回传 equivalent_task_id", "equivalent_task_id" in src_topo)


def main() -> int:
    print("== 仿真会话兜底自愈 ==")
    case_all_completed()
    case_still_running()
    case_failed()
    case_mcp_unreachable()
    case_no_task_id()
    case_terminal_step()
    case_endpoint_wiring()
    print()
    if _failures:
        print(f"FAILED: {len(_failures)} 项 -> {_failures}")
        return 1
    print("PASS: 全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
