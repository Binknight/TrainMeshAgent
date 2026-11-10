"""会话写路径并发保护回归测试（app/agent/session.py 长操作互斥 + 短写 session_lock）。

背景：会话是可变的共享对象，多个标签页/请求可能同时写。之前的并发控制只覆盖
仿真提交路径；Step1/Step2/Step3/对话这些「长操作」仍可能互相踩；参数/公式行等
短写也没有统一串行。

本测试锁住修复后的契约：
  1. 同一会话长操作（Step1/Step3/对话/仿真提交）互斥：第二个请求 409 task_busy，
     且不会产生副作用；
  2. 长操作结束后名额会释放（不会永久卡 busy）；
  3. 短写（仿真参数、公式行）走 session_lock 串行，两个并发写都成功且都落库；
  4. Step2 原子化：POST /workflow/step2 一次算完并落库（公式行 + 等效组网/模型 +
     step=equiv_generated）；GET /workflow/step2/stream 退化为纯回放，不占长操作
     名额、busy 时也能回放；两个任务可同时回放且互不污染（2026-11-09 两任务串台修复）；
  5. 删除会话在长操作进行中返回 409，结束后可以正常删除。

不依赖真实 MCP/仿真：临时 SQLite + Flask test client + monkeypatch MCP 客户端。

Run: python tests/test_session_write_locks.py
"""

import os
import sys
import json
import threading
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

_TMP_DB = REPO / ".tmp" / "session_write_locks_test"
_TMP_DB.mkdir(parents=True, exist_ok=True)
for _f in ("test.db", "test.db-wal", "test.db-shm"):
    try:
        (_TMP_DB / _f).unlink()
    except FileNotFoundError:
        pass
os.environ.setdefault("SQLITE_PATH", str(_TMP_DB / "test.db"))
os.environ.setdefault("DATABASE_URL", "")

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_BASE_PARAMS = {
    "device_type": "A3",
    "dp": 8,
    "tp": 16,
    "pp": 8,
    "model_name": "Qwen3-32B",
    "model_type": "dense",
    "L": 64,
    "H": 4096,
    "A": 32,
    "S": 2048,
    "B": 32,
    "b_micro": 4,
    "dff": 14336,
    "vocab_size": 32000,
    "strategy": "min_equiv",
}

_failures: list[str] = []


def check(label: str, cond: bool, detail: str = "") -> None:
    print(("  [PASS] " if cond else "  [FAIL] ") + label + ("" if cond else f"   {detail}"))
    if not cond:
        _failures.append(label)


def _new_session(client) -> str:
    r = client.post("/api/session")
    return (r.get_json() or {})["session_id"]


def _new_session_with_topologies(client):
    """建会话并造出两侧组网（跳过 step2/step3 生成流程）。"""
    from app.agent.session import session_manager

    sid = _new_session(client)
    r = client.post(f"/api/session/{sid}/workflow/step1", json=_BASE_PARAMS)
    assert r.status_code == 200, (r.status_code, r.get_json())
    session = session_manager.get_session(sid)
    eq_topo = session.original_topology.model_copy(deep=True)
    eq_topo.name = "等效组网"
    session.equivalent_topology = eq_topo
    session.equivalent_training_model = session.original_training_model
    return sid


def main() -> int:
    from app.main import create_app
    from app.agent.session import session_manager
    from app.mcp.client import mcp_client

    app = create_app()
    client = app.test_client()

    # ── 0. 长操作互斥 API 本身 ──
    print("[0] 长操作互斥 API：占用/冲突/释放")
    sid0 = _new_session(client)
    check("首次占用成功", session_manager.try_begin_exclusive(sid0) is True)
    check("同一会话二次占用失败", session_manager.try_begin_exclusive(sid0) is False)
    check("is_exclusive_busy=True", session_manager.is_exclusive_busy(sid0) is True)
    session_manager.end_exclusive(sid0)
    check("释放后可再次占用", session_manager.try_begin_exclusive(sid0) is True)
    session_manager.end_exclusive(sid0)
    check("释放后 busy=False", session_manager.is_exclusive_busy(sid0) is False)

    # ── 1. 并发 Step1：一个成功、一个 409 ──
    print("\n[1] 并发 Step1：一个 200、一个 409 task_busy")
    sid1 = _new_session(client)
    clients = [app.test_client() for _ in range(2)]
    barrier = threading.Barrier(2)
    results: list = []

    def call_step1(idx):
        barrier.wait()
        resp = clients[idx].post(f"/api/session/{sid1}/workflow/step1", json=_BASE_PARAMS)
        results.append((idx, resp.status_code, resp.get_json() or {}))

    threads = [threading.Thread(target=call_step1, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    codes = sorted(s for _, s, _ in results)
    print(f"       HTTP: {[(i, s) for i, s, _ in results]}")
    check("一个成功、一个 409", codes == [200, 409], str(results))
    check(
        "409 返回 task_busy",
        any((b or {}).get("error") == "task_busy" for _, s, b in results if s == 409),
        str(results),
    )
    session1 = session_manager.get_session(sid1)
    check("成功侧生成了原始组网", session1.original_topology is not None)
    check("结束后释放长操作名额", session_manager.is_exclusive_busy(sid1) is False)

    # ── 2. 短写并发：参数 + 公式行都成功且都落库 ──
    print("\n[2] 短写并发：仿真参数 + 公式行都成功、都落库")
    sid2 = _new_session(client)
    wclients = [app.test_client() for _ in range(2)]
    wbarrier = threading.Barrier(2)
    writes: list = []

    def write_params():
        wbarrier.wait()
        resp = wclients[0].post(
            f"/api/session/{sid2}/simulation-params",
            json={"epoch_num": 2, "model_name": "Qwen3-32B"},
        )
        writes.append(("params", resp.status_code))

    def write_formula():
        wbarrier.wait()
        resp = wclients[1].post(
            f"/api/session/{sid2}/formula-lines",
            json={"lines": [{"section": "test", "line": "tp_keep=1"}]},
        )
        writes.append(("formula", resp.status_code))

    threads = [threading.Thread(target=write_params), threading.Thread(target=write_formula)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    check("两个短写都 200", all(s == 200 for _, s in writes), str(writes))
    session2 = session_manager.get_session(sid2)
    check("仿真参数已落库", session2.simulation_params is not None, str(session2.simulation_params))
    check(
        "公式行已落库且内容正确",
        bool(session2.formula_lines)
        and session2.formula_lines[0].get("line") == "tp_keep=1",
        str(session2.formula_lines),
    )
    check("短写不占用长操作名额", session_manager.is_exclusive_busy(sid2) is False)

    # ── 3. 长操作占用中：仿真提交 409 且不下发任务 ──
    print("\n[3] 长操作占用中：run-simulation 409，且不调用 MCP")
    executed: list = []
    original_execute = mcp_client.execute_task

    def fake_execute(topology_data, params=None):
        executed.append(topology_data.get("name", ""))
        return f"task_{len(executed)}"

    mcp_client.execute_task = fake_execute
    original_card = mcp_client.get_card_details
    original_hbm = mcp_client.get_hbm_detail
    original_status = mcp_client.get_task_status
    mcp_client.get_card_details = lambda *a, **k: []
    mcp_client.get_hbm_detail = lambda *a, **k: {}
    mcp_client.get_task_status = lambda task_id: {"status": "running", "progress": 1}
    try:
        sid3 = _new_session_with_topologies(client)
        session_manager.try_begin_exclusive(sid3)
        try:
            r = client.post(f"/api/session/{sid3}/run-simulation", json={})
            body = r.get_json() or {}
            check(
                "返回 409 task_busy 且 MCP execute_task 未被调用",
                r.status_code == 409
                and body.get("error") == "task_busy"
                and executed == [],
                f"HTTP {r.status_code} {body} executed={executed}",
            )
        finally:
            session_manager.end_exclusive(sid3)
        r = client.post(f"/api/session/{sid3}/run-simulation", json={})
        check("释放后可提交（不永久卡 busy）", r.status_code == 200, f"HTTP {r.status_code}")
    finally:
        mcp_client.execute_task = original_execute
        mcp_client.get_card_details = original_card
        mcp_client.get_hbm_detail = original_hbm
        mcp_client.get_task_status = original_status

    # ── 4. 长操作占用中：Step3 409 ──
    print("\n[4] 长操作占用中：Step3 409 task_busy")
    sid4 = _new_session_with_topologies(client)
    session_manager.try_begin_exclusive(sid4)
    try:
        r = client.post(f"/api/session/{sid4}/workflow/step3")
        body = r.get_json() or {}
        check(
            "Step3 返回 409 task_busy",
            r.status_code == 409 and body.get("error") == "task_busy",
            f"HTTP {r.status_code} {body}",
        )
    finally:
        session_manager.end_exclusive(sid4)

    # ── 5. Step2 原子化：POST 仍互斥；stream 退化为纯回放（无锁/无写） ──
    print("\n[5] Step2 原子化：busy 时 POST 409；stream 纯回放不再 409")
    sid5 = _new_session_with_topologies(client)
    session5 = session_manager.get_session(sid5)
    # 公式行由 POST step2 原子落库；这里塞两条短行让回放瞬间完成（避免 22s 动画）
    session5.formula_lines = [
        {"section": "strategy", "line": "line-1"},
        {"section": "formula", "line": "line-2"},
    ]
    session_manager.try_begin_exclusive(sid5)
    try:
        r = client.post(f"/api/session/{sid5}/workflow/step2")
        body = r.get_json() or {}
        check(
            "busy 时 POST step2 返回 409 task_busy",
            r.status_code == 409 and body.get("error") == "task_busy",
            f"HTTP {r.status_code} {body}",
        )
        r = client.get(f"/api/session/{sid5}/workflow/step2/stream")
        text = r.get_data(as_text=True)
        check(
            "busy 时 stream 仍 200 回放（不占长操作名额）",
            r.status_code == 200 and '"type": "error"' not in text,
            f"HTTP {r.status_code} {text[:160]}",
        )
        check(
            "回放内容来自已落库公式行",
            '"line-1"' in text and '"line-2"' in text,
            text[:200],
        )
        check("回放带 done 事件", '"type": "done"' in text, text[-160:])
    finally:
        session_manager.end_exclusive(sid5)
    check("结束后名额释放", session_manager.is_exclusive_busy(sid5) is False)

    # ── 5b. 两个任务同时回放：各回放各的公式行，互不污染 ──
    print("\n[5b] 两个任务同时 Step2 回放：互不污染（2026-11-09 串台回归）")
    sid5b_1 = _new_session_with_topologies(client)
    sid5b_2 = _new_session_with_topologies(client)
    for tag, sidx in (("task-A", sid5b_1), ("task-B", sid5b_2)):
        sx = session_manager.get_session(sidx)
        sx.formula_lines = [
            {"section": "strategy", "line": tag + "-strategy"},
            {"section": "formula", "line": tag + "-formula"},
        ]
    replay = {}

    def replay_stream(sidx):
        replay[sidx] = app.test_client().get(
            f"/api/session/{sidx}/workflow/step2/stream"
        ).get_data(as_text=True)

    threads = [
        threading.Thread(target=replay_stream, args=(sid5b_1,)),
        threading.Thread(target=replay_stream, args=(sid5b_2,)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    text_a, text_b = replay[sid5b_1], replay[sid5b_2]
    check("A 回放只含 A 的公式行", "task-A-strategy" in text_a and "task-B-strategy" not in text_a, text_a[:200])
    check("B 回放只含 B 的公式行", "task-B-strategy" in text_b and "task-A-strategy" not in text_b, text_b[:200])
    check("两侧都带 done 事件", text_a.count('"type": "done"') == 1 and text_b.count('"type": "done"') == 1)

    # ── 5c. POST step2 原子完成：等效组网/模型/公式行一次落库 ──
    print("\n[5c] POST step2 原子完成：等效组网/模型/公式行一次落库")
    sid5c = _new_session(client)
    r = client.post(f"/api/session/{sid5c}/workflow/step1", json=_BASE_PARAMS)
    check("step1 200", r.status_code == 200, f"HTTP {r.status_code}")
    r = client.post(f"/api/session/{sid5c}/workflow/step2")
    check("step2 200", r.status_code == 200, f"HTTP {r.status_code} {r.get_json()}")
    s5c = session_manager.get_session(sid5c)
    check("step=equiv_generated（不再停在中转态）", getattr(s5c, "step", None) == "equiv_generated", str(getattr(s5c, "step", None)))
    check("等效组网已生成", s5c.equivalent_topology is not None)
    check("等效模型已生成", s5c.equivalent_training_model is not None)
    check(
        "公式行已落库",
        len(getattr(s5c, "formula_lines", None) or []) > 30,
        str(len(getattr(s5c, "formula_lines", None) or [])),
    )
    r = client.post(f"/api/session/{sid5c}/workflow/step2")
    check("重复 POST step2 幂等（仍 200）", r.status_code == 200, f"HTTP {r.status_code}")

    # ── 6. 长操作占用中：删除会话 409；释放后可删 ──
    print("\n[6] 长操作占用中：DELETE 409；释放后可删除")
    sid6 = _new_session(client)
    session_manager.try_begin_exclusive(sid6)
    try:
        r = client.delete(f"/api/session/{sid6}")
        check(
            "DELETE 返回 409 task_busy",
            r.status_code == 409 and (r.get_json() or {}).get("error") == "task_busy",
            f"HTTP {r.status_code} {r.get_json()}",
        )
    finally:
        session_manager.end_exclusive(sid6)
    r = client.delete(f"/api/session/{sid6}")
    check("释放后可正常删除", r.status_code == 200, f"HTTP {r.status_code}")

    print()
    if _failures:
        print(f"  [FAIL] {len(_failures)} 项未通过: {_failures}")
        return 1
    print("  [PASS] 会话写路径并发保护符合预期：长操作互斥、短写串行、busy 不产生副作用。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
