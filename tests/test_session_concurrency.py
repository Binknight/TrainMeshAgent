"""会话级并发安全回归测试（app/routes/session.py run_simulation）。

背景：`run_simulation()` 原先没有任何锁/幂等保护。两个并发请求（两个标签页各点
一次「开始仿真」、或完成回调 + 刷新兜底同时到达）都会读到 task_id=None，于是各自
向 MCP 下发一遍任务 —— 一个会话凭空多出 2 个孤儿仿真进程。已实测复现：2 次并发
调用产生 4 个 MCP 任务，会话只记住最后 2 个。

本测试锁住修复后的契约：
  1. 两个并发 POST /run-simulation 只产生 2 个 MCP 任务（原始/等效各一）；
  2. 再次 POST（非 force_retry）不重复下发（复用 task_id 补取结果）；
  3. force_retry=true 才会取消旧任务并重新下发（重试按钮依赖该语义）；
  4. 已完成的会话重复 POST 直接返回缓存结果，不再向 MCP 拉全量卡结果；
  5. 删除会话会尽力取消绑定的 MCP 任务；
  6. `session_manager.get_session()` 在缓存 miss 的并发场景只加载一份实例。

不启动真实服务、不碰用户数据：临时 SQLite + Flask test client + monkeypatch
MCP 客户端（不依赖 MCP Server / 仿真环境）。

Run: python tests/test_session_concurrency.py
"""

import os
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# 必须在 import app.* 之前设好（默认 SQLITE_PATH 是容器路径，Windows 上不存在）
_TMP_DB = REPO / ".tmp" / "session_concurrency_test"
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
    if cond:
        print(f"  [PASS] {label}")
    else:
        print(f"  [FAIL] {label} {detail}")
        _failures.append(label)


class MCPStub:
    """可编排的 MCP 客户端替身：记录调用、可控制卡结果是否就绪。"""

    def __init__(self):
        self.lock = threading.Lock()
        self.executed: list[str] = []        # 下发得到的 task_id 序列
        self.executed_names: list[str] = []  # 对应的组网名（便于报告）
        self.cancelled: list[str] = []
        self.card_calls = 0
        self.cards_ready = False

    # ── 被 app 代码调用的接口 ──
    def execute_task(self, topology_data, params=None):
        # 模拟真实提交的耗时（放大原竞态窗口）
        time.sleep(0.15)
        with self.lock:
            task_id = f"task_{len(self.executed) + 1}"
            self.executed.append(task_id)
            self.executed_names.append(topology_data.get("name", ""))
            return task_id

    def get_card_details(self, task_id, card_ids=None):
        with self.lock:
            self.card_calls += 1
            ready = self.cards_ready
        if not ready:
            return []
        return [
            {
                "card_id": f"card_{r}",
                "global_rank": r,
                "flops_per_card": 1.0e12,
                "hbm_gb": 10.0,
                "hbm_model_gb": 8.0,
                "tp_comm_gb_per_micro": 1.0,
                "pp_comm_mb_per_micro": 100.0,
                "dp_comm_gb_per_step": 2.0,
                "ep_comm_gb_per_step": 0.0,
            }
            for r in range(2)
        ]

    def get_hbm_detail(self, task_id, global_rank):
        return {}

    def get_task_status(self, task_id):
        return {"status": "running", "progress": 10}

    def cancel_task(self, task_id):
        with self.lock:
            self.cancelled.append(task_id)
        return {"task_id": task_id, "status": "cancelled"}


def _new_session_with_topologies(client):
    """建会话并造出原始+等效两侧组网/模型（跳过 step2/step3 的生成流程）。"""
    from app.agent.session import session_manager

    r = client.post("/api/session")
    sid = (r.get_json() or {})["session_id"]
    r = client.post(f"/api/session/{sid}/workflow/step1", json=_BASE_PARAMS)
    assert r.status_code == 200, (r.status_code, r.get_json())
    session = session_manager.get_session(sid)
    assert session and session.original_topology and session.original_training_model
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
    stub = MCPStub()
    mcp_client.execute_task = stub.execute_task
    mcp_client.get_card_details = stub.get_card_details
    mcp_client.get_hbm_detail = stub.get_hbm_detail
    mcp_client.get_task_status = stub.get_task_status
    mcp_client.cancel_task = stub.cancel_task

    # ── 1. 并发提交：只应产生 2 个 MCP 任务 ──
    print("[1] 并发 POST /run-simulation 不得重复下发任务")
    client = app.test_client()
    sid = _new_session_with_topologies(client)
    clients = [app.test_client() for _ in range(2)]
    barrier = threading.Barrier(2)
    responses: list = []

    def submit(idx):
        barrier.wait()
        resp = clients[idx].post(f"/api/session/{sid}/run-simulation", json={})
        responses.append((idx, resp.status_code, resp.get_json() or {}))

    threads = [threading.Thread(target=submit, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print(f"       HTTP: {[(i, s) for i, s, _ in responses]}")
    print(f"       MCP execute_task 调用: {list(zip(stub.executed, stub.executed_names))}")
    codes = sorted(s for _, s, _ in responses)
    check(
        "并发请求一个成功、一个被 409 互斥拦下（不会重复下发）",
        codes == [200, 409],
        str(responses),
    )
    check("被拦下的请求返回 task_busy", any(
        (b or {}).get("error") == "task_busy" for _, s, b in responses if s == 409
    ), str(responses))
    check("MCP 任务数恰为 2（无孤儿任务）", len(stub.executed) == 2, f"actual={len(stub.executed)}")
    session = session_manager.get_session(sid)
    check(
        "会话记录的两侧 task_id 都在实际下发集合内",
        {session.original_task_id, session.equivalent_task_id} == set(stub.executed),
        f"session=({session.original_task_id},{session.equivalent_task_id}) executed={stub.executed}",
    )

    # ── 2. 非 force_retry 重复调用：复用 task_id，不再下发 ──
    print("\n[2] 重复 POST（非 force_retry）不得重复下发")
    r = client.post(f"/api/session/{sid}/run-simulation", json={})
    check(
        "重复调用 200 且 execute_task 仍为 2 次",
        r.status_code == 200 and len(stub.executed) == 2,
        f"HTTP {r.status_code} executed={len(stub.executed)}",
    )

    # ── 3. force_retry：取消旧任务并重新下发 ──
    print("\n[3] force_retry=true 取消旧任务并重新下发")
    old_ids = (session.original_task_id, session.equivalent_task_id)
    session.step = "failed"
    r = client.post(f"/api/session/{sid}/run-simulation", json={"force_retry": True})
    body = r.get_json() or {}
    check(
        "force_retry 重新下发（executed=4，old ids 被取消）",
        r.status_code == 200
        and len(stub.executed) == 4
        and set(old_ids).issubset(set(stub.cancelled)),
        f"HTTP {r.status_code} executed={len(stub.executed)} cancelled={stub.cancelled}",
    )
    new_ids = (body.get("original_task_id"), body.get("equivalent_task_id"))
    check(
        "force_retry 后 task_id 已换成新下发的两个",
        set(new_ids).issubset(set(stub.executed)) and not set(new_ids) & set(old_ids),
        f"new={new_ids} old={old_ids}",
    )

    # ── 4. 已完成会话的重复调用走缓存，不再拉全量卡结果 ──
    print("\n[4] 已完成会话重复 POST 直接返回缓存结果")
    stub.cards_ready = True
    sid2 = _new_session_with_topologies(client)
    r = client.post(f"/api/session/{sid2}/run-simulation", json={})
    body = r.get_json() or {}
    session2 = session_manager.get_session(sid2)
    check(
        "首次调用即完成对比（两张卡结果就绪）",
        r.status_code == 200
        and (body.get("results") or {}).get("comparison") is not None
        and session2.step == "completed",
        f"HTTP {r.status_code} step={session2.step}",
    )
    card_calls_before = stub.card_calls
    r = client.post(f"/api/session/{sid2}/run-simulation", json={})
    body = r.get_json() or {}
    check(
        "重复调用返回 cached=true 且未再查卡结果",
        r.status_code == 200
        and body.get("cached") is True
        and stub.card_calls == card_calls_before,
        f"HTTP {r.status_code} cached={body.get('cached')} card_calls={stub.card_calls}",
    )

    # ── 4b. MCP 队列满（-32004）→ 429 背压，不能把会话标 failed ──
    print("\n[4b] MCP 队列满：返回 429 simulation_queue_full 且会话不置 failed")
    from app.mcp.client import MCPQueueFullError

    sid3 = _new_session_with_topologies(client)
    session3 = session_manager.get_session(sid3)

    def queue_full_execute(topology_data, params=None):
        raise MCPQueueFullError("仿真队列已满：最多 20 个排队任务，请稍后重试")

    mcp_client.execute_task = queue_full_execute
    try:
        r = client.post(f"/api/session/{sid3}/run-simulation", json={})
        body = r.get_json() or {}
        check(
            "队列满返回 429 + simulation_queue_full",
            r.status_code == 429 and body.get("error") == "simulation_queue_full",
            f"HTTP {r.status_code} {body}",
        )
        check(
            "队列满不把会话标成 failed",
            session3.step != "failed",
            f"step={session3.step}",
        )
        check(
            "队列满不留下 task_id（重试可正常提交）",
            not session3.original_task_id and not session3.equivalent_task_id,
            f"ids=({session3.original_task_id},{session3.equivalent_task_id})",
        )
    finally:
        mcp_client.execute_task = stub.execute_task

    # ── 4c. 同会话长操作占用中 → 409 task_busy，不重复下发 ──
    print("\n[4c] 会话长操作占用中：run-simulation 409 task_busy")
    executed_before = len(stub.executed)
    session_manager.try_begin_exclusive(sid3)
    try:
        r = client.post(f"/api/session/{sid3}/run-simulation", json={})
        body = r.get_json() or {}
        check(
            "长操作占用时返回 409 task_busy 且不下发任务",
            r.status_code == 409
            and body.get("error") == "task_busy"
            and len(stub.executed) == executed_before,
            f"HTTP {r.status_code} {body} executed={len(stub.executed)}",
        )
    finally:
        session_manager.end_exclusive(sid3)
    r = client.post(f"/api/session/{sid3}/run-simulation", json={})
    check("释放后可正常提交（不会永久卡 busy）", r.status_code == 200, f"HTTP {r.status_code}")

    # ── 5. 删除会话尽力取消 MCP 任务 ──
    print("\n[5] 删除会话时取消绑定的 MCP 任务")
    session2 = session_manager.get_session(sid2)
    ids_before = (session2.original_task_id, session2.equivalent_task_id)
    r = client.delete(f"/api/session/{sid2}")
    check(
        "DELETE 会话 200 且旧 task_id 被取消",
        r.status_code == 200 and set(ids_before).issubset(set(stub.cancelled)),
        f"HTTP {r.status_code} cancelled={stub.cancelled[-2:]}",
    )

    # ── 6. get_session 缓存 miss 的并发加载只返回一个实例 ──
    print("\n[6] 并发 get_session（缓存 miss）返回同一实例")
    from app.agent.session import _load_session  # noqa: F401
    session_manager.save_session(session)  # 确保已落库
    with session_manager._lock:
        session_manager._sessions.pop(sid, None)
    got = []

    def load():
        got.append(session_manager.get_session(sid))

    threads = [threading.Thread(target=load) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    check(
        "两个线程拿到同一个 SessionState 实例",
        len(got) == 2 and got[0] is not None and got[0] is got[1],
        f"ids={[id(g) for g in got]}",
    )

    print()
    if _failures:
        print(f"  [FAIL] {len(_failures)} 项未通过: {_failures}")
        return 1
    print("  [PASS] 会话级并发安全符合预期：无重复下发、force_retry 语义正确、删除会取消任务。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
