"""MCP Server 优雅停机 / 重启恢复的验收测试（standalone，非 pytest）。

覆盖 task-2 的 ④：

1. ``task_state.json`` 迁移写入：prepare / queued / running / terminal（completed、
   cancelled、执行超时、排队超时）都能 best-effort 落盘，字段齐全；无 workspace_dir
   的记录不落盘且不抛异常；
2. ``task_loader.load_task_record`` 恢复规则：terminal 原样采用；running/queued
   有结果→completed、无结果→failed（message 含「服务重启，任务中断（进程已丢失）」）；
   dry_run+submitted→submitted；无 task_state.json→legacy 推断；损坏文件回落 legacy；
3. ``recover_interrupted_tasks`` 启动扫描：只收敛 running/queued、统计准确、幂等
   （第二轮不重复改动）、不碰终态/dry_run/no-state 任务；
4. ``scheduler.shutdown(cancel_running=True)``：取消并 kill 运行中 fake 任务、写
   cancelled 终态、停 worker 线程、重复调用幂等；
5. FastAPI lifespan：启动顺序 = 恢复扫描 → scheduler.start → cleanup.start，停机顺序
   = cleanup.stop → scheduler.shutdown(cancel_running=True)；并用**真实 uvicorn**
   跑一遍启动/优雅停机（uvicorn 收到 SIGTERM/退出信号时走的就是 Server.shutdown →
   lifespan shutdown，这里用 ``server.should_exit`` 触发同一路径），验证启动恢复扫描
   与停机取消运行中任务都真实生效。

全程 fake process / 临时 workspace，不依赖真实 aicm 与真实仿真环境。
Run: python tests/test_restart_recovery.py
"""

import asyncio
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# Windows 控制台默认代码页与 UTF-8 报告不一致
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# 本机可能配了到外网的代理，会拦截 127.0.0.1 的 HTTP 探测
for _k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    os.environ.pop(_k, None)

try:
    import pydantic  # noqa: F401
except ImportError:
    print("  [SKIP] 未安装 pydantic，跳过重启恢复测试 —— 请先执行:")
    print("         pip install -r mcp_server/requirements.txt")
    sys.exit(0)

try:
    from mcp_server.config import settings
    from mcp_server.schemas.common import ExecuteTaskInput, TaskStatus
    from mcp_server.services import task_state
    from mcp_server.services.simulation_runner import prepare_task
    from mcp_server.services.task_loader import load_task_record, recover_interrupted_tasks
    from mcp_server.services.task_scheduler import TaskScheduler
    from mcp_server.services.task_store import TaskRecord, task_store
except ImportError as exc:  # pragma: no cover
    print(f"  [SKIP] 无法导入 mcp_server 依赖（{exc}）—— 请先执行:")
    print("         pip install -r mcp_server/requirements.txt")
    sys.exit(0)

_failures: list[str] = []


def check(label: str, cond: bool, detail: str = "") -> None:
    print(("  [PASS] " if cond else "  [FAIL] ") + label + ("" if cond else f"   {detail}"))
    if not cond:
        _failures.append(label)


def skip(label: str, why: str) -> None:
    print(f"  [SKIP] {label} —— {why}")


# ---------------------------------------------------------------------------
# 测试替身与夹具
# ---------------------------------------------------------------------------


class FakeProcess:
    """最小 subprocess.Popen 替身：poll/kill 可控，不产生真实进程。"""

    def __init__(self, pid: int = 4321) -> None:
        self.pid = pid
        self.returncode = None
        self.kill_count = 0

    def poll(self):
        return self.returncode

    def kill(self) -> None:
        self.kill_count += 1
        if self.returncode is None:
            self.returncode = -9

    def wait(self, timeout=None):  # noqa: ARG002
        return self.returncode


class FakeClock:
    """可推进的 time.monotonic 替身。"""

    def __init__(self, start: float = 5000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def make_launcher(procs: dict[str, FakeProcess]):
    def launcher(record: TaskRecord) -> FakeProcess:
        process = FakeProcess(pid=4400 + len(procs))
        procs[record.task_id] = process
        return process

    return launcher


def make_scheduler(
    *,
    clock: FakeClock,
    launcher,
    max_concurrent: int = 1,
    max_queue_size: int = 20,
    task_timeout: float = 1000.0,
    queue_timeout: float = 1000.0,
    auto_start: bool = False,
) -> TaskScheduler:
    return TaskScheduler(
        max_concurrent_tasks=max_concurrent,
        max_queue_size=max_queue_size,
        task_timeout_sec=task_timeout,
        queue_timeout_sec=queue_timeout,
        launcher=launcher,
        clock=clock,
        auto_start=auto_start,
    )


TOPO = {
    "name": "原始组网",
    "device_type": "A3",
    "dp_size": 2,
    "tp_size": 1,
    "pp_size": 1,
    "total_nodes": 2,
    "num_layers": 2,
    "hidden_dim": 128,
    "num_heads": 8,
}


def make_input():
    return ExecuteTaskInput.model_validate({"topology": dict(TOPO), "simulation_params": {}})


TASK_STATE_KEYS = {
    "status",
    "dry_run",
    "pid",
    "started_at",
    "finished_at",
    "message",
    "updated_at",
}


def read_state(task_dir: Path):
    path = Path(task_dir) / "task_state.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def seed_task_dir(
    ws: Path,
    name: str,
    *,
    state_status: str | None = None,
    dry_run: bool = False,
    message: str | None = None,
    results: bool = False,
    state_json: str | None = None,
) -> Path:
    """构造一个任务目录：task_meta.json + 可选 task_state.json / results。"""
    task_dir = ws / name
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "task_meta.json").write_text(
        json.dumps({"topology": dict(TOPO), "simulation_params": {}}),
        encoding="utf-8",
    )
    if results:
        mocked = task_dir / "results" / "mocked_workload"
        mocked.mkdir(parents=True, exist_ok=True)
        (mocked / "model_rank0_time.csv").write_text("comm_type,start_time\n", encoding="utf-8")
    if state_json is not None:
        (task_dir / "task_state.json").write_text(state_json, encoding="utf-8")
    elif state_status is not None:
        payload = {
            "status": state_status,
            "dry_run": dry_run,
            "pid": 12345,
            "started_at": "2026-01-01T00:00:00Z",
            "finished_at": None,
            "message": message,
            "updated_at": "2026-01-01T00:00:01Z",
        }
        (task_dir / "task_state.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )
    return task_dir


def fresh_load(task_id: str):
    """清空内存任务表后按磁盘恢复（避免内存命中掩盖 loader 行为）。"""
    task_store._tasks.clear()  # 测试专用：模拟「服务刚重启，内存为空」
    return load_task_record(task_id)


def wait_until(predicate, timeout: float = 5.0, interval: float = 0.01) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def health_ok(port: int) -> bool:
    """探测 127.0.0.1 上的 /health（显式禁用环境代理）。"""
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(f"http://127.0.0.1:{port}/health", timeout=2) as resp:
            return resp.status == 200
    except Exception:  # noqa: BLE001 — 启动期间连接失败是预期
        return False


work = REPO / ".tmp" / "mcp_restart_recovery"
shutil.rmtree(work, ignore_errors=True)
(work / "aicm" / "examples").mkdir(parents=True)
(work / "ws").mkdir(parents=True)
(work / "aicm" / "run.py").write_text("# fake simulation entry, never executed\n", encoding="utf-8")
(work / "aicm" / "examples" / "time_args.json").write_text("{}", encoding="utf-8")

_saved_settings = {
    name: getattr(settings, name)
    for name in ("sim_tool_home", "workspace_root", "dry_run", "conda_env")
}

print("=" * 78)
print("  MCP Server 优雅停机 / 重启恢复：task_state 落盘 / loader / 恢复扫描 / shutdown")
print("=" * 78)

try:
    settings.sim_tool_home = work / "aicm"
    settings.workspace_root = work / "ws"
    settings.dry_run = False
    settings.conda_env = ""

    # -----------------------------------------------------------------------
    # [1] task_state.json 迁移写入
    # -----------------------------------------------------------------------
    print("\n[1] task_state.json 迁移写入（prepare/queued/running/terminal）")
    ws = work / "ws"
    clock = FakeClock()
    procs: dict[str, FakeProcess] = {}
    sched = make_scheduler(clock=clock, launcher=make_launcher(procs))

    record = prepare_task(make_input())
    state = read_state(record.workspace_dir)
    check("prepare 迁移点写入 task_state.json", state is not None)
    state = state or {}
    check(
        "prepare 状态 = submitted、dry_run=false、pid/started/finished 为空",
        state.get("status") == "submitted"
        and state.get("dry_run") is False
        and state.get("pid") is None
        and state.get("started_at") is None
        and state.get("finished_at") is None,
        str(state),
    )
    check("task_state 字段齐全（status/dry_run/pid/started_at/finished_at/message/updated_at）", TASK_STATE_KEYS <= set(state), str(sorted(state)))

    sched.submit(record)
    state = read_state(record.workspace_dir) or {}
    check(
        "queued 迁移点：status=queued、pid/started_at/finished_at 为空",
        state.get("status") == "queued"
        and state.get("pid") is None
        and state.get("started_at") is None
        and state.get("finished_at") is None,
        str(state),
    )
    check("queued 状态 message 非空", bool(state.get("message")), str(state.get("message")))

    sched.tick()
    state = read_state(record.workspace_dir) or {}
    check(
        "running 迁移点：status=running、pid=子进程 pid、started_at 为 ISO 时间",
        state.get("status") == "running"
        and state.get("pid") == 4400
        and isinstance(state.get("started_at"), str)
        and str(state.get("started_at", "")).endswith("Z"),
        str(state),
    )

    procs[record.task_id].returncode = 0
    sched.tick()
    state = read_state(record.workspace_dir) or {}
    check(
        "terminal(completed) 迁移点：终态 + finished_at 落盘",
        state.get("status") == "completed"
        and state.get("pid") == 4400
        and isinstance(state.get("finished_at"), str)
        and "successfully" in (state.get("message") or ""),
        str(state),
    )

    # cancel running → terminal(cancelled)
    record2 = prepare_task(make_input())
    sched.submit(record2)
    sched.tick()
    sched.cancel(record2)
    state = read_state(record2.workspace_dir) or {}
    check(
        "cancel 迁移点：cancelled 终态落盘（含 finished_at）",
        state.get("status") == "cancelled" and isinstance(state.get("finished_at"), str),
        str(state),
    )
    check("cancel 的 fake 进程被 kill", procs[record2.task_id].kill_count == 1)

    # 执行超时 → terminal(failed)
    clock3 = FakeClock()
    procs3: dict[str, FakeProcess] = {}
    sched3 = make_scheduler(
        clock=clock3,
        launcher=make_launcher(procs3),
        task_timeout=120,
        queue_timeout=1000,
    )
    record3 = prepare_task(make_input())
    sched3.submit(record3)
    sched3.tick()
    clock3.advance(120)
    sched3.tick()
    state = read_state(record3.workspace_dir) or {}
    check(
        "执行超时迁移点：failed 终态落盘、message 含 task timeout",
        state.get("status") == "failed" and "task timeout" in (state.get("message") or ""),
        str(state),
    )

    # 排队超时 → terminal(failed)
    clock4 = FakeClock()
    procs4: dict[str, FakeProcess] = {}
    sched4 = make_scheduler(
        clock=clock4,
        launcher=make_launcher(procs4),
        task_timeout=1000,
        queue_timeout=30,
    )
    holder = prepare_task(make_input())
    sched4.submit(holder)
    sched4.tick()
    queued_rec = prepare_task(make_input())
    sched4.submit(queued_rec)
    clock4.advance(30)
    sched4.tick()
    state = read_state(queued_rec.workspace_dir) or {}
    check(
        "排队超时迁移点：failed 终态落盘、message 含 queue timeout",
        state.get("status") == "failed" and "queue timeout" in (state.get("message") or ""),
        str(state),
    )

    # dry_run prepare → submitted + dry_run=true
    settings.dry_run = True
    dry_rec = prepare_task(make_input())
    state = read_state(dry_rec.workspace_dir) or {}
    settings.dry_run = False
    check(
        "dry_run prepare 迁移点：status=submitted 且 dry_run=true",
        state.get("status") == "submitted" and state.get("dry_run") is True,
        str(state),
    )

    # best-effort：无 workspace_dir / 写不进去都不能抛异常影响主流程
    bare = TaskRecord(task_id="sim_no_workspace", topology=None, simulation_params={})
    wrote = task_state.write_task_state(bare, status=TaskStatus.QUEUED)
    check("无 workspace_dir 的记录不落盘且不抛异常（返回 False）", wrote is False)
    orphan_dir = work / "orphan_state_dir"
    wrote_orphan = task_state.write_task_state_fields(orphan_dir, status="failed", message="orphan")
    check(
        "写状态文件失败不影响主流程：目标目录不存在时会自动创建（best-effort 返回 True）",
        wrote_orphan is True and (read_state(orphan_dir) or {}).get("status") == "failed",
        str(read_state(orphan_dir)),
    )

    # -----------------------------------------------------------------------
    # [2] load_task_record 恢复规则
    # -----------------------------------------------------------------------
    print("\n[2] load_task_record 恢复规则（task_state 优先 / legacy 兜底）")
    loader_ws = work / "loader_ws"
    loader_ws.mkdir(parents=True, exist_ok=True)
    settings.workspace_root = loader_ws

    seed_task_dir(loader_ws, "sim_legacy_results", results=True)
    record = fresh_load("sim_legacy_results")
    check(
        "legacy：无 task_state 但有 results → completed / progress=100",
        record is not None and record.status == TaskStatus.COMPLETED and record.progress == 100.0,
        record.status.value if record else "None",
    )

    seed_task_dir(loader_ws, "sim_legacy_none")
    record = fresh_load("sim_legacy_none")
    check(
        "legacy：无 task_state 且无 results → submitted",
        record is not None and record.status == TaskStatus.SUBMITTED,
        record.status.value if record else "None",
    )

    seed_task_dir(loader_ws, "sim_state_completed", state_status="completed", message="done from state", results=True)
    record = fresh_load("sim_state_completed")
    check(
        "terminal(completed) → 原样采用该状态与 message",
        record is not None
        and record.status == TaskStatus.COMPLETED
        and record.progress == 100.0
        and record.message == "done from state",
        f"{record.status.value}/{record.message}" if record else "None",
    )

    seed_task_dir(loader_ws, "sim_state_failed", state_status="failed", message="boom")
    record = fresh_load("sim_state_failed")
    check(
        "terminal(failed) → failed 且保留落盘 message",
        record is not None and record.status == TaskStatus.FAILED and record.message == "boom",
        f"{record.status.value}/{record.message}" if record else "None",
    )

    seed_task_dir(loader_ws, "sim_state_error", state_status="error", message="bad result")
    record = fresh_load("sim_state_error")
    check(
        "terminal(error) → error",
        record is not None and record.status == TaskStatus.ERROR,
        record.status.value if record else "None",
    )

    seed_task_dir(loader_ws, "sim_state_cancelled", state_status="cancelled", message="user cancelled")
    record = fresh_load("sim_state_cancelled")
    check(
        "terminal(cancelled) → cancelled",
        record is not None and record.status == TaskStatus.CANCELLED,
        record.status.value if record else "None",
    )

    seed_task_dir(loader_ws, "sim_running_results", state_status="running", results=True)
    record = fresh_load("sim_running_results")
    check(
        "running + 有 results → completed（进程虽丢失但结果完整）",
        record is not None
        and record.status == TaskStatus.COMPLETED
        and record.progress == 100.0,
        f"{record.status.value}" if record else "None",
    )

    seed_task_dir(loader_ws, "sim_running_lost", state_status="running")
    record = fresh_load("sim_running_lost")
    check(
        "running + 无 results → failed 且 message 含「服务重启，任务中断（进程已丢失）」",
        record is not None
        and record.status == TaskStatus.FAILED
        and "服务重启，任务中断（进程已丢失）" in (record.message or ""),
        f"{record.status.value}/{record.message}" if record else "None",
    )

    seed_task_dir(loader_ws, "sim_queued_lost", state_status="queued")
    record = fresh_load("sim_queued_lost")
    check(
        "queued（从未拉起进程）→ failed 且 message 含中断提示",
        record is not None
        and record.status == TaskStatus.FAILED
        and "服务重启，任务中断（进程已丢失）" in (record.message or ""),
        f"{record.status.value}/{record.message}" if record else "None",
    )

    seed_task_dir(loader_ws, "sim_dryrun_submitted", state_status="submitted", dry_run=True, results=False)
    record = fresh_load("sim_dryrun_submitted")
    check(
        "dry_run + submitted → submitted（dry_run 任务不算中断）",
        record is not None and record.status == TaskStatus.SUBMITTED,
        record.status.value if record else "None",
    )

    seed_task_dir(loader_ws, "sim_corrupt_state", state_json="{not valid json")
    record = fresh_load("sim_corrupt_state")
    check(
        "task_state.json 损坏 → 回落 legacy 推断（无 results → submitted）",
        record is not None and record.status == TaskStatus.SUBMITTED,
        record.status.value if record else "None",
    )

    task_store._tasks.clear()
    check("不存在的 task_id 仍返回 None", load_task_record("sim_does_not_exist") is None)

    # -----------------------------------------------------------------------
    # [3] recover_interrupted_tasks 启动恢复扫描（幂等）
    # -----------------------------------------------------------------------
    print("\n[3] recover_interrupted_tasks 启动扫描（统计 / 幂等 / 不碰终态）")
    rec_ws = work / "recovery_ws"
    rec_ws.mkdir(parents=True, exist_ok=True)
    settings.workspace_root = rec_ws

    seed_task_dir(rec_ws, "sim_running_lost", state_status="running")
    seed_task_dir(rec_ws, "sim_running_done", state_status="running", results=True)
    seed_task_dir(rec_ws, "sim_queued_lost", state_status="queued")
    terminal_dir = seed_task_dir(rec_ws, "sim_terminal_completed", state_status="completed", message="old terminal", results=True)
    failed_dir = seed_task_dir(rec_ws, "sim_terminal_failed", state_status="failed", message="old failed")
    dryrun_dir = seed_task_dir(rec_ws, "sim_dryrun", state_status="submitted", dry_run=True)
    nostate_dir = seed_task_dir(rec_ws, "sim_nostate")
    (rec_ws / "not_a_task").mkdir(parents=True, exist_ok=True)

    before_terminal = read_state(terminal_dir)
    before_failed = read_state(failed_dir)
    before_dryrun = read_state(dryrun_dir)

    stats = recover_interrupted_tasks(rec_ws)
    check(
        "恢复扫描统计：scanned=6 interrupted=3 completed=1 failed=2 errors=0",
        stats == {"scanned": 6, "interrupted": 3, "completed": 1, "failed": 2, "errors": 0},
        str(stats),
    )
    check(
        "running 无结果 → failed + 中断提示",
        (read_state(rec_ws / "sim_running_lost") or {}).get("status") == "failed"
        and "服务重启，任务中断（进程已丢失）"
        in ((read_state(rec_ws / "sim_running_lost") or {}).get("message") or ""),
        str(read_state(rec_ws / "sim_running_lost")),
    )
    check(
        "running 有结果 → completed",
        (read_state(rec_ws / "sim_running_done") or {}).get("status") == "completed",
        str(read_state(rec_ws / "sim_running_done")),
    )
    check(
        "queued → failed + 中断提示",
        (read_state(rec_ws / "sim_queued_lost") or {}).get("status") == "failed",
        str(read_state(rec_ws / "sim_queued_lost")),
    )
    check("终态任务不被恢复扫描改写", read_state(terminal_dir) == before_terminal and read_state(failed_dir) == before_failed)
    check("dry_run/submitted 不被恢复扫描改写", read_state(dryrun_dir) == before_dryrun)
    check("无 task_state.json 的旧任务不被扫描", read_state(nostate_dir) is None)

    after_first = {
        name: read_state(rec_ws / name)
        for name in (
            "sim_running_lost",
            "sim_running_done",
            "sim_queued_lost",
            "sim_terminal_completed",
            "sim_terminal_failed",
            "sim_dryrun",
        )
    }
    stats_second = recover_interrupted_tasks(rec_ws)
    after_second = {
        name: read_state(rec_ws / name) for name in after_first
    }
    check(
        "恢复扫描幂等：第二轮不再改动任何任务",
        stats_second["interrupted"] == 0
        and stats_second["completed"] == 0
        and stats_second["failed"] == 0
        and after_first == after_second,
        str(stats_second),
    )
    check(
        "确认不存在 workspace 时扫描直接返回空统计",
        recover_interrupted_tasks(work / "no_such_ws")["scanned"] == 0,
    )

    # 恢复扫描后的任务再经 loader 读取仍正确（重启后 report_status 走这条路）
    record = fresh_load("sim_running_lost")
    check(
        "恢复扫描落盘后 loader 读到 failed + 中断提示",
        record is not None
        and record.status == TaskStatus.FAILED
        and "服务重启，任务中断（进程已丢失）" in (record.message or ""),
        f"{record.status.value if record else None}",
    )

    # -----------------------------------------------------------------------
    # [4] scheduler.shutdown(cancel_running=True) 优雅停机
    # -----------------------------------------------------------------------
    print("\n[4] scheduler.shutdown(cancel_running=True)：取消运行中任务 + 停 worker（幂等）")
    shut_ws = work / "shutdown_ws"
    shut_ws.mkdir(parents=True, exist_ok=True)
    settings.workspace_root = shut_ws
    settings.dry_run = False

    shut_procs: dict[str, FakeProcess] = {}
    running_sched = make_scheduler(
        clock=FakeClock(),
        launcher=make_launcher(shut_procs),
        auto_start=True,
    )
    running_sched.start()
    running_rec = TaskRecord(task_id="sim_shutdown_running", topology=None, simulation_params={})
    running_rec.workspace_dir = shut_ws / running_rec.task_id
    running_rec.workspace_dir.mkdir(parents=True, exist_ok=True)
    running_sched.submit(running_rec)
    queued_rec2 = TaskRecord(task_id="sim_shutdown_queued", topology=None, simulation_params={})
    queued_rec2.workspace_dir = shut_ws / queued_rec2.task_id
    queued_rec2.workspace_dir.mkdir(parents=True, exist_ok=True)
    running_sched.submit(queued_rec2)

    check(
        "worker 已拉起运行中任务、排队任务排队中",
        wait_until(lambda: running_sched.is_task_running("sim_shutdown_running"))
        and running_sched.queued_task_ids() == ["sim_shutdown_queued"],
        f"running={running_sched.running_task_ids()} queued={running_sched.queued_task_ids()}",
    )
    check(
        "运行中任务 running 状态已落盘",
        (read_state(running_rec.workspace_dir) or {}).get("status") == "running",
        str(read_state(running_rec.workspace_dir)),
    )

    running_sched.shutdown(cancel_running=True)
    check("shutdown 后 worker 线程停止", running_sched.worker_alive is False)
    check(
        "运行中任务被取消（cancelled 终态 + fake 进程被 kill）",
        running_rec.status == TaskStatus.CANCELLED
        and shut_procs["sim_shutdown_running"].kill_count == 1,
        f"{running_rec.status.value} kills={shut_procs['sim_shutdown_running'].kill_count}",
    )
    check(
        "cancelled 终态已落盘（重启后不会看到 running）",
        (read_state(running_rec.workspace_dir) or {}).get("status") == "cancelled",
        str(read_state(running_rec.workspace_dir)),
    )

    kill_count_before = shut_procs["sim_shutdown_running"].kill_count
    running_sched.shutdown(cancel_running=True)
    check(
        "重复 shutdown 幂等（不再 kill、不再改状态、不抛异常）",
        shut_procs["sim_shutdown_running"].kill_count == kill_count_before
        and running_rec.status == TaskStatus.CANCELLED
        and running_sched.worker_alive is False,
    )
    try:
        running_sched.submit(TaskRecord(task_id="sim_after_shutdown", topology=None, simulation_params={}))
        check("shutdown 之后 submit 被拒绝（scheduler closed）", False, "未抛异常")
    except RuntimeError:
        check("shutdown 之后 submit 被拒绝（scheduler closed）", True)

    # -----------------------------------------------------------------------
    # [5] FastAPI lifespan：启停顺序 + 真实 uvicorn 优雅停机
    # -----------------------------------------------------------------------
    print("\n[5] FastAPI lifespan：启停顺序 / 启动恢复扫描 / 真实 uvicorn 优雅停机")
    try:
        import uvicorn

        from mcp_server import main as mcp_main
        from mcp_server.services import task_scheduler as task_scheduler_module

        has_uvicorn = True
    except ImportError as exc:  # pragma: no cover
        print(f"  [SKIP] lifespan/uvicorn 集成检查 —— 缺少依赖: {exc}")
        has_uvicorn = False

    if has_uvicorn:
        # 5a. 顺序契约（用替身，不起真实线程）
        calls: list = []

        class FakeSched:
            def start(self):
                calls.append("scheduler.start")

            def shutdown(self, cancel_running: bool = False):
                calls.append(("scheduler.shutdown", cancel_running))

        class FakeCleanup:
            def start(self):
                calls.append("cleanup.start")

            def stop(self):
                calls.append("cleanup.stop")

        def fake_recover():
            calls.append("recover")
            return {"scanned": 0, "interrupted": 0, "completed": 0, "failed": 0, "errors": 0}

        saved_mod_attrs = (
            mcp_main.task_scheduler,
            mcp_main.workspace_cleaner,
            mcp_main.recover_interrupted_tasks,
        )
        try:
            mcp_main.task_scheduler = SimpleNamespace(scheduler=FakeSched())
            mcp_main.workspace_cleaner = SimpleNamespace(cleanup_worker=FakeCleanup())
            mcp_main.recover_interrupted_tasks = fake_recover

            async def drive_lifespan():
                async with mcp_main.app.router.lifespan_context(mcp_main.app):
                    pass

            check("FastAPI app 已接入 lifespan", mcp_main.app.router.lifespan_context is not None)
            asyncio.run(drive_lifespan())
            check(
                "启动顺序：recover → scheduler.start → cleanup.start",
                calls[:3] == ["recover", "scheduler.start", "cleanup.start"],
                str(calls),
            )
            check(
                "停机顺序：cleanup.stop → scheduler.shutdown(cancel_running=True)",
                calls[3:] == ["cleanup.stop", ("scheduler.shutdown", True)],
                str(calls),
            )
        finally:
            (
                mcp_main.task_scheduler,
                mcp_main.workspace_cleaner,
                mcp_main.recover_interrupted_tasks,
            ) = saved_mod_attrs

        check(
            "import 阶段不再启动调度线程（启动副作用已移入 lifespan）",
            task_scheduler_module.scheduler.worker_alive is False,
        )

        # 5b. 真实 uvicorn：启动触发恢复扫描，优雅停机触发取消运行中任务
        e2e_ws = work / "uvicorn_ws"
        e2e_ws.mkdir(parents=True, exist_ok=True)
        settings.workspace_root = e2e_ws
        settings.dry_run = True
        seeded_dir = seed_task_dir(e2e_ws, "sim_seeded_interrupted", state_status="running")

        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()
        config = uvicorn.Config(
            mcp_main.app,
            host="127.0.0.1",
            port=port,
            log_level="warning",
            lifespan="on",
        )
        server = uvicorn.Server(config)
        server_thread = threading.Thread(target=server.run, name="uvicorn-e2e", daemon=True)
        server_thread.start()
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

        live_rec = None
        try:
            check(
                "uvicorn 服务启动并执行 lifespan startup",
                wait_until(lambda: server.started, timeout=20.0),
            )
            if server.started:
                with opener.open(f"http://127.0.0.1:{port}/health", timeout=5) as resp:
                    health = json.loads(resp.read().decode())
                check("真实 HTTP GET /health 正常", resp.status == 200 and health.get("status") == "ok", str(health))

                seeded_state = read_state(seeded_dir) or {}
                check(
                    "启动恢复扫描：上次遗留的 running 任务被落成 failed（含中断提示）",
                    seeded_state.get("status") == "failed"
                    and "服务重启，任务中断（进程已丢失）" in (seeded_state.get("message") or ""),
                    str(seeded_state),
                )
                check(
                    "lifespan startup 后调度 worker 已启动",
                    wait_until(lambda: task_scheduler_module.scheduler.worker_alive, timeout=5.0),
                )

                # 灌一个真实运行中的 fake 任务，验证优雅停机 cancel_running 生效
                live_procs: dict[str, FakeProcess] = {}
                task_scheduler_module.scheduler._launcher = make_launcher(live_procs)
                live_rec = TaskRecord(task_id="sim_uvicorn_live", topology=None, simulation_params={})
                live_rec.workspace_dir = e2e_ws / live_rec.task_id
                live_rec.workspace_dir.mkdir(parents=True, exist_ok=True)
                task_scheduler_module.scheduler.submit(live_rec)
                check(
                    "fake 运行中任务已被调度器拉起",
                    wait_until(lambda: task_scheduler_module.scheduler.is_task_running("sim_uvicorn_live"), timeout=5.0),
                )

            server.should_exit = True  # 与 SIGTERM 走同一条 uvicorn 优雅停机路径
            server_thread.join(timeout=20.0)
            check("优雅停机后 uvicorn 线程退出", server_thread.is_alive() is False)
            check("优雅停机后调度 worker 线程退出", task_scheduler_module.scheduler.worker_alive is False)
            if live_rec is not None:
                check(
                    "优雅停机时取消并 kill 运行中任务（cancelled + 进程被 kill）",
                    live_rec.status == TaskStatus.CANCELLED
                    and live_procs[live_rec.task_id].kill_count == 1,
                    f"{live_rec.status.value} kills={live_procs.get(live_rec.task_id).kill_count if live_rec.task_id in live_procs else '?'}",
                )
                check(
                    "优雅停机取消的终态已落盘",
                    (read_state(live_rec.workspace_dir) or {}).get("status") == "cancelled",
                    str(read_state(live_rec.workspace_dir)),
                )
            # lifespan shutdown 已执行；再调一次 shutdown 验证幂等
            try:
                task_scheduler_module.scheduler.shutdown(cancel_running=True)
                check("lifespan 停机后重复 shutdown 幂等", True)
            except Exception as exc:  # noqa: BLE001
                check("lifespan 停机后重复 shutdown 幂等", False, repr(exc))
        finally:
            server.should_exit = True
            if server_thread.is_alive():
                server_thread.join(timeout=10.0)

    # -----------------------------------------------------------------------
    # [6] 真实 SIGTERM 优雅停机（POSIX 子进程；Windows 的 terminate 是硬杀）
    # -----------------------------------------------------------------------
    print("\n[6] 真实 SIGTERM 优雅停机（POSIX 子进程）")
    if not has_uvicorn:
        skip("真实 SIGTERM 优雅停机", "缺少 fastapi/uvicorn 依赖")
    elif os.name != "posix":
        skip(
            "真实 SIGTERM 优雅停机",
            "Windows 下 subprocess.terminate 是 TerminateProcess 硬杀，无法触发优雅停机；"
            "[5] 的 in-process uvicorn 已覆盖同一 Server.shutdown 路径，硬杀场景由启动恢复扫描兜底",
        )
    else:
        import signal

        sig_ws = work / "sigterm_ws"
        sig_ws.mkdir(parents=True, exist_ok=True)
        settings.workspace_root = sig_ws
        sig_dir = seed_task_dir(sig_ws, "sim_sigterm_interrupted", state_status="running")

        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        sig_port = sock.getsockname()[1]
        sock.close()

        sig_env = dict(os.environ)
        sig_env.update(
            {
                "AICM_MCP_HOST": "127.0.0.1",
                "AICM_MCP_PORT": str(sig_port),
                "AICM_MCP_DRY_RUN": "true",
                "AICM_MCP_SIM_TOOL_HOME": str(work / "aicm"),
                "AICM_MCP_WORKSPACE_ROOT": str(sig_ws),
                "AICM_MCP_LOG_LEVEL": "info",
            }
        )
        for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
            sig_env.pop(key, None)

        sig_log_path = work / "sigterm_server.log"
        sig_log = open(sig_log_path, "w", encoding="utf-8")
        sig_proc = subprocess.Popen(
            [sys.executable, "-m", "mcp_server"],
            cwd=str(REPO),
            env=sig_env,
            stdout=sig_log,
            stderr=subprocess.STDOUT,
        )
        try:
            up = False
            boot_deadline = time.time() + 30.0
            while time.time() < boot_deadline:
                if sig_proc.poll() is not None:
                    break
                if health_ok(sig_port):
                    up = True
                    break
                time.sleep(0.2)
            check("SIGTERM 子进程服务启动", up)
            if up:
                seeded = read_state(sig_dir) or {}
                check(
                    "子进程启动时执行恢复扫描（running → failed + 中断提示）",
                    seeded.get("status") == "failed"
                    and "服务重启，任务中断（进程已丢失）" in (seeded.get("message") or ""),
                    str(seeded),
                )
                sig_proc.send_signal(signal.SIGTERM)
                try:
                    returncode = sig_proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    returncode = None
                check("SIGTERM 后进程优雅退出（returncode=0）", returncode == 0, str(returncode))
                sig_log.flush()
                log_text = sig_log_path.read_text(encoding="utf-8", errors="replace")
                check(
                    "服务日志含启动恢复统计（日志统计）",
                    "task recovery scan" in log_text,
                    log_text[-600:],
                )
                check(
                    "服务日志含 lifespan 停机收尾（证明 SIGTERM 走到 lifespan shutdown）",
                    "shutdown complete" in log_text,
                    log_text[-600:],
                )
        finally:
            if sig_proc.poll() is None:
                sig_proc.kill()
                sig_proc.wait(timeout=10)
            sig_log.close()

finally:
    for name, value in _saved_settings.items():
        setattr(settings, name, value)
    task_store._tasks.clear()
    shutil.rmtree(work, ignore_errors=True)

print()
if _failures:
    print(f"  [FAIL] {len(_failures)} 项未通过:")
    for failure in _failures:
        print(f"    - {failure}")
    sys.exit(1)
print(
    "  [PASS] 重启恢复全部通过：task_state 迁移写入 / loader 六种恢复规则 / "
    "recover_interrupted_tasks 幂等 / shutdown cancel_running 停线程 / "
    "lifespan 启停顺序与真实 uvicorn 优雅停机。"
)
