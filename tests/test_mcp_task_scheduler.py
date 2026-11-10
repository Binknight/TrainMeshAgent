"""MCP Server 并发调度改造的验收测试（standalone，非 pytest）。

覆盖（对应 task-1 验收项）：

1. 并发上限：max=2 提交 5 个任务 → 最多 2 个 running，其余 queued 且 queue_position 正确
2. FIFO：槽位释放后按入队顺序拉起
3. 终态释放槽位（completed/failed/cancelled/timeout 后都能再拉起下一个）
4. 排队超时 → failed，且该任务从未拉起子进程
5. 执行超时 → kill 进程 + failed + 释放槽位
6. cancel queued：出队、不拉起、其余任务位置前移、后续 FIFO 正确
7. cancel running：kill 进程、置 cancelled、释放槽位、重复取消幂等
8. cancel 已终态：幂等返回当前状态
9. report_status 输出 queue_position 且**不读日志**；状态枚举覆盖 queued/cancelled
10. sync_logs 真正按字节增量（行偏移语义保持）；dry_run 返回 submitted 且不占槽位
11. 队列上限/背压（task-2）：队列满且无空闲槽位 → QueueFullError / JSON-RPC **-32004**
    （message 含「队列已满」），不静默入队、不残留记录；腾出槽位后可再提交；
    TaskNotReadyError 保持 -32003；经 dispatch + ASGI 端到端验证 JSON-RPC 错误体。

全程使用 fake process / 假时钟，不拉起真实 aicm 子进程。
Run: python tests/test_mcp_task_scheduler.py
"""

import asyncio
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# Windows 控制台默认代码页与 UTF-8 报告不一致
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

try:
    import pydantic  # noqa: F401
except ImportError:
    print("  [SKIP] 未安装 pydantic，跳过调度器测试 —— 请先执行:")
    print("         pip install -r mcp_server/requirements.txt")
    sys.exit(0)

try:
    import mcp_server.services.task_scheduler as task_scheduler_module
    from mcp_server.config import settings
    from mcp_server.errors import McpError, QueueFullError, TaskNotReadyError
    from mcp_server.schemas.common import (
        TERMINAL_TASK_STATUSES,
        CancelTaskInput,
        CancelTaskOutput,
        ReportStatusOutput,
        TaskStatus,
    )
    from mcp_server.services.task_scheduler import TaskScheduler
    from mcp_server.services.task_store import TaskRecord, task_store
    from mcp_server.tools import registry
except ImportError as exc:  # pragma: no cover
    print(f"  [SKIP] 无法导入 mcp_server 依赖（{exc}）—— 请先执行:")
    print("         pip install -r mcp_server/requirements.txt")
    sys.exit(0)

# ASGI 层验证（JSON-RPC 错误体）需要 fastapi；缺失时只跳过该子项，不影响其余用例。
try:
    from starlette.requests import Request

    from mcp_server import main as mcp_main

    _HAS_ASGI = True
except ImportError:  # pragma: no cover
    Request = None  # type: ignore[assignment,misc]
    mcp_main = None  # type: ignore[assignment]
    _HAS_ASGI = False

_failures: list[str] = []


def check(label: str, cond: bool, detail: str = "") -> None:
    print(("  [PASS] " if cond else "  [FAIL] ") + label + ("" if cond else f"   {detail}"))
    if not cond:
        _failures.append(label)


# ---------------------------------------------------------------------------
# 测试替身：fake process / 假时钟 / 可控 launcher
# ---------------------------------------------------------------------------


class FakeProcess:
    """最小 subprocess.Popen 替身：poll/kill 可控，不产生真实进程。"""

    def __init__(self) -> None:
        self.returncode = None
        self.kill_count = 0

    def poll(self):
        return self.returncode

    def kill(self) -> None:
        self.kill_count += 1
        if self.returncode is None:
            self.returncode = -9

    # 供 _terminate_process 的 POSIX 分支使用（Windows 下不会走到）
    pid = 424242

    def wait(self, timeout=None):  # noqa: ARG002
        return self.returncode


class FakeClock:
    """可推进的 time.monotonic 替身，让超时测试无需真实 sleep。"""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def make_record(task_id: str) -> TaskRecord:
    """调度器测试只关心生命周期字段，topology 不参与。"""
    return TaskRecord(task_id=task_id, topology=None, simulation_params={})


def make_scheduler(
    clock: FakeClock,
    *,
    max_concurrent: int = 2,
    max_queue_size: int = 20,
    task_timeout: float = 1000.0,
    queue_timeout: float = 1000.0,
):
    """构造手动驱动（不启动 worker 线程）的调度器 + fake launcher。"""
    launched: list[str] = []
    procs: dict[str, FakeProcess] = {}

    def launcher(record: TaskRecord) -> FakeProcess:
        process = FakeProcess()
        procs[record.task_id] = process
        launched.append(record.task_id)
        return process

    sched = TaskScheduler(
        max_concurrent_tasks=max_concurrent,
        max_queue_size=max_queue_size,
        task_timeout_sec=task_timeout,
        queue_timeout_sec=queue_timeout,
        launcher=launcher,
        clock=clock,
        auto_start=False,
    )
    return sched, launched, procs


def statuses(records: dict[str, TaskRecord]) -> dict[str, str]:
    return {name: record.status.value for name, record in records.items()}


def positions(records: dict[str, TaskRecord]) -> dict[str, int | None]:
    return {name: record.queue_position for name, record in records.items()}


print("=" * 78)
print("  MCP Server 并发调度：全局上限 / FIFO / 超时 / 取消 / 增量日志")
print("=" * 78)

# ---------------------------------------------------------------------------
# [1] 并发上限 + queue_position + FIFO + 终态释放槽位
# ---------------------------------------------------------------------------
print("\n[1] 并发上限 / FIFO / queue_position / 终态释放槽位")
clock = FakeClock()
sched, launched, procs = make_scheduler(clock, max_concurrent=2)
records = {name: make_record(name) for name in "ABCDE"}
for record in records.values():
    sched.submit(record)

check("提交后全部处于 queued", all(r.status == TaskStatus.QUEUED for r in records.values()))
check("FIFO 队列顺序 = 提交顺序", sched.queued_task_ids() == list("ABCDE"), str(sched.queued_task_ids()))
check(
    "queued 任务 queue_position 依次为 1..5",
    positions(records) == {name: i + 1 for i, name in enumerate("ABCDE")},
    str(positions(records)),
)

sched.tick()
check("槽位 2 → 只拉起 2 个 running", sched.running_task_ids() == ["A", "B"], str(sched.running_task_ids()))
check("launcher 按 FIFO 拉起", launched == ["A", "B"], str(launched))
check(
    "running 任务 queue_position 为 None，其余依次 1..3",
    positions(records)
    == {"A": None, "B": None, "C": 1, "D": 2, "E": 3},
    str(positions(records)),
)

# 多次 tick 不允许突破上限（进程都还活着）
for _ in range(5):
    sched.tick()
check("空转多次 tick 不突破并发上限", len(sched.running_task_ids()) == 2 and launched == ["A", "B"], str(launched))

# A 结束 → 释放槽位 → C 立即补位
procs["A"].returncode = 0
sched.tick()
check("A 退出后 completed 且 progress=100", records["A"].status == TaskStatus.COMPLETED and records["A"].progress == 100.0)
check("终态释放槽位后 C 补位", sched.running_task_ids() == ["B", "C"], str(sched.running_task_ids()))
check("队列位置前移：D=1, E=2", positions(records)["D"] == 1 and positions(records)["E"] == 2, str(positions(records)))

for name in ["B", "C", "D", "E"]:
    procs[name].returncode = 0
    sched.tick()
check("全部任务按 FIFO 依次被拉起", launched == ["A", "B", "C", "D", "E"], str(launched))
check(
    "全部完成后 running/queue 清空、状态为 completed",
    sched.running_task_ids() == []
    and sched.queued_task_ids() == []
    and all(r.status == TaskStatus.COMPLETED for r in records.values()),
    str(statuses(records)),
)

# ---------------------------------------------------------------------------
# [2] 排队超时 → failed，且从未拉起子进程
# ---------------------------------------------------------------------------
print("\n[2] 排队超时")
clock = FakeClock()
sched, launched, procs = make_scheduler(clock, max_concurrent=1, queue_timeout=100, task_timeout=1000)
records = {name: make_record(name) for name in ("F", "G")}
for record in records.values():
    sched.submit(record)
sched.tick()
check("F 占用唯一槽位、G 排队", sched.running_task_ids() == ["F"] and records["G"].queue_position == 1)

clock.advance(100)
sched.tick()
check("排队超时 → failed", records["G"].status == TaskStatus.FAILED, records["G"].status.value)
check("失败原因包含 queue timeout", "queue timeout" in (records["G"].message or ""), str(records["G"].message))
check("排队超时任务从未拉起子进程", launched == ["F"] and "G" not in procs, str(launched))
check("超时任务已出队", sched.queued_task_ids() == [], str(sched.queued_task_ids()))

procs["F"].returncode = 0
sched.tick()
check("槽位释放后也不会拉起已超时的 G", records["F"].status == TaskStatus.COMPLETED and launched == ["F"], str(launched))

# ---------------------------------------------------------------------------
# [3] 执行超时 → kill 进程 + failed + 释放槽位
# ---------------------------------------------------------------------------
print("\n[3] 执行超时")
clock = FakeClock()
sched, launched, procs = make_scheduler(clock, max_concurrent=1, task_timeout=100, queue_timeout=1000)
records = {name: make_record(name) for name in ("H", "I")}
for record in records.values():
    sched.submit(record)
sched.tick()
check("H 运行中、I 排队", sched.running_task_ids() == ["H"] and records["I"].queue_position == 1)

clock.advance(100)
sched.tick()
check("执行超时 → failed", records["H"].status == TaskStatus.FAILED, records["H"].status.value)
check("执行超时 → 进程被 kill", procs["H"].kill_count == 1, str(procs["H"].kill_count))
check("失败原因包含 task timeout", "task timeout" in (records["H"].message or ""), str(records["H"].message))
check("执行超时释放槽位后 I 补位", sched.running_task_ids() == ["I"] and launched == ["H", "I"], str(launched))

# ---------------------------------------------------------------------------
# [4] cancel queued：出队、不拉起、位置前移、FIFO 不受污染
# ---------------------------------------------------------------------------
print("\n[4] cancel queued")
clock = FakeClock()
sched, launched, procs = make_scheduler(clock, max_concurrent=1)
records = {name: make_record(name) for name in ("J", "K", "L")}
for record in records.values():
    sched.submit(record)
sched.tick()
check("J 运行、K/L 排队且位置 1/2", sched.running_task_ids() == ["J"] and positions(records)["K"] == 1 and positions(records)["L"] == 2)

cancelled = sched.cancel(records["K"])
check("cancel queued 返回 cancelled", cancelled == TaskStatus.CANCELLED and records["K"].status == TaskStatus.CANCELLED)
check("cancel queued 不启动子进程", "K" not in procs and launched == ["J"], str(launched))
check("出队后 L 前移到位置 1", sched.queued_task_ids() == ["L"] and records["L"].queue_position == 1, str(sched.queued_task_ids()))

procs["J"].returncode = 0
sched.tick()
check(
    "取消的任务不阻塞 FIFO：J 完成后拉起 L",
    records["J"].status == TaskStatus.COMPLETED
    and sched.running_task_ids() == ["L"]
    and launched == ["J", "L"],
    str(launched),
)

# ---------------------------------------------------------------------------
# [5] cancel running：kill 进程、cancelled、释放槽位、幂等
# ---------------------------------------------------------------------------
print("\n[5] cancel running")

clock = FakeClock()
sched, launched, procs = make_scheduler(clock, max_concurrent=1)
records = {name: make_record(name) for name in ("M", "N")}
for record in records.values():
    sched.submit(record)
sched.tick()
check("M 运行、N 排队", sched.running_task_ids() == ["M"] and records["N"].queue_position == 1)

cancelled = sched.cancel(records["M"])
check("cancel running 返回 cancelled", cancelled == TaskStatus.CANCELLED)
check("cancel running kill 了子进程", procs["M"].kill_count == 1, str(procs["M"].kill_count))
check("cancel running 释放槽位", sched.running_task_ids() == [])
check("排队任务 N 前移为位置 1", records["N"].queue_position == 1, str(records["N"].queue_position))

sched.tick()
check("取消后槽位可复用：N 被拉起", sched.running_task_ids() == ["N"] and launched == ["M", "N"], str(launched))

again = sched.cancel(records["M"])
check("重复 cancel 幂等（仍 cancelled、不再 kill）", again == TaskStatus.CANCELLED and procs["M"].kill_count == 1)

# ---------------------------------------------------------------------------
# [6] cancel 已终态：幂等返回当前状态
# ---------------------------------------------------------------------------
print("\n[6] cancel 已终态（completed）")
procs["N"].returncode = 0
sched.tick()
check("N completed", records["N"].status == TaskStatus.COMPLETED)
result = sched.cancel(records["N"])
check("已完成任务 cancel 幂等返回 completed", result == TaskStatus.COMPLETED and procs["N"].kill_count == 0)

# ---------------------------------------------------------------------------
# [7] 状态枚举 / 输出模型 / tool 注册契约
# ---------------------------------------------------------------------------
print("\n[7] 状态枚举、输出模型与 tool 注册")
expected_statuses = {"submitted", "queued", "running", "completed", "failed", "error", "cancelled"}
check("TaskStatus 覆盖 queued/cancelled 且无多余值", {s.value for s in TaskStatus} == expected_statuses, str(sorted(s.value for s in TaskStatus)))
check(
    "终态集合 = completed/failed/error/cancelled",
    {s.value for s in TERMINAL_TASK_STATUSES} == {"completed", "failed", "error", "cancelled"},
    str(sorted(s.value for s in TERMINAL_TASK_STATUSES)),
)
check(
    "ReportStatusOutput 接受 queue_position",
    ReportStatusOutput(status=TaskStatus.QUEUED, queue_position=2).queue_position == 2,
)
try:
    ReportStatusOutput(status=TaskStatus.QUEUED, queue_position=0)
    check("queue_position 必须 >=1", False, "未报错")
except Exception:
    check("queue_position 必须 >=1", True)
check(
    "CancelTask 入/出参模型",
    CancelTaskInput(task_id="sim_x").task_id == "sim_x"
    and CancelTaskOutput(task_id="sim_x", status=TaskStatus.CANCELLED).status == TaskStatus.CANCELLED,
)

tool_names = [tool["name"] for tool in registry.list_tools()]
check("registry 注册 10 个 tool", len(tool_names) == 10, str(tool_names))
check("registry 含 cancel_task", "cancel_task" in tool_names, str(tool_names))
cancel_schema = next(t for t in registry.list_tools() if t["name"] == "cancel_task")
check("cancel_task inputSchema 要求 task_id", cancel_schema["inputSchema"].get("required") == ["task_id"], str(cancel_schema["inputSchema"]))

# ---------------------------------------------------------------------------
# [8]/[9]/[10] handler 层：execute_task 排队、report_status 不读日志、sync_logs 增量、dry_run
# ---------------------------------------------------------------------------
print("\n[8] handler 层：execute_task 立即 queued / report_status / sync_logs / dry_run")

work = REPO / ".tmp" / "mcp_task_scheduler"
shutil.rmtree(work, ignore_errors=True)
(work / "aicm" / "examples").mkdir(parents=True)
(work / "ws").mkdir(parents=True)
(work / "aicm" / "run.py").write_text("# fake simulation entry, never executed\n", encoding="utf-8")
(work / "aicm" / "examples" / "time_args.json").write_text("{}", encoding="utf-8")

_saved_settings = {
    name: getattr(settings, name)
    for name in ("sim_tool_home", "workspace_root", "dry_run", "conda_env")
}
_original_scheduler = task_scheduler_module.scheduler

clock = FakeClock()
manual, manual_launched, manual_procs = make_scheduler(clock, max_concurrent=1)

try:
    settings.sim_tool_home = work / "aicm"
    settings.workspace_root = work / "ws"
    settings.dry_run = False
    task_scheduler_module.scheduler = manual

    payload = {
        "topology": {
            "name": "原始组网",
            "device_type": "A3",
            "dp_size": 2,
            "tp_size": 2,
            "pp_size": 2,
            "total_nodes": 8,
            "num_layers": 2,
            "hidden_dim": 128,
            "num_heads": 8,
        },
        "simulation_params": {},
    }
    result = registry.dispatch_tool("execute_task", payload)
    task_id = result.get("task_id", "")
    record = task_store.get(task_id)
    check("execute_task 真实模式立即返回 queued", result.get("status") == "queued", str(result))
    check(
        "返回时子进程尚未拉起（不阻塞等子进程）",
        manual_launched == [] and record is not None and record.status == TaskStatus.QUEUED,
        str(manual_launched),
    )
    check("任务已入队且 queue_position=1", manual.queued_task_ids() == [task_id] and record.queue_position == 1)

    status = registry.dispatch_tool("report_status", {"task_id": task_id})
    check(
        "report_status 反映 queued + queue_position",
        status.get("status") == "queued" and status.get("queue_position") == 1,
        str(status),
    )

    manual.tick()
    status = registry.dispatch_tool("report_status", {"task_id": task_id})
    check(
        "调度器拿到槽位后 report_status 变为 running 且 queue_position=None",
        status.get("status") == "running" and status.get("queue_position") is None,
        str(status),
    )
    check("running 时子进程已由调度器拉起", manual_launched == [task_id], str(manual_launched))

    # 模拟仿真进程写日志；此时 report_status 不得读日志
    log_path = record.log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("alpha\nbeta\n", encoding="utf-8")
    logs_before = len(record.logs)
    byte_before = record.log_byte_offset
    registry.dispatch_tool("report_status", {"task_id": task_id})
    check(
        "report_status 不读日志（logs/字节游标不变）",
        len(record.logs) == logs_before and record.log_byte_offset == byte_before,
        f"logs {logs_before}->{len(record.logs)}, offset {byte_before}->{record.log_byte_offset}",
    )

    sync1 = registry.dispatch_tool("sync_logs", {"task_id": task_id, "offset": 0})
    check(
        "sync_logs 读出完整行且 next_offset 为行偏移",
        sync1.get("lines", [])[-2:] == ["alpha", "beta"]
        and sync1.get("next_offset") == len(record.logs),
        str(sync1),
    )
    check(
        "sync_logs 字节游标推进到文件末尾",
        record.log_byte_offset == log_path.stat().st_size,
        f"{record.log_byte_offset} != {log_path.stat().st_size}",
    )

    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write("gamma\n")
    sync2 = registry.dispatch_tool(
        "sync_logs", {"task_id": task_id, "offset": sync1["next_offset"]}
    )
    check(
        "增量 sync_logs 只返回新增行",
        sync2.get("lines") == ["gamma"] and sync2.get("next_offset") == sync1["next_offset"] + 1,
        str(sync2),
    )

    sync3 = registry.dispatch_tool(
        "sync_logs", {"task_id": task_id, "offset": sync2["next_offset"]}
    )
    check("无新内容时不重复返回（不再整文件重读+去重）", sync3.get("lines") == [], str(sync3))

    # 半行（进程未写完换行）不消费，补上换行后才返回
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write("delta")
    sync4 = registry.dispatch_tool(
        "sync_logs", {"task_id": task_id, "offset": sync3["next_offset"]}
    )
    check("未写完的半行不返回", sync4.get("lines") == [], str(sync4))
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write("\n")
    sync5 = registry.dispatch_tool(
        "sync_logs", {"task_id": task_id, "offset": sync4["next_offset"]}
    )
    check("半行补齐后才返回", sync5.get("lines") == ["delta"], str(sync5))

    # 进程退出 → completed，cancel 幂等
    manual_procs[task_id].returncode = 0
    manual.tick()
    status = registry.dispatch_tool("report_status", {"task_id": task_id})
    check(
        "进程退出后 report_status=completed 且 queue_position=None",
        status.get("status") == "completed" and status.get("queue_position") is None,
        str(status),
    )
    cancelled = registry.dispatch_tool("cancel_task", {"task_id": task_id})
    check(
        "cancel 已终态任务幂等返回 completed",
        cancelled == {"task_id": task_id, "status": "completed"},
        str(cancelled),
    )

    # cancel 不存在的任务 → 既有 -32001 语义
    try:
        registry.dispatch_tool("cancel_task", {"task_id": "sim_does_not_exist"})
        check("cancel 未知 task_id -> -32001", False, "未报错")
    except McpError as exc:
        check("cancel 未知 task_id -> -32001", exc.code == -32001, str(exc.code))

    # dry_run：返回 submitted，不排队、不占槽位、不拉进程
    settings.dry_run = True
    launched_before = len(manual_launched)
    result = registry.dispatch_tool("execute_task", payload)
    dry_task_id = result.get("task_id", "")
    check("dry_run execute_task 返回 submitted", result.get("status") == "submitted", str(result))
    check(
        "dry_run 不占调度槽位、不拉子进程",
        dry_task_id not in manual.queued_task_ids()
        and dry_task_id not in manual.running_task_ids()
        and len(manual_launched) == launched_before,
        str(manual_launched),
    )
    status = registry.dispatch_tool("report_status", {"task_id": dry_task_id})
    check(
        "dry_run 任务 report_status=submitted、queue_position=None",
        status.get("status") == "submitted" and status.get("queue_position") is None,
        str(status),
    )
    cancelled = registry.dispatch_tool("cancel_task", {"task_id": dry_task_id})
    check("dry_run(submitted) 任务 cancel 幂等不改变状态", cancelled.get("status") == "submitted", str(cancelled))
finally:
    for name, value in _saved_settings.items():
        setattr(settings, name, value)
    task_scheduler_module.scheduler = _original_scheduler
    manual.shutdown()
    shutil.rmtree(work, ignore_errors=True)

# ---------------------------------------------------------------------------
# [9] 队列上限 / 背压（task-2）：QueueFullError -32004，不静默入队
# ---------------------------------------------------------------------------
print("\n[9] 队列上限/背压：QueueFull(-32004) / 不静默入队 / 腾出槽位后可再提交")

# 9a. 调度器层：队列满 **且** 无空闲槽位 → 抛 QueueFullError（-32004）
clock = FakeClock()
sched, launched, procs = make_scheduler(clock, max_concurrent=1, max_queue_size=2)
records = {name: make_record(name) for name in ("Q1", "Q2", "Q3", "Q4", "Q5")}
sched.submit(records["Q1"])
sched.tick()
check("Q1 占满唯一执行槽位", sched.running_task_ids() == ["Q1"] and sched.queue_size() == 0)
sched.submit(records["Q2"])
sched.submit(records["Q3"])
check(
    "队列达到 max_queue_size=2",
    sched.queued_task_ids() == ["Q2", "Q3"] and sched.queue_size() == 2,
    str(sched.queued_task_ids()),
)
try:
    sched.submit(records["Q4"])
    check("队列满且无空闲槽位 → 抛 QueueFullError", False, "未抛异常")
except QueueFullError as exc:
    check("队列满且无空闲槽位 → 抛 QueueFullError", True)
    check("QueueFullError JSON-RPC code = -32004（与 -32003 未就绪区分）", exc.code == -32004, str(exc.code))
    check("QueueFullError message 含「队列已满」", "队列已满" in exc.message, exc.message)
    check(
        "QueueFullError data 带 reason/上限",
        exc.data.get("reason") == "queue_full"
        and exc.data.get("max_queue_size") == 2
        and exc.data.get("queue_size") == 2,
        str(exc.data),
    )
check(
    "拒绝后不静默入队、不残留记录",
    sched.queued_task_ids() == ["Q2", "Q3"]
    and records["Q4"].status != TaskStatus.QUEUED
    and sched.queue_position("Q4") is None,
    f"{sched.queued_task_ids()} / {records['Q4'].status.value}",
)
check(
    "既有 TaskNotReadyError 仍为 -32003（不被队列满占用）",
    TaskNotReadyError("t", "running").code == -32003,
)

# 9b. 契约是「队列满 **且** 无空闲槽位」才拒绝：有空闲槽位时允许短暂超过队列长度
clock = FakeClock()
sched2, launched2, _procs2 = make_scheduler(clock, max_concurrent=2, max_queue_size=1)
r1, r2 = make_record("R1"), make_record("R2")
sched2.submit(r1)
try:
    sched2.submit(r2)  # len(queue)==max_queue_size，但还有空闲槽位 → 允许
    check("有空闲槽位时不受队列上限拒绝（避免刚空出槽位却拒任务）", True)
except QueueFullError as exc:
    check("有空闲槽位时不受队列上限拒绝（避免刚空出槽位却拒任务）", False, str(exc))
sched2.tick()
check(
    "随后 2 个任务立即拿到槽位、队列清空",
    sched2.running_task_ids() == ["R1", "R2"] and sched2.queue_size() == 0,
    str(sched2.running_task_ids()),
)

# 9c. 腾出槽位/队列后可以继续提交
procs["Q1"].returncode = 0
sched.tick()
check(
    "Q1 终态释放槽位后 Q2 补位、队列只剩 Q3",
    sched.running_task_ids() == ["Q2"] and sched.queued_task_ids() == ["Q3"],
    str(sched.queued_task_ids()),
)
sched.submit(records["Q5"])
check("队列有空间时可再提交 Q5", sched.queued_task_ids() == ["Q3", "Q5"], str(sched.queued_task_ids()))
try:
    sched.submit(make_record("Q6"))
    check("再次填满后继续拒绝（不会无限堆积）", False, "未抛异常")
except QueueFullError:
    check("再次填满后继续拒绝（不会无限堆积）", sched.queued_task_ids() == ["Q3", "Q5"])
procs["Q2"].returncode = 0
sched.tick()
sched.submit(records["Q4"])
check(
    "运行中任务再完成一个后 Q4 可被接受",
    records["Q4"].status == TaskStatus.QUEUED and sched.queued_task_ids() == ["Q5", "Q4"],
    str(sched.queued_task_ids()),
)

# 9d. execute_task 经 dispatch 原样抛出 -32004；ASGI 端点转成 JSON-RPC error
work = REPO / ".tmp" / "mcp_task_scheduler_queue"
shutil.rmtree(work, ignore_errors=True)
(work / "aicm" / "examples").mkdir(parents=True)
(work / "ws").mkdir(parents=True)
(work / "aicm" / "run.py").write_text("# fake simulation entry, never executed\n", encoding="utf-8")
(work / "aicm" / "examples" / "time_args.json").write_text("{}", encoding="utf-8")

_saved_settings = {
    name: getattr(settings, name)
    for name in ("sim_tool_home", "workspace_root", "dry_run", "conda_env")
}
_original_scheduler = task_scheduler_module.scheduler
full = TaskScheduler(
    max_concurrent_tasks=1,
    max_queue_size=1,
    launcher=lambda record: FakeProcess(),
    clock=FakeClock(),
    auto_start=False,
)


def asgi_mcp_call(body: dict) -> dict:
    """不依赖 httpx/TestClient 的 ASGI 调用：直接驱动 FastAPI 端点。"""
    raw = json.dumps(body).encode()

    async def receive():
        return {"type": "http.request", "body": raw, "more_body": False}

    async def send(_message):
        return None

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/mcp",
        "raw_path": b"/mcp",
        "query_string": b"",
        "headers": [(b"content-type", b"application/json")],
        "client": ("127.0.0.1", 1234),
        "server": ("127.0.0.1", 9000),
    }
    response = asyncio.run(mcp_main.mcp_endpoint(Request(scope, receive=receive, send=send)))
    return json.loads(response.body)


def count_queue_full_task_states(ws_root: Path) -> int:
    """统计磁盘上被背压拒绝、落成 failed 的孤儿任务（message 含队列已满）。"""
    count = 0
    for child in ws_root.glob("sim_*"):
        state_path = child / "task_state.json"
        if not state_path.is_file():
            continue
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("status") == "failed" and "队列已满" in (state.get("message") or ""):
            count += 1
    return count


try:
    settings.sim_tool_home = work / "aicm"
    settings.workspace_root = work / "ws"
    settings.dry_run = False
    # 把调度器灌满：1 个运行中 + 1 个排队（max_concurrent=1, max_queue_size=1）
    occupy_run = make_record("occupy_run")
    full._records[occupy_run.task_id] = occupy_run
    occupy_run.status = TaskStatus.RUNNING
    full._running[occupy_run.task_id] = occupy_run
    full.submit(make_record("occupy_q"))
    task_scheduler_module.scheduler = full

    queue_payload = {
        "topology": {
            "name": "原始组网",
            "device_type": "A3",
            "dp_size": 1,
            "tp_size": 1,
            "pp_size": 1,
            "total_nodes": 1,
            "num_layers": 1,
            "hidden_dim": 8,
            "num_heads": 1,
        },
        "simulation_params": {},
    }

    try:
        registry.dispatch_tool("execute_task", queue_payload)
        check("dispatch(execute_task) 队列满 → QueueFullError", False, "未抛异常")
    except QueueFullError as exc:
        check("dispatch(execute_task) 队列满 → QueueFullError", True)
        check("  code=-32004", exc.code == -32004, str(exc.code))
        check("  message 含「队列已满」", "队列已满" in exc.message, exc.message)
    except McpError as exc:  # 兜底：错误类型/码不符
        check("dispatch(execute_task) 队列满 → QueueFullError", False, f"got {type(exc).__name__} code={exc.code}")
    check(
        "背压拒绝后不把任务留在队列里",
        full.queued_task_ids() == ["occupy_q"] and full.queue_size() == 1,
        str(full.queued_task_ids()),
    )
    check("被拒绝任务在磁盘上落成 failed 终态（重启后不会永远 submitted）", count_queue_full_task_states(work / "ws") == 1)

    if _HAS_ASGI:
        rpc = asgi_mcp_call(
            {
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {"name": "execute_task", "arguments": queue_payload},
                "id": 42,
            }
        )
        check("ASGI: JSON-RPC error code = -32004", rpc.get("error", {}).get("code") == -32004, str(rpc.get("error")))
        check("ASGI: error.message 含「队列已满」", "队列已满" in (rpc.get("error", {}).get("message") or ""), str(rpc.get("error")))
        check(
            "ASGI: 错误响应仍带 result.error/message（§13）",
            rpc.get("result", {}).get("error") is True
            and "队列已满" in (rpc.get("result", {}).get("message") or ""),
            str(rpc.get("result")),
        )
        check("ASGI: 回显请求 id", rpc.get("id") == 42, str(rpc.get("id")))
        check("ASGI 拒绝后队列仍不残留", full.queued_task_ids() == ["occupy_q"], str(full.queued_task_ids()))
        check("两次拒绝任务都落成 failed 终态", count_queue_full_task_states(work / "ws") == 2)
    else:
        print("  [SKIP] 未安装 fastapi，跳过 ASGI JSON-RPC 错误体校验")

    # 腾出槽位后可再提交（端到端：dispatch 返回 queued 而不是报错）
    full._running.pop("occupy_run", None)
    full.tick()
    result = registry.dispatch_tool("execute_task", queue_payload)
    check(
        "腾出执行槽位后 execute_task 恢复 accepted(queued)",
        result.get("status") == "queued",
        str(result),
    )
finally:
    for name, value in _saved_settings.items():
        setattr(settings, name, value)
    task_scheduler_module.scheduler = _original_scheduler
    full.shutdown()
    shutil.rmtree(work, ignore_errors=True)

print()
if _failures:
    print(f"  [FAIL] {len(_failures)} 项未通过:")
    for failure in _failures:
        print(f"    - {failure}")
    sys.exit(1)
print(
    "  [PASS] 并发调度全部通过：全局上限 / FIFO / queue_position / 排队与执行超时 / "
    "cancel 幂等 / report_status 不读日志 / sync_logs 字节增量 / dry_run submitted / "
    "队列上限背压 -32004。"
)
