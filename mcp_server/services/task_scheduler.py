"""任务调度器：全局并发上限 + FIFO 排队 + 排队/执行超时 + 取消。

改造前的 MCP Server 在 ``execute_task`` 里同步 Popen 子进程，等于**无并发上限**；
并行仿真互相争抢 CPU/内存，出现「两个任务一起变慢」。本模块把任务生命周期从
调用路径上摘出来：

- ``submit()``：只入队（execute_task 立即返回 ``queued``），由后台 worker 在
  空闲槽位按 FIFO 顺序拉起；
- ``tick()``：调度状态机的一步 —— 收尾已退出进程、处理排队/执行超时、补位启动；
- ``cancel()``：排队中只出队（绝不拉起子进程）；运行中 kill 进程组；已终态幂等。

可测试性：``launcher`` / ``finalizer`` / ``clock`` 均可注入，测试用 fake process +
假时钟即可覆盖并发上限、FIFO、超时与取消，不依赖真实 aicm 子进程。

可靠性（task-2）：

- **背压**：``max_queue_size`` 限制排队长度；队列满且无空闲槽位时 ``submit()`` 抛
  ``QueueFullError``（JSON-RPC -32004，message 含「队列已满」），不静默入队、不入队；
- **状态落盘**：queued/running/terminal 几个迁移点 best-effort 写
  ``workspace/<task_id>/task_state.json``，供服务重启后恢复（见 task_state 模块）；
- **优雅停机**：``shutdown(cancel_running=True)`` 取消运行中任务（kill 进程组）、
  标记终态、停 worker 线程，可重复调用（幂等）。
"""

import logging
import os
import signal
import threading
import time
from collections import deque
from typing import Callable, Deque, Optional

from mcp_server.config import settings
from mcp_server.errors import QueueFullError
from mcp_server.schemas.common import TaskStatus
from mcp_server.services import simulation_runner, task_state
from mcp_server.services.task_store import TaskRecord

logger = logging.getLogger(__name__)

# launcher(record) -> subprocess.Popen；finalizer(record, returncode) -> None
Launcher = Callable[[TaskRecord], object]
Finalizer = Callable[[TaskRecord, int], None]


def _terminate_process(process: object) -> None:
    """尽力杀死子进程。

    POSIX：优先 ``killpg`` 杀掉整个进程组（Popen 时需 ``start_new_session=True``，
    见 simulation_runner.launch_task），否则 run.py 拉起的子进程会变成孤儿继续跑；
    取不到进程组（已退出/权限）时回退到 ``process.kill()``。
    Windows 没有进程组语义，直接 ``process.kill()``。
    """
    if process is None:
        return
    try:
        if os.name == "posix":
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)  # type: ignore[attr-defined]
                return
            except (AttributeError, ProcessLookupError, PermissionError, OSError):
                pass
        process.kill()  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 — 杀进程失败不应影响状态机
        logger.exception("failed to terminate simulation process")


class TaskScheduler:
    """FIFO + 全局并发上限的任务调度器（线程安全）。"""

    def __init__(
        self,
        *,
        # 默认 2 = 1 对仿真（原始 + 等效两个子进程）；压测确认有余量后再调大
        max_concurrent_tasks: int = 2,
        # 排队上限（背压）：<=0 表示不限（见 config.max_queue_size 的说明）
        max_queue_size: int = 20,
        task_timeout_sec: float = 3600,
        queue_timeout_sec: float = 900,
        poll_interval_sec: float = 0.2,
        launcher: Optional[Launcher] = None,
        finalizer: Optional[Finalizer] = None,
        clock: Callable[[], float] = time.monotonic,
        auto_start: bool = True,
    ) -> None:
        self.max_concurrent_tasks = max(1, int(max_concurrent_tasks))
        # 0/负数 = 不限制排队长度（便于特殊部署/测试关闭背压）
        self.max_queue_size = int(max_queue_size)
        # <=0 表示关闭对应超时（便于只关心其中一种的部署/测试）
        self.task_timeout_sec = float(task_timeout_sec)
        self.queue_timeout_sec = float(queue_timeout_sec)
        self.poll_interval_sec = max(0.01, float(poll_interval_sec))
        self._launcher: Launcher = launcher or simulation_runner.launch_task
        self._finalizer: Finalizer = finalizer or simulation_runner.finalize_task
        self._clock = clock
        self._auto_start = auto_start

        self._lock = threading.RLock()
        self._wakeup = threading.Event()
        self._queue: Deque[str] = deque()
        self._records: dict[str, TaskRecord] = {}
        self._running: dict[str, TaskRecord] = {}
        self._thread: Optional[threading.Thread] = None
        self._closed = False

    # ------------------------------------------------------------------
    # worker 线程生命周期
    # ------------------------------------------------------------------
    def start(self) -> None:
        """启动后台 worker（幂等）。"""
        with self._lock:
            if self._closed:
                return
            if self._thread is not None and self._thread.is_alive():
                return
            self._thread = threading.Thread(
                target=self._run_forever,
                name="mcp-task-scheduler",
                daemon=True,
            )
            self._thread.start()

    def shutdown(self, *, cancel_running: bool = False) -> None:
        """停止 worker 线程（幂等）；``cancel_running=True`` 时先取消所有运行中任务。

        停机顺序由 FastAPI lifespan 保证：``cleanup_worker.stop()`` →
        ``scheduler.shutdown(cancel_running=True)``。取消运行中任务会 kill 整个
        进程组并写终态（cancelled）落盘，避免重启后留下「永远 running」的任务。
        重复调用安全：running 集合已空、线程已退出时立即返回。
        """
        with self._lock:
            self._closed = True
            to_cancel = list(self._running.values()) if cancel_running else []
            thread = self._thread
        for record in to_cancel:
            try:
                self.cancel(record)
            except Exception:  # noqa: BLE001 — 单个任务取消失败不能阻断停机
                logger.exception("failed to cancel task %s on shutdown", record.task_id)
        self._wakeup.set()
        if (
            thread is not None
            and thread is not threading.current_thread()
            and thread.is_alive()
        ):
            thread.join(timeout=5.0)
        with self._lock:
            if thread is not None and not thread.is_alive():
                self._thread = None

    def _run_forever(self) -> None:
        while True:
            with self._lock:
                if self._closed:
                    return
            self._wakeup.clear()
            try:
                self.tick()
            except Exception:  # noqa: BLE001 — worker 不能因单次调度异常退出
                logger.exception("task scheduler tick failed")
            self._wakeup.wait(self.poll_interval_sec)

    # ------------------------------------------------------------------
    # 入队 / 取消 / 查询
    # ------------------------------------------------------------------
    def submit(self, record: TaskRecord) -> None:
        """把一个已 prepare 的任务放入 FIFO 队列（不阻塞、不拉子进程）。

        背压：``max_queue_size > 0``、排队数已达上限且没有空闲执行槽位时抛
        ``QueueFullError``（JSON-RPC -32004）。此时**不登记、不入队**，调用方
        （execute_task）可安全重试；任务目录/task_meta 由 prepare 留下，不影响
        后续重试（新任务用新 task_id）。
        """
        with self._lock:
            if self._closed:
                raise RuntimeError("task scheduler is shut down")
            if self._is_full_locked():
                raise QueueFullError(
                    queue_size=len(self._queue),
                    max_queue_size=self.max_queue_size,
                )
            self._records[record.task_id] = record
            record.status = TaskStatus.QUEUED
            record.queued_at = self._clock()
            record.started_at = None
            record.finished_at = None
            record.progress = 0.0
            record.message = "queued, waiting for a free execution slot"
            record.append_log("[queue] task queued")
            self._queue.append(record.task_id)
            self._refresh_queue_positions_locked()
            task_state.write_task_state(
                record,
                status=TaskStatus.QUEUED,
                message=record.message,
                started_at=None,
                finished_at=None,
            )
            self._wakeup.set()
        if self._auto_start:
            self.start()

    def cancel(self, record: TaskRecord) -> TaskStatus:
        """取消任务；返回取消后的状态（已终态时幂等返回当前状态）。"""
        task_id = record.task_id
        with self._lock:
            self._records.setdefault(task_id, record)

            # 1) 排队中：出队即可，**绝不拉起子进程**
            if record.status == TaskStatus.QUEUED or task_id in self._queue:
                self._remove_from_queue_locked(task_id)
                record.status = TaskStatus.CANCELLED
                record.message = "cancelled while queued (simulation never started)"
                record.queue_position = None
                record.finished_at = self._clock()
                record.append_log("[cancel] cancelled while queued")
                self._refresh_queue_positions_locked()
                self._persist_terminal_locked(record)
                return record.status

            # 2) 运行中：kill 进程组并释放槽位
            if task_id in self._running or record.process is not None:
                self._terminate_locked(record)
                record.status = TaskStatus.CANCELLED
                record.message = "cancelled while running (process group killed)"
                record.queue_position = None
                record.finished_at = self._clock()
                record.append_log("[cancel] cancelled while running")
                self._persist_terminal_locked(record)
                return record.status

            # 3) 已终态 / dry_run(submitted) / 本调度器不认识的任务：幂等
            return record.status

    def queue_position(self, task_id: str) -> Optional[int]:
        with self._lock:
            record = self._records.get(task_id)
            if record is None or record.status != TaskStatus.QUEUED:
                return None
            return record.queue_position

    def queued_task_ids(self) -> list[str]:
        with self._lock:
            return list(self._queue)

    def queue_size(self) -> int:
        with self._lock:
            return len(self._queue)

    @property
    def worker_alive(self) -> bool:
        """worker 线程是否仍在运行（停机测试/健康检查用）。"""
        thread = self._thread
        return thread is not None and thread.is_alive()

    def _is_full_locked(self) -> bool:
        """队列满且无空闲槽位 → 拒绝新任务（调用方需持有 self._lock）。

        契约定义：``len(queue) >= max_queue_size 且无空闲槽位``。有空闲槽位时
        允许短暂超队列上限（worker 会立即消费），避免「刚有空位却把任务拒掉」。
        """
        if self.max_queue_size <= 0:
            return False
        if len(self._queue) < self.max_queue_size:
            return False
        return len(self._running) >= self.max_concurrent_tasks

    def running_task_ids(self) -> list[str]:
        with self._lock:
            return list(self._running)

    def is_task_running(self, task_id: str) -> bool:
        with self._lock:
            return task_id in self._running

    # ------------------------------------------------------------------
    # 调度状态机的一步
    # ------------------------------------------------------------------
    def tick(self) -> None:
        """推进一步调度：收尾/超时/取消生效，然后在空闲槽位上按 FIFO 拉起任务。"""
        now = self._clock()
        with self._lock:
            if self._closed:
                return

            # 1) 运行中的任务：执行超时（kill）或进程已退出（收尾释放槽位）
            for task_id in list(self._running):
                record = self._running.get(task_id)
                if record is None:
                    continue
                if self._task_timed_out(record, now):
                    self._terminate_locked(record)
                    record.status = TaskStatus.FAILED
                    record.message = (
                        f"task timeout after {self.task_timeout_sec:g}s "
                        "(process group killed)"
                    )
                    record.finished_at = now
                    record.append_log(
                        f"[timeout] execution exceeded {self.task_timeout_sec:g}s, killed"
                    )
                    self._persist_terminal_locked(record)
                    continue
                process = record.process
                returncode = process.poll() if process is not None else None
                if returncode is None:
                    continue
                self._detach_process_locked(record)
                try:
                    self._finalizer(record, int(returncode))
                except Exception:  # noqa: BLE001 — 收尾失败也要释放槽位
                    logger.exception("failed to finalize task %s", task_id)
                    record.status = TaskStatus.FAILED
                    record.message = f"finalize failed for exit code {returncode}"
                record.finished_at = now
                self._persist_terminal_locked(record)

            # 2) 排队超时：从未拿到槽位的任务直接失败（不会拉起子进程）
            for task_id in list(self._queue):
                record = self._records.get(task_id)
                if record is None:
                    self._remove_from_queue_locked(task_id)
                    continue
                if self._queue_timed_out(record, now):
                    self._remove_from_queue_locked(task_id)
                    record.status = TaskStatus.FAILED
                    record.message = (
                        f"queue timeout after {self.queue_timeout_sec:g}s "
                        f"(no free execution slot, max_concurrent_tasks="
                        f"{self.max_concurrent_tasks})"
                    )
                    record.finished_at = now
                    record.append_log(
                        f"[timeout] queue wait exceeded {self.queue_timeout_sec:g}s"
                    )
                    self._persist_terminal_locked(record)

            # 3) 有空闲槽位则按 FIFO 拉起
            while self._queue and len(self._running) < self.max_concurrent_tasks:
                task_id = self._queue.popleft()
                record = self._records.get(task_id)
                if record is None or record.status != TaskStatus.QUEUED:
                    continue
                self._launch_locked(record, now)

            self._refresh_queue_positions_locked()

    # ------------------------------------------------------------------
    # 内部实现（调用方需持有 self._lock）
    # ------------------------------------------------------------------
    def _task_timed_out(self, record: TaskRecord, now: float) -> bool:
        if self.task_timeout_sec <= 0 or record.started_at is None:
            return False
        return (now - record.started_at) >= self.task_timeout_sec

    def _queue_timed_out(self, record: TaskRecord, now: float) -> bool:
        if self.queue_timeout_sec <= 0 or record.queued_at is None:
            return False
        return (now - record.queued_at) >= self.queue_timeout_sec

    def _launch_locked(self, record: TaskRecord, now: float) -> None:
        record.started_at = now
        record.queue_position = None
        try:
            process = self._launcher(record)
        except Exception as exc:  # noqa: BLE001 — 启动失败转 failed，不占槽位
            logger.warning("failed to launch task %s: %s", record.task_id, exc)
            record.status = TaskStatus.FAILED
            record.message = f"failed to launch simulation: {exc}"
            record.finished_at = now
            record.append_log(f"[launch] failed: {exc}")
            self._persist_terminal_locked(record)
            return
        if process is None:
            process = record.process
        if process is None:
            record.status = TaskStatus.FAILED
            record.message = "failed to launch simulation: launcher returned no process"
            record.finished_at = now
            record.append_log("[launch] failed: launcher returned no process")
            self._persist_terminal_locked(record)
            return
        record.process = process
        record.status = TaskStatus.RUNNING
        record.progress = 0.0
        record.message = "simulation running"
        self._running[record.task_id] = record
        # running 迁移点落盘（pid / started_at）：进程在服务重启/崩溃后丢失时，
        # 重启恢复扫描据此把任务收敛成 failed/completed（而不是永远 submitted）。
        task_state.write_task_state(
            record,
            status=TaskStatus.RUNNING,
            message=record.message,
            started_at=task_state.utc_now_iso(),
            finished_at=None,
        )

    def _detach_process_locked(self, record: TaskRecord) -> None:
        """把任务从 running 集合摘出并清理 IO（不 kill 进程）。"""
        record.process = None
        self._running.pop(record.task_id, None)
        simulation_runner.close_task_log(record)

    def _terminate_locked(self, record: TaskRecord) -> None:
        """kill 子进程（组）+ 摘出 running + 清理 IO，释放槽位。"""
        process = record.process
        self._detach_process_locked(record)
        _terminate_process(process)

    def _remove_from_queue_locked(self, task_id: str) -> None:
        try:
            self._queue.remove(task_id)
        except ValueError:
            pass

    def _refresh_queue_positions_locked(self) -> None:
        for record in self._records.values():
            record.queue_position = None
        for index, task_id in enumerate(self._queue):
            record = self._records.get(task_id)
            if record is not None:
                record.queue_position = index + 1

    def _persist_terminal_locked(self, record: TaskRecord) -> None:
        """终态落盘（best-effort，调用方已置好 status/message/finished_at）。"""
        task_state.write_task_state(
            record,
            status=record.status,
            message=record.message,
            finished_at=task_state.utc_now_iso(),
        )


# 进程级单例：execute_task / report_status / cancel_task 共享同一个槽位池
scheduler = TaskScheduler(
    max_concurrent_tasks=settings.max_concurrent_tasks,
    max_queue_size=settings.max_queue_size,
    task_timeout_sec=settings.task_timeout_sec,
    queue_timeout_sec=settings.queue_timeout_sec,
)
