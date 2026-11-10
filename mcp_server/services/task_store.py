"""内存任务存储。"""

import secrets
import string
import subprocess
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from mcp_server.schemas.common import SimulationTaskInput, TaskStatus


def _rand_suffix(length: int = 6) -> str:
    alphabet = string.ascii_lowercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


def generate_task_id() -> str:
    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    return f"sim_{ts}_{_rand_suffix()}"


@dataclass
class TaskRecord:
    task_id: str
    topology: SimulationTaskInput
    simulation_params: dict[str, Any]
    status: TaskStatus = TaskStatus.SUBMITTED
    progress: float = 0.0
    message: Optional[str] = None
    logs: list[str] = field(default_factory=list)
    # 对外语义见 sync_logs：offset/next_offset 都是「行偏移」
    log_offset: int = 0
    # 日志文件已消费到的字节偏移（增量读取游标，不对外暴露）
    log_byte_offset: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    # ---- 调度器字段（time.monotonic() 时间基，由 TaskScheduler 维护）----
    queued_at: Optional[float] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    # 1-based FIFO 位置；非 queued 时为 None
    queue_position: Optional[int] = None
    # ---- 子进程启动参数（prepare_task 构造，launch_task 使用）----
    launch_argv: Optional[list[str]] = None
    sim_home: Optional[Path] = None
    workspace_dir: Optional[Path] = None
    results_dir: Optional[Path] = None
    generated_script: Optional[Path] = None
    log_path: Optional[Path] = None
    process: Optional[subprocess.Popen] = None
    _log_file_handle: Any = field(default=None, repr=False)
    _logs_lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False, compare=False
    )

    def append_log(self, line: str) -> None:
        with self._logs_lock:
            self.logs.append(line)

    def sync_logs_from_file(self) -> None:
        """把 log_path 中**新增**的内容追加到 logs。

        只从上次消费到的字节偏移往后读，不再每次整文件 read_text + 对 logs 做
        去重（旧实现的 O(n²) 是并行仿真时 report_status/sync_logs 互相拖慢的根因）。
        末尾未写完的半行不消费，等下一次读取时再拼完整。
        """
        path = self.log_path
        if path is None:
            return
        with self._logs_lock:
            try:
                size = path.stat().st_size
            except OSError:
                return
            if size < self.log_byte_offset:
                # 日志被截断/重建：游标失效，从头再读
                self.log_byte_offset = 0
            if size == self.log_byte_offset:
                return
            try:
                with open(path, "rb") as handle:
                    handle.seek(self.log_byte_offset)
                    chunk = handle.read()
            except OSError:
                return
            if not chunk:
                return
            last_newline = chunk.rfind(b"\n")
            if last_newline < 0:
                # 还没有完整行（进程正在写），下次再读
                return
            complete = chunk[: last_newline + 1]
            self.log_byte_offset += len(complete)
            text = complete.decode("utf-8", errors="replace")
            for line in text.splitlines():
                if line:
                    self.logs.append(line)


class TaskStore:
    def __init__(self) -> None:
        self._tasks: dict[str, TaskRecord] = {}
        self._lock = threading.RLock()

    def create(
        self,
        topology: SimulationTaskInput,
        simulation_params: dict[str, Any],
    ) -> TaskRecord:
        expected = topology.dp_size * topology.tp_size * topology.pp_size
        if topology.total_nodes != expected:
            raise ValueError(
                f"total_nodes ({topology.total_nodes}) must equal "
                f"dp_size*tp_size*pp_size ({expected})"
            )
        task_id = generate_task_id()
        record = TaskRecord(
            task_id=task_id,
            topology=topology,
            simulation_params=simulation_params,
        )
        self.register(record)
        return record

    def register(self, record: TaskRecord) -> None:
        """登记一个已构造好的记录（task_loader 从磁盘恢复时使用）。"""
        with self._lock:
            self._tasks[record.task_id] = record

    def get(self, task_id: str) -> Optional[TaskRecord]:
        with self._lock:
            return self._tasks.get(task_id)

    def exists(self, task_id: str) -> bool:
        with self._lock:
            return task_id in self._tasks


task_store = TaskStore()
