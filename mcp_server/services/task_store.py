"""内存任务存储。"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
import secrets
import string
import subprocess

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
    log_offset: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    workspace_dir: Optional[Path] = None
    results_dir: Optional[Path] = None
    generated_script: Optional[Path] = None
    log_path: Optional[Path] = None
    process: Optional[subprocess.Popen] = None
    _log_file_handle: Any = field(default=None, repr=False)

    def append_log(self, line: str) -> None:
        self.logs.append(line)

    def sync_logs_from_file(self) -> None:
        if self.log_path and self.log_path.is_file():
            try:
                text = self.log_path.read_text(encoding="utf-8", errors="replace")
                for line in text.splitlines():
                    if line and line not in self.logs:
                        self.logs.append(line)
            except OSError:
                pass


class TaskStore:
    def __init__(self) -> None:
        self._tasks: dict[str, TaskRecord] = {}

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
        self._tasks[task_id] = record
        return record

    def get(self, task_id: str) -> Optional[TaskRecord]:
        return self._tasks.get(task_id)

    def exists(self, task_id: str) -> bool:
        return task_id in self._tasks


task_store = TaskStore()
