"""从内存或 workspace 磁盘加载任务记录。"""

import json
from pathlib import Path
from typing import Optional

from mcp_server.config import settings
from mcp_server.schemas.common import ExecuteTaskInput, SimulationRunnerParams, TaskStatus
from mcp_server.services.task_store import TaskRecord, task_store


def _workspace_task_dir(task_id: str) -> Path:
    return settings.workspace_root / task_id


def load_task_record(task_id: str) -> Optional[TaskRecord]:
    """优先内存；不存在则从 workspace/{task_id}/task_meta.json 恢复。"""
    record = task_store.get(task_id)
    if record is not None:
        return record

    task_dir = _workspace_task_dir(task_id)
    meta_path = task_dir / "task_meta.json"
    if not meta_path.is_file():
        return None

    data = json.loads(meta_path.read_text(encoding="utf-8"))
    inp = ExecuteTaskInput.model_validate(data)
    record = TaskRecord(
        task_id=task_id,
        topology=inp.topology,
        simulation_params=inp.simulation_params.model_dump(),
        workspace_dir=task_dir,
        results_dir=task_dir / "results",
        generated_script=task_dir / "topology_generated.sh",
        log_path=task_dir / "simulation.log",
    )

    # 根据磁盘产物推断状态
    mocked = record.results_dir / "mocked_workload"
    if mocked.is_dir() and any(mocked.glob("*_time.csv")):
        record.status = TaskStatus.COMPLETED
        record.progress = 100.0
    elif task_dir.is_dir():
        record.status = TaskStatus.SUBMITTED

    task_store._tasks[task_id] = record
    return record


def require_task(task_id: str) -> TaskRecord:
    record = load_task_record(task_id)
    if record is None:
        from mcp_server.errors import TaskNotFoundError

        raise TaskNotFoundError(task_id)
    return record
