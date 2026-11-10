"""从内存或 workspace 磁盘加载任务记录，并在启动时恢复「进程已丢失」的任务。

服务重启（SIGKILL / 容器漂移 / 崩溃）后内存任务表是空的，只能靠磁盘恢复：

- ``task_meta.json``：任务的入参（topology / simulation_params），一定由 prepare 写入；
- ``task_state.json``：最后一次已知状态（prepare/queued/running/terminal），由
  ``task_state`` 模块在状态迁移点 best-effort 写入。

``load_task_record`` 的恢复规则（优先级从高到低）：

1. ``task_state.json`` 是终态（completed/failed/error/cancelled）→ 采用该状态；
2. ``task_state.json`` 是 running/queued（进程已随服务一起丢失，本进程并未在跑它）：
   有 results/mocked_workload 产物 → completed，否则 failed 且 message 为
   「服务重启，任务中断（进程已丢失）」；
3. ``dry_run=true`` 且 submitted → submitted（dry_run 任务本来就不会拉起子进程，
   不应被当成「中断」判 failed）；
4. 没有 ``task_state.json``（旧版本任务）→ 保持 legacy 推断：有 results → completed，
   否则 submitted。

``recover_interrupted_tasks`` 是启动扫描：在 ``scheduler.start()`` **之前**把
workspace 下所有 running/queued 的任务按上述规则落成终态，避免调用方对重启前的
task_id 永远看到 running/queued（协议里没有「服务已重启」这个状态）。该函数幂等：
第二轮扫描时状态已是终态，什么也不做；它**不拉起任何子进程**。
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional

from mcp_server.config import settings
from mcp_server.schemas.common import (
    TERMINAL_TASK_STATUSES,
    ExecuteTaskInput,
    TaskStatus,
)
from mcp_server.services import task_state
from mcp_server.services.results_reader import has_results
from mcp_server.services.task_store import TaskRecord, task_store

logger = logging.getLogger(__name__)


def _workspace_task_dir(task_id: str) -> Path:
    return settings.workspace_root / task_id


def _has_completed_results(results_dir: Optional[Path]) -> bool:
    """results/mocked_workload 下是否已有可读的仿真产物。

    只看目录是否存在不够：任务可能刚建目录就被 kill（半成品），把它标 completed
    会让调用方拿到空结果。这里要求至少有一个 ``*_rank*_time.csv``（results_reader
    的 ``has_results`` 同口径）或 legacy 的 ``*_time.csv``。
    """
    if results_dir is None:
        return False
    mocked = results_dir / "mocked_workload"
    if not mocked.is_dir():
        return False
    return has_results(results_dir) or any(mocked.glob("*_time.csv"))


def _apply_task_state(record: TaskRecord, state: dict[str, Any]) -> bool:
    """按 task_state.json 恢复状态；返回是否采用了持久化状态。

    规则见模块 docstring；无法识别（未知/非法状态值、缺少 task_meta 的旧任务）
    时返回 False，由调用方回落 legacy 推断。
    """
    status_value = state.get("status")
    if not isinstance(status_value, str):
        return False
    try:
        status = TaskStatus(status_value)
    except ValueError:
        logger.warning(
            "unknown persisted task status %r for task %s",
            status_value,
            record.task_id,
        )
        return False

    message = state.get("message")
    if not isinstance(message, str):
        message = None

    if status in TERMINAL_TASK_STATUSES:
        # 1) 终态：原样采用（重启/加载不应改变已落定的结果）
        record.status = status
        if status == TaskStatus.COMPLETED:
            record.progress = 100.0
        if message:
            record.message = message
        return True

    if status in (TaskStatus.RUNNING, TaskStatus.QUEUED):
        # 2) 中间态 + 本进程没在跑它 ⇒ 上次服务的进程已丢失
        if _has_completed_results(record.results_dir):
            record.status = TaskStatus.COMPLETED
            record.progress = 100.0
            record.message = task_state.RESTART_COMPLETED_MESSAGE
        else:
            record.status = TaskStatus.FAILED
            record.progress = 0.0
            record.message = task_state.RESTART_INTERRUPTED_MESSAGE
        return True

    if status == TaskStatus.SUBMITTED and bool(state.get("dry_run")):
        # 3) dry_run 任务：永远停在 submitted 是正常语义，不是中断
        record.status = TaskStatus.SUBMITTED
        record.progress = 0.0
        if message:
            record.message = message
        return True

    return False


def _legacy_infer_status(record: TaskRecord, task_dir: Path) -> None:
    """旧版本任务（没有 task_state.json）的磁盘推断，保持改造前语义。"""
    if _has_completed_results(record.results_dir):
        record.status = TaskStatus.COMPLETED
        record.progress = 100.0
    elif task_dir.is_dir():
        record.status = TaskStatus.SUBMITTED


def load_task_record(task_id: str) -> Optional[TaskRecord]:
    """优先内存；不存在则从 workspace/{task_id}/ 的 task_meta + task_state 恢复。"""
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

    # 根据 task_state.json 恢复；旧任务（无状态文件）回落 legacy 推断
    persisted = task_state.read_task_state(task_dir)
    if persisted is None or not _apply_task_state(record, persisted):
        _legacy_infer_status(record, task_dir)

    task_store.register(record)
    return record


def recover_interrupted_tasks(
    workspace_root: Optional[Path] = None,
) -> dict[str, int]:
    """启动恢复扫描：把上次服务遗留的 running/queued 任务收敛成终态（幂等）。

    - 只处理 workspace 的直接子目录且含 ``task_state.json`` 的任务；
    - running/queued → 有完整 results 则 completed，否则 failed（message 含
      「服务重启，任务中断（进程已丢失）」）；
    - 已经是终态的任务不做任何修改（幂等，可重复调用）；
    - **不拉起任何子进程**，只改磁盘状态文件。
    """
    root = Path(workspace_root) if workspace_root is not None else settings.workspace_root
    stats = {
        "scanned": 0,
        "interrupted": 0,
        "completed": 0,
        "failed": 0,
        "errors": 0,
    }
    if not root.is_dir():
        return stats

    try:
        entries = list(root.iterdir())
    except OSError as exc:
        logger.warning("failed to scan workspace %s for task recovery: %s", root, exc)
        stats["errors"] += 1
        return stats

    for entry in entries:
        try:
            if not entry.is_dir():
                continue
        except OSError:
            stats["errors"] += 1
            continue
        state = task_state.read_task_state(entry)
        if state is None:
            continue
        stats["scanned"] += 1
        if state.get("status") not in task_state.INTERRUPTED_STATUSES:
            continue
        stats["interrupted"] += 1

        if _has_completed_results(entry / "results"):
            new_status = TaskStatus.COMPLETED.value
            message = task_state.RESTART_COMPLETED_MESSAGE
            stats["completed"] += 1
        else:
            new_status = TaskStatus.FAILED.value
            message = task_state.RESTART_INTERRUPTED_MESSAGE
            stats["failed"] += 1

        ok = task_state.write_task_state_fields(
            entry,
            status=new_status,
            message=message,
            finished_at=task_state.utc_now_iso(),
        )
        if not ok:
            stats["errors"] += 1

    return stats


def require_task(task_id: str) -> TaskRecord:
    record = load_task_record(task_id)
    if record is None:
        from mcp_server.errors import TaskNotFoundError

        raise TaskNotFoundError(task_id)
    return record
