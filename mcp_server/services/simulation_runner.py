"""在 workspace/{task_id} 下异步拉起 aicm/run.py。"""

import logging
import os
import subprocess
import threading
from pathlib import Path
from typing import IO, Optional

from mcp_server.config import settings
from mcp_server.errors import InvalidParamsError
from mcp_server.schemas.common import ExecuteTaskInput, SimulationRunnerParams, TaskStatus
from mcp_server.services.conda_launcher import build_simulation_command
from mcp_server.services.param_mapper import build_run_py_command
from mcp_server.services.script_generator import generate_topology_script
from mcp_server.services.task_store import TaskRecord, task_store

logger = logging.getLogger(__name__)


def _task_dir(task_id: str) -> Path:
    return settings.workspace_root / task_id


def _results_dir(task_id: str) -> Path:
    return _task_dir(task_id) / "results"


def prepare_and_launch(params: ExecuteTaskInput) -> TaskRecord:
    """创建任务、生成脚本、启动仿真子进程。"""
    sim_home = settings.sim_tool_home.resolve()
    if not sim_home.is_dir():
        raise InvalidParamsError(f"sim_tool_home not found: {sim_home}")

    try:
        record = task_store.create(
            topology=params.topology,
            simulation_params=params.simulation_params.model_dump(),
        )
    except ValueError as exc:
        raise InvalidParamsError(str(exc)) from exc

    task_dir = _task_dir(record.task_id)
    task_dir.mkdir(parents=True, exist_ok=True)
    record.workspace_dir = task_dir
    record.results_dir = _results_dir(record.task_id)

    script_path = task_dir / "topology_generated.sh"
    try:
        generate_topology_script(
            params.topology,
            script_path,
            params.simulation_params.model_dump(),
        )
    except ValueError as exc:
        record.status = TaskStatus.FAILED
        record.message = str(exc)
        raise InvalidParamsError(str(exc)) from exc

    record.generated_script = script_path

    # 保存任务元数据
    meta_path = task_dir / "task_meta.json"
    meta_path.write_text(
        params.model_dump_json(indent=2),
        encoding="utf-8",
    )

    if settings.dry_run:
        record.status = TaskStatus.SUBMITTED
        record.append_log("[dry_run] task prepared, subprocess not started")
        return record

    try:
        run_argv = build_run_py_command(
            sim_home=sim_home,
            task_dir=task_dir,
            script_path=script_path,
            topology=params.topology,
            params=params.simulation_params,
        )
    except FileNotFoundError as exc:
        record.status = TaskStatus.FAILED
        record.message = str(exc)
        raise InvalidParamsError(str(exc)) from exc

    log_path = task_dir / "simulation.log"
    record.log_path = log_path
    launch_cmd = build_simulation_command(run_argv, conda_env=settings.conda_env)
    record.append_log(f"[launch] cwd={task_dir}")
    record.append_log(f"[launch] conda_env={settings.conda_env or '(none)'}")
    record.append_log(f"[launch] cmd={' '.join(launch_cmd)}")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(sim_home) + os.pathsep + env.get("PYTHONPATH", "")

    log_file: Optional[IO[str]] = open(log_path, "a", encoding="utf-8")
    try:
        proc = subprocess.Popen(
            launch_cmd,
            cwd=str(task_dir),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
        )
    except OSError as exc:
        log_file.close()
        record.status = TaskStatus.FAILED
        record.message = str(exc)
        raise InvalidParamsError(f"failed to start simulation: {exc}") from exc

    record.process = proc
    record._log_file_handle = log_file
    record.status = TaskStatus.RUNNING
    record.progress = 0.0

    threading.Thread(
        target=_wait_process,
        args=(record.task_id, proc, log_file),
        daemon=True,
    ).start()

    return record


def _wait_process(task_id: str, proc: subprocess.Popen, log_file: IO[str]) -> None:
    returncode = proc.wait()
    try:
        log_file.close()
    except Exception:  # noqa: BLE001
        pass

    record = task_store.get(task_id)
    if record is None:
        return

    record.process = None
    record._log_file_handle = None

    if returncode == 0:
        record.status = TaskStatus.COMPLETED
        record.progress = 100.0
        record.message = "simulation finished successfully"
        record.append_log(f"[exit] code=0")
    else:
        record.status = TaskStatus.FAILED
        record.message = f"simulation exited with code {returncode}"
        record.append_log(f"[exit] code={returncode}")


def refresh_task_status(record: TaskRecord) -> None:
    """根据子进程状态更新任务记录。"""
    if record.process is not None:
        poll = record.process.poll()
        if poll is None:
            if record.status != TaskStatus.RUNNING:
                record.status = TaskStatus.RUNNING
            if record.progress < 90:
                record.progress = min(record.progress + 1.0, 90.0)
        return

    if record.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.ERROR):
        return

    # 进程已结束但未经过 waiter（例如服务重启）— 根据 results 目录推断
    if record.results_dir and record.results_dir.is_dir():
        has_mocked = (record.results_dir / "mocked_workload").is_dir()
        if has_mocked and record.status not in (TaskStatus.FAILED,):
            record.status = TaskStatus.COMPLETED
            record.progress = 100.0
