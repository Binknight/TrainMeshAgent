"""workspace/{task_id} 下仿真任务的准备、子进程启动与收尾。

职责拆分（并发调度改造）：

- ``prepare_task``：校验仿真工具、创建任务记录、生成训练脚本与 task_meta.json、
  构造 run.py 的 argv。**不拉起子进程**（真实模式的校验仍在 execute_task 同步
  完成，例如 sim_tool_home / run.py 缺失仍立即报 InvalidParamsError）。
- ``launch_task``：真正 Popen 子进程 —— 只由 ``TaskScheduler`` 的 worker 在拿到
  执行槽位后调用，因此 execute_task 可以立即返回 ``queued``。
- ``finalize_task`` / ``close_task_log``：子进程退出后的状态收尾，同样由调度器
  worker 调用，任务终态时释放并发槽位。
- ``refresh_task_status``：状态查询路径上的轻量刷新（看进程/结果目录，不读日志）。
"""

import logging
import os
import subprocess
from pathlib import Path
from typing import IO, Optional

from mcp_server.config import settings
from mcp_server.errors import InvalidParamsError
from mcp_server.schemas.common import (
    TERMINAL_TASK_STATUSES,
    ExecuteTaskInput,
    SimulationRunnerParams,
    TaskStatus,
)
from mcp_server.services import task_state
from mcp_server.services.conda_launcher import build_simulation_command
from mcp_server.services.param_mapper import build_run_py_command
from mcp_server.services.script_generator import generate_topology_script
from mcp_server.services.task_store import TaskRecord, task_store

logger = logging.getLogger(__name__)


def _task_dir(task_id: str) -> Path:
    return settings.workspace_root / task_id


def _results_dir(task_id: str) -> Path:
    return _task_dir(task_id) / "results"


def prepare_task(params: ExecuteTaskInput) -> TaskRecord:
    """创建任务、生成脚本/元数据并构造启动命令；dry_run 到此为止。"""
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
        # 终态落盘（best-effort）：重启后按 failed 恢复，而不是 legacy 的 submitted
        task_state.write_task_state(record, finished_at=task_state.utc_now_iso())
        raise InvalidParamsError(str(exc)) from exc

    record.generated_script = script_path

    # 保存任务元数据（服务重启后 task_loader 依赖它恢复任务）
    meta_path = task_dir / "task_meta.json"
    meta_path.write_text(
        params.model_dump_json(indent=2),
        encoding="utf-8",
    )

    if settings.dry_run:
        record.status = TaskStatus.SUBMITTED
        record.append_log("[dry_run] task prepared, subprocess not started")
        # dry_run 任务不会被调度器拉起：状态直接落成 submitted（带 dry_run=true），
        # 重启后 loader 仍恢复为 submitted，而不是被当成「中断的仿真」判 failed。
        task_state.write_task_state(record, dry_run=True)
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
        task_state.write_task_state(record, finished_at=task_state.utc_now_iso())
        raise InvalidParamsError(str(exc)) from exc

    record.sim_home = sim_home
    record.launch_argv = run_argv
    record.log_path = task_dir / "simulation.log"
    # prepare 迁移点落盘：真实模式随即由调度器入队（submit 会写成 queued）
    task_state.write_task_state(record, dry_run=False)
    return record


def launch_task(record: TaskRecord) -> subprocess.Popen:
    """拉起仿真子进程（调用方：TaskScheduler worker，已持有执行槽位）。"""
    if record.launch_argv is None or record.log_path is None:
        raise InvalidParamsError(
            f"task {record.task_id} has no prepared launch command"
        )
    if record.workspace_dir is None:
        raise InvalidParamsError(
            f"task {record.task_id} has no workspace directory"
        )

    sim_home = record.sim_home or settings.sim_tool_home.resolve()
    launch_cmd = build_simulation_command(record.launch_argv, conda_env=settings.conda_env)
    record.append_log(f"[launch] cwd={record.workspace_dir}")
    record.append_log(f"[launch] conda_env={settings.conda_env or '(none)'}")
    record.append_log(f"[launch] cmd={' '.join(launch_cmd)}")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(sim_home) + os.pathsep + env.get("PYTHONPATH", "")

    log_file: Optional[IO[str]] = open(record.log_path, "a", encoding="utf-8")
    try:
        proc = subprocess.Popen(
            launch_cmd,
            cwd=str(record.workspace_dir),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
            # POSIX 下独立进程组，超时/取消时能连同 run.py 的子进程一起杀
            start_new_session=(os.name == "posix"),
        )
    except OSError as exc:
        log_file.close()
        raise InvalidParamsError(f"failed to start simulation: {exc}") from exc

    record.process = proc
    record._log_file_handle = log_file
    record.status = TaskStatus.RUNNING
    record.progress = 0.0
    record.message = "simulation running"
    return proc


def close_task_log(record: TaskRecord) -> None:
    """关闭子进程 stdout 的日志句柄（可重复调用）。"""
    handle = record._log_file_handle
    record._log_file_handle = None
    if handle is not None:
        try:
            handle.close()
        except Exception:  # noqa: BLE001 — 收尾失败不应打断状态机
            pass


def finalize_task(record: TaskRecord, returncode: int) -> None:
    """根据子进程退出码写终态（调用方：TaskScheduler worker）。"""
    if returncode == 0:
        record.status = TaskStatus.COMPLETED
        record.progress = 100.0
        record.message = "simulation finished successfully"
        record.append_log("[exit] code=0")
    else:
        record.status = TaskStatus.FAILED
        record.message = f"simulation exited with code {returncode}"
        record.append_log(f"[exit] code={returncode}")


def refresh_task_status(record: TaskRecord) -> None:
    """根据子进程状态更新任务记录（状态查询路径，不读日志）。"""
    if record.status in TERMINAL_TASK_STATUSES or record.status == TaskStatus.QUEUED:
        return

    if record.process is not None:
        poll = record.process.poll()
        if poll is None:
            if record.status != TaskStatus.RUNNING:
                record.status = TaskStatus.RUNNING
            if record.progress < 90:
                record.progress = min(record.progress + 1.0, 90.0)
        return

    # 进程已结束但未经过调度器收尾（例如服务重启）— 根据 results 目录推断
    if record.results_dir and record.results_dir.is_dir():
        has_mocked = (record.results_dir / "mocked_workload").is_dir()
        if has_mocked and record.status not in (TaskStatus.FAILED,):
            record.status = TaskStatus.COMPLETED
            record.progress = 100.0
