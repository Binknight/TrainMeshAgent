"""MCP Tool 处理函数。"""

import logging

from mcp_server.errors import TaskNotFoundError, TaskNotReadyError
from mcp_server.schemas.common import (
    CardDetailInput,
    CardDetailOutput,
    CommDetailInput,
    CommDetailOutput,
    DeviceDetailInput,
    DeviceDetailOutput,
    ExecuteTaskInput,
    ExecuteTaskOutput,
    GetResultInput,
    GetResultOutput,
    GetTrainingScriptInput,
    GetTrainingScriptOutput,
    HbmDetailInput,
    HbmDetailOutput,
    ReportStatusInput,
    ReportStatusOutput,
    SyncLogsInput,
    SyncLogsOutput,
    TaskStatus,
)
from mcp_server.services import results_reader
from mcp_server.services.script_generator import resolve_script_filename
from mcp_server.services.simulation_runner import prepare_and_launch, refresh_task_status
from mcp_server.services.task_loader import require_task
from mcp_server.services.task_store import task_store

logger = logging.getLogger(__name__)


def execute_task(params: ExecuteTaskInput) -> ExecuteTaskOutput:
    """§3 下发仿真任务。"""
    record = prepare_and_launch(params)
    return ExecuteTaskOutput(task_id=record.task_id, status=record.status)


def report_status(params: ReportStatusInput) -> ReportStatusOutput:
    """§4 查询任务状态。"""
    record = require_task(params.task_id)
    record.sync_logs_from_file()
    refresh_task_status(record)
    return ReportStatusOutput(
        status=record.status,
        progress=record.progress,
        message=record.message,
    )


def sync_logs(params: SyncLogsInput) -> SyncLogsOutput:
    """§5 增量同步日志。"""
    record = require_task(params.task_id)
    record.sync_logs_from_file()
    lines = record.logs[params.offset :]
    return SyncLogsOutput(lines=lines, next_offset=len(record.logs))


def _is_running(record) -> bool:
    return record.process is not None and record.process.poll() is None


def _ensure_results_ready(record) -> None:
    refresh_task_status(record)
    results_dir = record.results_dir
    if results_dir and results_reader.has_results(results_dir):
        return
    if record.status != TaskStatus.COMPLETED:
        raise TaskNotReadyError(record.task_id, record.status.value)


def get_result(params: GetResultInput) -> GetResultOutput:
    """§6 获取整体仿真结果。"""
    record = require_task(params.task_id)
    _ensure_results_ready(record)
    cards = results_reader.load_all_cards(record)
    ranks = results_reader.list_available_ranks(record.results_dir)
    summary = {
        "topology_name": record.topology.name,
        "device_type": record.topology.device_type,
        "total_ranks": len(ranks),
        "results_dir": str(record.results_dir),
        "dp_size": record.topology.dp_size,
        "tp_size": record.topology.tp_size,
        "pp_size": record.topology.pp_size,
        "model_type": record.topology.model_type,
    }
    if record.topology.ep is not None:
        summary["ep_size"] = record.topology.ep
    if cards:
        summary["aggregate_flops"] = sum(c.flops_per_card for c in cards)
        summary["aggregate_hbm_gb"] = sum(c.hbm_gb for c in cards)
    return GetResultOutput(
        task_id=params.task_id,
        status=TaskStatus.COMPLETED,
        summary=summary,
        cards=cards,
    )


def card_detail(params: CardDetailInput) -> CardDetailOutput:
    """§7 获取单卡/多卡汇总指标。"""
    logger.info("card_detail request: %s", params.model_dump_json())
    record = require_task(params.task_id)
    _ensure_results_ready(record)
    cards = results_reader.load_all_cards(record, params.card_ids)
    output = CardDetailOutput(cards=cards)
    logger.info(
        "card_detail response: task_id=%s card_count=%d body=%s",
        params.task_id,
        len(output.cards),
        output.model_dump_json(),
    )
    return output


def get_device_detail(params: DeviceDetailInput) -> DeviceDetailOutput:
    """§8 获取单卡算子级 Trace（支持增量）。"""
    record = require_task(params.task_id)
    refresh_task_status(record)
    running = _is_running(record)
    if not running:
        _ensure_results_ready(record)
    return results_reader.get_device_detail(
        record,
        params.global_rank,
        params.offset,
        is_task_running=running,
    )


def get_hbm_detail(params: HbmDetailInput) -> HbmDetailOutput:
    """§9 获取单卡 HBM 分项。"""
    record = require_task(params.task_id)
    _ensure_results_ready(record)
    return results_reader.get_hbm_detail(record, params.global_rank)


def get_comm_detail(params: CommDetailInput) -> CommDetailOutput:
    """§10 获取单卡通信详情。"""
    record = require_task(params.task_id)
    _ensure_results_ready(record)
    return results_reader.get_comm_detail(
        record, params.global_rank, params.comm_type
    )


def get_training_script(params: GetTrainingScriptInput) -> GetTrainingScriptOutput:
    """§11 获取训练脚本（execute_task 返回后即可调用）。"""
    record = require_task(params.task_id)
    script_path = record.generated_script
    if script_path is None and record.workspace_dir is not None:
        script_path = record.workspace_dir / "topology_generated.sh"
    if script_path is None or not script_path.is_file():
        raise TaskNotFoundError(params.task_id)

    return GetTrainingScriptOutput(
        task_id=params.task_id,
        topology_name=record.topology.name,
        script_path=str(script_path.resolve()),
        script_filename=resolve_script_filename(record.topology.name),
        script_content=script_path.read_text(encoding="utf-8"),
    )
