"""Tool 注册表：名称 → 入参模型 + 处理函数。"""

from dataclasses import dataclass
from typing import Any, Callable, Type

from pydantic import BaseModel

from mcp_server.errors import InvalidParamsError, McpError, NotImplementedToolError, UnknownToolError
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
)
from mcp_server.tools import handlers


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    input_model: Type[BaseModel]
    handler: Callable[[BaseModel], BaseModel]


TOOL_REGISTRY: dict[str, ToolDefinition] = {
    "execute_task": ToolDefinition(
        name="execute_task",
        description="下发仿真任务（§3）",
        input_model=ExecuteTaskInput,
        handler=handlers.execute_task,
    ),
    "report_status": ToolDefinition(
        name="report_status",
        description="查询任务状态（§4）",
        input_model=ReportStatusInput,
        handler=handlers.report_status,
    ),
    "sync_logs": ToolDefinition(
        name="sync_logs",
        description="增量同步仿真日志（§5）",
        input_model=SyncLogsInput,
        handler=handlers.sync_logs,
    ),
    "get_result": ToolDefinition(
        name="get_result",
        description="获取整体仿真结果（§6）",
        input_model=GetResultInput,
        handler=handlers.get_result,
    ),
    "card_detail": ToolDefinition(
        name="card_detail",
        description="获取单卡/多卡汇总指标（§7）",
        input_model=CardDetailInput,
        handler=handlers.card_detail,
    ),
    "get_device_detail": ToolDefinition(
        name="get_device_detail",
        description="获取单卡算子级 Trace，支持增量轮询（§8）",
        input_model=DeviceDetailInput,
        handler=handlers.get_device_detail,
    ),
    "get_hbm_detail": ToolDefinition(
        name="get_hbm_detail",
        description="获取单卡 HBM 分项占用（§9）",
        input_model=HbmDetailInput,
        handler=handlers.get_hbm_detail,
    ),
    "get_comm_detail": ToolDefinition(
        name="get_comm_detail",
        description="获取单卡 TP/PP/DP 通信详情（§10）",
        input_model=CommDetailInput,
        handler=handlers.get_comm_detail,
    ),
    "get_training_script": ToolDefinition(
        name="get_training_script",
        description="获取任务绑定的 pretrain 训练脚本（§11）",
        input_model=GetTrainingScriptInput,
        handler=handlers.get_training_script,
    ),
}


def list_tools() -> list[dict[str, Any]]:
    """返回已注册 tool 的元信息（便于调试与文档生成）。"""
    items: list[dict[str, Any]] = []
    for tool in TOOL_REGISTRY.values():
        items.append(
            {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.input_model.model_json_schema(),
            }
        )
    return items


def dispatch_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """校验入参并调用 handler，返回可序列化的 result dict。"""
    tool = TOOL_REGISTRY.get(name)
    if tool is None:
        raise UnknownToolError(name)

    try:
        params = tool.input_model.model_validate(arguments)
    except Exception as exc:  # noqa: BLE001 — 统一转为 InvalidParamsError
        raise InvalidParamsError(str(exc)) from exc

    try:
        output = tool.handler(params)
    except McpError:
        raise
    except NotImplementedToolError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise NotImplementedToolError(name) from exc

    if isinstance(output, BaseModel):
        return output.model_dump(mode="json", by_alias=True)
    return dict(output)
