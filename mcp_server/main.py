"""FastAPI 应用：GET /health、POST /mcp（JSON-RPC tools/call）。"""

import logging
import mimetypes
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Optional, Union

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool

from mcp_server import __version__
from mcp_server.config import settings
from mcp_server.errors import McpError
from mcp_server.rpc import JsonRpcRequest, JsonRpcResponse, JsonRpcError, ToolsCallParams
from mcp_server.services import task_files, task_scheduler, workspace_cleaner
from mcp_server.services.task_loader import recover_interrupted_tasks
from mcp_server.tools.registry import dispatch_tool, list_tools

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """服务启停：启动任务调度/磁盘清理，停机时优雅收尾。

    启动顺序（**必须在 scheduler.start() 之前**做恢复扫描）：
    1. ``recover_interrupted_tasks()``：把上次进程遗留的 running/queued 任务在磁盘上
       落成 completed/failed —— 只改状态文件，不拉起任何子进程；
    2. ``scheduler.start()``：启动调度 worker（空闲槽位拉起排队任务）；
    3. ``cleanup_worker.start()``：启动 workspace 磁盘清理线程。

    停机顺序（uvicorn 收到 SIGTERM/退出信号后走 lifespan shutdown）：
    1. ``cleanup_worker.stop()``：先停后台清理线程；
    2. ``scheduler.shutdown(cancel_running=True)``：取消并 kill 运行中任务的进程组、
       标记终态（cancelled）并落盘、停调度 worker 线程（幂等）。

    ``python -m mcp_server`` 的 uvicorn 在 SIGTERM 下执行优雅停机 ⇒ 一定会走到本
    lifespan 的 finally（Windows 上 ``TerminateProcess`` 之类硬杀无法拦截，那种
    情况由 task_state.json + 下次启动的恢复扫描兜底）。
    """
    stats = None
    try:
        stats = recover_interrupted_tasks()
    except Exception:  # noqa: BLE001 — 恢复扫描失败不能阻止服务启动
        logger.exception("task recovery scan failed (server keeps starting)")
    if stats is not None:
        logger.info(
            "task recovery scan: scanned=%d interrupted=%d completed=%d failed=%d errors=%d",
            stats["scanned"],
            stats["interrupted"],
            stats["completed"],
            stats["failed"],
            stats["errors"],
        )

    # 调度器 worker：在空闲槽位上拉起排队中的仿真任务（幂等）。
    # 单 worker uvicorn 只有一个事件循环，任务拉起/收尾必须由该线程负责，
    # 不能占用请求处理路径。
    task_scheduler.scheduler.start()

    # 工作区磁盘清理：启动后先等 cleanup_initial_delay_sec 扫一次，之后每
    # cleanup_interval_hours（默认 24h）扫一次。优先级见 workspace_cleaner 模块说明：
    # 运行中/排队任务永不删 > 磁盘应急水位 > min_keep_tasks > 结果/日志保留期。
    workspace_cleaner.cleanup_worker.start()

    try:
        yield
    finally:
        # 先停清理线程，再取消运行中任务并停调度器 worker（顺序不要颠倒：
        # 清理线程依赖调度器的运行中/排队集合判定保护，先停调度器会拿到脏状态）。
        workspace_cleaner.cleanup_worker.stop()
        task_scheduler.scheduler.shutdown(cancel_running=True)
        logger.info(
            "shutdown complete: cleanup worker stopped, "
            "task scheduler cancelled running tasks and stopped"
        )


app = FastAPI(
    title="AICM Simulation MCP Server",
    description="equivalent-modeling-service 对接的仿真系统 MCP Server（HTTP JSON-RPC）",
    version=__version__,
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict[str, str]:
    """§2.2 健康检查。"""
    return {"status": "ok"}


@app.get("/tools")
async def tools_catalog() -> dict[str, Any]:
    """非规格接口：列出已注册 tool 及 JSON Schema（便于联调）。"""
    return {"tools": list_tools()}


# ── 仿真产物文件（REST，非 JSON-RPC；供 Flask 的「仿真流量文件」下载代理） ──
# 任务目录由 MCP Server 独占管理，Flask 不直接读 workspace；文件是否存在、路径
# 是否越权、任务是否已被磁盘清理，都以这里的结果为准（见 services/task_files.py）。


@app.get("/tasks/{task_id}/files")
def task_files_list(task_id: str) -> JSONResponse:
    """列出任务目录内的文件分组与任务状态（不读文件内容）。"""
    try:
        return JSONResponse(content=task_files.list_task_files(task_id))
    except task_files.TaskFilesError as exc:
        return JSONResponse(status_code=exc.status_code, content={"error": exc.message})


@app.get("/tasks/{task_id}/files/download")
def task_file_download(task_id: str, path: str) -> Any:
    """下载单个产物文件（path 为相对任务目录的路径）。"""
    try:
        file_path = task_files.resolve_file(task_id, path)
    except task_files.TaskFilesError as exc:
        return JSONResponse(status_code=exc.status_code, content={"error": exc.message})
    media_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    return FileResponse(file_path, media_type=media_type, filename=file_path.name)


@app.get("/tasks/{task_id}/files/bundle")
def task_files_bundle(task_id: str, category: Optional[str] = None) -> Any:
    """按类别（可省略）把任务产物打成 ZIP 流式下载；响应结束后删除临时文件。"""
    try:
        bundle_path, filename = task_files.create_bundle(task_id, category=category)
    except task_files.TaskFilesError as exc:
        return JSONResponse(status_code=exc.status_code, content={"error": exc.message})
    return FileResponse(
        bundle_path,
        media_type="application/zip",
        filename=filename,
        background=BackgroundTask(task_files.remove_quietly, bundle_path),
    )


@app.post("/mcp")
async def mcp_endpoint(request: Request) -> JSONResponse:
    """§2.1 MCP 主调用接口。"""
    try:
        body = await request.json()
    except Exception:
        return _json_rpc_error(
            id=None,
            code=-32700,
            message="Parse error",
            result_detail="invalid JSON body",
        )

    try:
        rpc = JsonRpcRequest.model_validate(body)
    except Exception as exc:
        return _json_rpc_error(
            id=body.get("id") if isinstance(body, dict) else None,
            code=-32600,
            message="Invalid Request",
            result_detail=str(exc),
        )

    if rpc.method != "tools/call":
        return _json_rpc_error(
            id=rpc.id,
            code=-32601,
            message="Method not found",
            result_detail=f"unsupported method: {rpc.method}",
        )

    if not rpc.params:
        return _json_rpc_error(
            id=rpc.id,
            code=-32602,
            message="Invalid params",
            result_detail="params is required",
        )

    try:
        call_params = ToolsCallParams.model_validate(rpc.params)
    except Exception as exc:
        return _json_rpc_error(
            id=rpc.id,
            code=-32602,
            message="Invalid params",
            result_detail=str(exc),
        )

    try:
        # 工具 handler 是同步函数（含子进程 Popen、结果解析等阻塞 IO）——
        # 放进线程池，避免单 worker 的 uvicorn 事件循环被一次调用卡住。
        result_data = await run_in_threadpool(
            dispatch_tool, call_params.name, call_params.arguments
        )
        response = JsonRpcResponse(id=rpc.id, result=result_data)
        return JSONResponse(content=response.model_dump_response())

    except McpError as exc:
        logger.warning("MCP tool error: %s", exc.message)
        return _json_rpc_error(
            id=rpc.id,
            code=exc.code,
            message=exc.message,
            result_detail=exc.message,
            data=exc.data,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unhandled MCP error")
        return _json_rpc_error(
            id=rpc.id,
            code=-32000,
            message="Internal error",
            result_detail=str(exc),
        )


def _json_rpc_error(
    *,
    id: Optional[Union[int, str]],
    code: int,
    message: str,
    result_detail: str,
    data: Optional[dict[str, Any]] = None,
) -> JSONResponse:
    """构造带 result 补充信息的 JSON-RPC 错误响应（§13）。"""
    response = JsonRpcResponse(
        id=id,
        result={"error": True, "message": result_detail},
        error=JsonRpcError(code=code, message=message, data=data),
    )
    return JSONResponse(content=response.model_dump_response())


def create_app() -> FastAPI:
    return app
