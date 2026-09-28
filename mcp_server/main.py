"""FastAPI 应用：GET /health、POST /mcp（JSON-RPC tools/call）。"""

import logging
from typing import Any, Optional, Union

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from mcp_server import __version__
from mcp_server.config import settings
from mcp_server.errors import McpError
from mcp_server.rpc import JsonRpcRequest, JsonRpcResponse, JsonRpcError, ToolsCallParams
from mcp_server.tools.registry import dispatch_tool, list_tools

logger = logging.getLogger(__name__)

app = FastAPI(
    title="AICM Simulation MCP Server",
    description="TrainMeshAgent 对接的仿真系统 MCP Server（HTTP JSON-RPC）",
    version=__version__,
)


@app.get("/health")
async def health() -> dict[str, str]:
    """§2.2 健康检查。"""
    return {"status": "ok"}


@app.get("/tools")
async def tools_catalog() -> dict[str, Any]:
    """非规格接口：列出已注册 tool 及 JSON Schema（便于联调）。"""
    return {"tools": list_tools()}


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
        result_data = dispatch_tool(call_params.name, call_params.arguments)
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
