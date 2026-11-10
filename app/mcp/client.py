"""
MCP (Model Context Protocol) Client for simulation system communication.
The simulation system is an independent Python program exposing tools via MCP Server.
"""
import json
import logging
import requests
from typing import Any
from app.config import config

logger = logging.getLogger(__name__)

# MCP Server 的 JSON-RPC 错误码约定（见 mcp_server/errors.py）：
# -32004 = 资源不足（当前用于「仿真队列已满」，app 侧应转成 HTTP 429 背压）
MCP_ERROR_QUEUE_FULL = -32004


class MCPQueueFullError(RuntimeError):
    """MCP 侧仿真队列已满（背压）。

    不是任务失败：调用方不能把会话标成 failed，应返回 429 让用户稍后重试。
    """


class MCPClient:
    """
    MCP Client that communicates with the simulation system's MCP Server.
    Supports: task_execute, status_report, sync_logs, get_result, card_detail, get_training_script.
    """

    def __init__(self, server_url: str | None = None):
        self.server_url = (server_url or config.MCP_SERVER_URL).rstrip("/")
        self._session = requests.Session()
        # MCP server is only reachable via direct VPN routing; system/env proxy (HTTP_PROXY)
        # cannot reach it and returns 504 Gateway Time-out. Force direct connection.
        self._session.trust_env = False
        self._session.headers.update({"Content-Type": "application/json"})

    def _call_tool(self, tool_name: str, params: dict[str, Any]) -> dict[str, Any]:
        """Call a tool on the MCP server via JSON-RPC."""
        payload = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": params
            },
            "id": 1
        }
        try:
            resp = self._session.post(
                f"{self.server_url}/mcp",
                json=payload,
                timeout=30
            )
            resp.raise_for_status()
            body = resp.json()
        except requests.RequestException as e:
            logger.warning(f"MCP call '{tool_name}' failed: {e}")
            return {"error": str(e), "status": "unavailable"}
        except ValueError as e:
            # Non-JSON body (proxy error page / truncated response) — same handling
            logger.warning(f"MCP call '{tool_name}' returned invalid JSON: {e}")
            return {"error": str(e), "status": "unavailable"}

        # JSON-RPC 层错误（如任务不存在 / 队列已满）必须显式带出，否则下游会把
        # 响应体当作正常结果读，从而把「查不到任务」误判成「任务还在跑」。
        # 同时保留 code：-32004 表示队列满（背压），不能按普通失败处理。
        rpc_error = body.get("error") if isinstance(body, dict) else None
        if isinstance(rpc_error, dict) and rpc_error:
            result_part = body.get("result") if isinstance(body.get("result"), dict) else {}
            detail = result_part.get("message") or rpc_error.get("message") or "MCP error"
            logger.warning(
                f"MCP call '{tool_name}' returned error code={rpc_error.get('code')}: {detail}"
            )
            return {
                "error": detail,
                "message": detail,
                "status": "unavailable",
                "code": rpc_error.get("code"),
            }

        result = body.get("result", {}) if isinstance(body, dict) else {}
        if isinstance(result, dict) and result.get("error") and result.get("status") != "unavailable":
            logger.warning(f"MCP call '{tool_name}' returned error: {result.get('error')}")
            return {"error": result.get("error"), "status": "unavailable"}
        return result

    def execute_task(self, topology_data: dict, params: dict | None = None) -> str:
        """Execute a simulation task on the MCP server. Returns task_id.

        MCP 队列已满时抛 `MCPQueueFullError`（背压，调用方应返回 429 并允许重试），
        其余情况返回 ""（真正的失败由上层按失败处理）。
        """
        result = self._call_tool("execute_task", {
            "topology": topology_data,
            "simulation_params": params or {},
        })
        if result.get("code") == MCP_ERROR_QUEUE_FULL:
            raise MCPQueueFullError(
                result.get("message") or result.get("error") or "仿真队列已满，请稍后重试"
            )
        return result.get("task_id", "")

    def get_task_status(self, task_id: str) -> dict[str, Any]:
        """Get current status of a simulation task."""
        return self._call_tool("report_status", {"task_id": task_id})

    def sync_logs(self, task_id: str, offset: int = 0) -> dict[str, Any]:
        """Sync logs from simulation task."""
        return self._call_tool("sync_logs", {"task_id": task_id, "offset": offset})

    def get_result(self, task_id: str) -> dict[str, Any]:
        """Get simulation result data."""
        return self._call_tool("get_result", {"task_id": task_id})

    def get_card_details(self, task_id: str, card_ids: list[str] | None = None) -> list[dict[str, Any]]:
        """Get per-card details from simulation result."""
        result = self._call_tool("card_detail", {
            "task_id": task_id,
            "card_ids": card_ids or []
        })
        return result.get("cards", [])

    def get_device_detail(self, task_id: str, global_rank: int, offset: int = 0) -> dict[str, Any]:
        """Get per-device operator trace and timeline. Supports incremental polling via offset."""
        return self._call_tool("get_device_detail", {
            "task_id": task_id,
            "global_rank": global_rank,
            "offset": offset,
        })

    def get_hbm_detail(self, task_id: str, global_rank: int) -> dict[str, Any]:
        """Get per-device HBM usage breakdown."""
        return self._call_tool("get_hbm_detail", {
            "task_id": task_id,
            "global_rank": global_rank,
        })

    def get_comm_detail(self, task_id: str, global_rank: int, comm_type: str) -> dict[str, Any]:
        """Get per-device communication detail for TP/PP/DP."""
        return self._call_tool("get_comm_detail", {
            "task_id": task_id,
            "global_rank": global_rank,
            "comm_type": comm_type,
        })

    def get_training_script(self, task_id: str) -> dict[str, Any]:
        """Get the server-generated pretrain.sh training script for a task."""
        return self._call_tool("get_training_script", {"task_id": task_id})

    def cancel_task(self, task_id: str) -> dict[str, Any]:
        """Cancel a simulation task on the MCP server (idempotent).

        queued 任务直接出队（不会拉起子进程）；running 任务 kill 进程组并置
        cancelled；已终态任务原样返回。旧版 MCP Server 没有该 tool 时，
        `_call_tool` 会返回 {"error": ..., "status": "unavailable"}，调用方
        需要按「尽力取消」处理。
        """
        return self._call_tool("cancel_task", {"task_id": task_id})

    def check_health(self) -> bool:
        """Check if MCP server is reachable."""
        try:
            resp = self._session.get(f"{self.server_url}/health", timeout=5)
            return resp.status_code == 200
        except requests.RequestException:
            return False


# Singleton
mcp_client = MCPClient()
