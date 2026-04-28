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


class MCPClient:
    """
    MCP Client that communicates with the simulation system's MCP Server.
    Supports: task_execute, status_report, sync_logs, get_result, card_detail.
    """

    def __init__(self, server_url: str | None = None):
        self.server_url = (server_url or config.MCP_SERVER_URL).rstrip("/")
        self._session = requests.Session()
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
            return resp.json().get("result", {})
        except requests.RequestException as e:
            logger.warning(f"MCP call '{tool_name}' failed: {e}")
            return {"error": str(e), "status": "unavailable"}

    def execute_task(self, topology_data: dict, params: dict | None = None) -> str:
        """Execute a simulation task on the MCP server. Returns task_id."""
        result = self._call_tool("execute_task", {
            "topology": topology_data,
            "simulation_params": params or {},
        })
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

    def check_health(self) -> bool:
        """Check if MCP server is reachable."""
        try:
            resp = self._session.get(f"{self.server_url}/health", timeout=5)
            return resp.status_code == 200
        except requests.RequestException:
            return False


# Singleton
mcp_client = MCPClient()
