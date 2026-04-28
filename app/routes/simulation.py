"""WebSocket endpoint for real-time simulation status streaming."""
import json
import logging
import time
from flask import Blueprint, request
from flask_sock import Sock

from app.mcp.client import mcp_client
from app.agent.session import session_manager

logger = logging.getLogger(__name__)
sim_bp = Blueprint("simulation", __name__, url_prefix="/ws")
sock = Sock()


@sock.route("/ws/simulation/<session_id>")
def simulation_ws(ws, session_id: str):
    """
    WebSocket endpoint for real-time simulation status.

    Client sends JSON: { "type": "subscribe", "task_ids": ["task_orig", "task_eq"] }
    Server pushes: { "type": "status", "task_id": "...", "status": "...", "progress": 50, "log": "..." }
                    { "type": "complete", "task_id": "...", "result": {...} }
                    { "type": "error", "task_id": "...", "message": "..." }
    """
    session = session_manager.get_session(session_id)
    if not session:
        ws.send(json.dumps({"type": "error", "message": "Session not found"}))
        return

    task_ids = []

    while True:
        try:
            raw = ws.receive(timeout=120)
            if raw is None:
                # Timeout — send heartbeat
                ws.send(json.dumps({"type": "heartbeat", "ts": time.time()}))
                continue

            msg = json.loads(raw)
            msg_type = msg.get("type", "")

            if msg_type == "subscribe":
                task_ids = msg.get("task_ids", [])
                ws.send(json.dumps({"type": "subscribed", "task_ids": task_ids}))

                # Start polling loop for subscribed tasks
                _poll_simulation_tasks(ws, task_ids)

            elif msg_type == "unsubscribe":
                break

        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            try:
                ws.send(json.dumps({"type": "error", "message": str(e)}))
            except Exception:
                pass
            break


def _poll_simulation_tasks(ws, task_ids: list[str], interval: float = 1.0, timeout: float = 300.0):
    """Poll simulation tasks status and stream to WebSocket."""
    start_time = time.time()
    completed = set()

    while len(completed) < len(task_ids):
        if time.time() - start_time > timeout:
            for tid in task_ids:
                if tid not in completed:
                    ws.send(json.dumps({
                        "type": "error", "task_id": tid, "message": "Task timeout"
                    }))
            break

        for task_id in task_ids:
            if task_id in completed:
                continue
            try:
                status = mcp_client.get_task_status(task_id)
                st = status.get("status", "unknown")
                progress = status.get("progress", 0)

                ws.send(json.dumps({
                    "type": "status",
                    "task_id": task_id,
                    "status": st,
                    "progress": progress,
                }))

                # Sync logs if available
                logs = mcp_client.sync_logs(task_id)
                if logs.get("lines"):
                    for line in logs["lines"]:
                        ws.send(json.dumps({
                            "type": "log",
                            "task_id": task_id,
                            "line": line,
                        }))

                if st in ("completed", "failed", "error"):
                    completed.add(task_id)
                    if st == "completed":
                        result = mcp_client.get_result(task_id)
                        ws.send(json.dumps({
                            "type": "complete",
                            "task_id": task_id,
                            "result": result,
                        }))
                    else:
                        ws.send(json.dumps({
                            "type": "error",
                            "task_id": task_id,
                            "message": status.get("message", "Task failed"),
                        }))

            except Exception as e:
                logger.warning(f"Poll error for task {task_id}: {e}")

        time.sleep(interval)
