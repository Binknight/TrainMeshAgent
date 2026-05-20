"""WebSocket endpoint for real-time simulation status streaming."""
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
            raw = ws.receive(timeout=30)
            if raw is None:
                # Timeout — send heartbeat to keep proxy/load-balancer alive
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


def _poll_task_status(task_id: str) -> dict:
    """Poll a single task's status. Called from a thread with its own timeout."""
    try:
        status = mcp_client.get_task_status(task_id)
        return {"task_id": task_id, "result": status, "error": None}
    except Exception as e:
        return {"task_id": task_id, "result": None, "error": str(e)}


def _poll_simulation_tasks(ws, task_ids: list[str], interval: float = 1.0):
    """Poll simulation tasks in parallel threads — heartbeat every interval to keep WS alive.
    No hard timeout: MCP Server is the authority on task completion.
    """
    completed = set()
    log_offsets = {}  # {task_id: next_offset} for incremental log sync

    with ThreadPoolExecutor(max_workers=len(task_ids)) as executor:
        while len(completed) < len(task_ids):

            # Send heartbeat BEFORE polling to keep connection alive during MCP wait
            ws.send(json.dumps({"type": "heartbeat", "ts": time.time()}))

            # Fire parallel polls for all incomplete tasks
            pending = [tid for tid in task_ids if tid not in completed]
            futures = {executor.submit(_poll_task_status, tid): tid for tid in pending}

            for future in as_completed(futures):
                tid = futures[future]
                try:
                    poll = future.result()
                except Exception:
                    ws.send(json.dumps({
                        "type": "status", "task_id": tid,
                        "status": "poll_error", "progress": -1,
                    }))
                    continue

                if poll["error"]:
                    ws.send(json.dumps({
                        "type": "status", "task_id": tid,
                        "status": "poll_error", "progress": -1,
                    }))
                    continue

                status = poll["result"]
                st = status.get("status", "unknown")
                progress = status.get("progress", 0)

                ws.send(json.dumps({
                    "type": "status",
                    "task_id": tid,
                    "status": st,
                    "progress": progress,
                }))

                # Sync logs incrementally
                offset = log_offsets.get(tid, 0)
                try:
                    logs = mcp_client.sync_logs(tid, offset=offset)
                    if logs.get("next_offset") is not None:
                        log_offsets[tid] = logs["next_offset"]
                    if logs.get("lines"):
                        for line in logs["lines"]:
                            ws.send(json.dumps({"type": "log", "task_id": tid, "line": line}))
                except Exception:
                    pass

                if st in ("completed", "failed", "error"):
                    completed.add(tid)
                    if st == "completed":
                        try:
                            result = mcp_client.get_result(tid)
                            ws.send(json.dumps({
                                "type": "complete",
                                "task_id": tid,
                                "result": result,
                            }))
                        except Exception as e:
                            ws.send(json.dumps({
                                "type": "error", "task_id": tid,
                                "message": f"get_result failed: {e}",
                            }))
                    else:
                        ws.send(json.dumps({
                            "type": "error",
                            "task_id": tid,
                            "message": status.get("message", "Task failed"),
                        }))

            if len(completed) < len(task_ids):
                time.sleep(interval)
