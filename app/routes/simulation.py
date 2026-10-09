"""WebSocket endpoint for real-time simulation status streaming."""
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import Blueprint, request
from flask_sock import Sock

from app.mcp.client import mcp_client
from app.agent.session import session_manager
from app.config import config

logger = logging.getLogger(__name__)
sim_bp = Blueprint("simulation", __name__, url_prefix="/ws")
sock = Sock()

# 连续多少轮「任务查不到」就判定任务已从 MCP 侧消失（MCP Server 重启）。
# MCP 的任务表是进程内存字典，重启后旧 task_id 一律查不到；不设终止条件的话
# 这个 while 循环会永远 poll_error，前端也就永远停在"仿真验证中"。
_MAX_TASK_LOST_POLLS = 30


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
                _poll_simulation_tasks(ws, task_ids, interval=config.SIM_POLL_INTERVAL,
                                       session=session)

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


def _mark_session_failed(session, reason: str) -> None:
    """把会话落到 `failed` 并落库（供 WS 轮询的终止分支使用）。"""
    if session is None:
        return
    try:
        session.step = "failed"
        session.history.append({"role": "system", "content": f"❌ 仿真任务失败: {reason}"})
        session_manager.save_session(session)
    except Exception as e:  # 落库失败不能反过来打断 WS 的收尾
        logger.error(f"[ws] mark session failed failed: {e}")
    logger.warning(f"[ws] {reason}")


def _poll_simulation_tasks(ws, task_ids: list[str], interval: float, session=None):
    """Poll simulation tasks in parallel threads — heartbeat every interval to keep WS alive.
    No hard timeout: MCP Server is the authority on task completion.

    唯一的终止条件是「所有任务都到终态」，而"查不到任务"也算终态：
    MCP Server 的任务表是内存字典，重启后旧 task_id 永远查不到，没有这条
    兜底的话循环会一直空转、前端一直停在"仿真验证中"。
    """
    completed = set()
    log_offsets = {}  # {task_id: next_offset} for incremental log sync
    lost_polls = {tid: 0 for tid in task_ids}  # 连续查不到该任务的轮数

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
                    lost_polls[tid] += 1
                    continue

                status = poll["result"]
                st = status.get("status", "unknown")
                progress = status.get("progress", 0)

                if st in ("unavailable", "unknown"):
                    # MCP 可达但查不到该任务（例如 MCP Server 重启后内存任务表清空）。
                    # 不能把它当作"还在跑"——否则这个循环永远不退出。
                    lost_polls[tid] += 1
                    lost_streak = lost_polls[tid]
                    ws.send(json.dumps({
                        "type": "status", "task_id": tid,
                        "status": st, "progress": progress,
                    }))
                    if lost_streak >= _MAX_TASK_LOST_POLLS:
                        reason = (f"任务 {tid} 在 MCP Server 上已不存在"
                                  f"（连续 {lost_streak} 次查询不到，通常意味着 MCP Server 重启过）")
                        _mark_session_failed(session, reason)
                        completed.add(tid)
                        ws.send(json.dumps({
                            "type": "error", "task_id": tid, "message": reason,
                        }))
                    continue

                lost_polls[tid] = 0

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
