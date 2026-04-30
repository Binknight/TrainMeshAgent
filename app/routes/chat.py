"""SSE streaming chat endpoint — main agent interaction interface."""
from __future__ import annotations

import json
import asyncio
import logging
import traceback
from flask import Blueprint, request, Response, stream_with_context

from app.agent.orchestrator import agent_stream
from app.agent.session import session_manager
from app.models.schemas import AgentEvent

logger = logging.getLogger(__name__)
chat_bp = Blueprint("chat", __name__, url_prefix="/api/chat")


@chat_bp.route("/stream", methods=["POST"])
def chat_stream():
    """
    POST /api/chat/stream

    SSE streaming chat endpoint for agent interaction.
    Accepts JSON: { "session_id": "...", "message": "..." }
    Returns SSE stream with AgentEvent types:
      - thinking: agent is processing
      - tool_call: executing a tool
      - guard_check: input/output validation result
      - mesh_json: topology JSON generated
      - sim_data: simulation results (cards pushed via SSE)
      - message: text from agent
      - error: error occurred
      - done: processing complete
    """
    data = request.get_json(silent=True) or {}
    session_id = data.get("session_id", "")
    message = data.get("message", "")

    if not message:
        return {"error": "message is required"}, 400

    session = session_manager.get_session(session_id)
    if not session:
        session = session_manager.create_session()

    logger.info(f"[chat_stream] session={session.session_id} message={message[:80]}...")

    def generate():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            async def stream():
                async for event in agent_stream(session, message):
                    payload = json.dumps(event.model_dump(), ensure_ascii=False)
                    sse_line = f"event: {event.event_type}\ndata: {payload}\n\n"
                    if len(payload) > 1000:
                        logger.info(f"[chat_stream] SSE event={event.event_type} size={len(payload)} chars")
                    yield sse_line
            gen = stream()
            while True:
                try:
                    chunk = loop.run_until_complete(gen.__anext__())
                    yield chunk
                except StopAsyncIteration:
                    break
                except Exception as e:
                    logger.exception("[chat_stream] inner generator error")
                    err_event = AgentEvent(
                        event_type="error",
                        message=f"处理异常: {e}",
                        data={"detail": traceback.format_exc()[-500:]}
                    )
                    yield f"event: error\ndata: {json.dumps(err_event.model_dump(), ensure_ascii=False)}\n\n"
                    break
        except Exception as e:
            logger.exception("[chat_stream] outer generator error")
            err_event = AgentEvent(
                event_type="error",
                message=f"服务异常: {e}",
                data={"detail": traceback.format_exc()[-500:]}
            )
            yield f"event: error\ndata: {json.dumps(err_event.model_dump(), ensure_ascii=False)}\n\n"
        finally:
            loop.close()

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "X-Session-Id": session.session_id,
        }
    )
