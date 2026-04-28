import uuid
from typing import Optional
from app.models.schemas import SessionState


class SessionManager:
    """Manages agent session state: original + equivalent topology context."""

    def __init__(self):
        self._sessions: dict[str, SessionState] = {}

    def create_session(self) -> SessionState:
        session_id = str(uuid.uuid4())[:8]
        state = SessionState(session_id=session_id)
        self._sessions[session_id] = state
        return state

    def get_session(self, session_id: str) -> Optional[SessionState]:
        return self._sessions.get(session_id)

    def list_sessions(self) -> list[SessionState]:
        return list(self._sessions.values())

    def delete_session(self, session_id: str) -> bool:
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    def update_session(self, session_id: str, **kwargs) -> Optional[SessionState]:
        state = self._sessions.get(session_id)
        if not state:
            return None
        for key, value in kwargs.items():
            if hasattr(state, key):
                setattr(state, key, value)
        return state


session_manager = SessionManager()
