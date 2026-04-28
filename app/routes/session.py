"""REST endpoints for session management."""
from flask import Blueprint, request, jsonify

from app.agent.session import session_manager
from app.models.schemas import SessionState

session_bp = Blueprint("session", __name__, url_prefix="/api/session")


@session_bp.route("", methods=["POST"])
def create_session():
    """Create a new agent session."""
    session = session_manager.create_session()
    return jsonify(session.model_dump())


@session_bp.route("/<session_id>", methods=["GET"])
def get_session(session_id: str):
    """Get session state by ID."""
    session = session_manager.get_session(session_id)
    if not session:
        return {"error": "session not found"}, 404
    return jsonify(session.model_dump())


@session_bp.route("", methods=["GET"])
def list_sessions():
    """List all active sessions."""
    sessions = session_manager.list_sessions()
    return jsonify([s.model_dump() for s in sessions])


@session_bp.route("/<session_id>", methods=["DELETE"])
def delete_session(session_id: str):
    """Delete a session."""
    ok = session_manager.delete_session(session_id)
    if not ok:
        return {"error": "session not found"}, 404
    return {"status": "deleted", "session_id": session_id}


@session_bp.route("/<session_id>/topology", methods=["GET"])
def get_topology(session_id: str):
    """Get the mesh topology data for a session."""
    session = session_manager.get_session(session_id)
    if not session:
        return {"error": "session not found"}, 404

    return jsonify({
        "session_id": session_id,
        "original_topology": session.original_topology.model_dump() if session.original_topology else None,
        "equivalent_topology": session.equivalent_topology.model_dump() if session.equivalent_topology else None,
        "step": session.step,
    })


@session_bp.route("/<session_id>/simulation", methods=["GET"])
def get_simulation(session_id: str):
    """Get simulation results for a session."""
    session = session_manager.get_session(session_id)
    if not session:
        return {"error": "session not found"}, 404

    return jsonify({
        "session_id": session_id,
        "original_simulation": session.original_simulation.model_dump() if session.original_simulation else None,
        "equivalent_simulation": session.equivalent_simulation.model_dump() if session.equivalent_simulation else None,
        "comparison_report": session.comparison_report.model_dump() if session.comparison_report else None,
        "step": session.step,
    })
