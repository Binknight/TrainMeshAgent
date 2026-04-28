"""TrainMesh Agent — main Flask application entry point."""
import logging
import os
from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS

from app.config import config
from app.routes.chat import chat_bp
from app.routes.session import session_bp
from app.routes.simulation import sim_bp, sock

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")


def create_app() -> Flask:
    app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="/static")
    app.config["SECRET_KEY"] = "trainmesh-agent-secret"

    # CORS support
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    # Register blueprints
    app.register_blueprint(chat_bp)
    app.register_blueprint(session_bp)
    app.register_blueprint(sim_bp)

    # Initialize WebSocket
    sock.init_app(app)

    @app.route("/")
    def index():
        return send_from_directory(STATIC_DIR, "index.html")

    # Health check
    @app.route("/api/health")
    def health():
        return jsonify({"status": "ok", "service": "trainmesh-agent"})

    # API info
    @app.route("/api")
    def api_info():
        return jsonify({
            "service": "TrainMesh Agent",
            "version": "0.1.0",
            "endpoints": {
                "chat_stream": "POST /api/chat/stream  (SSE)",
                "create_session": "POST /api/session",
                "get_session": "GET /api/session/<id>",
                "list_sessions": "GET /api/session",
                "delete_session": "DELETE /api/session/<id>",
                "get_topology": "GET /api/session/<id>/topology",
                "get_simulation": "GET /api/session/<id>/simulation",
                "simulation_ws": "WS /ws/simulation/<session_id>",
            },
            "sse_event_types": [
                "thinking", "tool_call", "guard_check",
                "mesh_json", "message", "error", "done"
            ]
        })

    return app


app = create_app()

if __name__ == "__main__":
    logger.info(f"Starting TrainMesh Agent on {config.FLASK_HOST}:{config.FLASK_PORT}")
    app.run(
        host=config.FLASK_HOST,
        port=config.FLASK_PORT,
        debug=config.FLASK_DEBUG,
    )
