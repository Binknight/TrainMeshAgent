import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
    OPENAI_SSL_VERIFY = os.getenv("OPENAI_SSL_VERIFY", "true").lower() not in ("false", "0", "no")

    MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:9000")

    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@127.0.0.1:5432/train_mesh_agent")

    FLASK_HOST = os.getenv("FLASK_HOST", "0.0.0.0")
    FLASK_PORT = int(os.getenv("FLASK_PORT", "5000"))
    FLASK_DEBUG = os.getenv("FLASK_DEBUG", "false").lower() == "true"

    # Simulation polling interval (seconds)
    SIM_POLL_INTERVAL = float(os.getenv("SIM_POLL_INTERVAL", "1.0"))

    # Guardrail rules
    VALID_DEVICE_TYPES = {"A2", "A3", "A5"}
    DP_MIN = 1
    DP_MAX = 1024
    TP_MIN = 1
    TP_MAX = 32
    PP_MIN = 1
    PP_MAX = 128


config = Config()
