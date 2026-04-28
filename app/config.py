import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

    MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:9000")

    FLASK_HOST = os.getenv("FLASK_HOST", "0.0.0.0")
    FLASK_PORT = int(os.getenv("FLASK_PORT", "5000"))
    FLASK_DEBUG = os.getenv("FLASK_DEBUG", "false").lower() == "true"

    # Guardrail rules
    VALID_DEVICE_TYPES = {"A2", "A3", "A5"}
    DP_MIN = 1
    DP_MAX = 1024
    TP_MIN = 1
    TP_MAX = 32
    PP_MIN = 1
    PP_MAX = 128


config = Config()
