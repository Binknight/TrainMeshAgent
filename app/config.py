import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
    OPENAI_SSL_VERIFY = os.getenv("OPENAI_SSL_VERIFY", "true").lower() not in ("false", "0", "no")
    EXTERNAL_PROXY = os.getenv("EXTERNAL_PROXY", os.getenv("LLM_PROXY", ""))

    MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:9000")

    # ── 数据库 ──
    # 默认值刻意保留「TCP 连本机 5432」这一形态：Windows 本地开发直连本机
    # PostgreSQL（如 18.x）时零改动。容器内由镜像 ENV 覆盖为 Unix socket 形态
    # （见仓库根 Dockerfile），环境变量优先级高于这里的默认值。
    #
    # 容器内形态：postgresql://postgres@/train_mesh_agent?host=/home/aicm/db/run
    #   - 空 hostname + host=/path 查询参数 => psycopg2 走 Unix socket
    #   - 不走 TCP，因此没有 listen_addresses / 端口占用 / 主机名解析的问题
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@127.0.0.1:5432/train_mesh_agent")

    # 内嵌 PostgreSQL 的数据目录（仅容器内模式使用；与镜像 ENV 一致）。
    # 由 docker/entrypoint.sh 做「版本守卫 -> 空目录 initdb -> 启动」。
    # PGDATA 是 PostgreSQL 自身的标准环境变量，postgres / initdb / pg_ctl
    # 都直接识别它，因此这里只做读取与默认值对齐，不改写语义。
    PGDATA = os.getenv("PGDATA", "/home/aicm/db/data")

    # Unix socket 所在目录，用于 pg_isready / 就绪探测。
    PG_SOCKET_DIR = os.getenv("PG_SOCKET_DIR", "/home/aicm/db/run")

    # 本地开发豁免开关：跳过「PGDATA 必须是挂载点」的自检（等价于
    # AICM_MCP_WORKSPACE_ALLOW_LOCAL，但独立控制数据库一侧）。
    DB_ALLOW_LOCAL = os.getenv("DB_ALLOW_LOCAL", "").strip().lower() in ("1", "true", "yes", "on")

    FLASK_HOST = os.getenv("FLASK_HOST", "0.0.0.0")
    FLASK_PORT = int(os.getenv("FLASK_PORT", "5000"))
    FLASK_DEBUG = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    FLASK_USE_RELOADER = os.getenv("FLASK_USE_RELOADER", "false").lower() == "true"

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
