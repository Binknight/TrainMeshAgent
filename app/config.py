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
    # 双后端：默认 SQLite；`DATABASE_URL` 指向 PG 时自动切回 PostgreSQL（逃生门）。
    #
    # 后端由 DSN 前缀侦测（见 app/dbapi.detect_backend），刻意不引入 `DB_BACKEND`
    # 独立开关 —— 开关与 URL 不一致会产生第四种状态。
    #
    # 空串 = 用下面 SQLITE_PATH 指定的 SQLite 文件。
    # 注意：这里**不再是**那串 TCP 默认值 —— 显式注入 PG 串的部署行为不变，
    # 但依赖旧默认值的部署需要显式设置 DATABASE_URL（见迁移说明）。
    DATABASE_URL = os.getenv("DATABASE_URL", "")

    # SQLite 数据文件（默认后端）。父目录即部署侧的挂载点 /home/data/db。
    # WAL 模式下同目录还会出现 -wal / -shm 两个伴生文件，备份需整目录拷贝。
    SQLITE_PATH = os.getenv("SQLITE_PATH", "/home/data/db/equivalent_modeling_service.db")

    # SQLite 写锁等待上限（毫秒）。WAL 是库级单写者，并发写靠这个等待而非立刻报错。
    SQLITE_BUSY_TIMEOUT_MS = int(os.getenv("SQLITE_BUSY_TIMEOUT_MS", "5000"))

    # SQLite 同步级别：NORMAL 在 WAL 下只在 checkpoint 时 fsync，
    # 极端掉电可能丢最后若干事务 —— 会话历史非关键数据，这是有意折衷。
    SQLITE_SYNCHRONOUS = os.getenv("SQLITE_SYNCHRONOUS", "NORMAL").strip().upper()
    if SQLITE_SYNCHRONOUS not in ("OFF", "NORMAL", "FULL", "EXTRA"):
        SQLITE_SYNCHRONOUS = "NORMAL"

    # 外部/内嵌 PostgreSQL 数据目录（仅 PG 后端与版本排障使用；SQLite 路径忽略）。
    # PGDATA 是 PostgreSQL 自身的标准环境变量，这里只做读取与默认值对齐。
    PGDATA = os.getenv("PGDATA", "/home/aicm/db/data")

    # Unix socket 所在目录，仅 PG 后端的就绪探测与排障使用。
    PG_SOCKET_DIR = os.getenv("PG_SOCKET_DIR", "/home/aicm/db/run")

    # 本地开发豁免开关：跳过「数据库目录必须是挂载点」的自检（等价于
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
