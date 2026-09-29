"""Database migration script. Creates all tables if they don't exist.

支持两种方言（由 `app.db.backend()` 决定，源自 `DATABASE_URL` 前缀）：

  - `postgres`：外部 PostgreSQL（逃生门），DDL 与迁移前逐字一致
  - `sqlite`  ：默认后端，见 `SCHEMA_SQLITE_SQL`

历史说明（数据库内嵌化重构时留下的两条约束，**两者都必须保持**）：
  1. 不恢复「静默吞 DDL 异常」的写法。当前实现把 DDL 错误分为「可忽略」
     （`BENIGN_DDL_ERRORS`，如建表重复）与「致命」，并在收尾用
     `REQUIRED_TABLES` / `REQUIRED_COLUMNS` 做硬校验，缺失即抛 `RuntimeError`。
     否则会出现「schema 不完整但服务照常跑」，直到有人发现历史标题全是「新建任务」。
  2. 启动路径上不放破坏性 DDL（`DROP COLUMN` 之类）。每次重启都会重放。

两种方言的幂等策略不同，但**收尾硬校验完全共用**：
  - PG  ：`CREATE TABLE IF NOT EXISTS` + 30 条 `ALTER TABLE ... ADD COLUMN IF NOT EXISTS`
  - SQLite：不支持 `ADD COLUMN IF NOT EXISTS`（表/索引的 `IF NOT EXISTS` 是支持的），
    因此建表直接给**最终列集合**（新库无历史包袱），加列逻辑由
    `PRAGMA table_info()` 探测后动态补齐 —— 用于兼容改造前留下的旧 SQLite 文件。
"""

from __future__ import annotations

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
    id              VARCHAR(8) PRIMARY KEY,
    step            VARCHAR(20) NOT NULL DEFAULT 'idle',
    original_task_id    VARCHAR(64),
    equivalent_task_id  VARCHAR(64),
    created_at      TIMESTAMP DEFAULT NOW(),
    updated_at      TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS topology_params (
    id              UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    session_id      VARCHAR(8) REFERENCES sessions(id) ON DELETE CASCADE,
    role            VARCHAR(10) NOT NULL CHECK (role IN ('original', 'equivalent')),
    name            VARCHAR(100) NOT NULL,
    device_type     VARCHAR(10) NOT NULL,
    dp_size         INT NOT NULL,
    tp_size         INT NOT NULL,
    pp_size         INT NOT NULL,
    total_nodes     INT NOT NULL,
    model_name      VARCHAR(100),
    num_layers      INT,
    hidden_dim      INT,
    num_heads       INT,
    d_ffn           INT,
    seq_len         INT,
    batch_size      INT,
    micro_batch_size INT,
    vocab_size      INT,
    -- MoE-specific fields
    model_type      VARCHAR(10) DEFAULT 'dense',
    num_experts     INT,
    moe_router_topk INT,
    num_moe_layers  INT,
    moe_ffn_hidden_size INT,
    has_shared_expert BOOLEAN DEFAULT FALSE,
    expert_tensor_parallel_size INT DEFAULT 1,
    ep              INT,
    UNIQUE (session_id, role)
);

CREATE TABLE IF NOT EXISTS simulation_params (
    id              UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    session_id      VARCHAR(8) REFERENCES sessions(id) ON DELETE CASCADE,
    role            VARCHAR(10) NOT NULL CHECK (role IN ('original', 'equivalent')),
    script_path     VARCHAR(500),
    epoch_num       INT DEFAULT 1,
    model_name      VARCHAR(100),
    device_type     VARCHAR(50),
    vocab_size      VARCHAR(20),
    frame           VARCHAR(50),
    rank            INT DEFAULT 0,
    rank_range      INT,
    comp_filepath   VARCHAR(500),
    no_time_accumulation BOOLEAN DEFAULT FALSE,
    level0_config   JSONB,
    level1_config   JSONB,
    visual_json_output   BOOLEAN DEFAULT TRUE,
    comm_group_output    BOOLEAN DEFAULT TRUE,
    debug_time      BOOLEAN DEFAULT FALSE,
    UNIQUE (session_id, role)
);

CREATE TABLE IF NOT EXISTS simulation_results (
    id              UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    session_id      VARCHAR(8) REFERENCES sessions(id) ON DELETE CASCADE,
    role            VARCHAR(10) NOT NULL CHECK (role IN ('original', 'equivalent')),
    topology_name   VARCHAR(100),
    device_type     VARCHAR(10),
    total_nodes     INT,
    is_simulated    BOOLEAN DEFAULT FALSE,
    cards           JSONB DEFAULT '[]',
    UNIQUE (session_id, role)
);

CREATE TABLE IF NOT EXISTS comparison_reports (
    id              UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    session_id      VARCHAR(8) REFERENCES sessions(id) ON DELETE CASCADE,
    original_id     UUID REFERENCES simulation_results(id),
    equivalent_id   UUID REFERENCES simulation_results(id),
    flops_diff_pct  FLOAT,
    hbm_diff_pct    FLOAT,
    tp_comm_diff_pct FLOAT,
    pp_comm_diff_pct FLOAT,
    dp_comm_diff_pct FLOAT,
    is_equivalent   BOOLEAN,
    error_tolerance FLOAT DEFAULT 5.0,
    details         JSONB
);

CREATE TABLE IF NOT EXISTS conversation_messages (
    id              UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    session_id      VARCHAR(8) REFERENCES sessions(id) ON DELETE CASCADE,
    msg_index       INT NOT NULL,
    role            VARCHAR(20) NOT NULL,
    content         TEXT,
    timestamp       TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS model_catalog (
    id              UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    model_name      VARCHAR(100) NOT NULL UNIQUE,
    name_key        VARCHAR(100) NOT NULL UNIQUE,
    model_type      VARCHAR(10) NOT NULL DEFAULT 'dense',
    num_layers      INT NOT NULL,
    d_model         INT NOT NULL,
    num_heads       INT NOT NULL,
    d_ffn           INT NOT NULL,
    vocab_size      INT NOT NULL,
    num_kv_heads    INT,
    source          VARCHAR(20),
    reference       TEXT,
    description     TEXT,
    created_at      TIMESTAMP DEFAULT NOW(),
    updated_at      TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_model_catalog_name_key ON model_catalog(name_key);

ALTER TABLE topology_params ADD COLUMN IF NOT EXISTS d_ffn INT;
ALTER TABLE topology_params ADD COLUMN IF NOT EXISTS micro_batch_size INT;
ALTER TABLE topology_params ADD COLUMN IF NOT EXISTS vocab_size INT;
ALTER TABLE topology_params ADD COLUMN IF NOT EXISTS model_type VARCHAR(10) DEFAULT 'dense';  -- MoE-specific columns
ALTER TABLE topology_params ADD COLUMN IF NOT EXISTS num_experts INT;
ALTER TABLE topology_params ADD COLUMN IF NOT EXISTS moe_router_topk INT;
ALTER TABLE topology_params ADD COLUMN IF NOT EXISTS num_moe_layers INT;
ALTER TABLE topology_params ADD COLUMN IF NOT EXISTS moe_ffn_hidden_size INT;
ALTER TABLE topology_params ADD COLUMN IF NOT EXISTS has_shared_expert BOOLEAN DEFAULT FALSE;
ALTER TABLE topology_params ADD COLUMN IF NOT EXISTS expert_tensor_parallel_size INT DEFAULT 1;
ALTER TABLE topology_params ADD COLUMN IF NOT EXISTS ep INT;

ALTER TABLE model_catalog ADD COLUMN IF NOT EXISTS reference TEXT;

ALTER TABLE model_catalog ADD COLUMN IF NOT EXISTS tp INT;
ALTER TABLE model_catalog ADD COLUMN IF NOT EXISTS pp INT;
ALTER TABLE model_catalog ADD COLUMN IF NOT EXISTS dp INT;
ALTER TABLE model_catalog ADD COLUMN IF NOT EXISTS seq_len INT;
ALTER TABLE model_catalog ADD COLUMN IF NOT EXISTS global_batch_size INT;
ALTER TABLE model_catalog ADD COLUMN IF NOT EXISTS micro_batch_size INT;
ALTER TABLE model_catalog ADD COLUMN IF NOT EXISTS device_type VARCHAR(10);

-- MoE (Mixture of Experts) columns — NULL for dense models;
ALTER TABLE model_catalog ADD COLUMN IF NOT EXISTS num_experts INT;
ALTER TABLE model_catalog ADD COLUMN IF NOT EXISTS moe_ffn_hidden_size INT;
ALTER TABLE model_catalog ADD COLUMN IF NOT EXISTS moe_router_topk INT;
ALTER TABLE model_catalog ADD COLUMN IF NOT EXISTS moe_layer_freq TEXT;
ALTER TABLE model_catalog ADD COLUMN IF NOT EXISTS num_moe_layers INT;
ALTER TABLE model_catalog ADD COLUMN IF NOT EXISTS has_shared_expert BOOLEAN;
ALTER TABLE model_catalog ADD COLUMN IF NOT EXISTS shared_expert_intermediate_size INT;
ALTER TABLE model_catalog ADD COLUMN IF NOT EXISTS expert_tensor_parallel_size INT;
ALTER TABLE model_catalog ADD COLUMN IF NOT EXISTS ep INT;

ALTER TABLE sessions ADD COLUMN IF NOT EXISTS original_task_id VARCHAR(64);
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS equivalent_task_id VARCHAR(64);
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS formula_lines JSONB;

CREATE INDEX IF NOT EXISTS idx_topology_params_session ON topology_params(session_id, role);
CREATE INDEX IF NOT EXISTS idx_simulation_params_session ON simulation_params(session_id, role);
CREATE INDEX IF NOT EXISTS idx_simulation_results_session ON simulation_results(session_id, role);
CREATE INDEX IF NOT EXISTS idx_comparison_reports_session ON comparison_reports(session_id);
CREATE INDEX IF NOT EXISTS idx_conversation_messages_session ON conversation_messages(session_id);

-- comparison_reports 的唯一性：dao.save_comparison_report 用的是
-- `ON CONFLICT DO NOTHING`（无冲突目标）。PG 允许不带目标的写法，
-- 但 SQLite 要求给出冲突目标，因此两侧都必须补上这个约束，
-- 否则「同一 session 重复保存报告」在 SQLite 下会插入多行而非幂等。
CREATE UNIQUE INDEX IF NOT EXISTS uq_comparison_reports_session ON comparison_reports(session_id);
"""

# ── SQLite 方言 ──
# 与 PG 版的差异（全部是方言必需，无设计取舍）：
#   1. `UUID DEFAULT gen_random_uuid()` → `TEXT PRIMARY KEY`，**主键由 DAO 侧 uuid4() 生成**
#   2. `JSONB` → `TEXT`（DAO 自行 json.dumps/loads，读写形态与 PG 侧 psycopg2 默认行为一致）
#   3. `TIMESTAMP DEFAULT NOW()` → `DATETIME DEFAULT CURRENT_TIMESTAMP`
#   4. BOOLEAN → INTEGER（SQLite 无布尔类型；出口由 app.dbapi.normalize_row 归一为 bool）
#   5. 直接给**最终列集合**，不再需要 30 条 ADD COLUMN IF NOT EXISTS
#   6. 外键级联需要每连接 `PRAGMA foreign_keys=ON`（见 app.dbapi.configure_sqlite_connection）
SCHEMA_SQLITE_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
    id                  TEXT PRIMARY KEY,
    step                TEXT NOT NULL DEFAULT 'idle',
    original_task_id    TEXT,
    equivalent_task_id  TEXT,
    created_at          DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at          DATETIME DEFAULT CURRENT_TIMESTAMP,
    formula_lines       TEXT
);

CREATE TABLE IF NOT EXISTS topology_params (
    id              TEXT PRIMARY KEY,
    session_id      TEXT REFERENCES sessions(id) ON DELETE CASCADE,
    role            TEXT NOT NULL CHECK (role IN ('original', 'equivalent')),
    name            TEXT NOT NULL,
    device_type     TEXT NOT NULL,
    dp_size         INTEGER NOT NULL,
    tp_size         INTEGER NOT NULL,
    pp_size         INTEGER NOT NULL,
    total_nodes     INTEGER NOT NULL,
    model_name      TEXT,
    num_layers      INTEGER,
    hidden_dim      INTEGER,
    num_heads       INTEGER,
    d_ffn           INTEGER,
    seq_len         INTEGER,
    batch_size      INTEGER,
    micro_batch_size INTEGER,
    vocab_size      INTEGER,
    model_type      TEXT DEFAULT 'dense',
    num_experts     INTEGER,
    moe_router_topk INTEGER,
    num_moe_layers  INTEGER,
    moe_ffn_hidden_size INTEGER,
    has_shared_expert INTEGER DEFAULT 0,
    expert_tensor_parallel_size INTEGER DEFAULT 1,
    ep              INTEGER,
    UNIQUE (session_id, role)
);

CREATE TABLE IF NOT EXISTS simulation_params (
    id              TEXT PRIMARY KEY,
    session_id      TEXT REFERENCES sessions(id) ON DELETE CASCADE,
    role            TEXT NOT NULL CHECK (role IN ('original', 'equivalent')),
    script_path     TEXT,
    epoch_num       INTEGER DEFAULT 1,
    model_name      TEXT,
    device_type     TEXT,
    vocab_size      TEXT,
    frame           TEXT,
    rank            INTEGER DEFAULT 0,
    rank_range      INTEGER,
    comp_filepath   TEXT,
    no_time_accumulation INTEGER DEFAULT 0,
    level0_config   TEXT,
    level1_config   TEXT,
    visual_json_output   INTEGER DEFAULT 1,
    comm_group_output    INTEGER DEFAULT 1,
    debug_time      INTEGER DEFAULT 0,
    UNIQUE (session_id, role)
);

CREATE TABLE IF NOT EXISTS simulation_results (
    id              TEXT PRIMARY KEY,
    session_id      TEXT REFERENCES sessions(id) ON DELETE CASCADE,
    role            TEXT NOT NULL CHECK (role IN ('original', 'equivalent')),
    topology_name   TEXT,
    device_type     TEXT,
    total_nodes     INTEGER,
    is_simulated    INTEGER DEFAULT 0,
    cards           TEXT DEFAULT '[]',
    UNIQUE (session_id, role)
);

CREATE TABLE IF NOT EXISTS comparison_reports (
    id              TEXT PRIMARY KEY,
    session_id      TEXT REFERENCES sessions(id) ON DELETE CASCADE,
    original_id     TEXT REFERENCES simulation_results(id),
    equivalent_id   TEXT REFERENCES simulation_results(id),
    flops_diff_pct  REAL,
    hbm_diff_pct    REAL,
    tp_comm_diff_pct REAL,
    pp_comm_diff_pct REAL,
    dp_comm_diff_pct REAL,
    is_equivalent   INTEGER,
    error_tolerance REAL DEFAULT 5.0,
    details         TEXT,
    UNIQUE (session_id)
);

CREATE TABLE IF NOT EXISTS conversation_messages (
    id              TEXT PRIMARY KEY,
    session_id      TEXT REFERENCES sessions(id) ON DELETE CASCADE,
    msg_index       INTEGER NOT NULL,
    role            TEXT NOT NULL,
    content         TEXT,
    timestamp       DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS model_catalog (
    id              TEXT PRIMARY KEY,
    model_name      TEXT NOT NULL UNIQUE,
    name_key        TEXT NOT NULL UNIQUE,
    model_type      TEXT NOT NULL DEFAULT 'dense',
    num_layers      INTEGER NOT NULL,
    d_model         INTEGER NOT NULL,
    num_heads       INTEGER NOT NULL,
    d_ffn           INTEGER NOT NULL,
    vocab_size      INTEGER NOT NULL,
    num_kv_heads    INTEGER,
    source          TEXT,
    reference       TEXT,
    description     TEXT,
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
    tp              INTEGER,
    pp              INTEGER,
    dp              INTEGER,
    seq_len         INTEGER,
    global_batch_size INTEGER,
    micro_batch_size INTEGER,
    device_type     TEXT,
    num_experts     INTEGER,
    moe_ffn_hidden_size INTEGER,
    moe_router_topk INTEGER,
    moe_layer_freq  TEXT,
    num_moe_layers  INTEGER,
    has_shared_expert INTEGER,
    shared_expert_intermediate_size INTEGER,
    expert_tensor_parallel_size INTEGER,
    ep              INTEGER
);

CREATE INDEX IF NOT EXISTS idx_model_catalog_name_key ON model_catalog(name_key);
CREATE INDEX IF NOT EXISTS idx_topology_params_session ON topology_params(session_id, role);
CREATE INDEX IF NOT EXISTS idx_simulation_params_session ON simulation_params(session_id, role);
CREATE INDEX IF NOT EXISTS idx_simulation_results_session ON simulation_results(session_id, role);
CREATE INDEX IF NOT EXISTS idx_comparison_reports_session ON comparison_reports(session_id);
CREATE INDEX IF NOT EXISTS idx_conversation_messages_session ON conversation_messages(session_id);
"""

# SQLite 下「旧库补列」用的列定义表：`PRAGMA table_info` 探测到缺失才执行 ALTER TABLE。
# 只列 SQLite 允许 ALTER 追加的形态（不能加 UNIQUE / NOT NULL 无默认值）。
SQLITE_ADDABLE_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("sessions", "original_task_id", "TEXT"),
    ("sessions", "equivalent_task_id", "TEXT"),
    ("sessions", "formula_lines", "TEXT"),
    ("topology_params", "d_ffn", "INTEGER"),
    ("topology_params", "micro_batch_size", "INTEGER"),
    ("topology_params", "vocab_size", "INTEGER"),
    ("topology_params", "model_type", "TEXT DEFAULT 'dense'"),
    ("topology_params", "num_experts", "INTEGER"),
    ("topology_params", "moe_router_topk", "INTEGER"),
    ("topology_params", "num_moe_layers", "INTEGER"),
    ("topology_params", "moe_ffn_hidden_size", "INTEGER"),
    ("topology_params", "has_shared_expert", "INTEGER DEFAULT 0"),
    ("topology_params", "expert_tensor_parallel_size", "INTEGER DEFAULT 1"),
    ("topology_params", "ep", "INTEGER"),
    ("model_catalog", "reference", "TEXT"),
    ("model_catalog", "tp", "INTEGER"),
    ("model_catalog", "pp", "INTEGER"),
    ("model_catalog", "dp", "INTEGER"),
    ("model_catalog", "seq_len", "INTEGER"),
    ("model_catalog", "global_batch_size", "INTEGER"),
    ("model_catalog", "micro_batch_size", "INTEGER"),
    ("model_catalog", "device_type", "TEXT"),
    ("model_catalog", "num_experts", "INTEGER"),
    ("model_catalog", "moe_ffn_hidden_size", "INTEGER"),
    ("model_catalog", "moe_router_topk", "INTEGER"),
    ("model_catalog", "moe_layer_freq", "TEXT"),
    ("model_catalog", "num_moe_layers", "INTEGER"),
    ("model_catalog", "has_shared_expert", "INTEGER"),
    ("model_catalog", "shared_expert_intermediate_size", "INTEGER"),
    ("model_catalog", "expert_tensor_parallel_size", "INTEGER"),
    ("model_catalog", "ep", "INTEGER"),
)

# ── 迁移收尾硬校验 ──
# 这些是 dao 层实际读写的表与列。缺任何一个，服务「看起来能启动」但会在运行时
# 反复出错（最常见的症状是历史列表标题退化成「新建任务」）。
REQUIRED_TABLES = (
    "sessions",
    "topology_params",
    "simulation_params",
    "simulation_results",
    "comparison_reports",
    "conversation_messages",
    "model_catalog",
)

# 只校验最容易被漏掉的「后加列」：全表列校验噪音太大，也没必要。
REQUIRED_COLUMNS = (
    ("sessions", "formula_lines"),
    ("topology_params", "model_type"),
    ("simulation_results", "cards"),
    ("simulation_params", "level0_config"),
    ("model_catalog", "name_key"),
)

# 重放 DDL 时可预期的「无害」错误码，命中即放行（只打日志）。PG 方言专用：
#   42P07 duplicate_table / 42710 duplicate_object —— 并发建表/建索引
#   42701 duplicate_column                        —— 列已存在但语句没带 IF NOT EXISTS
#   42P16 invalid_table_definition                 —— 同名列 + 默认值差异
#   42P01 undefined_table                          —— 表由更早的语句创建失败（会被硬校验抓住）
#   23505 unique_violation                         —— 建索引时撞唯一约束
BENIGN_DDL_ERRORS = {"42P07", "42710", "42701", "42P16", "23505"}

# SQLite 方言的对应「无害」错误：sqlite3 不给错误码，只能匹配文案。
# 匹配刻意保守 —— 宁可把无害错误报成致命（会显式失败、容易发现），
# 也不要把真错误吞掉（那正是历史上「schema 不完整仍照常启动」的成因）。
BENIGN_SQLITE_ERRORS = (
    "already exists",
    "duplicate column name",
)


def _is_benign_sqlite_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(token in message for token in BENIGN_SQLITE_ERRORS)


def _active_backend() -> str:
    """当前后端。延迟 import，避免 `app.db` ←→ `app.db_migration` 循环依赖。"""
    from app.db import backend
    return backend()


def _preflight_sqlite(path: str) -> None:
    """SQLite 启动前可写性预检 —— 失败必须早且信息明确。

    原实现由 `docker/wait_for_db.py` 承担「等内嵌 PG 就绪 + 建库」；SQLite 无服务端、
    无库的概念，那一段整体消失，只留下这件真正需要的事：**目录不可写/未挂载要
    立刻报错**，而不是等 sqlite3 抛一句 `unable to open database file`。

    刻意**不自动创建父目录**：目录本该由镜像预建、部署侧挂载。自动创建会让数据库
    悄悄落在可写镜像层里，容器重建即丢全部会话历史（与 workspace 同源的失效模式）。
    """
    import tempfile
    from pathlib import Path

    target = Path(path)
    parent = target.parent
    if not parent.is_dir():
        raise RuntimeError(
            f"[migration] SQLite 数据目录不存在：{parent}\n"
            f"  SQLITE_PATH={path}\n"
            f"  提示：容器内该目录应由镜像预建并由部署侧挂载宿主机目录"
            f"（/home/aicm/db）。不自动创建，是为了避免数据库落在可写镜像层里。"
        )
    try:
        with tempfile.NamedTemporaryFile(dir=parent, prefix=".migcheck-", delete=True):
            pass
    except OSError as exc:
        raise RuntimeError(
            f"[migration] SQLite 数据目录不可写：{parent}（{exc}）\n"
            f"  SQLITE_PATH={path}\n"
            f"  提示：检查挂载目录属主是否与容器运行用户一致（k8s 下由 initContainer chown）。"
        ) from exc
    if target.exists() and not target.is_file():
        raise RuntimeError(f"[migration] SQLITE_PATH 已存在但不是普通文件：{path}")


def _schema_statements(backend: str) -> list[str]:
    """返回当前方言需要逐条执行的 DDL 语句。"""
    scheme = SCHEMA_SQL if backend == "postgres" else SCHEMA_SQLITE_SQL
    return [s.strip() for s in scheme.split(";") if s.strip() and not s.strip().startswith("--")]


def _read_existing_tables(cur, backend: str) -> set[str]:
    if backend == "postgres":
        cur.execute(
            """SELECT table_name FROM information_schema.tables
               WHERE table_schema = current_schema()"""
        )
    else:
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    return {row[0] for row in cur.fetchall()}


def _read_columns(cur, backend: str) -> set[tuple[str, str]]:
    if backend == "postgres":
        cur.execute(
            """SELECT table_name, column_name FROM information_schema.columns
               WHERE table_schema = current_schema()"""
        )
        return {(row[0], row[1]) for row in cur.fetchall()}

    columns: set[tuple[str, str]] = set()
    for table in REQUIRED_TABLES:
        cur.execute(f"PRAGMA table_info({table})")
        for row in cur.fetchall():
            # PRAGMA table_info: (cid, name, type, notnull, dflt_value, pk)
            columns.add((table, row[1]))
    return columns


def _verify_schema(cur, backend: str | None = None) -> list[str]:
    """返回缺失项描述列表；空列表表示校验通过。

    ⚠️ `backend` 省略时会按**进程的 `DATABASE_URL`** 侦测，而不是「这个 cursor 连的是什么」。
    两者在正常启动路径上一致（迁移与连接同源），但**直连别的库做校验时必须显式传入** ——
    否则会拿 SQLite 的取数方式（`sqlite_master`）去查 PG 连接（或反之），抛方言无关的
    报错。`scripts/pg_real_check.py` 走 ADMIN_DSN 直连，因此它显式传 `"postgres"`。
    仅当 cursor 就来自当前进程配置的那个后端时才可省略。
    """
    if backend is None:
        backend = _active_backend()
    missing: list[str] = []
    present = _read_existing_tables(cur, backend)
    missing.extend(f"表 {t}" for t in REQUIRED_TABLES if t not in present)

    columns = _read_columns(cur, backend)
    missing.extend(
        f"列 {t}.{c}" for t, c in REQUIRED_COLUMNS if (t, c) not in columns
    )
    return missing


def _ensure_sqlite_columns(cur, existing_columns: set[tuple[str, str]], skipped: list) -> None:
    """SQLite 不支持 `ADD COLUMN IF NOT EXISTS`，改为探测后动态补齐。

    新库走 `SCHEMA_SQLITE_SQL` 已经是最终列集合，这里只对「改造前遗留的旧
    SQLite 文件」生效，属于纯兼容路径。
    """
    for table, column, ddl_type in SQLITE_ADDABLE_COLUMNS:
        if (table, column) in existing_columns:
            continue
        try:
            cur.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl_type}")
            print(f"[migration] 补列 {table}.{column}")
        except Exception as e:
            if _is_benign_sqlite_error(e):
                skipped.append((f"ADD COLUMN {table}.{column}", str(e)))
                continue
            raise


def init_db():
    """Run migration to create all tables. 幂等，可在每次启动重放。"""
    from app.db import get_db

    active = _active_backend()
    if active == "sqlite":
        # 先做可写性预检：目录不可写时给出「哪里不对、怎么修」，而不是 SQLite 的
        # `unable to open database file`。原 wait_for_db.py 的等库语义在 SQLite 下
        # 无对应物，此预检即是它的替代。
        from app.config import config as _config
        _preflight_sqlite(_config.SQLITE_PATH)

    skipped: list[tuple[str, str]] = []   # (语句首行, 错误) —— 已放行的无害失败
    failed: list[tuple[str, str]] = []    # (语句首行, 错误) —— 需要人工关注的失败

    with get_db() as conn:
        # Enable autocommit so each DDL statement runs in its own
        # transaction — a failure in one won't abort the rest.
        conn.autocommit = True
        with conn.cursor() as cur:
            # 先记下迁移前已存在的表：用于区分「首次建库」与「重启复用已有库」。
            # 仅凭 DDL 是否报错区分不了 —— 语句都带 IF NOT EXISTS，重放会静默成功。
            pre_existing = _read_existing_tables(cur, active)

            # 两种方言的 execute() 都只处理一条语句 —— 按分号切分逐条执行。
            for stmt in _schema_statements(active):
                first_line = stmt.splitlines()[0].strip()[:80]
                try:
                    cur.execute(stmt)
                except Exception as e:
                    if active == "postgres":
                        benign = getattr(e, "pgcode", None) in BENIGN_DDL_ERRORS
                    else:
                        benign = _is_benign_sqlite_error(e)
                    if benign:
                        skipped.append((first_line, f"{e}"))
                        print(f"[migration] 已存在/可忽略: {e}")
                        continue
                    failed.append((first_line, f"{e}"))
                    print(f"[migration] FAIL: {e}  <- {first_line}")

            if active == "sqlite":
                _ensure_sqlite_columns(cur, _read_columns(cur, active), skipped)

            # 收尾硬校验：只看「语句是否报错」是不够的 —— 表可能压根没建出来，
            # 也可能被更早的失败连累。这里直接查 schema 定论。
            missing = _verify_schema(cur, active)
        conn.autocommit = False

    if skipped:
        print(f"[migration] 有 {len(skipped)} 条 DDL 可安全跳过（对象已存在）。")

    if failed or missing:
        detail = []
        if missing:
            detail.append("缺失对象：" + "、".join(missing))
        for stmt, err in failed:
            detail.append(f"执行失败：{stmt} -> {err}")
        raise RuntimeError(
            f"[migration] 数据库 schema 不完整（后端 {active}），拒绝以半可用状态启动服务。\n  "
            + "\n  ".join(detail)
            + "\n  排查提示：SQLite 后端确认 SQLITE_PATH 的父目录存在且可写；"
            + "PG 后端确认 DATABASE_URL 指向的库与用户名有 DDL 权限。"
        )

    # Seed model catalog entries (idempotent upsert).
    # Each category runs independently so one failure doesn't skip the rest.
    try:
        from app.dao import seed_model_catalog_builtin
        from app.models.model_catalog import (
            MINDSPEED_DENSE_MODELS, MEGATRON_DENSE_MODELS,
            MINDSPEED_MOE_MODELS, MEGATRON_MOE_MODELS,
        )

        def _safe_seed(label, models):
            try:
                return seed_model_catalog_builtin(models)
            except Exception as e:
                print(f"[migration] {label} seed skipped: {e}")
                return 0

        count1 = _safe_seed("mindspeed dense", MINDSPEED_DENSE_MODELS)
        count2 = _safe_seed("megatron dense", MEGATRON_DENSE_MODELS)
        count3 = _safe_seed("mindspeed moe", MINDSPEED_MOE_MODELS)
        count4 = _safe_seed("megatron moe", MEGATRON_MOE_MODELS)
        print(f"[migration] model_catalog seeded {count1} mindspeed dense + {count2} megatron dense + {count3} mindspeed moe + {count4} megatron moe models.")
    except Exception as e:
        print(f"[migration] model_catalog seed skipped: {e}")

    # 明确区分「首次建表」与「重启复用已有库」：两者的日志否则完全一样，
    # 排障时无法从容器的 stdout 判断数据卷是不是空的（这是很实际的需求）。
    created = [t for t in REQUIRED_TABLES if t not in pre_existing]
    if created:
        print(f"[migration] 首次建表完成，本次新建 {len(created)} 张：{'、'.join(created)}")
    else:
        print(
            f"[migration] 复用已有数据库（后端 {active}，{len(REQUIRED_TABLES)} 张表均已存在），"
            "迁移以幂等方式重放"
        )

    print(f"[migration] All tables created successfully. (backend={active})")


if __name__ == "__main__":
    init_db()
