"""Database migration script. Creates all tables if they don't exist.

改动说明（数据库内嵌化重构）：
  1. 删除了 5 条 `ALTER TABLE simulation_results DROP COLUMN IF EXISTS ...`。
     它们是一次性历史清理（把 total_flops / total_hbm / total_tp_comm /
     total_pp_comm / total_dp_comm 换成了 cards JSONB），却在**每次启动**都
     重新执行一遍：白拿 5 把表锁，且对已有部署毫无意义。需要清理旧库时手工执行。
  2. 不再静默吞 DDL 异常。原实现把每条失败只 `print` 出来，末尾打一句 WARNING
     就继续启动 —— 结果是「schema 不完整但服务照常跑」，直到有人发现历史标题
     全是「新建任务」。现在改为：可预期的重放类错误仍放行（打日志），
     **收尾硬校验核心表与关键列**，缺失即抛异常让容器启动失败。
"""

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
"""

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

# 重放 DDL 时可预期的「无害」错误码，命中即放行（只打日志）：
#   42P07 duplicate_table / 42710 duplicate_object —— 并发建表/建索引
#   42701 duplicate_column                        —— 列已存在但语句没带 IF NOT EXISTS
#   42P16 invalid_table_definition                 —— 同名列 + 默认值差异（IF NOT EXISTS 下不会触发）
#   42P01 undefined_table                          —— 表由更早的语句创建失败（会在收尾校验里被抓住）
#   23505 unique_violation                         —— 建索引时撞唯一约束
BENIGN_DDL_ERRORS = {"42P07", "42710", "42701", "42P16", "23505"}


def _verify_schema(cur) -> list[str]:
    """返回缺失项描述列表；空列表表示校验通过。"""
    missing: list[str] = []

    cur.execute(
        """SELECT table_name FROM information_schema.tables
           WHERE table_schema = current_schema()"""
    )
    present = {row[0] for row in cur.fetchall()}
    missing.extend(f"表 {t}" for t in REQUIRED_TABLES if t not in present)

    cur.execute(
        """SELECT table_name, column_name FROM information_schema.columns
           WHERE table_schema = current_schema()"""
    )
    columns = {(row[0], row[1]) for row in cur.fetchall()}
    missing.extend(
        f"列 {t}.{c}" for t, c in REQUIRED_COLUMNS if (t, c) not in columns
    )
    return missing


def init_db():
    """Run migration to create all tables (UUID PKs from the start)."""
    from app.db import get_db

    skipped: list[tuple[str, str]] = []   # (语句首行, 错误) —— 已放行的无害失败
    failed: list[tuple[str, str]] = []    # (语句首行, 错误) —— 需要人工关注的失败

    with get_db() as conn:
        # Enable autocommit so each DDL statement runs in its own
        # transaction — a failure in one won't abort the rest.
        conn.autocommit = True
        with conn.cursor() as cur:
            # psycopg2 execute() only handles one statement per call.
            # Split on semicolons and execute each individually.
            for stmt in SCHEMA_SQL.split(";"):
                stmt = stmt.strip()
                if not stmt or stmt.startswith("--"):
                    continue
                first_line = stmt.splitlines()[0].strip()[:80]
                try:
                    cur.execute(stmt)
                except Exception as e:
                    # psycopg2 errors expose .pgcode; treat a missing one as unknown.
                    code = getattr(e, "pgcode", None)
                    if code in BENIGN_DDL_ERRORS:
                        skipped.append((first_line, f"{e}"))
                        print(f"[migration] 已存在/可忽略: {e}")
                        continue
                    failed.append((first_line, f"{e}"))
                    print(f"[migration] FAIL: {e}  <- {first_line}")

            # 收尾硬校验：只看「语句是否报错」是不够的 —— 表可能压根没建出来，
            # 也可能被更早的失败连累。这里直接查 information_schema 定论。
            missing = _verify_schema(cur)
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
            "[migration] 数据库 schema 不完整，拒绝以半可用状态启动服务。\n  "
            + "\n  ".join(detail)
            + "\n  排查提示：内嵌模式下先确认 entrypoint 已完成 initdb 且目标库存在；"
            + "若是从外部 PG 迁入，确认 DATABASE_URL 指向的库与用户名有 DDL 权限。"
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

    print("[migration] All tables created successfully.")


if __name__ == "__main__":
    init_db()


