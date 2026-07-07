"""Database migration script. Creates all tables if they don't exist."""

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

ALTER TABLE simulation_results DROP COLUMN IF EXISTS total_flops;
ALTER TABLE simulation_results DROP COLUMN IF EXISTS total_hbm;
ALTER TABLE simulation_results DROP COLUMN IF EXISTS total_tp_comm;
ALTER TABLE simulation_results DROP COLUMN IF EXISTS total_pp_comm;
ALTER TABLE simulation_results DROP COLUMN IF EXISTS total_dp_comm;

ALTER TABLE topology_params ADD COLUMN IF NOT EXISTS d_ffn INT;
ALTER TABLE topology_params ADD COLUMN IF NOT EXISTS micro_batch_size INT;

ALTER TABLE model_catalog ADD COLUMN IF NOT EXISTS reference TEXT;

ALTER TABLE model_catalog ADD COLUMN IF NOT EXISTS tp INT;
ALTER TABLE model_catalog ADD COLUMN IF NOT EXISTS pp INT;
ALTER TABLE model_catalog ADD COLUMN IF NOT EXISTS dp INT;
ALTER TABLE model_catalog ADD COLUMN IF NOT EXISTS seq_len INT;
ALTER TABLE model_catalog ADD COLUMN IF NOT EXISTS global_batch_size INT;
ALTER TABLE model_catalog ADD COLUMN IF NOT EXISTS micro_batch_size INT;
ALTER TABLE model_catalog ADD COLUMN IF NOT EXISTS device_type VARCHAR(10);

CREATE INDEX IF NOT EXISTS idx_topology_params_session ON topology_params(session_id, role);
CREATE INDEX IF NOT EXISTS idx_simulation_params_session ON simulation_params(session_id, role);
CREATE INDEX IF NOT EXISTS idx_simulation_results_session ON simulation_results(session_id, role);
CREATE INDEX IF NOT EXISTS idx_comparison_reports_session ON comparison_reports(session_id);
CREATE INDEX IF NOT EXISTS idx_conversation_messages_session ON conversation_messages(session_id);
"""

def init_db():
    """Run migration to create all tables (UUID PKs from the start)."""
    from app.db import get_db

    with get_db() as conn:
        with conn.cursor() as cur:
            # psycopg2 execute() only handles one statement per call.
            # Split on semicolons and execute each individually.
            for stmt in SCHEMA_SQL.split(";"):
                stmt = stmt.strip()
                if stmt and not stmt.startswith("--"):
                    try:
                        cur.execute(stmt)
                    except Exception as e:
                        print(f"[migration] SKIP: {e}")

    # Seed model catalog entries (idempotent upsert)
    try:
        from app.dao import seed_model_catalog_builtin
        from app.models.model_catalog import MINDSPEED_DENSE_MODELS, MEGATRON_DENSE_MODELS
        count1 = seed_model_catalog_builtin(MINDSPEED_DENSE_MODELS)
        count2 = seed_model_catalog_builtin(MEGATRON_DENSE_MODELS)
        print(f"[migration] model_catalog seeded {count1} mindspeed + {count2} megatron models.")
    except Exception as e:
        print(f"[migration] model_catalog seed skipped: {e}")

    print("[migration] All tables created successfully.")


if __name__ == "__main__":
    init_db()
