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

ALTER TABLE simulation_results DROP COLUMN IF EXISTS total_flops;
ALTER TABLE simulation_results DROP COLUMN IF EXISTS total_hbm;
ALTER TABLE simulation_results DROP COLUMN IF EXISTS total_tp_comm;
ALTER TABLE simulation_results DROP COLUMN IF EXISTS total_pp_comm;
ALTER TABLE simulation_results DROP COLUMN IF EXISTS total_dp_comm;

ALTER TABLE topology_params ADD COLUMN IF NOT EXISTS d_ffn INT;
ALTER TABLE topology_params ADD COLUMN IF NOT EXISTS micro_batch_size INT;

CREATE INDEX IF NOT EXISTS idx_topology_params_session ON topology_params(session_id, role);
CREATE INDEX IF NOT EXISTS idx_simulation_params_session ON simulation_params(session_id, role);
CREATE INDEX IF NOT EXISTS idx_simulation_results_session ON simulation_results(session_id, role);
CREATE INDEX IF NOT EXISTS idx_comparison_reports_session ON comparison_reports(session_id);
CREATE INDEX IF NOT EXISTS idx_conversation_messages_session ON conversation_messages(session_id);
"""

# Migration from SERIAL/INT PKs to UUID.
# Safe to run on a fresh DB (all ALTERs use IF EXISTS / IF NOT EXISTS).
MIGRATE_UUID_SQL = """
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.table_constraints
               WHERE constraint_name = 'comparison_reports_original_id_fkey') THEN
        ALTER TABLE comparison_reports DROP CONSTRAINT comparison_reports_original_id_fkey;
    END IF;
    IF EXISTS (SELECT 1 FROM information_schema.table_constraints
               WHERE constraint_name = 'comparison_reports_equivalent_id_fkey') THEN
        ALTER TABLE comparison_reports DROP CONSTRAINT comparison_reports_equivalent_id_fkey;
    END IF;
END $$;

ALTER TABLE topology_params ADD COLUMN IF NOT EXISTS new_id UUID DEFAULT gen_random_uuid();
ALTER TABLE topology_params DROP COLUMN IF EXISTS id CASCADE;
ALTER TABLE topology_params RENAME COLUMN new_id TO id;
ALTER TABLE topology_params ADD PRIMARY KEY (id);

ALTER TABLE simulation_params ADD COLUMN IF NOT EXISTS new_id UUID DEFAULT gen_random_uuid();
ALTER TABLE simulation_params DROP COLUMN IF EXISTS id CASCADE;
ALTER TABLE simulation_params RENAME COLUMN new_id TO id;
ALTER TABLE simulation_params ADD PRIMARY KEY (id);

ALTER TABLE simulation_results ADD COLUMN IF NOT EXISTS new_id UUID DEFAULT gen_random_uuid();
ALTER TABLE simulation_results DROP COLUMN IF EXISTS id CASCADE;
ALTER TABLE simulation_results RENAME COLUMN new_id TO id;
ALTER TABLE simulation_results ADD PRIMARY KEY (id);

ALTER TABLE comparison_reports ADD COLUMN IF NOT EXISTS new_original_id UUID;
ALTER TABLE comparison_reports ADD COLUMN IF NOT EXISTS new_equivalent_id UUID;
ALTER TABLE comparison_reports DROP COLUMN IF EXISTS original_id;
ALTER TABLE comparison_reports DROP COLUMN IF EXISTS equivalent_id;
ALTER TABLE comparison_reports RENAME COLUMN new_original_id TO original_id;
ALTER TABLE comparison_reports RENAME COLUMN new_equivalent_id TO equivalent_id;

ALTER TABLE comparison_reports ADD COLUMN IF NOT EXISTS new_id UUID DEFAULT gen_random_uuid();
ALTER TABLE comparison_reports DROP COLUMN IF EXISTS id CASCADE;
ALTER TABLE comparison_reports RENAME COLUMN new_id TO id;
ALTER TABLE comparison_reports ADD PRIMARY KEY (id);

ALTER TABLE conversation_messages ADD COLUMN IF NOT EXISTS new_id UUID DEFAULT gen_random_uuid();
ALTER TABLE conversation_messages DROP COLUMN IF EXISTS id CASCADE;
ALTER TABLE conversation_messages RENAME COLUMN new_id TO id;
ALTER TABLE conversation_messages ADD PRIMARY KEY (id);

ALTER TABLE comparison_reports ADD CONSTRAINT comparison_reports_original_id_fkey
    FOREIGN KEY (original_id) REFERENCES simulation_results(id);
ALTER TABLE comparison_reports ADD CONSTRAINT comparison_reports_equivalent_id_fkey
    FOREIGN KEY (equivalent_id) REFERENCES simulation_results(id);
"""


def init_db():
    """Run migration to create all tables and migrate SERIAL→UUID if needed."""
    from app.db import get_db

    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(SCHEMA_SQL)
            cur.execute(MIGRATE_UUID_SQL)
    print("[migration] All tables created and UUID migration applied successfully.")


if __name__ == "__main__":
    init_db()
