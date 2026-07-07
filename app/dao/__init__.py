"""Data access layer for PostgreSQL persistence."""

from __future__ import annotations
import json
from typing import Any
from datetime import datetime, timezone

from app.db import get_db


# ── sessions ──

def create_session(session_id: str) -> None:
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO sessions (id) VALUES (%s) ON CONFLICT DO NOTHING",
                (session_id,),
            )


def update_session_step(session_id: str, step: str, original_task_id: str | None = None, equivalent_task_id: str | None = None) -> None:
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """UPDATE sessions SET step=%s, original_task_id=COALESCE(%s, original_task_id),
                   equivalent_task_id=COALESCE(%s, equivalent_task_id), updated_at=NOW()
                   WHERE id=%s""",
                (step, original_task_id, equivalent_task_id, session_id),
            )


def delete_session(session_id: str) -> None:
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM sessions WHERE id=%s", (session_id,))


def list_session_ids() -> list[dict[str, Any]]:
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id, step, created_at, updated_at FROM sessions ORDER BY updated_at DESC")
            rows = cur.fetchall()
    return [{"session_id": r[0], "step": r[1], "created_at": r[2].isoformat() if r[2] else None, "updated_at": r[3].isoformat() if r[3] else None} for r in rows]


def get_session_summaries() -> list[dict[str, Any]]:
    """Return lightweight session list for history panel, with topology-derived titles."""
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT s.id, s.step, s.created_at, s.updated_at,
                       o.name AS orig_name, e.name AS eq_name
                FROM sessions s
                LEFT JOIN topology_params o ON o.session_id = s.id AND o.role = 'original'
                LEFT JOIN topology_params e ON e.session_id = s.id AND e.role = 'equivalent'
                ORDER BY s.updated_at DESC
            """)
            rows = cur.fetchall()
    result = []
    for r in rows:
        orig_name = r[4]
        eq_name = r[5]
        if orig_name and eq_name:
            title = f"{orig_name} → {eq_name}"
        elif orig_name:
            title = orig_name
        elif eq_name:
            title = eq_name
        else:
            title = "新建任务"
        result.append({
            "session_id": r[0],
            "title": title,
            "step": r[1],
            "created_at": r[2].isoformat() if r[2] else None,
            "updated_at": r[3].isoformat() if r[3] else None,
        })
    return result


def get_session_summary(session_id: str) -> dict[str, Any] | None:
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id, step, original_task_id, equivalent_task_id, created_at, updated_at FROM sessions WHERE id=%s", (session_id,))
            row = cur.fetchone()
    if not row:
        return None
    return {"session_id": row[0], "step": row[1], "original_task_id": row[2], "equivalent_task_id": row[3], "created_at": row[4].isoformat() if row[4] else None, "updated_at": row[5].isoformat() if row[5] else None}


# ── topology_params ──

def save_topology_params(session_id: str, role: str, data: dict[str, Any]) -> None:
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO topology_params (session_id, role, name, device_type, dp_size, tp_size, pp_size, total_nodes,
                   model_name, num_layers, hidden_dim, num_heads, d_ffn, seq_len, batch_size, micro_batch_size)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                   ON CONFLICT (session_id, role) DO UPDATE SET
                   name=EXCLUDED.name, device_type=EXCLUDED.device_type, dp_size=EXCLUDED.dp_size,
                   tp_size=EXCLUDED.tp_size, pp_size=EXCLUDED.pp_size, total_nodes=EXCLUDED.total_nodes,
                   model_name=EXCLUDED.model_name, num_layers=EXCLUDED.num_layers, hidden_dim=EXCLUDED.hidden_dim,
                   num_heads=EXCLUDED.num_heads, d_ffn=EXCLUDED.d_ffn,
                   seq_len=EXCLUDED.seq_len, batch_size=EXCLUDED.batch_size,
                   micro_batch_size=EXCLUDED.micro_batch_size""",
                (
                    session_id, role,
                    data.get("name"), data.get("device_type"), data.get("dp_size"),
                    data.get("tp_size"), data.get("pp_size"), data.get("total_nodes"),
                    data.get("model_name"), data.get("num_layers"), data.get("hidden_dim"),
                    data.get("num_heads"), data.get("d_ffn"),
                    data.get("seq_len"), data.get("batch_size"), data.get("micro_batch_size"),
                ),
            )


def get_topology_params(session_id: str, role: str) -> dict[str, Any] | None:
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT name, device_type, dp_size, tp_size, pp_size, total_nodes, model_name, num_layers, hidden_dim, num_heads, d_ffn, seq_len, batch_size, micro_batch_size FROM topology_params WHERE session_id=%s AND role=%s",
                (session_id, role),
            )
            row = cur.fetchone()
    if not row:
        return None
    keys = ["name", "device_type", "dp_size", "tp_size", "pp_size", "total_nodes", "model_name", "num_layers", "hidden_dim", "num_heads", "d_ffn", "seq_len", "batch_size", "micro_batch_size"]
    return dict(zip(keys, row))


# ── simulation_params ──

def save_simulation_params(session_id: str, role: str, data: dict[str, Any]) -> None:
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO simulation_params (session_id, role, script_path, epoch_num, model_name, device_type,
                   vocab_size, frame, rank, rank_range, comp_filepath, no_time_accumulation,
                   level0_config, level1_config, visual_json_output, comm_group_output, debug_time)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                   ON CONFLICT (session_id, role) DO UPDATE SET
                   script_path=EXCLUDED.script_path, epoch_num=EXCLUDED.epoch_num, model_name=EXCLUDED.model_name,
                   device_type=EXCLUDED.device_type, vocab_size=EXCLUDED.vocab_size, frame=EXCLUDED.frame,
                   rank=EXCLUDED.rank, rank_range=EXCLUDED.rank_range, comp_filepath=EXCLUDED.comp_filepath,
                   no_time_accumulation=EXCLUDED.no_time_accumulation, level0_config=EXCLUDED.level0_config,
                   level1_config=EXCLUDED.level1_config, visual_json_output=EXCLUDED.visual_json_output,
                   comm_group_output=EXCLUDED.comm_group_output, debug_time=EXCLUDED.debug_time""",
                (
                    session_id, role,
                    data.get("script_path"), data.get("epoch_num", 1), data.get("model_name", ""),
                    data.get("device_type"), data.get("vocab_size"), data.get("frame"),
                    data.get("rank", 0), data.get("rank_range"), data.get("comp_filepath"),
                    data.get("no_time_accumulation", False),
                    json.dumps(data.get("level0_config")) if data.get("level0_config") else None,
                    json.dumps(data.get("level1_config")) if data.get("level1_config") else None,
                    data.get("visual_json_output", True), data.get("comm_group_output", True),
                    data.get("debug_time", False),
                ),
            )


def get_simulation_params(session_id: str, role: str) -> dict[str, Any] | None:
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT script_path, epoch_num, model_name, device_type, vocab_size, frame, rank, rank_range, comp_filepath, no_time_accumulation, level0_config, level1_config, visual_json_output, comm_group_output, debug_time FROM simulation_params WHERE session_id=%s AND role=%s",
                (session_id, role),
            )
            row = cur.fetchone()
    if not row:
        return None
    keys = ["script_path", "epoch_num", "model_name", "device_type", "vocab_size", "frame", "rank", "rank_range", "comp_filepath", "no_time_accumulation", "level0_config", "level1_config", "visual_json_output", "comm_group_output", "debug_time"]
    result = dict(zip(keys, row))
    for k in ("level0_config", "level1_config"):
        if result[k] and isinstance(result[k], str):
            result[k] = json.loads(result[k])
    return result


# ── simulation_results ──

def save_simulation_result(session_id: str, role: str, data: dict[str, Any]) -> str:
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO simulation_results (session_id, role, topology_name, device_type, total_nodes,
                   is_simulated, cards)
                   VALUES (%s,%s,%s,%s,%s,%s,%s)
                   ON CONFLICT (session_id, role) DO UPDATE SET
                   topology_name=EXCLUDED.topology_name, device_type=EXCLUDED.device_type,
                   total_nodes=EXCLUDED.total_nodes,
                   is_simulated=EXCLUDED.is_simulated, cards=EXCLUDED.cards
                   RETURNING id""",
                (
                    session_id, role,
                    data.get("topology_name"), data.get("device_type"), data.get("total_nodes"),
                    data.get("is_simulated", False),
                    json.dumps(data.get("cards", [])),
                ),
            )
            row = cur.fetchone()
    return row[0]


def get_simulation_result(session_id: str, role: str) -> dict[str, Any] | None:
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, topology_name, device_type, total_nodes, is_simulated, cards FROM simulation_results WHERE session_id=%s AND role=%s",
                (session_id, role),
            )
            row = cur.fetchone()
    if not row:
        return None
    keys = ["id", "topology_name", "device_type", "total_nodes", "is_simulated", "cards"]
    result = dict(zip(keys, row))
    if isinstance(result.get("cards"), str):
        result["cards"] = json.loads(result["cards"])
    return result


# ── comparison_reports ──

def save_comparison_report(session_id: str, original_id: str, equivalent_id: str, data: dict[str, Any]) -> None:
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO comparison_reports (session_id, original_id, equivalent_id, flops_diff_pct,
                   hbm_diff_pct, tp_comm_diff_pct, pp_comm_diff_pct, dp_comm_diff_pct,
                   is_equivalent, error_tolerance, details)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                   ON CONFLICT DO NOTHING""",
                (
                    session_id, original_id, equivalent_id,
                    data.get("flops_diff_pct"), data.get("hbm_diff_pct"),
                    data.get("tp_comm_diff_pct"), data.get("pp_comm_diff_pct"),
                    data.get("dp_comm_diff_pct"), data.get("is_equivalent"),
                    data.get("error_tolerance", 5.0), json.dumps(data.get("details", {})),
                ),
            )


def get_comparison_report(session_id: str) -> dict[str, Any] | None:
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT flops_diff_pct, hbm_diff_pct, tp_comm_diff_pct, pp_comm_diff_pct, dp_comm_diff_pct, is_equivalent, error_tolerance, details FROM comparison_reports WHERE session_id=%s",
                (session_id,),
            )
            row = cur.fetchone()
    if not row:
        return None
    keys = ["flops_diff_pct", "hbm_diff_pct", "tp_comm_diff_pct", "pp_comm_diff_pct", "dp_comm_diff_pct", "is_equivalent", "error_tolerance", "details"]
    result = dict(zip(keys, row))
    if isinstance(result.get("details"), str):
        result["details"] = json.loads(result["details"])
    return result


# ── conversation_messages ──

def save_message(session_id: str, msg_index: int, role: str, content: str) -> None:
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO conversation_messages (session_id, msg_index, role, content) VALUES (%s,%s,%s,%s) ON CONFLICT DO NOTHING",
                (session_id, msg_index, role, content),
            )


def get_messages(session_id: str, limit: int = 20) -> list[dict[str, Any]]:
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT role, content FROM (SELECT role, content, msg_index FROM conversation_messages WHERE session_id=%s ORDER BY msg_index DESC LIMIT %s) sub ORDER BY msg_index ASC",
                (session_id, limit),
            )
            rows = cur.fetchall()
    result = []
    for r in rows:
        try:
            msg = json.loads(r[1])
            if isinstance(msg, dict):
                result.append(msg)
            else:
                result.append({"role": r[0], "content": r[1]})
        except (json.JSONDecodeError, TypeError):
            result.append({"role": r[0], "content": r[1]})
    return result


def delete_messages(session_id: str) -> None:
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM conversation_messages WHERE session_id=%s", (session_id,))


# ── model_catalog ──

def _name_key(name: str) -> str:
    """Normalize a model name for fuzzy matching: lowercase, strip - _ / space."""
    return (name or "").lower().replace("-", "").replace("_", "").replace("/", "").replace(" ", "")


def get_model_catalog_entry(model_name: str) -> dict[str, Any] | None:
    """Look up a model by exact (case-insensitive) name or normalized name_key.

    When model_name contains '/' (e.g. Qwen/Qwen2.5-72B), we also try the
    bare name (Qwen2.5-72B) so LLM-auto-expanded repo-ids still hit the
    builtin catalog entries that are stored under bare names.
    """
    if not model_name:
        return None
    # Build candidate name_keys: full name + bare name (strip org prefix)
    nk_candidates = {_name_key(model_name)}
    if "/" in model_name:
        bare = model_name.rsplit("/", 1)[-1]
        nk_candidates.add(_name_key(bare))
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT model_name, model_type, num_layers, d_model, num_heads,
                          d_ffn, vocab_size, num_kv_heads, source, reference
                   FROM model_catalog
                   WHERE model_name ILIKE %s OR name_key = ANY(%s)
                   ORDER BY
                     CASE source
                       WHEN 'mindspeed' THEN 1
                       WHEN 'megatron' THEN 2
                       WHEN 'huggingface' THEN 3
                       WHEN 'modelscope' THEN 4
                       ELSE 5
                     END,
                     (model_name ILIKE %s) DESC
                   LIMIT 1""",
                (model_name, list(nk_candidates), model_name),
            )
            row = cur.fetchone()
    if not row:
        return None
    source = row[8] or "pg"
    reference = row[9]
    # Auto-generate reference URL for remote-sourced models that were cached
    # before the reference column was added (backfill on read)
    if not reference and source == "huggingface":
        reference = f"https://huggingface.co/{row[0]}"
    elif not reference and source == "modelscope":
        reference = f"https://modelscope.cn/models/{row[0]}"
    return {
        "model_name": row[0],
        "model_type": row[1],
        "num_layers": row[2],
        "d_model": row[3],
        "num_heads": row[4],
        "d_ffn": row[5],
        "vocab_size": row[6],
        "num_key_value_heads": row[7],
        "_source": source,
        "reference": reference,
    }


def upsert_model_catalog(
    model_name: str, cfg: dict[str, Any], description: str | None = None
) -> None:
    """Insert or update a single model catalog entry.

    cfg is the internal-shape dict produced by the resolver (num_layers,
    d_model, num_heads, d_ffn, vocab_size, model_type, optional
    num_key_value_heads, _source). description is an optional admin note;
    on update a NULL description preserves the existing value (so resolver
    upserts never wipe a manually-set description).
    """
    if not model_name or not cfg:
        return
    nk = _name_key(model_name)
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO model_catalog
                   (model_name, name_key, model_type, num_layers, d_model, num_heads,
                    d_ffn, vocab_size, num_kv_heads, source, reference, description)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                   ON CONFLICT (model_name) DO UPDATE SET
                   name_key=EXCLUDED.name_key, model_type=EXCLUDED.model_type,
                   num_layers=EXCLUDED.num_layers, d_model=EXCLUDED.d_model,
                   num_heads=EXCLUDED.num_heads, d_ffn=EXCLUDED.d_ffn,
                   vocab_size=EXCLUDED.vocab_size, num_kv_heads=EXCLUDED.num_kv_heads,
                   source=EXCLUDED.source,
                   reference=EXCLUDED.reference,
                   description=COALESCE(EXCLUDED.description, model_catalog.description),
                   updated_at=NOW()""",
                (
                    model_name, nk, cfg.get("model_type", "dense"),
                    cfg["num_layers"], cfg["d_model"], cfg["num_heads"],
                    cfg["d_ffn"], cfg["vocab_size"], cfg.get("num_key_value_heads"),
                    cfg.get("_source"), cfg.get("reference"), description,
                ),
            )


def list_model_catalog() -> list[dict[str, Any]]:
    """Return all model catalog entries (for admin/management views)."""
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT model_name, model_type, num_layers, d_model, num_heads,
                          d_ffn, vocab_size, num_kv_heads, source, reference, description, updated_at
                   FROM model_catalog ORDER BY model_name"""
            )
            rows = cur.fetchall()
    return [{
        "model_name": r[0], "model_type": r[1], "num_layers": r[2], "d_model": r[3],
        "num_heads": r[4], "d_ffn": r[5], "vocab_size": r[6], "num_key_value_heads": r[7],
        "source": r[8], "reference": r[9], "description": r[10],
        "updated_at": r[11].isoformat() if r[11] else None,
    } for r in rows]


def delete_model_catalog_entry(model_name: str) -> bool:
    """Delete a model catalog entry by exact name. Returns True if a row was deleted."""
    if not model_name:
        return False
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM model_catalog WHERE model_name ILIKE %s", (model_name,)
            )
            return cur.rowcount > 0


def seed_model_catalog_builtin(entries: dict[str, dict[str, Any]]) -> int:
    """Bulk-upsert model catalog entries. Source comes from each entry's _source field
    (e.g. 'megatron', 'mindspeed'); falls back to 'builtin' for backward compat."""
    if not entries:
        return 0
    rows = [
        (
            name, _name_key(name), cfg.get("model_type", "dense"),
            cfg["num_layers"], cfg["d_model"], cfg["num_heads"],
            cfg["d_ffn"], cfg["vocab_size"], cfg.get("num_key_value_heads"),
            cfg.get("_source") or "builtin", cfg.get("reference"),
        )
        for name, cfg in entries.items()
    ]
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.executemany(
                """INSERT INTO model_catalog
                   (model_name, name_key, model_type, num_layers, d_model, num_heads,
                    d_ffn, vocab_size, num_kv_heads, source, reference)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                   ON CONFLICT (model_name) DO UPDATE SET
                   name_key=EXCLUDED.name_key, model_type=EXCLUDED.model_type,
                   num_layers=EXCLUDED.num_layers, d_model=EXCLUDED.d_model,
                   num_heads=EXCLUDED.num_heads, d_ffn=EXCLUDED.d_ffn,
                   vocab_size=EXCLUDED.vocab_size, num_kv_heads=EXCLUDED.num_kv_heads,
                   source=EXCLUDED.source, reference=EXCLUDED.reference, updated_at=NOW()""",
                rows,
            )
    return len(rows)
