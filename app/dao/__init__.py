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
                   model_name, num_layers, hidden_dim, num_heads, d_ffn, seq_len, batch_size)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                   ON CONFLICT (session_id, role) DO UPDATE SET
                   name=EXCLUDED.name, device_type=EXCLUDED.device_type, dp_size=EXCLUDED.dp_size,
                   tp_size=EXCLUDED.tp_size, pp_size=EXCLUDED.pp_size, total_nodes=EXCLUDED.total_nodes,
                   model_name=EXCLUDED.model_name, num_layers=EXCLUDED.num_layers, hidden_dim=EXCLUDED.hidden_dim,
                   num_heads=EXCLUDED.num_heads, d_ffn=EXCLUDED.d_ffn,
                   seq_len=EXCLUDED.seq_len, batch_size=EXCLUDED.batch_size""",
                (
                    session_id, role,
                    data.get("name"), data.get("device_type"), data.get("dp_size"),
                    data.get("tp_size"), data.get("pp_size"), data.get("total_nodes"),
                    data.get("model_name"), data.get("num_layers"), data.get("hidden_dim"),
                    data.get("num_heads"), data.get("d_ffn"),
                    data.get("seq_len"), data.get("batch_size"),
                ),
            )


def get_topology_params(session_id: str, role: str) -> dict[str, Any] | None:
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT name, device_type, dp_size, tp_size, pp_size, total_nodes, model_name, num_layers, hidden_dim, num_heads, d_ffn, seq_len, batch_size FROM topology_params WHERE session_id=%s AND role=%s",
                (session_id, role),
            )
            row = cur.fetchone()
    if not row:
        return None
    keys = ["name", "device_type", "dp_size", "tp_size", "pp_size", "total_nodes", "model_name", "num_layers", "hidden_dim", "num_heads", "d_ffn", "seq_len", "batch_size"]
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
