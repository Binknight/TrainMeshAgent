"""Data access layer for dual-backend persistence (SQLite default / PostgreSQL).

方言差异由 `app.dbapi` 在语句下发前处理（`%s`→`?`、`NOW()`→`CURRENT_TIMESTAMP`、
`EXCLUDED`→`excluded`），因此本层的 SQL 文案保持与迁移前一致，改动集中在：

  1. **主键由这里的 `uuid4()` 生成**。SQLite 没有 `gen_random_uuid()`，
     `id` 列是普通 `TEXT PRIMARY KEY`（无默认值），必须显式传值。
     这也让两种后端产出同一种主键形态（TEXT/UUID 字符串）。
  2. **布尔列的出口归一**（`normalize_row`）。SQLite 无布尔类型，写入 True/False
     读出是 1/0；对外 JSON 形态必须稳定，否则前端 `=== true` 之类的判断会坏。
  3. **JSON 列统一按「可能是 str」处理**。PG 的 psycopg2 默认不解析 JSONB（读回 str），
     SQLite 下就是 TEXT，因此两侧行为一致，解析逻辑无需再分方言。
  4. `ILIKE` → `LIKE`、`name_key = ANY(%s)` → `name_key IN (...)`：SQLite 没有这两个构造。
"""

from __future__ import annotations
import json
import uuid
from typing import Any

from app.db import get_db
from app.dbapi import normalize_row


def _new_id() -> str:
    """主键。两种后端都用 UUID 字符串：SQLite 无 gen_random_uuid()，必须应用侧生成。"""
    return str(uuid.uuid4())


def _iso(value: Any) -> str | None:
    """时间戳出口归一。

    PG 返回 datetime、SQLite 返回 'YYYY-MM-DD HH:MM:SS' 字符串（CURRENT_TIMESTAMP
    的格式）。对外一律是 ISO 字符串，避免下游 `jsonify` 拿到两种形态。
    """
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    text = str(value).strip()
    if not text:
        return None
    # SQLite 的 'YYYY-MM-DD HH:MM:SS' → 'YYYY-MM-DDTHH:MM:SS'
    if len(text) >= 19 and text[10] == " ":
        return text[:10] + "T" + text[11:]
    return text


def _loads(value: Any, default: Any = None) -> Any:
    """JSON 列出口：两种后端读回都是 str（PG 的 JSONB 默认不解析），存 None 时给默认值。"""
    if value is None:
        return default
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return default
    return value


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
    return [{"session_id": r[0], "step": r[1], "created_at": _iso(r[2]), "updated_at": _iso(r[3])} for r in rows]


def get_session_summaries() -> list[dict[str, Any]]:
    """Return lightweight session list for history panel, with topology-derived titles."""
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT s.id, s.step, s.created_at, s.updated_at,
                       o.name AS orig_name, e.name AS eq_name,
                       o.model_name AS orig_model, e.model_name AS eq_model,
                       o.model_type AS orig_type, e.model_type AS eq_type
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
        # 模型名称/类型：优先取原始组网；等效侧由 save_session 统一加 "_eq" 后缀
        # （见 app/agent/session.py），仅作兜底时展示、并去掉该内部标记。
        # model_type 与 model_name 取同一侧的行；旧库行可能为 NULL，按列默认值
        # 归一为 dense（MoE 列上线之前的会话均为稠密模型）。
        if r[6]:
            model_name, model_type = r[6], r[8]
        else:
            model_name, model_type = r[7], r[9]
        if model_name and model_name.endswith("_eq"):
            model_name = model_name[:-3]
        model_type = (model_type or "dense") if model_name else None
        result.append({
            "session_id": r[0],
            "title": title,
            "step": r[1],
            "model_name": model_name,
            "model_type": model_type,
            "created_at": _iso(r[2]),
            "updated_at": _iso(r[3]),
        })
    return result


def get_session_summary(session_id: str) -> dict[str, Any] | None:
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id, step, original_task_id, equivalent_task_id, formula_lines, created_at, updated_at FROM sessions WHERE id=%s", (session_id,))
            row = cur.fetchone()
    if not row:
        return None
    return {"session_id": row[0], "step": row[1], "original_task_id": row[2], "equivalent_task_id": row[3], "formula_lines": _loads(row[4]), "created_at": _iso(row[5]), "updated_at": _iso(row[6])}


def save_formula_lines(session_id: str, formula_lines: list[dict[str, Any]]) -> None:
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE sessions SET formula_lines=%s, updated_at=NOW() WHERE id=%s",
                (json.dumps(formula_lines, ensure_ascii=False), session_id),
            )


# ── topology_params ──

def save_topology_params(session_id: str, role: str, data: dict[str, Any]) -> None:
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO topology_params (id, session_id, role, name, device_type, dp_size, tp_size, pp_size, total_nodes,
                   model_name, num_layers, hidden_dim, num_heads, d_ffn, seq_len, batch_size, micro_batch_size, vocab_size,
                   model_type, num_experts, moe_router_topk, num_moe_layers, moe_ffn_hidden_size, has_shared_expert, expert_tensor_parallel_size, ep)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                   ON CONFLICT (session_id, role) DO UPDATE SET
                   name=EXCLUDED.name, device_type=EXCLUDED.device_type, dp_size=EXCLUDED.dp_size,
                   tp_size=EXCLUDED.tp_size, pp_size=EXCLUDED.pp_size, total_nodes=EXCLUDED.total_nodes,
                   model_name=EXCLUDED.model_name, num_layers=EXCLUDED.num_layers, hidden_dim=EXCLUDED.hidden_dim,
                   num_heads=EXCLUDED.num_heads, d_ffn=EXCLUDED.d_ffn,
                   seq_len=EXCLUDED.seq_len, batch_size=EXCLUDED.batch_size,
                   micro_batch_size=EXCLUDED.micro_batch_size, vocab_size=EXCLUDED.vocab_size,
                   model_type=EXCLUDED.model_type, num_experts=EXCLUDED.num_experts,
                   moe_router_topk=EXCLUDED.moe_router_topk, num_moe_layers=EXCLUDED.num_moe_layers,
                   moe_ffn_hidden_size=EXCLUDED.moe_ffn_hidden_size, has_shared_expert=EXCLUDED.has_shared_expert,
                   expert_tensor_parallel_size=EXCLUDED.expert_tensor_parallel_size, ep=EXCLUDED.ep""",
                (
                    _new_id(), session_id, role,
                    data.get("name"), data.get("device_type"), data.get("dp_size"),
                    data.get("tp_size"), data.get("pp_size"), data.get("total_nodes"),
                    data.get("model_name"), data.get("num_layers"), data.get("hidden_dim"),
                    data.get("num_heads"), data.get("d_ffn"),
                    data.get("seq_len"), data.get("batch_size"), data.get("micro_batch_size"),
                    data.get("vocab_size"),
                    data.get("model_type"), data.get("num_experts"),
                    data.get("moe_router_topk"), data.get("num_moe_layers"),
                    data.get("moe_ffn_hidden_size"), data.get("has_shared_expert"),
                    data.get("expert_tensor_parallel_size"), data.get("ep"),
                ),
            )


def get_topology_params(session_id: str, role: str) -> dict[str, Any] | None:
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT name, device_type, dp_size, tp_size, pp_size, total_nodes, model_name, num_layers, hidden_dim, num_heads, d_ffn, seq_len, batch_size, micro_batch_size, vocab_size, model_type, num_experts, moe_router_topk, num_moe_layers, moe_ffn_hidden_size, has_shared_expert, expert_tensor_parallel_size, ep FROM topology_params WHERE session_id=%s AND role=%s",
                (session_id, role),
            )
            row = cur.fetchone()
    if not row:
        return None
    keys = ["name", "device_type", "dp_size", "tp_size", "pp_size", "total_nodes", "model_name", "num_layers", "hidden_dim", "num_heads", "d_ffn", "seq_len", "batch_size", "micro_batch_size", "vocab_size", "model_type", "num_experts", "moe_router_topk", "num_moe_layers", "moe_ffn_hidden_size", "has_shared_expert", "expert_tensor_parallel_size", "ep"]
    return normalize_row(dict(zip(keys, row)))


# ── simulation_params ──

def save_simulation_params(session_id: str, role: str, data: dict[str, Any]) -> None:
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO simulation_params (id, session_id, role, script_path, epoch_num, model_name, device_type,
                   vocab_size, frame, rank, rank_range, comp_filepath, no_time_accumulation,
                   level0_config, level1_config, visual_json_output, comm_group_output, debug_time)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                   ON CONFLICT (session_id, role) DO UPDATE SET
                   script_path=EXCLUDED.script_path, epoch_num=EXCLUDED.epoch_num, model_name=EXCLUDED.model_name,
                   device_type=EXCLUDED.device_type, vocab_size=EXCLUDED.vocab_size, frame=EXCLUDED.frame,
                   rank=EXCLUDED.rank, rank_range=EXCLUDED.rank_range, comp_filepath=EXCLUDED.comp_filepath,
                   no_time_accumulation=EXCLUDED.no_time_accumulation, level0_config=EXCLUDED.level0_config,
                   level1_config=EXCLUDED.level1_config, visual_json_output=EXCLUDED.visual_json_output,
                   comm_group_output=EXCLUDED.comm_group_output, debug_time=EXCLUDED.debug_time""",
                (
                    _new_id(), session_id, role,
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
    result = normalize_row(dict(zip(keys, row)))
    for k in ("level0_config", "level1_config"):
        if result[k]:
            result[k] = _loads(result[k])
    return result


# ── simulation_results ──

def save_simulation_result(session_id: str, role: str, data: dict[str, Any]) -> str:
    """Upsert 并返回该行的主键。

    主键由应用侧生成，因此 upsert 命中冲突时**不能**图省事用新 uuid 覆盖 ——
    `comparison_reports.original_id/equivalent_id` 外键引用的是这里返回的值。
    先查一次已存在的 id（(session_id, role) 上有 UNIQUE 索引，代价极小），
    没有才新建；这样 upsert 语义与「同一会话只有一行」的实际数据形态一致。
    """
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM simulation_results WHERE session_id=%s AND role=%s",
                (session_id, role),
            )
            existing = cur.fetchone()
            row_id = existing[0] if existing else _new_id()
            cur.execute(
                """INSERT INTO simulation_results (id, session_id, role, topology_name, device_type, total_nodes,
                   is_simulated, cards)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                   ON CONFLICT (session_id, role) DO UPDATE SET
                   topology_name=EXCLUDED.topology_name, device_type=EXCLUDED.device_type,
                   total_nodes=EXCLUDED.total_nodes,
                   is_simulated=EXCLUDED.is_simulated, cards=EXCLUDED.cards""",
                (
                    row_id, session_id, role,
                    data.get("topology_name"), data.get("device_type"), data.get("total_nodes"),
                    data.get("is_simulated", False),
                    json.dumps(data.get("cards", [])),
                ),
            )
    return row_id


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
    result = normalize_row(dict(zip(keys, row)))
    result["cards"] = _loads(result["cards"], default=[])
    return result


# ── comparison_reports ──

def save_comparison_report(session_id: str, original_id: str, equivalent_id: str, data: dict[str, Any]) -> None:
    """保存对比报告。`ON CONFLICT DO NOTHING` 在 SQLite 下必须带冲突目标，
    因此两套 DDL 都建了 `UNIQUE(session_id)`（PG 侧是 `uq_comparison_reports_session`）。"""
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO comparison_reports (id, session_id, original_id, equivalent_id, flops_diff_pct,
                   hbm_diff_pct, tp_comm_diff_pct, pp_comm_diff_pct, dp_comm_diff_pct,
                   is_equivalent, error_tolerance, details)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                   ON CONFLICT DO NOTHING""",
                (
                    _new_id(), session_id, original_id, equivalent_id,
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
    result = normalize_row(dict(zip(keys, row)))
    result["details"] = _loads(result["details"], default={})
    return result


# ── conversation_messages ──

def save_message(session_id: str, msg_index: int, role: str, content: str) -> None:
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO conversation_messages (id, session_id, msg_index, role, content) VALUES (%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING",
                (_new_id(), session_id, msg_index, role, content),
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

    方言说明：原实现用 `ILIKE` + `name_key = ANY(%s)`（PG 专有）。
    SQLite 无 ANY 构造，因此改为 `name_key IN (?,?,...)`；大小写不敏感由
    `LIKE` 承担 —— SQLite 的 LIKE 对 ASCII 天然不敏感，PG 侧则由 name_key
    精确匹配兜底，两侧行为一致。
    """
    if not model_name:
        return None
    # Build candidate name_keys: full name + bare name (strip org prefix)
    nk_candidates = {_name_key(model_name)}
    if "/" in model_name:
        bare = model_name.rsplit("/", 1)[-1]
        nk_candidates.add(_name_key(bare))
    nk_list = sorted(nk_candidates)
    placeholders = ",".join(["%s"] * len(nk_list))
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""SELECT model_name, model_type, num_layers, d_model, num_heads,
                          d_ffn, vocab_size, num_kv_heads, source, reference,
                          tp, pp, dp, ep, seq_len, global_batch_size, micro_batch_size, device_type,
                          num_experts, moe_ffn_hidden_size, moe_router_topk,
                          moe_layer_freq, num_moe_layers, has_shared_expert,
                          shared_expert_intermediate_size, expert_tensor_parallel_size
                   FROM model_catalog
                   WHERE model_name LIKE %s OR name_key IN ({placeholders})
                   ORDER BY
                     CASE source
                       WHEN 'mindspeed' THEN 1
                       WHEN 'megatron' THEN 2
                       WHEN 'huggingface' THEN 3
                       WHEN 'modelscope' THEN 4
                       ELSE 5
                     END,
                     (model_name LIKE %s) DESC
                   LIMIT 1""",
                [model_name, *nk_list, model_name],
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
    return normalize_row({
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
        "tp": row[10],
        "pp": row[11],
        "dp": row[12],
        "ep": row[13],
        "seq_len": row[14],
        "global_batch_size": row[15],
        "micro_batch_size": row[16],
        "device_type": row[17],
        # MoE fields (None for dense models)
        "num_experts": row[18],
        "moe_ffn_hidden_size": row[19],
        "moe_router_topk": row[20],
        "moe_layer_freq": row[21],
        "num_moe_layers": row[22],
        "has_shared_expert": row[23],
        "shared_expert_intermediate_size": row[24],
        "expert_tensor_parallel_size": row[25],
    })


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
                   (id, model_name, name_key, model_type, num_layers, d_model, num_heads,
                    d_ffn, vocab_size, num_kv_heads, source, reference, description,
                    tp, pp, dp, ep, seq_len, global_batch_size, micro_batch_size, device_type,
                    num_experts, moe_ffn_hidden_size, moe_router_topk,
                    moe_layer_freq, num_moe_layers, has_shared_expert,
                    shared_expert_intermediate_size, expert_tensor_parallel_size)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                   ON CONFLICT (model_name) DO UPDATE SET
                   name_key=EXCLUDED.name_key, model_type=EXCLUDED.model_type,
                   num_layers=EXCLUDED.num_layers, d_model=EXCLUDED.d_model,
                   num_heads=EXCLUDED.num_heads, d_ffn=EXCLUDED.d_ffn,
                   vocab_size=EXCLUDED.vocab_size, num_kv_heads=EXCLUDED.num_kv_heads,
                   source=EXCLUDED.source,
                   reference=EXCLUDED.reference,
                   description=COALESCE(EXCLUDED.description, model_catalog.description),
                   tp=EXCLUDED.tp, pp=EXCLUDED.pp, dp=EXCLUDED.dp, ep=EXCLUDED.ep,
                   seq_len=EXCLUDED.seq_len, global_batch_size=EXCLUDED.global_batch_size,
                   micro_batch_size=EXCLUDED.micro_batch_size, device_type=EXCLUDED.device_type,
                   num_experts=EXCLUDED.num_experts, moe_ffn_hidden_size=EXCLUDED.moe_ffn_hidden_size,
                   moe_router_topk=EXCLUDED.moe_router_topk, moe_layer_freq=EXCLUDED.moe_layer_freq,
                   num_moe_layers=EXCLUDED.num_moe_layers, has_shared_expert=EXCLUDED.has_shared_expert,
                   shared_expert_intermediate_size=EXCLUDED.shared_expert_intermediate_size,
                   expert_tensor_parallel_size=EXCLUDED.expert_tensor_parallel_size,
                   updated_at=NOW()""",
                (
                    _new_id(), model_name, nk, cfg.get("model_type", "dense"),
                    cfg["num_layers"], cfg["d_model"], cfg["num_heads"],
                    cfg["d_ffn"], cfg["vocab_size"], cfg.get("num_key_value_heads"),
                    cfg.get("_source"), cfg.get("reference"), description,
                    cfg.get("tp"), cfg.get("pp"), cfg.get("dp"), cfg.get("ep"),
                    cfg.get("seq_len"), cfg.get("global_batch_size"),
                    cfg.get("micro_batch_size"), cfg.get("device_type"),
                    cfg.get("num_experts"), cfg.get("moe_ffn_hidden_size"),
                    cfg.get("moe_router_topk"), cfg.get("moe_layer_freq"),
                    cfg.get("num_moe_layers"), cfg.get("has_shared_expert"),
                    cfg.get("shared_expert_intermediate_size"),
                    cfg.get("expert_tensor_parallel_size"),
                ),
            )


def list_model_catalog() -> list[dict[str, Any]]:
    """Return all model catalog entries (for admin/management views)."""
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT model_name, model_type, num_layers, d_model, num_heads,
                          d_ffn, vocab_size, num_kv_heads, source, reference, description, updated_at,
                          tp, pp, dp, ep, seq_len, global_batch_size, micro_batch_size, device_type,
                          num_experts, moe_ffn_hidden_size, moe_router_topk,
                          moe_layer_freq, num_moe_layers, has_shared_expert,
                          shared_expert_intermediate_size, expert_tensor_parallel_size
                   FROM model_catalog ORDER BY model_name"""
            )
            rows = cur.fetchall()
    return [normalize_row({
        "model_name": r[0], "model_type": r[1], "num_layers": r[2], "d_model": r[3],
        "num_heads": r[4], "d_ffn": r[5], "vocab_size": r[6], "num_key_value_heads": r[7],
        "source": r[8], "reference": r[9], "description": r[10],
        "updated_at": _iso(r[11]),
        "tp": r[12], "pp": r[13], "dp": r[14], "ep": r[15],
        "seq_len": r[16], "global_batch_size": r[17], "micro_batch_size": r[18],
        "device_type": r[19],
        "num_experts": r[20], "moe_ffn_hidden_size": r[21], "moe_router_topk": r[22],
        "moe_layer_freq": r[23], "num_moe_layers": r[24], "has_shared_expert": r[25],
        "shared_expert_intermediate_size": r[26], "expert_tensor_parallel_size": r[27],
    }) for r in rows]


def delete_model_catalog_entry(model_name: str) -> bool:
    """Delete a model catalog entry by exact name. Returns True if a row was deleted."""
    if not model_name:
        return False
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM model_catalog WHERE model_name LIKE %s", (model_name,)
            )
            return cur.rowcount > 0


def seed_model_catalog_builtin(entries: dict[str, dict[str, Any]]) -> int:
    """Bulk-upsert model catalog entries. Source comes from each entry's _source field
    (e.g. 'megatron', 'mindspeed'); falls back to 'builtin' for backward compat."""
    if not entries:
        return 0
    rows = [
        (
            _new_id(), name, _name_key(name), cfg.get("model_type", "dense"),
            cfg["num_layers"], cfg["d_model"], cfg["num_heads"],
            cfg["d_ffn"], cfg["vocab_size"], cfg.get("num_key_value_heads"),
            cfg.get("_source") or "builtin", cfg.get("reference"),
            cfg.get("tp"), cfg.get("pp"), cfg.get("dp"), cfg.get("ep"),
            cfg.get("seq_len"), cfg.get("global_batch_size"),
            cfg.get("micro_batch_size"), cfg.get("device_type"),
            cfg.get("num_experts"), cfg.get("moe_ffn_hidden_size"),
            cfg.get("moe_router_topk"), cfg.get("moe_layer_freq"),
            cfg.get("num_moe_layers"), cfg.get("has_shared_expert"),
            cfg.get("shared_expert_intermediate_size"),
            cfg.get("expert_tensor_parallel_size"),
            cfg.get("description"),
        )
        for name, cfg in entries.items()
    ]
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.executemany(
                """INSERT INTO model_catalog
                   (id, model_name, name_key, model_type, num_layers, d_model, num_heads,
                    d_ffn, vocab_size, num_kv_heads, source, reference,
                    tp, pp, dp, ep, seq_len, global_batch_size, micro_batch_size, device_type,
                    num_experts, moe_ffn_hidden_size, moe_router_topk,
                    moe_layer_freq, num_moe_layers, has_shared_expert,
                    shared_expert_intermediate_size, expert_tensor_parallel_size,
                    description)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                   ON CONFLICT (model_name) DO UPDATE SET
                   name_key=EXCLUDED.name_key, model_type=EXCLUDED.model_type,
                   num_layers=EXCLUDED.num_layers, d_model=EXCLUDED.d_model,
                   num_heads=EXCLUDED.num_heads, d_ffn=EXCLUDED.d_ffn,
                   vocab_size=EXCLUDED.vocab_size, num_kv_heads=EXCLUDED.num_kv_heads,
                   source=EXCLUDED.source, reference=EXCLUDED.reference,
                   tp=EXCLUDED.tp, pp=EXCLUDED.pp, dp=EXCLUDED.dp, ep=EXCLUDED.ep,
                   seq_len=EXCLUDED.seq_len, global_batch_size=EXCLUDED.global_batch_size,
                   micro_batch_size=EXCLUDED.micro_batch_size, device_type=EXCLUDED.device_type,
                   num_experts=EXCLUDED.num_experts, moe_ffn_hidden_size=EXCLUDED.moe_ffn_hidden_size,
                   moe_router_topk=EXCLUDED.moe_router_topk, moe_layer_freq=EXCLUDED.moe_layer_freq,
                   num_moe_layers=EXCLUDED.num_moe_layers, has_shared_expert=EXCLUDED.has_shared_expert,
                   shared_expert_intermediate_size=EXCLUDED.shared_expert_intermediate_size,
                   expert_tensor_parallel_size=EXCLUDED.expert_tensor_parallel_size,
                   description=COALESCE(EXCLUDED.description, model_catalog.description),
                   updated_at=NOW()""",
                rows,
            )
    return len(rows)
