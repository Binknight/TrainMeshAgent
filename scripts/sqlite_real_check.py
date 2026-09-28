"""真实 SQLite 验证：把 SQLite 分支的建表 DDL 与 DAO 全路径跑在真实文件库上。

设计对齐 `scripts/pg_real_check.py`（真实引擎、可复跑、有正负向），额外覆盖
SQLite 特有的三个必须实测项：

  1. `PRAGMA foreign_keys=ON` 是否真的生效（`ON DELETE CASCADE` 依赖它，
     且它是**每连接**属性 —— 池里每条连接都得设，漏一条就静默失效）
  2. `journal_mode` 是否真的是 WAL
  3. **并发写（M2）**：N 线程 × M 次写入不得出现 `database is locked`
     —— 这是整个 SQLite 方案唯一可能失效的地方，必须实测而非推断

用法：
    python scripts/sqlite_real_check.py            # 在临时目录建库，跑完即删
    SQLITE_PATH=/tmp/foo.db python scripts/sqlite_real_check.py   # 指定文件（仍会清理）
"""
from __future__ import annotations

import os
import shutil
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _console import ensure_utf8_console  # noqa: E402

# Windows 控制台默认 cp1252，中文/× 字符会让 print 抛 UnicodeEncodeError，
# 把「校验失败」伪装成编码崩溃。统一切到 UTF-8。
ensure_utf8_console()

# 必须在 import app.* 之前定好环境：app.config 在 import 期就读取环境变量。
#
# 临时库放仓库内 `.tmp/`（已在 .gitignore 中）而非系统临时目录：
# 一是与仓库既有测试产物位置一致，二是某些受限执行环境只允许在工作区内写文件。
_TMP_ROOT = os.path.join(ROOT, ".tmp")
_TMP_DIR = os.path.join(_TMP_ROOT, f"sqlite_check_{uuid.uuid4().hex[:8]}")
os.makedirs(_TMP_DIR, exist_ok=True)
_DB_PATH = os.environ.get("SQLITE_PATH") or os.path.join(_TMP_DIR, "check.db")
os.environ["SQLITE_PATH"] = _DB_PATH
os.environ.pop("DATABASE_URL", None)          # 确保落到 SQLite 后端
os.environ["AICM_MCP_WORKSPACE_ALLOW_LOCAL"] = "1"

from app.db import backend, get_db, reset_for_tests  # noqa: E402
from app import dao  # noqa: E402
from app.db_migration import (  # noqa: E402
    SCHEMA_SQLITE_SQL,
    _verify_schema,
    init_db,
)
from app.dbapi import translate  # noqa: E402

FAILURES: list[str] = []


def log(msg: str) -> None:
    print(f"[sqlite-real] {msg}", flush=True)


def check(condition: bool, ok_msg: str, fail_msg: str) -> bool:
    if condition:
        log(f"OK: {ok_msg}")
        return True
    FAILURES.append(fail_msg)
    log(f"FAIL: {fail_msg}")
    return False


# ────────────────────────── 1. 建表 / 幂等 / 硬校验 ──────────────────────────


def check_schema() -> None:
    log(f"数据库文件: {_DB_PATH}")
    check(backend() == "sqlite", "后端侦测为 sqlite", f"后端侦测错误，实际 {backend()}")

    init_db()
    log("init_db() 第 1 次执行完成")
    # 幂等重放：第二次不得抛错
    try:
        init_db()
        log("OK: init_db() 幂等重放无异常")
    except Exception as exc:  # noqa: BLE001
        FAILURES.append(f"init_db() 幂等重放抛错: {exc}")

    with get_db() as conn:
        with conn.cursor() as cur:
            missing = _verify_schema(cur)
            check(not missing, "硬校验 _verify_schema 通过（7 表 + 关键列）", f"硬校验报缺失: {missing}")

            # 负向：删一张核心表，校验必须发现
            cur.execute("DROP TABLE comparison_reports")
            missing2 = _verify_schema(cur)
            check(
                any("comparison_reports" in m for m in missing2),
                f"负向测试：删表后正确报出 -> {missing2}",
                f"负向测试失败：删表后校验未报出，实际 {missing2}",
            )
        conn.autocommit = True
        with conn.cursor() as cur:
            # 复原（后续 DAO 用例需要这张表）
            for stmt in SCHEMA_SQLITE_SQL.split(";"):
                stmt = stmt.strip()
                if stmt and not stmt.startswith("--"):
                    try:
                        cur.execute(stmt)
                    except Exception:  # noqa: BLE001  已存在的对象
                        pass
        conn.autocommit = False


# ────────────────────────── 2. SQLite 特有的行为契约 ──────────────────────────


def check_engine_contracts() -> None:
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("PRAGMA journal_mode")
            mode = cur.fetchone()[0]
            check(str(mode).lower() == "wal", f"journal_mode=WAL（实际 {mode}）", f"journal_mode 不是 WAL：{mode}")

            cur.execute("PRAGMA foreign_keys")
            fk = cur.fetchone()[0]
            check(bool(fk), "foreign_keys=ON（本连接）", f"foreign_keys 未开启：{fk}")

            cur.execute("PRAGMA synchronous")
            sync = cur.fetchone()[0]
            log(f"  synchronous={sync}（1=NORMAL）")

            # 时间戳默认值：created_at 必须自动填充成可解析的形态
            cur.execute("INSERT INTO sessions (id) VALUES ('zzzzzzzz') ON CONFLICT DO NOTHING")
            cur.execute("SELECT step, created_at FROM sessions WHERE id='zzzzzzzz'")
            row = cur.fetchone()
            check(
                bool(row) and row[0] == "idle" and bool(row[1]),
                f"sessions 默认值生效：step={row[0]!r} created_at={row[1]!r}",
                f"sessions 默认值异常: {row}",
            )

            # 错误恢复：把连接交还给池之后，下一次取用不得继承半截事务
            cur.execute("DELETE FROM sessions WHERE id='zzzzzzzz'")

    # 跨连接验证 FK 与级联（池里另一条连接也必须开着 foreign_keys）
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO sessions (id) VALUES ('cascade1')")
            cur.execute(
                "INSERT INTO conversation_messages (id, session_id, msg_index, role, content) VALUES (%s,'cascade1',1,'user','hi')",
                (str(uuid.uuid4()),),
            )
            cur.execute("DELETE FROM sessions WHERE id='cascade1'")
            cur.execute("SELECT COUNT(*) FROM conversation_messages WHERE session_id='cascade1'")
            left = cur.fetchone()[0]
            check(left == 0, "ON DELETE CASCADE 生效（对话消息随会话删除）", f"级联未生效，残留 {left} 条消息")


# ────────────────────────── 3. UNIQUE 约束（ON CONFLICT 目标合法性） ──────────────────────────


def check_unique_targets() -> None:
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO sessions (id) VALUES ('uniq0001') ON CONFLICT DO NOTHING")
            cur.execute(
                "INSERT INTO simulation_results (id, session_id, role) VALUES (%s,'uniq0001','original')",
                (str(uuid.uuid4()),),
            )
            try:
                cur.execute(
                    "INSERT INTO simulation_results (id, session_id, role) VALUES (%s,'uniq0001','original')",
                    (str(uuid.uuid4()),),
                )
                FAILURES.append("UNIQUE(session_id, role) 未生效：重复插入未报错")
                log("FAIL: UNIQUE(session_id, role) 未生效")
            except Exception:  # noqa: BLE001
                log("OK: UNIQUE(session_id, role) 生效（dao 的 ON CONFLICT 目标合法）")

            # comparison_reports 的 UNIQUE(session_id)：save_comparison_report 的
            # `ON CONFLICT DO NOTHING` 在 SQLite 下必须能命中一个目标，否则语义不同
            cur.execute(
                "INSERT INTO comparison_reports (id, session_id) VALUES (%s,'uniq0001')",
                (str(uuid.uuid4()),),
            )
            try:
                cur.execute(
                    "INSERT INTO comparison_reports (id, session_id) VALUES (%s,'uniq0001')",
                    (str(uuid.uuid4()),),
                )
                FAILURES.append("UNIQUE(session_id) on comparison_reports 未生效")
                log("FAIL: comparison_reports 的 UNIQUE(session_id) 未生效")
            except Exception:  # noqa: BLE001
                log("OK: comparison_reports 的 UNIQUE(session_id) 生效（ON CONFLICT DO NOTHING 目标合法）")


# ────────────────────────── 4. DAO 全路径往返 ──────────────────────────


def check_dao_roundtrip() -> None:
    sid = "dao" + uuid.uuid4().hex[:5]

    dao.create_session(sid)
    dao.update_session_step(sid, "topology", original_task_id="task-a")

    summary = dao.get_session_summary(sid)
    check(
        bool(summary) and summary["step"] == "topology" and summary["original_task_id"] == "task-a",
        f"create_session/update_session_step/get_session_summary 往返一致: {summary}",
        f"session 往返不一致: {summary}",
    )
    check(
        bool(summary) and isinstance(summary["created_at"], str) and "T" in summary["created_at"],
        f"时间戳出口为 ISO 字符串: {summary['created_at']}",
        f"时间戳出口形态异常: {summary.get('created_at')!r}",
    )

    # COALESCE 语义：只传 step 不得清空已存在的 task_id
    dao.update_session_step(sid, "simulating")
    summary2 = dao.get_session_summary(sid)
    check(
        summary2["original_task_id"] == "task-a",
        "update_session_step 的 COALESCE 语义保持（未传的字段不被清空）",
        f"COALESCE 语义被破坏: {summary2['original_task_id']!r}",
    )

    # formula_lines（JSON）
    dao.save_formula_lines(sid, [{"k": "计算强度", "v": 1.5}])
    check(
        dao.get_session_summary(sid)["formula_lines"] == [{"k": "计算强度", "v": 1.5}],
        "formula_lines JSON 往返一致（含非 ASCII）",
        f"formula_lines 往返不一致: {dao.get_session_summary(sid)['formula_lines']}",
    )

    # topology_params：含 MoE 字段与布尔列
    dao.save_topology_params(sid, "original", {
        "name": "A3×4", "device_type": "A3", "dp_size": 2, "tp_size": 2, "pp_size": 1,
        "total_nodes": 4, "model_name": "Qwen2.5-72B", "num_layers": 80,
        "hidden_dim": 8192, "num_heads": 64, "d_ffn": 29568, "seq_len": 4096,
        "batch_size": 16, "micro_batch_size": 1, "vocab_size": 152064,
        "model_type": "sparse", "num_experts": 64, "moe_router_topk": 6,
        "num_moe_layers": 60, "moe_ffn_hidden_size": 2048,
        "has_shared_expert": True, "expert_tensor_parallel_size": 2, "ep": 8,
    })
    tp = dao.get_topology_params(sid, "original")
    check(
        bool(tp) and tp["name"] == "A3×4" and tp["ep"] == 8 and tp["num_experts"] == 64,
        f"topology_params 往返一致（MoE 字段）: name={tp['name']!r} ep={tp['ep']}",
        f"topology_params 往返不一致: {tp}",
    )
    check(
        tp["has_shared_expert"] is True,
        "布尔列读回为真 bool（has_shared_expert is True）—— pydantic 严格布尔校验依赖它",
        f"布尔列类型异常: {tp['has_shared_expert']!r} (type={type(tp['has_shared_expert']).__name__})",
    )

    # upsert：同一 (session_id, role) 必须更新而非新增
    dao.save_topology_params(sid, "original", {"name": "A3×8", "device_type": "A3", "dp_size": 4,
                                              "tp_size": 2, "pp_size": 1, "total_nodes": 8,
                                              "has_shared_expert": False})
    tp2 = dao.get_topology_params(sid, "original")
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM topology_params WHERE session_id=%s", (sid,))
            n_rows = cur.fetchone()[0]
    check(
        tp2["name"] == "A3×8" and n_rows == 1,
        f"topology_params upsert 语义正确（更新而非新增，行数={n_rows}）",
        f"upsert 语义错误：name={tp2['name']!r} 行数={n_rows}",
    )

    # simulation_params：JSON 列 + 布尔列
    dao.save_simulation_params(sid, "original", {
        "script_path": "/x/t.sh", "epoch_num": 3, "model_name": "m",
        "level0_config": {"a": [1, 2]}, "level1_config": {"b": "中文"},
        "no_time_accumulation": True, "visual_json_output": False, "debug_time": True,
    })
    sp = dao.get_simulation_params(sid, "original")
    check(
        bool(sp) and sp["level0_config"] == {"a": [1, 2]} and sp["level1_config"] == {"b": "中文"},
        f"simulation_params JSON 列往返一致: {sp['level0_config']} / {sp['level1_config']}",
        f"simulation_params JSON 往返不一致: {sp}",
    )
    check(
        sp["no_time_accumulation"] is True and sp["visual_json_output"] is False and sp["debug_time"] is True,
        "simulation_params 三个布尔列读回为真 bool",
        f"布尔列异常: {sp['no_time_accumulation']!r}/{sp['visual_json_output']!r}/{sp['debug_time']!r}",
    )

    # simulation_results + 返回 id 稳定性（外键引用它）
    rid = dao.save_simulation_result(sid, "original", {
        "topology_name": "A3×8", "device_type": "A3", "total_nodes": 8,
        "is_simulated": True, "cards": [{"rank": 0, "flops": 1.23}],
    })
    rid2 = dao.save_simulation_result(sid, "original", {
        "topology_name": "A3×8", "device_type": "A3", "total_nodes": 8,
        "is_simulated": True, "cards": [{"rank": 0, "flops": 2.34}],
    })
    check(
        rid == rid2,
        f"save_simulation_result 幂等返回同一主键（{rid}）—— comparison_reports 外键依赖它",
        f"主键在 upsert 后漂移：{rid} -> {rid2}",
    )
    sr = dao.get_simulation_result(sid, "original")
    check(
        sr["cards"] == [{"rank": 0, "flops": 2.34}] and sr["is_simulated"] is True,
        f"simulation_results 往返一致: cards={sr['cards']} is_simulated={sr['is_simulated']!r}",
        f"simulation_results 往返不一致: {sr}",
    )

    # comparison_reports（外键指向 simulation_results.id）
    dao.save_comparison_report(sid, rid, rid, {
        "flops_diff_pct": 1.1, "hbm_diff_pct": 2.2, "tp_comm_diff_pct": 0.5,
        "pp_comm_diff_pct": 0.0, "dp_comm_diff_pct": 3.3,
        "is_equivalent": True, "error_tolerance": 5.0, "details": {"note": "等效"},
    })
    cr = dao.get_comparison_report(sid)
    check(
        bool(cr) and cr["is_equivalent"] is True and cr["details"] == {"note": "等效"},
        f"comparison_reports 往返一致（含外键 original_id/equivalent_id）: {cr['details']}",
        f"comparison_reports 往返不一致: {cr}",
    )
    # ON CONFLICT DO NOTHING：重复保存不得报错、不得新增行
    dao.save_comparison_report(sid, rid, rid, {"flops_diff_pct": 9.9})
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM comparison_reports WHERE session_id=%s", (sid,))
            n_reports = cur.fetchone()[0]
    check(
        n_reports == 1,
        "save_comparison_report 的 ON CONFLICT DO NOTHING 幂等（重复保存不新增行）",
        f"重复保存新增了行，实际 {n_reports} 行",
    )

    # conversation_messages
    for i, (role, content) in enumerate([("user", "你好"), ("assistant", '{"role":"assistant","content":"hi"}')]):
        dao.save_message(sid, i, role, content)
    msgs = dao.get_messages(sid)
    check(
        len(msgs) == 2 and msgs[0] == {"role": "user", "content": "你好"},
        f"conversation_messages 往返一致（含 JSON 内容解析）: {msgs}",
        f"conversation_messages 往返不一致: {msgs}",
    )

    # 历史列表：标题必须来自 topology_params（不能退化成「新建任务」）
    summaries = {s["session_id"]: s["title"] for s in dao.get_session_summaries()}
    check(
        summaries.get(sid, "").startswith("A3×8"),
        f"get_session_summaries 标题来自 topology_params: {summaries.get(sid)!r}",
        f"历史标题退化: {summaries.get(sid)!r}",
    )

    # 删除级联 + 列表
    dao.delete_session(sid)
    remaining = [s["session_id"] for s in dao.list_session_ids()]
    check(
        dao.get_session_summary(sid) is None and sid not in remaining,
        "delete_session 生效并从列表中消失",
        "delete_session 未生效",
    )


# ────────────────────────── 5. model_catalog（ILIKE/ANY 改写后的匹配语义） ──────────────────────────


def check_model_catalog() -> None:
    from app.models.model_catalog import MEGATRON_DENSE_MODELS, MINDSPEED_MOE_MODELS

    # init_db() 在收尾会 seed 全部四类内置清单，因此这里先把基线记下来。
    baseline = len(dao.list_model_catalog())
    log(f"  init_db 已 seed 基线 {baseline} 条")

    n1 = dao.seed_model_catalog_builtin(MEGATRON_DENSE_MODELS)
    n2 = dao.seed_model_catalog_builtin(MINDSPEED_MOE_MODELS)
    log(f"  seed: megatron dense {n1} 条 / mindspeed moe {n2} 条")

    # 幂等：再 seed 一遍不得新增行（count 必须回到基线）
    dao.seed_model_catalog_builtin(MEGATRON_DENSE_MODELS)
    dao.seed_model_catalog_builtin(MINDSPEED_MOE_MODELS)
    listed = dao.list_model_catalog()
    check(
        len(listed) == baseline,
        f"seed_model_catalog_builtin 幂等（重 seed 后仍为基线 {len(listed)} 条）",
        f"seed 不幂等：基线 {baseline} 条，重 seed 后 {len(listed)} 条",
    )

    sample = next(iter(MEGATRON_DENSE_MODELS))
    entry = dao.get_model_catalog_entry(sample)
    check(
        bool(entry) and entry["model_name"] == sample,
        f"get_model_catalog_entry 精确命中: {sample}",
        f"get_model_catalog_entry 未命中 {sample}: {entry}",
    )
    entry_ci = dao.get_model_catalog_entry(sample.lower())
    check(
        bool(entry_ci),
        f"get_model_catalog_entry 大小写不敏感命中（ILIKE→LIKE 改写后）: {sample.lower()}",
        f"大小写不敏感匹配失效: {sample.lower()} -> {entry_ci}",
    )
    # 带 org 前缀的 repo id 也要能命中裸名（name_key IN (...) 路径）
    entry_repo = dao.get_model_catalog_entry(f"someorg/{sample}")
    check(
        bool(entry_repo),
        f"repo-id 形式（someorg/{sample}）通过 name_key 兜底命中",
        f"repo-id 形式未命中: someorg/{sample} -> {entry_repo}",
    )

    # upsert 覆盖 + 删除
    dao.upsert_model_catalog("Unit-Test-Model", {
        "num_layers": 1, "d_model": 2, "num_heads": 3, "d_ffn": 4, "vocab_size": 5,
        "model_type": "dense", "_source": "builtin",
    }, description="t")
    check(
        bool(dao.get_model_catalog_entry("Unit-Test-Model")),
        "upsert_model_catalog 新增并可按名命中（name_key 归一化：连字符被剥离）",
        "upsert_model_catalog 未生效",
    )
    check(
        dao.delete_model_catalog_entry("Unit-Test-Model") and not dao.delete_model_catalog_entry("Unit-Test-Model"),
        "delete_model_catalog_entry 删除返回值正确（True→False）",
        "delete_model_catalog_entry 返回值异常",
    )


# ────────────────────────── 6. 并发写（M2：方案唯一可能失效处） ──────────────────────────


def check_concurrency(threads: int = 8, per_thread: int = 200) -> None:
    errors: list[str] = []
    lock = threading.Lock()

    def worker(idx: int) -> None:
        sid = f"c{idx:02d}"
        for i in range(per_thread):
            try:
                dao.create_session(sid)
                dao.update_session_step(sid, f"step{i}")
                dao.save_message(sid, i, "user", f"msg-{i}")
            except Exception as exc:  # noqa: BLE001
                with lock:
                    errors.append(f"线程{idx} 第{i}次: {type(exc).__name__}: {exc}")
                return

    with ThreadPoolExecutor(max_workers=threads) as pool:
        list(pool.map(worker, range(threads)))

    locked = [e for e in errors if "locked" in e.lower()]
    check(
        not errors,
        f"并发写 {threads} 线程 × {per_thread} 次：0 错误（无 database is locked）",
        f"并发写出现 {len(errors)} 个错误（其中 locked {len(locked)} 个），首例: {errors[0] if errors else '-'}",
    )

    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM sessions WHERE id LIKE 'c%'")
            n_sessions = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM conversation_messages WHERE session_id LIKE 'c%'")
            n_msgs = cur.fetchone()[0]
    check(
        n_sessions == threads and n_msgs == threads * per_thread,
        f"并发写总量正确（会话 {n_sessions}/{threads}，消息 {n_msgs}/{threads * per_thread}）",
        f"并发写总量不符：会话 {n_sessions}/{threads}，消息 {n_msgs}/{threads * per_thread}",
    )

    # 清理，避免影响后续统计
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM conversation_messages WHERE session_id LIKE 'c%'")
            cur.execute("DELETE FROM sessions WHERE id LIKE 'c%'")


# ────────────────────────── 7. 方言转换器（词法安全） ──────────────────────────


def check_translator() -> None:
    cases = [
        ("SELECT %s, %s", "SELECT ?, ?", "占位符"),
        ("UPDATE t SET a=NOW() WHERE b=%s", "UPDATE t SET a=CURRENT_TIMESTAMP WHERE b=?", "NOW()"),
        ("ON CONFLICT (a) DO UPDATE SET b=EXCLUDED.b", "ON CONFLICT (a) DO UPDATE SET b=excluded.b", "EXCLUDED"),
        ("SELECT 'NOW()' , %s", "SELECT 'NOW()' , ?", "单引号字面量内的 NOW() 不得被替换"),
        ("SELECT \"EXCLUDED\" , %s", 'SELECT "EXCLUDED" , ?', "双引号标识符内的 EXCLUDED 不得被替换"),
        ("SELECT '100%s' , %s", "SELECT '100%s' , ?", "单引号字面量内的 %s 不得被替换"),
        ("SELECT 'it''s %s', %s", "SELECT 'it''s %s', ?", "转义单引号后的字面量仍受保护"),
    ]
    for sql, expected, label in cases:
        got = translate(sql, "sqlite")
        check(got == expected, f"translate: {label}", f"translate 失败（{label}）: {sql!r} -> {got!r}，期望 {expected!r}")

    pg_sql = "SELECT %s, NOW(), EXCLUDED.x"
    check(
        translate(pg_sql, "postgres") == pg_sql,
        "PG 后端下 translate 原样返回（逃生门逐字不变）",
        "PG 后端下 translate 改动了语句",
    )


def main() -> int:
    try:
        check_schema()
        check_translator()
        check_engine_contracts()
        check_unique_targets()
        check_dao_roundtrip()
        check_model_catalog()
        check_concurrency()
    finally:
        reset_for_tests()
        # 只清理自己造的临时目录；用户显式指定的 SQLITE_PATH 不动。
        if _DB_PATH.startswith(_TMP_DIR):
            shutil.rmtree(_TMP_DIR, ignore_errors=True)
            log(f"已清理临时目录 {_TMP_DIR}")

    if FAILURES:
        print(f"\n失败 {len(FAILURES)} 项:")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("\nOK: 真实 SQLite 上 建表/幂等/硬校验正负向/引擎契约/UNIQUE 目标/DAO 全路径/并发写 全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
