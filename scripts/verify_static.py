"""静态校验脚本（可复跑）：db_migration.py 的结构不变量 + 全仓库 Python 语法。

本次迁移（PostgreSQL → 默认 SQLite，保留 PG 逃生门）新增了**两套手写 DDL**，
它们是最容易长期漂移的地方，因此校验重点随之扩展：

  [1] db_migration.py 顶层结构不变量（init_db 唯一、必需常量齐全）
  [2] PG 分支 SCHEMA_SQL 的内容抽查（表集合、关键列、无破坏性 DDL）
  [3] SQLite 分支 SCHEMA_SQLITE_SQL 的内容抽查
  [4] **D1：两套 DDL 的「表→列」映射必须逐一致** —— 这是本文件最重要的一条。
      它不是格式检查，而是功能等价性检查：任何一侧漏列，都会造成
      「PG 下能跑、SQLite 下运行期才炸」这类最难查的问题。
  [5] **D2：upsert 依赖的冲突目标在两套 DDL 中都存在**
  [6] 全仓库 Python 语法
"""
from __future__ import annotations

import ast
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _console import ensure_utf8_console  # noqa: E402

ensure_utf8_console()

ROOT = pathlib.Path(__file__).resolve().parent.parent

REQUIRED_TABLES = {
    "sessions",
    "topology_params",
    "simulation_params",
    "simulation_results",
    "comparison_reports",
    "conversation_messages",
    "model_catalog",
}

# 约束/表级修饰符：不是列名，解析时要排除
_NON_COLUMN_KEYWORDS = {
    "UNIQUE", "PRIMARY", "FOREIGN", "CHECK", "CONSTRAINT", "KEY", "INDEX",
}


def _split_statements(schema: str) -> list[str]:
    return [s.strip() for s in schema.split(";") if s.strip() and not s.strip().startswith("--")]


def _strip_comments(stmt: str) -> str:
    """去掉行内 `--` 注释，避免注释里的词被误认为列名。"""
    lines = [line.split("--", 1)[0] for line in stmt.splitlines()]
    return "\n".join(lines)


def _parse_table_columns(statements: list[str]) -> dict[str, set[str]]:
    """解析 CREATE TABLE 的列名，并把 ALTER TABLE ... ADD COLUMN 合并进去。

    合并 ALTER 是必需的：PG 分支建表用「最小列集合 + 30 条 ADD COLUMN IF NOT
    EXISTS」，只解析 CREATE TABLE 会漏掉 sessions.formula_lines 这类后加列，
    从而把「PG 有、SQLite 无」的差异**反向**误报成一致。
    """
    tables: dict[str, set[str]] = {}

    for raw in statements:
        stmt = _strip_comments(raw)
        head = stmt.lstrip().upper()

        if head.startswith("CREATE TABLE"):
            m = re.search(r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?([A-Za-z_][A-Za-z0-9_]*)",
                          stmt, re.IGNORECASE)
            if not m:
                continue
            table = m.group(1)
            body_start = stmt.find("(", m.end())
            if body_start == -1:
                continue
            # 取到与第一个 ( 配对的 ) ，忽略其后的表级子句
            depth = 0
            body_end = len(stmt)
            for idx in range(body_start, len(stmt)):
                if stmt[idx] == "(":
                    depth += 1
                elif stmt[idx] == ")":
                    depth -= 1
                    if depth == 0:
                        body_end = idx
                        break
            body = stmt[body_start + 1:body_end]

            cols = tables.setdefault(table, set())
            for part in _split_top_level(body):
                part = part.strip()
                if not part:
                    continue
                first = re.match(r"([A-Za-z_][A-Za-z0-9_]*)", part)
                if not first:
                    continue
                token = first.group(1)
                if token.upper() in _NON_COLUMN_KEYWORDS:
                    continue
                cols.add(token.lower())

        elif head.startswith("ALTER TABLE") and "ADD COLUMN" in stmt.upper():
            m = re.search(
                r"ALTER\s+TABLE\s+([A-Za-z_][A-Za-z0-9_]*)\s+ADD\s+COLUMN\s+"
                r"(?:IF\s+NOT\s+EXISTS\s+)?([A-Za-z_][A-Za-z0-9_]*)",
                stmt, re.IGNORECASE,
            )
            if m:
                tables.setdefault(m.group(1), set()).add(m.group(2).lower())

    return tables


def _split_top_level(body: str) -> list[str]:
    """按顶层逗号切分表定义体（忽略括号内的逗号，如 `UNIQUE (a, b)`）。"""
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for ch in body:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
    if current:
        parts.append("".join(current))
    return parts


def _extract_top_level_assign(tree: ast.Module, name: str) -> str | None:
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            getattr(t, "id", "") == name for t in node.targets
        ):
            try:
                return ast.literal_eval(node.value)
            except Exception:  # noqa: BLE001 非常量赋值
                return None
        # `NAME: tuple[...] = (...)` 形式（带类型注解）
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == name
        ):
            try:
                return ast.literal_eval(node.value)
            except Exception:  # noqa: BLE001
                return None
    return None


def check_migration() -> int:
    path = ROOT / "app" / "db_migration.py"
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)

    funcs = [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    assigns = [
        t.id
        for n in tree.body
        if isinstance(n, (ast.Assign, ast.AnnAssign))
        for t in (n.targets if isinstance(n, ast.Assign) else [n.target])
        if isinstance(t, ast.Name)
    ]
    print(f"[1] 顶层赋值: {assigns}")
    print(f"[1] 顶层函数: {funcs}")

    problems: list[str] = []
    if funcs.count("init_db") != 1:
        problems.append(f"init_db 定义次数 = {funcs.count('init_db')}，应为 1")
    for need in (
        "SCHEMA_SQL", "SCHEMA_SQLITE_SQL", "REQUIRED_TABLES", "REQUIRED_COLUMNS",
        "BENIGN_DDL_ERRORS", "BENIGN_SQLITE_ERRORS", "SQLITE_ADDABLE_COLUMNS",
    ):
        if need not in assigns:
            problems.append(f"缺少顶层赋值 {need}")
    if "_verify_schema" not in funcs:
        problems.append("缺少 _verify_schema")

    schema = _extract_top_level_assign(tree, "SCHEMA_SQL") or ""
    sqlite_schema = _extract_top_level_assign(tree, "SCHEMA_SQLITE_SQL") or ""
    if not schema or not sqlite_schema:
        problems.append("SCHEMA_SQL / SCHEMA_SQLITE_SQL 不是可解析的字符串常量")

    pg_stmts = _split_statements(schema)
    lite_stmts = _split_statements(sqlite_schema)
    for label, stmts in (("PG", pg_stmts), ("SQLite", lite_stmts)):
        tables = [s for s in stmts if s.upper().startswith("CREATE TABLE")]
        indexes = [s for s in stmts if s.upper().startswith("CREATE INDEX")]
        alters = [s for s in stmts if s.upper().startswith("ALTER TABLE")]
        print(f"[2] {label} SCHEMA 语句数={len(stmts)}  CREATE TABLE={len(tables)}  "
              f"CREATE INDEX={len(indexes)}  ALTER TABLE={len(alters)}")

    # PG 分支：建表集合与内容抽查
    pg_tables = {t for t in _parse_table_columns(pg_stmts)}
    if pg_tables != REQUIRED_TABLES:
        problems.append(
            f"PG 建表集合不匹配: 多={pg_tables - REQUIRED_TABLES} 少={REQUIRED_TABLES - pg_tables}"
        )
    for kw in ("gen_random_uuid()", "JSONB"):
        if kw not in schema:
            problems.append(f"SCHEMA_SQL 中缺少 {kw}（PG 分支 schema 可能被误删）")
    if "DROP COLUMN" in schema.upper():
        problems.append("SCHEMA_SQL 仍含破坏性 DDL（DROP COLUMN）")
    if "LISTEN_ADDRESSES" in schema.upper():
        problems.append("SCHEMA_SQL 混入了非 schema 内容")

    # SQLite 分支：不得出现 PG 专有构造
    for kw in ("gen_random_uuid()", "JSONB"):
        if kw in sqlite_schema:
            problems.append(f"SCHEMA_SQLITE_SQL 含 PG 专有构造 {kw}")
    if "DROP COLUMN" in sqlite_schema.upper():
        problems.append("SCHEMA_SQLITE_SQL 含破坏性 DDL（DROP COLUMN）")
    lite_tables = {t for t in _parse_table_columns(lite_stmts)}
    if lite_tables != REQUIRED_TABLES:
        problems.append(
            f"SQLite 建表集合不匹配: 多={lite_tables - REQUIRED_TABLES} 少={REQUIRED_TABLES - lite_tables}"
        )

    # 每张表的必备列抽查（防止拼接时丢掉某张表的中间部分）
    col_checks = {
        "sessions": ["step", "formula_lines"],
        "topology_params": ["ep", "moe_ffn_hidden_size", "UNIQUE (session_id, role)"],
        "simulation_params": ["level1_config", "debug_time"],
        "simulation_results": ["cards", "is_simulated"],
        "comparison_reports": ["details", "error_tolerance"],
        "conversation_messages": ["msg_index", "content"],
        "model_catalog": ["name_key", "num_kv_heads", "updated_at"],
    }
    for table, cols in col_checks.items():
        for col in cols:
            if col not in schema:
                problems.append(f"SCHEMA_SQL：{table} 缺少列/约束 {col}")
            if col not in sqlite_schema:
                problems.append(f"SCHEMA_SQLITE_SQL：{table} 缺少列/约束 {col}")
    print(f"[3] 两套 DDL 列级抽查通过项: {sum(len(v) for v in col_checks.values()) * 2}")

    # ── D1：两套 DDL 的「表→列」映射必须逐一致 ──
    pg_cols = _parse_table_columns(pg_stmts)
    lite_cols = _parse_table_columns(lite_stmts)
    d1_diffs: list[str] = []
    for table in sorted(REQUIRED_TABLES):
        pg_set = pg_cols.get(table, set())
        lite_set = lite_cols.get(table, set())
        only_pg = pg_set - lite_set
        only_lite = lite_set - pg_set
        if only_pg:
            d1_diffs.append(f"{table}: 仅 PG 有 {sorted(only_pg)}")
        if only_lite:
            d1_diffs.append(f"{table}: 仅 SQLite 有 {sorted(only_lite)}")
    if d1_diffs:
        problems.append("[D1] 两套 DDL 列集合不一致：")
        problems.extend(f"    - {d}" for d in d1_diffs)
    else:
        total_cols = sum(len(v) for v in pg_cols.values())
        print(f"[D1] OK: 两套 DDL 的 {len(REQUIRED_TABLES)} 张表、{total_cols} 个列名逐一致")

    # ── D2：upsert 冲突目标必须两侧都有 ──
    # dao 里用到两类冲突目标：
    #   (session_id, role) → topology_params / simulation_params / simulation_results
    #   (session_id)       → comparison_reports（save_comparison_report 用的是
    #                        无目标 ON CONFLICT DO NOTHING，SQLite 必须能定位到目标）
    #   (model_name)       → model_catalog
    d2_problems: list[str] = []
    for table in ("topology_params", "simulation_params", "simulation_results"):
        for label, text in (("PG", schema), ("SQLite", sqlite_schema)):
            if not re.search(
                rf"CREATE TABLE\s+(?:IF NOT EXISTS\s+)?{table}\b.*?UNIQUE\s*\(\s*session_id\s*,\s*role\s*\)",
                text, re.S | re.IGNORECASE,
            ):
                d2_problems.append(f"{label} 的 {table} 缺少 UNIQUE (session_id, role)")
    for label, text in (("PG", schema), ("SQLite", sqlite_schema)):
        has_unique = re.search(
            r"UNIQUE\s*\(\s*session_id\s*\)", text, re.IGNORECASE
        ) or re.search(
            r"CREATE\s+UNIQUE\s+INDEX\s+(?:IF\s+NOT\s+EXISTS\s+)?uq_comparison_reports_session",
            text, re.IGNORECASE,
        )
        if not has_unique:
            d2_problems.append(f"{label} 的 comparison_reports 缺少 session_id 唯一约束")
    for label, text in (("PG", schema), ("SQLite", sqlite_schema)):
        # model_catalog 的冲突目标来自建表时的 `model_name ... UNIQUE` 列约束
        # （不是 ON CONFLICT 子句 —— 那个在 dao 的 INSERT 里，不在 DDL 里）。
        if not re.search(
            r"model_name\s+\w+(?:\s*\(\s*\d+\s*\))?\s+NOT\s+NULL\s+UNIQUE",
            text, re.IGNORECASE,
        ):
            d2_problems.append(f"{label} 的 model_catalog 缺少 model_name 唯一约束（ON CONFLICT 目标）")
        if not re.search(
            r"name_key\s+\w+(?:\s*\(\s*\d+\s*\))?\s+NOT\s+NULL\s+UNIQUE",
            text, re.IGNORECASE,
        ):
            d2_problems.append(f"{label} 的 model_catalog 缺少 name_key 唯一约束")
    if d2_problems:
        problems.append("[D2] upsert 冲突目标缺失：")
        problems.extend(f"    - {d}" for d in d2_problems)
    else:
        print("[D2] OK: upsert 依赖的冲突目标（session_id,role / session_id / model_name）两套 DDL 齐备")

    if problems:
        print("\n失败项:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\nOK: db_migration.py 结构不变量全部满足")
    return 0


def compile_all() -> int:
    bad = []
    for path in sorted(ROOT.rglob("*.py")):
        if any(part in {".git", "__pycache__", ".tmp", "node_modules"} for part in path.parts):
            continue
        try:
            ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as exc:
            bad.append(f"{path.relative_to(ROOT)}: {exc}")
    if bad:
        print("\n语法错误:")
        for b in bad:
            print(f"  - {b}")
        return 1
    print("\nOK: 全仓库 Python 文件语法通过")
    return 0


if __name__ == "__main__":
    rc = check_migration()
    rc |= compile_all()
    sys.exit(rc)
