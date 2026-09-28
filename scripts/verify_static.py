"""一次性静态校验脚本（非交付物，验证后删除）。

校验对象：app/db_migration.py 的结构不变量，以及全仓库 Python 语法。
用途：确认重构过程中「删除 schema 重复段」没有破坏 SCHEMA_SQL 内容。
"""
from __future__ import annotations

import ast
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

def check_migration() -> int:
    path = ROOT / "app" / "db_migration.py"
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)

    funcs = [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    assigns = [
        t.id
        for n in tree.body
        if isinstance(n, ast.Assign)
        for t in n.targets
        if isinstance(t, ast.Name)
    ]
    print(f"[1] 顶层赋值: {assigns}")
    print(f"[1] 顶层函数: {funcs}")

    problems = []
    if funcs.count("init_db") != 1:
        problems.append(f"init_db 定义次数 = {funcs.count('init_db')}，应为 1")
    for need in ("SCHEMA_SQL", "REQUIRED_TABLES", "REQUIRED_COLUMNS", "BENIGN_DDL_ERRORS"):
        if need not in assigns:
            problems.append(f"缺少顶层赋值 {need}")
    if "_verify_schema" not in funcs:
        problems.append("缺少 _verify_schema")

    schema = next(
        n.value.value
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(getattr(t, "id", "") == "SCHEMA_SQL" for t in n.targets)
    )
    stmts = [
        s.strip() for s in schema.split(";") if s.strip() and not s.strip().startswith("--")
    ]
    tables = [s for s in stmts if s.upper().startswith("CREATE TABLE")]
    indexes = [s for s in stmts if s.upper().startswith("CREATE INDEX")]
    alters = [s for s in stmts if s.upper().startswith("ALTER TABLE")]
    print(f"[2] SCHEMA_SQL 语句数={len(stmts)}  CREATE TABLE={len(tables)}  "
          f"CREATE INDEX={len(indexes)}  ALTER TABLE={len(alters)}")

    expected_tables = {
        "sessions", "topology_params", "simulation_params",
        "simulation_results", "comparison_reports", "conversation_messages",
        "model_catalog",
    }
    seen_tables = set()
    for stmt in tables:
        # CREATE TABLE IF NOT EXISTS <name> (...)
        token = stmt.split("EXISTS", 1)[-1].strip().split("(")[0].strip()
        seen_tables.add(token)
    if seen_tables != expected_tables:
        problems.append(f"建表集合不匹配: 多={seen_tables - expected_tables} 少={expected_tables - seen_tables}")

    for kw in ("gen_random_uuid()", "JSONB"):
        if kw not in schema:
            problems.append(f"SCHEMA_SQL 中缺少 {kw}（schema 内容可能被误删）")
    if "DROP COLUMN" in schema.upper():
        problems.append("SCHEMA_SQL 仍含破坏性 DDL（DROP COLUMN）")
    if "LISTEN_ADDRESSES" in schema.upper():
        problems.append("SCHEMA_SQL 混入了非 schema 内容")

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
                problems.append(f"{table} 缺少列/约束 {col}")

    print(f"[3] 列级抽查通过项: {sum(len(v) for v in col_checks.values())}")

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
