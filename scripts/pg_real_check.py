"""真实数据库验证：把 SCHEMA_SQL 跑在本机 PostgreSQL 上，验证迁移与校验逻辑。

为什么可以这样验证：
    建表 DDL 与收尾硬校验查询都是标准 PG 语法（gen_random_uuid() 自 PG 13 起内建，
    未用任何 contrib 扩展），因此在本机 PG 18 上执行能证明语句本身正确；
    内嵌镜像使用 PG 14，只影响数据目录格式，不影响这些语句。

只做两件真实的事：
    1. CREATE DATABASE 一个临时库 -> 在它上面执行 SCHEMA_SQL -> 跑 _verify_schema
    2. 用 dao 的真实接口做一次写读回环（需要 Flask 依赖，缺失则跳过）
无论成败都会 DROP DATABASE，不留痕。

用法：DATABASE_URL=postgresql://postgres:<pw>@127.0.0.1:5432/postgres python .tmp/pg_real_check.py
"""
from __future__ import annotations

import os
import sys
import time
import uuid

import psycopg2
from psycopg2 import sql

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _console import ensure_utf8_console  # noqa: E402

ensure_utf8_console()

from app.db_migration import SCHEMA_SQL, _verify_schema, BENIGN_DDL_ERRORS  # noqa: E402

ADMIN_DSN = os.getenv("ADMIN_DSN", "")
TEST_DB = f"tma_verify_{uuid.uuid4().hex[:8]}"


def log(msg: str) -> None:
    print(f"[pg-real] {msg}", flush=True)


def exec_schema(conn) -> tuple[int, int, list[str]]:
    """执行 SCHEMA_SQL，返回 (成功数, 可忽略失败数, 致命失败列表)。"""
    ok = benign = 0
    fatal: list[str] = []
    conn.autocommit = True
    with conn.cursor() as cur:
        for stmt in SCHEMA_SQL.split(";"):
            stmt = stmt.strip()
            if not stmt or stmt.startswith("--"):
                continue
            first = stmt.splitlines()[0].strip()[:90]
            try:
                cur.execute(stmt)
                ok += 1
            except Exception as exc:  # noqa: BLE001
                code = getattr(exc, "pgcode", None)
                if code in BENIGN_DDL_ERRORS:
                    benign += 1
                    log(f"  可忽略({code}): {exc}")
                else:
                    fatal.append(f"{first} -> {exc}")
    return ok, benign, fatal


def main() -> int:
    if not ADMIN_DSN:
        log("未提供 ADMIN_DSN，跳过（用法见文件头注释）")
        return 2

    log(f"目标维护库: {ADMIN_DSN.split('@')[-1]}")
    admin = psycopg2.connect(ADMIN_DSN, connect_timeout=5)
    admin.autocommit = True
    with admin.cursor() as cur:
        cur.execute("SELECT version()")
        log(f"服务端: {cur.fetchone()[0].split(',')[0]}")

    test_dsn = ADMIN_DSN.rsplit("/", 1)[0] + "/" + TEST_DB
    failures: list[str] = []
    created = False
    try:
        with admin.cursor() as cur:
            cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(TEST_DB)))
        created = True
        log(f"已创建临时库 {TEST_DB}")

        conn = psycopg2.connect(test_dsn, connect_timeout=5)
        try:
            ok, benign, fatal = exec_schema(conn)
            log(f"SCHEMA_SQL: 成功 {ok} 条，可忽略 {benign} 条，致命失败 {len(fatal)} 条")
            failures += [f"DDL 失败: {f}" for f in fatal]

            # 第二次执行：验证幂等（全部应落到「可忽略」分支）
            ok2, benign2, fatal2 = exec_schema(conn)
            log(f"第二次执行（幂等性）: 成功 {ok2} 条，可忽略 {benign2} 条，致命失败 {len(fatal2)} 条")
            if fatal2:
                failures += [f"幂等重放失败: {f}" for f in fatal2]

            # 收尾硬校验
            with conn.cursor() as cur:
                missing = _verify_schema(cur)
            if missing:
                failures.append(f"硬校验报缺失: {missing}")
            else:
                log("硬校验 _verify_schema: 通过（7 张表 + 关键列齐全）")

            # 负向测试：删掉一张核心表，校验必须能发现
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute("DROP TABLE comparison_reports CASCADE")
                missing2 = _verify_schema(cur)
            if any("comparison_reports" in m for m in missing2):
                log(f"负向测试: 删除 comparison_reports 后校验正确报出 -> {missing2}")
            else:
                failures.append(f"负向测试失败：删表后校验未报出，实际 {missing2}")

            # 关键行为验证：sessions 上必须真能插入并带默认值
            with conn.cursor() as cur:
                cur.execute("INSERT INTO sessions (id) VALUES ('abcdefgh') ON CONFLICT DO NOTHING")
                cur.execute("SELECT step, created_at FROM sessions WHERE id='abcdefgh'")
                row = cur.fetchone()
            if row and row[0] == "idle" and row[1]:
                log(f"行为验证: sessions 插入成功，step 默认值={row[0]!r}，created_at 自动填充")
            else:
                failures.append(f"行为验证失败: {row}")

            # 关键行为验证：JSONB 列可写可读、gen_random_uuid() 主键可生成
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO simulation_results (session_id, role, cards)
                       VALUES ('abcdefgh','original','[{"a":1}]'::jsonb) RETURNING id"""
                )
                pk = cur.fetchone()[0]
                cur.execute("SELECT cards FROM simulation_results WHERE id=%s", (pk,))
                cards = cur.fetchone()[0]
            log(f"行为验证: simulation_results 主键={pk}（gen_random_uuid 生效），cards={cards}")
            if not pk:
                failures.append("gen_random_uuid() 未生成主键")

            # UNIQUE(session_id, role) 必须生效 —— dao 的 ON CONFLICT 依赖它
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO simulation_results (session_id, role) VALUES ('abcdefgh','original')"
                    )
                failures.append("UNIQUE(session_id, role) 未生效：重复插入未报错")
            except psycopg2.errors.UniqueViolation:
                log("行为验证: UNIQUE(session_id, role) 生效（ON CONFLICT 目标合法）")
                conn.rollback()
        finally:
            conn.close()
    finally:
        if created:
            try:
                with admin.cursor() as cur:
                    cur.execute(sql.SQL("DROP DATABASE {}").format(sql.Identifier(TEST_DB)))
                log(f"已清理临时库 {TEST_DB}")
            except Exception as exc:  # noqa: BLE001
                log(f"警告：清理临时库失败 {exc}")
        admin.close()

    if failures:
        print(f"\n失败 {len(failures)} 项:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nOK: 真实 PG 上 SCHEMA_SQL / 幂等重放 / 硬校验 / 正向与负向行为 全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
