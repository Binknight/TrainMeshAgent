"""收尾一致性检查：跨文件契约对齐。

本次迁移（PostgreSQL → 默认 SQLite，保留 PG 逃生门）的核心风险，是同一个路径
在四处重复声明而写歪：镜像 ENV（Dockerfile）、app/config.py 默认值、chart
（values.yaml）、entrypoint 兜底值。因此这里做跨文件比对，而非单文件语法检查：

  1. 镜像 ENV 的 SQLITE_PATH 与 app/config.py 默认值一致
  2. chart 的 config.sqlitePath 与镜像 ENV 一致
  3. SQLITE_PATH 必须落在 database.containerPath（挂载点）之内 —— 否则数据库
     会落进镜像层，容器重建即丢全部会话历史
  4. 默认后端是 SQLite：镜像 ENV 的 DATABASE_URL 必须为空（留空 = 用 SQLite；
     填了非 postgres:// 前缀的串会静默回落 SQLite，那才是最难查的配置错误）
  5. entrypoint 兜底默认值与上面三处一致
  6. 旧 PG 契约已彻底移除：不再有 PGDATA / PG_SOCKET_DIR / pgSocketDir /
     postgresql-14 / wait_for_db.py；SVG 用到的启动文件都在
  7. workspace 路径四处一致（历史盲区，保持）
  8. 破坏性 DDL / 静默吞异常写法未回归
"""
from __future__ import annotations

import pathlib
import re
import sys

import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _console import ensure_utf8_console  # noqa: E402

ensure_utf8_console()

ROOT = pathlib.Path(__file__).resolve().parent.parent


def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def env_from_dockerfile(text: str) -> dict[str, str]:
    """解析 Dockerfile 的 ENV 段（KEY=value 多行反斜杠续行，直到下一个指令行）。

    值的部分允许为空（`DATABASE_URL=` 是**有意**留空的默认值，表示走 SQLite），
    因此正则末段用 `*` 而非 `+` —— 否则这个刻意留空的赋值会被解析丢掉，
    表现为「未声明 DATABASE_URL」这种反向误报。
    """
    envs: dict[str, str] = {}
    collecting = False
    payload = ""
    for line in text.splitlines():
        if not collecting:
            # 注意：Dockerfile 里可能先出现旧式 `ENV LANG C.UTF-8`（空格分隔、无 =），
            # 那不是我们要找的块，跳过继续找 KEY=value 形式的那一处。
            m = re.match(r"^ENV\s+(.*)$", line)
            if m and "=" in m.group(1):
                collecting = True
                payload = m.group(1)
            else:
                continue
        else:
            # 上一行是续行：拼上本行
            payload = payload.rstrip().rstrip("\\") + " " + line

        for key, value in re.findall(
            r"([A-Z_][A-Z0-9_]*)=(\"[^\"]*\"|'[^']*'|[^\s]*)", payload
        ):
            envs[key] = value.strip('"').strip("'")

        if not payload.rstrip().endswith("\\"):
            break  # 本指令（含续行）到此结束
    return envs


def strip_comments(text: str) -> str:
    """去掉整行注释。

    「旧 PG 契约是否残留」这类检查必须只看**可执行内容**：本次改造的说明性注释里
    正当地解释了「原先有 PGDATA / initdb / pg_isready」等历史，把它们当成残留会
    产生假失败，进而诱使后来者删掉有价值的上下文。
    本仓库的 Dockerfile / shell 脚本里没有行尾内联注释，按整行剥离即可。
    """
    return "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    )


def extract_config_default(text: str, name: str) -> str | None:
    """取 `NAME = os.getenv("NAME", "默认值")` 里的默认值；无默认值时返回 None。"""
    m = re.search(rf'{name}\s*=\s*os\.getenv\(\s*"{name}"\s*,\s*"([^"]*)"', text)
    return m.group(1) if m else None


def main() -> int:
    errors: list[str] = []
    notes: list[str] = []

    base_df = read("docker/base/Dockerfile")
    app_df = read("Dockerfile")
    config_py = read("app/config.py")
    migration_py = read("app/db_migration.py")
    entrypoint = read("docker/entrypoint.sh")
    values_text = read("charts/values.yaml")
    values = yaml.safe_load(re.sub(r"@([A-Za-z0-9_.]+)@", r'"PH_\1"', values_text))
    configmap = read("charts/templates/configMap.yaml")

    # ── 1. 镜像 ENV vs app/config.py ──
    envs = env_from_dockerfile(app_df)
    print(f"[1] Dockerfile ENV 解析到 {len(envs)} 个键")

    sqlite_env = envs.get("SQLITE_PATH", "")
    sqlite_cfg = extract_config_default(config_py, "SQLITE_PATH") or ""
    sqlite_chart = values["config"].get("sqlitePath", "")
    mount_path = values["database"]["containerPath"]

    expected_sqlite = "/home/aicm/db/train_mesh_agent.db"
    if sqlite_env != expected_sqlite:
        errors.append(f"Dockerfile ENV SQLITE_PATH={sqlite_env!r}，期望 {expected_sqlite!r}")
    if sqlite_cfg != expected_sqlite:
        errors.append(f"app/config.py 默认 SQLITE_PATH={sqlite_cfg!r}，期望 {expected_sqlite!r}")
    if sqlite_chart != expected_sqlite:
        errors.append(f"chart config.sqlitePath={sqlite_chart!r}，期望 {expected_sqlite!r}")
    if sqlite_env == sqlite_cfg == sqlite_chart == expected_sqlite:
        print(f"    OK: SQLITE_PATH 三处一致 = {expected_sqlite}")

    # chart 的 configMap 必须下发同一个值（否则部署态与镜像 ENV 漂移）
    if "SQLITE_PATH" not in configmap:
        errors.append("configMap.yaml 未下发 SQLITE_PATH")
    else:
        print("    OK: configMap.yaml 下发 SQLITE_PATH")

    # ── 2. 默认后端必须是 SQLite（DATABASE_URL 留空）──
    if "DATABASE_URL" not in envs:
        errors.append("镜像 Dockerfile 未显式声明 DATABASE_URL（应为空串 = 默认 SQLite）")
    elif envs["DATABASE_URL"] != "":
        errors.append(
            f"镜像 ENV DATABASE_URL 应为空串（默认走 SQLite），实际 {envs['DATABASE_URL']!r}；"
            "非空且不是 postgres:// 前缀会被静默当成 SQLite，属于最难排查的配置错误"
        )
    else:
        print("    OK: 镜像 DATABASE_URL 为空串 → 默认后端 SQLite")

    dsn_cfg = extract_config_default(config_py, "DATABASE_URL")
    if dsn_cfg is None:
        # 形如 os.getenv("DATABASE_URL", "") —— 第二个参数是空串，
        # 上面的正则要求非空值，因此这里单独确认「默认值是空」。
        m = re.search(r'DATABASE_URL\s*=\s*os\.getenv\(\s*"DATABASE_URL"\s*,\s*""\s*\)', config_py)
        if not m:
            errors.append("app/config.py 的 DATABASE_URL 默认值应为空串（= SQLite 后端）")
        else:
            print("    OK: config.py DATABASE_URL 默认值为空串（默认 SQLite）")
    elif dsn_cfg != "":
        errors.append(f"app/config.py 的 DATABASE_URL 默认值应为空串，实际 {dsn_cfg!r}")

    # ── 3. SQLITE_PATH 必须落在挂载点内 ──
    if not sqlite_env.startswith(mount_path.rstrip("/") + "/"):
        errors.append(
            f"SQLITE_PATH({sqlite_env}) 不在 chart 的 database.containerPath({mount_path}) 之下 —— "
            "数据库会落进镜像层，容器重建即丢全部会话历史"
        )
    else:
        print(f"    OK: SQLITE_PATH 位于挂载点 {mount_path}/ 之下")
    if mount_path != "/home/aicm/db":
        errors.append(f"chart database.containerPath={mount_path!r}，期望 /home/aicm/db")

    # ── 4. entrypoint 兜底值一致 ──
    m_sqlite_entry = re.search(r'SQLITE_PATH:=([^}]+)', entrypoint)
    sqlite_entry = m_sqlite_entry.group(1).strip() if m_sqlite_entry else ""
    if not sqlite_entry:
        errors.append("entrypoint.sh 缺少 SQLITE_PATH 兜底默认值")
    elif sqlite_entry != sqlite_env:
        errors.append(f"entrypoint.sh SQLITE_PATH 兜底值 {sqlite_entry!r} != 镜像 ENV {sqlite_env!r}")
    else:
        print(f"    OK: entrypoint.sh SQLITE_PATH 兜底值与镜像 ENV 一致 = {sqlite_entry}")

    # 工作区路径：chart / Dockerfile(mkdir) / Dockerfile(ENV) / entrypoint 兜底 四处一致。
    ws_chart = values["workspace"]["containerPath"]
    ws_env = envs.get("AICM_MCP_WORKSPACE_ROOT", "")
    m_ws_entry = re.search(r'AICM_MCP_WORKSPACE_ROOT:=([^}]+)', entrypoint)
    ws_entry = m_ws_entry.group(1).strip() if m_ws_entry else ""
    if not ws_env:
        errors.append("镜像 Dockerfile 缺少 AICM_MCP_WORKSPACE_ROOT")
    if not ws_entry:
        errors.append("entrypoint.sh 缺少 AICM_MCP_WORKSPACE_ROOT 兜底默认值")
    if ws_env and f"mkdir -p {ws_env}" not in app_df:
        errors.append(f"Dockerfile 未预建工作区挂载点 {ws_env}（entrypoint 自检会误判为未挂载）")
    ws_sources = {
        "chart workspace.containerPath": ws_chart,
        "镜像 ENV": ws_env,
        "entrypoint 兜底": ws_entry,
    }
    if len(set(ws_sources.values())) != 1:
        for label, val in ws_sources.items():
            print(f"      {label} = {val!r}")
        errors.append("工作区路径在上述四处不一致")
    else:
        print(f"    OK: 工作区路径四处一致 = {ws_chart}")

    # 数据库挂载点也必须被 Dockerfile 预建（否则自检把「未挂载」判定成「目录不存在」）
    db_dir = sqlite_env.rsplit("/", 1)[0]
    if f"mkdir -p {db_dir}" not in app_df:
        errors.append(f"Dockerfile 未预建数据库挂载点 {db_dir}")
    else:
        print(f"    OK: Dockerfile 预建数据库挂载点 {db_dir}")

    # ── 5. 旧 PG 契约必须已从**可执行内容**中移除 ──
    # 只看剥离注释后的文本：改造说明性注释里会正当地提到 PGDATA/initdb 等历史。
    exec_texts = {
        "app Dockerfile": strip_comments(app_df),
        "base Dockerfile": strip_comments(base_df),
        "entrypoint.sh": strip_comments(entrypoint),
        "values.yaml": strip_comments(values_text),
        "charts/configMap.yaml": strip_comments(configmap),
    }
    stale_tokens = [
        "PGDATA",
        "PG_SOCKET_DIR",
        "PG_BIN",
        "pgSocketDir",
        "postgresql-14",
        "pg_isready",
        # 不要用裸 "initdb" 或 "/usr/lib/postgresql" 作为 token：前者是常见词
        # （「不再需要 initdb」是正当说明），后者会撞上基础镜像里
        # 「断言 postgres 服务端**不存在**」的那条检查。用带参数的调用形态。
        "initdb -D",
        "wait_for_db",
        "ucf-shim",
    ]
    stale_found = False
    for token in stale_tokens:
        hits = [name for name, text in exec_texts.items() if token in text]
        if hits:
            stale_found = True
            errors.append(f"旧 PG 契约残留：{token} 仍出现在 {hits}")
    if not stale_found:
        print(f"    OK: 已无旧 PG 契约残留（检查 {len(stale_tokens)} 个 token × {len(exec_texts)} 个文件）")

    # ── 6. 旧写法已清除 ──
    # 先取出 SCHEMA_SQL 的字面量（不能用整文件匹配：文件头部文档注释里
    # 正当地讨论了「已删除 DROP COLUMN IF EXISTS ...」）
    schema_match = re.search(r'SCHEMA_SQL\s*=\s*"""(.*?)"""', migration_py, re.S)
    if not schema_match:
        errors.append("无法从 db_migration.py 提取 SCHEMA_SQL 字面量")
    else:
        schema_sql = schema_match.group(1)
        if "DROP COLUMN IF EXISTS" in schema_sql:
            errors.append("SCHEMA_SQL 仍含 DROP COLUMN IF EXISTS")
        else:
            print("    OK: SCHEMA_SQL 已无破坏性 DDL")
        # PG 分支保留 gen_random_uuid()/JSONB（逃生门必须仍能建表）；
        # SQLite 分支则**不该**出现它们（SQLite 无这两个类型/函数）。
        if "gen_random_uuid()" not in schema_sql or "JSONB" not in schema_sql:
            errors.append("SCHEMA_SQL（PG 分支）缺少 gen_random_uuid()/JSONB，schema 可能被误删")
        else:
            print("    OK: SCHEMA_SQL（PG 分支）保留 gen_random_uuid() 与 JSONB")
    sqlite_schema_match = re.search(r'SCHEMA_SQLITE_SQL\s*=\s*"""(.*?)"""', migration_py, re.S)
    if not sqlite_schema_match:
        errors.append("无法从 db_migration.py 提取 SCHEMA_SQLITE_SQL 字面量")
    else:
        sqlite_schema = sqlite_schema_match.group(1)
        if "gen_random_uuid()" in sqlite_schema or "JSONB" in sqlite_schema:
            errors.append("SCHEMA_SQLITE_SQL 含 PG 专有构造（gen_random_uuid()/JSONB）")
        else:
            print("    OK: SCHEMA_SQLITE_SQL 无 PG 专有构造")

    if "schema may be incomplete" in migration_py:
        errors.append("db_migration.py 仍保留『schema 可能不完整但继续启动』的旧告警")
    else:
        print("    OK: db_migration.py 不再以警告收场（改为硬校验 + 抛错）")

    # ── 6b. 后端前缀判定的两处实现必须一致 ──
    # app/dbapi.detect_backend 与 docker/check_db.py 的 backend_of 是重复实现
    # （后者要早于 app 包被 import，故不共享代码）。判定分叉会产生
    # 「自检认为走 SQLite、app 实际连 PG」这类极难排查的状态，因此在这里钉住前缀元组。
    check_db = read("docker/check_db.py")
    prefix_tuple = '("postgres://", "postgresql://")'
    if prefix_tuple not in check_db:
        errors.append(
            f"docker/check_db.py 的 PG 前缀元组不再是 {prefix_tuple}，"
            "可能已与 app/dbapi.detect_backend 分叉"
        )
    elif ".strip().lower()" not in check_db:
        errors.append("docker/check_db.py 的前缀判定未做 strip().lower()，与 app/dbapi 行为不一致")
    else:
        print("    OK: 后端前缀判定两处实现一致（dbapi / check_db）")

    # ── 6c. 直连校验脚本必须显式声明被测后端 ──
    # `_verify_schema(cur, backend=None)` 的默认是**按进程 DATABASE_URL 侦测**，而不是
    # 「这个 cursor 连的是什么」。pg_real_check.py 走 ADMIN_DSN 直连、sqlite_real_check.py
    # 走 SQLITE_PATH 直连，两者都与 DATABASE_URL 无关 —— 一旦漏传 backend，就会拿另一种
    # 方言的取数语句去查连接，抛方言无关的报错，且**只在特定环境变量组合下才暴露**。
    # 已经因此踩过一次（pg_real_check 的硬校验永远跑不通），故把调用形态钉死。
    for script in ("scripts/pg_real_check.py", "scripts/sqlite_real_check.py"):
        text = read(script)
        bare = re.findall(r"_verify_schema\(\s*cur\s*\)", text)
        if bare:
            errors.append(
                f"{script} 有 {len(bare)} 处 `_verify_schema(cur)` 未显式传 backend —— "
                "会按进程 DATABASE_URL 猜后端，与直连的连接无关"
            )
        else:
            print(f"    OK: {script} 的 _verify_schema 调用均显式声明后端")

    # ── 7. 关键文件存在性 ──
    for rel in (
        "docker/check_db.py",
        "docker/checkpoint_db.py",
        "docker/check_workspace.py",
        "docker/entrypoint.sh",
        "app/dbapi.py",
        "scripts/sqlite_real_check.py",
    ):
        if not (ROOT / rel).exists():
            errors.append(f"缺少文件 {rel}")
    for rel in ("docker/wait_for_db.py", "docker/base/ucf-shim.sh"):
        if (ROOT / rel).exists():
            errors.append(f"已废弃文件仍存在：{rel}")
    print("    OK: 启动链所需文件齐全，已废弃文件已清除")

    if notes:
        print("\n提示:")
        for n in notes:
            print(f"  - {n}")

    if errors:
        print(f"\n发现 {len(errors)} 个问题:")
        for e in errors:
            print(f"  - {e}")
        return 1

    print("\nOK: 跨文件契约一致")
    return 0


if __name__ == "__main__":
    sys.exit(main())
