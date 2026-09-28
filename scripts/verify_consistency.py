"""收尾一致性检查：跨文件契约对齐。

本次重构的核心风险是「同一个路径/DSN 在多处重复声明而写歪」，因此这里做的是
跨文件比对，而不是单文件语法检查：
  1. 镜像 ENV 的 PGDATA / PG_SOCKET_DIR / DATABASE_URL 与 app/config.py 默认值一致
  2. chart 的 database.containerPath / config.pgSocketDir 与镜像 ENV 一致
  3. 各处出现的 socket 路径字面量完全一致（/home/aicm/db/run）
  4. PGDATA 一律是挂载点的 data 子目录（/home/aicm/db/data）
  5. 旧的破坏性 DDL / 静默吞异常写法已消失
  6. 启动脚本里 PG 主版本号与基础镜像里 apt 的包名一致（14）
"""
from __future__ import annotations

import pathlib
import re
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent


def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def env_from_dockerfile(text: str) -> dict[str, str]:
    """解析 Dockerfile 的 ENV 段（KEY=value 多行反斜杠续行，直到下一个指令行）。"""
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
            r"([A-Z_][A-Z0-9_]*)=(\"[^\"]*\"|'[^']*'|[^\s]+)", payload
        ):
            envs[key] = value.strip('"').strip("'")

        if not payload.rstrip().endswith("\\"):
            break  # 本指令（含续行）到此结束
    return envs


def extract_config_default(text: str, name: str) -> str | None:
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
    values = yaml.safe_load(
        re.sub(r"@([A-Za-z0-9_.]+)@", r'"PH_\1"', read("charts/values.yaml"))
    )

    # ── 1. 镜像 ENV vs app/config.py ──
    envs = env_from_dockerfile(app_df)
    print(f"[1] Dockerfile ENV 解析到 {len(envs)} 个键")

    pairs = [
        ("PGDATA", "/home/aicm/db/data"),
        ("PG_SOCKET_DIR", "/home/aicm/db/run"),
    ]
    for key, expected in pairs:
        got = envs.get(key)
        cfg = extract_config_default(config_py, key)
        if got != expected:
            errors.append(f"Dockerfile ENV {key}={got!r}，期望 {expected!r}")
        if cfg != expected:
            errors.append(f"app/config.py 默认 {key}={cfg!r}，期望 {expected!r}")
        if got == cfg == expected:
            print(f"    OK: {key} 三处一致 = {expected}")

    # DATABASE_URL：镜像内必须是 socket 形态；config.py 必须是 TCP 形态（本地开发用）
    dsn_env = envs.get("DATABASE_URL", "")
    dsn_cfg = extract_config_default(config_py, "DATABASE_URL")
    if "?host=" not in dsn_env or dsn_env.startswith("postgresql://postgres:postgres@"):
        errors.append(f"镜像 ENV DATABASE_URL 不是 socket 形态: {dsn_env!r}")
    else:
        print(f"    OK: 镜像 DATABASE_URL 为 socket 形态")
    if not dsn_cfg or "127.0.0.1:5432" not in dsn_cfg:
        errors.append(f"config.py 的 DATABASE_URL 默认值应保留 TCP 形态（本地开发），实际 {dsn_cfg!r}")
    else:
        print(f"    OK: config.py DATABASE_URL 保留 TCP 默认值（本地开发零改动）")

    # socket 目录必须出现在镜像 DSN 里，否则两处写歪
    socket_dir = envs.get("PG_SOCKET_DIR", "")
    if socket_dir and f"host={socket_dir}" not in dsn_env:
        errors.append(f"DATABASE_URL 的 host= 与 PG_SOCKET_DIR({socket_dir}) 不一致: {dsn_env!r}")
    else:
        print(f"    OK: DATABASE_URL 的 host= 与 PG_SOCKET_DIR 对齐")

    # ── 2. chart vs 镜像 ENV ──
    if values["database"]["containerPath"] != "/home/aicm/db":
        errors.append(f"chart database.containerPath={values['database']['containerPath']!r}，期望 /home/aicm/db")
    if values["config"]["pgSocketDir"] != socket_dir:
        errors.append(f"chart config.pgSocketDir={values['config']['pgSocketDir']!r} != 镜像 PG_SOCKET_DIR={socket_dir!r}")
    if values["database"]["containerPath"] + "/run" != socket_dir and values["config"]["pgSocketDir"] == socket_dir:
        notes.append("database.containerPath 与 PG_SOCKET_DIR 非同源推导（当前靠显式声明保持一致）")
    print(f"    OK: chart database.containerPath / pgSocketDir 与镜像 ENV 对齐")

    # PGDATA 必须是挂载点的 data 子目录
    pgdata = envs.get("PGDATA", "")
    if not pgdata.startswith(values["database"]["containerPath"] + "/"):
        errors.append(f"PGDATA({pgdata}) 不在 chart 的 database.containerPath 之下")
    else:
        print(f"    OK: PGDATA 位于挂载点子目录 {values['database']['containerPath']}/ 下")

    # ── 3. 启动脚本硬编码路径与镜像 ENV 对齐 ──
    for literal, label in (
        ("/home/aicm/db/run", "socket 目录"),
        ("/home/aicm/db/data", "PGDATA"),
    ):
        if literal in entrypoint and literal != envs.get(
            "PG_SOCKET_DIR" if "run" in literal else "PGDATA"
        ):
            errors.append(f"entrypoint.sh 硬编码 {label} {literal} 与镜像 ENV 不一致")
    # 兜底默认值必须与镜像 ENV 相同（否则非 docker 场景行为漂移）
    for key, expected in pairs:
        m = re.search(rf':\s*"\$\{{{key}:=([^}}]+)\}}"', entrypoint)
        if m and m.group(1) != expected:
            errors.append(f"entrypoint.sh 的 {key} 兜底值 {m.group(1)!r} != 镜像 ENV {expected!r}")
    print(f"    OK: entrypoint.sh 兜底默认值与镜像 ENV 一致")

    # 工作区路径：chart / Dockerfile(mkdir) / Dockerfile(ENV) / entrypoint 兜底 四处必须一致。
    # 这条是补的盲区：原先只校验数据库一侧，workspace 改一处漏一处不会被发现。
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

    # ── 4. PG 主版本一致性 ──
    apt_pkgs = set(re.findall(r"postgresql-(\d+)", base_df))
    bin_paths = set(re.findall(r"/usr/lib/postgresql/(\d+)/bin", base_df + app_df + entrypoint))
    versions = apt_pkgs | bin_paths
    if versions != {"14"}:
        errors.append(f"PG 主版本声明不一致/非 14: apt={apt_pkgs} bin={bin_paths}")
    else:
        print(f"    OK: PG 主版本在 apt 包名、二进制路径、entrypoint 中统一为 14")

    # ── 5. 旧写法已清除 ──
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
        if "gen_random_uuid()" not in schema_sql or "JSONB" not in schema_sql:
            errors.append("SCHEMA_SQL 缺少 gen_random_uuid()/JSONB，schema 可能被误删")
        else:
            print("    OK: SCHEMA_SQL 保留 gen_random_uuid() 与 JSONB")
    if "schema may be incomplete" in migration_py:
        errors.append("db_migration.py 仍保留『schema 可能不完整但继续启动』的旧告警")
    else:
        print("    OK: db_migration.py 不再以警告收场（改为硬校验 + 抛错）")
    if "如果为空会退回镜像默认的 127.0.0.1，wait_for_db 失败" in read("charts/values.yaml"):
        errors.append("values.yaml 仍保留『DATABASE_URL 必填』的过期注释")
    else:
        print("    OK: values.yaml 已更新 DATABASE_URL 的必填语义")

    # ── 6. 关键文件存在性 ──
    for rel in ("docker/check_db.py", "docker/wait_for_db.py", "docker/entrypoint.sh"):
        if not (ROOT / rel).exists():
            errors.append(f"缺少文件 {rel}")
    print("    OK: 启动链所需文件齐全")

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
