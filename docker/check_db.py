#!/usr/bin/env python3
"""启动前校验数据库目录（默认后端 SQLite）。

为什么要有这个检查（与 docker/check_workspace.py 同一套理由）：
    数据库文件若只是镜像层里的普通文件，容器重建、镜像升级时整个数据库都会消失。
    更糟的是服务表面上一切正常（建表成功、会话可写），直到下次重建才发现历史
    全没了。因此默认要求数据库目录必须落在「挂载点」之下，不满足就直接退出。

    本地开发（docker run 不挂卷）确实不想挂载时，设 DB_ALLOW_LOCAL=1 可豁免
    挂载点校验（会打印醒目告警，可写性检查照旧）。

检查项（SQLite 后端，默认）：
    1. SQLITE_PATH 的父目录存在且是目录
    2. 该目录自身或其任一父目录是挂载点（= 落在持久化卷里）
    3. 该目录可写（真写一个临时文件，不用 os.access）
    4. 若数据库文件已存在，确认它是普通文件且可写

检查项（PG 逃生门：DATABASE_URL 指向 postgresql:// 时）：
    数据库目录不参与持久化（数据在外部 PG 上），因此只做提示，不阻断启动。

与改造前的差异：原实现校验 PGDATA 的 0700 权限与属主一致性 —— 那是 PostgreSQL
特有的敏感点（权限过宽会拒绝启动），SQLite 没有这个要求，一并去掉。

退出码：0 = 通过；1 = 未通过。
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

DEFAULT_SQLITE_PATH = "/home/aicm/db/train_mesh_agent.db"
MOUNTINFO_PATH = "/proc/self/mountinfo"
ALLOW_LOCAL_ENV = "DB_ALLOW_LOCAL"
_TRUTHY = {"1", "true", "yes", "on"}

# /proc/self/mountinfo 的路径字段用八进制转义空格等字符
_ESCAPES = (("\\040", " "), ("\\011", "\t"), ("\\012", "\n"), ("\\134", "\\"))


def log(message: str) -> None:
    print(f"[db-check] {message}", flush=True)


def current_uid() -> int:
    return os.getuid() if hasattr(os, "getuid") else -1


def unescape_mount_field(field: str) -> str:
    for escaped, plain in _ESCAPES:
        field = field.replace(escaped, plain)
    return field


def read_mount_points(mountinfo_path: str = MOUNTINFO_PATH) -> set[str]:
    """返回 mountinfo 中的所有挂载点（第 5 列，0 基下标 4）。"""
    try:
        raw = Path(mountinfo_path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return set()

    points: set[str] = set()
    for line in raw.splitlines():
        fields = line.split()
        if len(fields) >= 5:
            points.add(unescape_mount_field(fields[4]).rstrip("/") or "/")
    return points


def is_under_mount_point(target: Path, mount_points: set[str]) -> str | None:
    """target 自身或其任一祖先是否落在挂载点上；返回命中的挂载点。"""
    if not mount_points:
        return None
    chain = [target, *target.parents]
    for candidate in chain:
        key = str(candidate).rstrip("/") or "/"
        if key in mount_points:
            return key
    return None


def writable_probe(directory: Path) -> bool:
    """真写一个临时文件（不使用 os.access —— 受限环境下它可能失真）。"""
    try:
        with tempfile.NamedTemporaryFile(dir=directory, prefix=".dbcheck-", delete=True):
            pass
        return True
    except OSError as exc:
        log(f"目录不可写：{directory}（{exc}）")
        return False


def backend_of(database_url: str) -> str:
    """按 DSN 前缀判定后端。

    ⚠️ 这段逻辑与 `app/dbapi.detect_backend` **有意重复**：本脚本在容器里以
    `/home/docker/check_db.py` 独立运行，早于仓库 `app/` 包被 import，为了不引入
    路径/依赖耦合而自带一份。两处必须保持一致 —— 判定一旦分叉，会出现
    「check_db 认为要用 SQLite（于是校验挂载点）而 app 实际连 PG」这类诡异现象。
    `scripts/verify_consistency.py` 会对这段前缀元组做断言，改一处忘了另一处会直接失败。
    """
    url = (database_url or "").strip().lower()
    return "postgres" if url.startswith(("postgres://", "postgresql://")) else "sqlite"


def main() -> int:
    failures: list[str] = []

    database_url = os.getenv("DATABASE_URL", "").strip()
    active = backend_of(database_url)

    if active == "postgres":
        # PG 逃生门：数据在外部 PG 上，本地数据库目录不参与持久化。
        log(f"检测到 DATABASE_URL 指向 PostgreSQL（{database_url.split('@')[-1]}），"
            f"跳过本地数据库目录持久化校验。")
        log("提示：此时容器内不再运行任何数据库进程，数据生命周期由外部 PG 负责；")
        log("     若为空库，首次启动会自动建表并 seed 内置模型清单。")
        return 0

    sqlite_path = Path(os.getenv("SQLITE_PATH") or DEFAULT_SQLITE_PATH)
    db_dir = sqlite_path.parent

    log(f"后端=sqlite  数据文件={sqlite_path}")
    log(f"数据库目录={db_dir}  运行用户 uid={current_uid()}")

    # 1. 父目录存在且是目录
    if not db_dir.exists():
        failures.append(
            f"数据库目录不存在：{db_dir}（应由镜像预建并由部署侧挂载宿主机目录）"
        )
    elif not db_dir.is_dir():
        failures.append(f"数据库目录不是目录：{db_dir}")

    if failures:
        for f in failures:
            log(f"FATAL: {f}")
        _print_hint(db_dir)
        return 1

    # 2. 挂载点校验（唯一可豁免项）
    mount_points = read_mount_points()
    hit = is_under_mount_point(db_dir, mount_points)
    allow_local = os.getenv(ALLOW_LOCAL_ENV, "").strip().lower() in _TRUTHY

    if hit:
        log(f"挂载点校验通过：{db_dir} 位于挂载点 {hit} 之下")
    elif not mount_points:
        log("警告：本平台无 /proc/self/mountinfo，跳过挂载点校验（非 Linux 环境正常）")
    elif allow_local:
        log("=" * 68)
        log(f"警告：{db_dir} 不在任何挂载点之下，但 {ALLOW_LOCAL_ENV} 已设置，继续启动。")
        log("      容器重建或镜像升级会**丢失全部会话历史**。仅供本地开发使用！")
        log("=" * 68)
    else:
        failures.append(
            f"数据库目录 {db_dir} 不在挂载点之下 —— 数据库会落在可写镜像层里，"
            f"容器重建即丢全部会话历史"
        )

    # 3. 目录可写
    if not writable_probe(db_dir):
        failures.append(f"数据库目录不可写：{db_dir}（检查属主与权限）")

    # 4. 已存在的数据库文件必须是普通文件且可写
    if sqlite_path.exists():
        if not sqlite_path.is_file():
            failures.append(f"数据库路径已存在但不是普通文件：{sqlite_path}")
        elif not os.access(sqlite_path, os.W_OK):
            failures.append(f"数据库文件不可写：{sqlite_path}")
        else:
            log(f"数据库文件已存在且可写（{sqlite_path.stat().st_size} 字节）")
        for suffix in ("-wal", "-shm"):
            companion = sqlite_path.with_name(sqlite_path.name + suffix)
            if companion.exists():
                log(f"  WAL 伴生文件存在：{companion.name}（正常，停机时会自动合并）")
    else:
        log("数据库文件尚不存在，将在首次启动时创建")

    if failures:
        for f in failures:
            log(f"FATAL: {f}")
        _print_hint(db_dir)
        return 1

    log("数据库目录自检通过")
    return 0


def _print_hint(db_dir: Path) -> None:
    log(f"  排查建议：把宿主机目录挂到 {db_dir}（docker -v / k8s hostPath），")
    log(f"           并确保其属主是容器运行用户（uid={current_uid()}）。")
    log("           k8s 下由 initContainer 执行 chown（见 charts/templates/deployment.yaml）。")
    log(f"           仅本地验证时可设 {ALLOW_LOCAL_ENV}=1 豁免挂载点校验。")


if __name__ == "__main__":
    sys.exit(main())
