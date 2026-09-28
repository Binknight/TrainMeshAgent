#!/usr/bin/env python3
"""启动前校验内嵌 PostgreSQL 的数据目录（PGDATA）。

为什么要有这个检查（与 docker/check_workspace.py 同一套理由）：
    PGDATA 若只是镜像层里的普通目录，容器重建、镜像升级时整个数据库都会消失。
    更糟的是服务表面上一切正常（建表成功、会话可写），直到下次重建才发现历史
    全没了。因此默认要求数据目录必须落在「挂载点」之下，不满足就直接退出。

    本地开发（docker run 不挂卷）确实不想挂载时，设 DB_ALLOW_LOCAL=1 可豁免
    挂载点校验（会打印醒目告警，权限与可写性检查照旧）。

检查项：
    1. PGDATA 存在且是目录
    2. PGDATA 自身或其任一父目录是挂载点（= 落在持久化卷里）
    3. PGDATA 可写（真写一个临时文件，不用 os.access）
    4. PGDATA 权限为 0700 —— PostgreSQL 对数据目录权限敏感，权限过宽会拒绝启动
    5. PGDATA 属主与父目录一致（提示用，PG 要求数据目录属主 = 运行进程用户）

退出码：0 = 通过；1 = 未通过。
"""

from __future__ import annotations

import os
import shutil
import stat
import sys
import tempfile
from pathlib import Path

DEFAULT_PGDATA = "/home/aicm/db/data"
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
        fields = line.split(" ")
        if len(fields) >= 5:
            points.add(unescape_mount_field(fields[4]))
    return points


def first_mount_ancestor(path: Path, mountinfo_path: str = MOUNTINFO_PATH) -> str | None:
    """返回 path 自身或最近的挂载点祖先；都不命中则返回 None。

    与 check_workspace.py 的 is_mount_point() 不同：这里不只比较 path 自身。
    PGDATA 是「挂载目录的子目录」（/home/aicm/db/data 挂在 /home/aicm/db 之下），
    真正被挂载的是它的父目录，因此必须向上回溯。
    """
    points = read_mount_points(mountinfo_path)
    if not points:
        # 非 Linux 或读不到 mountinfo：无法证明它是挂载点
        return None

    real_points = {os.path.realpath(p) for p in points}
    try:
        candidate = Path(os.path.realpath(path))
    except OSError:
        return None

    while True:
        if str(candidate) in real_points:
            return str(candidate)
        if str(candidate) in ("/", "") or candidate == candidate.parent:
            return None
        candidate = candidate.parent


def writability_error(path: Path) -> str:
    """真实写一个临时文件来判断可写性。

    不用 os.access(W_OK)：容器以 root 运行时它对目录权限位不敏感，
    但仍会被只读挂载挡住 —— 那种情况只有真写一次才能发现。
    """
    try:
        with tempfile.NamedTemporaryFile(dir=str(path), prefix=".db_check_"):
            pass
    except OSError as exc:
        return str(exc)
    return ""


def check(pgdata: Path, allow_local: bool, mountinfo_path: str = MOUNTINFO_PATH) -> list[str]:
    """返回致命问题列表，空列表表示通过。"""
    if not pgdata.is_dir():
        return [
            f"{pgdata} 不存在或不是目录"
            "（提示：应指向挂载进容器的持久化目录，如 /home/aicm/db/data）"
        ]

    problems: list[str] = []

    ancestor = first_mount_ancestor(pgdata, mountinfo_path)
    if ancestor:
        log(f"{pgdata} 落在挂载点 {ancestor} 上（持久化）")
    else:
        reason = f"{pgdata} 不在任何挂载点下：数据库会随镜像层/容器一起丢失"
        if allow_local:
            log(f"警告：{reason}（{ALLOW_LOCAL_ENV}=1，仅限本地开发）")
        else:
            problems.append(reason)

    error = writability_error(pgdata)
    if error:
        problems.append(f"{pgdata} 不可写（当前 uid={current_uid()}）：{error}")
    else:
        try:
            mode = stat.S_IMODE(pgdata.stat().st_mode)
            if mode != 0o700:
                problems.append(
                    f"{pgdata} 权限为 {mode:04o}，PostgreSQL 要求数据目录为 0700"
                )
            uid = pgdata.stat().st_uid
            if uid != current_uid():
                problems.append(
                    f"{pgdata} 属主 uid={uid} 与容器运行用户 uid={current_uid()} 不一致"
                    "（PG 要求数据目录属主即运行用户；k8s 下由 initContainer chown 修正）"
                )
        except OSError as exc:
            problems.append(f"无法读取 {pgdata} 的权限/属主：{exc}")

    return problems


def describe(pgdata: Path) -> None:
    try:
        st = pgdata.stat()
        free_gib = shutil.disk_usage(pgdata).free / 1024**3
        version_file = pgdata / "PG_VERSION"
        version = version_file.read_text().strip() if version_file.exists() else "未初始化"
        log(
            f"就绪：{pgdata} owner={st.st_uid}:{st.st_gid} mode={stat.S_IMODE(st.st_mode):04o} "
            f"PG_VERSION={version} 剩余空间={free_gib:.1f}GiB"
        )
    except OSError as exc:
        log(f"就绪：{pgdata}（无法读取磁盘信息：{exc}）")


def main() -> int:
    pgdata = Path(os.getenv("PGDATA") or DEFAULT_PGDATA)
    allow_local = os.getenv(ALLOW_LOCAL_ENV, "").strip().lower() in _TRUTHY

    problems = check(pgdata, allow_local, MOUNTINFO_PATH)
    if problems:
        for problem in problems:
            log(f"FATAL: {problem}")
        log(
            "提示：把宿主机目录挂到 PGDATA 的父目录（k8s hostPath / docker -v），"
            f"并保证容器运行用户（uid={current_uid()}）可写；"
            f"本地开发可设 {ALLOW_LOCAL_ENV}=1 豁免挂载点校验。"
        )
        return 1

    describe(pgdata)
    return 0


if __name__ == "__main__":
    sys.exit(main())
