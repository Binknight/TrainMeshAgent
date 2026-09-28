#!/usr/bin/env python3
"""启动前校验仿真工作区（AICM_MCP_WORKSPACE_ROOT）。

为什么要有这个检查：
    仿真产物（task_meta.json / simulation.log / topology_generated.sh / results/）
    全部写在 AICM_MCP_WORKSPACE_ROOT 下。如果该路径只是镜像层里的普通目录，
    容器重建、镜像升级时产物会随层一起消失；而服务表面上一切正常，直到有人来
    取结果才发现。因此默认要求它必须是「挂载点」，不满足就直接退出，把问题
    暴露在启动阶段而不是排查阶段。

    本地开发确实不想挂载时，设 AICM_MCP_WORKSPACE_ALLOW_LOCAL=1 可豁免挂载点
    校验（会打印醒目告警，其余检查照旧）。

退出码：0 = 通过；1 = 未通过。
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

DEFAULT_WORKSPACE = "/data/aicm/workspace"
MOUNTINFO_PATH = "/proc/self/mountinfo"
ALLOW_LOCAL_ENV = "AICM_MCP_WORKSPACE_ALLOW_LOCAL"
_TRUTHY = {"1", "true", "yes", "on"}

# /proc/self/mountinfo 的路径字段用八进制转义空格等字符
_ESCAPES = (("\\040", " "), ("\\011", "\t"), ("\\012", "\n"), ("\\134", "\\"))


def log(message: str) -> None:
    print(f"[workspace] {message}", flush=True)


def current_uid() -> int:
    return os.getuid() if hasattr(os, "getuid") else -1


def unescape_mount_field(field: str) -> str:
    for escaped, plain in _ESCAPES:
        field = field.replace(escaped, plain)
    return field


def read_mount_points(mountinfo_path: str = MOUNTINFO_PATH) -> set[str]:
    """返回 mountinfo 中的所有挂载点（第 5 列，0 基下标 4）。

    mountinfo 行格式：
        36 35 98:0 /mnt1 /mnt2 rw,noatime master:1 - ext3 /dev/root rw
        ①  ②  ③    ④     ⑤     ⑥
    """
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


def is_mount_point(workspace: Path, mountinfo_path: str = MOUNTINFO_PATH) -> bool:
    """workspace 是否为挂载点（含 bind mount 与 k8s hostPath）。"""
    points = read_mount_points(mountinfo_path)
    if not points:
        # 非 Linux 或读不到 mountinfo：无法证明它是挂载点
        return False
    try:
        target = os.path.realpath(workspace)
    except OSError:
        return False
    return any(os.path.realpath(point) == target for point in points)


def writability_error(workspace: Path) -> str:
    """真实写一个临时文件来判断可写性。

    不用 os.access(W_OK)：容器以 root 运行时它对目录权限位不敏感，
    但仍会被只读挂载挡住——那种情况只有真写一次才能发现。
    """
    try:
        with tempfile.NamedTemporaryFile(dir=str(workspace), prefix=".aicm_ws_check_"):
            pass
    except OSError as exc:
        return str(exc)
    return ""


def check(
    workspace: Path,
    allow_local: bool,
    mountinfo_path: str = MOUNTINFO_PATH,
) -> list[str]:
    """返回致命问题列表，空列表表示通过。"""
    if not workspace.is_dir():
        return [f"{workspace} 不存在或不是目录"]

    problems: list[str] = []

    if is_mount_point(workspace, mountinfo_path):
        log(f"{workspace} 是挂载点")
    else:
        reason = f"{workspace} 不是挂载点：产物会写进镜像层，容器重建即丢失"
        if allow_local:
            log(f"警告：{reason}（{ALLOW_LOCAL_ENV}=1，仅限本地开发）")
        else:
            problems.append(reason)

    error = writability_error(workspace)
    if error:
        problems.append(f"{workspace} 不可写（当前 uid={current_uid()}）：{error}")

    return problems


def describe(workspace: Path) -> None:
    try:
        stat = workspace.stat()
        free_gib = shutil.disk_usage(workspace).free / 1024**3
        log(
            f"就绪：{workspace} owner={stat.st_uid}:{stat.st_gid} "
            f"剩余空间={free_gib:.1f}GiB"
        )
    except OSError as exc:
        log(f"就绪：{workspace}（无法读取磁盘信息：{exc}）")


def main() -> int:
    workspace = Path(os.getenv("AICM_MCP_WORKSPACE_ROOT") or DEFAULT_WORKSPACE)
    allow_local = os.getenv(ALLOW_LOCAL_ENV, "").strip().lower() in _TRUTHY

    problems = check(workspace, allow_local, MOUNTINFO_PATH)
    if problems:
        for problem in problems:
            log(f"FATAL: {problem}")
        log(
            "提示：把宿主机目录挂到该路径（k8s hostPath / docker -v），"
            f"并保证容器运行用户（uid={current_uid()}）可写；"
            f"本地开发可设 {ALLOW_LOCAL_ENV}=1 豁免挂载点校验。"
        )
        return 1

    describe(workspace)
    return 0


if __name__ == "__main__":
    sys.exit(main())
