"""workspace/<task_id> 下仿真产物的文件列举 / 定位 / ZIP 打包。

供 MCP Server 的 REST 下载接口使用（``GET /tasks/{task_id}/files*``）：

- **列举**：返回任务目录内所有常规文件的相对路径、大小、类别与任务状态；
  类别用于前端「仿真流量文件」下载面板分组（流量明细 / 算子时间线 / 统计汇总 /
  训练脚本 / 运行日志 / 其他）。
- **单文件下载**：把相对路径安全地解析到任务目录内，拒绝 ``..``、绝对路径、
  符号链接逃逸（解析后必须仍位于任务目录内）。
- **打包下载**：在 MCP Server 侧生成 ZIP（按类别可选过滤），供 Flask 代理流式
  下载；文件较大时只在 MCP 侧产生一次性临时文件，响应完成后删除。

设计约束：
- 任务目录由 ``settings.workspace_root`` 决定，**不允许**跨任务/跨目录读取；
- workspace 可能被磁盘清理器删除，这里一律按「任务不存在 / 文件不存在」返回
  404，由调用方展示「已被清理」；
- 打包上限由 ``settings.files_bundle_max_bytes`` / ``files_max_list`` 把守，
  避免超大目录把临时盘写满。
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Optional

from mcp_server.config import settings
from mcp_server.services.task_state import read_task_state

logger = logging.getLogger(__name__)

# task_id 由 task_store.generate_task_id 生成（sim_<ts>_<rand>）；测试/历史任务
# 也允许普通字母数字组合。拒绝路径分隔符，防止 task_id 本身造成目录穿越。
_TASK_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{1,127}$")

CATEGORY_ORDER: tuple[str, ...] = (
    "traffic_detail",
    "operator_timeline",
    "statistic_summary",
    "script",
    "logs",
    "other",
)
CATEGORY_LABELS: dict[str, str] = {
    "traffic_detail": "流量明细",
    "operator_timeline": "算子时间线",
    "statistic_summary": "统计汇总",
    "script": "训练脚本",
    "logs": "运行日志",
    "other": "其他文件",
}


class TaskFilesError(Exception):
    """文件接口的可预期错误（带 HTTP 状态码）。"""

    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.message = message


def _safe_task_dir(task_id: str) -> Path:
    if not task_id or not _TASK_ID_RE.match(task_id):
        raise TaskFilesError(400, "invalid task_id")
    root = settings.workspace_root.resolve()
    task_dir = (root / task_id).resolve()
    # 双保险：task_id 已限制字符集，这里再确认解析后仍是 workspace 的直接子目录
    if task_dir.parent != root:
        raise TaskFilesError(400, "invalid task_id")
    if not task_dir.is_dir():
        raise TaskFilesError(404, f"task workspace not found or cleaned: {task_id}")
    return task_dir


def classify_rel_path(rel_path: str) -> str:
    """按目录/文件名给产物分类（与前端下载面板的分组顺序一致）。"""
    posix = rel_path.replace("\\", "/").lower()
    name = posix.rsplit("/", 1)[-1]
    if name.endswith("_time.csv"):
        return "operator_timeline"
    if name.endswith("_workload.csv"):
        return "traffic_detail"
    if "/statistic_data/" in f"/{posix}" or posix.startswith("statistic_data/"):
        return "statistic_summary"
    if name.endswith(".log"):
        return "logs"
    if name.endswith(".sh"):
        return "script"
    return "other"


def _iter_task_files(task_dir: Path) -> list[tuple[Path, str, int]]:
    """返回 [(绝对路径, 相对 posix 路径, 字节数)]，按路径排序。

    遍历上限由 ``files_max_list`` 把守；超过上限返回截断标记由调用方决定是否报错。
    """
    entries: list[tuple[Path, str, int]] = []
    for path in sorted(task_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(task_dir).as_posix()
        # 打包/测试残留、原子写的临时文件不对外暴露
        if rel.endswith(".tmp") or rel.endswith(".zip"):
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        entries.append((path, rel, size))
    return entries


def _group_files(files: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for f in files:
        category = f["category"]
        group = groups.setdefault(
            category,
            {
                "category": category,
                "label": CATEGORY_LABELS.get(category, category),
                "count": 0,
                "size_bytes": 0,
                "formats": set(),
            },
        )
        group["count"] += 1
        group["size_bytes"] += f["size"]
        suffix = Path(f["path"]).suffix.lstrip(".").lower()
        if suffix:
            group["formats"].add(suffix)
    ordered: list[dict[str, Any]] = []
    for category in CATEGORY_ORDER:
        if category not in groups:
            continue
        group = groups[category]
        group["formats"] = sorted(group["formats"])
        ordered.append(group)
    return ordered


def list_task_files(task_id: str) -> dict[str, Any]:
    """列举任务目录下的文件与任务状态（不读取文件内容）。"""
    task_dir = _safe_task_dir(task_id)
    all_entries = _iter_task_files(task_dir)
    truncated = len(all_entries) > settings.files_max_list
    entries = all_entries[: settings.files_max_list]

    files: list[dict[str, Any]] = []
    total_size = 0
    for path, rel, size in entries:
        files.append(
            {
                "path": rel,
                "size": size,
                "category": classify_rel_path(rel),
                "modified_at": _mtime_iso(path),
            }
        )
        total_size += size

    state = read_task_state(task_dir) or {}
    status = state.get("status")
    if not status:
        # 无 task_state 的旧任务：有结果就按 completed，否则 unknown（前端会显示未知）
        has_results = (task_dir / "results" / "mocked_workload").is_dir()
        status = "completed" if has_results else "unknown"

    return {
        "task_id": task_id,
        "status": str(status),
        "message": state.get("message"),
        "started_at": state.get("started_at"),
        "finished_at": state.get("finished_at"),
        "updated_at": state.get("updated_at"),
        "files": files,
        "groups": _group_files(files),
        "total_files": len(all_entries),
        "total_size_bytes": total_size,
        "truncated": truncated,
        "limits": {
            "max_files": settings.files_max_list,
            "max_bundle_bytes": settings.files_bundle_max_bytes,
        },
    }


def _mtime_iso(path: Path) -> Optional[str]:
    try:
        from datetime import datetime, timezone

        return datetime.fromtimestamp(
            path.stat().st_mtime, tz=timezone.utc
        ).isoformat()
    except OSError:
        return None


def _normalize_rel_path(rel_path: str) -> str:
    if not rel_path or "\x00" in rel_path:
        raise TaskFilesError(400, "invalid file path")
    normalized = rel_path.replace("\\", "/")
    if normalized.startswith("/"):
        raise TaskFilesError(400, "invalid file path")
    parts = normalized.split("/")
    if any(part in ("..", "") for part in parts):
        raise TaskFilesError(400, "invalid file path")
    return "/".join(parts)


def resolve_file(task_id: str, rel_path: str) -> Path:
    """把相对路径解析为任务目录内的真实文件；任何越权/穿越都抛 TaskFilesError。"""
    task_dir = _safe_task_dir(task_id)
    normalized = _normalize_rel_path(rel_path)
    target = (task_dir / normalized).resolve()
    try:
        target.relative_to(task_dir)
    except ValueError:
        raise TaskFilesError(400, "invalid file path") from None
    if not target.is_file():
        raise TaskFilesError(404, f"file not found: {normalized}")
    return target


def create_bundle(task_id: str, category: Optional[str] = None) -> tuple[Path, str]:
    """把任务目录（可按类别过滤）打成 ZIP，返回 (临时文件路径, 下载文件名)。

    临时文件由 HTTP 响应结束后删除（见 main.py 的 BackgroundTask）；失败时这里
    负责清理，不把半截 ZIP 留给下一个人。
    """
    if category and category not in CATEGORY_LABELS:
        raise TaskFilesError(400, f"unknown category: {category}")

    task_dir = _safe_task_dir(task_id)
    all_entries = _iter_task_files(task_dir)
    if len(all_entries) > settings.files_max_list:
        raise TaskFilesError(
            413,
            f"too many files to bundle ({len(all_entries)} > {settings.files_max_list})",
        )
    entries = [
        item
        for item in all_entries
        if not category or classify_rel_path(item[1]) == category
    ]
    if not entries:
        label = CATEGORY_LABELS.get(category or "", "全部文件")
        raise TaskFilesError(404, f"no files to bundle for {label}")

    total_bytes = sum(size for _, _, size in entries)
    if settings.files_bundle_max_bytes > 0 and total_bytes > settings.files_bundle_max_bytes:
        raise TaskFilesError(
            413,
            f"bundle too large: {total_bytes} bytes > {settings.files_bundle_max_bytes}",
        )

    fd, tmp_name = tempfile.mkstemp(
        prefix=f"mcp_bundle_{task_id}_", suffix=".zip"
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path, rel, _size in entries:
                zf.write(path, arcname=rel)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    suffix = f"_{category}" if category else "_all"
    filename = f"{task_id}{suffix}.zip"
    logger.info(
        "[task_files] bundled %d files (%d bytes) for %s -> %s",
        len(entries),
        total_bytes,
        task_id,
        tmp_path.name,
    )
    return tmp_path, filename


def remove_quietly(path: Path) -> None:
    """响应结束后的 best-effort 清理（Windows 上文件可能仍被占用，失败只记日志）。"""
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:  # noqa: BLE001
        logger.warning("[task_files] failed to remove temp bundle %s: %s", path, exc)
