"""任务状态落盘：``workspace/<task_id>/task_state.json``（服务重启恢复用）。

**为什么需要**：任务表在内存里（``task_store``），进程被 SIGKILL / 容器漂移 /
崩溃重启后，磁盘上只剩 ``task_meta.json``（没有状态）。旧版 ``task_loader`` 只能按
「有 results → completed，否则 submitted」推断，于是**运行中被打断的任务会永远停在
submitted**，调用方拿着 task_id 永远等不到终态。本模块在 prepare / queued /
running / terminal 几个迁移点把状态 best-effort 落盘，重启时由
``task_loader.load_task_record`` 与 ``task_loader.recover_interrupted_tasks`` 据此收敛。

**约定**：

- 写入必须 *best-effort*：磁盘满 / 权限 / 序列化错误都只记日志，绝不影响任务主流程；
- 状态文件是「最新快照」，每次写入都整体重写（先写临时文件再 ``os.replace``，
  避免进程恰好被 kill 时留下半截 JSON）；
- ``started_at`` / ``finished_at`` / ``updated_at`` 是 **UTC ISO-8601 字符串**
  （跨重启可读）；内存 ``TaskRecord.started_at`` 是 ``time.monotonic()`` 时间基
  （调度器算超时用），两者不要混用；
- ``dry_run`` 记录写入时刻的 ``settings.dry_run``：重启后 dry_run 任务仍应显示
  ``submitted`` 而不是被当成「中断的仿真」；
- 没有 ``workspace_dir`` 的记录（prepare 未完成）不落盘。
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from mcp_server.config import settings

logger = logging.getLogger(__name__)

TASK_STATE_FILENAME = "task_state.json"

# 进程已丢失的中间态：重启后必须收敛成终态（见 task_loader.recover_interrupted_tasks）
INTERRUPTED_STATUSES: frozenset[str] = frozenset({"running", "queued"})

# 中断/重启恢复时使用的统一提示语（契约要求 message 含这句话）
RESTART_INTERRUPTED_MESSAGE = "服务重启，任务中断（进程已丢失）"

# 恢复时「有结果」的提示语（与上面成对，便于调用方区分两种恢复结果）
RESTART_COMPLETED_MESSAGE = "服务重启，检测到已有完整仿真结果，恢复为 completed"

KEEP = object()  # write_task_state 的哨兵：保持现值


def utc_now_iso() -> str:
    """当前 UTC 时间（秒级 ISO-8601，``Z`` 结尾）。"""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def state_path(task_dir: Path) -> Path:
    return Path(task_dir) / TASK_STATE_FILENAME


def read_task_state(task_dir: Path) -> Optional[dict[str, Any]]:
    """读取 ``task_dir/task_state.json``；不存在/损坏返回 None（绝不抛异常）。"""
    try:
        raw = state_path(task_dir).read_text(encoding="utf-8")
    except OSError:
        return None
    except ValueError:  # 极端编码错误
        return None
    try:
        data = json.loads(raw)
    except ValueError:
        logger.warning("invalid task_state.json under %s", task_dir)
        return None
    return data if isinstance(data, dict) else None


def _write_payload(task_dir: Path, payload: dict[str, Any]) -> bool:
    """原子写 task_state.json（先 tmp 再 replace）；失败只记日志。"""
    task_dir = Path(task_dir)
    target = state_path(task_dir)
    tmp = target.with_name(target.name + ".tmp")
    try:
        task_dir.mkdir(parents=True, exist_ok=True)
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(tmp, target)
        return True
    except Exception as exc:  # noqa: BLE001 — 落盘失败绝不能影响任务主流程
        logger.warning("failed to write task_state.json under %s: %s", task_dir, exc)
        try:
            tmp.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            pass
        return False


def _status_value(status: Any) -> Optional[str]:
    if status is None:
        return None
    return getattr(status, "value", str(status))


def write_task_state(
    record: Any,
    *,
    status: Any = None,
    message: Any = KEEP,
    pid: Any = KEEP,
    started_at: Any = KEEP,
    finished_at: Any = KEEP,
    dry_run: Optional[bool] = None,
    extra: Optional[dict[str, Any]] = None,
) -> bool:
    """把 ``record`` 的当前状态合并进 task_state.json（best-effort）。

    未显式传入的字段保持文件里的旧值（``KEEP``）；``status`` 缺省取
    ``record.status``，``message`` 缺省取 ``record.message``，``pid`` 缺省取
    ``record.process.pid``（进程已 detach 时保持旧值，便于排查崩溃进程）。
    """
    task_dir = getattr(record, "workspace_dir", None)
    if task_dir is None:
        # prepare 尚未完成（没有任务目录）：无落盘位置，静默跳过
        return False

    payload = read_task_state(task_dir) or {}
    payload.setdefault("task_id", getattr(record, "task_id", None))

    resolved_status = _status_value(status) or _status_value(
        getattr(record, "status", None)
    )
    if resolved_status is not None:
        payload["status"] = resolved_status

    if message is not KEEP:
        payload["message"] = message
    elif "message" not in payload:
        payload["message"] = getattr(record, "message", None)

    if pid is KEEP:
        process = getattr(record, "process", None)
        live_pid = getattr(process, "pid", None) if process is not None else None
        if live_pid is not None:
            payload["pid"] = live_pid
        else:
            payload.setdefault("pid", None)
    else:
        payload["pid"] = pid

    if started_at is not KEEP:
        payload["started_at"] = started_at
    else:
        payload.setdefault("started_at", None)
    if finished_at is not KEEP:
        payload["finished_at"] = finished_at
    else:
        payload.setdefault("finished_at", None)

    if dry_run is not None:
        payload["dry_run"] = bool(dry_run)
    else:
        payload.setdefault("dry_run", bool(settings.dry_run))

    payload["updated_at"] = utc_now_iso()
    if extra:
        payload.update(extra)
    return _write_payload(Path(task_dir), payload)


def write_task_state_fields(task_dir: Path, **fields: Any) -> bool:
    """按字段更新状态文件（恢复扫描用：不依赖 TaskRecord，只认任务目录）。"""
    payload = read_task_state(task_dir) or {}
    payload.update(fields)
    payload["updated_at"] = utc_now_iso()
    return _write_payload(Path(task_dir), payload)
