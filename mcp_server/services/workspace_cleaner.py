"""workspace 磁盘清理：按优先级删除旧的仿真任务产物。

优先级（从高到低）：

1. **运行中 / 排队中的任务永不删除**（由调度器集合 + 任务记录状态共同判定）；
2. **磁盘水位**：使用率 >= ``disk_watermark_pct`` 进入应急模式（保留期降到
   ``emergency_retention_days``、必留数降到 ``emergency_min_keep_tasks``）；
   >= ``disk_critical_pct`` 进入临界模式（只保留最新 1 个任务、超过 1 天的全部删除，
   尽快把盘腾出来）；
3. **正常水位下 ``min_keep_tasks`` 优先**：最近 N 个终态任务无论多老都保留；
4. ``workspace_retention_days``：超过该天数先删 ``results/``（体积大头，通常占 99%）；
5. ``log_retention_days``：超过该天数删整个任务目录（日志/脚本/task_meta）。

所以「两个配置冲突」时的语义是：**正常水位下 min_keep_tasks 赢（保留更多）；
磁盘告急时水位规则赢（两条常规配置一起让位）**。

安全约束：
- 只处理 ``workspace_root`` 的直接子目录，且必须含 ``task_meta.json``（防路径穿越/误删）；
- 运行中/排队中的 task_id 一律跳过；
- 目录年龄取「目录 + results/ + simulation.log + task_meta.json」里最新的 mtime，
  避免文件内容更新但目录 mtime 未变导致误判为旧任务；
- 单任务删除失败只记错误、不中断整轮清理；
- ``dry_run`` 模式只统计、不落盘删除（上线前观察用）。
"""

from __future__ import annotations

import logging
import os
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

from mcp_server.config import settings

logger = logging.getLogger(__name__)

TERMINAL_STATUSES = frozenset({"completed", "failed", "error", "cancelled"})
ACTIVE_STATUSES = frozenset({"queued", "running"})

MODE_NORMAL = "normal"
MODE_EMERGENCY = "emergency"
MODE_CRITICAL = "critical"

# 临界模式固定策略：只保留最新 1 个任务，其余超过 1 天的都删（尽快止血，不再加配置项）。
_CRITICAL_KEEP_TASKS = 1
_CRITICAL_RETENTION_DAYS = 1.0

_SECONDS_PER_DAY = 86400.0


@dataclass
class CleanupStats:
    """一轮清理的统计（dry_run 时 deleted_* 表示「本来会删」的数量）。"""

    mode: str = MODE_NORMAL
    disk_pct: float = 0.0
    scanned: int = 0
    active_skipped: int = 0
    kept_min_keep: int = 0
    kept_recent: int = 0
    kept_log_only: int = 0
    skipped_unknown: int = 0
    deleted_results: int = 0
    deleted_dirs: int = 0
    freed_bytes: int = 0
    errors: int = 0
    dry_run: bool = False


def _default_protected_ids() -> set[str]:
    """默认从调度器取「运行中 + 排队中」的 task_id（导入延迟，避免循环依赖）。"""
    try:
        from mcp_server.services.task_scheduler import scheduler

        return set(scheduler.running_task_ids()) | set(scheduler.queued_task_ids())
    except Exception:  # noqa: BLE001 — 清理不能因为调度器不可用而失效
        return set()


def _default_task_status(task_id: str) -> Optional[str]:
    """默认从内存任务表取状态；未加载的任务返回 None（由年龄兜底）。"""
    try:
        from mcp_server.services.task_store import task_store

        record = task_store.get(task_id)
    except Exception:  # noqa: BLE001
        return None
    if record is None:
        return None
    status = record.status
    return getattr(status, "value", str(status))


class WorkspaceCleaner:
    """按优先级清理 workspace 下的旧任务目录。

    ``now`` / ``disk_usage`` / ``protected_ids_provider`` / ``status_provider`` 均可注入，
    便于用假时钟 + 临时目录做单测，不依赖真实磁盘水位与调度器状态。
    """

    def __init__(
        self,
        *,
        workspace_root: Path,
        result_retention_days: float = 7,
        log_retention_days: float = 14,
        min_keep_tasks: int = 20,
        disk_watermark_pct: float = 80,
        disk_critical_pct: float = 90,
        emergency_retention_days: float = 2,
        emergency_min_keep_tasks: int = 5,
        dry_run: bool = False,
        now: Callable[[], float] = time.time,
        disk_usage: Callable[[Path], tuple[int, int, int]] = shutil.disk_usage,
        protected_ids_provider: Optional[Callable[[], Iterable[str]]] = None,
        status_provider: Optional[Callable[[str], Optional[str]]] = None,
    ) -> None:
        self.workspace_root = Path(workspace_root)
        self.result_retention_days = float(result_retention_days)
        self.log_retention_days = float(log_retention_days)
        self.min_keep_tasks = max(0, int(min_keep_tasks))
        self.disk_watermark_pct = float(disk_watermark_pct)
        self.disk_critical_pct = float(disk_critical_pct)
        self.emergency_retention_days = float(emergency_retention_days)
        self.emergency_min_keep_tasks = max(0, int(emergency_min_keep_tasks))
        self.dry_run = bool(dry_run)
        self._now = now
        self._disk_usage = disk_usage
        self._protected_ids_provider = protected_ids_provider or _default_protected_ids
        self._status_provider = status_provider or _default_task_status

    # ------------------------------------------------------------------
    # 对外入口
    # ------------------------------------------------------------------
    def run_once(self) -> CleanupStats:
        stats = CleanupStats(dry_run=self.dry_run)
        root = self.workspace_root
        if not root.is_dir():
            logger.info("[cleanup] workspace 目录不存在，跳过：%s", root)
            return stats

        stats.disk_pct = self._disk_pct()
        stats.mode = self._mode(stats.disk_pct)
        retention_days, min_keep = self._effective_policy(stats.mode)

        try:
            protected = set(self._protected_ids_provider())
        except Exception:  # noqa: BLE001
            protected = set()

        candidates: list[tuple[str, Path, float, Optional[str]]] = []
        for task_id, task_dir in self._iter_task_dirs(stats):
            try:
                age_days = self._age_days(task_dir)
            except OSError:
                stats.errors += 1
                continue
            status = self._safe_status(task_id)
            candidates.append((task_id, task_dir, age_days, status))
        # 最新在前：min_keep 取的是「最近 N 个任务」
        candidates.sort(key=lambda item: item[2])

        kept = 0
        for task_id, task_dir, age_days, status in candidates:
            stats.scanned += 1
            if task_id in protected or status in ACTIVE_STATUSES:
                stats.active_skipped += 1
                continue
            if status is not None and status not in TERMINAL_STATUSES:
                # 状态未知/非终态（如 dry_run 的 submitted、磁盘恢复的 submitted）：
                # 不直接删，交给年龄兜底（保留期内一律保留）
                if age_days <= retention_days:
                    stats.kept_recent += 1
                    continue
            if kept < min_keep:
                kept += 1
                stats.kept_min_keep += 1
                continue
            if age_days <= retention_days:
                stats.kept_recent += 1
                continue

            results_dir = task_dir / "results"
            if stats.mode == MODE_NORMAL:
                if age_days <= self.log_retention_days:
                    if results_dir.is_dir():
                        self._remove(results_dir, stats, is_results=True)
                    else:
                        stats.kept_log_only += 1
                    continue
            # 超出日志保留期（正常）、或处于应急/临界模式：整目录删除
            self._remove(task_dir, stats, is_results=False)

        return stats

    # ------------------------------------------------------------------
    # 策略
    # ------------------------------------------------------------------
    def _mode(self, disk_pct: float) -> str:
        if disk_pct >= self.disk_critical_pct:
            return MODE_CRITICAL
        if disk_pct >= self.disk_watermark_pct:
            return MODE_EMERGENCY
        return MODE_NORMAL

    def _effective_policy(self, mode: str) -> tuple[float, int]:
        if mode == MODE_CRITICAL:
            return _CRITICAL_RETENTION_DAYS, _CRITICAL_KEEP_TASKS
        if mode == MODE_EMERGENCY:
            return self.emergency_retention_days, self.emergency_min_keep_tasks
        return self.result_retention_days, self.min_keep_tasks

    def _disk_pct(self) -> float:
        try:
            total, used, _free = self._disk_usage(self.workspace_root)
        except OSError:
            return 0.0
        if not total:
            return 0.0
        return round(used / total * 100.0, 2)

    # ------------------------------------------------------------------
    # 目录扫描
    # ------------------------------------------------------------------
    def _iter_task_dirs(self, stats: CleanupStats) -> list[tuple[str, Path]]:
        root = self.workspace_root
        result: list[tuple[str, Path]] = []
        try:
            entries = list(root.iterdir())
        except OSError:
            return result
        for entry in entries:
            try:
                if not entry.is_dir():
                    continue
                # 只认 workspace 的直接子目录 + 含 task_meta.json 的任务目录
                if not (entry / "task_meta.json").is_file():
                    stats.skipped_unknown += 1
                    continue
                if entry.resolve().parent != root.resolve():
                    stats.skipped_unknown += 1
                    continue
            except OSError:
                stats.skipped_unknown += 1
                continue
            result.append((entry.name, entry))
        return result

    def _age_days(self, task_dir: Path) -> float:
        last = task_dir.stat().st_mtime
        for sub in (task_dir / "results", task_dir / "simulation.log", task_dir / "task_meta.json"):
            try:
                last = max(last, sub.stat().st_mtime)
            except OSError:
                continue
        return (self._now() - last) / _SECONDS_PER_DAY

    def _safe_status(self, task_id: str) -> Optional[str]:
        try:
            return self._status_provider(task_id)
        except Exception:  # noqa: BLE001
            return None

    # ------------------------------------------------------------------
    # 删除
    # ------------------------------------------------------------------
    def _remove(self, path: Path, stats: CleanupStats, *, is_results: bool) -> None:
        try:
            size = self._dir_size(path)
        except OSError:
            size = 0
        if self.dry_run:
            stats.freed_bytes += size
            if is_results:
                stats.deleted_results += 1
            else:
                stats.deleted_dirs += 1
            logger.info("[cleanup][dry_run] would remove %s (%.1f MB)", path, size / (1024 * 1024))
            return
        try:
            shutil.rmtree(path)
        except OSError as exc:
            stats.errors += 1
            logger.warning("[cleanup] 删除失败 path=%s err=%s", path, exc)
            return
        stats.freed_bytes += size
        if is_results:
            stats.deleted_results += 1
        else:
            stats.deleted_dirs += 1
        logger.info("[cleanup] removed %s (%.1f MB)", path, size / (1024 * 1024))

    @staticmethod
    def _dir_size(path: Path) -> int:
        total = 0
        for current, _dirs, files in os.walk(path):
            for name in files:
                try:
                    total += (Path(current) / name).stat().st_size
                except OSError:
                    continue
        return total


def build_cleaner_from_settings() -> WorkspaceCleaner:
    return WorkspaceCleaner(
        workspace_root=settings.workspace_root,
        result_retention_days=settings.workspace_retention_days,
        log_retention_days=settings.log_retention_days,
        min_keep_tasks=settings.min_keep_tasks,
        disk_watermark_pct=settings.disk_watermark_pct,
        disk_critical_pct=settings.disk_critical_pct,
        emergency_retention_days=settings.emergency_retention_days,
        emergency_min_keep_tasks=settings.emergency_min_keep_tasks,
        dry_run=settings.cleanup_dry_run,
    )


class CleanupWorker:
    """后台清理线程：启动后先等 initial_delay 扫一次，之后每 interval 扫一次。"""

    def __init__(
        self,
        *,
        enabled: bool = True,
        interval_sec: float = 86400.0,
        initial_delay_sec: float = 10.0,
        cleaner_factory: Callable[[], WorkspaceCleaner] = build_cleaner_from_settings,
        name: str = "workspace-cleanup",
    ) -> None:
        self.enabled = bool(enabled)
        self.interval_sec = max(1.0, float(interval_sec))
        self.initial_delay_sec = max(0.0, float(initial_delay_sec))
        self._cleaner_factory = cleaner_factory
        self._name = name
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if not self.enabled:
            logger.info("[cleanup] workspace cleanup disabled by config")
            return
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name=self._name, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=5.0)
        self._thread = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _run(self) -> None:
        if self._stop.wait(self.initial_delay_sec):
            return
        while not self._stop.is_set():
            try:
                stats = self._cleaner_factory().run_once()
                logger.info(
                    "[cleanup] mode=%s disk=%.1f%% scanned=%d active_skipped=%d "
                    "kept(min_keep=%d recent=%d log_only=%d) deleted(results=%d dirs=%d) "
                    "freed=%.1fMB errors=%d dry_run=%s",
                    stats.mode,
                    stats.disk_pct,
                    stats.scanned,
                    stats.active_skipped,
                    stats.kept_min_keep,
                    stats.kept_recent,
                    stats.kept_log_only,
                    stats.deleted_results,
                    stats.deleted_dirs,
                    stats.freed_bytes / (1024 * 1024),
                    stats.errors,
                    stats.dry_run,
                )
            except Exception:  # noqa: BLE001 — 清理线程不能因单轮异常退出
                logger.exception("[cleanup] run failed")
            if self._stop.wait(self.interval_sec):
                return


# 进程级单例：main.py 启动时 start()（启动后先等 initial_delay，再每天扫一次）
cleanup_worker = CleanupWorker(
    enabled=settings.workspace_cleanup_enabled,
    interval_sec=float(settings.cleanup_interval_hours) * 3600.0,
    initial_delay_sec=float(settings.cleanup_initial_delay_sec),
)
