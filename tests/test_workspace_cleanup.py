"""workspace 磁盘清理回归测试（mcp_server/services/workspace_cleaner.py）。

锁住「优先级」语义（这是最容易改错的点）：
  1. 运行中/排队中任务永不删；
  2. 正常水位下 min_keep_tasks 优先于 workspace_retention_days（冲突时保留更多）；
  3. 保留期两阶段：超过结果保留期先删 results/，超过日志保留期才删整个目录；
  4. 磁盘排水位：>=80% 应急（更短保留期/更小必留数）、>=90% 临界（只留最新 1 个）；
  5. dry_run 只统计不删除；未知目录/非任务目录不动；
  6. CleanupWorker 启动后会先扫一次，禁用时不启动。

不依赖真实磁盘水位、不依赖调度器/真实仿真：用临时目录 + 假时钟 + 假 disk_usage
+ 注入 status/protected provider。

Run: python tests/test_workspace_cleanup.py
"""

import os
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

try:
    from mcp_server.services import workspace_cleaner as wc
except ImportError as exc:  # pragma: no cover - 环境缺依赖时按项目惯例 SKIP
    print(f"  [SKIP] 未安装 mcp_server 依赖（{exc}）—— 请先执行:")
    print("         pip install -r mcp_server/requirements.txt")
    sys.exit(0)

NOW = 1_700_000_000.0
BASE = REPO / ".tmp" / "workspace_cleanup_test"
_failures: list[str] = []


def check(label: str, cond: bool, detail: str = "") -> None:
    print(("  [PASS] " if cond else "  [FAIL] ") + label + ("" if cond else f"   {detail}"))
    if not cond:
        _failures.append(label)


def fresh_root(name: str) -> Path:
    root = BASE / name
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    return root


def make_task(
    root: Path,
    task_id: str,
    age_days: float,
    *,
    now: float = NOW,
    with_results: bool = True,
    with_log: bool = True,
    result_files: int = 2,
    result_bytes: int = 1024,
) -> Path:
    """造一个任务目录，并把「最后活动时间」设为 now - age_days 天。"""
    td = root / task_id
    td.mkdir(parents=True, exist_ok=True)
    touched: list[Path] = [td, td / "task_meta.json"]
    (td / "task_meta.json").write_text("{}", encoding="utf-8")
    if with_results:
        mocked = td / "results" / "mocked_workload"
        mocked.mkdir(parents=True, exist_ok=True)
        touched.append(td / "results")
        for i in range(result_files):
            f = mocked / f"model_rank{i}_time.csv"
            f.write_bytes(b"x" * result_bytes)
            touched.append(f)
    if with_log:
        log = td / "simulation.log"
        log.write_bytes(b"l" * 512)
        touched.append(log)
    ts = now - age_days * 86400.0
    # 先设文件、后设目录（目录 mtime 不随子文件 utime 改变，但顺序上先深后浅更稳）
    for p in sorted(touched, key=lambda x: len(x.parts), reverse=True):
        os.utime(p, (ts, ts))
    return td


def make_cleaner(
    root: Path,
    *,
    pct: float = 50.0,
    statuses: dict | None = None,
    protected: set | None = None,
    dry_run: bool = False,
    now: float = NOW,
    **kwargs,
):
    statuses = statuses or {}
    protected = protected or set()

    def _disk_usage(_path):
        total = 1000.0
        used = total * pct / 100.0
        return int(total), int(used), int(total - used)

    defaults = dict(
        workspace_root=root,
        result_retention_days=7,
        log_retention_days=14,
        min_keep_tasks=0,
        disk_watermark_pct=80,
        disk_critical_pct=90,
        emergency_retention_days=2,
        emergency_min_keep_tasks=5,
        dry_run=dry_run,
        now=lambda: now,
        disk_usage=_disk_usage,
        protected_ids_provider=lambda: set(protected),
        status_provider=lambda tid: statuses.get(tid),
    )
    defaults.update(kwargs)
    return wc.WorkspaceCleaner(**defaults)


# ──────────────────────────────────────────────────────────────
# 场景 1：正常水位 + 两阶段删除
# ──────────────────────────────────────────────────────────────
def case_normal_two_phase() -> None:
    print("[1] 正常水位：先删 results/，超日志保留期才删整个目录")
    root = fresh_root("normal")
    make_task(root, "t_recent", 1)
    make_task(root, "t_mid", 10)
    make_task(root, "t_old", 20)
    make_task(root, "t_failed", 10, with_results=False)
    statuses = {t: "completed" for t in ("t_recent", "t_mid", "t_old", "t_failed")}
    cleaner = make_cleaner(root, pct=50, statuses=statuses, min_keep_tasks=1)
    stats = cleaner.run_once()

    check("mode=normal", stats.mode == wc.MODE_NORMAL, stats.mode)
    check("最近 1 个任务被 min_keep 保护", stats.kept_min_keep == 1, str(stats))
    check("10 天的任务只删 results/", stats.deleted_results == 1, str(stats))
    check("t_mid 目录仍在但 results/ 已删",
          (root / "t_mid").is_dir() and not (root / "t_mid" / "results").exists(),
          str(list((root / "t_mid").glob("*"))))
    check("20 天的任务整个目录被删", not (root / "t_old").exists())
    check("results 缺失的旧任务只保留日志（不提前删目录）",
          (root / "t_failed").is_dir() and stats.kept_log_only >= 1,
          str(stats))
    check("释放字节数 > 0", stats.freed_bytes > 0, str(stats.freed_bytes))


# ──────────────────────────────────────────────────────────────
# 场景 2：min_keep_tasks 与 retention 冲突 —— min_keep 赢
# ──────────────────────────────────────────────────────────────
def case_min_keep_wins() -> None:
    print("[2] 冲突场景：最近 20 个任务必留（min_keep 优先于 7 天保留期）")
    root = fresh_root("min_keep")
    # 25 个已完成任务，年龄 10~34 天，全部超过 7 天保留期
    ids = [f"t_{i:02d}" for i in range(25)]
    for i, tid in enumerate(ids):
        make_task(root, tid, 10 + i)
    statuses = {tid: "completed" for tid in ids}
    cleaner = make_cleaner(root, pct=50, statuses=statuses, min_keep_tasks=20)
    stats = cleaner.run_once()

    remaining = sorted(p.name for p in root.iterdir() if p.is_dir())
    check("保留 20 个（最近 20 个）", len(remaining) == 20, f"{len(remaining)}: {remaining[:5]}...")
    check("删掉最老的 5 个", stats.deleted_dirs == 5, str(stats))
    check("最老的 5 个确实被删", all(not (root / ids[i]).exists() for i in range(20, 25)))
    check("最近 20 个都还在", all((root / ids[i]).exists() for i in range(20)))
    check("min_keep 统计=20", stats.kept_min_keep == 20, str(stats))


# ──────────────────────────────────────────────────────────────
# 场景 3：运行中/排队中任务永不删
# ──────────────────────────────────────────────────────────────
def case_active_never_deleted() -> None:
    print("[3] 运行中/排队中任务永不删（即使超龄 100 天）")
    root = fresh_root("active")
    make_task(root, "t_protected", 100)
    make_task(root, "t_running", 100)
    make_task(root, "t_done", 100)
    cleaner = make_cleaner(
        root,
        pct=50,
        statuses={"t_running": "running", "t_done": "completed"},
        protected={"t_protected"},
        min_keep_tasks=0,
    )
    stats = cleaner.run_once()

    check("protected 任务保留", (root / "t_protected").is_dir())
    check("running 任务保留", (root / "t_running").is_dir())
    check("active_skipped=2", stats.active_skipped == 2, str(stats))
    check("其余超龄任务被删", not (root / "t_done").exists())


# ──────────────────────────────────────────────────────────────
# 场景 4：磁盘应急水位（>=80%）
# ──────────────────────────────────────────────────────────────
def case_emergency_watermark() -> None:
    print("[4] 磁盘 85%：应急模式（保留期 2 天、必留 5 个）")
    root = fresh_root("emergency")
    ages = [1.5, 3, 5, 8, 10, 12]
    ids = [f"e_{i}" for i in range(len(ages))]
    for tid, age in zip(ids, ages):
        make_task(root, tid, age)
    cleaner = make_cleaner(
        root,
        pct=85,
        statuses={tid: "completed" for tid in ids},
        min_keep_tasks=20,  # 正常配置很大，但在应急模式应被忽略
        emergency_min_keep_tasks=5,
        emergency_retention_days=2,
    )
    stats = cleaner.run_once()

    check("mode=emergency", stats.mode == wc.MODE_EMERGENCY, stats.mode)
    check("最近 5 个被应急 min_keep 保护", stats.kept_min_keep == 5, str(stats))
    check("最老的 1 个被删", stats.deleted_dirs == 1 and not (root / ids[-1]).exists(), str(stats))
    check("1.5 天的任务保留（保留期内）", (root / ids[0]).exists())


# ──────────────────────────────────────────────────────────────
# 场景 5：磁盘临界水位（>=90%）
# ──────────────────────────────────────────────────────────────
def case_critical_watermark() -> None:
    print("[5] 磁盘 95%：临界模式（只保留最新 1 个任务）")
    root = fresh_root("critical")
    ages = [0.5, 2, 5, 8]
    ids = [f"c_{i}" for i in range(len(ages))]
    for tid, age in zip(ids, ages):
        make_task(root, tid, age)
    cleaner = make_cleaner(root, pct=95, statuses={tid: "completed" for tid in ids})
    stats = cleaner.run_once()

    check("mode=critical", stats.mode == wc.MODE_CRITICAL, stats.mode)
    check("最新 1 个保留", (root / ids[0]).exists())
    check("其余 3 个被删", stats.deleted_dirs == 3, str(stats))
    check("被删的确实不在了", all(not (root / t).exists() for t in ids[1:]))


# ──────────────────────────────────────────────────────────────
# 场景 6：dry_run 只统计不删除
# ──────────────────────────────────────────────────────────────
def case_dry_run() -> None:
    print("[6] dry_run：只统计「本来会删」，不落盘")
    root = fresh_root("dry_run")
    make_task(root, "d_old", 30)
    cleaner = make_cleaner(root, pct=50, statuses={"d_old": "completed"}, dry_run=True)
    stats = cleaner.run_once()

    check("dry_run=True", stats.dry_run is True)
    check("统计到 1 个待删目录", stats.deleted_dirs == 1, str(stats))
    check("目录实际仍存在", (root / "d_old").is_dir())
    check("统计释放字节 > 0", stats.freed_bytes > 0)


# ──────────────────────────────────────────────────────────────
# 场景 7：未知目录/非任务目录不动；submitted 老任务可清
# ──────────────────────────────────────────────────────────────
def case_unknown_dirs() -> None:
    print("[7] 未知目录/非任务目录不动；超龄 submitted 任务可清")
    root = fresh_root("unknown")
    (root / "random_dir").mkdir()
    (root / "random_dir" / "keep.txt").write_text("keep", encoding="utf-8")
    (root / "not_a_dir.txt").write_text("x", encoding="utf-8")
    make_task(root, "t_submitted_old", 40)
    cleaner = make_cleaner(
        root,
        pct=50,
        statuses={"t_submitted_old": "submitted"},
        log_retention_days=14,
    )
    stats = cleaner.run_once()

    check("非任务目录（无 task_meta）被跳过且保留",
          (root / "random_dir").is_dir() and stats.skipped_unknown >= 1, str(stats))
    check("根目录普通文件不处理", (root / "not_a_dir.txt").is_file())
    check("超龄 submitted 任务被清", stats.deleted_dirs == 1 and not (root / "t_submitted_old").exists(),
          str(stats))


# ──────────────────────────────────────────────────────────────
# 场景 8：CleanupWorker 启动即扫一次 / 禁用不启动
# ──────────────────────────────────────────────────────────────
def case_worker() -> None:
    print("[8] CleanupWorker：启动后先扫一次；enabled=False 不启动")
    root = fresh_root("worker")
    make_task(root, "w_old", 30, now=time.time())
    cleaner = make_cleaner(
        root,
        pct=50,
        statuses={"w_old": "completed"},
        now=time.time(),
        disk_usage=lambda _p: (100, 50, 50),
    )
    worker = wc.CleanupWorker(
        enabled=True,
        interval_sec=60,
        initial_delay_sec=0.05,
        cleaner_factory=lambda: cleaner,
    )
    worker.start()
    deadline = time.time() + 5
    while time.time() < deadline and (root / "w_old").exists():
        time.sleep(0.05)
    running = worker.is_running
    worker.stop()
    check("worker 启动并完成一轮清理", not (root / "w_old").exists())
    check("stop 后线程退出", running is True and worker.is_running is False)

    disabled = wc.CleanupWorker(enabled=False, cleaner_factory=lambda: cleaner)
    disabled.start()
    check("enabled=False 时不启动", disabled.is_running is False)
    disabled.stop()


# ──────────────────────────────────────────────────────────────
# 场景 9：workspace 不存在时安全跳过
# ──────────────────────────────────────────────────────────────
def case_missing_root() -> None:
    print("[9] workspace 不存在：安全跳过、不抛异常")
    cleaner = make_cleaner(BASE / "does_not_exist")
    stats = cleaner.run_once()
    check("scanned=0 且无错误", stats.scanned == 0 and stats.errors == 0, str(stats))


def main() -> int:
    print("== workspace 磁盘清理（优先级 / 两阶段 / 应急 / 安全） ==")
    case_normal_two_phase()
    case_min_keep_wins()
    case_active_never_deleted()
    case_emergency_watermark()
    case_critical_watermark()
    case_dry_run()
    case_unknown_dirs()
    case_worker()
    case_missing_root()
    print()
    if _failures:
        print(f"  [FAIL] {len(_failures)} 项未通过: {_failures}")
        return 1
    print("  [PASS] workspace 清理优先级符合预期：运行中/排队 > 水位应急 > min_keep > 保留期。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
