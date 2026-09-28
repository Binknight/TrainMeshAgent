#!/usr/bin/env python3
"""停机前对 SQLite 做一次 WAL checkpoint（TRUNCATE）。

为什么需要它：
    WAL 模式下已提交的数据可能还留在 `<db>-wal` 里。TRUNCATE checkpoint 会把它们
    并回主库并清空伴生文件，因此：
      * 停机后 `cp <db> <backup>` 得到的是**完整**快照（否则要连着 -wal 一起拷）；
      * 卷占用回归单文件大小，排查「磁盘为什么没释放」时不会被 -wal 误导。
    合并本身不需要这一步（下次打开会自动恢复），这里纯粹是让停机后的状态干净。

容错：SQLite 后端才做；PG 逃生门、文件不存在、任何异常都只打印并返回 0 ——
      checkpoint 失败绝不阻断容器退出流程。
"""

from __future__ import annotations

import os
import sqlite3
import sys

DEFAULT_SQLITE_PATH = "/home/aicm/db/train_mesh_agent.db"


def log(message: str) -> None:
    print(f"[checkpoint] {message}", flush=True)


def main() -> int:
    url = os.getenv("DATABASE_URL", "").strip().lower()
    if url.startswith(("postgres://", "postgresql://")):
        log("后端=postgres（外部 PG），无需 checkpoint")
        return 0

    path = os.getenv("SQLITE_PATH") or DEFAULT_SQLITE_PATH
    if not os.path.exists(path):
        log(f"数据文件不存在（{path}），无需 checkpoint")
        return 0

    try:
        conn = sqlite3.connect(path, timeout=10)
        try:
            mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchall()
            size = os.path.getsize(path)
            log(f"checkpoint 完成（journal_mode={mode}，主库 {size} 字节）")
        finally:
            conn.close()
    except Exception as exc:  # noqa: BLE001 —— 绝不因清理动作阻断退出
        log(f"警告：checkpoint 失败：{exc}（数据不会丢失，下次启动会自动合并 WAL）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
