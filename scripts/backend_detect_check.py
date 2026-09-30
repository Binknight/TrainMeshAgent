"""DATABASE_URL 前缀侦测的判定矩阵验证。

为什么值得单独验：这条规则是**静默**的 —— DSN 写错前缀不会报错，只会悄悄连到
SQLite，表现为"数据丢了/看不到历史"。这类错误没有异常可抓，只能靠把边界情形
逐条钉住。配套文档见 docs/使用指南.md §3.2。
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _console import ensure_utf8_console  # noqa: E402

ensure_utf8_console()

from app import dbapi  # noqa: E402

CASES = [
    ("", "sqlite", "留空 = 默认 SQLite"),
    ("postgresql://u:p@h:5432/db", "postgres", "标准 postgresql:// 前缀"),
    ("postgres://u:p@h:5432/db", "postgres", "postgres:// 短前缀（Heroku 风格）"),
    ("POSTGRESQL://u:p@h/db", "postgres", "大写前缀"),
    ("  postgresql://u:p@h/db  ", "postgres", "带首尾空白"),
    ("postgres:/u:p@h/db", "sqlite", "写坏的前缀（单斜杠）"),
    ("postgres@Data", "sqlite", "DSN 简写 → 静默回落 SQLite"),
    ("mysql://u:p@h/db", "sqlite", "别的协议 → 静默回落 SQLite"),
    ("/home/data/db/x.db", "sqlite", "误把文件路径写进 DATABASE_URL"),
    ("sqlite:///home/data/db/x.db", "sqlite", "显式 sqlite:// 也归 SQLite"),
]


def main() -> int:
    bad = 0
    for dsn, want, note in CASES:
        got = dbapi.detect_backend(dsn)
        ok = got == want
        bad += not ok
        print(f"{'OK  ' if ok else 'FAIL'}: {note:34s} {dsn!r:38s} -> {got}")
        if not ok:
            print(f"      期望 {want!r}")
    print()
    if bad:
        print(f"FAIL: {bad} 条判定不符合预期")
        return 1
    print(f"OK: {len(CASES)} 条前缀判定全部符合预期")
    return 0


if __name__ == "__main__":
    sys.exit(main())
