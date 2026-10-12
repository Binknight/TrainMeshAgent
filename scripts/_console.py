"""给 scripts/ 下的校验脚本统一加 UTF-8 控制台兜底。

为什么需要：这些脚本的输出是中文，而 Windows 控制台默认 cp1252 —— 一条
`print("...中文...")` 就会抛 UnicodeEncodeError，把「校验通过/失败」伪装成
编码崩溃，很容易误导排查方向（本仓库在改造过程中就踩到过一次）。
容器内 LANG=C.UTF-8 不受影响，这里纯粹是兜住本地开发。

用法（必须在任何 print 之前调用）：

    from _console import ensure_utf8_console
    ensure_utf8_console()

注意：脚本以 `python scripts/xxx.py` 运行时，`scripts/` 自身在 sys.path[0]，
因此 `import _console` 直接可用；被 tests/ 或其它入口 import 时不受影响。
"""

from __future__ import annotations

import sys

_done = False


def ensure_utf8_console() -> None:
    global _done
    if _done:
        return
    _done = True
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001 非 tty / 被重定向 / 旧解释器
            pass
