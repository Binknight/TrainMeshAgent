"""实测 README 里记录的 Flask 启动命令，确保文档与代码一致。

本脚本固化两条 README 结论（避免文档随代码漂移）：
  1. `python -m app.main`      —— 唯一可用的启动形式，必须 PASS
  2. `python app/main.py`      —— 直接跑脚本会 ModuleNotFoundError: No module named 'app'，
                                  因此 README 只把它列为"不可用"并说明原因，此处断言其为 FAIL

第 2 条是实测发现的结果，不是猜测：Python 执行脚本时把**脚本所在目录**（app/）
放入 sys.path，而不是 CWD（仓库根），故 `from app.config import ...` 找不到包。

刻意用非默认端口，避免与已运行的服务冲突；不依赖 aicm/ 仿真工具。
输出重定向到文件而非 PIPE：受限沙箱禁止命名管道。
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
PORT = 5199
BASE = f"http://127.0.0.1:{PORT}"


def probe(health_path: str, timeout: float = 3.0):
    with urllib.request.urlopen(f"{BASE}{health_path}", timeout=timeout) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


def run_case(label: str, argv: list[str], health_path: str) -> bool:
    env = dict(os.environ)
    env["FLASK_PORT"] = str(PORT)
    env["FLASK_HOST"] = "127.0.0.1"

    # 标签含 ' '、'/'、'.' 等字符，需彻底净化为安全文件名（否则 '/' 会被当作路径分隔符）
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in label)
    log = pathlib.Path(tempfile.gettempdir()) / f"flask_probe_{safe}.log"
    with open(log, "w", encoding="utf-8") as fh:
        proc = subprocess.Popen(argv, env=env, cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT)

    print(f"\n[{label}] 命令: {' '.join(argv)}   pid={proc.pid}  日志 {log.name}")
    ok = False
    try:
        for _ in range(60):
            if proc.poll() is not None:
                print(f"[{label}] 进程过早退出 exit={proc.returncode}")
                break
            try:
                status, body = probe(health_path)
                print(f"[{label}] GET {health_path} -> {status} {body}")
                ok = status == 200
                if ok:
                    break
            except (urllib.error.URLError, OSError, json.JSONDecodeError):
                time.sleep(0.5)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)

    text = log.read_text(encoding="utf-8", errors="replace")
    print(f"[{label}] 启动日志关键行:")
    for line in text.splitlines():
        if any(k in line for k in ("migration", "Starting TrainMesh", "Running on", "Traceback", "Error")):
            print(f"    {line.strip()[:150]}")
    if not ok:
        print(f"[{label}] 完整日志尾部:")
        for line in text.splitlines()[-15:]:
            print(f"    {line[:150]}")

    verdict = "PASS" if ok else "FAIL"
    print(f"[{label}] {verdict}")
    return ok


def main() -> int:
    # (标签, argv, 期望能否启动)
    cases = [
        ("python -m app.main", [sys.executable, "-m", "app.main"], True),
        ("python app/main.py", [sys.executable, "app/main.py"], False),
    ]
    results = []
    for label, argv, _expected in cases:
        try:
            results.append(run_case(label, argv, "/api/health"))
        except Exception as exc:  # noqa: BLE001 —— 单个用例异常不应中断其余探测
            print(f"\n[{label}] 探测本身异常: {type(exc).__name__}: {exc}")
            results.append(False)

    print("\n=== 汇总 ===")
    all_ok = True
    for (label, _argv, expected), ok in zip(cases, results):
        match = ok == expected
        all_ok = all_ok and match
        expect_txt = "可启动" if expected else "应失败"
        got_txt = "可启动" if ok else "失败"
        print(f"  {label:<22} 期望={expect_txt:<8} 实测={got_txt:<8} {'OK' if match else '不符合预期'}")

    if all_ok:
        print("OK: README 关于 Flask 启动方式的说明与实际行为一致")
    else:
        print("FAIL: 实际行为与 README 说明不符，需修正 README")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
