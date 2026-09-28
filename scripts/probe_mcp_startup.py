"""实测 MCP 仿真 Server 能否启动并响应（README 命令的可复现验证）。

启动 python -m mcp_server，探测 /health 与 /tools，然后干净关闭。

刻意不使用 subprocess.PIPE：受限沙箱禁止创建命名管道，管道式 stdio 会直接
EPERM。改为把子进程输出重定向到文件，既避开该限制，又能保留日志用于诊断。

不依赖 aicm/ 仿真工具：sim_tool_home 只在真正下发任务时才校验，服务本身能起来。
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

PORT = 9123  # 刻意避开默认 9000，防止与已运行的服务冲突
BASE = f"http://127.0.0.1:{PORT}"


def get(path: str, timeout: float = 3.0):
    with urllib.request.urlopen(f"{BASE}{path}", timeout=timeout) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


def main() -> int:
    env = dict(os.environ)
    env["AICM_MCP_PORT"] = str(PORT)
    env["AICM_MCP_HOST"] = "127.0.0.1"

    log_path = pathlib.Path(tempfile.gettempdir()) / "mcp_probe_server.log"
    log_file = open(log_path, "w", encoding="utf-8")

    proc = subprocess.Popen(
        [sys.executable, "-m", "mcp_server"],
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        cwd=str(pathlib.Path(__file__).resolve().parent.parent),
    )
    print(f"[mcp-probe] 已启动 pid={proc.pid}，端口 {PORT}，日志 {log_path}")

    ok = False
    try:
        for _ in range(30):
            if proc.poll() is not None:
                print(f"[mcp-probe] 进程过早退出，exit={proc.returncode}")
                break
            try:
                status, body = get("/health")
                print(f"[mcp-probe] /health -> {status} {body}")
                if status == 200 and body.get("status") == "ok":
                    ok = True
                    break
            except (urllib.error.URLError, OSError, json.JSONDecodeError):
                time.sleep(0.5)

        if ok:
            status, body = get("/tools")
            names = [t.get("name") for t in body.get("tools", [])]
            print(f"[mcp-probe] /tools -> {status}，注册 {len(names)} 个工具")
            print(f"[mcp-probe] 工具清单: {names}")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        log_file.close()
        print(f"[mcp-probe] 已停止，exit={proc.returncode}")

    if not ok and log_path.is_file():
        print("[mcp-probe] 服务端日志:")
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-25:]:
            print(f"    {line}")

    print("OK: MCP Server 可正常启动与响应" if ok else "FAIL: MCP Server 未能正常响应")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
