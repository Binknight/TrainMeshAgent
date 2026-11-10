"""仿真流量文件下载：MCP REST + Flask 端到端回归测试。

覆盖（真实 HTTP，非 mock）：
  MCP Server（临时 workspace + 临时端口）
    1. /tasks/<id>/files 列举（文件/分类/状态/total）
    2. /tasks/<id>/files/download 单文件，Content-Disposition 为附件
    3. /tasks/<id>/files/bundle 按分类打包 ZIP
    4. 路径穿越（../、绝对路径、反斜杠）被拒绝；任务不存在 404
  Flask（真实把 MCP 作为上游）
    5. GET  /api/session/<sid>/simulation-files（两侧文件分组与状态）
    6. GET  .../download?side=original&path=...（流式代理，内容一致）
    7. GET  .../bundle?side=original / category / both（ZIP 内容正确）
    8. 无 task_id 的一侧 → available=false / 下载 404；非法 side/category → 400
    9. MCP 不可达 → 列表降级为 unavailable、下载 503（不会 500）
   10. 不同会话不能下到别人的任务文件（side 绑定会话 task_id）

不依赖 aicm 真实仿真：只手工在临时 workspace 里放置产物文件。

Run: python tests/test_simulation_files_api.py
"""

from __future__ import annotations

import io
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
TEST_DIR = REPO / ".tmp" / "sim_files_api_test"
WS = TEST_DIR / "ws"
MCP_PORT = 8797
MCP_BASE = f"http://127.0.0.1:{MCP_PORT}"

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_failures: list[str] = []


def check(label: str, cond: bool, detail: str = "") -> None:
    print(("  [PASS] " if cond else "  [FAIL] ") + label + ("" if cond else f"   {detail}"))
    if not cond:
        _failures.append(label)


def port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        return s.connect_ex((host, port)) == 0


def wait_health(timeout: float = 30) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(MCP_BASE + "/health", timeout=2) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(0.25)
    return False


def write_workspace_task(task_id: str, marker: str) -> None:
    """在临时 workspace 里造一个已完成任务的产物目录。"""
    task_dir = WS / task_id
    (task_dir / "results" / "mocked_workload").mkdir(parents=True)
    (task_dir / "results" / "statistic_data").mkdir(parents=True)
    (task_dir / "results" / "mocked_workload" / f"{marker}_rank0_workload.csv").write_text(
        f"operator,bytes\n{marker}_op,1024\n", encoding="utf-8"
    )
    (task_dir / "results" / "mocked_workload" / f"{marker}_rank0_time.csv").write_text(
        f"operator,time\n{marker}_op,1.5\n", encoding="utf-8"
    )
    (task_dir / "results" / "statistic_data" / "rank0.txt").write_text(
        "flops: 1.0e12\nhbm: 1.0e9\n", encoding="utf-8"
    )
    (task_dir / "simulation.log").write_text(
        f"[{marker}] simulation done\n", encoding="utf-8"
    )
    (task_dir / "topology_generated.sh").write_text(
        f"#!/bin/bash\necho {marker}\n", encoding="utf-8"
    )
    (task_dir / "task_state.json").write_text(
        json.dumps({"status": "completed", "dry_run": True}), encoding="utf-8"
    )


def mcp_get_json(path: str) -> tuple[int, dict]:
    try:
        with urllib.request.urlopen(MCP_BASE + path, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            body = json.loads(exc.read().decode("utf-8"))
        except Exception:
            body = {}
        return exc.code, body


def mcp_get_bytes(path: str) -> tuple[int, bytes, dict]:
    try:
        with urllib.request.urlopen(MCP_BASE + path, timeout=30) as resp:
            return resp.status, resp.read(), dict(resp.headers)
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(), dict(exc.headers)


def main() -> int:
    shutil.rmtree(TEST_DIR, ignore_errors=True)
    WS.mkdir(parents=True)
    os.environ["SQLITE_PATH"] = str(TEST_DIR / "app.db")
    os.environ["DATABASE_URL"] = ""

    orig_id = "sim_it_orig_0001"
    eq_id = "sim_it_eq_0001"
    write_workspace_task(orig_id, "orig")
    write_workspace_task(eq_id, "eq")

    env = dict(os.environ)
    env.update(
        {
            "AICM_MCP_PORT": str(MCP_PORT),
            "AICM_MCP_DRY_RUN": "true",
            "AICM_MCP_WORKSPACE_ROOT": str(WS),
            "AICM_MCP_WORKSPACE_CLEANUP_ENABLED": "false",
            "AICM_MCP_MAX_CONCURRENT_TASKS": "2",
        }
    )
    server_log = open(TEST_DIR / "mcp_server.log", "w", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, "-m", "mcp_server"],
        cwd=str(REPO),
        env=env,
        stdout=server_log,
        stderr=subprocess.STDOUT,
    )

    from app.main import create_app
    from app.agent.session import session_manager
    from app.mcp.client import mcp_client

    app = create_app()
    app.testing = True
    client = app.test_client()

    try:
        print("== 仿真流量文件下载端到端 ==")
        check("MCP Server 启动并响应 /health", wait_health(), "server not up")
        if _failures:
            server_log.flush()
            print((TEST_DIR / "mcp_server.log").read_text(encoding="utf-8", errors="replace")[-3000:])
            return 1

        mcp_client.server_url = MCP_BASE

        # ── 1. MCP REST 列举 ──
        print("\n[1] MCP REST：文件列举")
        status, listing = mcp_get_json(f"/tasks/{orig_id}/files")
        check("列举返回 200", status == 200, str(status))
        check("任务状态 completed", listing.get("status") == "completed", str(listing.get("status")))
        categories = {g["category"]: g for g in listing.get("groups", [])}
        check(
            "分类覆盖 流量/时间线/统计/脚本/日志/其他",
            {"traffic_detail", "operator_timeline", "statistic_summary", "script", "logs", "other"}
            <= set(categories),
            str(sorted(categories)),
        )
        check("total_files=6", listing.get("total_files") == 6, str(listing.get("total_files")))
        check(
            "流量明细为 1 个 CSV",
            categories.get("traffic_detail", {}).get("count") == 1
            and categories.get("traffic_detail", {}).get("formats") == ["csv"],
            str(categories.get("traffic_detail")),
        )

        # ── 2. MCP REST 单文件 + 路径穿越 ──
        print("\n[2] MCP REST：单文件下载与路径穿越防护")
        status, data, headers = mcp_get_bytes(
            f"/tasks/{orig_id}/files/download?path=results/mocked_workload/orig_rank0_workload.csv"
        )
        check("单文件返回 200 且内容正确", status == 200 and b"orig_op,1024" in data, f"HTTP {status}")
        check(
            "Content-Disposition 为附件",
            "attachment" in headers.get("content-disposition", "").lower(),
            str(headers.get("content-disposition")),
        )
        for bad_path in ["../task_meta.json", "/etc/passwd", "results/../../x", "results\\..\\x"]:
            status, _body = mcp_get_json(f"/tasks/{orig_id}/files/download?path={bad_path}")
            check(f"路径穿越被拒绝: {bad_path!r}", status == 400, f"HTTP {status}")
        status, _body = mcp_get_json("/tasks/sim_missing_0001/files")
        check("不存在的任务返回 404", status == 404, f"HTTP {status}")

        # ── 3. MCP REST 分类打包 ──
        print("\n[3] MCP REST：分类打包 ZIP")
        status, zip_bytes, _ = mcp_get_bytes(
            f"/tasks/{orig_id}/files/bundle?category=traffic_detail"
        )
        names = []
        if status == 200:
            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
                names = zf.namelist()
        check(
            "分类 ZIP 只含流量明细",
            status == 200 and names == ["results/mocked_workload/orig_rank0_workload.csv"],
            f"HTTP {status} names={names}",
        )

        # ── 4. Flask：列表 ──
        print("\n[4] Flask：GET simulation-files 列表")
        create_resp = client.post("/api/session")
        session_id = (create_resp.get_json() or {}).get("session_id") or ""
        check("创建会话", bool(session_id), str(create_resp.status_code))
        session = session_manager.get_session(session_id)
        session.original_task_id = orig_id
        session.equivalent_task_id = eq_id
        session.step = "completed"

        resp = client.get(f"/api/session/{session_id}/simulation-files")
        body = resp.get_json() or {}
        sides = body.get("sides") or {}
        check("列表返回 200", resp.status_code == 200, f"HTTP {resp.status_code}")
        check(
            "两侧均可用且状态 completed",
            sides.get("original", {}).get("available") is True
            and sides.get("equivalent", {}).get("available") is True
            and sides.get("original", {}).get("status_label") == "已完成",
            json.dumps({k: {kk: v.get(kk) for kk in ("available", "status_label")} for k, v in sides.items()}, ensure_ascii=False),
        )
        check(
            "两侧 task_id 回传正确",
            sides.get("original", {}).get("task_id") == orig_id
            and sides.get("equivalent", {}).get("task_id") == eq_id,
            str([sides.get("original", {}).get("task_id"), sides.get("equivalent", {}).get("task_id")]),
        )
        check(
            "分类分组可用（groups 非空）",
            len(sides.get("original", {}).get("groups") or []) >= 5,
            str(sides.get("original", {}).get("groups")),
        )

        # ── 5. Flask：单文件 ──
        print("\n[5] Flask：单文件流式下载")
        resp = client.get(
            f"/api/session/{session_id}/simulation-files/download",
            query_string={"side": "original", "path": "results/mocked_workload/orig_rank0_workload.csv"},
        )
        check(
            "单文件返回 200 且内容正确",
            resp.status_code == 200 and b"orig_op,1024" in resp.data,
            f"HTTP {resp.status_code}",
        )
        check(
            "单文件 Content-Disposition 附件",
            "attachment" in (resp.headers.get("Content-Disposition") or "").lower(),
            str(resp.headers.get("Content-Disposition")),
        )
        resp = client.get(
            f"/api/session/{session_id}/simulation-files/download",
            query_string={"side": "original", "path": "../task_meta.json"},
        )
        check("Flask 路径穿越被拒绝", resp.status_code in (400, 404), f"HTTP {resp.status_code}")
        resp = client.get(
            f"/api/session/{session_id}/simulation-files/download",
            query_string={"side": "equivalent", "path": "simulation.log"},
        )
        check("另一侧文件可下载", resp.status_code == 200 and b"[eq]" in resp.data, f"HTTP {resp.status_code}")

        # ── 6. Flask：ZIP 打包 ──
        print("\n[6] Flask：ZIP 打包（单侧 / 分类 / 双侧合并）")
        resp = client.get(
            f"/api/session/{session_id}/simulation-files/bundle", query_string={"side": "original"}
        )
        names = []
        if resp.status_code == 200:
            with zipfile.ZipFile(io.BytesIO(resp.data)) as zf:
                names = zf.namelist()
        check(
            "单侧 ZIP 含全部文件",
            resp.status_code == 200 and "results/mocked_workload/orig_rank0_time.csv" in names,
            f"HTTP {resp.status_code} names={names}",
        )
        resp = client.get(
            f"/api/session/{session_id}/simulation-files/bundle",
            query_string={"side": "original", "category": "logs"},
        )
        names = []
        if resp.status_code == 200:
            with zipfile.ZipFile(io.BytesIO(resp.data)) as zf:
                names = zf.namelist()
        check(
            "分类 ZIP 只含日志",
            resp.status_code == 200 and names == ["simulation.log"],
            f"HTTP {resp.status_code} names={names}",
        )
        resp = client.get(
            f"/api/session/{session_id}/simulation-files/bundle", query_string={"side": "both"}
        )
        names = []
        if resp.status_code == 200:
            with zipfile.ZipFile(io.BytesIO(resp.data)) as zf:
                names = zf.namelist()
        check(
            "双侧合并 ZIP 按两侧分目录",
            resp.status_code == 200
            and any(n.startswith("原始组网/") for n in names)
            and any(n.startswith("等效组网/") for n in names),
            f"HTTP {resp.status_code} names={names[:5]}",
        )
        check(
            "双侧 ZIP 文件名合规",
            "attachment" in (resp.headers.get("Content-Disposition") or "").lower(),
            str(resp.headers.get("Content-Disposition")),
        )

        # ── 7. 非法参数 / 无 task_id 的一侧 ──
        print("\n[7] 参数校验与缺失任务")
        resp = client.get(
            f"/api/session/{session_id}/simulation-files/download",
            query_string={"side": "nope", "path": "simulation.log"},
        )
        check("非法 side → 400", resp.status_code == 400, f"HTTP {resp.status_code}")
        resp = client.get(
            f"/api/session/{session_id}/simulation-files/bundle",
            query_string={"side": "original", "category": "nope"},
        )
        check("非法 category → 400", resp.status_code == 400, f"HTTP {resp.status_code}")
        resp = client.post("/api/session")
        empty_sid = (resp.get_json() or {}).get("session_id") or ""
        resp = client.get(f"/api/session/{empty_sid}/simulation-files")
        empty_body = resp.get_json() or {}
        check(
            "无 task_id 的会话两侧 available=false",
            resp.status_code == 200
            and all(
                empty_body.get("sides", {}).get(side, {}).get("available") is False
                for side in ("original", "equivalent")
            ),
            json.dumps(empty_body.get("sides", {}), ensure_ascii=False)[:300],
        )
        resp = client.get(
            f"/api/session/{empty_sid}/simulation-files/bundle", query_string={"side": "both"}
        )
        check("无任务双侧打包 → 404", resp.status_code == 404, f"HTTP {resp.status_code}")

        # ── 8. MCP 不可达降级 ──
        print("\n[8] MCP 不可达：列表降级 / 下载 503（不是 500）")
        real_url = mcp_client.server_url
        try:
            mcp_client.server_url = "http://127.0.0.1:1"
            resp = client.get(f"/api/session/{session_id}/simulation-files")
            degraded = resp.get_json() or {}
            check(
                "列表降级为 unavailable 且仍 200",
                resp.status_code == 200
                and degraded.get("sides", {}).get("original", {}).get("available") is False
                and degraded.get("sides", {}).get("original", {}).get("status") == "unavailable",
                f"HTTP {resp.status_code} {json.dumps(degraded.get('sides', {}), ensure_ascii=False)[:300]}",
            )
            resp = client.get(
                f"/api/session/{session_id}/simulation-files/download",
                query_string={"side": "original", "path": "simulation.log"},
            )
            check("下载在 MCP 不可达时返回 503", resp.status_code == 503, f"HTTP {resp.status_code}")
            resp = client.get(
                f"/api/session/{session_id}/simulation-files/bundle", query_string={"side": "original"}
            )
            check("打包在 MCP 不可达时返回 503", resp.status_code == 503, f"HTTP {resp.status_code}")
        finally:
            mcp_client.server_url = real_url

        # ── 9. 会话隔离：另一个会话拿不到本任务文件 ──
        print("\n[9] 会话隔离")
        other_sid = (client.post("/api/session").get_json() or {}).get("session_id") or ""
        resp = client.get(
            f"/api/session/{other_sid}/simulation-files/download",
            query_string={"side": "original", "path": "results/mocked_workload/orig_rank0_workload.csv"},
        )
        check("新会话无 task_id → 无法下载他人任务文件", resp.status_code == 404, f"HTTP {resp.status_code}")
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=10)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        server_log.close()

    print()
    if _failures:
        print(f"  [FAIL] {len(_failures)} 项未通过: {_failures}")
        return 1
    print("  [PASS] 仿真流量文件下载端到端通过：MCP REST + Flask 代理 + ZIP 打包 + 安全校验。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
