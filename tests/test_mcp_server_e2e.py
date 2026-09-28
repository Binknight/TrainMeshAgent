"""MCP Server end-to-end test: boot it and call it over real HTTP.

Boots uvicorn via `python -m mcp_server` in dry-run mode (the aicm simulation
tool is not needed), seeds a minimal `results/` tree so the result-reading
tools return data, then exercises GET /health, GET /tools and POST /mcp for
both success and error paths.

The (dp, tp, pp) triple returned by `get_device_detail` is cross-checked against
`app.rank_layout.decompose`, so this test also guards the TP-DP-PP layout
contract end to end — see tests/test_mcp_server_rank_layout.py for the
in-process version.

Skipped (exit 0) when the server dependencies are absent.
Run: python tests/test_mcp_server_e2e.py
"""

import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

# Windows consoles default to a legacy code page; this report is UTF-8.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# The host may export a proxy whose upstream cannot reach PyPI/localhost;
# keep this test self-contained.
for _k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    os.environ.pop(_k, None)

try:
    import fastapi  # noqa: F401
except ImportError:
    print("  [SKIP] 未安装服务端依赖 fastapi，跳过端到端测试 —— 请先执行:")
    print("         pip install -r mcp_server/requirements.txt")
    sys.exit(0)

from app.rank_layout import decompose  # noqa: E402

PORT = 8799
BASE = f"http://127.0.0.1:{PORT}"
DP, TP, PP = 8, 16, 8

_failures: list[str] = []


def check(label: str, cond: bool, detail: str = "") -> None:
    print(("  [PASS] " if cond else "  [FAIL] ") + label + ("" if cond else f"   {detail}"))
    if not cond:
        _failures.append(label)


def rpc(tool: str, args: dict) -> dict:
    payload = json.dumps(
        {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {"name": tool, "arguments": args},
            "id": 1,
        }
    ).encode()
    req = urllib.request.Request(
        BASE + "/mcp", data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def rpc_ok(label: str, tool: str, args: dict) -> dict:
    """Call a tool, failing loudly on a JSON-RPC error instead of reading Nones."""
    r = rpc(tool, args)
    if "error" in r:
        check(label, False, f"JSON-RPC error: {r['error']}")
        return {}
    return r.get("result", {}) or {}


def get(path: str) -> tuple[int, dict]:
    with urllib.request.urlopen(BASE + path, timeout=10) as r:
        return r.status, json.loads(r.read().decode())


def topology(name: str, dp: int, tp: int, pp: int, **extra) -> dict:
    t = {
        "name": name,
        "device_type": "A3",
        "dp_size": dp,
        "tp_size": tp,
        "pp_size": pp,
        "total_nodes": dp * tp * pp,
        "num_layers": 64,
        "hidden_dim": 4096,
        "num_heads": 32,
    }
    t.update(extra)
    return t


_CSV_HEADER = (
    "comm_type,comm_group,comm_group_size,msg_size,stage,dst,src,additional,"
    "nonblock,wait_n,_elapsed_time,start_time,end_time,single_flops\n"
)
_CSV_ROWS = (
    "computation,None,None,None,forward/layer0,None,None,matmul,1,None,120.0,0.0,120.0,1.5e11\n"
    "all_reduce,tp_group,16,1048576,backward/layer0,None,None,Ring,0,None,80.0,120.0,200.0,None\n"
    "send,pp_group,8,524288,forward/layer0,8,None,None,1,None,40.0,200.0,240.0,None\n"
)
_STAT_TXT = """HBM Detail
weight: 1000000000
gradient: 500000000
optimizer: 2000000000
activation: 300000000
comm_buf: 100000000
total: 3900000000

Flops Detail
Total Flops: 1.5e12
Forward Flops: 5e11
Backward B Flops: 5e11
Backward W Flops: 5e11

Communication Detail
breakdown_by_group(counts, bytes):
tp_group: count=10, bytes=1000000
pp_group: count=2, bytes=200000
dp_group: count=4, bytes=400000
ep_group: count=3, bytes=300000
"""


def seed_results(task_dir: Path, ranks) -> None:
    """Fabricate the minimum `results/` tree the reader expects."""
    mocked = task_dir / "results" / "mocked_workload"
    stats = task_dir / "results" / "statistic_data"
    mocked.mkdir(parents=True, exist_ok=True)
    stats.mkdir(parents=True, exist_ok=True)
    for r in ranks:
        (mocked / f"model_rank{r}_time.csv").write_text(_CSV_HEADER + _CSV_ROWS, encoding="utf-8")
        (stats / f"rank{r}.txt").write_text(_STAT_TXT, encoding="utf-8")


work = REPO / ".tmp" / "mcp_server_e2e"
(work / "aicm").mkdir(parents=True, exist_ok=True)
ws_root = work / "ws"

env = dict(os.environ)
env.update(
    {
        "AICM_MCP_PORT": str(PORT),
        "AICM_MCP_DRY_RUN": "true",
        "AICM_MCP_SIM_TOOL_HOME": str(work / "aicm"),
        "AICM_MCP_WORKSPACE_ROOT": str(ws_root),
    }
)

print("=" * 74)
print("  MCP Server end-to-end (real HTTP)")
print("=" * 74)

server_log_path = work / "server.log"
server_log = open(server_log_path, "w", encoding="utf-8")
proc = subprocess.Popen(
    [sys.executable, "-m", "mcp_server"],
    cwd=str(REPO),
    env=env,
    stdout=server_log,
    stderr=subprocess.STDOUT,
)
try:
    # ── boot ──
    up = False
    for _ in range(60):
        if proc.poll() is not None:
            break
        try:
            get("/health")
            up = True
            break
        except Exception:
            time.sleep(0.5)
    check("服务启动并响应 GET /health", up)
    if not up:
        server_log.flush()
        print("--- server.log ---")
        print(server_log_path.read_text(encoding="utf-8", errors="replace")[-2500:])
        raise SystemExit(1)

    status, body = get("/health")
    check("GET /health -> 200 {status: ok}", status == 200 and body.get("status") == "ok", str(body))

    status, body = get("/tools")
    tools = body.get("tools", [])
    check("GET /tools 列出 9 个 tool", len(tools) == 9, str(len(tools)))

    # ── error paths ──
    r = rpc("no_such_tool", {})
    check("未知 tool -> JSON-RPC -32601", r.get("error", {}).get("code") == -32601, str(r.get("error")))
    check("错误响应仍含 result 字段", "result" in r, str(sorted(r)))

    r = rpc("report_status", {"task_id": "sim_does_not_exist"})
    check("非法 task_id -> -32001", r.get("error", {}).get("code") == -32001, str(r.get("error")))

    r = rpc(
        "execute_task",
        {"topology": topology("原始组网", DP, TP, PP, model_type="sparse"), "simulation_params": {}},
    )
    check(
        "sparse 缺 MoE 字段 -> -32602 参数错误",
        r.get("error", {}).get("code") == -32602,
        str(r.get("error", {}).get("message"))[:120],
    )

    # ── dense task: submit + seed results ──
    result = rpc_ok(
        "execute_task(dense)",
        "execute_task",
        {"topology": topology("原始组网", DP, TP, PP), "simulation_params": {}},
    )
    task_id = result.get("task_id", "")
    check("execute_task(dense) 返回 task_id", task_id.startswith("sim_"), str(result))
    check("execute_task 状态为 submitted", result.get("status") == "submitted")

    if task_id:
        probe_ranks = (0, 100, 200, 777, 1023)
        seed_results(ws_root / task_id, probe_ranks)

        result = rpc_ok("report_status", "report_status", {"task_id": task_id})
        check(
            "report_status 可用",
            result.get("status") in ("submitted", "running", "completed"),
            str(result),
        )

        result = rpc_ok("sync_logs", "sync_logs", {"task_id": task_id, "offset": 0})
        check("sync_logs 返回增量日志", len(result.get("lines", [])) > 0, str(result))

        result = rpc_ok("get_training_script", "get_training_script", {"task_id": task_id})
        script = result.get("script_content", "")
        check("get_training_script 返回脚本", bool(script))
        for token in ("DP=8", "TP=16", "PP=8", "NUM_LAYERS=64", "GLOBAL_BATCH_SIZE="):
            check(f"  脚本含 {token}", token in script)

        # ── the TP-DP-PP contract, through real HTTP ──
        for rank in probe_ranks:
            res = rpc_ok(
                f"get_device_detail rank={rank}",
                "get_device_detail",
                {"task_id": task_id, "global_rank": rank, "offset": 0},
            )
            got = (res.get("dp_rank"), res.get("tp_rank"), res.get("pp_rank"))
            want = decompose(rank, DP, TP, PP)
            check(f"get_device_detail rank={rank} -> (dp,tp,pp)={want}", got == want, f"got {got}")
            check(
                f"  rank={rank} 解析出算子",
                len(res.get("operators", [])) == 3,
                str(len(res.get("operators", []))),
            )
            tl = res.get("timeline") or {}
            check(
                f"  rank={rank} 时间线计算/通信耗时",
                tl.get("compute_time_ms", 0) > 0 and tl.get("comm_time_ms", 0) > 0,
                str(tl),
            )

        # ── other result-reading tools ──
        res = rpc_ok("card_detail", "card_detail", {"task_id": task_id})
        cards = res.get("cards", [])
        check("card_detail 返回卡指标", len(cards) == len(probe_ranks), str(len(cards)))
        if cards:
            c = cards[0]
            check(
                "  card_detail 含 flops/hbm 明细",
                "flops_detail" in c and "hbm_detail" in c,
                str(sorted(c)),
            )
            check("  card_detail ep_comm 字段存在", "ep_comm_gb_per_step" in c, str(sorted(c)))

        res = rpc_ok("get_hbm_detail", "get_hbm_detail", {"task_id": task_id, "global_rank": 100})
        check(
            "get_hbm_detail 返回四项分解",
            all(k in res for k in ("weights_gb", "gradients_gb", "optimizer_gb", "activations_gb")),
            str(sorted(res)),
        )

        for ct in ("tp", "pp", "dp"):
            res = rpc_ok(
                f"get_comm_detail({ct})",
                "get_comm_detail",
                {"task_id": task_id, "global_rank": 100, "comm_type": ct},
            )
            check(
                f"get_comm_detail({ct}) 返回通信详情",
                res.get("comm_count", 0) > 0 and res.get("total_comm_gb", 0) > 0,
                str(res),
            )

        res = rpc_ok("get_result", "get_result", {"task_id": task_id})
        check(
            "get_result 返回 summary + cards",
            bool(res.get("summary")) and len(res.get("cards", [])) == len(probe_ranks),
            str(res.get("summary")),
        )

    # ── MoE task ──
    moe = topology(
        "原始组网", DP, TP, PP,
        model_type="sparse", ep=8, num_experts=256, moe_router_topk=8,
        num_moe_layers=58, moe_ffn_hidden_size=2048, has_shared_expert=True,
        shared_expert_intermediate_size=2048,
    )
    result = rpc_ok("execute_task(sparse/MoE)", "execute_task", {"topology": moe, "simulation_params": {}})
    moe_task = result.get("task_id", "")
    check("execute_task(sparse/MoE) 返回 task_id", moe_task.startswith("sim_"), str(result))
    if moe_task:
        result = rpc_ok("MoE get_training_script", "get_training_script", {"task_id": moe_task})
        mscript = result.get("script_content", "")
        for token in (
            "NUM_EXPERTS=256",
            "MOE_ROUTER_TOPK=8",
            "NUM_MOE_LAYERS=58",
            "MOE_LAYER_FREQ=[0]*6+[1]*58",
            "MOE_FFN_HIDDEN_SIZE=2048",
            "HAS_SHARED_EXPERT=true",
            "SHARED_EXPERT_INTERMEDIATE_SIZE=2048",
            "EP=8",
            "--expert-model-parallel-size ${EP}",
            "--num-experts 256",
            "--moe-router-topk 8",
            "--moe-layer-freq [0]*6+[1]*58",
            "--moe-ffn-hidden-size 2048",
            "--n-shared-experts 1",
            "--moe-shared-expert-intermediate-size 2048",
        ):
            check(f"  MoE 脚本含 {token}", token in mscript)

        # GPT_ARGS 逐行结构：续行符齐备且每行只有一个参数
        # （曾把 --expert-model-parallel-size 粘到 --overlap-param-gather 同一行）
        body = mscript.split('GPT_ARGS="', 1)[1].rsplit('"', 1)[0]
        arg_lines = body.strip("\n").splitlines()
        glued = [ln for ln in arg_lines if re.search(r"\\\s+--", ln)]
        check("  MoE GPT_ARGS 无「反斜杠+空格+参数」粘连行", not glued, str(glued))
        check(
            "  MoE GPT_ARGS 每行一个参数且续行符齐备",
            all(ln.count("--") == 1 for ln in arg_lines)
            and all(ln.rstrip().endswith("\\") for ln in arg_lines[:-1])
            and not arg_lines[-1].rstrip().endswith("\\"),
            str(arg_lines),
        )

        # ── MoE 结果读取：EP 通信量 ──
        seed_results(ws_root / moe_task, (0,))
        res = rpc_ok("MoE card_detail", "card_detail", {"task_id": moe_task})
        moe_cards = res.get("cards", [])
        check(
            "  MoE card_detail 返回 ep_comm_gb_per_step > 0",
            bool(moe_cards) and moe_cards[0].get("ep_comm_gb_per_step", 0) > 0,
            str(moe_cards[:1]),
        )
        check(
            "  MoE card_detail comm_detail 含 ep 二级明细",
            bool(moe_cards)
            and moe_cards[0].get("comm_detail", {}).get("ep", {}).get("comm_count", 0) > 0,
            str(moe_cards[:1]),
        )
        res = rpc_ok(
            "MoE get_comm_detail(ep)",
            "get_comm_detail",
            {"task_id": moe_task, "global_rank": 0, "comm_type": "ep"},
        )
        check(
            "  MoE get_comm_detail(ep) 返回 EP 通信详情",
            res.get("comm_type") == "ep"
            and res.get("comm_count", 0) > 0
            and res.get("comm_cards") == 8,
            str(res),
        )

finally:
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
    server_log.close()

print()
if _failures:
    print(f"  [FAIL] {len(_failures)} 项未通过: {_failures}")
    sys.exit(1)
print("  [PASS] 端到端全部通过：服务可启动，JSON-RPC 正常，rank 分解与契约一致。")
