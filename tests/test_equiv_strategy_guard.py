"""等效策略入口校验回归测试（app/routes/session.py workflow_step1）。

背景：Step 1 参数表单里有「单卡极限等效」(single_card_extreme) 选项，但参数
推导只实现了「最小集群等效」(min_equiv)。修复前选择单卡极限等效照样能提交，
后端静默按 min_equiv 计算 —— 用户以为做了极限等效，实际拿到的是最小集群等效。

本测试锁住入口校验：
  1. single_card_extreme → 400 + error=strategy_not_implemented，且会话不被推进
  2. min_equiv → 200，正常生成原始组网/模型
  3. 缺省 strategy → 按 min_equiv 处理（老前端兼容）
  4. 未知 strategy → 400（不能静默回落）

不启动服务器、不碰用户数据：数据库指向 .tmp/ 下的临时 SQLite，
用 Flask test client 直接打路由。

Run: python tests/test_equiv_strategy_guard.py
"""

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# 必须在 import app.* 之前设好：app.main 导入时会执行 init_db()，
# 而默认 SQLITE_PATH 是容器路径 /home/data/db/...，在 Windows 上不存在。
_TMP_DB = REPO / ".tmp" / "strategy_guard_test"
_TMP_DB.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("SQLITE_PATH", str(_TMP_DB / "test.db"))
os.environ.setdefault("DATABASE_URL", "")

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_failures: list[str] = []


def check(label: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  [PASS] {label}")
    else:
        print(f"  [FAIL] {label} {detail}")
        _failures.append(label)


# 一份能通过 guardrail 的原始组网/模型参数（A3 DP8 TP16 PP8）
_BASE_PARAMS = {
    "device_type": "A3",
    "dp": 8,
    "tp": 16,
    "pp": 8,
    "model_name": "Qwen3-32B",
    "model_type": "dense",
    "L": 64,
    "H": 4096,
    "A": 32,
    "S": 2048,
    "B": 32,
    "b_micro": 4,
    "dff": 14336,
    "vocab_size": 32000,
}


def main() -> int:
    from app.main import create_app

    app = create_app()
    client = app.test_client()

    r = client.post("/api/session")
    check("创建临时会话", r.status_code == 200, f"HTTP {r.status_code}")
    sid = (r.get_json() or {})["session_id"]

    # 1) 未实现策略必须 400，且不能产生工作流副作用
    bad = client.post(
        f"/api/session/{sid}/workflow/step1",
        json={**_BASE_PARAMS, "strategy": "single_card_extreme"},
    )
    bad_json = bad.get_json() or {}
    check(
        "single_card_extreme 被拒绝（400 + strategy_not_implemented）",
        bad.status_code == 400
        and bad_json.get("error") == "strategy_not_implemented",
        f"HTTP {bad.status_code} {bad_json}",
    )
    check(
        "拒绝后错误信息提示最小集群等效",
        "最小集群等效" in str(bad_json.get("message", "")),
        str(bad_json.get("message", ""))[:120],
    )
    topo = client.get(f"/api/session/{sid}/topology").get_json() or {}
    check(
        "拒绝后会话未被推进（无原始组网）",
        not topo.get("original_topology"),
    )

    # 2) 已实现策略仍可正常提交（不能误伤）
    ok = client.post(
        f"/api/session/{sid}/workflow/step1",
        json={**_BASE_PARAMS, "strategy": "min_equiv"},
    )
    ok_json = ok.get_json() or {}
    check(
        "min_equiv 正常通过（原始组网 + 模型已生成）",
        ok.status_code == 200
        and bool(ok_json.get("original_mesh"))
        and bool(ok_json.get("original_model")),
        f"HTTP {ok.status_code} keys={list(ok_json)[:6]}",
    )

    # 3) 不传 strategy 时默认 min_equiv（老前端兼容）
    r2 = client.post("/api/session")
    sid2 = (r2.get_json() or {})["session_id"]
    default = client.post(f"/api/session/{sid2}/workflow/step1", json=_BASE_PARAMS)
    default_json = default.get_json() or {}
    check(
        "缺省 strategy 走 min_equiv（200）",
        default.status_code == 200 and bool(default_json.get("original_mesh")),
        f"HTTP {default.status_code} keys={list(default_json)[:6]}",
    )

    # 4) 未知策略同样拒绝（禁止静默回落）
    r3 = client.post("/api/session")
    sid3 = (r3.get_json() or {})["session_id"]
    unknown = client.post(
        f"/api/session/{sid3}/workflow/step1",
        json={**_BASE_PARAMS, "strategy": "not_a_strategy"},
    )
    check(
        "未知策略被拒绝（400 + strategy_not_implemented）",
        unknown.status_code == 400
        and (unknown.get_json() or {}).get("error") == "strategy_not_implemented",
        f"HTTP {unknown.status_code} {unknown.get_json()}",
    )

    print()
    if _failures:
        print(f"  [FAIL] {len(_failures)} 项未通过: {_failures}")
        return 1
    print("  [PASS] 等效策略入口校验符合预期：未实现的策略不会静默按 min_equiv 执行。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
