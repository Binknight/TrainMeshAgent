"""会话从 DB 恢复时的等效组网判定回归测试（app.agent.session._load_session）。

背景（2026-11 用户实测）：很多「只完成 Step1」的任务（session.step=params_collected）
在重启服务后，打开时会显示一个「等效组网」但**没有等效模型结构、推导卡空白**，而且
步骤条把后续节点标成已生成/仿真中，看起来像功能坏了。

根因：`_persist_topology()` 对**两侧**都写 `topology_params` 表 —— Step1 除了真实
生成的原始组网，还会把「等效目标参数」（dp/tp/pp 等，用于重启后继续跑 Step2）写成
`role='equivalent'` 的行。而 `_load_session()` 原先只要该行有 `name` 就重建
`equivalent_topology`，于是 Step1-only 会话被伪装成「Step2 已完成」：前端据此按两侧
组网渲染，但没有等效模型（num_layers 为空）和 formula_lines，于是缺模型/空卡片。

修复约定（本测试锁定）：
  · 等效侧只有 `topology_params.num_layers` 非空（= Step2 真的生成过等效模型）时才恢复
    `equivalent_topology`；
  · 无论如何都恢复 `equivalent_params`（否则重启后 Step2 无法继续执行）；
  · 原始侧拓扑照旧恢复（Step1 必定生成过）。

Run: python tests/test_session_restore_equiv.py
"""

import os
import shutil
import sys
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# 必须在 import app.* 之前设好：app.main 导入时会执行 init_db()，
# 而默认 SQLITE_PATH 是容器路径 /home/data/db/...，在 Windows 上不存在。
_TMP_DB = REPO / ".tmp" / "session_restore_equiv_test"
_TMP_DB.mkdir(parents=True, exist_ok=True)
_DB_FILE = _TMP_DB / "test.db"
for suffix in ("", "-wal", "-shm"):
    try:
        Path(str(_DB_FILE) + suffix).unlink()
    except OSError:
        pass
os.environ["SQLITE_PATH"] = str(_DB_FILE)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" — {detail}" if detail and not ok else ""))
    if not ok:
        _failures.append(label)


# ── Step1 落库的等效目标参数（没有 num_layers，模拟真实的 params_collected 会话） ──
_ORIG_PARAMS = {
    "name": "原始组网 (A3 DP8 TP8 PP16)",
    "device_type": "A3",
    "dp_size": 8, "tp_size": 8, "pp_size": 16, "total_nodes": 1024,
    "model_name": "GPT-3-175B",
    "num_layers": 96, "hidden_dim": 12288, "num_heads": 96, "d_ffn": 49152,
    "seq_len": 2048, "batch_size": 1536, "micro_batch_size": 1, "vocab_size": 50257,
    "model_type": "dense",
}
_EQ_TARGET_PARAMS = {
    # Step1 就算出来的等效目标参数：只有组网形状，没有模型参数
    "name": "等效组网",
    "device_type": "A3",
    "dp_size": 2, "tp_size": 8, "pp_size": 3, "total_nodes": 48,
    "model_name": "GPT-3-175B_eq",
    "seq_len": 2048, "batch_size": 384, "micro_batch_size": 1,
}
_EQ_GENERATED_PARAMS = dict(
    _EQ_TARGET_PARAMS,
    num_layers=21, hidden_dim=12288, num_heads=96, d_ffn=49152, model_type="dense",
)


def main() -> int:
    from app.main import create_app  # noqa: F401 — 导入即 init_db()

    app = create_app()  # noqa: F841 — 起库/建表（本测试只直接调 DAO + _load_session）
    from app.dao import create_session, save_topology_params, update_session_step
    from app.agent.session import _load_session

    # ── 1. Step1-only 会话：重启恢复后不得出现「假的等效组网」 ──
    sid = "restore-step1-" + uuid.uuid4().hex[:8]
    create_session(sid)
    update_session_step(sid, "params_collected", None, None)
    save_topology_params(sid, "original", _ORIG_PARAMS)
    save_topology_params(sid, "equivalent", _EQ_TARGET_PARAMS)

    state = _load_session(sid)
    check("Step1-only: 会话能从 DB 恢复", state is not None)
    check("Step1-only: step 仍为 params_collected", state is not None and state.step == "params_collected",
          getattr(state, "step", None))
    check("Step1-only: 原始组网恢复", state is not None and state.original_topology is not None)
    check("Step1-only: 原始模型恢复", state is not None and state.original_training_model is not None)
    check("Step1-only: 不重建等效组网（核心修复点）",
          state is not None and state.equivalent_topology is None,
          repr(getattr(state, "equivalent_topology", "missing")))
    check("Step1-only: 不重建等效模型",
          state is not None and state.equivalent_training_model is None)
    eq_params = getattr(state, "equivalent_params", None) if state else None
    check("Step1-only: 等效目标参数仍恢复（重启后可继续跑 Step2）",
          eq_params is not None and eq_params.dp == 2 and eq_params.tp == 8 and eq_params.pp == 3,
          repr(eq_params))

    # ── 2. Step2 已生成等效组网/模型：重启后必须完整恢复（不能误伤正常会话） ──
    save_topology_params(sid, "equivalent", _EQ_GENERATED_PARAMS)
    state2 = _load_session(sid)
    check("Step2 done: 等效组网恢复", state2 is not None and state2.equivalent_topology is not None)
    check("Step2 done: 等效组网节点数正确（2*8*3=48）",
          state2 is not None and state2.equivalent_topology is not None
          and state2.equivalent_topology.total_nodes == 48,
          repr(getattr(getattr(state2, "equivalent_topology", None), "total_nodes", None)))
    eq_model = getattr(state2, "equivalent_training_model", None) if state2 else None
    check("Step2 done: 等效模型恢复（num_layers=21）",
          eq_model is not None and eq_model.config.num_layers == 21,
          repr(getattr(getattr(eq_model, "config", None), "num_layers", None)))

    # ── 3. 完全没有 topology_params 的空会话：能恢复且不伪造拓扑 ──
    sid2 = "restore-empty-" + uuid.uuid4().hex[:8]
    create_session(sid2)
    update_session_step(sid2, "idle", None, None)
    state3 = _load_session(sid2)
    check("空会话: 能恢复", state3 is not None)
    check("空会话: 不伪造两侧组网/参数",
          state3 is not None and state3.original_topology is None
          and state3.equivalent_topology is None
          and getattr(state3, "original_params", None) is None
          and getattr(state3, "equivalent_params", None) is None)

    # ── 4. 前端配套修复：无公式行时不得把推导卡清空 ──
    topo_src = (REPO / "static" / "topo-renderer.js").read_text(encoding="utf-8")
    replay_start = topo_src.find("function _replayFormulaLines()")
    replay_body = topo_src[replay_start:replay_start + 4000] if replay_start >= 0 else ""
    empty_guard = replay_body.find("if (!lines.length)")
    remove_call = replay_body.find('cardG.select(".formula-content-group").remove();')
    check("前端: _replayFormulaLines 在无公式行时保留默认占位（先判空再删除）",
          replay_start >= 0 and empty_guard >= 0 and remove_call >= 0 and empty_guard < remove_call,
          f"empty_guard={empty_guard} remove_call={remove_call}")
    # 占位内容必须带标记，且流式打字机开始写真实公式行前先清占位；
    # 否则点 Step2 后真实公式行会叠在占位模板上（2026-11 用户实测文字堆叠）。
    check("前端: 占位内容带 formula-content-placeholder 标记",
          "formula-content-group formula-content-placeholder" in topo_src)
    check("前端: 打字机开始前清占位并复位公式行游标",
          'classed("formula-content-placeholder")' in topo_src
          and "_formulaLineY = _formulaCardHeaderH;" in topo_src)

    print()
    if _failures:
        print(f"  [FAIL] {len(_failures)} 项未通过: {_failures}")
        return 1
    print("  [PASS] Step1-only 会话重启后不再伪造等效组网；Step2 已生成的会话恢复正常。")
    print("  [PASS] 等效目标参数仍可恢复（重启后 Step2 可继续）；无公式行时推导卡不再空白。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
