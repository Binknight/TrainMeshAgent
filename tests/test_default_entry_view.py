"""
Default entry view regression test (static/topo-renderer.js + static/index.html).

Product rule locked here (2026-11 confirmed):

  1. 点击历史任务 / 刷新恢复时，**两侧仿真结果都齐** → 默认「等效仿真验证」页签；
     否则（未开始 / 进行中 / 失败 / 只有一侧结果 / 无组网） → 「等效建模分析」。
  2. 默认页签固定**原始组网 global rank 0**；有等效组网时同时固定
     `_mapRankToOtherSide("orig", 0)` 映射出的等效侧 rank（两侧都无 / 只有原始
     组网时的行为分别是「不固定」和「mappedRank=null」）。
  3. 「等效结果判定」页签不作为默认入口（用户手动点开）。
  4. 默认固定必须在异步 `fetchSimulationData()` 回填数据后重放：`meshRebuild()`
     会清空 pinned rank，若不重放，首帧固定会在数据到达后消失。

What is actually executed: the pure decision/mapping functions between their
`@testable-entry-view` markers in the shipped `topo-renderer.js`, in Node (no DOM).
Plus a few static wiring guards so the restore path cannot silently drop the
auto-pin / re-apply hooks.

Run: python tests/test_default_entry_view.py
"""

import json
import os
import shutil
import subprocess
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Windows consoles default to a legacy code page; the report below is UTF-8.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TOPO = os.path.join(_ROOT, "static", "topo-renderer.js")
_INDEX = os.path.join(_ROOT, "static", "index.html")

_START = "/* @testable-entry-view:start */"
_END = "/* @testable-entry-view:end */"

_failures: list[str] = []


def check(label: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  [PASS] {label}")
    else:
        print(f"  [FAIL] {label} {detail}")
        _failures.append(label)


# ── Fixtures ────────────────────────────────────────────────────────────────
# Tab decision: (name, state, expected_tab)
_TAB_CASES = [
    ("两侧结果齐 + 两侧组网", dict(simCompleted=True, hasOriginalTopology=True, hasEquivalentTopology=True), "simulation"),
    ("两侧结果齐 + 仅原始组网（残缺数据兜底）", dict(simCompleted=True, hasOriginalTopology=True, hasEquivalentTopology=False), "modeling"),
    ("两侧结果齐但无组网（防御性回退）", dict(simCompleted=True, hasOriginalTopology=False, hasEquivalentTopology=False), "modeling"),
    ("仿真未开始", dict(simCompleted=False, hasOriginalTopology=True, hasEquivalentTopology=True), "modeling"),
    ("仿真进行中/失败（只有一侧结果）", dict(simCompleted=False, hasOriginalTopology=True, hasEquivalentTopology=True), "modeling"),
    ("仅等效推导完成未仿真", dict(simCompleted=False, hasOriginalTopology=True, hasEquivalentTopology=False), "modeling"),
    ("空会话", dict(simCompleted=False, hasOriginalTopology=False, hasEquivalentTopology=False), "modeling"),
]

# Rank plan: (name, orig, eq, expected) — expected rank fields for global rank 0.
_ORIG = dict(tp=4, pp=2, dp=4)
_EQ = dict(tp=2, pp=2, dp=2)
_ORIG_ONLY = dict(tp=4, pp=2, dp=4)
_PLAN_CASES = [
    ("两侧组网：rank0 联动映射 rank0", _ORIG, _EQ, 0),
    ("异构组网：rank0 仍映射 rank0", dict(tp=8, pp=4, dp=2), dict(tp=2, pp=2, dp=1), 0),
    ("仅原始组网：无映射 rank", _ORIG_ONLY, None, None),
    ("无原始组网：不固定", None, _EQ, None),
]

_DRIVER = """
var tabCases = __TAB_CASES__;
var planCases = __PLAN_CASES__;

var out = { tabs: {}, plans: {}, mapExplicit: null, mapFallback: null };

for (var i = 0; i < tabCases.length; i++) {
  var tc = tabCases[i];
  out.tabs[tc.name] = resolveEntryCanvasTab(tc.state);
}

for (var j = 0; j < planCases.length; j++) {
  var pc = planCases[j];
  out.plans[pc.name] = resolveEntryRankPlan(pc.orig, pc.eq);
}

// rank0 之外：带显式组网参数的重载必须与全局回退路径结果一致（防止改动
// _mapRankToOtherSide 签名时破坏原有 2 参调用）。
var orig = { tp: 4, pp: 2, dp: 4 };
var eq = { tp: 2, pp: 2, dp: 2 };
out.mapExplicit = _mapRankToOtherSide("orig", 17, orig, eq);
meshOriginal = orig;
meshEquivalent = eq;
out.mapFallback = _mapRankToOtherSide("orig", 17);
out.decomposeRank0 = meshDecomposeRank(0, orig.tp, orig.dp);

process.stdout.write(JSON.stringify(out));
"""


def _run_node_cases():
    with open(_TOPO, encoding="utf-8") as fh:
        src = fh.read()
    if _START not in src or _END not in src:
        check("topo-renderer.js 里能找到 @testable-entry-view 标记", False,
              f"缺少标记，无法取出默认进入视图纯函数（{_START!r} / {_END!r}）")
        return None
    body = src.split(_START, 1)[1].split(_END, 1)[0]

    script = (
        body
        + _DRIVER.replace("__TAB_CASES__", json.dumps([{"name": n, "state": s} for n, s, _e in _TAB_CASES]))
        .replace("__PLAN_CASES__", json.dumps([{"name": n, "orig": o, "eq": e} for n, o, e, _m in _PLAN_CASES]))
    )

    # 探针脚本落在 .tmp/ 下（已在 .gitignore 中）。不要用 tempfile.mkdtemp：
    # 它按 0o700 建目录，在受限令牌下新建的子目录反而写不进去。
    tmp_root = os.path.join(_ROOT, ".tmp")
    os.makedirs(tmp_root, exist_ok=True)
    js_path = os.path.join(tmp_root, f"default_entry_view_probe_{uuid.uuid4().hex}.js")
    try:
        with open(js_path, "w", encoding="utf-8") as fh:
            fh.write(script)
        proc = subprocess.run(["node", js_path], capture_output=True, timeout=60)
        if proc.returncode != 0:
            check("node 执行入口视图纯函数", False,
                  proc.stderr.decode("utf-8", "replace").strip()[:400])
            return None
        return json.loads(proc.stdout.decode("utf-8"))
    finally:
        # 只删自己刚建的那个探针文件
        if os.path.realpath(os.path.dirname(js_path)) == os.path.realpath(tmp_root):
            try:
                os.remove(js_path)
            except OSError:
                pass


def _check_static_wiring() -> None:
    with open(_TOPO, encoding="utf-8") as fh:
        topo = fh.read()
    with open(_INDEX, encoding="utf-8") as fh:
        html = fh.read()

    # 恢复入口：默认 tab 由状态判定，不再写死 resetTab / 沿用 sessionStorage 旧逻辑
    check("index.html 用 resolveEntryCanvasTab 决定默认页签",
          "resolveEntryCanvasTab(" in html)
    check("index.html 以「两侧结果齐」为完成判据",
          "data.original_simulation && data.equivalent_simulation" in html)
    check("index.html 不再使用旧的 resetTab 强制建模页逻辑",
          "resetTab" not in html, "仍能找到 resetTab")
    check("index.html 不再在异步恢复前预切 sessionStorage 旧页签",
          "Switch to saved tab BEFORE" not in html)

    # 默认 rank 固定：进入时 stage，建模页补画 bar，异步数据回填后 reapply
    check("index.html 在仿真/建模两个入口都放了默认 rank pin 状态",
          html.count("window._stageEntryRankPin(") >= 2,
          f"调用点 {html.count('window._stageEntryRankPin(')} 个")
    check("index.html 建模页重建后补画联动卡 rank bar",
          "window._applyEntryRankPin(\"modeling\"" in html)
    check("default rank 固定暴露了 clear/stage/apply/reapply 四个入口",
          all(k in topo for k in (
              "window._clearEntryAutoPin", "window._stageEntryRankPin",
              "window._applyEntryRankPin", "window._reapplyEntryRankPin",
          )))
    check("fetchSimulationData 在 meshRebuild 清 pin 后重放默认固定",
          "window._reapplyEntryRankPin" in topo)
    check("用户手动操作会清除自动固定意图（防止异步回填覆盖用户选择）",
          "clearEntryAutoPin();" in topo and "clearEntryAutoPin()" in html)

    # 幽灵 pin / 悬空详情 / 迟到响应三条配套守卫（评审发现的回归点）
    check("切会话时清空全部 rank pin 状态（不只清意图）",
          "window._clearAllRankPins" in topo
          and "window._clearAllRankPins" in html
          and "function clearAllRankPins()" in topo)
    check("fetchSimulationData 带会话快照守卫（迟到响应丢弃）",
          "reqSessionId !== sessionId" in topo)
    check("meshRebuild 清画布 pin 时同步清底部详情用的 window._pinnedSim",
          "_simPinnedRank = null;\n    _simPinnedTpInfo = null;\n    // 与画布 pin 同步清空" in topo
          or ("与画布 pin 同步清空" in topo and "window._pinnedSim = null" in topo))

    # 默认固定复用既有 rank 布局单一事实来源，不允许重写换算公式
    check("默认 rank 计划使用 meshDecomposeRank / _mapRankToOtherSide",
          "meshDecomposeRank(0, origTopo.tp, origTopo.dp)" in topo
          and "_mapRankToOtherSide(\"orig\", 0, origTopo, eqTopo)" in topo)

    # 拓扑到达后建立/刷新默认联动（空会话 Step1、Step2 等效组网后到）
    check("index.html 在空会话/新建任务/新会话处登记「待建立默认联动」",
          html.count("window._requestEntryAutoPin(") >= 3,
          f"调用点 {html.count('window._requestEntryAutoPin(')} 个")
    check("topo-renderer.js 提供待建立/拓扑变化刷新机制",
          all(k in topo for k in (
              "_entryAutoPinWanted", "window._requestEntryAutoPin",
              "_maybeEstablishEntryPin()", "_refreshEntryRankPinForTopologyChange()",
          )))
    check("loadMeshData 在拓扑赋值后调用建立/刷新默认联动",
          "if (typeof _maybeEstablishEntryPin === \"function\") _maybeEstablishEntryPin();" in topo
          and "if (typeof _refreshEntryRankPinForTopologyChange === \"function\") {" in topo)
    check("三栏 canvasRebuild 末尾按 _entryAutoPin 补画联动卡 bar",
          "if (_entryAutoPin && _entryAutoPin.mappedRank != null && _centerPanelState.barG)" in topo)
    check("自动固定 rank 的重复重建不重播 bar 动画（防联动卡闪烁）",
          topo.count("_centerPanelState._skipBarAnim = true;") >= 3,
          f"skip flag 次数 {topo.count('_centerPanelState._skipBarAnim = true;')}")


def main() -> int:
    if shutil.which("node") is None:
        print("  [SKIP] 未找到 node，跳过 (需要 node 执行 topo-renderer.js 里的纯函数)")
        return 0

    data = _run_node_cases()
    if data is None:
        print()
        print(f"  [FAIL] {len(_failures)} 项未通过: {_failures}")
        return 1

    # ── A. 默认页签判定 ──
    for name, _state, expected in _TAB_CASES:
        got = data["tabs"].get(name)
        check(f"默认页签: {name} -> {expected}", got == expected, f"got {got!r}")

    # ── B. 第一个 rank 固定计划 ──
    for name, orig, eq, expected_mapped in _PLAN_CASES:
        plan = data["plans"].get(name)
        if orig is None:
            check(f"rank 计划: {name} -> null", plan is None, f"got {plan!r}")
            continue
        check(f"rank 计划: {name} 固定原始 rank 0", plan is not None
              and plan.get("side") == "orig" and plan.get("globalRank") == 0,
              f"got {plan!r}")
        check(f"rank 计划: {name} 映射 rank 符合预期（{expected_mapped}）",
              plan is not None and plan.get("mappedRank") == expected_mapped,
              f"got {plan!r}")
        check(f"rank 计划: {name} 的 TP/PP 下标为 0",
              plan is not None and plan.get("tpIndex") == 0 and plan.get("ppIndex") == 0,
              f"got {plan!r}")

    check("映射函数：显式组网参数重载与全局回退结果一致",
          data["mapExplicit"] == data["mapFallback"] == 5,
          f"explicit={data['mapExplicit']} fallback={data['mapFallback']}")
    check("rank0 分解：tpIdx/dpIdx/ppIdx 均为 0",
          data["decomposeRank0"] == {"tpIdx": 0, "dpIdx": 0, "ppIdx": 0},
          f"got {data['decomposeRank0']}")

    # ── C. 静态接线守卫（保证恢复路径不会悄悄丢掉默认固定）──
    _check_static_wiring()

    print()
    if _failures:
        print(f"  [FAIL] {len(_failures)} 项未通过: {_failures}")
        return 1
    print("  [PASS] 默认进入视图规则正确：两侧结果齐 -> 仿真验证页，否则建模分析页。")
    print("  [PASS] 默认固定原始组网 global rank 0，并联动等效侧映射 rank（仅原始组网时不渲染对比卡）。")
    print("  [PASS] 默认固定与异步数据回填、用户手动操作之间的重放/清理接线完整。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
