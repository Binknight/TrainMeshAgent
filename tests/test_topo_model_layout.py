"""
Model-section layout regression test (static/topo-renderer.js).

`_modelLayoutGeom()` decides the x/width of the two model cards drawn under the
topology. The invariant it must never break:

    hasBothModels == true  =>  x0 / areaW / x0Eq / areaWEq 全是有限数

That invariant used to hold only "by accident": the geometry was assigned per
*topology* layout mode, so only the three-part branch and the "both meshes
present" branch produced the equivalent side. But whether the equivalent model
is *rendered* depends on `hasBothModels`, which looks at the model payloads
only. On session restore the models arrive synchronously while the topologies
arrive one awaited `loadMeshData()` at a time, so the in-between rebuild had
both models and a single mesh => `modelX0Eq`/`modelAreaWEq` were `undefined`,
`undefined + NaN` became NaN and d3 reported

    <text> attribute x: Expected length, "NaN"
    <g> attribute transform: Expected number, "translate(NaN,457) scale(N…"

and dropped the whole equivalent-model group.

This test runs the real function from the shipped file (between the
`@testable-model-layout` markers) in Node, with no DOM needed.

Run: python tests/test_topo_model_layout.py
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
_START = "/* @testable-model-layout:start */"
_END = "/* @testable-model-layout:end */"

_failures: list[str] = []


def check(label: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  [PASS] {label}")
    else:
        print(f"  [FAIL] {label} {detail}")
        _failures.append(label)


# ── Fixtures ────────────────────────────────────────────────────────────────
# dpW comes from the stubbed calcDims below: dpW = tp * pp * 40.
_ORIG = {"tp": 4, "pp": 2}  # dpW 320
_EQ = {"tp": 2, "pp": 2}  # dpW 160
_W = 1200
_THREE = {"mode": "three", "origW": 400, "eqW": 420, "gap": 10}
_TWO = {"mode": "two", "origW": 700, "cardW": 440, "gap": 10}

# Split fallback: gap 24, avail 1176, share 0.5 => moW/meW 588, areaW 572, x0Eq 620
_SPLIT = {"x0": 8, "areaW": 572, "x0Eq": 620, "areaWEq": 572}
# Both meshes, one model: share = 320/480 clamped to 0.6
_SHARED = {"x0": 8, "areaW": 705.6 - 16, "x0Eq": 705.6 + 32, "areaWEq": 1176 * 0.4 - 16}

# name, topoLayout, meshOriginal, meshEquivalent, hasBothModels, expected (or None)
_CASES: list[tuple] = [
    # ── the reported crash: restore in flight, models in, one mesh only ──
    ("restore-mid, orig mesh only, both models", None, _ORIG, None, True, _SPLIT),
    ("restore-mid, two-part layout, both models", _TWO, _ORIG, None, True, _SPLIT),
    ("restore-mid, equiv mesh only, both models", None, None, _EQ, True, _SPLIT),
    # ── states that already worked: behaviour must not move ──
    ("three-part, both models", _THREE, _ORIG, _EQ, True,
     {"x0": 8, "areaW": 384, "x0Eq": 418, "areaWEq": 404}),
    ("three-part, single model", _THREE, _ORIG, _EQ, False,
     {"x0": 8, "areaW": 384, "x0Eq": 418, "areaWEq": 404}),
    ("both meshes, single model", None, _ORIG, _EQ, False, _SHARED),
    ("two-part, single model", _TWO, _ORIG, None, False,
     {"x0": 8, "areaW": 684, "x0Eq": None, "areaWEq": None}),
    ("no mesh, single model", None, None, None, False,
     {"x0": 16, "areaW": _W - 32, "x0Eq": None, "areaWEq": None}),
]

_DRIVER = """
function calcDims(tp, pp) { return { dpW: tp * pp * 40 }; }
var cases = __CASES__;
var out = {};
for (var i = 0; i < cases.length; i++) {
  var c = cases[i];
  out[c.name] = _modelLayoutGeom({
    meshWidth: c.meshWidth,
    topoLayout: c.topoLayout,
    meshOriginal: c.meshOriginal,
    meshEquivalent: c.meshEquivalent,
    hasBothModels: c.hasBothModels,
    calcDims: calcDims,
  });
}
process.stdout.write(JSON.stringify(out));
"""


def _finite(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v == v and abs(v) != float("inf")


def _run_cases():
    with open(_TOPO, encoding="utf-8") as fh:
        src = fh.read()
    if _START not in src or _END not in src:
        check("topo-renderer.js 里能找到 @testable-model-layout 标记", False,
              f"缺少标记，无法取出 _modelLayoutGeom（{_START!r} / {_END!r}）")
        return None
    body = src.split(_START, 1)[1].split(_END, 1)[0]

    payload = [
        {
            "name": name,
            "meshWidth": _W,
            "topoLayout": topo,
            "meshOriginal": mo,
            "meshEquivalent": me,
            "hasBothModels": both,
        }
        for name, topo, mo, me, both, _exp in _CASES
    ]
    script = body + _DRIVER.replace("__CASES__", json.dumps(payload))

    # 探针脚本落在 .tmp/ 下（已在 .gitignore 中）。注意不要用 tempfile.mkdtemp：
    # 它按 0o700 建目录，在受限令牌下新建的子目录反而写不进去。
    tmp_root = os.path.join(_ROOT, ".tmp")
    os.makedirs(tmp_root, exist_ok=True)
    js_path = os.path.join(tmp_root, f"topo_layout_probe_{uuid.uuid4().hex}.js")
    try:
        with open(js_path, "w", encoding="utf-8") as fh:
            fh.write(script)
        proc = subprocess.run(
            ["node", js_path], capture_output=True, timeout=60,
        )
        if proc.returncode != 0:
            check("node 执行 _modelLayoutGeom", False,
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


def main() -> int:
    if shutil.which("node") is None:
        print("  [SKIP] 未找到 node，跳过 (需要 node 执行 topo-renderer.js 里的纯函数)")
        return 0

    geom = _run_cases()
    if geom is None:
        print()
        print(f"  [FAIL] {len(_failures)} 项未通过: {_failures}")
        return 1

    for name, _topo, _mo, _me, both, expected in _CASES:
        g = geom.get(name, {})
        x0, areaW, x0Eq, areaWEq = g.get("x0"), g.get("areaW"), g.get("x0Eq"), g.get("areaWEq")

        # The invariant that the NaN bug violated.
        if both:
            check(f"{name}: 两侧模型 => 四个几何量均为有限正数",
                  all(_finite(v) for v in (x0, areaW, x0Eq, areaWEq)) and areaW > 0 and areaWEq > 0,
                  f"got {g}")
        else:
            # 单模型时等效侧可能预留了栏位（三栏 / 两侧组网），也可能留空；
            # 但任何一种情况下都不允许出现 NaN。
            check(f"{name}: 单模型 => 左侧有限正数，等效侧为 null 或有限数",
                  _finite(x0) and _finite(areaW) and areaW > 0
                  and (x0Eq is None or _finite(x0Eq))
                  and (areaWEq is None or _finite(areaWEq)),
                  f"got {g}")

        if expected is not None:
            ok = all(
                (_finite(g.get(k)) and _finite(v) and abs(g[k] - v) < 1e-6)
                if v is not None else g.get(k) is None
                for k, v in expected.items()
            )
            check(f"{name}: 数值与既有布局一致", ok, f"expected {expected}, got {g}")

    print()
    if _failures:
        print(f"  [FAIL] {len(_failures)} 项未通过: {_failures}")
        return 1
    print("  [PASS] 模型区几何量不变量成立：凡是渲染两侧模型的状态都没有 NaN 坐标。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
