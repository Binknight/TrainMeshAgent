"""MoE（稀疏模型）在 MCP 侧的契约测试 —— 进程内运行，不需要启动服务。

覆盖三处历史上不完整/出错的环节：

1. **脚本生成**：GPT_ARGS 逐行结构 + MoE 参数。
   旧实现里 ``_overlap_optimizer_args`` 返回的片段没有尾换行，拼接后把 MoE 的
   ``--expert-model-parallel-size`` 粘到了 ``--overlap-param-gather`` 同一行，
   续行语义失效（dense 任务的 ``--use-flash-attn`` 同样被粘住）。
2. **参数接受**：sparse 条件必填 + 组合合法性；``moe_layer_freq`` 的透传与推导
   （旧实现把 ``MOE_LAYER_FREQ`` 硬编码为 1 = 全层 MoE，与同一脚本里的
   ``NUM_MOE_LAYERS=58`` 自相矛盾）。
3. **数据返回**：EP 通信（``ep_group``）解析、``card_detail.comm_detail.ep``
   与 ``get_comm_detail(comm_type="ep")``。

Run: python tests/test_moe_mcp_contract.py
"""

import re
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

try:
    import pydantic  # noqa: F401
    from pydantic import ValidationError
except ImportError:
    print("  [SKIP] 未安装 pydantic，跳过 MoE 契约测试 —— 请先执行:")
    print("         pip install -r mcp_server/requirements.txt")
    sys.exit(0)

try:
    from mcp_server.schemas.common import SimulationTaskInput
    from mcp_server.services.results_reader import (
        _effective_hbm_bytes,
        _parse_statistic_file,
        build_card_metrics,
        comm_consistency_warnings,
        ep_zero_warning,
        get_comm_detail,
    )
    from mcp_server.services.script_generator import (
        generate_topology_script,
        resolve_moe_layer_freq,
    )
except ImportError as exc:  # pragma: no cover - 依赖缺失时跳过而非报错
    print(f"  [SKIP] 无法导入 mcp_server 依赖（{exc}）—— 请先执行:")
    print("         pip install -r mcp_server/requirements.txt")
    sys.exit(0)

# Windows 控制台默认代码页与本报告（UTF-8）不一致
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_failures: list[str] = []

# 仿真侧实际输出的 rank0.txt 样例（稠密运行）：注意 `grad` 别名、
# `Backward_B_Flops` 下划线形式、以及 edp_group / cp_group 与 typed 分组同时出现。
_REAL_RANK0 = """rank0 HBM Detail (bytes):
total: 1.23e+11
weight: 5.01e+09
grad: 5.01e+09
optimizer: 3.01e+10
activation: 8.10e+10
comm_buf: 2.00e+09

rank0 Flops Detail:
Total Flops: 1.26e+14
Forward Flops: 4.22e+13
Backward_B_Flops: 4.28e+13
Backward_W_Flops: 4.08e+13
breakdown_by_operator(count, flops):
  rmsnorm_1: count=20, flops=5.37e+09
  attention_qkv: count=20, flops=8.25e+12
  mlp_linear_1: count=20, flops=1.98e+13
  rmsnorm_1_B: count=20, flops=5.37e+09

rank0 Communication Detail(bytes):
total_comm_count: 167
total_comm_bytes: 31310872588
breakdown_by_group(counts, bytes):
  dp_group: count=40, bytes=14936178688
  tp_group: count=126, bytes=16240476172
  edp_group: count=0, bytes=0
  ep_group: count=0, bytes=0
  pp_group: count=1, bytes=134217728
  cp_group: count=0, bytes=0
breakdown_by_group_and_type(counts, bytes):
  dp_group_all_reduce: count=1, bytes=1.00e+10
  dp_group_reduce_scatter: count=39, bytes=4.91e+09
  pp_group_recv: count=1, bytes=1.34e+08
  pp_group_send: count=1, bytes=1.34e+08
  tp_group_all_reduce: count=84, bytes=1.09e+10
  tp_group_broadcast: count=2, bytes=1.31e+05
  tp_group_wait: count=40, bytes=5.37e+09
"""


def check(label: str, cond: bool, detail: str = "") -> None:
    print(("  [PASS] " if cond else "  [FAIL] ") + label + ("" if cond else f"   {detail}"))
    if not cond:
        _failures.append(label)


class WorkDir:
    """仓库内临时目录上下文管理器。

    与 tests/test_mcp_server_e2e.py 一致：产物写在 ``<repo>/.tmp`` 下，
    不使用系统临时目录（受限环境下可能不可写）。
    """

    def __init__(self, path: Path) -> None:
        self.path = path

    def __enter__(self) -> str:
        shutil.rmtree(self.path, ignore_errors=True)
        self.path.mkdir(parents=True, exist_ok=True)
        return str(self.path)

    def __exit__(self, *exc) -> bool:
        shutil.rmtree(self.path, ignore_errors=True)
        return False


def dense_topology(**extra) -> SimulationTaskInput:
    base = dict(
        name="原始组网", device_type="A3",
        dp_size=8, tp_size=16, pp_size=8, total_nodes=1024,
        num_layers=64, hidden_dim=4096, num_heads=32, d_ffn=14336,
    )
    base.update(extra)
    return SimulationTaskInput(**base)


def moe_topology(**extra) -> SimulationTaskInput:
    base = dict(
        model_type="sparse", ep=8,
        num_experts=256, moe_router_topk=8,
        num_moe_layers=58, moe_ffn_hidden_size=2048,
        has_shared_expert=True, shared_expert_intermediate_size=2048,
    )
    base.update(extra)
    return dense_topology(**base)


def gpt_args_lines(script: str) -> list[str]:
    """取出 GPT_ARGS="..." 的内容行。"""
    body = script.split('GPT_ARGS="', 1)[1].rsplit('"', 1)[0]
    return [ln for ln in body.strip("\n").splitlines()]


def eval_layer_pattern(expr: str) -> list[int]:
    """求值 0/1 层分布表达式（仅允许数字、方括号、星号、加号）。"""
    if not re.fullmatch(r"[\d\[\]\*\+]+", expr):
        raise ValueError(f"非法的层分布表达式: {expr!r}")
    return eval(expr, {"__builtins__": {}}, {})  # noqa: S307 - 字符集已白名单校验


def check_gpt_args_structure(label: str, script: str) -> list[str]:
    """逐行结构校验：续行符齐备、没有把两个参数挤到同一行。"""
    lines = gpt_args_lines(script)
    glued = [ln for ln in lines if re.search(r"\\\s+--", ln)]
    check(f"{label}: 无「反斜杠+空格+参数」粘连行", not glued, str(glued))
    multi = [ln for ln in lines if ln.count("--") != 1]
    check(f"{label}: 每行恰好一个 CLI 参数", not multi, str(multi))
    missing_cont = [ln for ln in lines[:-1] if not ln.rstrip().endswith("\\")]
    check(f"{label}: 除末行外均以续行符结尾", not missing_cont, str(missing_cont))
    check(f"{label}: 末行无续行符", not lines[-1].rstrip().endswith("\\"), repr(lines[-1]))
    return lines


print("=" * 74)
print("  MoE MCP 契约（脚本生成 / 参数接受 / 数据返回）")
print("=" * 74)

with WorkDir(REPO / ".tmp" / "moe_mcp_contract") as tmp:
    tmp_dir = Path(tmp)

    # ── 1. 脚本生成：dense（DP>1 时 overlap 参数与 --use-flash-attn 必须各占一行）──
    print("\n[1] dense 脚本")
    dense_script = generate_topology_script(
        dense_topology(), tmp_dir / "dense.sh"
    ).read_text(encoding="utf-8")
    dense_lines = check_gpt_args_structure("dense", dense_script)
    check(
        "dense: --use-flash-attn 独占一行",
        "    --use-flash-attn \\" in dense_lines,
        str(dense_lines),
    )
    check(
        "dense: 不含 MoE 参数",
        "--num-experts" not in dense_script and "--moe-layer-freq" not in dense_script,
        "",
    )
    check(
        "dense: 不含 --sequence-parallel（aicm 仅 MoE 校验）",
        "--sequence-parallel" not in dense_script,
        "",
    )
    check(
        "dense: 仍暴露 EP 键（兜底 1）",
        bool(re.search(r"^EP=1$", dense_script, re.M)),
        "",
    )

    # ── 2. 脚本生成：MoE（推导层分布）──
    print("\n[2] MoE 脚本（按 num_moe_layers 推导层分布）")
    moe_script = generate_topology_script(moe_topology(), tmp_dir / "moe.sh").read_text("utf-8")
    moe_lines = check_gpt_args_structure("MoE", moe_script)
    for token in (
        "    --sequence-parallel \\",
        "    --expert-model-parallel-size ${EP} \\",
        "    --num-experts 256 \\",
        "    --moe-router-topk 8 \\",
        "    --moe-layer-freq [0]*6+[1]*58 \\",
        "    --moe-ffn-hidden-size 2048 \\",
        "    --expert-tensor-parallel-size 1 \\",
        "    --n-shared-experts 1 \\",
        "    --moe-shared-expert-intermediate-size 2048 \\",
    ):
        check(f"MoE: GPT_ARGS 含行 {token.strip()!r}", token in moe_lines, str(moe_lines))
    for pattern, label in (
        (r"^NUM_EXPERTS=256$", "NUM_EXPERTS"),
        (r"^MOE_ROUTER_TOPK=8$", "MOE_ROUTER_TOPK"),
        (r"^NUM_MOE_LAYERS=58$", "NUM_MOE_LAYERS"),
        (r"^MOE_FFN_HIDDEN_SIZE=2048$", "MOE_FFN_HIDDEN_SIZE"),
        (r"^HAS_SHARED_EXPERT=true$", "HAS_SHARED_EXPERT"),
        (r"^EXPERT_TP=1$", "EXPERT_TP"),
        (r"^SHARED_EXPERT_INTERMEDIATE_SIZE=2048$", "SHARED_EXPERT_INTERMEDIATE_SIZE"),
        (r"^EP=8$", "EP"),
    ):
        check(f"MoE: 脚本含键 {label}", bool(re.search(pattern, moe_script, re.M)), "")

    # MOE_LAYER_FREQ 必须与 NUM_LAYERS / NUM_MOE_LAYERS 自洽（旧实现硬编码 1 = 全 MoE）
    m = re.search(r"^MOE_LAYER_FREQ=(.+)$", moe_script, re.M)
    check("MoE: 脚本含 MOE_LAYER_FREQ", m is not None, "")
    if m:
        expr = m.group(1).strip()
        try:
            pat = eval_layer_pattern(expr)
        except ValueError as exc:
            check("MoE: MOE_LAYER_FREQ 可求值", False, str(exc))
            pat = []
        check("MoE: MOE_LAYER_FREQ 长度 = num_layers", len(pat) == 64, f"{expr} -> {len(pat)}")
        check("MoE: MOE_LAYER_FREQ 中 MoE 层数 = num_moe_layers", sum(pat) == 58, f"{expr} -> {sum(pat)}")
        check("MoE: MOE_LAYER_FREQ 非「全层 MoE」", expr != "1", expr)

    # ── 3. 显式 moe_layer_freq 透传（交替 MoE 层）与全 MoE 推导 ──
    print("\n[3] MoE 层分布：显式透传 / 全 MoE")
    alt_script = generate_topology_script(
        moe_topology(
            num_layers=48, num_moe_layers=24, moe_layer_freq="([0,1]*24)",
        ),
        tmp_dir / "alt.sh",
    ).read_text(encoding="utf-8")
    check(
        "显式 moe_layer_freq: 键为原表达式",
        bool(re.search(r"^MOE_LAYER_FREQ=\(\[0,1\]\*24\)$", alt_script, re.M)),
        "",
    )
    check(
        "显式 moe_layer_freq: 同时进 GPT_ARGS",
        "    --moe-layer-freq ([0,1]*24) \\" in gpt_args_lines(alt_script),
        "",
    )
    full_moe = resolve_moe_layer_freq(
        moe_topology(num_layers=48, num_moe_layers=48)
    )
    check("全 MoE: 推导为 '1'（等价 -1 哨兵）", full_moe == "1", full_moe)

    # ── 4. 参数接受：条件必填 + 组合合法性 ──
    print("\n[4] 参数接受与校验")
    try:
        moe_topology(ep=None)
        check("sparse 缺 ep -> 报错", False, "未报错")
    except ValidationError as exc:
        check("sparse 缺 ep -> 报错", "ep" in str(exc), str(exc)[:120])

    for label, kwargs in (
        ("num_moe_layers > num_layers", {"num_moe_layers": 65}),
        ("moe_router_topk > num_experts", {"moe_router_topk": 512}),
        ("num_experts 不能被 ep 整除", {"ep": 7, "num_experts": 256}),
        ("ep > dp*tp", {"ep": 256}),
    ):
        try:
            moe_topology(**kwargs)
            check(f"拒绝非法组合: {label}", False, "未报错")
        except ValidationError:
            check(f"拒绝非法组合: {label}", True)

    ok = moe_topology(moe_layer_freq="[0]*6+[1]*58")
    check(
        "接受可选字段 moe_layer_freq / shared_expert_intermediate_size",
        ok.moe_layer_freq == "[0]*6+[1]*58" and ok.shared_expert_intermediate_size == 2048,
        "",
    )
    dense = dense_topology()
    check(
        "dense 缺省：MoE 字段为 None / False（行为与旧版兼容）",
        dense.model_type == "dense"
        and dense.num_experts is None
        and dense.moe_layer_freq is None
        and dense.has_shared_expert is False,
        "",
    )

    # ── 5. 数据返回：EP 通信解析 ──
    print("\n[5] 数据返回：EP 通信")
    task_dir = tmp_dir / "sim_moe_task"
    (task_dir / "results" / "mocked_workload").mkdir(parents=True)
    (task_dir / "results" / "statistic_data").mkdir(parents=True)
    (task_dir / "results" / "statistic_data" / "rank0.txt").write_text(
        "HBM Detail\n"
        "weight: 1000000000\n"
        "gradient: 500000000\n"
        "optimizer: 2000000000\n"
        "activation: 300000000\n"
        "comm_buf: 100000000\n"
        "total: 3900000000\n"
        "\n"
        "Flops Detail\n"
        "Total Flops: 1.5e12\n"
        "Forward Flops: 5e11\n"
        "Backward B Flops: 5e11\n"
        "Backward W Flops: 5e11\n"
        "\n"
        "Communication Detail\n"
        "breakdown_by_group(counts, bytes):\n"
        "tp_group: count=10, bytes=1000000\n"
        "pp_group: count=2, bytes=200000\n"
        "dp_group: count=4, bytes=400000\n"
        "ep_group: count=3, bytes=805306368\n",
        encoding="utf-8",
    )
    topo = moe_topology()
    card = build_card_metrics(0, topo, task_dir / "results")
    check(
        "card_detail: ep_comm_gb_per_step 解析出 EP 通信量",
        abs(card.ep_comm_gb_per_step - 805306368 / (1024 ** 3)) < 1e-9,
        str(card.ep_comm_gb_per_step),
    )
    check(
        "card_detail: comm_detail 含 ep 二级明细",
        card.comm_detail.ep.comm_count == 3 and card.comm_detail.ep.comm_cards == 8,
        str(card.comm_detail.model_dump()),
    )
    record = SimpleNamespace(results_dir=task_dir / "results", topology=topo)
    ep_detail = get_comm_detail(record, 0, "ep")
    check(
        "get_comm_detail(ep) 返回 EP 通信详情",
        ep_detail.comm_type == "ep"
        and ep_detail.comm_count == 3
        and ep_detail.comm_cards == 8
        and ep_detail.total_comm_gb > 0,
        str(ep_detail.model_dump()),
    )

    # ── 6. 真实 rank0.txt 格式回归（edp_group / cp_group 与 typed 分组并存）──
    print("\n[6] 真实 rank0.txt 格式（稠密运行样例）")
    real_dir = tmp_dir / "sim_real"
    (real_dir / "results" / "statistic_data").mkdir(parents=True, exist_ok=True)
    real_path = real_dir / "results" / "statistic_data" / "rank0.txt"
    real_path.write_text(_REAL_RANK0, encoding="utf-8")
    real = _parse_statistic_file(real_path)
    check(
        "HBM: grad 别名归一为 gradient",
        real["hbm"].get("gradient") == 5.01e9 and real["hbm"].get("total") == 1.23e11,
        str(real["hbm"]),
    )
    check(
        "有效 HBM = total - activation - comm_buf",
        abs(_effective_hbm_bytes(real["hbm"]) - (1.23e11 - 8.10e10 - 2.00e9)) < 1.0,
        repr(_effective_hbm_bytes(real["hbm"])),
    )
    check(
        "Flops: Backward_B_Flops / Backward_W_Flops 归一为 backward_B/W_flops",
        real["flops"].get("backward_B_flops") == 4.28e13
        and real["flops"].get("backward_W_flops") == 4.08e13,
        str(real["flops"]),
    )
    groups = real["comm_by_group"]
    check(
        "分组汇总: dp_group 未被前缀相同的 edp_group 覆盖",
        groups.get("dp_group", {}).get("count") == 40
        and groups.get("dp_group", {}).get("bytes") == 14936178688.0,
        str(groups.get("dp_group")),
    )
    check(
        "edp_group / cp_group / ep_group 各自独立成键（互不覆盖）",
        groups.get("edp_group", {}).get("bytes") == 0.0
        and groups.get("cp_group", {}).get("bytes") == 0.0
        and groups.get("ep_group", {}).get("bytes") == 0.0,
        str({k: v for k, v in groups.items() if k.endswith("_group")}),
    )
    check(
        "typed 分组不与汇总冲突（dp_group_all_reduce 独立于 dp_group）",
        groups.get("dp_group_all_reduce", {}).get("bytes") == 1.0e10
        and groups.get("dp_group", {}).get("bytes") == 14936178688.0,
        str(sorted(groups)),
    )
    check(
        "分组自洽: Σ group bytes == total_comm_bytes (31310872588)",
        sum(v["bytes"] for k, v in groups.items() if k.endswith("_group")) == 31310872588.0,
        repr(sum(v["bytes"] for k, v in groups.items() if k.endswith("_group"))),
    )
    check(
        "分组自洽: Σ group counts == total_comm_count (167)",
        sum(v["count"] for k, v in groups.items() if k.endswith("_group")) == 167,
        repr(sum(v["count"] for k, v in groups.items() if k.endswith("_group"))),
    )
    real_card = build_card_metrics(0, topo, real_dir / "results")
    check(
        "稠密运行的真实文件: ep_comm=0 且不报错（ep_group=0 忠实返回 0）",
        real_card.ep_comm_gb_per_step == 0.0
        and real_card.dp_comm_gb_per_step > 0
        and real_card.comm_detail.ep.comm_count == 0,
        str(real_card.comm_detail.model_dump()),
    )
    real_ep = get_comm_detail(
        SimpleNamespace(results_dir=real_dir / "results", topology=topo), 0, "ep"
    )
    check(
        "真实文件 get_comm_detail(ep) 返回零值而不抛错",
        real_ep.comm_type == "ep" and real_ep.comm_count == 0 and real_ep.total_comm_gb == 0.0,
        str(real_ep.model_dump()),
    )

    # ── 7. 自洽校验与 EP 诊断（口径：EP 记在 ep_group，不做 edp 兜底）──
    print("\n[7] 分组自洽校验与 EP 零值诊断")
    check(
        "自洽校验: 真实文件无告警",
        comm_consistency_warnings(real) == [],
        str(comm_consistency_warnings(real)),
    )
    broken_path = real_dir / "results" / "statistic_data" / "rank1.txt"
    broken_path.write_text(
        _REAL_RANK0.replace("  dp_group: count=40, bytes=14936178688\n", "").replace(
            "rank0 ", "rank1 "
        ),
        encoding="utf-8",
    )
    broken_warnings = comm_consistency_warnings(_parse_statistic_file(broken_path))
    check(
        "自洽校验: 分组漏读时 bytes 与 counts 各报一条告警",
        len(broken_warnings) == 2,
        str(broken_warnings),
    )
    check(
        "自洽校验: 无 total_comm_* 时不误报（旧版输出）",
        comm_consistency_warnings({"comm_by_group": {"dp_group": {"count": 1, "bytes": 1.0}}}) == [],
        "",
    )
    check(
        "EP 诊断: 稀疏 ep=8 且 ep_group=0 -> 提示",
        ep_zero_warning(topo, real["comm_by_group"]) is not None,
        str(ep_zero_warning(topo, real["comm_by_group"])),
    )
    check(
        "EP 诊断: 稀疏 ep=1 -> 不提示（专家通信在卡内完成）",
        ep_zero_warning(moe_topology(ep=1), {"ep_group": {"count": 0, "bytes": 0.0}}) is None,
        "",
    )
    check(
        "EP 诊断: ep_group 有量 -> 不提示",
        ep_zero_warning(topo, {"ep_group": {"count": 3, "bytes": 12345.0}}) is None,
        "",
    )
    check(
        "EP 诊断: dense 任务 -> 不提示",
        ep_zero_warning(dense_topology(), {"ep_group": {"count": 0, "bytes": 0.0}}) is None,
        "",
    )

print()
if _failures:
    print(f"  [FAIL] {len(_failures)} 项未通过:")
    for f in _failures:
        print(f"    - {f}")
    sys.exit(1)
print("  [PASS] MoE MCP 契约全部通过：脚本逐行结构、MoE 参数/层分布、参数校验、EP 数据返回。")
