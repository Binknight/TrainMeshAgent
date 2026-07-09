"""
测试等效前后理论估算指标对比
模型: DeepSeek-R1 (megatron) [MoE]

验证目标：
  1. 等效组网 num_moe_layers 应正确缩减 (58→18)
  2. 每卡 FLOPs / HBM / 通信量应大致一致 (±15%)
     （等效组网保持每卡处理层数相近：61/8≈7.6 vs 21/3=7）

用法：
  python tests/test_estimate_equivalence.py          # 通过 HTTP API 测试
  python tests/test_estimate_equivalence.py --local  # 直接调用公式验证（无需启动服务）
"""

import json
import os
import sys
import math
import urllib.request
import urllib.error

# Ensure project root is on Python path for local imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

API_BASE = "http://127.0.0.1:5000"
ESTIMATE_URL = f"{API_BASE}/api/session/estimate"


# ═══════════════════════════════════════════════════════════════════════════
# 等效参数计算
# ═══════════════════════════════════════════════════════════════════════════

def compute_equivalent_params(orig: dict) -> dict:
    """根据原始组网参数计算等效组网参数。

    等效策略:
      eq_pp = min(pp-1, 3)   — PP 缩减到 ≤3
      eq_dp = 2              — DP 缩减到 2
      eq_L  = (L//pp)*3       — 层数按 PP 比例缩减
      eq_B  = B * eq_dp/dp   — 批次按 DP 比例缩减
      eq_num_moe_layers = eq_L - (L - num_moe_layers)  — 保持稠密层数不变
    """
    dp = orig["dp"]
    tp = orig["tp"]
    pp = orig["pp"]
    L = orig["num_layers"]
    B = orig["total_batch"]
    n_moe = orig.get("num_moe_layers", L)
    ep = orig.get("ep", dp)

    eq_pp = max(1, min(pp - 1, 3)) if pp > 3 else pp
    eq_dp = 2 if dp >= 2 else 1
    eq_L = (L // pp) * 3 if pp > 3 else L
    eq_B = max(1, int(B * eq_dp / dp)) if dp > 1 else B
    eq_ep = ep
    eq_n_moe = max(1, eq_L - (L - n_moe))

    eq_nodes = eq_dp * tp * eq_pp

    eq = dict(orig)
    eq.update({
        "total_nodes": eq_nodes,
        "dp": eq_dp,
        "pp": eq_pp,
        "num_layers": eq_L,
        "total_batch": eq_B,
        "ep": eq_ep,
        "num_moe_layers": eq_n_moe,
    })
    return eq


# ═══════════════════════════════════════════════════════════════════════════
# 本地公式验证（绕过 HTTP，直接调用 Python 公式）
# ═══════════════════════════════════════════════════════════════════════════

def local_flops(params: dict) -> float:
    """直接调用 compute_moe_flops 计算单卡 FLOPs."""
    import importlib
    _moe = importlib.import_module("app.skills.training-mesh-profiler-skill.moe_estimator")
    compute_moe_flops = _moe.compute_moe_flops
    n_moe = params["num_moe_layers"]
    return compute_moe_flops(
        micro_batch_size=params["micro_batch"],
        seq_len=params["seq_len"],
        num_layers=params["num_layers"],
        hidden_size=params["hidden_dim"],
        tensor_parallel=params["tp"],
        num_moe_layers=n_moe,
        expert_ffn_hidden_size=params["moe_ffn_hidden_size"],
        expert_parallel=params.get("ep", params["dp"]),
        topk=params["moe_router_topk"],
        num_shared_expert_layers=n_moe if params.get("has_shared_expert") else 0,
        pipeline_parallel=params["pp"],
    )


def local_hbm(params: dict) -> float:
    """直接调用 calculate_moe_hbm 计算单卡 HBM (GB)."""
    import importlib
    _moe = importlib.import_module("app.skills.training-mesh-profiler-skill.moe_estimator")
    calculate_moe_hbm = _moe.calculate_moe_hbm
    n_moe = params["num_moe_layers"]
    n_dense = params["num_layers"] - n_moe
    hbm_bytes = calculate_moe_hbm(
        num_dense_layers=n_dense,
        num_moe_layers=n_moe,
        pipeline_parallel=params["pp"],
        hidden_size=params["hidden_dim"],
        ffn_hidden_size=params["d_ffn"],
        tensor_parallel=params["tp"],
        expert_ffn_hidden_size=params["moe_ffn_hidden_size"],
        num_experts=params["num_experts"],
        expert_parallel=params.get("ep", params["dp"]),
        vocab_size=params["vocab_size"],
        expert_tensor_parallel=params.get("expert_tensor_parallel_size", 1),
    )
    return hbm_bytes / 1e9


def local_tp_comm(params: dict) -> float:
    """TP 通信量 (GB/micro-batch)."""
    import importlib
    _est = importlib.import_module("app.skills.training-mesh-profiler-skill")
    return _est._estimate_tp_comm_gb(
        params["num_layers"], params["hidden_dim"],
        params["seq_len"], params["micro_batch"], params["pp"],
    )


def local_pp_comm(params: dict) -> float:
    """PP 通信量 (MB/micro-batch)."""
    import importlib
    _est = importlib.import_module("app.skills.training-mesh-profiler-skill")
    return _est._estimate_pp_comm_mb(
        params["hidden_dim"], params["seq_len"], params["micro_batch"],
    )


def local_dp_comm(params: dict) -> float:
    """DP 通信量 (GB/step)."""
    import importlib
    _est = importlib.import_module("app.skills.training-mesh-profiler-skill")
    return _est._estimate_dp_comm_gb(
        params["num_layers"], params["hidden_dim"], params["d_ffn"],
        params["dp"], params["tp"], params["pp"],
    )


# ═══════════════════════════════════════════════════════════════════════════
# HTTP API
# ═══════════════════════════════════════════════════════════════════════════

def post_estimate(payload: dict) -> dict:
    req = urllib.request.Request(
        ESTIMATE_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise SystemExit(f"HTTP {e.code}: {body}")
    except urllib.error.URLError as e:
        raise SystemExit(f"API 连接失败: {e.reason}")


def aggregate(est: dict) -> dict:
    cards = est["cards"]
    n = len(cards)
    total_flops = sum(c["flops_per_card"] for c in cards)
    total_hbm = sum(c["hbm_gb"] for c in cards)
    first = cards[0]
    return {
        "nodes": n,
        "total_flops_tflops": total_flops / 1e3,
        "per_card_flops_tflops": total_flops / n / 1e3,
        "total_hbm_gb": total_hbm,
        "per_card_hbm_gb": total_hbm / n,
        "tp_comm_gb": first["tp_comm_gb_per_micro"],
        "pp_comm_mb": first["pp_comm_mb_per_micro"],
        "dp_comm_gb": first["dp_comm_gb_per_step"],
    }


def pct_diff(a, b):
    if a == 0:
        return float("inf")
    return (b - a) / a * 100


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    use_local = "--local" in sys.argv

    # ── DeepSeek-R1 原始参数 ──
    original = {
        "device_type": "A3",
        "total_nodes": 1024,
        "dp": 8,
        "tp": 16,
        "pp": 8,
        "num_layers": 61,
        "hidden_dim": 7168,
        "d_ffn": 18432,
        "seq_len": 4096,
        "total_batch": 32,
        "micro_batch": 1,
        "vocab_size": 129280,
        "model_type": "sparse",
        "num_experts": 256,
        "moe_router_topk": 8,
        "num_moe_layers": 58,
        "moe_ffn_hidden_size": 2048,
        "has_shared_expert": True,
        "expert_tensor_parallel_size": 1,
        "ep": 8,
    }

    equivalent = compute_equivalent_params(original)

    # ── 参数对比 ──
    print("=" * 72)
    print("DeepSeek-R1 (MoE) 等效前后理论估算指标对比")
    print("=" * 72)
    print(f"\n等效公式:")
    print(f"  eq_pp = min(pp-1, 3)           = min({original['pp']}-1, 3)  = {equivalent['pp']}")
    print(f"  eq_dp = 2                       = {equivalent['dp']}")
    print(f"  eq_L  = (L//pp)*3               = ({original['num_layers']}//{original['pp']})*3 = {equivalent['num_layers']}")
    print(f"  eq_B  = B * eq_dp/dp           = {original['total_batch']} * {equivalent['dp']}/{original['dp']} = {equivalent['total_batch']}")
    print(f"  稠密层 = L - num_moe_layers      = {original['num_layers']} - {original['num_moe_layers']} = {original['num_layers'] - original['num_moe_layers']}")
    print(f"  eq_moe = eq_L - 稠密层           = {equivalent['num_layers']} - {original['num_layers'] - original['num_moe_layers']} = {equivalent['num_moe_layers']}")
    print(f"  eq_nodes = eq_dp * tp * eq_pp  = {equivalent['dp']}*{equivalent['tp']}*{equivalent['pp']} = {equivalent['total_nodes']}")

    print(f"\n参数对比:")
    print(f"  {'':<22} {'原始':>18} {'等效':>18}")
    print(f"  {'-' * 58}")
    for key, label in [
        ("total_nodes", "节点数"),
        ("dp", "DP"),
        ("tp", "TP"),
        ("pp", "PP"),
        ("ep", "EP"),
        ("num_layers", "L (总层数)"),
        ("total_batch", "B (全局批次)"),
        ("num_moe_layers", "MoE 层数"),
    ]:
        print(f"  {label:<22} {original[key]:>18} {equivalent[key]:>18}")

    # ── 等效性预期 ──
    orig_layers_per_card = original["num_layers"] / original["pp"]
    eq_layers_per_card = equivalent["num_layers"] / equivalent["pp"]
    print(f"\n每卡层数: 原始 {orig_layers_per_card:.2f} vs 等效 {eq_layers_per_card:.2f}  (比值 {eq_layers_per_card/orig_layers_per_card:.3f})")

    # ── 计算指标 ──
    if use_local:
        print("\n[模式] 本地公式直接调用\n")
        o_flops = local_flops(original)
        o_hbm = local_hbm(original)
        o_tp = local_tp_comm(original)
        o_pp_comm = local_pp_comm(original)
        o_dp = local_dp_comm(original)

        e_flops = local_flops(equivalent)
        e_hbm = local_hbm(equivalent)
        e_tp = local_tp_comm(equivalent)
        e_pp_comm = local_pp_comm(equivalent)
        e_dp = local_dp_comm(equivalent)

        o_nodes = original["total_nodes"]
        e_nodes = equivalent["total_nodes"]
    else:
        print("\n[模式] HTTP API 请求\n")
        print("请求原始模型估算...")
        orig_est = post_estimate(original)
        print("请求等效模型估算...")
        eq_est = post_estimate(equivalent)

        orig_agg = aggregate(orig_est)
        eq_agg = aggregate(eq_est)

        o_nodes = orig_agg["nodes"]
        e_nodes = eq_agg["nodes"]
        o_flops = orig_agg["per_card_flops_tflops"] * 1e3  # back to raw
        e_flops = eq_agg["per_card_flops_tflops"] * 1e3
        o_hbm = orig_agg["per_card_hbm_gb"]
        e_hbm = eq_agg["per_card_hbm_gb"]
        o_tp = orig_agg["tp_comm_gb"]
        e_tp = eq_agg["tp_comm_gb"]
        o_pp_comm = orig_agg["pp_comm_mb"]
        e_pp_comm = eq_agg["pp_comm_mb"]
        o_dp = orig_agg["dp_comm_gb"]
        e_dp = eq_agg["dp_comm_gb"]

    # ── 指标对比 ──
    print(f"\n指标对比 (单卡):")
    print(f"  {'指标':<32} {'原始':>16} {'等效':>16} {'差异':>10}")
    print(f"  {'-' * 74}")

    rows = [
        ("FLOPs (TFLOPs)",        o_flops / 1e3,  e_flops / 1e3,  ".2f"),
        ("HBM (GB)",              o_hbm,          e_hbm,          ".1f"),
        ("TP 通信 (GB/micro)",    o_tp,           e_tp,           ".4f"),
        ("PP 通信 (MB/micro)",    o_pp_comm,      e_pp_comm,      ".1f"),
        ("DP 通信 (GB/step)",     o_dp,           e_dp,           ".2f"),
    ]

    for label, o_val, e_val, fmt in rows:
        diff = pct_diff(o_val, e_val)
        print(f"  {label:<32} {o_val:{fmt}} {e_val:{fmt}} {diff:>+9.1f}%")

    # ── 等效性检查 ──
    print(f"\n等效性检查:")
    print(f"  {'-' * 60}")

    checks = []

    # 1. num_moe_layers 应正确缩减
    expected_eq_moe = max(1, equivalent["num_layers"] - (original["num_layers"] - original["num_moe_layers"]))
    checks.append((
        f"MoE 层数缩减: {original['num_moe_layers']} -> {equivalent['num_moe_layers']} (预期 {expected_eq_moe})",
        equivalent["num_moe_layers"] == expected_eq_moe,
    ))

    # 2. 每卡 FLOPs 应接近 (ratio ~ L_pp_eq / L_pp_orig)
    expected_ratio = eq_layers_per_card / orig_layers_per_card
    actual_ratio = e_flops / o_flops if o_flops else 0
    flops_diff = abs(pct_diff(o_flops, e_flops))
    checks.append((
        f"每卡 FLOPs (±15%): 比值 {actual_ratio:.3f} (预期 ~{expected_ratio:.3f})",
        flops_diff < 15,
    ))

    # 3. 每卡 HBM 应接近
    hbm_diff = abs(pct_diff(o_hbm, e_hbm))
    checks.append((
        f"每卡 HBM (±15%): 差异 {hbm_diff:.1f}%",
        hbm_diff < 15,
    ))

    # 4. TP 通信应与每卡层数比例一致 (TP 不变)
    tp_diff = abs(pct_diff(o_tp, e_tp))
    checks.append((
        f"TP 通信: 差异 {tp_diff:.1f}% (应跟随 L/pp 变化 ~{(1-expected_ratio)*100:.1f}%)",
        tp_diff < 15,
    ))

    # 5. PP 通信应完全一致 (per-micro-batch 的 PP 边界通信量仅与 H/S/b 有关)
    pp_diff_val = abs(e_pp_comm - o_pp_comm)
    checks.append((
        f"PP 通信一致: 差异 {pp_diff_val:.6f} MB",
        pp_diff_val < 1e-6,
    ))

    # 6. DP 通信应随 batch 缩减
    dp_ratio = e_dp / o_dp if o_dp else 0
    expected_dp_ratio = equivalent["total_batch"] / original["total_batch"]
    dp_check = abs(dp_ratio - expected_dp_ratio) < 0.05
    checks.append((
        f"DP 通信按 batch 缩放: 实际 {dp_ratio:.2%} vs 预期 {expected_dp_ratio:.2%}",
        dp_check,
    ))

    all_pass = True
    for label, passed in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{status}] {label}")

    # ── 总结 ──
    print(f"\n{'=' * 72}")
    if all_pass:
        print("全部通过 - 等效组网指标一致")
    else:
        print("存在未通过项 - 请检查公式逻辑")
    print(f"{'=' * 72}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
