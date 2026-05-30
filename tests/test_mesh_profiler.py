"""
Output metrics for Original and Equivalent topologies.
Run: python tests/test_mesh_profiler.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging

logging.getLogger().setLevel(logging.ERROR)

import importlib

_skill = importlib.import_module("app.skills.training-mesh-profiler-skill")

_estimate_flops = _skill._estimate_flops
_estimate_flops_old = _skill._estimate_flops_old
_estimate_hbm_gb = _skill._estimate_hbm_gb
_estimate_hbm_gb_old = _skill._estimate_hbm_gb_old
_estimate_dp_comm_gb = _skill._estimate_dp_comm_gb
_estimate_dp_comm_gb_old_lh = _skill._estimate_dp_comm_gb_old_lh
_estimate_tp_comm_gb = _skill._estimate_tp_comm_gb
_estimate_pp_comm_mb = _skill._estimate_pp_comm_mb
_MODEL_CONFIG = _skill._MODEL_CONFIG

S = 2048
B = 32
dff = 14336
a = 1

# (name, device_type, dp, tp, pp, overrides)
topologies = [
    (
        "Original",
        "A3",
        8,
        16,
        8,
        {"hidden_dim": 4096, "num_layers": 64, "batch_size": 32},
    ),
    (
        "Equivalent",
        "A3",
        2,
        16,
        3,
        {"hidden_dim": 4096, "num_layers": 24, "batch_size": 8},
    ),
]

print("=" * 100)

for name, device_type, dp, tp, pp, overrides in topologies:
    cfg = _MODEL_CONFIG[device_type]
    H = int(overrides.get("hidden_dim", cfg["hidden_dim"]))
    L = int(overrides.get("num_layers", cfg["num_layers"]))
    B_val = int(overrides.get("batch_size", B))
    dff_val = int(overrides.get("dff", dff))
    total_nodes = dp * tp * pp

    # -- Compute --
    flops = _estimate_flops(L, H, S, B_val, dff_val, dp, tp, pp)
    flops_old = _estimate_flops_old(L, H, S, B_val, dp, tp, pp)
    hbm = _estimate_hbm_gb(L, H, dff_val, tp, pp)
    hbm_old = _estimate_hbm_gb_old(L, H, S, B_val, dp, tp, pp, a)
    dp_comm = _estimate_dp_comm_gb(L, H, dff_val, dp, tp, pp)
    dp_comm_old = _estimate_dp_comm_gb_old_lh(L, H, dp)
    tp_comm = _estimate_tp_comm_gb(L, H, S, B_val, pp)
    pp_comm = _estimate_pp_comm_mb(H, S, B_val)

    # -- Print --
    print(f"\n{'=' * 80}")
    print(
        f"  {name}  |  {device_type}  |  DP={dp}  TP={tp}  PP={pp}  --> {total_nodes} NPU"
    )
    print(f"{'=' * 80}")

    print(f"\n  -- Input Parameters --")
    print(f"  device_type  = {device_type}")
    print(f"  DP           = {dp}")
    print(f"  TP           = {tp}")
    print(f"  PP           = {pp}")
    print(f"  L  (layers)  = {L}")
    print(f"  H  (hidden)  = {H}")
    print(f"  S  (seq_len) = {S}")
    print(f"  B  (batch)   = {B_val}")
    print(f"  dff          = {dff_val}")
    print(f"  quant_coeff  = {a}")

    print(f"\n  -- NEW FLOPs Formula --")
    print(f"  FLOPs = (6*B*S*L*H/(DP*PP*TP)) * (4*H + 3*dff + 2*S)")
    print(
        f"        = (6*{B_val}*{S}*{L}*{H}/({dp}*{pp}*{tp})) * (4*{H} + 3*{dff_val} + 2*{S})"
    )
    numerator = 6 * B_val * S * L * H
    denominator = dp * pp * tp
    factor2 = 4 * H + 3 * dff_val + 2 * S
    print(f"        = ({numerator} / {denominator}) * {factor2}")
    print(f"        = {numerator / denominator:.2f} * {factor2}")
    print(f"        = {flops:.4e} FLOPs/card")

    print(f"\n  -- OLD FLOPs Formula (reference) --")
    print(f"  FLOPs = (72*B*S*H^2 + 12*B*S^2*H) / TP * L/PP")
    print(f"        = (72*{B_val}*{S}*{H}^2 + 12*{B_val}*{S}^2*{H}) / {tp} * {L}/{pp}")
    print(f"        = {flops_old:.4e} FLOPs/card")

    print(f"\n  -- NEW HBM Formula --")
    print(f"  HBM = L/PP * ((4*H^2 + 3*H*dff)/TP + 2*H) / 1e9")
    term_a = 4 * H**2 + 3 * H * dff_val
    term_b = term_a / tp
    term_c = term_b + 2 * H
    print(f"      = {L}/{pp} * ((4*{H}^2 + 3*{H}*{dff_val})/{tp} + 2*{H}) / 1e9")
    print(f"      = {L}/{pp} * (({4*H**2} + {3*H*dff_val})/{tp} + {2*H}) / 1e9")
    print(f"      = {L}/{pp} * ({term_a}/{tp} + {2*H}) / 1e9")
    print(f"      = {L}/{pp} * ({term_b:.2f} + {2*H}) / 1e9")
    print(f"      = {L}/{pp} * {term_c:.2f} / 1e9")
    print(f"      = {hbm:.4f} GB")
    print(f"\n  -- OLD HBM Formula (reference) --")
    print(f"  HBM = [L*(12*H^2+4*H)/(TP*PP) + B*S*H*L/PP + L*(12*H^2+4*H)/(TP*PP)] * a / 1e9")
    print(f"       = {hbm_old:.4f} GB")

    print(f"\n  -- Communication --")
    print(f"  DP comm = 2*(DP-1)/DP * 4 * L/PP * (4*H^2/TP + 3*H*dff/TP) / 1e9")
    print(f"          = {dp_comm:.4f} GB/step")
    print(f"  (OLD)   = 2*(DP-1)/DP * 12*L*H^2 / 1e9")
    print(f"          = {dp_comm_old:.4f} GB/step")
    print(f"  TP comm = L/PP * 15*B*S*H / 1e9")
    print(f"          = {tp_comm:.2f} GB/micro-step")
    print(f"  PP comm = 4*B*S*H / 1e6")
    print(f"          = {pp_comm:.4f} MB/micro-step")

# -- B scaling verification --
print(f"\n{'=' * 100}")
print("  B Scaling Verification  (eq_B = max(1, int(B * eq_dp / dp)))")
print(f"{'=' * 100}")
test_cases = [
    (32, 8, 2, 8),
    (32, 4, 1, 8),
    (32, 2, 1, 16),
    (32, 1, 1, 32),
    (64, 8, 2, 16),
    (1, 8, 2, 1),
]
print(
    f"  {'B':>4}  {'dp':>4}  {'eq_dp':>6}  {'eq_B(calc)':>12}  {'eq_B(expected)':>14}  {'Result':>6}"
)
print(f"  {'-' * 58}")
all_pass = True
for B_in, dp_in, eq_dp_in, expected in test_cases:
    eq_B_calc = max(1, int(B_in * eq_dp_in / dp_in)) if dp_in > 1 else B_in
    ok = eq_B_calc == expected
    if not ok:
        all_pass = False
    print(
        f"  {B_in:>4}  {dp_in:>4}  {eq_dp_in:>6}  {eq_B_calc:>12}  {expected:>14}  {'PASS' if ok else 'FAIL':>6}"
    )

print()
if all_pass:
    print("  [PASS] All B scaling tests passed.")
else:
    print("  [FAIL] SOME B SCALING TESTS FAILED!")
    sys.exit(1)
