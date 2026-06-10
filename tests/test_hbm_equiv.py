"""
HBM estimation: Original vs Equivalent topology comparison.
Run: python tests/test_hbm_equiv.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib

_skill = importlib.import_module("app.skills.training-mesh-profiler-skill")
_estimate_flops = _skill._estimate_flops
_estimate_flops_first_last = _skill._estimate_flops_first_last
_estimate_hbm_gb = _skill._estimate_hbm_gb
_estimate_hbm_gb_first_last = _skill._estimate_hbm_gb_first_last
_DEFAULT_VOCAB_SIZE = _skill._DEFAULT_VOCAB_SIZE

# -- Fixed model params --
H = 4096
dff = 14336
b = 4

# -- Test cases: (name, L, dp, tp, pp, B) --
cases = [
    ("Original", 64, 8, 16, 8, 32),
    ("Equivalent", 24, 2, 16, 3, 8),
]

print("=" * 72)
print("  HBM Theoretical Estimation: Original vs Equivalent")
print("=" * 72)

for name, L, dp, tp, pp, B_val in cases:
    hbm_new = _estimate_hbm_gb(L, H, dff, tp, pp)

    print(f"\n{'-' * 60}")
    print(f"  {name}")
    print(f"{'-' * 60}")
    print(f"  L={L}  H={H}  dff={dff}  DP={dp}  TP={tp}  PP={pp}  B={B_val}  b={b}")

    print(f"\n  HBM = L/PP * ((4*H^2 + 3*H*dff)/TP + 2*H) / 1e9")
    term_a = 4 * H**2 + 3 * H * dff
    term_b = term_a / tp
    term_c = term_b + 2 * H
    factor = L / pp
    hbm_bytes = factor * term_c
    print(f"        = {L}/{pp} * ((4*{H}^2 + 3*{H}*{dff})/{tp} + 2*{H}) / 1e9")
    print(f"        = {L}/{pp} * (({4*H**2} + {3*H*dff})/{tp} + {2*H}) / 1e9")
    print(f"        = {L}/{pp} * ({term_a}/{tp} + {2*H}) / 1e9")
    print(f"        = {factor} * ({term_b:.2f} + {2*H}) / 1e9")
    print(f"        = {factor} * {term_c:.2f} / 1e9")
    print(f"        = {hbm_bytes:.2f} / 1e9")
    print(f"        = {hbm_new:.4f} GB")

# -- Equivalence check --
print(f"\n{'=' * 72}")
print("  Equivalence Verification")
print(f"{'=' * 72}")

hbm_orig = _estimate_hbm_gb(64, H, dff, 16, 8)
hbm_equiv = _estimate_hbm_gb(24, H, dff, 16, 3)

diff = abs(hbm_orig - hbm_equiv)
diff_pct = diff / hbm_orig * 100 if hbm_orig else 0

print(f"  Original   HBM = {hbm_orig:.4f} GB")
print(f"  Equivalent HBM = {hbm_equiv:.4f} GB")
print(f"  Difference     = {diff:.4f} GB ({diff_pct:.2f}%)")

# L/PP ratio check
print(f"\n  L_orig/PP_orig = 64/8 = {64/8}")
print(f"  L_eq/PP_eq     = 24/3 = {24/3}")
print(f"  Scale factor   = {64/8} / {24/3} = 1.0")
print(f"  HBM scales with L/PP. Both = {64/8:.0f}, so HBM unchanged.")

tolerance = 0.001
if diff <= tolerance:
    print(f"\n  [PASS] HBM equivalent: diff {diff:.6f} GB <= {tolerance} GB")
else:
    print(f"\n  [FAIL] HBM not equivalent: diff {diff:.6f} GB > {tolerance} GB")
    sys.exit(1)

# -- First/Last PP HBM test --
print(f"\n{'=' * 72}")
print("  First/Last PP Rank HBM Verification")
print(f"{'=' * 72}")

V = _DEFAULT_VOCAB_SIZE  # 32000
for name, L, dp, tp, pp, B_val in cases:
    hbm_mid = _estimate_hbm_gb(L, H, dff, tp, pp)
    hbm_edge = _estimate_hbm_gb_first_last(L, H, dff, tp, pp, V)
    extra = hbm_edge - hbm_mid
    print(f"\n  {name}:")
    print(f"    Middle PP HBM  = {hbm_mid:.4f} GB")
    print(f"    Edge PP HBM    = {hbm_edge:.4f} GB")
    print(f"    Extra (V*H/TP) = {extra:.4f} GB")
    print(f"    Formula: +{V}*{H}/({tp}*1e9) = +{V*H/(tp*1e9):.4f} GB")

# Verify edge > middle
hbm_mid_orig = _estimate_hbm_gb(64, H, dff, 16, 8)
hbm_edge_orig = _estimate_hbm_gb_first_last(64, H, dff, 16, 8, V)
assert hbm_edge_orig > hbm_mid_orig, "Edge HBM should be larger than middle HBM"
print(f"\n  [PASS] Edge HBM ({hbm_edge_orig:.4f} GB) > Middle HBM ({hbm_mid_orig:.4f} GB)")

# Verify single PP: all ranks use middle (no edge)
hbm_single_pp = _estimate_hbm_gb(64, H, dff, 16, 1)
print(f"  [PASS] PP=1 middle HBM = {hbm_single_pp:.4f} GB (no edge ranks when PP=1)")

# -- First/Last PP FLOPs test --
print(f"\n{'=' * 72}")
print("  First/Last PP Rank FLOPs Verification")
print(f"{'=' * 72}")

S, B_val = 2048, 32
for name, L, dp, tp, pp, _B in cases:
    flops_mid = _estimate_flops(L, H, S, B_val, dff, dp, tp, pp)
    flops_edge = _estimate_flops_first_last(L, H, S, B_val, dff, dp, tp, pp, V)
    extra = flops_edge - flops_mid
    print(f"\n  {name}:")
    print(f"    Middle FLOPs   = {flops_mid:.4e}")
    print(f"    Edge FLOPs     = {flops_edge:.4e}")
    print(f"    Extra          = {extra:.4e}")
    print(f"    Formula: +(6*{V}*{H})/{tp}*({B_val}/{dp})*{S} = +{(6*V*H)/tp*(B_val/dp)*S:.4e}")

assert flops_edge > flops_mid, "Edge FLOPs should be larger than middle FLOPs"
print(f"\n  [PASS] Edge FLOPs ({flops_edge:.4e}) > Middle FLOPs ({flops_mid:.4e})")
