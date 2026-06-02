"""
HBM estimation: Original vs Equivalent topology comparison.
Run: python tests/test_hbm_equiv.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib

_skill = importlib.import_module("app.skills.training-mesh-profiler-skill")
_estimate_hbm_gb = _skill._estimate_hbm_gb
_estimate_hbm_gb_old = _skill._estimate_hbm_gb_old

# -- Fixed model params --
H = 4096
dff = 14336
S = 2048
B = 32
b = 4
a = 1

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
    hbm_old = _estimate_hbm_gb_old(L, H, S, B_val, dp, tp, pp, a)

    print(f"\n{'-' * 60}")
    print(f"  {name}")
    print(f"{'-' * 60}")
    print(f"  L={L}  H={H}  dff={dff}  DP={dp}  TP={tp}  PP={pp}  B={B_val}  b={b}")

    print(f"\n  [NEW] HBM = L/PP * ((4*H^2 + 3*H*dff)/TP + 2*H) / 1e9")
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

    print(f"\n  [OLD] HBM = (params*16 + B*S*H*L/PP*2) / 1e9")
    params = L * (12 * H**2 + 4 * H) / (tp * pp)
    act = B_val * S * H * L / pp * 2
    hbm_old_bytes = params * 16 + act
    print(f"        = ({params:.2f} * 16 + {act:.2f}) / 1e9")
    print(f"        = {hbm_old_bytes:.2f} / 1e9")
    print(f"        = {hbm_old:.4f} GB")

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
