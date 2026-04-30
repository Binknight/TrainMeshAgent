"""
Output metrics for Original and Equivalent topologies.
Run: python tests/test_mesh_profiler.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
logging.getLogger().setLevel(logging.ERROR)

import importlib

_skill = importlib.import_module("app.skills.training-mesh-profiler-skill")

_estimate_flops = _skill._estimate_flops
_estimate_hbm_gb = _skill._estimate_hbm_gb
_estimate_tp_comm_gb = _skill._estimate_tp_comm_gb
_estimate_pp_comm_mb = _skill._estimate_pp_comm_mb
_estimate_dp_comm_gb = _skill._estimate_dp_comm_gb

S = 2048
B = 32
a = 1
H = 4096

topologies = [
    ("Original",   128, 8, 8, 8),
    ("Equivalent",  48, 3, 8, 3),
]

print(f"{'Topology':<12} {'FLOPs/card':>22} {'HBM(GB)':>10} {'TP(GB/m)':>10} {'PP(MB/m)':>12} {'DP(GB/s)':>10}")
print("-" * 80)

for name, L, dp, tp, pp in topologies:
    f = _estimate_flops(L, H, S, B, dp, tp, pp)
    h = _estimate_hbm_gb(L, H, S, B, dp, tp, pp, a)
    tp_c = _estimate_tp_comm_gb(L, H, dp)
    pp_c = _estimate_pp_comm_mb(L, H, S, B, tp)
    dp_c = _estimate_dp_comm_gb(H, S, B)
    print(f"{name:<12} {f:>22.4e} {h:>10.4f} {tp_c:>10.4f} {pp_c:>12.2f} {dp_c:>10.4f}")
