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
_estimate_hbm_gb = _skill._estimate_hbm_gb
_estimate_tp_comm_gb = _skill._estimate_tp_comm_gb
_estimate_pp_comm_mb = _skill._estimate_pp_comm_mb
_estimate_dp_comm_gb = _skill._estimate_dp_comm_gb
_MODEL_CONFIG = _skill._MODEL_CONFIG

S = 2048
B = 32
a = 1

# (name, device_type, dp, tp, pp, overrides)
# overrides can include "num_layers" and/or "hidden_dim"; omitted keys fall back to _MODEL_CONFIG
topologies = [
    ("Original", "A3", 8, 16, 8, {"hidden_dim": 6144, "num_layers": 128}),
    ("Equivalent", "A3", 3, 16, 3, {"hidden_dim": 6144, "num_layers": 48}),
]

print(
    f"{'Topology':<12} {'L':>4} {'H':>6} {'FLOPs/card':>22} {'HBM(GB)':>10} {'TP(GB/m)':>10} {'PP(MB/m)':>12} {'DP(GB/s)':>10}"
)
print("-" * 100)

for name, device_type, dp, tp, pp, overrides in topologies:
    cfg = _MODEL_CONFIG[device_type]
    H = int(overrides.get("hidden_dim", cfg["hidden_dim"]))
    L = int(overrides.get("num_layers", cfg["num_layers"]))

    f = _estimate_flops(L, H, S, B, dp, tp, pp)
    h = _estimate_hbm_gb(L, H, S, B, dp, tp, pp, a)
    tp_c = _estimate_tp_comm_gb(L, H, dp)
    pp_c = _estimate_pp_comm_mb(L, H, S, B, tp)
    dp_c = _estimate_dp_comm_gb(H, S, B)
    print(
        f"{name:<12} {L:>4} {H:>6} {f:>22.4e} {h:>10.4f} {tp_c:>10.4f} {pp_c:>12.2f} {dp_c:>10.4f}"
    )
