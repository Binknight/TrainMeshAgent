"""REST endpoints for session management."""
import importlib
import math
import random
from flask import Blueprint, request, jsonify

from app.agent.session import session_manager
from app.models.schemas import (
    CardMetrics, DeviceType, DeviceSimulationDetail,
    OperatorTrace, SessionState, TimelineSummary,
)

session_bp = Blueprint("session", __name__, url_prefix="/api/session")


@session_bp.route("", methods=["POST"])
def create_session():
    """Create a new agent session."""
    session = session_manager.create_session()
    return jsonify(session.model_dump())


@session_bp.route("/<session_id>", methods=["GET"])
def get_session(session_id: str):
    """Get session state by ID."""
    session = session_manager.get_session(session_id)
    if not session:
        return {"error": "session not found"}, 404
    return jsonify(session.model_dump())


@session_bp.route("", methods=["GET"])
def list_sessions():
    """List all active sessions."""
    sessions = session_manager.list_sessions()
    return jsonify([s.model_dump() for s in sessions])


@session_bp.route("/<session_id>", methods=["DELETE"])
def delete_session(session_id: str):
    """Delete a session."""
    ok = session_manager.delete_session(session_id)
    if not ok:
        return {"error": "session not found"}, 404
    return {"status": "deleted", "session_id": session_id}


@session_bp.route("/<session_id>/topology", methods=["GET"])
def get_topology(session_id: str):
    """Get the mesh topology data for a session."""
    session = session_manager.get_session(session_id)
    if not session:
        return {"error": "session not found"}, 404

    return jsonify({
        "session_id": session_id,
        "original_topology": session.original_topology.model_dump() if session.original_topology else None,
        "equivalent_topology": session.equivalent_topology.model_dump() if session.equivalent_topology else None,
        "step": session.step,
    })


@session_bp.route("/<session_id>/simulation", methods=["GET"])
def get_simulation(session_id: str):
    """Get simulation results for a session."""
    session = session_manager.get_session(session_id)
    if not session:
        return {"error": "session not found"}, 404

    return jsonify({
        "session_id": session_id,
        "original_simulation": session.original_simulation.model_dump() if session.original_simulation else None,
        "equivalent_simulation": session.equivalent_simulation.model_dump() if session.equivalent_simulation else None,
        "comparison_report": session.comparison_report.model_dump() if session.comparison_report else None,
        "step": session.step,
    })


@session_bp.route("/estimate", methods=["POST"])
def estimate_metrics():
    """Compute per-card metrics using the Python estimation formulas."""
    data = request.get_json()
    if not data:
        return {"error": "request body is required"}, 400

    try:
        device_type = DeviceType(data["device_type"].upper())
        total_nodes = int(data["total_nodes"])
        dp = int(data["dp"])
        tp = int(data["tp"])
        pp = int(data["pp"])
    except (KeyError, ValueError) as e:
        return {"error": f"invalid or missing parameter: {e}"}, 400

    _est = importlib.import_module("app.skills.training-mesh-profiler-skill")

    cfg = _est._MODEL_CONFIG[device_type]
    L = cfg["num_layers"]
    H = cfg["hidden_dim"]
    S = int(data.get("seq_len", _est._SEQ_LEN))
    B = int(data.get("total_batch", _est._TOTAL_BATCH))
    a = float(data.get("quant_coeff", _est._QUANT_COEFF))

    flops = _est._estimate_flops(L, H, S, B, dp, tp, pp)
    hbm = _est._estimate_hbm_gb(L, H, S, B, dp, tp, pp, a)
    tp_comm = _est._estimate_tp_comm_gb(L, H, dp)
    pp_comm = _est._estimate_pp_comm_mb(L, H, S, B, tp)
    dp_comm = _est._estimate_dp_comm_gb(H, S, B)

    cards = []
    for rank in range(total_nodes):
        cards.append(CardMetrics(
            card_id=f"card_{rank}",
            global_rank=rank,
            flops_per_card=flops,
            hbm_gb=hbm,
            tp_comm_gb_per_micro=tp_comm,
            pp_comm_mb_per_micro=pp_comm,
            dp_comm_gb_per_step=dp_comm,
        ))

    return jsonify({
        "device_type": data["device_type"].upper(),
        "total_nodes": total_nodes,
        "dp": dp,
        "tp": tp,
        "pp": pp,
        "cards": [c.model_dump() for c in cards],
    })


# ── Mock operator trace generation ──

_OP_NAMES_FWD = [
    ("Embedding", "computation", "fwd", 0, {"shape": "vocab×d_model"}),
    ("LayerNorm0", "computation", "fwd", 1, {}),
    ("MHA_QKV_Proj", "computation", "fwd", 1, {"heads": "32", "d_head": "128"}),
    ("MHA_Reshape", "computation", "fwd", 2, {}),
    ("MHA_Attention", "computation", "fwd", 2, {}),
    ("MHA_OutProj", "computation", "fwd", 2, {}),
    ("TP_AllReduce_MHA", "communication", "fwd", 2, {"size": "≈24MB"}),
    ("LayerNorm1", "computation", "fwd", 1, {}),
    ("FFN_FC1", "computation", "fwd", 1, {"d_ffn": "11008"}),
    ("FFN_GELU", "computation", "fwd", 2, {}),
    ("FFN_FC2", "computation", "fwd", 2, {}),
    ("TP_AllReduce_FFN", "communication", "fwd", 2, {"size": "≈32MB"}),
]

_OP_NAMES_BWD = [
    ("Grad_FFN_FC2", "computation", "bwd", 1, {}),
    ("Grad_GELU", "computation", "bwd", 2, {}),
    ("Grad_FFN_FC1", "computation", "bwd", 2, {}),
    ("TP_ReduceScatter_FFN", "communication", "bwd", 2, {"size": "≈32MB"}),
    ("Grad_LayerNorm1", "computation", "bwd", 1, {}),
    ("Grad_MHA_OutProj", "computation", "bwd", 1, {}),
    ("Grad_Attention", "computation", "bwd", 2, {}),
    ("Grad_MHA_Reshape", "computation", "bwd", 2, {}),
    ("Grad_QKV_Proj", "computation", "bwd", 2, {}),
    ("TP_ReduceScatter_MHA", "communication", "bwd", 2, {"size": "≈24MB"}),
    ("Grad_LayerNorm0", "computation", "bwd", 1, {}),
    ("Grad_Embedding", "computation", "bwd", 0, {"shape": "vocab×d_model"}),
]

_OP_NAMES_OPT = [
    ("DP_AllReduce_Grads", "collective", "optimizer", 0, {"size": "≈134MB"}),
    ("Adam_Update", "computation", "optimizer", 0, {}),
]


def _generate_mock_operators(global_rank: int, tp: int, pp: int, num_layers_per_pp: int) -> tuple[list[OperatorTrace], TimelineSummary]:
    """Generate realistic mock operator traces for a training step on one device."""
    rng = random.Random(global_rank * 137 + tp * 7 + pp * 13)
    operators: list[OperatorTrace] = []
    current_time_us: float = 0.0
    total_flops: float = 0.0
    total_comm_bytes: float = 0.0
    comm_time_us: float = 0.0

    # Scale durations based on device type context (subtle variation per rank)
    dur_scale = 0.85 + rng.random() * 0.3  # 0.85-1.15

    # Embedding (first PP stage only, rank 0 of PP group)
    if pp == 1 or global_rank % pp == 0:
        for op_name, op_type, cat, depth, details in _OP_NAMES_FWD[:1]:
            dur = 1200 * dur_scale
            if depth == 0:
                dur = 800 * dur_scale
            ops = _OP_NAMES_FWD[0][3]  # depth for Embedding is 0
            operators.append(OperatorTrace(
                op_name=op_name, op_type=op_type, category=cat,
                start_us=current_time_us, duration_us=dur,
                flops=1.5e12 * dur_scale,
                parent_op="Forward" if depth > 0 else "",
                depth=0, details=dict(details),
            ))
            total_flops += 1.5e12 * dur_scale
            current_time_us += dur

    # Per-layer operators (forward + backward)
    for layer_idx in range(num_layers_per_pp):
        layer_prefix = f"TransformerBlock_{layer_idx}"

        # Forward pass
        for op_name, op_type, cat, depth, details in _OP_NAMES_FWD:
            if op_name == "Embedding":
                continue  # Already handled
            if depth == 1:
                parent = f"{layer_prefix}/Fwd"
            elif depth >= 2:
                parent = f"{layer_prefix}/Fwd/{op_name.split('_')[0]}"
            else:
                parent = f"{layer_prefix}/Fwd"

            if op_type == "communication":
                dur = rng.uniform(80, 200) * dur_scale
                comm_bytes = rng.uniform(8e6, 40e6)
                flops = 0
            else:
                dur = rng.uniform(200, 1800) * dur_scale
                comm_bytes = 0
                flops = rng.uniform(0.1e12, 3e12) * dur_scale

            operators.append(OperatorTrace(
                op_name=op_name, op_type=op_type, category=cat,
                start_us=current_time_us, duration_us=dur,
                flops=flops, comm_bytes=comm_bytes,
                parent_op=parent, depth=depth,
                details=dict(details) if details else {},
            ))
            current_time_us += dur
            total_flops += flops
            total_comm_bytes += comm_bytes
            if op_type == "communication":
                comm_time_us += dur

        # PP Send activations (if not last PP stage)
        if pp > 1 and global_rank % pp != pp - 1:
            dur = rng.uniform(50, 150) * dur_scale
            comm_bytes = rng.uniform(20e6, 60e6)
            operators.append(OperatorTrace(
                op_name="PP_Send_Activations", op_type="communication", category="fwd",
                start_us=current_time_us, duration_us=dur,
                comm_bytes=comm_bytes,
                parent_op=f"{layer_prefix}/Fwd", depth=0,
                details={"direction": "fwd", "size": f"{comm_bytes/1e6:.1f}MB"},
            ))
            current_time_us += dur
            total_comm_bytes += comm_bytes
            comm_time_us += dur

        # Backward pass
        for op_name, op_type, cat, depth, details in _OP_NAMES_BWD:
            if depth == 1:
                parent = f"{layer_prefix}/Bwd"
            elif depth >= 2:
                parent = f"{layer_prefix}/Bwd/{op_name.split('_')[0]}"
            else:
                parent = f"{layer_prefix}/Bwd"

            if op_type == "communication":
                dur = rng.uniform(80, 200) * dur_scale
                comm_bytes = rng.uniform(8e6, 40e6)
                flops = 0
            else:
                dur = rng.uniform(300, 2500) * dur_scale
                comm_bytes = 0
                flops = rng.uniform(0.15e12, 4e12) * dur_scale

            operators.append(OperatorTrace(
                op_name=op_name, op_type=op_type, category=cat,
                start_us=current_time_us, duration_us=dur,
                flops=flops, comm_bytes=comm_bytes,
                parent_op=parent, depth=depth,
                details=dict(details) if details else {},
            ))
            current_time_us += dur
            total_flops += flops
            total_comm_bytes += comm_bytes
            if op_type == "communication":
                comm_time_us += dur

        # PP Recv gradients (if not first PP stage)
        if pp > 1 and global_rank % pp != 0:
            dur = rng.uniform(50, 150) * dur_scale
            comm_bytes = rng.uniform(20e6, 60e6)
            operators.append(OperatorTrace(
                op_name="PP_Recv_Grads", op_type="communication", category="bwd",
                start_us=current_time_us, duration_us=dur,
                comm_bytes=comm_bytes,
                parent_op=f"{layer_prefix}/Bwd", depth=0,
                details={"direction": "bwd", "size": f"{comm_bytes/1e6:.1f}MB"},
            ))
            current_time_us += dur
            total_comm_bytes += comm_bytes
            comm_time_us += dur

    # Optimizer step
    for op_name, op_type, cat, depth, details in _OP_NAMES_OPT:
        if op_type == "collective" and tp <= 1 and pp <= 1:
            # AllReduce only for DP
            dur = rng.uniform(500, 2000) * dur_scale
            comm_bytes = rng.uniform(100e6, 200e6)
            flops = 0
        elif op_type == "collective":
            dur = rng.uniform(200, 1000) * dur_scale
            comm_bytes = rng.uniform(50e6, 150e6)
            flops = 0
        else:
            dur = rng.uniform(300, 1000) * dur_scale
            comm_bytes = 0
            flops = rng.uniform(0.5e12, 1.5e12) * dur_scale

        operators.append(OperatorTrace(
            op_name=op_name, op_type=op_type, category=cat,
            start_us=current_time_us, duration_us=dur,
            flops=flops, comm_bytes=comm_bytes,
            parent_op="Optimizer", depth=depth,
            details=dict(details) if details else {},
        ))
        current_time_us += dur
        total_flops += flops
        total_comm_bytes += comm_bytes
        if op_type in ("communication", "collective"):
            comm_time_us += dur

    total_time_ms = current_time_us / 1000.0
    compute_time_ms = (current_time_us - comm_time_us) / 1000.0
    comm_time_ms_v = comm_time_us / 1000.0

    timeline = TimelineSummary(
        total_time_ms=round(total_time_ms, 3),
        compute_time_ms=round(compute_time_ms, 3),
        comm_time_ms=round(comm_time_ms_v, 3),
        compute_pct=round(compute_time_ms / max(total_time_ms, 0.001) * 100, 1),
        comm_pct=round(comm_time_ms_v / max(total_time_ms, 0.001) * 100, 1),
        total_flops=round(total_flops, 1),
        total_comm_gb=round(total_comm_bytes / 1e9, 4),
    )

    return operators, timeline


@session_bp.route("/<session_id>/simulation/<side>/<int:global_rank>/detail", methods=["GET"])
def get_device_detail(session_id: str, side: str, global_rank: int):
    """Get per-device simulation detail with operator timeline (mock data for now)."""
    session = session_manager.get_session(session_id)
    if not session:
        return {"error": "session not found"}, 404

    if side not in ("original", "equivalent"):
        return {"error": "side must be 'original' or 'equivalent'"}, 400

    topo = session.original_topology if side == "original" else session.equivalent_topology
    task_id = session.original_task_id if side == "original" else session.equivalent_task_id

    if not topo:
        return {"error": f"no {side} topology found"}, 404

    # Find the node with matching global_rank
    node = None
    for n in topo.nodes:
        if n.global_rank == global_rank:
            node = n
            break

    if not node:
        return {"error": f"global_rank {global_rank} not found in {side} topology"}, 404

    # Default parallelism params
    tp = topo.tp_size
    pp = topo.pp_size
    dp = topo.dp_size

    # Calculate layers per PP stage
    total_layers = {"A2": 32, "A3": 48, "A5": 64}.get(topo.device_type.value if hasattr(topo.device_type, 'value') else str(topo.device_type), 32)
    num_layers_per_pp = max(1, math.ceil(total_layers / pp))

    operators, timeline = _generate_mock_operators(global_rank, tp, pp, num_layers_per_pp)

    detail = DeviceSimulationDetail(
        card_id=f"card_{global_rank}",
        global_rank=global_rank,
        task_id=task_id or "mock_task_id",
        topology_name=topo.name,
        device_type=topo.device_type.value if hasattr(topo.device_type, 'value') else str(topo.device_type),
        dp_rank=node.dp_rank,
        tp_rank=node.tp_rank,
        pp_rank=node.pp_rank,
        operators=operators,
        timeline=timeline,
    )

    return jsonify(detail.model_dump())
