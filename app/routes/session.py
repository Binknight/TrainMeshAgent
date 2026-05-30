"""REST endpoints for session management."""
import asyncio
import importlib
import json
import logging
import math
import random
import uuid
from flask import Blueprint, request, jsonify, Response

logger = logging.getLogger(__name__)

# Server boot identifier — changes on every restart, so the frontend can
# detect a server restart and discard cached sessions from the last run.
SERVER_BOOT_ID = str(uuid.uuid4())[:8]

from app.agent.session import session_manager
from app.agent.guardrails import validate_input_params
from app.config import config
from app.mcp.client import mcp_client
from app.models.schemas import (
    CardMetrics, CommDetail, DeviceType, DeviceSimulationDetail,
    HbmDetail, OperatorTrace, SessionState, TimelineSummary,
    SimulationResult, ComparisonReport, MeshTopology, TrainingModel,
    TopologyParams,
)
from app.skills.base import SkillContext, SkillResult
from app.skills.registry import registry

_estimator = importlib.import_module("app.skills.training-mesh-profiler-skill")

session_bp = Blueprint("session", __name__, url_prefix="/api/session")


@session_bp.route("", methods=["POST"])
def create_session():
    """Create a new agent session."""
    session = session_manager.create_session()
    result = session.model_dump()
    result["server_boot_id"] = SERVER_BOOT_ID
    return jsonify(result)


@session_bp.route("/summaries", methods=["GET"])
def list_session_summaries():
    """List session summaries for the history panel (lightweight, with topology titles)."""
    from app.dao import get_session_summaries
    return jsonify(get_session_summaries())


@session_bp.route("/<session_id>/simulation-params", methods=["GET"])
def get_simulation_params(session_id: str):
    """Get saved simulation params for a session."""
    from app.dao import get_simulation_params as dao_get_sim_params
    params = dao_get_sim_params(session_id, "original")
    if not params:
        return {"simulation_params": None}
    return {"simulation_params": params}


@session_bp.route("/<session_id>/simulation-params", methods=["POST"])
def save_simulation_params(session_id: str):
    """Save simulation params without triggering simulation."""
    session = session_manager.get_session(session_id)
    if not session:
        return {"error": "session not found"}, 404
    from app.models.schemas import SimulationParams
    from app.dao import save_simulation_params as dao_save_sim_params
    data = request.get_json(silent=True) or {}
    sim_params = SimulationParams(**data)
    session.simulation_params = sim_params
    d = sim_params.model_dump()
    dao_save_sim_params(session_id, "original", d)
    dao_save_sim_params(session_id, "equivalent", d)
    session_manager.save_session(session)
    return jsonify({"status": "saved", "simulation_params": d})


@session_bp.route("", methods=["GET"])
def list_sessions():
    """List all active sessions."""
    sessions = session_manager.list_sessions()
    return jsonify([s.model_dump() for s in sessions])


@session_bp.route("/<session_id>", methods=["GET"])
def get_session(session_id: str):
    """Get session state by ID."""
    session = session_manager.get_session(session_id)
    if not session:
        return {"error": "session not found"}, 404
    return jsonify(session.model_dump())


@session_bp.route("/<session_id>", methods=["DELETE"])
def delete_session(session_id: str):
    """Delete a session."""
    ok = session_manager.delete_session(session_id)
    if not ok:
        return {"error": "session not found"}, 404
    return {"status": "deleted", "session_id": session_id}


def _topo_with_model(topo, training_model, seq_len=None, batch_size=None, model_name=None, d_ffn=None):
    """Attach model config + runtime params onto a topology dict for MCP execute_task."""
    if topo is None:
        return None
    d = topo.model_dump()
    if training_model:
        d["num_layers"] = training_model.config.num_layers
        d["hidden_dim"] = training_model.config.d_model
        d["num_heads"] = training_model.config.num_heads
    if seq_len is not None:
        d["seq_len"] = seq_len
    if batch_size is not None:
        d["batch_size"] = batch_size
    if model_name is not None:
        d["model_name"] = model_name
    if d_ffn is not None:
        d["d_ffn"] = d_ffn
    elif training_model and training_model.config.d_ffn:
        d["d_ffn"] = training_model.config.d_ffn
    return d


@session_bp.route("/<session_id>/topology", methods=["GET"])
def get_topology(session_id: str):
    """Get the mesh topology data for a session."""
    session = session_manager.get_session(session_id)
    if not session:
        return {"error": "session not found"}, 404

    return jsonify({
        "session_id": session_id,
        "server_boot_id": SERVER_BOOT_ID,
        "original_topology": _topo_with_model(
            session.original_topology, session.original_training_model,
            seq_len=session.original_seq_len,
            batch_size=session.original_batch_size,
            model_name=session.original_model_name,
            d_ffn=session.original_dff,
        ),
        "equivalent_topology": _topo_with_model(
            session.equivalent_topology, session.equivalent_training_model,
            seq_len=session.equivalent_seq_len,
            batch_size=session.equivalent_batch_size,
            model_name=session.original_model_name,  # model is the same, just different topo
            d_ffn=session.equivalent_dff,
        ),
        "original_training_model": session.original_training_model.model_dump() if session.original_training_model else None,
        "equivalent_training_model": session.equivalent_training_model.model_dump() if session.equivalent_training_model else None,
        "original_simulation": session.original_simulation is not None,
        "equivalent_simulation": session.equivalent_simulation is not None,
        "comparison_report": session.comparison_report is not None,
        "simulation_params": session.simulation_params.model_dump() if session.simulation_params else None,
        "step": session.step,
        "messages": session.history,
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

    cfg = _estimator._MODEL_CONFIG[device_type]
    L = int(data.get("num_layers", cfg["num_layers"]))
    H = int(data.get("hidden_dim", cfg["hidden_dim"]))
    S = int(data.get("seq_len", _estimator._SEQ_LEN))
    B = int(data.get("total_batch", _estimator._TOTAL_BATCH))
    dff_val = int(data.get("d_ffn", 14336))

    flops = _estimator._estimate_flops(L, H, S, B, dff_val, dp, tp, pp)
    hbm = _estimator._estimate_hbm_gb(L, H, dff_val, tp, pp)
    dp_comm = _estimator._estimate_dp_comm_gb(L, H, dff_val, dp, tp, pp)
    tp_comm = _estimator._estimate_tp_comm_gb(L, H, S, B, pp)
    pp_comm = _estimator._estimate_pp_comm_mb(H, S, B)

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


def _run_simulation_for_topology(topo, training_model, task_id_in: str | None, label: str, sim_params: dict | None = None, seq_len=None, batch_size=None, model_name=None) -> tuple[str | None, SimulationResult | None]:
    """Submit MCP task for a single topology. Returns (task_id, SimulationResult or None if not ready)."""
    if not topo:
        return task_id_in, None

    # Submit MCP task (fire-and-forget)
    task_id = task_id_in
    if not task_id:
        topo_payload = _topo_with_model(topo, training_model, seq_len=seq_len, batch_size=batch_size, model_name=model_name) or topo.model_dump()
        task_id = mcp_client.execute_task(topo_payload, params=sim_params)
        if not task_id:
            raise RuntimeError(f"MCP execute_task returned empty task_id for {label}")

    # Try to get card metrics from MCP (may be empty if simulation not yet complete)
    try:
        card_details = mcp_client.get_card_details(task_id)
        logger.info(f"[run_simulation] card_detail for {label} (task_id={task_id}): "
                    f"type={type(card_details).__name__}, len={len(card_details) if hasattr(card_details, '__len__') else 'N/A'}, "
                    f"truthy={bool(card_details)}, raw_keys={list(card_details[0].keys()) if card_details else 'EMPTY'}")
        if card_details:
            cards = []
            for detail in card_details:
                cards.append(CardMetrics(
                    card_id=detail.get("card_id", ""),
                    global_rank=detail.get("global_rank", 0),
                    flops_per_card=detail.get("flops_per_card", 0),
                    hbm_gb=detail.get("hbm_gb", 0),
                    tp_comm_gb_per_micro=detail.get("tp_comm_gb_per_micro", 0),
                    pp_comm_mb_per_micro=detail.get("pp_comm_mb_per_micro", 0),
                    dp_comm_gb_per_step=detail.get("dp_comm_gb_per_step", 0),
                ))
            # Log first card as sample
            if cards:
                c0 = cards[0]
                logger.info(f"[run_simulation] card_detail for {label}: {len(cards)} cards. "
                            f"sample[0]: flops={c0.flops_per_card}, hbm={c0.hbm_gb}, "
                            f"tp={c0.tp_comm_gb_per_micro}, pp={c0.pp_comm_mb_per_micro}, dp={c0.dp_comm_gb_per_step}")
            device_type = topo.device_type if isinstance(topo.device_type, DeviceType) else DeviceType(topo.device_type.value)
            result = SimulationResult(
                topology_name=topo.name,
                device_type=device_type,
                total_nodes=topo.dp_size * topo.tp_size * topo.pp_size,
                cards=cards,
            )
            return task_id, result
        else:
            logger.warning(f"[run_simulation] card_detail for {label}: falsy result (empty list or None), skipping")
    except Exception as exc:
        logger.warning(f"[run_simulation] MCP card_detail failed for {label}: {exc}", exc_info=True)

    return task_id, None


def _build_comparison(original: SimulationResult, equivalent: SimulationResult) -> ComparisonReport:
    eps = 1e-9

    def _diff_pct(ov, ev):
        return round(abs(ov - ev) / max(abs(ov), eps) * 100, 2)

    def _per_card(cards: list[CardMetrics], attr: str) -> float:
        return sum(getattr(c, attr) for c in cards) / max(len(cards), 1)

    flops_diff = _diff_pct(_per_card(original.cards, "flops_per_card"), _per_card(equivalent.cards, "flops_per_card"))
    hbm_diff = _diff_pct(_per_card(original.cards, "hbm_gb"), _per_card(equivalent.cards, "hbm_gb"))
    tp_diff = _diff_pct(_per_card(original.cards, "tp_comm_gb_per_micro"), _per_card(equivalent.cards, "tp_comm_gb_per_micro"))
    pp_diff = _diff_pct(_per_card(original.cards, "pp_comm_mb_per_micro"), _per_card(equivalent.cards, "pp_comm_mb_per_micro"))
    dp_diff = _diff_pct(_per_card(original.cards, "dp_comm_gb_per_step"), _per_card(equivalent.cards, "dp_comm_gb_per_step"))

    tolerance = 5.0
    is_eq = all(d <= tolerance for d in [flops_diff, hbm_diff, tp_diff, pp_diff, dp_diff])

    return ComparisonReport(
        original=original,
        equivalent=equivalent,
        flops_diff_pct=flops_diff,
        hbm_diff_pct=hbm_diff,
        tp_comm_diff_pct=tp_diff,
        pp_comm_diff_pct=pp_diff,
        dp_comm_diff_pct=dp_diff,
        is_equivalent=is_eq,
        error_tolerance_pct=tolerance,
        details={
            "conclusion": "✅ 等效验证通过" if is_eq else "❌ 等效验证不通过",
            "max_diff_pct": round(max(flops_diff, hbm_diff, tp_diff, pp_diff, dp_diff), 2),
        }
    )


@session_bp.route("/<session_id>/run-simulation", methods=["POST"])
def run_simulation(session_id: str):
    """Directly run simulations for both topologies and return results (no agent/SSE)."""
    session = session_manager.get_session(session_id)
    if not session:
        return {"error": "session not found"}, 404

    # Accept optional simulation params from request body
    from app.models.schemas import SimulationParams
    body = request.get_json(silent=True) or {}
    sim_params_data = body.get("simulation_params", None)
    if sim_params_data:
        session.simulation_params = SimulationParams(**sim_params_data)
    elif session.simulation_params is None:
        session.simulation_params = SimulationParams()

    sim_params_dict = session.simulation_params.model_dump()

    is_retry = bool(session.original_task_id or session.equivalent_task_id)
    logger.info(f"[run_simulation] session={session_id}, step={session.step}, "
                f"orig_task={session.original_task_id}, eq_task={session.equivalent_task_id}, "
                f"is_retry={is_retry}")
    session.history.append({"role": "system", "content": "📊 开始仿真任务..."})
    results = {}

    try:
        # ── Original topology ──
        orig_tid, orig_sim = _run_simulation_for_topology(
            session.original_topology, session.original_training_model,
            session.original_task_id, "original", sim_params_dict,
            seq_len=session.original_seq_len,
            batch_size=session.original_batch_size,
            model_name=session.original_model_name,
        )
        session.original_task_id = orig_tid or session.original_task_id
        if orig_sim:
            session.original_simulation = orig_sim
            results["original"] = orig_sim.model_dump()

        # ── Equivalent topology ──
        eq_tid, eq_sim = _run_simulation_for_topology(
            session.equivalent_topology, session.equivalent_training_model,
            session.equivalent_task_id, "equivalent", sim_params_dict,
            seq_len=session.equivalent_seq_len,
            batch_size=session.equivalent_batch_size,
            model_name=session.original_model_name,
        )
        session.equivalent_task_id = eq_tid or session.equivalent_task_id
        if eq_sim:
            session.equivalent_simulation = eq_sim
            results["equivalent"] = eq_sim.model_dump()

        # ── Comparison ──
        report = None
        if session.original_simulation and session.equivalent_simulation:
            report = _build_comparison(session.original_simulation, session.equivalent_simulation)
            session.comparison_report = report
            session.step = "completed"
            results["comparison"] = report.model_dump(exclude={"original", "equivalent"})
            if report.is_equivalent:
                session.history.append({"role": "system", "content": "✅ 仿真验证已通过，等效性对比一致"})
            else:
                session.history.append({"role": "system", "content": "⚠️ 仿真完成，等效性对比存在差异，请检查"})
        else:
            session.step = "simulating"
            session.history.append({"role": "system", "content": "📊 仿真任务已提交，任务ID: " + str(session.original_task_id or "")})

    except Exception as exc:
        logger.error(f"[run_simulation] Failed: {exc}")
        session.step = "failed"
        session.history.append({"role": "system", "content": f"❌ 仿真任务失败: {exc}"})
        session_manager.save_session(session)
        return {"error": str(exc), "step": "failed"}, 502

    session_manager.save_session(session)

    return jsonify({
        "session_id": session_id,
        "original_task_id": session.original_task_id,
        "equivalent_task_id": session.equivalent_task_id,
        "results": results,
        "step": session.step,
    })


# ── Mock operator trace generation (aligned with CSV headers + MCP computed fields) ──

# Operator templates: (operator_name, comm_type, stage_suffix, [op-specific fields])
# stage_suffix is combined with layer index at generation time, e.g. "forward/layer0"
_FWD_OPS = [
    ("embedding",        "computation",   "forward",  dict(data_shape="[B,S,vocab] → [B,S,d_model]", data_type="bf16", algo_name="lookup")),
    ("qkv_projection",   "computation",   "forward",  dict(data_shape="[B,S,d_model] → [B,S,3*d_model]", data_type="bf16", algo_name="linear")),
    ("mha_reshape",      "computation",   "forward",  dict(data_shape="[B,S,3*d_model] → [3,B,heads,S,d_head]", data_type="bf16")),
    ("mha_attention",    "computation",   "forward",  dict(data_shape="[B,heads,S,S]", data_type="bf16", algo_name="softmax")),
    ("mha_outproj",      "computation",   "forward",  dict(data_shape="[B,S,d_model] → [B,S,d_model]", data_type="bf16", algo_name="linear")),
    ("all_reduce",       "all_reduce",    "forward",  dict(comm_group="tp_group", data_shape="[B,S,d_model]", data_type="bf16", additional="≈24MB")),
    ("ffn_fc1",          "computation",   "forward",  dict(data_shape="[B,S,d_model] → [B,S,d_ffn]", data_type="bf16", algo_name="linear")),
    ("ffn_gelu",         "computation",   "forward",  dict(data_shape="[B,S,d_ffn]", data_type="bf16", algo_name="GELU")),
    ("ffn_fc2",          "computation",   "forward",  dict(data_shape="[B,S,d_ffn] → [B,S,d_model]", data_type="bf16", algo_name="linear")),
    ("all_reduce",       "all_reduce",    "forward",  dict(comm_group="tp_group", data_shape="[B,S,d_model]", data_type="bf16", additional="≈32MB")),
]

_BWD_OPS = [
    ("grad_ffn_fc2",        "computation",    "backward", dict(data_shape="[B,S,d_model] → [B,S,d_ffn]", data_type="bf16")),
    ("grad_gelu",           "computation",    "backward", dict(data_shape="[B,S,d_ffn]", data_type="bf16")),
    ("grad_ffn_fc1",        "computation",    "backward", dict(data_shape="[B,S,d_ffn] → [B,S,d_model]", data_type="bf16", algo_name="linear")),
    ("reduce_scatter",      "reduce_scatter", "backward", dict(comm_group="tp_group", data_shape="[B,S,d_model]", data_type="bf16", additional="≈32MB")),
    ("grad_mha_outproj",    "computation",    "backward", dict(data_shape="[B,S,d_model] → [B,S,d_model]", data_type="bf16", algo_name="linear")),
    ("grad_attention",      "computation",    "backward", dict(data_shape="[B,heads,S,S]", data_type="bf16")),
    ("grad_mha_reshape",    "computation",    "backward", dict(data_shape="[3,B,heads,S,d_head] → [B,S,3*d_model]", data_type="bf16")),
    ("grad_qkv_projection", "computation",    "backward", dict(data_shape="[B,S,3*d_model] → [B,S,d_model]", data_type="bf16", algo_name="linear")),
    ("reduce_scatter",      "reduce_scatter", "backward", dict(comm_group="tp_group", data_shape="[B,S,d_model]", data_type="bf16", additional="≈24MB")),
    ("grad_embedding",      "computation",    "backward", dict(data_shape="[B,S,d_model] → [B,S,vocab]", data_type="bf16")),
]

_OPT_OPS = [
    ("all_reduce",  "all_reduce",  "optimizer", dict(comm_group="dp_group", data_shape="≈134MB per param", additional="≈134MB")),
    ("adam_update", "computation", "optimizer", dict(data_shape="per-param", data_type="fp32", algo_name="Adam")),
]


def _generate_mock_operators(global_rank: int, tp: int, pp: int, num_layers_per_pp: int) -> tuple[list[OperatorTrace], TimelineSummary]:
    """Generate mock operator traces aligned with CSV schema + MCP computed fields."""
    rng = random.Random(global_rank * 137 + tp * 7 + pp * 13)
    operators: list[OperatorTrace] = []
    current_time_us: float = 0.0
    total_flops: float = 0.0
    total_comm_bytes: float = 0.0
    comm_time_us: float = 0.0
    op_index: int = 0

    dur_scale = 0.85 + rng.random() * 0.3
    dp = max(1, tp * pp)  # approximate DP

    def _make_op(name, comm_type, stage, extra, start_us, dur, flops_val, msg_bytes):
        nonlocal op_index
        is_comm = comm_type not in ("computation",)
        op = OperatorTrace(
            index=op_index,
            operator_name=name,
            comm_type=comm_type,
            stage=stage,
            start_time=start_us,
            end_time=start_us + dur,
            duration=dur,
            elapsed_time=dur,
            single_flops=flops_val if flops_val > 0 else None,
            msg_size=msg_bytes if msg_bytes > 0 else None,
            comm_group=extra.get("comm_group") if is_comm else None,
            comm_group_size=(tp if extra.get("comm_group") == "tp_group" else dp if extra.get("comm_group") == "dp_group" else None) if is_comm else None,
            data_shape=extra.get("data_shape"),
            data_type=extra.get("data_type"),
            algo_name=extra.get("algo_name"),
            additional=extra.get("additional"),
            nonblock=0 if comm_type in ("all_reduce",) else 1,
            dst=None,
            src=None,
            wait_n=None,
        )
        op_index += 1
        return op

    # Embedding (first PP stage only)
    if pp == 1 or global_rank % pp == 0:
        emb_op = _FWD_OPS[0]
        dur = 800 * dur_scale
        flops = 1.5e12 * dur_scale
        operators.append(_make_op(
            emb_op[0], emb_op[1], f"{emb_op[2]}/layer0", emb_op[3],
            current_time_us, dur, flops, 0,
        ))
        total_flops += flops
        current_time_us += dur

    # Per-layer operators
    for layer_idx in range(num_layers_per_pp):
        stage_fwd = f"forward/layer{layer_idx}"
        stage_bwd = f"backward/layer{layer_idx}"

        # Forward pass
        for name, comm_type, _suffix, extra in _FWD_OPS:
            if name == "embedding":
                continue
            is_comm = comm_type != "computation"
            if is_comm:
                dur = rng.uniform(80, 200) * dur_scale
                msg_bytes = rng.uniform(8e6, 40e6)
                flops = 0
            else:
                dur = rng.uniform(200, 1800) * dur_scale
                msg_bytes = 0
                flops = rng.uniform(0.1e12, 3e12) * dur_scale

            operators.append(_make_op(name, comm_type, stage_fwd, extra, current_time_us, dur, flops, msg_bytes))
            current_time_us += dur
            total_flops += flops
            if is_comm:
                total_comm_bytes += msg_bytes
                comm_time_us += dur

        # PP Send (if not last PP stage)
        if pp > 1 and global_rank % pp != pp - 1:
            dur = rng.uniform(50, 150) * dur_scale
            msg_bytes = rng.uniform(20e6, 60e6)
            extra_send = {"data_shape": "[B,S,d_model]", "data_type": "bf16", "comm_group": "pp_group", "additional": f"{msg_bytes/1e6:.1f}MB"}
            operators.append(_make_op("send", "send", stage_fwd, extra_send, current_time_us, dur, 0, msg_bytes))
            current_time_us += dur
            total_comm_bytes += msg_bytes
            comm_time_us += dur

        # Backward pass
        for name, comm_type, _suffix, extra in _BWD_OPS:
            is_comm = comm_type != "computation"
            if is_comm:
                dur = rng.uniform(80, 200) * dur_scale
                msg_bytes = rng.uniform(8e6, 40e6)
                flops = 0
            else:
                dur = rng.uniform(300, 2500) * dur_scale
                msg_bytes = 0
                flops = rng.uniform(0.15e12, 4e12) * dur_scale

            operators.append(_make_op(name, comm_type, stage_bwd, extra, current_time_us, dur, flops, msg_bytes))
            current_time_us += dur
            total_flops += flops
            if is_comm:
                total_comm_bytes += msg_bytes
                comm_time_us += dur

        # PP Recv (if not first PP stage)
        if pp > 1 and global_rank % pp != 0:
            dur = rng.uniform(50, 150) * dur_scale
            msg_bytes = rng.uniform(20e6, 60e6)
            extra_recv = {"data_shape": "[B,S,d_model]", "data_type": "bf16", "comm_group": "pp_group", "additional": f"{msg_bytes/1e6:.1f}MB"}
            operators.append(_make_op("recv", "recv", stage_bwd, extra_recv, current_time_us, dur, 0, msg_bytes))
            current_time_us += dur
            total_comm_bytes += msg_bytes
            comm_time_us += dur

    # Optimizer step
    for name, comm_type, _suffix, extra in _OPT_OPS:
        is_comm = comm_type != "computation"
        if comm_type == "all_reduce":
            dur = rng.uniform(200, 2000) * dur_scale
            msg_bytes = rng.uniform(50e6, 200e6)
            flops = 0
        else:
            dur = rng.uniform(300, 1000) * dur_scale
            msg_bytes = 0
            flops = rng.uniform(0.5e12, 1.5e12) * dur_scale

        operators.append(_make_op(name, comm_type, "optimizer", extra, current_time_us, dur, flops, msg_bytes))
        current_time_us += dur
        total_flops += flops
        if is_comm:
            total_comm_bytes += msg_bytes
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

    mcp_result = mcp_client.get_device_detail(task_id, global_rank)
    raw_ops = mcp_result.get("operators", [])
    operators = [OperatorTrace(**op) if isinstance(op, dict) else op for op in raw_ops]
    timeline_data = mcp_result.get("timeline")
    timeline = TimelineSummary(**timeline_data) if timeline_data and isinstance(timeline_data, dict) else None

    detail = DeviceSimulationDetail(
        card_id=mcp_result.get("card_id", f"card_{global_rank}"),
        global_rank=mcp_result.get("global_rank", global_rank),
        task_id=mcp_result.get("task_id", task_id),
        topology_name=mcp_result.get("topology_name", topo.name),
        device_type=mcp_result.get("device_type", str(topo.device_type.value if hasattr(topo.device_type, 'value') else topo.device_type)),
        dp_rank=mcp_result.get("dp_rank", node.dp_rank),
        tp_rank=mcp_result.get("tp_rank", node.tp_rank),
        pp_rank=mcp_result.get("pp_rank", node.pp_rank),
        operators=operators,
        timeline=timeline,
    )
    return jsonify(detail.model_dump())


# ── Metric detail mock data generators ──

def _generate_mock_hbm_detail(global_rank: int) -> HbmDetail:
    """Generate mock HBM breakdown: weights, gradients, optimizer, activations."""
    rng = random.Random(global_rank * 137 + 42)
    weights = round(rng.uniform(1.0, 3.0), 2)
    gradients = round(rng.uniform(1.0, 3.0), 2)
    optimizer = round(rng.uniform(2.0, 6.0), 2)
    activations = round(rng.uniform(5.0, 15.0), 2)
    return HbmDetail(
        global_rank=global_rank,
        weights_gb=weights,
        gradients_gb=gradients,
        optimizer_gb=optimizer,
        activations_gb=activations,
        total_hbm_gb=round(weights + gradients + optimizer + activations, 2),
    )


def _generate_mock_comm_detail(global_rank: int, comm_type: str, tp: int, pp: int, dp: int) -> CommDetail:
    """Generate mock communication detail for TP/PP/DP."""
    rng = random.Random(global_rank * 137 + hash(comm_type) * 31 + tp * 7 + pp * 13 + dp * 3)
    if comm_type == "tp":
        comm_cards = tp
        comm_count = rng.randint(24, 96)
        size_per_time = round(rng.uniform(0.01, 0.05), 4)
    elif comm_type == "pp":
        comm_cards = pp
        comm_count = rng.randint(2, 16)
        size_per_time = round(rng.uniform(0.005, 0.03), 4)
    else:  # dp
        comm_cards = dp
        comm_count = rng.randint(1, 8)
        size_per_time = round(rng.uniform(0.05, 0.2), 4)
    return CommDetail(
        global_rank=global_rank,
        comm_type=comm_type,
        comm_count=comm_count,
        comm_cards=comm_cards,
        comm_size_per_time_gb=size_per_time,
        total_comm_gb=round(comm_count * size_per_time, 4),
    )


# ── Metric detail endpoints ──

@session_bp.route("/<session_id>/simulation/<side>/<int:global_rank>/hbm-detail", methods=["GET"])
def get_hbm_detail(session_id: str, side: str, global_rank: int):
    """Get HBM usage breakdown for a device."""
    session = session_manager.get_session(session_id)
    if not session:
        return {"error": "session not found"}, 404
    if side not in ("original", "equivalent"):
        return {"error": "side must be 'original' or 'equivalent'"}, 400

    task_id = session.original_task_id if side == "original" else session.equivalent_task_id
    mcp_result = mcp_client.get_hbm_detail(task_id, global_rank)
    return jsonify(HbmDetail(**mcp_result).model_dump())


@session_bp.route("/<session_id>/simulation/<side>/<int:global_rank>/tp-comm-detail", methods=["GET"])
def get_tp_comm_detail(session_id: str, side: str, global_rank: int):
    """Get TP communication detail for a device."""
    session = session_manager.get_session(session_id)
    if not session:
        return {"error": "session not found"}, 404
    if side not in ("original", "equivalent"):
        return {"error": "side must be 'original' or 'equivalent'"}, 400
    topo = session.original_topology if side == "original" else session.equivalent_topology
    task_id = session.original_task_id if side == "original" else session.equivalent_task_id
    mcp_result = mcp_client.get_comm_detail(task_id, global_rank, "tp")
    return jsonify(CommDetail(**mcp_result).model_dump())


@session_bp.route("/<session_id>/simulation/<side>/<int:global_rank>/pp-comm-detail", methods=["GET"])
def get_pp_comm_detail(session_id: str, side: str, global_rank: int):
    """Get PP communication detail for a device."""
    session = session_manager.get_session(session_id)
    if not session:
        return {"error": "session not found"}, 404
    if side not in ("original", "equivalent"):
        return {"error": "side must be 'original' or 'equivalent'"}, 400
    topo = session.original_topology if side == "original" else session.equivalent_topology
    task_id = session.original_task_id if side == "original" else session.equivalent_task_id
    mcp_result = mcp_client.get_comm_detail(task_id, global_rank, "pp")
    return jsonify(CommDetail(**mcp_result).model_dump())


@session_bp.route("/<session_id>/simulation/<side>/<int:global_rank>/dp-comm-detail", methods=["GET"])
def get_dp_comm_detail(session_id: str, side: str, global_rank: int):
    """Get DP communication detail for a device."""
    session = session_manager.get_session(session_id)
    if not session:
        return {"error": "session not found"}, 404
    if side not in ("original", "equivalent"):
        return {"error": "side must be 'original' or 'equivalent'"}, 400
    topo = session.original_topology if side == "original" else session.equivalent_topology
    task_id = session.original_task_id if side == "original" else session.equivalent_task_id
    mcp_result = mcp_client.get_comm_detail(task_id, global_rank, "dp")
    return jsonify(CommDetail(**mcp_result).model_dump())


# ══════════════════════════════════════════════════════════════════════════════
# Workflow API — direct skill calls (API mode, bypasses Agent)
# ══════════════════════════════════════════════════════════════════════════════

@session_bp.route("/<session_id>/workflow/step1", methods=["POST"])
def workflow_step1(session_id: str):
    """Step 1: Receive original params → generate original mesh + model. Returns JSON."""
    session = session_manager.get_session(session_id)
    if not session:
        return {"error": "session not found"}, 404

    data = request.get_json(silent=True) or {}
    device_type_str = data.get("device_type", "A3")
    dp = data.get("dp", 8)
    tp = data.get("tp", 16)
    pp = data.get("pp", 8)
    model_name = data.get("model_name", "Qwen3-32B")
    L = data.get("L", 64)
    H = data.get("H", 4096)
    A = data.get("A", 32)
    S = data.get("S", 2048)
    B = data.get("B", 32)
    dff = data.get("dff", 14336)
    strategy = data.get("strategy", "min_equiv")

    # 1. Guardrail validation (silent — no workflow node)
    validation = validate_input_params(
        device_type=device_type_str,
        dp=dp, tp=tp, pp=pp,
    )
    if not validation.passed:
        return {
            "error": "guardrail_failed",
            "message": "; ".join(validation.errors),
            "warnings": validation.warnings,
        }, 400

    context = SkillContext(session=session, mcp_client=mcp_client, config=config)

    # 2. Generate original mesh topology
    mesh_args = {
        "name": "原始组网 (" + device_type_str + " DP" + str(dp) + " TP" + str(tp) + " PP" + str(pp) + ")",
        "device_type": device_type_str,
        "dp": dp, "tp": tp, "pp": pp,
    }
    mesh_result: SkillResult = registry.execute_tool("training-mesh-gen-skill", mesh_args, context)
    if not mesh_result.success:
        return {"error": "mesh_gen_failed", "message": mesh_result.error}, 500

    orig_mesh = mesh_result.data
    session.original_topology = orig_mesh
    session.original_params = TopologyParams(
        device_type=DeviceType(device_type_str) if device_type_str in [d.value for d in DeviceType] else DeviceType.A3,
        dp=dp, tp=tp, pp=pp,
    )

    # 3. Generate original model structure
    model_args = {
        "num_layers": L,
        "d_model": H,
        "num_heads": A,
        "d_ffn": dff,
        "pp": pp,
        "is_equivalent": False,
    }
    model_result: SkillResult = registry.execute_tool("training-model-gen-skill", model_args, context)
    if not model_result.success:
        return {"error": "model_gen_failed", "message": model_result.error}, 500

    orig_model = model_result.data
    session.original_training_model = orig_model

    # Store equivalent params for later steps (based on strategy)
    eq_pp = max(1, min(pp - 1, 3)) if pp > 3 else pp
    eq_dp = max(1, dp // 4) if dp > 1 else 1
    eq_L = L if pp <= 3 else (L // pp) * 3
    eq_B = max(1, int(B * eq_dp / dp)) if dp > 1 else B
    session.equivalent_params = TopologyParams(
        device_type=DeviceType(device_type_str) if device_type_str in [d.value for d in DeviceType] else DeviceType.A3,
        dp=eq_dp, tp=tp, pp=eq_pp,
    )
    # Store model params for equivalent model
    session._equiv_model_params = {
        "L": eq_L, "H": H, "A": A, "S": S, "B": eq_B, "dff": dff,
        "strategy": strategy,
        "L_orig": L,  # original layer count for SSE formula display
        "B_orig": B,  # original batch size for SSE formula display
    }
    session.original_dff = dff
    session.original_batch_size = B
    session.equivalent_batch_size = eq_B
    session.equivalent_dff = dff  # dff unchanged between original and equivalent

    session.step = "params_collected"
    session_manager.save_session(session)

    # Build response with topology/model as dicts (enrich mesh with model + runtime params)
    orig_model_dict = orig_model.model_dump() if hasattr(orig_model, "model_dump") else orig_model
    orig_model_dict["seq_len"] = S
    orig_model_dict["batch_size"] = B
    return jsonify({
        "original_mesh": _topo_with_model(
            orig_mesh, orig_model,
            seq_len=S, batch_size=B, model_name=model_name, d_ffn=dff,
        ),
        "original_model": orig_model_dict,
        "equivalent_params": session.equivalent_params.model_dump() if hasattr(session.equivalent_params, "model_dump") else session.equivalent_params,
        "step": session.step,
    })


@session_bp.route("/<session_id>/workflow/step2", methods=["POST"])
def workflow_step2(session_id: str):
    """Step 2: Trigger equivalent calculation. Returns task_id for SSE streaming."""
    session = session_manager.get_session(session_id)
    if not session:
        return {"error": "session not found"}, 404

    if not session.original_topology:
        return {"error": "original topology not found — run step1 first"}, 400

    session.step = "equiv_calculating"
    session_manager.save_session(session)

    return jsonify({
        "task_id": session_id,
        "status": "started",
    })


@session_bp.route("/<session_id>/workflow/step2/stream", methods=["GET"])
def workflow_step2_stream(session_id: str):
    """Step 2 SSE stream: push formula lines one by one for dynamic loading."""
    session = session_manager.get_session(session_id)
    if not session:
        return Response("data: " + json.dumps({"type": "error", "message": "session not found"}) + "\n\n",
                        mimetype="text/event-stream")

    orig = session.original_topology
    eq_params = session.equivalent_params if hasattr(session, "equivalent_params") else None
    model_meta = getattr(session, "_equiv_model_params", {}) or {}

    orig_dp = orig.dp_size if orig else 8
    tp = orig.tp_size if orig else 8
    pp = orig.pp_size if orig else 8
    eq_pp = eq_params.pp if eq_params else 3
    eq_tp = eq_params.tp if eq_params else 16
    eq_dp = eq_params.dp if eq_params else 2
    eq_L = model_meta.get("L", 24)
    H_val = model_meta.get("H", 4096)
    A_val = model_meta.get("A", 32)
    S_val = model_meta.get("S", 2048)
    B_val = model_meta.get("B", 32)
    B_orig = model_meta.get("B_orig", B_val)
    dff_val = model_meta.get("dff", 14336)
    L_orig = model_meta.get("L_orig", eq_L)  # fallback to eq_L if not stored
    strategy = model_meta.get("strategy", "min_equiv")
    strategy_label = "最小集群等效" if strategy == "min_equiv" else ("单卡极限等效" if strategy == "single_card_extreme" else strategy)

    # Pre-compute numeric values for richer display
    npu_orig = orig_dp * tp * pp
    npu_eq = eq_dp * eq_tp * eq_pp
    comp_ratio = round(npu_orig / npu_eq, 1) if npu_eq else 0

    def generate():
        # ═══ Phase 1: 策略加载 ═══
        lines_strategy = [
            f"▸ 等效策略: {strategy_label} ({strategy})",
            f"  原始组网  {orig.device_type.value if orig and orig.device_type else 'A3'}  DP={orig_dp}  TP={tp}  PP={pp}  →  {npu_orig} NPU",
            f"  等效组网  {eq_params.device_type.value if eq_params and eq_params.device_type else 'A3'}  DP={eq_dp}  TP={eq_tp}  PP={eq_pp}  →  {npu_eq} NPU",
            f"  模型配置  L={L_orig}  H={H_val}  A={A_val}  dff={dff_val}  S={S_val}  B={B_orig}  →  B_eq={B_val}",
            f"  NPU 压缩比  {npu_orig} : {npu_eq}  ≈  {comp_ratio} : 1",
        ]
        for line in lines_strategy:
            yield f"data: {json.dumps({'type': 'equiv_formula_line', 'section': 'strategy', 'line': line})}\n\n"
            import time; time.sleep(0.45)
        yield f"data: {json.dumps({'type': 'equiv_formula_line', 'section': 'strategy', 'section_done': True, 'line': ''})}\n\n"

        # ═══ Phase 2: 指标分析 ═══
        flops_per_card = _estimator._estimate_flops(L_orig, H_val, S_val, B_orig, dff_val, orig_dp, tp, pp)
        flops_str = f"{flops_per_card / 1e15:.2f} × 10¹⁵" if flops_per_card >= 1e15 else f"{flops_per_card / 1e12:.2f} × 10¹²"

        hbm_gb = _estimator._estimate_hbm_gb(L_orig, H_val, dff_val, tp, pp)
        tp_comm = _estimator._estimate_tp_comm_gb(L_orig, H_val, S_val, B_orig, pp)
        dp_comm = _estimator._estimate_dp_comm_gb(L_orig, H_val, dff_val, orig_dp, tp, pp)
        pp_comm = _estimator._estimate_pp_comm_mb(H_val, S_val, B_orig)

        lines_metrics = [
            f"▸ 单卡计算量 (FLOPs)",
            f"  FLOPs = (6·B·S·L·H/(DP·PP·TP)) × (4·H + 3·dff + 2·S)",
            f"  = (6×{B_orig}×{S_val}×{L_orig}×{H_val}/({orig_dp}×{pp}×{tp})) × (4×{H_val} + 3×{dff_val} + 2×{S_val})",
            f"  ≈ {flops_str} FLOPs",
            f"▸ 显存占用 (HBM)",
            f"  HBM = L/PP × ((4·H² + 3·H·dff)/TP + 2·H) / 1e9",
            f"  = {L_orig}/{pp} × ((4×{H_val}² + 3×{H_val}×{dff_val})/{tp} + 2×{H_val}) / 1e9",
            f"  ≈ {hbm_gb:.1f} GB",
            f"▸ 通信流量 (GB / step)",
            f"  TP 通信 = L/PP * 15 * B * S * H / 1e9",
            f"  = {L_orig}/{pp} * 15 * {B_val} * {S_val} * {H_val} / 1e9",
            f"  ≈ {tp_comm:.2f} GB/micro-step  (TP 全规约)",
            f"  DP 通信 = 2*(DP-1)/DP * 4 * L/PP * (4*H^2/TP + 3*H*dff/TP) / 1e9",
            f"  = 2*({orig_dp}-1)/{orig_dp} * 4 * {L_orig}/{pp} * (4*{H_val}^2/{tp} + 3*{H_val}*{dff_val}/{tp}) / 1e9",
            f"  ≈ {dp_comm:.2f} GB/step",
            f"  PP 通信 = 4·B·S·H / 1e6  (激活值 send/recv)",
            f"  = 4 × {B_val} × {S_val} × {H_val} / 1e6",
            f"  ≈ {pp_comm:.2f} MB/micro-step  (per PP boundary)",
        ]
        for line in lines_metrics:
            yield f"data: {json.dumps({'type': 'equiv_formula_line', 'section': 'metrics', 'line': line})}\n\n"
            import time; time.sleep(0.4)
        yield f"data: {json.dumps({'type': 'equiv_formula_line', 'section': 'metrics', 'section_done': True, 'line': ''})}\n\n"

        # ═══ Phase 3: 公式计算 ═══
        lines_formula = [
            f"▸ 等效变换推导",
            f"  TP 保持:  TP_eq = TP = {tp}",
            f"  PP 降维:  PP_eq = min(PP-1, 3) = {eq_pp}",
            f"  DP 缩减:  DP_eq = max(DP/4, 1) = {eq_dp}",
            f"  层数调整:  L_eq = (L/PP) × PP_eq = ({L_orig}/{pp}) × {eq_pp} = {eq_L}",
            f"  批次缩减:  B_eq = B × eq_dp/dp = {B_orig}×{eq_dp}/{orig_dp} ≈ {B_val}",
            f"▸ 等效结果:  {npu_orig} NPU → {npu_eq} NPU  (压缩 {comp_ratio}:1)",
        ]
        for line in lines_formula:
            yield f"data: {json.dumps({'type': 'equiv_formula_line', 'section': 'formula', 'line': line})}\n\n"
            import time; time.sleep(0.35)
        yield f"data: {json.dumps({'type': 'equiv_formula_line', 'section': 'formula', 'section_done': True, 'line': ''})}\n\n"

        # Signal done
        yield f"data: {json.dumps({'type': 'done', 'data': {'stage': 'step2'}})}\n\n"

    return Response(generate(), mimetype="text/event-stream")


@session_bp.route("/<session_id>/workflow/step3", methods=["POST"])
def workflow_step3(session_id: str):
    """Step 3: Generate equivalent mesh + model. Returns JSON."""
    session = session_manager.get_session(session_id)
    if not session:
        return {"error": "session not found"}, 404

    eq_params = session.equivalent_params if hasattr(session, "equivalent_params") else None
    if not eq_params:
        return {"error": "equivalent params not computed — run step2 first"}, 400

    model_meta = getattr(session, "_equiv_model_params", {}) or {}
    context = SkillContext(session=session, mcp_client=mcp_client, config=config)

    # 1. Generate equivalent mesh
    eq_dev = eq_params.device_type.value if hasattr(eq_params.device_type, "value") else str(eq_params.device_type)
    eq_mesh_args = {
        "name": "等效组网 (" + eq_dev + " DP" + str(eq_params.dp) + " TP" + str(eq_params.tp) + " PP" + str(eq_params.pp) + ")",
        "device_type": eq_dev,
        "dp": eq_params.dp, "tp": eq_params.tp, "pp": eq_params.pp,
    }
    eq_mesh_result: SkillResult = registry.execute_tool("training-mesh-gen-skill", eq_mesh_args, context)
    if not eq_mesh_result.success:
        return {"error": "eq_mesh_gen_failed", "message": eq_mesh_result.error}, 500

    eq_mesh = eq_mesh_result.data
    session.equivalent_topology = eq_mesh

    # 2. Generate equivalent model
    eq_model_args = {
        "num_layers": model_meta.get("L", 24),
        "d_model": model_meta.get("H", 4096),
        "num_heads": model_meta.get("A", 32),
        "d_ffn": model_meta.get("dff", 14336),
        "pp": eq_params.pp,
        "is_equivalent": True,
    }
    eq_model_result: SkillResult = registry.execute_tool("training-model-gen-skill", eq_model_args, context)
    if not eq_model_result.success:
        return {"error": "eq_model_gen_failed", "message": eq_model_result.error}, 500

    eq_model = eq_model_result.data
    session.equivalent_training_model = eq_model

    session.step = "equiv_generated"
    session_manager.save_session(session)

    eq_model_dict = eq_model.model_dump() if hasattr(eq_model, "model_dump") else eq_model
    eq_model_dict["seq_len"] = model_meta.get("S")
    eq_model_dict["batch_size"] = model_meta.get("B")
    return jsonify({
        "equivalent_mesh": _topo_with_model(
            eq_mesh, eq_model,
            seq_len=model_meta.get("S"), batch_size=model_meta.get("B"),
            model_name=model_meta.get("model_name"), d_ffn=model_meta.get("dff"),
        ),
        "equivalent_model": eq_model_dict,
        "step": session.step,
    })
