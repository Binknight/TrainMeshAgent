import json
import uuid
import threading
import logging
from typing import Optional
from app.models.schemas import SessionState

logger = logging.getLogger(__name__)


class SessionManager:
    """Manages agent session state with PostgreSQL persistence.

    In-memory cache (dict) for fast access, backed by database for durability.
    Call save_session() after significant state changes to persist to DB.
    """

    def __init__(self):
        self._sessions: dict[str, SessionState] = {}
        self._lock = threading.Lock()

    def create_session(self) -> SessionState:
        session_id = str(uuid.uuid4())[:8]
        state = SessionState(session_id=session_id)
        with self._lock:
            self._sessions[session_id] = state
        _persist_new_session(session_id)
        return state

    def get_session(self, session_id: str) -> Optional[SessionState]:
        with self._lock:
            if session_id in self._sessions:
                return self._sessions[session_id]
        state = _load_session(session_id)
        if state:
            with self._lock:
                self._sessions[session_id] = state
        return state

    def list_sessions(self) -> list[SessionState]:
        from app.dao import list_session_ids
        summaries = list_session_ids()
        result = []
        for s in summaries:
            sid = s["session_id"]
            with self._lock:
                if sid in self._sessions:
                    result.append(self._sessions[sid])
                    continue
            state = _load_session(sid)
            if state:
                with self._lock:
                    self._sessions[sid] = state
                result.append(state)
        return result

    def delete_session(self, session_id: str) -> bool:
        from app.dao import delete_session as dao_delete
        dao_delete(session_id)
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                return True
        return False

    def update_session(self, session_id: str, **kwargs) -> Optional[SessionState]:
        with self._lock:
            state = self._sessions.get(session_id)
        if not state:
            return None
        for key, value in kwargs.items():
            if hasattr(state, key):
                setattr(state, key, value)
        return state

    def save_session(self, session: SessionState) -> None:
        """Persist the full session state to PostgreSQL."""
        try:
            _persist_session(session)
        except Exception as e:
            logger.error(f"[session] Failed to save session {session.session_id}: {e}")


session_manager = SessionManager()


def _persist_new_session(session_id: str) -> None:
    try:
        from app.dao import create_session
        create_session(session_id)
    except Exception as e:
        logger.error(f"[session] Failed to create session in DB: {e}")


def _persist_session(session: SessionState) -> None:
    from app.dao import (
        update_session_step, save_topology_params, save_simulation_params,
        save_simulation_result, save_comparison_report, delete_messages, save_message,
        save_formula_lines,
    )

    sid = session.session_id

    update_session_step(sid, session.step, session.original_task_id, session.equivalent_task_id)

    for role in ("original", "equivalent"):
        topo = getattr(session, f"{role}_topology", None)
        model = getattr(session, f"{role}_training_model", None)
        params_obj = getattr(session, f"{role}_params", None)
        if topo or model or params_obj:
            import importlib
            _profiler = importlib.import_module("app.skills.training-mesh-profiler-skill")
            params: dict = {}
            if topo:
                params.update({
                    "name": topo.name,
                    "device_type": topo.device_type.value if hasattr(topo.device_type, 'value') else str(topo.device_type),
                    "dp_size": topo.dp_size,
                    "tp_size": topo.tp_size,
                    "pp_size": topo.pp_size,
                    "total_nodes": topo.total_nodes,
                })
            elif params_obj:
                dev = params_obj.device_type.value if hasattr(params_obj.device_type, 'value') else str(params_obj.device_type)
                params.update({
                    "name": "原始组网" if role == "original" else "等效组网",
                    "device_type": dev,
                    "dp_size": params_obj.dp,
                    "tp_size": params_obj.tp,
                    "pp_size": params_obj.pp,
                    "total_nodes": params_obj.dp * params_obj.tp * params_obj.pp,
                })
            if model and hasattr(model, "config"):
                cfg = model.config
                params.update({
                    "num_layers": cfg.num_layers,
                    "hidden_dim": cfg.d_model,
                    "num_heads": cfg.num_heads,
                    "d_ffn": cfg.d_ffn,
                    # ── MoE fields (None for dense models) ──
                    "model_type": cfg.model_type,
                    "num_experts": cfg.num_experts,
                    "moe_router_topk": cfg.moe_router_topk,
                    "num_moe_layers": cfg.num_moe_layers,
                    "moe_ffn_hidden_size": cfg.moe_ffn_hidden_size,
                    "has_shared_expert": cfg.has_shared_expert,
                    "expert_tensor_parallel_size": cfg.expert_tensor_parallel_size,
                })
                step1_model_name = getattr(session, f"{role}_model_name", None)
                params.setdefault("model_name", step1_model_name or model.model_name or model.type)
                # ── EP from session state (not in TrainingModelConfig) ──
                step1_ep = getattr(session, f"{role}_ep", None)
                if step1_ep is not None:
                    params.setdefault("ep", step1_ep)
            else:
                step1_model_name = getattr(session, f"{role}_model_name", None)
                if step1_model_name:
                    params.setdefault("model_name", step1_model_name)
                # ── EP from session state (persists even without model) ──
                step1_ep = getattr(session, f"{role}_ep", None)
                if step1_ep is not None:
                    params.setdefault("ep", step1_ep)
            step1_seq_len = getattr(session, f"{role}_seq_len", None)
            step1_batch_size = getattr(session, f"{role}_batch_size", None)
            step1_dff = getattr(session, f"{role}_dff", None)
            step1_vocab_size = getattr(session, f"{role}_vocab_size", None)
            step1_micro_batch = getattr(session, f"{role}_micro_batch", None)
            params.setdefault("seq_len", step1_seq_len or _profiler._SEQ_LEN)
            params.setdefault("batch_size", step1_batch_size or _profiler._TOTAL_BATCH)
            params.setdefault("d_ffn", step1_dff or 14336)
            if step1_vocab_size is not None:
                params.setdefault("vocab_size", step1_vocab_size)
            if step1_micro_batch is not None:
                params.setdefault("micro_batch_size", step1_micro_batch)
            if role == "equivalent":
                orig_model = getattr(session, "original_training_model", None)
                base_name = (
                    getattr(session, "original_model_name", None)
                    or (orig_model.model_name if orig_model else None)
                    or (orig_model.type if orig_model else None)
                )
                if base_name:
                    params["model_name"] = base_name + "_eq"
            if params:
                save_topology_params(sid, role, params)

    for role in ("original", "equivalent"):
        sim = getattr(session, f"{role}_simulation", None)
        if sim:
            save_simulation_result(sid, role, sim.model_dump())

    if session.simulation_params:
        d = session.simulation_params.model_dump()
        save_simulation_params(sid, "original", d)
        save_simulation_params(sid, "equivalent", d)

    if session.comparison_report:
        from app.dao import get_simulation_result
        orig_sim = get_simulation_result(sid, "original")
        eq_sim = get_simulation_result(sid, "equivalent")
        orig_id = orig_sim["id"] if orig_sim else None
        eq_id = eq_sim["id"] if eq_sim else None
        save_comparison_report(sid, orig_id, eq_id, session.comparison_report.model_dump())

    if session.history:
        delete_messages(sid)
        for i, msg in enumerate(session.history):
            content = json.dumps(msg, ensure_ascii=False)
            save_message(sid, i, msg.get("role", "unknown"), content)

    if session.formula_lines:
        save_formula_lines(sid, session.formula_lines)


def _load_session(session_id: str) -> Optional[SessionState]:
    try:
        from app.dao import get_session_summary, get_topology_params, get_simulation_result, get_comparison_report, get_messages, get_simulation_params
        from app.models.schemas import DeviceType, MeshNode, MeshTopology, TopologyParams, TrainingModel, TrainingModelConfig, TrainingModelComputed, SimulationResult, ComparisonReport, CardMetrics

        summary = get_session_summary(session_id)
        if not summary:
            return None

        state = SessionState(session_id=session_id)
        state.step = summary.get("step", "idle")
        state.original_task_id = summary.get("original_task_id")
        state.equivalent_task_id = summary.get("equivalent_task_id")
        state.formula_lines = summary.get("formula_lines")

        for role in ("original", "equivalent"):
            tp = get_topology_params(session_id, role)
            if tp and tp.get("name"):
                device_type = DeviceType(tp.get("device_type", "A3"))
                dp_size = tp.get("dp_size", 1)
                tp_size = tp.get("tp_size", 1)
                pp_size = tp.get("pp_size", 1)
                total_nodes = tp.get("total_nodes", dp_size * tp_size * pp_size)
                # Rebuild nodes from dp/tp/pp (deterministic, no need to persist separately)
                nodes = []
                ranks_per_dp = tp_size * pp_size
                for g in range(total_nodes):
                    dp_rank = g // ranks_per_dp
                    remainder = g % ranks_per_dp
                    # 与前端 meshBuildData 一致：TP 最低位、PP 居中
                    # global_rank = dp*(tp*pp) + pp*tp + tp
                    pp_rank = remainder // tp_size
                    tp_rank = remainder % tp_size
                    nodes.append(MeshNode(
                        id=f"node_{g}",
                        device_type=device_type,
                        dp_rank=dp_rank,
                        tp_rank=tp_rank,
                        pp_rank=pp_rank,
                        global_rank=g,
                    ))
                state_attr = f"{role}_topology"
                setattr(state, state_attr, MeshTopology(
                    name=tp.get("name", ""),
                    device_type=device_type,
                    dp_size=dp_size,
                    tp_size=tp_size,
                    pp_size=pp_size,
                    total_nodes=total_nodes,
                    nodes=nodes,
                ))

                params_attr = f"{role}_params"
                setattr(state, params_attr, TopologyParams(
                    device_type=device_type,
                    dp=tp.get("dp_size", 1),
                    tp=tp.get("tp_size", 1),
                    pp=tp.get("pp_size", 1),
                    ep=tp.get("ep"),
                ))

                if tp.get("num_layers"):
                    d_model = tp.get("hidden_dim", 4096)
                    num_heads = tp.get("num_heads", 32)
                    d_ffn = tp.get("d_ffn", 11008)
                    # ── MoE fields (None for dense models) ──
                    model_type = tp.get("model_type", "dense")
                    num_experts = tp.get("num_experts")
                    moe_router_topk = tp.get("moe_router_topk")
                    num_moe_layers = tp.get("num_moe_layers")
                    moe_ffn_hidden_size = tp.get("moe_ffn_hidden_size")
                    has_shared_expert = tp.get("has_shared_expert", False)
                    expert_tp = tp.get("expert_tensor_parallel_size", 1)
                    config = TrainingModelConfig(
                        num_layers=tp.get("num_layers", 32),
                        d_model=d_model,
                        num_heads=num_heads,
                        d_ffn=d_ffn,
                        model_type=model_type,
                        num_experts=num_experts,
                        moe_router_topk=moe_router_topk,
                        num_moe_layers=num_moe_layers,
                        moe_ffn_hidden_size=moe_ffn_hidden_size,
                        has_shared_expert=has_shared_expert,
                        expert_tensor_parallel_size=expert_tp,
                    )
                    d_head = d_model // num_heads
                    computed = TrainingModelComputed(
                        d_head=d_head,
                        total_params_billions="~0.0",
                    )
                    import importlib
                    _model_gen = importlib.import_module("app.skills.training-model-gen-skill")
                    layers = _model_gen._build_layers(
                        config.num_layers, num_heads, d_head, d_ffn, "GELU",
                        num_experts=num_experts,
                        moe_ffn_hidden_size=moe_ffn_hidden_size,
                        num_moe_layers=num_moe_layers,
                        has_shared_expert=has_shared_expert,
                    )
                    model_attr = f"{role}_training_model"
                    setattr(state, model_attr, TrainingModel(
                        type="transformer",
                        model_name=tp.get("model_name"),
                        config=config,
                        computed=computed,
                        layers=layers,
                    ))

        # Restore session-level runtime fields from topology_params (survive server restart)
        orig_tp = get_topology_params(session_id, "original")
        if orig_tp:
            state.original_seq_len = orig_tp.get("seq_len")
            state.original_batch_size = orig_tp.get("batch_size")
            state.original_micro_batch = orig_tp.get("micro_batch_size")
            state.original_dff = orig_tp.get("d_ffn")
            state.original_vocab_size = orig_tp.get("vocab_size")
            state.original_model_name = orig_tp.get("model_name")
            # ── MoE session-level fields ──
            state.original_model_type = orig_tp.get("model_type", "dense")
            state.original_ep = orig_tp.get("ep")
            state.original_num_experts = orig_tp.get("num_experts")
            state.original_moe_topk = orig_tp.get("moe_router_topk")
            state.original_num_moe_layers = orig_tp.get("num_moe_layers")
            state.original_moe_ffn_hidden_size = orig_tp.get("moe_ffn_hidden_size")
            state.original_has_shared_expert = orig_tp.get("has_shared_expert")
            state.original_expert_tensor_parallel_size = orig_tp.get("expert_tensor_parallel_size")

        eq_tp = get_topology_params(session_id, "equivalent")
        if eq_tp:
            state.equivalent_seq_len = eq_tp.get("seq_len")
            state.equivalent_batch_size = eq_tp.get("batch_size")
            state.equivalent_micro_batch = eq_tp.get("micro_batch_size")
            state.equivalent_dff = eq_tp.get("d_ffn")
            # ── MoE session-level fields ──
            state.equivalent_ep = eq_tp.get("ep")
            state.equivalent_num_moe_layers = eq_tp.get("num_moe_layers")
            state.equivalent_num_experts = eq_tp.get("num_experts")

        for role in ("original", "equivalent"):
            sr = get_simulation_result(session_id, role)
            if sr:
                cards = [CardMetrics(**c) for c in (sr.get("cards") or [])]
                sim = SimulationResult(
                    topology_name=sr.get("topology_name", ""),
                    device_type=sr.get("device_type", "A3"),
                    total_nodes=sr.get("total_nodes", 0),
                    cards=cards,
                )
                sim_attr = f"{role}_simulation"
                setattr(state, sim_attr, sim)

        cr = get_comparison_report(session_id)
        if cr:
            state.comparison_report = ComparisonReport(
                original=state.original_simulation or SimulationResult(
                    topology_name="", device_type="A3", total_nodes=0, cards=[],
                ),
                equivalent=state.equivalent_simulation or SimulationResult(
                    topology_name="", device_type="A3", total_nodes=0, cards=[],
                ),
                flops_diff_pct=cr.get("flops_diff_pct", 0),
                hbm_diff_pct=cr.get("hbm_diff_pct", 0),
                tp_comm_diff_pct=cr.get("tp_comm_diff_pct", 0),
                pp_comm_diff_pct=cr.get("pp_comm_diff_pct", 0),
                dp_comm_diff_pct=cr.get("dp_comm_diff_pct", 0),
                is_equivalent=cr.get("is_equivalent", False),
                error_tolerance_pct=cr.get("error_tolerance", 5.0),
                details=cr.get("details", {}),
            )

        # Restore simulation params
        sim_params_data = get_simulation_params(session_id, "original")
        if sim_params_data:
            from app.models.schemas import SimulationParams
            state.simulation_params = SimulationParams(**sim_params_data)

        state.history = get_messages(session_id)
        return state
    except Exception as e:
        logger.error(f"[session] Failed to load session {session_id}: {e}")
        return None
