import uuid
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

    def create_session(self) -> SessionState:
        session_id = str(uuid.uuid4())[:8]
        state = SessionState(session_id=session_id)
        self._sessions[session_id] = state
        _persist_new_session(session_id)
        return state

    def get_session(self, session_id: str) -> Optional[SessionState]:
        if session_id in self._sessions:
            return self._sessions[session_id]
        state = _load_session(session_id)
        if state:
            self._sessions[session_id] = state
        return state

    def list_sessions(self) -> list[SessionState]:
        from app.dao import list_session_ids
        summaries = list_session_ids()
        result = []
        for s in summaries:
            sid = s["session_id"]
            if sid in self._sessions:
                result.append(self._sessions[sid])
            else:
                state = _load_session(sid)
                if state:
                    self._sessions[sid] = state
                    result.append(state)
        return result

    def delete_session(self, session_id: str) -> bool:
        from app.dao import delete_session as dao_delete
        dao_delete(session_id)
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    def update_session(self, session_id: str, **kwargs) -> Optional[SessionState]:
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
                params.update({
                    "num_layers": model.config.num_layers,
                    "hidden_dim": model.config.d_model,
                    "num_heads": model.config.num_heads,
                    "d_ffn": model.config.d_ffn,
                })
                step1_model_name = getattr(session, f"{role}_model_name", None)
                params.setdefault("model_name", step1_model_name or model.model_name or model.type)
            else:
                step1_model_name = getattr(session, f"{role}_model_name", None)
                if step1_model_name:
                    params.setdefault("model_name", step1_model_name)
            step1_seq_len = getattr(session, f"{role}_seq_len", None)
            step1_batch_size = getattr(session, f"{role}_batch_size", None)
            params.setdefault("seq_len", step1_seq_len or _profiler._SEQ_LEN)
            params.setdefault("batch_size", step1_batch_size or _profiler._TOTAL_BATCH)
            if role == "equivalent" and params.get("model_name"):
                params["model_name"] = params["model_name"] + "_eq"
            if params:
                save_topology_params(sid, role, params)

    for role in ("original", "equivalent"):
        sim = getattr(session, f"{role}_simulation", None)
        if sim:
            save_simulation_result(sid, role, sim.model_dump())

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
            content = msg.get("content", "")
            if content:
                save_message(sid, i, msg.get("role", "unknown"), str(content))


def _load_session(session_id: str) -> Optional[SessionState]:
    try:
        from app.dao import get_session_summary, get_topology_params, get_simulation_result, get_comparison_report, get_messages
        from app.models.schemas import DeviceType, MeshTopology, TopologyParams, TrainingModel, TrainingModelConfig, TrainingModelComputed, SimulationResult, ComparisonReport, CardMetrics

        summary = get_session_summary(session_id)
        if not summary:
            return None

        state = SessionState(session_id=session_id)
        state.step = summary.get("step", "idle")
        state.original_task_id = summary.get("original_task_id")
        state.equivalent_task_id = summary.get("equivalent_task_id")

        for role in ("original", "equivalent"):
            tp = get_topology_params(session_id, role)
            if tp and tp.get("name"):
                device_type = DeviceType(tp.get("device_type", "A3"))
                state_attr = f"{role}_topology"
                setattr(state, state_attr, MeshTopology(
                    name=tp.get("name", ""),
                    device_type=device_type,
                    dp_size=tp.get("dp_size", 1),
                    tp_size=tp.get("tp_size", 1),
                    pp_size=tp.get("pp_size", 1),
                    total_nodes=tp.get("total_nodes", 1),
                    nodes=[],
                ))

                params_attr = f"{role}_params"
                setattr(state, params_attr, TopologyParams(
                    device_type=device_type,
                    dp=tp.get("dp_size", 1),
                    tp=tp.get("tp_size", 1),
                    pp=tp.get("pp_size", 1),
                ))

                if tp.get("num_layers"):
                    d_model = tp.get("hidden_dim", 4096)
                    num_heads = tp.get("num_heads", 32)
                    d_ffn = tp.get("d_ffn", 11008)
                    config = TrainingModelConfig(
                        num_layers=tp.get("num_layers", 32),
                        d_model=d_model,
                        num_heads=num_heads,
                        d_ffn=d_ffn,
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
                    )
                    model_attr = f"{role}_training_model"
                    setattr(state, model_attr, TrainingModel(
                        type="transformer",
                        model_name=tp.get("model_name"),
                        config=config,
                        computed=computed,
                        layers=layers,
                    ))

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

        state.history = get_messages(session_id)
        return state
    except Exception as e:
        logger.error(f"[session] Failed to load session {session_id}: {e}")
        return None
