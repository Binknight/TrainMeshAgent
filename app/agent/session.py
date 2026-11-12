import json
import uuid
import threading
import logging
from contextlib import contextmanager
from typing import Optional
from app.models.schemas import SessionState
from app.rank_layout import decompose

logger = logging.getLogger(__name__)


class SessionManager:
    """Manages agent session state with PostgreSQL persistence.

    In-memory cache (dict) for fast access, backed by database for durability.
    Call save_session() after significant state changes to persist to DB.

    并发约定：`_lock` 只保护 `_sessions` 字典本身；`SessionState` 对象是**可变的
    共享对象**，多个线程（HTTP 请求线程 / 仿真 WebSocket 轮询线程 / SSE 线程）都会
    改它。凡是要对同一个会话做「读-改-写」的业务路径，必须用 `session_lock()`
    串行化，否则会出现任务重复下发、step 互相覆盖、消息丢失等竞态。
    """

    def __init__(self):
        self._sessions: dict[str, SessionState] = {}
        self._lock = threading.Lock()
        # session_id -> RLock。RLock 允许已持锁的调用链里再次 save_session()。
        self._session_locks: dict[str, threading.RLock] = {}
        self._session_locks_guard = threading.Lock()
        # 会话级「长操作」互斥：对话/Step1/Step2/Step3/仿真提交这些耗时操作
        # （可能秒级甚至分钟级）同一会话同时只允许一个，冲突返回 409。
        # 只做占用标记，不阻塞等待 —— 否则一个长对话会把另一个标签页的请求
        # 卡到 LLM 超时。
        self._exclusive_ops: set[str] = set()
        self._exclusive_guard = threading.Lock()

    def _lock_for(self, session_id: str) -> threading.RLock:
        with self._session_locks_guard:
            lock = self._session_locks.get(session_id)
            if lock is None:
                lock = threading.RLock()
                self._session_locks[session_id] = lock
            return lock

    @contextmanager
    def session_lock(self, session_id: str):
        """串行化同一会话的「读-改-写」。

        `run-simulation` 提交、`_fallback_reconcile_simulation` 补数、
        `_mark_session_failed` 置失败这三条路径来自不同线程，必须共用同一把锁。
        """
        with self._lock_for(session_id):
            yield

    # ------------------------------------------------------------------
    # 长操作互斥（非阻塞）：同一会话同一时刻只允许一个对话/生成/仿真提交
    # ------------------------------------------------------------------
    def try_begin_exclusive(self, session_id: str) -> bool:
        """尝试占用会话的长操作名额；已被占用返回 False。"""
        with self._exclusive_guard:
            if session_id in self._exclusive_ops:
                return False
            self._exclusive_ops.add(session_id)
            return True

    def end_exclusive(self, session_id: str) -> None:
        """释放长操作名额（必须与 try_begin_exclusive 成对调用）。"""
        with self._exclusive_guard:
            self._exclusive_ops.discard(session_id)

    def is_exclusive_busy(self, session_id: str) -> bool:
        with self._exclusive_guard:
            return session_id in self._exclusive_ops

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
        # cache miss 必须只加载一份：并发 miss 时若各自 _load_session()，两个线程会
        # 拿到两个独立对象，随后分别改、分别落库，状态互相覆盖（真实竞态）。
        with self._lock_for(session_id):
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
        found = False
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                found = True
        # 会话级锁也一并清理，避免删除后同 id 重建时复用旧锁表/内存泄漏
        with self._session_locks_guard:
            self._session_locks.pop(session_id, None)
        with self._exclusive_guard:
            self._exclusive_ops.discard(session_id)
        return found

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
        """Persist the full session state to PostgreSQL.

        Each table persists independently — one failure (e.g. a DB schema
        mismatch) is logged and recorded in _persist_failures instead of
        aborting the remaining tables. Failures are surfaced to the history
        panel via get_persist_failure(), so missing data no longer silently
        degrades to fallback titles like "新建任务".

        同一会话的保存必须串行：`_persist_session()` 是「先删全部消息再逐条插入」，
        两个线程交错执行会丢消息/互相覆盖；且并行仿真完成时多个线程会同时保存。
        """
        sid = session.session_id
        with self._lock_for(sid):
            try:
                if _persist_session(session):
                    _persist_failures.pop(sid, None)
            except Exception as e:
                # Safety net — per-table errors are caught inside _persist_session.
                _persist_failures[sid] = f"unexpected: {e}"
                logger.error(f"[session] Failed to save session {sid}: {e}")


session_manager = SessionManager()

# session_id -> last persist error. Written by _safe_persist on failure and
# cleared by save_session when every table persisted cleanly.
_persist_failures: dict[str, str] = {}


def _safe_persist(session_id: str, label: str, fn) -> bool:
    """Run one persist step in isolation so a single table failure doesn't
    block persisting the remaining tables. Returns True on success."""
    try:
        fn()
        return True
    except Exception as e:
        msg = f"{label}: {e}"
        _persist_failures[session_id] = msg
        logger.error(f"[session] Failed to save session {session_id}: {msg}")
        return False


def get_persist_failure(session_id: str) -> str | None:
    """Return the last persistence error for a session, if any."""
    return _persist_failures.get(session_id)


def _persist_new_session(session_id: str) -> None:
    try:
        from app.dao import create_session
        create_session(session_id)
    except Exception as e:
        logger.error(f"[session] Failed to create session in DB: {e}")


def _persist_session(session: SessionState) -> bool:
    """Persist every table independently; returns True when all succeed.

    Previously any failure (e.g. topology_params missing a column on an
    old DB schema) aborted the whole chain, so session step, simulation
    results and comparison reports were silently lost too — and the history
    panel fell back to "新建任务" titles with no visible error."""
    from app.dao import (
        update_session_step, save_topology_params, save_simulation_params,
        save_simulation_result, save_comparison_report, delete_messages, save_message,
        save_formula_lines,
    )

    sid = session.session_id
    ok = True

    ok &= _safe_persist(
        sid, "session_step",
        lambda: update_session_step(sid, session.step, session.original_task_id, session.equivalent_task_id),
    )

    for role in ("original", "equivalent"):
        def _persist_topology(r=role):
            topo = getattr(session, f"{r}_topology", None)
            model = getattr(session, f"{r}_training_model", None)
            params_obj = getattr(session, f"{r}_params", None)
            if not (topo or model or params_obj):
                return
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
                    "name": "原始组网" if r == "original" else "等效组网",
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
                step1_model_name = getattr(session, f"{r}_model_name", None)
                params.setdefault("model_name", step1_model_name or model.model_name or model.type)
                # ── EP from session state (not in TrainingModelConfig) ──
                step1_ep = getattr(session, f"{r}_ep", None)
                if step1_ep is not None:
                    params.setdefault("ep", step1_ep)
            else:
                step1_model_name = getattr(session, f"{r}_model_name", None)
                if step1_model_name:
                    params.setdefault("model_name", step1_model_name)
                # ── EP from session state (persists even without model) ──
                step1_ep = getattr(session, f"{r}_ep", None)
                if step1_ep is not None:
                    params.setdefault("ep", step1_ep)
            step1_seq_len = getattr(session, f"{r}_seq_len", None)
            step1_batch_size = getattr(session, f"{r}_batch_size", None)
            step1_dff = getattr(session, f"{r}_dff", None)
            step1_vocab_size = getattr(session, f"{r}_vocab_size", None)
            step1_micro_batch = getattr(session, f"{r}_micro_batch", None)
            params.setdefault("seq_len", step1_seq_len or _profiler._SEQ_LEN)
            params.setdefault("batch_size", step1_batch_size or _profiler._TOTAL_BATCH)
            params.setdefault("d_ffn", step1_dff or 14336)
            if step1_vocab_size is not None:
                params.setdefault("vocab_size", step1_vocab_size)
            if step1_micro_batch is not None:
                params.setdefault("micro_batch_size", step1_micro_batch)
            if r == "equivalent":
                orig_model = getattr(session, "original_training_model", None)
                base_name = (
                    getattr(session, "original_model_name", None)
                    or (orig_model.model_name if orig_model else None)
                    or (orig_model.type if orig_model else None)
                )
                if base_name:
                    params["model_name"] = base_name + "_eq"
            if params:
                save_topology_params(sid, r, params)

        ok &= _safe_persist(sid, f"topology_params:{role}", _persist_topology)

    for role in ("original", "equivalent"):
        sim = getattr(session, f"{role}_simulation", None)
        if sim:
            ok &= _safe_persist(
                sid, f"simulation_result:{role}",
                lambda s=sim, r=role: save_simulation_result(sid, r, s.model_dump()),
            )

    if session.simulation_params:
        d = session.simulation_params.model_dump()
        ok &= _safe_persist(sid, "simulation_params:original", lambda: save_simulation_params(sid, "original", d))
        ok &= _safe_persist(sid, "simulation_params:equivalent", lambda: save_simulation_params(sid, "equivalent", d))

    if session.comparison_report:
        def _persist_comparison():
            from app.dao import get_simulation_result
            orig_sim = get_simulation_result(sid, "original")
            eq_sim = get_simulation_result(sid, "equivalent")
            orig_id = orig_sim["id"] if orig_sim else None
            eq_id = eq_sim["id"] if eq_sim else None
            save_comparison_report(sid, orig_id, eq_id, session.comparison_report.model_dump())

        ok &= _safe_persist(sid, "comparison_report", _persist_comparison)

    if session.history:
        def _persist_messages():
            delete_messages(sid)
            for i, msg in enumerate(session.history):
                content = json.dumps(msg, ensure_ascii=False)
                save_message(sid, i, msg.get("role", "unknown"), content)

        ok &= _safe_persist(sid, "conversation_messages", _persist_messages)

    if session.formula_lines:
        ok &= _safe_persist(sid, "formula_lines", lambda: save_formula_lines(sid, session.formula_lines))

    # Note: &= (not `and`) so every table is still attempted after a failure.
    return bool(ok)


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
            if not (tp and tp.get("name")):
                continue
            device_type = DeviceType(tp.get("device_type", "A3"))
            dp_size = tp.get("dp_size", 1)
            tp_size = tp.get("tp_size", 1)
            pp_size = tp.get("pp_size", 1)
            total_nodes = tp.get("total_nodes", dp_size * tp_size * pp_size)
            # topology_params 同一张表既存「已生成的组网」，也存 Step1 算出的
            # 「等效目标参数」（用于重启后继续执行 Step2）。等效侧只有带模型参数
            # （num_layers）时才证明 Step2 真的生成过等效组网；否则只恢复
            # equivalent_params，绝不能重建 equivalent_topology —— 否则 Step1-only
            # 会话重启后会被伪装成「已完成等效推导」（前端渲染出没有模型/公式的
            # 等效组网，2026-11 用户截图问题：缺等效模型 + 推导卡空白）。
            has_generated_model = tp.get("num_layers") is not None
            if role == "original" or has_generated_model:
                # Rebuild nodes from dp/tp/pp (deterministic, no need to persist separately)
                nodes = []
                for g in range(total_nodes):
                    # Rank layout = TP-DP-PP, calibrated to the simulation system:
                    # global_rank = pp*(tp*dp) + dp*tp + tp  (see app.rank_layout)
                    dp_rank, tp_rank, pp_rank = decompose(g, dp_size, tp_size, pp_size)
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

            # 参数始终恢复：重启后 Step2 依赖 equivalent_params 才能继续执行
            params_attr = f"{role}_params"
            setattr(state, params_attr, TopologyParams(
                device_type=device_type,
                dp=tp.get("dp_size", 1),
                tp=tp.get("tp_size", 1),
                pp=tp.get("pp_size", 1),
                ep=tp.get("ep"),
            ))

            if has_generated_model:
                # DB 里这些列对老会话可能是 NULL（规范列/旧版本缺省），必须给
                # 非空兜底：否则 pydantic 校验（如 has_shared_expert 必填 bool）
                # 会抛错，整个会话加载失败（表现成历史任务打不开）。
                d_model = tp.get("hidden_dim") or 4096
                num_heads = tp.get("num_heads") or 32
                d_ffn = tp.get("d_ffn") or 11008
                # ── MoE fields (None for dense models) ──
                model_type = tp.get("model_type") or "dense"
                num_experts = tp.get("num_experts")
                moe_router_topk = tp.get("moe_router_topk")
                num_moe_layers = tp.get("num_moe_layers")
                moe_ffn_hidden_size = tp.get("moe_ffn_hidden_size")
                has_shared_expert = bool(tp.get("has_shared_expert"))
                expert_tp = tp.get("expert_tensor_parallel_size") or 1
                config = TrainingModelConfig(
                    num_layers=tp.get("num_layers") or 32,
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
