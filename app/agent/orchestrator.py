"""
Agent Orchestrator — core agent loop using OpenAI SDK with SkillRegistry.

Flow:
1. Receive user message + session context
2. LLM parses intent, decides which tools (skills + utilities) to call
3. Skills dispatched via SkillRegistry (with guardrails + retry)
4. Utility tools handled directly (MCP dispatch, comparison)
5. Stream progress back via SSE
"""

from __future__ import annotations

import json
import logging
from typing import AsyncGenerator

import httpx
from openai import OpenAI

from app.agent.guardrails import validate_input_params
from app.config import config
from app.mcp.client import mcp_client
from app.models.schemas import (
    AgentEvent,
    CardMetrics,
    ComparisonReport,
    DeviceType,
    MeshTopology,
    SessionState,
    SimulationResult,
    TopologyParams,
    TrainingModel,
)
from app.skills.base import SkillContext, SkillResult
from app.skills.registry import registry

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是 TrainMesh Agent，一个专为 AI 训练组网仿真测试设计的智能助手。

你的职责：
1. 帮助测试人员输入原始组网参数（设备类型 A2/A3/A5、DP/TP/PP、模型参数）
2. 调用 training-mesh-gen-skill 生成原始组网拓扑 JSON，再调用 training-model-gen-skill 生成原始模型结构
3. 用户确认等效计算后，逐条推送等效策略、计算指标、公式（equiv_formula_line），然后生成等效组网和等效模型
4. 通过 MCP 客户端向仿真系统下发仿真任务
5. 调用 training-mesh-profiler-skill 分析仿真结果
6. 对比原始组网和等效组网的仿真结果，判断等效性

分阶段工作流程：
- Step 1 (等效参数输入): 接收参数 → 护栏校验(后端静默) → 生成原始组网 → 生成原始模型结构 → 前端渲染
- Step 2 (等效计算): 用户确认 → 逐条推送等效策略/指标/公式 → 完成等效计算
- Step 3 (等效组网及模型渲染): 生成等效组网 → 生成等效模型结构 → 前端渲染
- Step 4 (仿真验证): 用户确认 → 下发仿真 → 切换到仿真验证tab
- Step 5 (结果分析): 自动对比 → 输出等效性结论

护栏校验在后端静默执行，不在工作流节点中展示。校验通过则继续，失败则返回错误提示用户修正参数。
请用中文与用户交互。每次只执行当前阶段的任务，等待用户确认后再进入下一阶段。"""

# ── Utility tool schemas (non-skill operations) ──

_UTILITY_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "validate_mesh_params",
            "description": "校验组网参数（输入护栏）：检查设备类型(A2/A3/A5)、DP、TP、PP 是否合法",
            "parameters": {
                "type": "object",
                "properties": {
                    "device_type": {
                        "type": "string",
                        "description": "设备类型: A2, A3, A5",
                    },
                    "dp": {"type": "integer", "description": "数据并行度"},
                    "tp": {"type": "integer", "description": "张量并行度"},
                    "pp": {"type": "integer", "description": "流水线并行度"},
                },
                "required": ["device_type", "dp", "tp", "pp"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_simulation",
            "description": "向仿真系统下发仿真任务（通过 MCP Client）。支持可选的仿真参数。当用户描述仿真参数时，提取并传入。未指定的参数使用默认值。",
            "parameters": {
                "type": "object",
                "properties": {
                    "topology_name": {"type": "string", "description": "组网名称"},
                    "topology_json": {
                        "type": "string",
                        "description": "组网 JSON 字符串",
                    },
                    "epoch_num": {
                        "type": "integer",
                        "description": "训练 epoch 数，默认 1",
                    },
                    "model_name": {"type": "string", "description": "模型名称"},
                    "device_type": {
                        "type": "string",
                        "description": "仿真硬件设备类型，默认 ASCEND_910B",
                    },
                    "vocab_size": {
                        "type": "string",
                        "description": "词表大小，默认 18277",
                    },
                    "frame": {
                        "type": "string",
                        "description": "框架类型，默认 Mindspeed",
                    },
                    "rank": {"type": "integer", "description": "起始 rank，默认 0"},
                    "rank_range": {
                        "type": "integer",
                        "description": "rank 范围，默认 1023",
                    },
                    "comp_filepath": {"type": "string", "description": "计算文件路径"},
                    "no_time_accumulation": {
                        "type": "boolean",
                        "description": "是否禁用时间累积，默认 false",
                    },
                    "visual_json_output": {
                        "type": "boolean",
                        "description": "是否输出可视化 JSON，默认 true",
                    },
                    "comm_group_output": {
                        "type": "boolean",
                        "description": "是否输出通信组，默认 true",
                    },
                    "debug_time": {
                        "type": "boolean",
                        "description": "是否调试时间，默认 false",
                    },
                },
                "required": ["topology_name", "topology_json"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_results",
            "description": "对比原始组网和等效组网的仿真结果，输出等效性分析报告。无需传入仿真数据，系统会自动从已保存的仿真结果中读取。",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "emit_formula_line",
            "description": "逐行推送等效计算分析内容（等效策略、计算指标、公式）。每条调用发送一行，前端逐行动态加载。三个section: strategy(等效策略), metrics(计算指标), formula(等效公式)。每个section结束时设置section_done=true。",
            "parameters": {
                "type": "object",
                "properties": {
                    "section": {
                        "type": "string",
                        "description": "所属区域: strategy, metrics, formula",
                    },
                    "line": {"type": "string", "description": "单行文本内容"},
                    "section_done": {
                        "type": "boolean",
                        "description": "该section是否完成，最后一行为true",
                    },
                },
                "required": ["section", "line"],
            },
        },
    },
]


def _build_tools() -> list[dict]:
    """Build combined tool list: skills from registry + utility tools."""
    skill_tools = registry.list_tools()
    return skill_tools + _UTILITY_TOOLS


def _execute_utility_tool(
    tool_name: str, arguments: dict, session: SessionState
) -> dict:
    """Execute a non-skill utility tool (guard check, MCP dispatch, comparison)."""
    if tool_name == "validate_mesh_params":
        result = validate_input_params(
            device_type=arguments["device_type"],
            dp=arguments["dp"],
            tp=arguments["tp"],
            pp=arguments["pp"],
        )
        if result.passed:
            from app.models.schemas import DeviceType, TopologyParams

            params = TopologyParams(
                device_type=DeviceType(arguments["device_type"].upper()),
                dp=arguments["dp"],
                tp=arguments["tp"],
                pp=arguments["pp"],
            )
            if session.original_params is None:
                session.original_params = params
            else:
                session.equivalent_params = params
        return result.model_dump()

    elif tool_name == "run_simulation":
        topology_json = json.loads(arguments["topology_json"])
        # Build simulation params from arguments, falling back to defaults
        sim_fields = [
            "epoch_num",
            "model_name",
            "device_type",
            "vocab_size",
            "frame",
            "rank",
            "rank_range",
            "comp_filepath",
            "no_time_accumulation",
            "visual_json_output",
            "comm_group_output",
            "debug_time",
        ]
        sim_data = {k: arguments[k] for k in sim_fields if k in arguments}
        if sim_data:
            from app.models.schemas import SimulationParams

            session.simulation_params = SimulationParams(**sim_data)
        sim_params_dict = (
            session.simulation_params.model_dump()
            if session.simulation_params
            else None
        )
        task_id = mcp_client.execute_task(topology_json, params=sim_params_dict)
        topo_name = arguments.get("topology_name", "")
        if "原始" in topo_name:
            session.original_task_id = task_id
        else:
            session.equivalent_task_id = task_id
        return {"task_id": task_id, "status": "submitted", "topology_name": topo_name}

    elif tool_name == "compare_results":
        if not session.original_simulation:
            return {
                "error": "缺少原始组网仿真结果，请先运行 training-mesh-profiler-skill 获取原始组网仿真数据"
            }
        if not session.equivalent_simulation:
            return {
                "error": "缺少等效组网仿真结果，请先运行 training-mesh-profiler-skill 获取等效组网仿真数据"
            }
        report = _build_comparison_report(
            session.original_simulation, session.equivalent_simulation
        )
        session.comparison_report = report
        session.step = "completed"
        return report.model_dump(exclude={"original", "equivalent"})

    elif tool_name == "emit_formula_line":
        section = arguments.get("section", "")
        line = arguments.get("line", "")
        section_done = arguments.get("section_done", False)
        return {
            "_event_type": "equiv_formula_line",
            "section": section,
            "line": line,
            "section_done": section_done,
        }

    return {"error": f"Unknown utility tool: {tool_name}"}


def _execute_skill_tool(tool_name: str, arguments: dict, session: SessionState) -> dict:
    """
    Execute a registered skill via SkillRegistry.
    The registry handles input guard → execute → output guard (with retry) automatically.
    """
    context = SkillContext(session=session, mcp_client=mcp_client, config=config)
    result: SkillResult = registry.execute_tool(tool_name, arguments, context)

    if not result.success:
        return {
            "error": result.error or "skill_execution_failed",
            "guardrail": result.guardrail_result.model_dump()
            if result.guardrail_result
            else None,
        }

    data = result.data

    if tool_name == "training-mesh-gen-skill" and isinstance(data, MeshTopology):
        params = TopologyParams(
            device_type=data.device_type,
            dp=data.dp_size,
            tp=data.tp_size,
            pp=data.pp_size,
        )
        if "原始" in data.name:
            session.original_topology = data
            session.original_params = params
            session.step = "params_collected"
        else:
            session.equivalent_topology = data
            session.equivalent_params = params
            session.step = "equiv_generated"
        # Return lightweight summary (full data via REST GET /api/session/<id>/topology)
        return {
            "name": data.name,
            "device_type": data.device_type.value,
            "total_nodes": data.total_nodes,
            "dp_size": data.dp_size,
            "tp_size": data.tp_size,
            "pp_size": data.pp_size,
            "comm_dp_groups": len(data.communication_groups.get("dp", [])),
            "comm_tp_groups": len(data.communication_groups.get("tp", [])),
            "comm_pp_groups": len(data.communication_groups.get("pp", [])),
            "session_id": session.session_id,
            "_type": "topology_summary",
        }

    elif tool_name == "training-mesh-profiler-skill" and isinstance(
        data, SimulationResult
    ):
        if "原始" in data.topology_name:
            session.original_simulation = data
        else:
            session.equivalent_simulation = data
        session.step = "simulating"
        # Return lightweight summary — full per-card data available via REST
        return {
            "topology_name": data.topology_name,
            "device_type": data.device_type.value,
            "total_nodes": data.total_nodes,
            "cards_count": len(data.cards),
            "session_id": session.session_id,
            "_type": "profiler_summary",
        }

    elif tool_name == "training-model-gen-skill" and isinstance(data, TrainingModel):
        role = "equivalent" if arguments.get("is_equivalent") else "original"
        if arguments.get("is_equivalent"):
            session.equivalent_training_model = data
        else:
            session.original_training_model = data
        return {
            "type": data.type,
            "config": data.config.model_dump(),
            "computed": data.computed.model_dump(),
            "layers_count": len(data.layers),
            "layer_types": [l.type for l in data.layers],
            "layers": [l.model_dump() for l in data.layers],
            "output_layer": data.output_layer.model_dump(),
            "_role": role,
            "session_id": session.session_id,
            "_type": "training_model_summary",
        }
    return data.model_dump() if hasattr(data, "model_dump") else data


def _build_comparison_report(
    original: SimulationResult, equivalent: SimulationResult
) -> ComparisonReport:
    eps = 1e-9

    def _diff_pct(ov: float, ev: float) -> float:
        return round(abs(ov - ev) / max(abs(ov), eps) * 100, 2)

    def _per_card(cards: list[CardMetrics], attr: str) -> float:
        return sum(getattr(c, attr) for c in cards) / max(len(cards), 1)

    of = _per_card(original.cards, "flops_per_card")
    ef = _per_card(equivalent.cards, "flops_per_card")
    oh = _per_card(original.cards, "hbm_gb")
    eh = _per_card(equivalent.cards, "hbm_gb")
    otp = _per_card(original.cards, "tp_comm_gb_per_micro")
    etp = _per_card(equivalent.cards, "tp_comm_gb_per_micro")
    opp = _per_card(original.cards, "pp_comm_mb_per_micro")
    epp = _per_card(equivalent.cards, "pp_comm_mb_per_micro")
    odp = _per_card(original.cards, "dp_comm_gb_per_step")
    edp = _per_card(equivalent.cards, "dp_comm_gb_per_step")

    flops_diff = _diff_pct(of, ef)
    hbm_diff = _diff_pct(oh, eh)
    tp_comm_diff = _diff_pct(otp, etp)
    pp_comm_diff = _diff_pct(opp, epp)
    dp_comm_diff = _diff_pct(odp, edp)

    tolerance = 5.0
    is_equivalent = (
        flops_diff <= tolerance
        and hbm_diff <= tolerance
        and tp_comm_diff <= tolerance
        and pp_comm_diff <= tolerance
        and dp_comm_diff <= tolerance
    )

    return ComparisonReport(
        original=original,
        equivalent=equivalent,
        flops_diff_pct=flops_diff,
        hbm_diff_pct=hbm_diff,
        tp_comm_diff_pct=tp_comm_diff,
        pp_comm_diff_pct=pp_comm_diff,
        dp_comm_diff_pct=dp_comm_diff,
        is_equivalent=is_equivalent,
        error_tolerance_pct=tolerance,
        details={
            "conclusion": "✅ 等效验证通过" if is_equivalent else "❌ 等效验证不通过",
            "max_diff_pct": round(
                max(flops_diff, hbm_diff, tp_comm_diff, pp_comm_diff, dp_comm_diff), 2
            ),
        },
    )


# ── Tool name classification ──

_SKILL_TOOLS = {
    "training-mesh-gen-skill",
    "training-mesh-profiler-skill",
    "training-model-gen-skill",
}

_SSE_EVENT_MAP = {
    "validate_mesh_params": "guard_check",
    "training-mesh-gen-skill": "mesh_json",
    "training-mesh-profiler-skill": "message",
    "training-model-gen-skill": "model_json",
    "run_simulation": "message",
    "compare_results": "message",
    "emit_formula_line": "equiv_formula_line",
}


def _build_workflow_state(session: SessionState) -> dict:
    """Build a lightweight workflow state dict for the frontend workflow panel."""
    return {
        "step": session.step,
        "original_topology": session.original_topology is not None,
        "equivalent_topology": session.equivalent_topology is not None,
        "original_training_model": session.original_training_model is not None,
        "equivalent_training_model": session.equivalent_training_model is not None,
        "original_simulation": session.original_simulation is not None,
        "equivalent_simulation": session.equivalent_simulation is not None,
        "comparison_report": session.comparison_report is not None,
        "comparison_equivalent": session.comparison_report.is_equivalent
        if session.comparison_report
        else None,
    }


async def agent_stream(
    session: SessionState,
    user_message: str,
) -> AsyncGenerator[AgentEvent, None]:
    """Main agent streaming loop. Skills dispatched via registry, utilities handled directly."""
    http_client = httpx.Client(verify=config.OPENAI_SSL_VERIFY)
    client = OpenAI(
        api_key=config.OPENAI_API_KEY,
        base_url=config.OPENAI_BASE_URL,
        http_client=http_client,
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]
    for h in session.history[-20:]:
        messages.append(h)
    messages.append({"role": "user", "content": user_message})
    session.history.append({"role": "user", "content": user_message})

    tools = _build_tools()
    _msg_size = lambda msgs: sum(len(json.dumps(m, ensure_ascii=False)) for m in msgs)
    logger.info(
        f"[agent_stream] session={session.session_id} tools={len(tools)} context_size={_msg_size(messages)} chars"
    )

    yield AgentEvent(event_type="thinking", message="正在分析您的请求...")

    max_rounds = 5
    for _round_num in range(max_rounds):
        logger.info(
            f"[agent_stream] round={_round_num + 1} context_size={_msg_size(messages)} chars"
        )
        try:
            response = client.chat.completions.create(
                model=config.OPENAI_MODEL,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.3,
            )
        except Exception as e:
            logger.exception(
                f"[agent_stream] OpenAI API error at round {_round_num + 1}"
            )
            yield AgentEvent(event_type="error", message=f"OpenAI API 异常: {e}")
            break

        msg = response.choices[0].message
        logger.info(
            f"[agent_stream] round={_round_num + 1} finish_reason={response.choices[0].finish_reason} "
            f"tool_calls={len(msg.tool_calls or [])} content_len={len(msg.content or '')}"
        )

        if not msg.tool_calls:
            if msg.content:
                yield AgentEvent(event_type="message", message=msg.content)
                session.history.append({"role": "assistant", "content": msg.content})
            break

        # Append assistant message with ALL tool_calls once before the loop
        assistant_msg = {
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
        }
        if hasattr(msg, "reasoning_content") and msg.reasoning_content:
            assistant_msg["reasoning_content"] = msg.reasoning_content
        messages.append(assistant_msg)
        session.history.append(assistant_msg)

        for tool_call in msg.tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)

            # Auto-inject task_id from session if LLM didn't provide one
            if tool_name == "training-mesh-profiler-skill" and not arguments.get(
                "task_id"
            ):
                topo_name = arguments.get("topology_name", "")
                sid = (
                    session.original_task_id
                    if "原始" in topo_name
                    else session.equivalent_task_id
                )
                if sid:
                    arguments["task_id"] = sid
                    logger.info(
                        f"[agent_stream] auto-injected task_id={sid} for profiler"
                    )

            yield AgentEvent(
                event_type="tool_call",
                message=f"调用工具: {tool_name}",
                data={"tool_name": tool_name, "arguments": arguments},
            )

            try:
                if tool_name in _SKILL_TOOLS:
                    result = _execute_skill_tool(tool_name, arguments, session)
                else:
                    result = _execute_utility_tool(tool_name, arguments, session)
            except Exception as e:
                logger.exception(f"[agent_stream] tool execution error: {tool_name}")
                yield AgentEvent(event_type="error", message=f"工具执行异常: {e}")
                break

            logger.info(
                f"[agent_stream] tool={tool_name} result_size={len(json.dumps(result, ensure_ascii=False))} chars"
            )

            event_type = _SSE_EVENT_MAP.get(tool_name, "message")

            # Override event type if result specifies one
            if result.get("_event_type"):
                event_type = result.pop("_event_type")

            if "error" in result:
                event_type = "error"
                result_msg = f"执行失败: {result.get('error')}"
                session.history.append(
                    {"role": "system", "content": "❌ " + result_msg}
                )
            elif tool_name == "validate_mesh_params":
                result_msg = "护栏校验" + ("通过" if result.get("passed") else "失败")
                session.history.append(
                    {"role": "system", "content": "✅ " + result_msg}
                    if result.get("passed")
                    else {"role": "system", "content": "❌ " + result_msg}
                )
            elif tool_name == "training-mesh-gen-skill":
                result_msg = f"组网 '{result.get('name', '')}' 生成成功"
            elif tool_name == "training-model-gen-skill":
                result_msg = f"模型结构 'transformer_model' 生成成功, {result.get('layers_count', 0)} 层"
            elif tool_name == "training-mesh-profiler-skill":
                result_msg = f"仿真分析完成: {result.get('topology_name', '')} — {result.get('total_nodes', 0)} 节点"
            elif tool_name == "run_simulation":
                result_msg = f"仿真任务已下发: {result.get('task_id', '')}"
            elif tool_name == "compare_results":
                result_msg = (
                    f"对比分析完成: {result.get('details', {}).get('conclusion', '')}"
                )
            else:
                result_msg = f"工具 {tool_name} 执行完成"

            yield AgentEvent(
                event_type=event_type,
                message=result_msg,
                data=result,
            )

            tool_content = json.dumps(result, ensure_ascii=False)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_content,
                }
            )
            session.history.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_content,
                }
            )

    # Auto-profiling: if topologies have task_ids but no simulation data, run profiler now
    for label, topo, task_id in [
        ("original", session.original_topology, session.original_task_id),
        ("equivalent", session.equivalent_topology, session.equivalent_task_id),
    ]:
        sim = (
            session.original_simulation
            if label == "original"
            else session.equivalent_simulation
        )
        if topo and task_id and not sim:
            logger.info(
                f"[agent_stream] auto-profiling {label} topology (task_id={task_id})"
            )
            try:
                training_model = (
                    session.original_training_model
                    if label == "original"
                    else session.equivalent_training_model
                )
                profiler_args = {
                    "topology_name": topo.name,
                    "device_type": topo.device_type.value,
                    "total_nodes": topo.total_nodes,
                    "dp": topo.dp_size,
                    "tp": topo.tp_size,
                    "pp": topo.pp_size,
                    "task_id": task_id,
                }
                if session.simulation_params:
                    profiler_args["simulation_params"] = (
                        session.simulation_params.model_dump()
                    )
                if training_model:
                    profiler_args["num_layers"] = training_model.config.num_layers
                    profiler_args["hidden_dim"] = training_model.config.d_model
                _execute_skill_tool(
                    "training-mesh-profiler-skill", profiler_args, session
                )
            except Exception as e:
                logger.exception(f"[agent_stream] auto-profiling {label} failed: {e}")

    # Push simulation results via SSE so the frontend doesn't need a separate HTTP request
    sim_payload = {}
    if session.original_simulation:
        sim_payload["original"] = session.original_simulation.model_dump()
    if session.equivalent_simulation:
        sim_payload["equivalent"] = session.equivalent_simulation.model_dump()
    if sim_payload:
        yield AgentEvent(event_type="sim_data", data=sim_payload, message="仿真数据")

    # Push workflow_state so frontend can sync the flow diagram
    yield AgentEvent(
        event_type="workflow_state",
        message="",
        data=_build_workflow_state(session),
    )

    from app.agent.session import session_manager

    session_manager.save_session(session)

    yield AgentEvent(event_type="done", message="处理完成")
