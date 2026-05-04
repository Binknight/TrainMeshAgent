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

from openai import OpenAI

from app.config import config
from app.models.schemas import (
    AgentEvent, ComparisonReport, DeviceType, MeshTopology,
    SessionState, SimulationResult, TopologyParams, TrainingModel,
)
from app.agent.guardrails import validate_input_params
from app.skills.base import SkillContext, SkillResult
from app.skills.registry import registry
from app.mcp.client import mcp_client

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是 TrainMesh Agent，一个专为 AI 训练组网仿真测试设计的智能助手。

你的职责：
1. 帮助测试人员输入原始组网和等效组网的参数（设备类型 A2/A3/A5、DP/TP/PP）
2. 调用 training-mesh-gen-skill 生成结构化 JSON 组网图
3. 调用 training-model-gen-skill 生成等效 Transformer 训练模型结构
4. 通过 MCP 客户端向仿真系统下发仿真任务
5. 调用 training-mesh-profiler-skill 分析仿真结果
6. 对比原始组网和等效组网的仿真结果，判断等效性

工作流程：
- 收集参数 → 护栏校验 → 生成拓扑 → 生成模型结构 → 前端渲染 → 用户确认 → 仿真 → 分析 → 对比

请用中文与用户交互。当用户提供参数时，主动进行护栏校验。
当拓扑生成完成，提醒用户在前端检查并确认。
当仿真完成，自动进行对比分析并给出结论。"""

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
                    "device_type": {"type": "string", "description": "设备类型: A2, A3, A5"},
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
            "description": "向仿真系统下发仿真任务（通过 MCP Client）",
            "parameters": {
                "type": "object",
                "properties": {
                    "topology_name": {"type": "string", "description": "组网名称"},
                    "topology_json": {"type": "string", "description": "组网 JSON 字符串"},
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
]


def _build_tools() -> list[dict]:
    """Build combined tool list: skills from registry + utility tools."""
    skill_tools = registry.list_tools()
    return skill_tools + _UTILITY_TOOLS


def _execute_utility_tool(tool_name: str, arguments: dict, session: SessionState) -> dict:
    """Execute a non-skill utility tool (guard check, MCP dispatch, comparison)."""
    if tool_name == "validate_mesh_params":
        result = validate_input_params(
            device_type=arguments["device_type"],
            dp=arguments["dp"],
            tp=arguments["tp"],
            pp=arguments["pp"],
        )
        return result.model_dump()

    elif tool_name == "run_simulation":
        topology_json = json.loads(arguments["topology_json"])
        task_id = mcp_client.execute_task(topology_json)
        topo_name = arguments.get("topology_name", "")
        if "原始" in topo_name:
            session.original_task_id = task_id
        else:
            session.equivalent_task_id = task_id
        return {"task_id": task_id, "status": "submitted", "topology_name": topo_name}

    elif tool_name == "compare_results":
        if not session.original_simulation:
            return {"error": "缺少原始组网仿真结果，请先运行 training-mesh-profiler-skill 获取原始组网仿真数据"}
        if not session.equivalent_simulation:
            return {"error": "缺少等效组网仿真结果，请先运行 training-mesh-profiler-skill 获取等效组网仿真数据"}
        report = _build_comparison_report(session.original_simulation, session.equivalent_simulation)
        session.comparison_report = report
        session.step = "completed"
        return report.model_dump(exclude={"original", "equivalent"})

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
            "guardrail": result.guardrail_result.model_dump() if result.guardrail_result else None,
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
        else:
            session.equivalent_topology = data
            session.equivalent_params = params
        session.step = "topology_generated"
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

    elif tool_name == "training-mesh-profiler-skill" and isinstance(data, SimulationResult):
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
            "aggregate_flops": data.aggregate_flops,
            "aggregate_hbm_gb": data.aggregate_hbm_gb,
            "aggregate_tp_comm_gb_per_micro": data.aggregate_tp_comm_gb_per_micro,
            "aggregate_pp_comm_mb_per_micro": data.aggregate_pp_comm_mb_per_micro,
            "aggregate_dp_comm_gb_per_step": data.aggregate_dp_comm_gb_per_step,
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
            "session_id": session.session_id,
            "_type": "training_model_summary",
            "_role": role,
        }
    return data.model_dump() if hasattr(data, "model_dump") else data


def _build_comparison_report(original: SimulationResult, equivalent: SimulationResult) -> ComparisonReport:
    eps = 1e-9

    def _diff_pct(ov: float, ev: float) -> float:
        return round(abs(ov - ev) / max(abs(ov), eps) * 100, 2)

    of = original.aggregate_flops
    ef = equivalent.aggregate_flops
    oh = original.aggregate_hbm_gb
    eh = equivalent.aggregate_hbm_gb
    otp = original.aggregate_tp_comm_gb_per_micro
    etp = equivalent.aggregate_tp_comm_gb_per_micro
    opp = original.aggregate_pp_comm_mb_per_micro
    epp = equivalent.aggregate_pp_comm_mb_per_micro
    odp = original.aggregate_dp_comm_gb_per_step
    edp = equivalent.aggregate_dp_comm_gb_per_step

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
            "max_diff_pct": round(max(flops_diff, hbm_diff, tp_comm_diff, pp_comm_diff, dp_comm_diff), 2),
        }
    )


# ── Tool name classification ──

_SKILL_TOOLS = {"training-mesh-gen-skill", "training-mesh-profiler-skill", "training-model-gen-skill"}

_SSE_EVENT_MAP = {
    "validate_mesh_params": "guard_check",
    "training-mesh-gen-skill": "mesh_json",
    "training-mesh-profiler-skill": "message",
    "training-model-gen-skill": "model_json",
    "run_simulation": "message",
    "compare_results": "message",
}


async def agent_stream(
    session: SessionState,
    user_message: str,
) -> AsyncGenerator[AgentEvent, None]:
    """Main agent streaming loop. Skills dispatched via registry, utilities handled directly."""
    client = OpenAI(
        api_key=config.OPENAI_API_KEY,
        base_url=config.OPENAI_BASE_URL,
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]
    for h in session.history[-20:]:
        messages.append(h)
    messages.append({"role": "user", "content": user_message})

    tools = _build_tools()
    _msg_size = lambda msgs: sum(len(json.dumps(m, ensure_ascii=False)) for m in msgs)
    logger.info(f"[agent_stream] session={session.session_id} tools={len(tools)} context_size={_msg_size(messages)} chars")

    yield AgentEvent(event_type="thinking", message="正在分析您的请求...")

    max_rounds = 5
    for _round_num in range(max_rounds):
        logger.info(f"[agent_stream] round={_round_num+1} context_size={_msg_size(messages)} chars")
        try:
            response = client.chat.completions.create(
                model=config.OPENAI_MODEL,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.3,
            )
        except Exception as e:
            logger.exception(f"[agent_stream] OpenAI API error at round {_round_num+1}")
            yield AgentEvent(event_type="error", message=f"OpenAI API 异常: {e}")
            break

        msg = response.choices[0].message
        logger.info(f"[agent_stream] round={_round_num+1} finish_reason={response.choices[0].finish_reason} "
                    f"tool_calls={len(msg.tool_calls or [])} content_len={len(msg.content or '')}")

        if not msg.tool_calls:
            if msg.content:
                yield AgentEvent(event_type="message", message=msg.content)
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

        for tool_call in msg.tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)

            # Auto-inject task_id from session if LLM didn't provide one
            if tool_name == "training-mesh-profiler-skill" and not arguments.get("task_id"):
                topo_name = arguments.get("topology_name", "")
                sid = session.original_task_id if "原始" in topo_name else session.equivalent_task_id
                if sid:
                    arguments["task_id"] = sid
                    logger.info(f"[agent_stream] auto-injected task_id={sid} for profiler")

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

            logger.info(f"[agent_stream] tool={tool_name} result_size={len(json.dumps(result, ensure_ascii=False))} chars")

            event_type = _SSE_EVENT_MAP.get(tool_name, "message")

            if "error" in result:
                event_type = "error"
                result_msg = f"执行失败: {result.get('error')}"
            elif tool_name == "validate_mesh_params":
                result_msg = "护栏校验" + ("通过" if result.get("passed") else "失败")
            elif tool_name == "training-mesh-gen-skill":
                result_msg = f"组网 '{result.get('name', '')}' 生成成功"
            elif tool_name == "training-model-gen-skill":
                result_msg = f"模型结构 'transformer_model' 生成成功, {result.get('layers_count', 0)} 层"
            elif tool_name == "training-mesh-profiler-skill":
                agg_flops = result.get('aggregate_flops', 0)
                agg_hbm = result.get('aggregate_hbm_gb', 0)
                result_msg = f"仿真分析完成: {result.get('topology_name', '')} — {result.get('total_nodes', 0)} 节点, 聚合FLOPs {agg_flops:.2e}, HBM {agg_hbm:.1f}GB"
            elif tool_name == "run_simulation":
                result_msg = f"仿真任务已下发: {result.get('task_id', '')}"
            elif tool_name == "compare_results":
                result_msg = f"对比分析完成: {result.get('details', {}).get('conclusion', '')}"
            else:
                result_msg = f"工具 {tool_name} 执行完成"

            yield AgentEvent(
                event_type=event_type,
                message=result_msg,
                data=result,
            )

            tool_content = json.dumps(result, ensure_ascii=False)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_content,
            })

    # Auto-profiling: if topologies have task_ids but no simulation data, run profiler now
    for label, topo, task_id in [
        ("original", session.original_topology, session.original_task_id),
        ("equivalent", session.equivalent_topology, session.equivalent_task_id),
    ]:
        sim = session.original_simulation if label == "original" else session.equivalent_simulation
        if topo and task_id and not sim:
            logger.info(f"[agent_stream] auto-profiling {label} topology (task_id={task_id})")
            try:
                _execute_skill_tool("training-mesh-profiler-skill", {
                    "topology_name": topo.name,
                    "device_type": topo.device_type.value,
                    "total_nodes": topo.total_nodes,
                    "dp": topo.dp_size,
                    "tp": topo.tp_size,
                    "pp": topo.pp_size,
                    "task_id": task_id,
                }, session)
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

    session.history.append({"role": "user", "content": user_message})
    yield AgentEvent(event_type="done", message="处理完成")
