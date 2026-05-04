# TrainMesh Agent — 前端与 Agent 交互逻辑分析

## 一、整体架构

```
┌─────────────────────────────────────────────────────────┐
│  前端 (static/index.html)                                │
│  ┌──────────┐  ┌──────────────┐  ┌───────────────────┐ │
│  │ 历史列表  │  │  D3 拓扑画布  │  │  SSE 聊天面板      │ │
│  │ (左侧)   │  │   (中间)     │  │  (右侧)            │ │
│  └──────────┘  └──────────────┘  └───────────────────┘ │
│                     ▲                    │                │
│                     │ loadMeshData()     │ SSE / WS       │
└─────────────────────┼────────────────────┼────────────────┘
                      │                    │
┌─────────────────────┼────────────────────┼────────────────┐
│  后端 Flask          │                    │                │
│                     │                    ▼                │
│  ┌─────────────────────────────────────────────────────┐ │
│  │  orchestrator.py (Agent 主循环)                      │ │
│  │  ┌──────────┐  ┌────────────┐  ┌────────────────┐  │ │
│  │  │ OpenAI   │  │ Skill      │  │ Utility Tools  │  │ │
│  │  │ LLM 调用 │  │ Registry   │  │ (护栏/MCP/对比) │  │ │
│  │  └──────────┘  └────────────┘  └────────────────┘  │ │
│  └─────────────────────────────────────────────────────┘ │
│  ┌────────────┐  ┌──────────────┐  ┌──────────────────┐ │
│  │ Session    │  │ MCP Client   │  │ Guardrails       │ │
│  │ Manager    │  │ (仿真通信)    │  │ (输入+输出校验)    │ │
│  └────────────┘  └──────────────┘  └──────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

### 文件清单

| 层次 | 文件 | 职责 |
|------|------|------|
| 前端 | `static/index.html` | 单页应用：三栏布局、SSE 消费、D3 拓扑渲染、WebSocket |
| 入口 | `app/main.py` | Flask 应用工厂，注册 Blueprint、WebSocket、静态文件 |
| 路由 | `app/routes/chat.py` | `POST /api/chat/stream` SSE 流式端点 |
| 路由 | `app/routes/session.py` | RESTful 会话 CRUD + 拓扑/仿真数据查询 |
| 路由 | `app/routes/simulation.py` | WebSocket 仿真状态实时推送 |
| 核心 | `app/agent/orchestrator.py` | Agent 主循环：OpenAI 调用、工具编排、SSE 事件流 |
| 核心 | `app/agent/session.py` | 会话管理器（内存字典存储） |
| 核心 | `app/agent/guardrails.py` | 输入/输出护栏校验函数 |
| 模型 | `app/models/schemas.py` | Pydantic 数据模型定义 |
| 技能 | `app/skills/base.py` | BaseSkill 抽象基类 + SkillContext + SkillResult |
| 技能 | `app/skills/registry.py` | SkillRegistry：发现、注册、调度、护栏+重试 |
| 技能 | `app/skills/training-mesh-gen-skill/__init__.py` | 组网拓扑生成 Skill |
| 技能 | `app/skills/training-model-gen-skill/__init__.py` | Transformer 模型结构生成 Skill |
| 技能 | `app/skills/training-mesh-profiler-skill/__init__.py` | 仿真性能分析 Skill |
| 通信 | `app/mcp/client.py` | MCP JSON-RPC 客户端（仿真系统通信） |
| 配置 | `app/config.py` | 环境变量配置（OpenAI / MCP / Flask / 护栏参数） |
| 模拟 | `mock/train.py` | GPT-2 分布式训练脚本（仿真被测对象） |
| 模拟 | `mock/model.py` | GPT 模型定义 |
| 模拟 | `mock/bench.py` | 性能基准测试脚本 |

---

## 二、通信通道

项目使用 **三种通信方式**，各司其职：

| 通道 | 端点 | 方向 | 用途 |
|------|------|------|------|
| **SSE** | `POST /api/chat/stream` | 服务端→客户端 | Agent 主交互流（思考/工具调用/结果/完成） |
| **WebSocket** | `WS /ws/simulation/<id>` | 双向 | 仿真任务状态实时轮询推送 |
| **REST** | `/api/session/*` | 请求-响应 | 会话 CRUD、拓扑数据获取、仿真结果查询 |

### SSE 事件类型（9种）

```
thinking    → Agent 正在分析
tool_call   → 调用某个工具（携带工具名+参数）
guard_check → 护栏校验结果（通过/失败+告警）
mesh_json   → 组网拓扑生成完毕（触发画布渲染）
model_json  → 训练模型结构生成完毕
sim_data    → 仿真数据推送（task_id + 状态）
message     → Agent 文本回复 / 仿真结果 / 对比报告
error       → 异常信息
done        → 处理完成
```

### SSE 数据流转路径

```
用户输入 → POST /api/chat/stream
         → chat.py: generate() 创建 asyncio event loop
         → agent_stream() 异步生成器逐事件 yield AgentEvent
         → 序列化为 JSON
         → 封装为 SSE 格式: "event: {type}\ndata: {json}\n\n"
         → Flask Response(mimetype="text/event-stream")
         → 前端 fetch + ReadableStream.getReader()
         → 按 \n 分割解析 event: / data: 行
         → handleSSEEvent(type, data) 分发处理
```

---

## 三、Agent 主循环 (`orchestrator.py`)

### 3.1 System Prompt

```python
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
```

### 3.2 核心流程图

```
用户消息 + 会话上下文
        │
        ▼
┌─────────────────────────────┐
│ 1. 构造 messages:           │
│    system_prompt            │
│    + session.history[-20:]  │  ← 最近 20 轮对话
│    + user_message           │
│                             │
│ 2. 合并 tools:              │
│    registry.list_tools()    │  ← 3 个 Skill 工具
│    + _UTILITY_TOOLS         │  ← 3 个 Utility 工具
│    = 共 6 个 function calls │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│ 3. OpenAI API 调用          │
│    model: gpt-4o (可配置)    │
│    tool_choice: "auto"      │
│    temperature: 0.3         │
│    max_rounds: 5            │
└──────────┬──────────────────┘
           │
     ┌─────┴──────────┐
     │ finish_reason?  │
     └─────┬──────────┘
    tool_calls  │  stop (无工具调用)
                │
         ┌──────┴──────┐
         │              │
         ▼              ▼
   遍历 tool_calls   yield message
         │           yield done
         │           break
         ▼
┌───────────────────────────────────────┐
│ 4. 逐个执行工具调用:                    │
│                                       │
│  if tool_name in _SKILL_TOOLS:        │
│    └→ SkillRegistry.execute_tool()    │
│        ├ input_guard(args)   ← 输入护栏│
│        ├ execute(args, ctx)  ← 核心逻辑│
│        └ output_guard(data)  ← 输出护栏│
│           └ 失败 → retry (最多3次)     │
│                                       │
│  else (Utility):                      │
│    ├ validate_mesh_params → 护栏校验    │
│    ├ run_simulation      → MCP 下发    │
│    └ compare_results     → 等效对比    │
│                                       │
│ 5. 结果写入 session state             │
│    追加 assistant + tool 消息到 messages│
└──────────┬────────────────────────────┘
           │
           ▼ 回到步骤 3（最多 5 轮）
```

### 3.3 六个工具详解

| 工具名 | 类型 | 功能 | 输入 | SSE 事件 |
|--------|------|------|------|----------|
| `training-mesh-gen-skill` | Skill | 根据 DP/TP/PP 生成结构化 MeshTopology JSON | name, device_type, dp, tp, pp | `mesh_json` |
| `training-model-gen-skill` | Skill | 生成 Transformer 模型结构 JSON，支持等效缩放 | num_layers, d_model, num_heads, d_ffn, pp, is_equivalent | `model_json` |
| `training-mesh-profiler-skill` | Skill | 通过 MCP 获取仿真结果或使用估算模式 | task_id?, topology_name, device_type, total_nodes, dp, tp, pp | `message` |
| `validate_mesh_params` | Utility | 输入护栏校验：设备类型(A2/A3/A5) + DP/TP/PP 范围 | device_type, dp, tp, pp | `guard_check` |
| `run_simulation` | Utility | 通过 MCP Client 向仿真系统下发任务 | topology_name, topology_json | `message` |
| `compare_results` | Utility | 对比原始/等效组网仿真结果，输出等效性报告 | original_json, equivalent_json | `message` |

### 3.4 Skill 工具执行后的 Session 状态更新

```python
# orchestrator.py: _execute_skill_tool()

if tool_name == "training-mesh-gen-skill":
    if "原始" in data.name:
        session.original_topology = data       # 存储原始组网拓扑
        session.original_params = params       # 存储原始组网参数
    else:
        session.equivalent_topology = data     # 存储等效组网拓扑
        session.equivalent_params = params
    session.step = "topology_generated"

elif tool_name == "training-mesh-profiler-skill":
    if "原始" in data.topology_name:
        session.original_simulation = data     # 存储原始组网仿真结果
    else:
        session.equivalent_simulation = data   # 存储等效组网仿真结果
    session.step = "simulating"

elif tool_name == "run_simulation":
    if "原始" in data.get("topology_name", ""):
        session.original_task_id = data["task_id"]   # 存储原始组网 MCP 任务 ID
    else:
        session.equivalent_task_id = data["task_id"]  # 存储等效组网 MCP 任务 ID
    # task_id 用于 profiler-skill 自动注入仿真查询参数

elif tool_name == "training-model-gen-skill":
    if is_equivalent:
        session.equivalent_training_model = data
    else:
        session.original_training_model = data

elif tool_name == "compare_results":
    session.comparison_report = report         # 存储等效性对比报告
    session.step = "completed"
```

### 3.5 等效性对比算法

```python
# orchestrator.py: _build_comparison_report()

def _build_comparison_report(original, equivalent):
    # 计算原始与等效的 5 项差异百分比
    flops_diff_pct  = abs(orig - equiv) / max(abs(orig), eps) * 100   # 计算强度差异
    hbm_diff_pct    = abs(orig - equiv) / max(abs(orig), eps) * 100   # HBM 内存差异
    tp_comm_diff_pct = abs(orig - equiv) / max(abs(orig), eps) * 100  # TP 通信差异
    pp_comm_diff_pct = abs(orig - equiv) / max(abs(orig), eps) * 100  # PP 通信差异
    dp_comm_diff_pct = abs(orig - equiv) / max(abs(orig), eps) * 100  # DP 通信差异

    tolerance = 5.0   # 容忍度 5%

    is_equivalent = (
        flops_diff_pct <= tolerance
        and hbm_diff_pct <= tolerance
        and tp_comm_diff_pct <= tolerance
        and pp_comm_diff_pct <= tolerance
        and dp_comm_diff_pct <= tolerance
    )

    return ComparisonReport(
        flops_diff_pct,
        hbm_diff_pct,
        tp_comm_diff_pct,
        pp_comm_diff_pct,
        dp_comm_diff_pct,
        is_equivalent,
        conclusion: "✅ 等效验证通过" or "❌ 等效验证不通过"
    )
```

---

## 四、Session 状态管理

### 4.1 SessionState 数据模型

```python
class SessionState(BaseModel):
    session_id: str                      # UUID4 前8位
    original_params: TopologyParams      # 原始组网参数 (device_type, dp, tp, pp)
    equivalent_params: TopologyParams    # 等效组网参数
    original_topology: MeshTopology      # 原始组网拓扑（含节点列表+通信域）
    equivalent_topology: MeshTopology    # 等效组网拓扑
    original_simulation: SimulationResult   # 原始组网仿真结果
    equivalent_simulation: SimulationResult # 等效组网仿真结果
    comparison_report: ComparisonReport     # 等效性对比报告
    original_training_model: TrainingModel  # 原始训练模型结构
    equivalent_training_model: TrainingModel # 等效训练模型结构
    original_task_id: str | None = None     # 原始组网 MCP 仿真任务 ID
    equivalent_task_id: str | None = None   # 等效组网 MCP 仿真任务 ID
    step: str = "idle"                  # 状态机步骤
    history: list[dict]                 # 完整对话历史
```

### 4.2 状态机

```
idle ──→ topology_generated ──→ simulating ──→ completed
```

注：`params_collected` 状态在 `SessionState.step` 类型中已定义，但 orchestrator 中无代码路径设置该状态，实际不经过此中间态。

### 4.3 SessionManager

```python
class SessionManager:
    _sessions: dict[str, SessionState]  # 内存字典存储

    create_session()    → 生成8位UUID，创建空SessionState
    get_session(id)     → 按ID查找
    list_sessions()     → 返回所有会话
    delete_session(id)  → 删除会话
    update_session(id)  → 按key更新会话字段
```

注意：会话存储在内存中，重启丢失，无持久化。

---

## 五、Skill 执行机制

### 5.1 BaseSkill 抽象基类

```python
class BaseSkill(ABC):
    name: str                # 唯一标识, e.g. 'training-mesh-gen-skill'
    description: str         # 人类可读描述
    tool_schema: dict        # OpenAI function-calling 工具定义

    input_guard(args)  → GuardrailResult   # 输入护栏（可选覆写）
    execute(args, ctx) → SkillResult       # 核心逻辑（必须实现）
    output_guard(data) → GuardrailResult   # 输出护栏（可选覆写）
```

### 5.2 SkillRegistry 调度流程

```
registry.execute_tool(name, args, context, max_retries=3)
        │
        ├─ 1. 按 name 查找 skill
        │     └─ 未找到 → SkillResult(success=False, error="Unknown skill")
        │
        ├─ 2. skill.input_guard(args)
        │     └─ 不通过 → SkillResult(success=False, error="input_guardrail_failed")
        │
        ├─ 3. skill.execute(args, context)  [最多重试 3 次]
        │     ├─ result.success = False → 不重试，直接返回错误
        │     └─ result.success = True  → 进入输出护栏
        │
        └─ 4. skill.output_guard(result.data)
              ├─ 通过 → 返回 SkillResult(success=True, data=..., guardrail_result=...)
              └─ 失败 → 回到步骤 3 重试
                        └─ 耗尽重试 → SkillResult(success=False, error="output_guardrail_failed_after_retries")
```

### 5.3 Skill 自动发现

```python
def discover_from_directory(directory):
    """
    扫描子目录中的 SKILL.md 文件:
      1. 解析 YAML 前置元数据 (name, description)
      2. 导入对应 Python 模块 (app.skills.{dir_name})
      3. 查找 BaseSkill 子类实例
      4. 注册到 registry
    """
```

### 5.4 三个 Skill 对比

| Skill | 输入护栏 | 核心逻辑 | 输出护栏 | 数据产出 |
|-------|---------|---------|---------|---------|
| `training-mesh-gen-skill` | 校验 device_type + DP/TP/PP 范围 | 遍历 dp×tp×pp 生成 MeshNode 列表 + 通信域分组 | 校验 total_nodes = len(nodes) = dp×tp×pp | `MeshTopology` → 存储到 session |
| `training-model-gen-skill` | 校验 num_layers > 0, d_model % num_heads == 0, 等效缩放整除 | 等效模型: layers = original_layers / pp × 3; 构建 TransformerBlock 列表 | 校验 block_count == num_layers, 首层为 input_embedding | `TrainingModel` → 存储到 session |
| `training-mesh-profiler-skill` | 校验 total_nodes > 0, device_type 合法 | 有 task_id → MCP 查询; 无 task_id → 按设备类型估算 | 无自定义输出护栏（使用默认通过） | `SimulationResult` → 存储到 session |

---

## 六、护栏系统 (`guardrails.py`)

### 6.1 输入护栏 (`validate_input_params`)

```
device_type: 必须在 {A2, A3, A5} 内
dp:          [1, 1024], >512 告警
tp:          [1, 32],   非2的幂次告警
pp:          [1, 128]
world_size:  [1, 100000], >10000 告警
```

### 6.2 输出护栏 (`validate_topology_output`)

```
- topology.name 非空
- topology.total_nodes > 0
- topology.nodes 非空
- len(nodes) == total_nodes
- total_nodes == dp_size × tp_size × pp_size
- 通信域 dp/tp/pp 分组大小与维度的期望值一致
```

---

## 七、MCP 通信 (`mcp/client.py`)

### 7.1 MCPClient

```python
class MCPClient:
    server_url: str                    # MCP Server 地址 (默认 localhost:9000)

    execute_task(topology, params) → task_id    # 下发仿真任务
    get_task_status(task_id)    → status dict   # 查询任务状态
    sync_logs(task_id, offset)  → log lines     # 同步任务日志
    get_result(task_id)         → result dict   # 获取仿真结果
    get_card_details(task_id)   → card list     # 获取单卡详情
    check_health()              → bool          # 健康检查
```

### 7.2 通信协议

使用 **JSON-RPC 2.0** 协议：

```json
{
    "jsonrpc": "2.0",
    "method": "tools/call",
    "params": {
        "name": "execute_task",
        "arguments": { "topology": {...}, "simulation_params": {...} }
    },
    "id": 1
}
```

### 7.3 WebSocket 仿真轮询

```
客户端订阅:  {"type": "subscribe", "task_ids": ["task_orig", "task_eq"]}
服务端循环:
  ├─ 每 1s 轮询 MCP get_task_status()
  ├─ 推送 status 事件 (含进度)
  ├─ 推送 log 事件 (含日志行)
  ├─ 状态为 completed → 推送 complete 事件 (含结果)
  ├─ 状态为 failed    → 推送 error 事件
  ├─ 超时 300s       → 推送 error (timeout)
  └─ 心跳 (120s 无消息时)
```

---

## 八、前端实现细节

### 8.1 三栏布局

```
┌────────────┬──────────────────────┬──────────────┐
│  左侧 300px │      中间 flex:1     │  右侧 440px   │
│  历史列表   │    D3 拓扑画布        │  SSE 聊天面板  │
│            │                      │              │
│  - 任务列表 │  - 单视图模式         │  - 快捷操作   │
│  - 状态徽标 │  - 对比模式(左右并排)  │  - 消息列表   │
│  - 新建按钮 │  - DP 下拉切换        │  - Markdown   │
│            │  - 缩放自适应         │  - 输入框     │
└────────────┴──────────────────────┴──────────────┘
```

### 8.2 SSE 事件处理映射

```javascript
function handleSSEEvent(type, data) {
    switch (type) {
        case 'thinking':
            → addMessage('agent', data.message)  // 思考动画
        case 'tool_call':
            → addMessage('tool', toolName + args) // 工具调用
        case 'guard_check':
            → addMessage(passed ? 'system' : 'error', result) // 校验结果
        case 'mesh_json':
            → addMessage('mesh', summary)  // 拓扑摘要
            → loadMeshData(topoData)       // 触发 D3 渲染
        case 'model_json':
            → setStatus('generated')       // 设置状态
            → addMessage('agent', summary, 'model')  // 显示模型摘要
            → _pendingModelList.push(data)  // 暂存，等 done 事件批量加载
        case 'sim_data':
            → addMessage('system', `仿真任务 ${task_id}: ${status}`)  // 仿真状态
        case 'message':
            → addMessage('agent', marked.parse(content)) // Markdown 渲染
        case 'error':
            → addMessage('error', message)
        case 'done':
            → loadModelData(_pendingModelList)  // 批量加载模型数据
            → setStatus('idle')
    }
}
```

### 8.3 D3 拓扑渲染逻辑

**单视图模式（仅一个组网）：**

```
meshBuildData(tp, pp, dpCount, activeDp)
    │
    ├─ DP 卡片 (外层):
    │   ├─ 三层阴影（层叠视觉效果）
    │   ├─ 主卡片 (圆角矩形)
    │   ├─ Header：DP 下拉选择器 + 信息文本 (TP×PP×DP | N NPUs)
    │   └─ Body：PP 列列表
    │
    └─ PP 列 (内层):
        ├─ PP Header (橙色)
        └─ TP 行列表 (绿色边框)
            └─ 每行: TP 标签 + Rank 标签

PP 列超 8 个时：显示前3 + "..." + 后4 折叠模式
```

**对比模式（两个组网并排）：**

```
meshRebuild() 检测 meshOriginal && meshEquivalent 都存在时:
    ├─ 隐藏配置 Toolbar
    ├─ 按 DP 宽度比例分配左右宽度 (40%~60%)
    ├─ 统一双方缩放比例
    ├─ 左侧标题 + _meshBuildView(原始)
    └─ 右侧标题 + _meshBuildView(等效)
```

### 8.4 数据从 Agent 到画布的桥接

```javascript
// Agent 产出的 MeshTopology 通过 SSE mesh_json 事件到达前端时:
// 注: 当前前端 mesh_json 处理中访问 data.data || data 来提取拓扑数据

function loadMeshData(topoData) {
    // 从 SSE 数据中提取关键字段:
    tp = topoData.tp_size || topoData.tp
    pp = topoData.pp_size || topoData.pp
    dp = topoData.dp_size || topoData.dp
    name = topoData.name → 判断 "原始" / "等效"

    if (name 包含 "原始") → meshOriginal = entry
    else                  → meshEquivalent = entry

    meshRebuild()  // 重新渲染画布
}
```

---

## 九、完整交互时序

以一次典型用户请求为例：

```
用户发送: "A3 DP=8 TP=4 PP=2"
         │
         ▼
┌─ SSE Stream ─────────────────────────────────────────────┐
│                                                          │
│ event: thinking    data: {"message":"正在分析您的请求..."}  │
│ event: tool_call   data: {"tool_name":"validate_mesh_params",...}
│ event: guard_check data: {"passed":true,...}              │
│ event: tool_call   data: {"tool_name":"training-mesh-gen-skill",...}
│ event: mesh_json   data: {"name":"原始组网","total_nodes":64,...}
│        → 前端 loadMeshData() → meshRebuild() 渲染 D3 拓扑 │
│ event: tool_call   data: {"tool_name":"training-model-gen-skill",...}
│ event: model_json  data: {"type":"transformer_model",...} │
│ event: message     data: {"message":"请在前端检查并确认"}   │
│ event: tool_call   data: {"tool_name":"run_simulation",...}
│ event: message     data: {"task_id":"task_xxx","status":"submitted"}
│ event: sim_data    data: {"task_id":"task_xxx","status":"running",...}     │
│ event: tool_call   data: {"tool_name":"training-mesh-profiler-skill",...}
│ event: sim_data    data: {"task_id":"task_xxx","status":"completed",...}   │
│ event: message     data: {"aggregate_compute_intensity_tflops":...}
│ event: tool_call   data: {"tool_name":"compare_results",...}
│ event: message     data: {"is_equivalent":true,"details":{"conclusion":"✅ 等效验证通过"}}
│ event: done        data: {"message":"处理完成"}           │
│                                                          │
└──────────────────────────────────────────────────────────┘
         │
         ▼
   前端恢复 idle 状态，等待下一条用户输入
```

---

## 十、关键设计决策与注意事项

### 10.1 优点

- **LLM 驱动编排**：Agent 自主决策工具调用顺序，灵活应对不同用户请求路径
- **SSE 流式反馈**：用户可实时看到每一步进展（思考→工具调用→结果），体验好
- **护栏+重试**：输入/输出双层校验，输出失败自动重试最多3次，保证数据质量
- **技能可扩展**：SkillRegistry 自动发现机制，新增技能只需创建目录+SKILL.md+Python 模块
- **前端拓扑可视化**：D3.js 渲染 3D 参数空间（DP×TP×PP），支持单视图和对比模式

### 10.2 待改进点

- **会话无持久化**：`SessionManager` 使用内存字典，重启后所有会话丢失
- **无身份认证**：CORS 开放所有来源，无用户身份校验
- **state 机 `params_collected` 为死状态**：`SessionState.step` 类型注释中定义了该状态，但 orchestrator 中无任何代码路径设置它，实际直接从 `idle` 跳到 `topology_generated`
- **MCP 无重试机制**：网络异常时直接返回 error dict，无自动重试
- **WebSocket 轮询而非推送**：仿真状态通过前端订阅+后端轮询 MCP 实现，非真正的事件驱动
- **单文件前端**：`index.html` 约 1100 行，CSS/JS/HTML 混合，无模块化
- **SSE 事件类型声明不一致**：`app/main.py` 的 `api_info()` 端点声明遗漏了 `model_json` 和 `sim_data` 两种事件
