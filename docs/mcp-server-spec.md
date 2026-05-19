# 仿真系统 MCP Server 需求规格

> 更新时间：2026-05-18  
> 依据代码：`app/mcp/client.py`、`app/routes/session.py`、`app/routes/simulation.py`、`app/skills/training-mesh-profiler-skill/__init__.py`、`app/agent/orchestrator.py`

---

## 1. 目标与范围

本文档定义 TrainMeshAgent 对接"仿真系统 MCP Server"的完整需求规格，覆盖：

- 接口协议与传输约定
- 8 个 MCP tools 完整定义（核心 5 + 详情 3，含完整入参/出参）
- 当前所有 mock 点列举及接入方式分类
- 各接口类型归属（纯 REST / 纯 MCP / 两者均需）
- 任务状态机与错误处理约定
- 联调与验收清单

---

## 2. 协议与基础接口

### 2.1 MCP 主调用接口

- 方法：`POST`
- 路径：`/mcp`
- 协议：JSON-RPC 2.0，固定 `method: "tools/call"`

请求结构：

```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "<tool_name>",
    "arguments": { ... }
  },
  "id": 1
}
```

响应约定：

- HTTP 200 + JSON；业务数据在 `result` 字段中
- 失败时返回 JSON-RPC 标准 `error`，并在 `result` 中补充可读错误信息

### 2.2 健康检查接口

- 方法：`GET`
- 路径：`/health`
- 成功条件：HTTP 200（内容不限）

---

## 3. execute_task — 下发仿真任务

### 3.1 仿真任务入参对象 `SimulationTaskInput`

每次调用 `execute_task` 传入一个 `SimulationTaskInput` 对象，作为 `arguments.topology` 的值。  
该对象由 TrainMeshAgent 在 Workflow Step 1 完成后自动组装，包含**组网参数**、**模型参数**和**运行时参数**三部分。

#### 顶层结构

```json
{
  "topology": { <SimulationTaskInput> },
  "simulation_params": { <SimulationRunnerParams> }
}
```

#### `SimulationTaskInput` 完整字段定义

| 分组 | 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|------|
| **组网标识** | `name` | string | ✅ | 组网名称，如 `"原始组网"` / `"等效组网"` |
| | `device_type` | string | ✅ | 设备类型，枚举：`A2` / `A3` / `A5` |
| **并行策略** | `dp_size` | integer | ✅ | 数据并行度 DP |
| | `tp_size` | integer | ✅ | 张量并行度 TP |
| | `pp_size` | integer | ✅ | 流水线并行度 PP |
| | `total_nodes` | integer | ✅ | 总卡数 = `dp_size × tp_size × pp_size` |
| **模型配置** | `model_name` | string | ⬜ 可选 | 模型名称，如 `"Llama-3.1-8B"` |
| | `num_layers` | integer | ✅ | Transformer 层数 L |
| | `hidden_dim` | integer | ✅ | 隐藏层维度 H |
| | `num_heads` | integer | ✅ | 注意力头数 A |
| **运行时参数** | `seq_len` | integer | ✅ | 序列长度 S |
| | `batch_size` | integer | ✅ | 总批次大小 B |

> **额外字段**：`MeshTopology.model_dump()` 还会输出 `nodes`（`MeshNode[]`）和 `communication_groups`（通信组列表）。这些是组网拓扑的内部结构，MCP Server 可忽略，但 `execute_task` 的 `topology` 参数中可能包含。后续版本考虑剥离。

---

### 3.2 真实 Step 1 入参示例

以下为实际使用场景中 Step 1 产出的完整 `execute_task` 调用体。

#### 原始组网（A3，DP=8，TP=16，PP=8，共 1024 卡）

```json
{
  "topology": {
    "name": "原始组网",
    "device_type": "A3",
    "dp_size": 8,
    "tp_size": 16,
    "pp_size": 8,
    "total_nodes": 1024,
    "model_name": "Llama-3.1-8B",
    "num_layers": 64,
    "hidden_dim": 4096,
    "num_heads": 32,
    "seq_len": 2048,
    "batch_size": 32
  },
  "simulation_params": {
    "script_path": "/opt/ascend/script/pretrain_xxxx.sh",
    "epoch_num": 1,
    "model_name": "",
    "device_type": "ASCEND_910B",
    "vocab_size": "18277",
    "frame": "Mindspeed",
    "rank": 0,
    "rank_range": 1023,
    "comp_filepath": "/opt/traffic_modeling/aicm/default.txt",
    "no_time_accumulation": false,
    "level0_config": null,
    "level1_config": null,
    "visual_json_output": true,
    "comm_group_output": true,
    "debug_time": false
  }
}
```

#### 等效组网（A3，DP=2，TP=16，PP=3，共 96 卡）

```json
{
  "topology": {
    "name": "等效组网",
    "device_type": "A3",
    "dp_size": 2,
    "tp_size": 16,
    "pp_size": 3,
    "total_nodes": 96,
    "model_name": "Llama-3.1-8B",
    "num_layers": 24,
    "hidden_dim": 4096,
    "num_heads": 32,
    "seq_len": 2048,
    "batch_size": 32
  },
  "simulation_params": {
    "script_path": "/opt/ascend/script/pretrain_xxxx.sh",
    "epoch_num": 1,
    "model_name": "",
    "device_type": "ASCEND_910B",
    "vocab_size": "18277",
    "frame": "Mindspeed",
    "rank": 0,
    "rank_range": 1023,
    "comp_filepath": "/opt/traffic_modeling/aicm/default.txt",
    "no_time_accumulation": false,
    "level0_config": null,
    "level1_config": null,
    "visual_json_output": true,
    "comm_group_output": true,
    "debug_time": false
  }
}
```

---

### 3.3 出参

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `task_id` | string | ✅ | 本次任务唯一标识，由仿真系统生成；建议格式 `sim_<timestamp>_<rand6>` 或 UUID |
| `status` | string | ⬜ 可选 | 建议返回 `submitted` |

---

## 4. report_status — 状态查询

**调用场景**：前端通过 WebSocket 订阅后，TrainMeshAgent 每秒轮询，实时推送进度。

### 入参

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `task_id` | string | ✅ | `execute_task` 返回的任务 ID |

### 出参

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `status` | string | ✅ | 当前状态，见状态机约定 |
| `progress` | number | ⬜ 可选 | 进度 0~100 |
| `message` | string | ⬜ 可选 | 失败原因或附加说明 |

---

## 5. sync_logs — 日志增量同步

**调用场景**：与 `report_status` 同步轮询，将仿真运行日志逐行推送到前端。

### 入参

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `task_id` | string | ✅ | 任务 ID |
| `offset` | integer | ⬜ 可选，默认 0 | 上次已读偏移，支持增量拉取 |

### 出参

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `lines` | array[string] | ✅ | 本次增量日志行；无新增时返回 `[]` |
| `next_offset` | integer | ⬜ 可选 | 下次调用建议使用的 offset |

---

## 6. get_result — 获取整体仿真结果

**调用场景**：`report_status.status == "completed"` 时由 WebSocket 轮询逻辑触发，结果原样透传到前端。

### 入参

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `task_id` | string | ✅ | 任务 ID |

### 出参

结构可由仿真系统自定义扩展，要求 JSON 可序列化，可被前端直接展示。建议至少包含：

| 字段 | 类型 | 说明 |
|------|------|------|
| `task_id` | string | 回显任务 ID |
| `status` | string | `completed` |
| `summary` | object | 整体指标摘要（自定义）|
| `cards` | array | 每卡详细指标（与 `card_detail` 格式保持一致）|

---

## 7. card_detail — 获取单卡指标

**调用场景**：`training-mesh-profiler-skill` 有 `task_id` 时调用，将结果映射为 `SimulationResult.cards`。

### 入参

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `task_id` | string | ✅ | 任务 ID |
| `card_ids` | array[string] | ⬜ 可选 | 指定卡 ID 列表；为空时返回全部卡 |

### 出参

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `cards` | array[object] | ✅ | 每卡指标列表 |

每个 card 对象字段（缺失字段将被调用方按 0/空值兜底）：

| 字段 | 类型 | 说明 |
|------|------|------|
| `card_id` | string | 如 `card_0` |
| `global_rank` | integer | 全局 rank |
| `flops_per_card` | float | 单卡 FLOPs |
| `hbm_gb` | float | HBM 总占用 (GB) |
| `tp_comm_gb_per_micro` | float | TP 通信量 (GB/micro-step) |
| `pp_comm_mb_per_micro` | float | PP 通信量 (MB/micro-step) |
| `dp_comm_gb_per_step` | float | DP 通信量 (GB/step) |

---

## 8. get_device_detail — 获取单卡算子级 Trace

**调用场景**：前端点击 Rank 卡片查看详情时，通过 REST 接口触发。返回该卡全部算子的执行时序和 Timeline 摘要，用于渲染**算子时序图**和**负载描述文件表格**。

> 与 `card_detail` 的区别：`card_detail` 返回卡级别的 7 个汇总指标（轻量，列表页用），`get_device_detail` 返回算子级别的完整 Trace（数据量大，按需点开单个 Rank 时用）。

### 入参

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `task_id` | string | ✅ | 任务 ID |
| `global_rank` | integer | ✅ | 全局 Rank 编号 |

### 出参

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `card_id` | string | ✅ | 卡标识，如 `card_0` |
| `global_rank` | integer | ✅ | 全局 Rank |
| `task_id` | string | ✅ | 回显任务 ID |
| `topology_name` | string | ✅ | 组网名称 |
| `device_type` | string | ✅ | 设备类型 |
| `dp_rank` | integer | ⬜ | DP 维度 rank |
| `tp_rank` | integer | ⬜ | TP 维度 rank |
| `pp_rank` | integer | ⬜ | PP 维度 rank |
| `operators` | array[OperatorTrace] | ✅ | 算子执行列表 |
| `timeline` | TimelineSummary | ⬜ | 时序汇总统计 |

**OperatorTrace** 每条记录：

| 字段 | 类型 | 说明 |
|------|------|------|
| `op_name` | string | 算子名称，如 `MHA_QKV_Proj`、`AllReduce` |
| `op_type` | string | 类型枚举：`computation` / `communication` / `collective` |
| `category` | string | 阶段分类：`fwd` / `bwd` / `optimizer` |
| `start_us` | float | 开始时间 (微秒) |
| `duration_us` | float | 持续时间 (微秒) |
| `flops` | float | 计算量 (FLOPs)，计算类算子填充 |
| `comm_bytes` | float | 通信量 (bytes)，通信类算子填充 |
| `parent_op` | string | 父算子名称，用于 tracing 层次 |
| `depth` | integer | 层次深度 |
| `details` | object | 额外信息键值对 |

**TimelineSummary**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `total_time_ms` | float | 总耗时 (ms) |
| `compute_time_ms` | float | 纯计算耗时 (ms) |
| `comm_time_ms` | float | 纯通信耗时 (ms) |
| `compute_pct` | float | 计算占比 (%) |
| `comm_pct` | float | 通信占比 (%) |
| `total_flops` | float | 总 FLOPs |
| `total_comm_gb` | float | 总通信量 (GB) |

---

## 9. get_hbm_detail — 获取单卡 HBM 分项占用

**调用场景**：前端展示单卡 HBM 内存分解（权重 / 梯度 / 优化器 / 激活值）。

### 入参

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `task_id` | string | ✅ | 任务 ID |
| `global_rank` | integer | ✅ | 全局 Rank 编号 |

### 出参

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `global_rank` | integer | ✅ | 全局 Rank |
| `weights_gb` | float | ✅ | 权重占用 (GB) |
| `gradients_gb` | float | ✅ | 梯度占用 (GB) |
| `optimizer_gb` | float | ✅ | 优化器状态占用 (GB) |
| `activations_gb` | float | ✅ | 激活值占用 (GB) |
| `total_hbm_gb` | float | ✅ | HBM 总占用 (GB) |

---

## 10. get_comm_detail — 获取单卡通信详情

**调用场景**：前端展示单卡 TP / PP / DP 通信详情（通信次数、参与卡数、单次/总量）。

### 入参

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `task_id` | string | ✅ | 任务 ID |
| `global_rank` | integer | ✅ | 全局 Rank 编号 |
| `comm_type` | string | ✅ | 通信类型枚举：`tp` / `pp` / `dp` |

### 出参

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `global_rank` | integer | ✅ | 全局 Rank |
| `comm_type` | string | ✅ | 回显通信类型 |
| `comm_count` | integer | ✅ | 每 step 通信次数 |
| `comm_cards` | integer | ✅ | 参与通信的卡数 |
| `comm_size_per_time_gb` | float | ✅ | 单次通信量 (GB) |
| `total_comm_gb` | float | ✅ | 总通信量 (GB) |

---

## 11. 任务状态机

```
submitted  →  running  →  completed
                       ↘  failed
                       ↘  error
```

约束：

- `completed` 时 `get_result(task_id)` 必须可用
- `failed` / `error` 时 `report_status.message` 应给出可读原因
- `progress` 应单调非递减
- 轮询超时阈值（TrainMeshAgent 侧）：**300 秒**，超时后前端收到 `error` 事件

---

## 12. 当前所有 Mock 点及接口归属分析

### 12.1 Mock 点全览

| 编号 | 文件 | Mock 内容 | 对应 REST 接口 |
|------|------|-----------|---------------|
| M1 | `app/routes/session.py:_generate_mock_operators` | 算子级 Trace（算子名、类型、耗时、FLOPs、通信量等） | `GET /api/session/<id>/simulation/<side>/<rank>/detail` |
| M2 | `app/routes/session.py:_generate_mock_hbm_detail` | HBM 分项（权重/梯度/优化器/激活值，单位 GB） | `GET /api/session/<id>/simulation/<side>/<rank>/hbm-detail` |
| M3 | `app/routes/session.py:_generate_mock_comm_detail` | TP/PP/DP 通信详情（次数/参与卡数/单次量/总量） | `GET /api/session/<id>/simulation/<side>/<rank>/tp-comm-detail` |
| M4 | 同上 | PP 通信详情 | `GET /api/session/<id>/simulation/<side>/<rank>/pp-comm-detail` |
| M5 | 同上 | DP 通信详情 | `GET /api/session/<id>/simulation/<side>/<rank>/dp-comm-detail` |
| M6 | `app/routes/session.py:614` | `task_id = "mock_task_id"`（无真实任务时占位）| 兜底值，不需单独接口 |
| M7 | `app/skills/training-mesh-profiler-skill/__init__.py` + `app/routes/session.py:_run_simulation_for_topology` | 无 task_id 时使用本地估算公式代替仿真结果（两处独立副本） | 估算模式，不需接口 |

### 12.2 接口类型归属

#### 类型一：纯 REST API（TrainMeshAgent 内部，无需 MCP）

| 接口 | 说明 |
|------|------|
| `POST /api/session` | 创建 session，生成 session_id |
| `GET /api/session/summaries` | 列出所有 session 摘要 |
| `GET /api/session/<id>` | 获取 session 状态 |
| `DELETE /api/session/<id>` | 删除 session |
| `GET /api/session/<id>/topology` | 获取已生成的拓扑 JSON（本地 session 数据）|
| `GET /api/session/<id>/simulation` | 获取比较报告（本地计算结果）|
| `POST /api/session/estimate` | 用本地估算公式计算指标（无需外部系统）|
| `POST /api/chat/stream` (SSE) | Agent 对话流（本地 orchestrator 驱动）|

#### 类型二：纯 MCP Tool（调用外部仿真系统，8 个）

| MCP Tool | 章节 | 触发路径 |
|----------|------|---------|
| `execute_task` | §3 | `run_simulation` utility → `mcp_client.execute_task()` |
| `report_status` | §4 | WebSocket 轮询 → `mcp_client.get_task_status()` |
| `sync_logs` | §5 | WebSocket 轮询 → `mcp_client.sync_logs()` |
| `get_result` | §6 | WebSocket `completed` 事件 → `mcp_client.get_result()` |
| `card_detail` | §7 | profiler skill → `mcp_client.get_card_details()` |
| `get_device_detail` | §8 | REST 路由 → `mcp_client.get_device_detail()` |
| `get_hbm_detail` | §9 | REST 路由 → `mcp_client.get_hbm_detail()` |
| `get_comm_detail` | §10 | REST 路由 → `mcp_client.get_comm_detail()` |

#### 类型三：需同时满足（REST 入口 + MCP 数据源）

以下接口当前全为 Mock，上线后需由 MCP Server 提供真实数据，TrainMeshAgent REST 负责转发/格式化给前端：

| REST 接口（TrainMeshAgent） | 需要的 MCP 数据 | 入参（TrainMeshAgent → MCP）|
|----------------------------|----------------|---------------------------|
| `GET /api/session/<id>/simulation/<side>/<rank>/detail` | 算子 Trace、Timeline | `task_id` + `global_rank` |
| `GET /api/session/<id>/simulation/<side>/<rank>/hbm-detail` | HBM 分项占用 | `task_id` + `global_rank` |
| `GET /api/session/<id>/simulation/<side>/<rank>/tp-comm-detail` | TP 通信详情 | `task_id` + `global_rank` |
| `GET /api/session/<id>/simulation/<side>/<rank>/pp-comm-detail` | PP 通信详情 | `task_id` + `global_rank` |
| `GET /api/session/<id>/simulation/<side>/<rank>/dp-comm-detail` | DP 通信详情 | `task_id` + `global_rank` |
| `POST /api/session/<id>/run-simulation`（直接仿真） | `card_detail`（整卡指标） | `task_id`（fire-and-forget）|
| WebSocket `/ws/simulation/<id>` | `report_status` + `sync_logs` + `get_result` | `task_id` 列表 |

> 以下三个 tool 已正式纳入规格，完整定义见 §8 `get_device_detail`、§9 `get_hbm_detail`、§10 `get_comm_detail`。若仿真系统不方便新增独立 Tool，也可将 §8~§10 的数据扩展到 `card_detail` 的每个 card 对象中，TrainMeshAgent 侧按需取用。

---

## 13. 错误处理与兼容性

- 参数错误：返回可识别错误信息（字段缺失/类型错误）
- `task_id` 不存在：明确返回 `task_id not found`
- 未知 tool：返回 `unknown tool`
- 系统异常：统一错误码 + 日志记录
- tool 名称必须稳定，不得随意改名
- `result` 字段必须始终存在
- 新增字段应向后兼容（优先可选字段）

---

## 14. 性能与可靠性建议

- `report_status`、`sync_logs` 响应建议 < 1s
- `get_device_detail`（§8）、`get_hbm_detail`（§9）、`get_comm_detail`（§10）等详情接口响应建议 < 3s
- 支持同时查询多个 `task_id`
- 服务重启后建议能恢复最近任务状态

---

## 15. 联调验收清单

**基础连通**

- [ ] `GET /health` 返回 200
- [ ] `POST /mcp` 未知 tool 返回明确错误

**核心链路**

- [ ] `execute_task` 成功返回 `task_id`（格式稳定）
- [ ] `report_status` 可从 `submitted` → `running` → `completed` 完整流转
- [ ] `sync_logs` 可返回增量日志（空日志合法返回 `[]`）
- [ ] 终态 `completed` 后 `get_result` 返回完整结果
- [ ] `card_detail` 返回 `cards`，字段可映射到前端 tooltip 指标

**详情数据（M1~M5 解 mock）**

- [ ] `get_device_detail`（§8）返回 `operators[]` + `timeline{}`（或扩展 card_detail）
- [ ] `get_hbm_detail`（§9）返回 HBM 四项分解
- [ ] `get_comm_detail`（§10）支持 `comm_type: tp/pp/dp` 三种查询

**容错**

- [ ] 非法 `task_id` 返回稳定错误，不 500
- [ ] 非法参数返回可读错误，不 500
- [ ] 并发多个 `task_id` 查询不互相干扰

---

## 16. 非目标（当前阶段不要求）

- 不强制 MCP Server 主动推送（调用方为轮询模型）
- 不强制限定 `get_result` 完整 schema（由仿真侧自行扩展）
- 不要求复杂权限系统（可先内网白名单）
- `simulation_params` 字段内容不作限制（当前传空对象即可）
