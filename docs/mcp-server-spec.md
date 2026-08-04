# 仿真系统 MCP Server 需求规格

> 更新时间：2026-08-04  
> 依据代码：`app/mcp/client.py`、`app/routes/session.py`、`app/routes/simulation.py`、`app/skills/training-mesh-profiler-skill/__init__.py`、`app/agent/orchestrator.py`
>
> **2026-08-04 更新（MoE 模型适配）**：在现有接口上适配稀疏/MoE 模型，不新增 tool，仅扩展已有接口——
> `SimulationTaskInput` 新增 EP 及 MoE 模型配置字段（§3.1）、补充 MoE 入参示例（§3.2）、`card_detail` 新增 EP 通信指标（§7）、算子 Trace 补充 MoE 算子命名约定（§8）、训练脚本解析契约新增 MoE 键名（§11）、联调验收清单新增 MoE 用例（§16）。dense 模型不传 MoE 字段，行为与旧版完全兼容。

---

## 1. 目标与范围

本文档定义 TrainMeshAgent 对接"仿真系统 MCP Server"的完整需求规格，覆盖：

- 接口协议与传输约定
- 9 个 MCP tools 完整定义（核心 5 + 详情 3 + 脚本下载 1，含完整入参/出参）
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
| | `ep` | integer | ⬜ 条件必填¹ | 专家并行度 EP（仅 `model_type="sparse"` 时必填） |
| **模型配置** | `model_name` | string | ⬜ 可选 | 模型名称，如 `"Llama-3.1-8B"` / `"DeepSeek-R1"` |
| | `num_layers` | integer | ✅ | Transformer 层数 L |
| | `hidden_dim` | integer | ✅ | 隐藏层维度 H |
| | `num_heads` | integer | ✅ | 注意力头数 A |
| | `d_ffn` | integer | ✅ | Dense FFN 隐藏层维度，默认 14336 |
| | `model_type` | string | ⬜ 可选 | 模型类型：`"dense"`（默认）或 `"sparse"`（MoE）；缺省视为 dense |
| | `num_experts` | integer | ⬜ 条件必填¹ | MoE：每层专家数 E |
| | `moe_router_topk` | integer | ⬜ 条件必填¹ | MoE：每个 token 激活的专家数 Top-K |
| | `num_moe_layers` | integer | ⬜ 条件必填¹ | MoE：MoE 层数 L_moe（等效组网传入该组网的等效缩减值，如 58→18） |
| | `moe_ffn_hidden_size` | integer | ⬜ 条件必填¹ | MoE：单个专家 FFN 中间维度 F_expert |
| | `has_shared_expert` | boolean | ⬜ 可选 | MoE：是否含共享专家（如 DeepSeek-V3/R1），默认 `false` |
| | `expert_tensor_parallel_size` | integer | ⬜ 可选 | MoE：专家内部张量并行度 TP_e，默认 `1` |
| **运行时参数** | `seq_len` | integer | ✅ | 序列长度 S |
| | `batch_size` | integer | ✅ | 总批次大小 B |
| | `micro_batch_size` | integer | ✅ | 微批次大小 b（per pipeline stage micro-batch） |

> **¹ 条件必填**：`model_type="sparse"` 时 `ep`、`num_experts`、`moe_router_topk`、`num_moe_layers`、`moe_ffn_hidden_size` 必须提供；缺失按 §14 参数错误约定返回可读错误，不得静默忽略后按稠密模型执行。`model_type="dense"`（或缺省）时上述 MoE 字段缺失或为 `null`，MCP Server 按稠密模型处理。

> **额外字段**：`MeshTopology.model_dump()` 还会输出 `nodes`（`MeshNode[]`）和 `communication_groups`（通信组列表）。这些是组网拓扑的内部结构，MCP Server 可忽略，但 `execute_task` 的 `topology` 参数中可能包含。后续版本考虑剥离。
>
> **注意**：上表中的 MoE 字段（`model_type` / `ep` / `num_experts` 等）决定仿真语义，MCP Server **不得忽略**——收到 `model_type="sparse"` 时必须以 MoE 模型语义执行仿真（专家 FFN、Top-K 路由、EP 通信等）。

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
    "d_ffn": 14336,
    "seq_len": 2048,
    "batch_size": 32,
    "micro_batch_size": 4
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
    "d_ffn": 14336,
    "seq_len": 2048,
    "batch_size": 8,
    "micro_batch_size": 4
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

#### MoE 模型示例（DeepSeek-R1，Megatron 配置，稀疏）

原始组网与等效组网均携带 MoE 字段。注意等效组网中 `num_layers`、`num_moe_layers`、`batch_size` 为等效缩减值，`ep` 与原始组网保持一致。

##### 原始组网（A3，DP=8，TP=16，PP=8，EP=8，共 1024 卡）

```json
{
  "topology": {
    "name": "原始组网",
    "device_type": "A3",
    "dp_size": 8,
    "tp_size": 16,
    "pp_size": 8,
    "total_nodes": 1024,
    "ep": 8,
    "model_name": "DeepSeek-R1",
    "num_layers": 61,
    "hidden_dim": 7168,
    "num_heads": 128,
    "d_ffn": 18432,
    "vocab_size": 129280,
    "model_type": "sparse",
    "num_experts": 256,
    "moe_router_topk": 8,
    "num_moe_layers": 58,
    "moe_ffn_hidden_size": 2048,
    "has_shared_expert": true,
    "expert_tensor_parallel_size": 1,
    "seq_len": 4096,
    "batch_size": 32,
    "micro_batch_size": 1
  },
  "simulation_params": { }
}
```

##### 等效组网（A3，DP=2，TP=16，PP=3，EP=8，共 96 卡）

```json
{
  "topology": {
    "name": "等效组网",
    "device_type": "A3",
    "dp_size": 2,
    "tp_size": 16,
    "pp_size": 3,
    "total_nodes": 96,
    "ep": 8,
    "model_name": "DeepSeek-R1",
    "num_layers": 21,
    "hidden_dim": 7168,
    "num_heads": 128,
    "d_ffn": 18432,
    "vocab_size": 129280,
    "model_type": "sparse",
    "num_experts": 256,
    "moe_router_topk": 8,
    "num_moe_layers": 18,
    "moe_ffn_hidden_size": 2048,
    "has_shared_expert": true,
    "expert_tensor_parallel_size": 1,
    "seq_len": 4096,
    "batch_size": 8,
    "micro_batch_size": 1
  },
  "simulation_params": { }
}
```

> 等效组网推导：`L_eq = (61/8)×3 = 21`、`PP_eq = 3`、`DP_eq = 2`、`B_eq = 32×2/8 = 8`、`L_moe_eq = 21 - (61-58) = 18`、`EP_eq = EP = 8`。

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
| `ep_comm_gb_per_step` | float | EP 通信量 (GB/step)，MoE（`model_type="sparse"`）任务建议返回；dense 任务可缺失，调用方按 0 兜底 |

---

## 8. get_device_detail — 获取单卡算子级 Trace（支持增量轮询）

**调用场景**：前端点击 Rank 卡片查看详情时，通过 REST 接口触发。

- **一次性模式**：不传 `offset`，返回该卡全部算子，用于仿真已完成后的静态查看
- **增量模式**：传入 `offset`，每次仅返回新增算子。仿真运行过程中每秒轮询，前端渐进渲染算子时序图和负载描述文件表格，直到 `is_complete=true`

> 与 `card_detail` 的区别：`card_detail` 返回卡级别的 7 个汇总指标（轻量，列表页用），`get_device_detail` 返回算子级别的完整 Trace（数据量大，按需点开单个 Rank 时用）。

### 入参

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `task_id` | string | ✅ | 任务 ID |
| `global_rank` | integer | ✅ | 全局 Rank 编号 |
| `offset` | integer | ⬜ 默认 0 | 增量拉取起始位置（算子索引）。0 = 从头返回全量；>0 = 仅返回 `index >= offset` 的算子 |

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
| `operators` | array[OperatorTrace] | ✅ | 算子执行列表（增量模式下仅返回新增部分） |
| `next_offset` | integer | ⬜ | 下次轮询应使用的 offset = 本次返回最后一条算子的 `index + 1`；一次性模式可省略 |
| `is_complete` | bool | ⬜ 默认 true | `false` = 仿真仍在运行，还有数据生成中；`true` = 已返回全量数据 |
| `timeline` | TimelineSummary | ⬜ | 时序汇总统计（建议仅在 `is_complete=true` 时填充完整值） |

### 增量轮询时序

```
仿真开始 → MCP Server 为每个 rank 持续写入算子到 CSV/内存

轮询1: get_device_detail(task_id, rank=0, offset=0)
       → operators[0..11], next_offset=12, is_complete=false
       前端追加 12 条算子到时序图

轮询2: get_device_detail(task_id, rank=0, offset=12)
       → operators[12..28], next_offset=29, is_complete=false
       前端追加 17 条算子

轮询3: get_device_detail(task_id, rank=0, offset=29)
       → operators[29..45], next_offset=46, is_complete=true
       前端追加最后 17 条，渲染完整视图，停止轮询
```

轮询间隔由 TrainMeshAgent 侧控制（建议 1~2 秒），`offset` 从 0 开始，每次用上次返回的 `next_offset` 作为下次的 `offset`。与 `sync_logs`（§5）的增量模式语义一致。

### 兼容性

- 不传 `offset` → 全量返回，`next_offset` 可省略，`is_complete` 默认 `true`，现有逻辑不受影响
- 传 `offset=0` → 等价于不传，全量返回，但必须返回 `next_offset` 和 `is_complete`

**OperatorTrace** 每条记录。字段以仿真系统 CSV 输出为基准，MCP Server 负责直传 CSV 字段 + 补充少量计算字段：

#### CSV 直传字段（MCP Server 从仿真 CSV 读取后原样返回）

| 字段 | 类型 | CSV 列 | 说明 |
|------|------|--------|------|
| `comm_type` | string | ✅ | 算子/通信类型，如 `computation`、`all_reduce`、`send`、`recv` |
| `comm_group` | string\|null | ✅ | 通信组名，如 `tp_group`、`dp_group`；计算类算子为 null |
| `comm_group_size` | int\|null | ✅ | 通信组参与卡数；计算类算子为 null |
| `msg_size` | float\|null | ✅ | 通信消息大小 (bytes)；计算类算子为 null |
| `stage` | string | ✅ | 执行阶段，如 `forward/layer0`、`backward/layer14`、`optimizer`、`init` |
| `dst` | string\|null | ✅ | 目标 rank 或组；计算类算子为 null |
| `src` | string\|null | ✅ | 源 rank 或组；计算类算子为 null |
| `additional` | string\|null | ✅ | 附加说明，如 `matmul`、`seed-sync` |
| `nonblock` | int | ✅ | 是否非阻塞：`1` = 非阻塞，`0` = 阻塞 |
| `wait_n` | int\|null | ✅ | 等待数量 |
| `elapsed_time` | float | ✅ | 仿真系统原始耗时 (微秒)，CSV 列为 `_elapsed_time`，MCP Server 返回时去掉前导下划线 |
| `start_time` | float | ✅ | 算子开始时间 (微秒) |
| `end_time` | float | ✅ | 算子结束时间 (微秒) |
| `single_flops` | float\|null | ✅ | 单算子 FLOPs；通信类算子为 null |

#### MCP Server 计算补充字段（从上述字段推导）

| 字段 | 类型 | 说明 |
|------|------|------|
| `index` | integer | 算子序号（从 0 递增），用于增量 offset 对齐 |
| `operator_name` | string | 算子可读名称，如 `qkv_projection`、`AllReduce_grad`，由 MCP 根据 `comm_type` + `stage` 推导 |
| `data_shape` | string\|null | 数据形状描述，如 `[32,4096,4096] → [32,4096,12288]` |
| `data_type` | string\|null | 数据类型，如 `bf16`、`fp32` |
| `algo_name` | string\|null | 算法名，如 `linear`、`Ring` |
| `duration` | float | = `end_time - start_time` (微秒)，方便前端直接使用 |

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

#### MoE 算子命名约定（可选，MoE 任务适用）

MoE 层（`model_type="sparse"`）的算子建议以 `moe_` 前缀区分于稠密 FFN 算子，便于前端识别与展示。仅为约定，不强制——MCP Server 也可沿用现有 CSV 直传字段结构（`comm_type` 为自由字符串）：

| `operator_name` | `comm_type` | 说明 |
|---|---|---|
| `moe_router` | `computation` | 路由计算（`softmax(TopK(W_r·x))`），`stage` 形如 `forward/layer3` |
| `moe_dispatch` | `all_to_all` / `send`+`recv` | token 分发到专家（EP 通信） |
| `moe_expert_ffn` | `computation` | 专家 FFN（fc1 / activation / fc2） |
| `moe_combine` | `all_to_all` / `send`+`recv` | token 收集合并（EP 通信） |
| `moe_shared_expert_ffn` | `computation` | 共享专家 FFN（`has_shared_expert=true` 时） |

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

## 11. get_training_script — 获取训练脚本

**调用场景**：`execute_task` 下发后，MCP Server 已在服务端生成对应的 pretrain.sh 训练脚本。前端「等效结果」页的「输出训练脚本」按钮，以及组网参数 / 模型结构参数 / 模型训练参数三张对比卡片，均通过本接口获取脚本文本并解析。

> 脚本由 MCP Server 依据 `execute_task` 传入的 `topology` + `simulation_params` 生成，是组网参数、模型结构参数、训练运行时参数的权威载体。TrainMeshAgent 不再 mock 这些参数，统一从本接口返回的 `script_content` 解析。

### 入参

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `task_id` | string | ✅ | `execute_task` 返回的任务 ID（原始组网用 `original_task_id`，等效组网用 `equivalent_task_id`） |

### 出参

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `task_id` | string | ✅ | 回显任务 ID |
| `topology_name` | string | ✅ | 组网名称，`原始组网` / `等效组网`，用于前端标签 |
| `script_path` | string | ⬜ | 服务端脚本路径，如 `/opt/ascend/script/pretrain_xxxx.sh` |
| `script_filename` | string | ⬜ | 建议下载文件名，如 `pretrain_orig.sh` / `pretrain_equiv.sh` |
| `script_content` | string | ✅ | pretrain.sh 完整脚本文本（UTF-8）；TrainMeshAgent 原样作为文件下载，并从中解析下列参数 |

### 脚本须包含的可解析字段（解析契约）

TrainMeshAgent 从 `script_content` 解析以下三组参数，分别填充「等效结果」页的三张对比卡片。脚本可以 bash 变量赋值（`KEY=value`）或启动参数（`--key value`）形式暴露，但下述键名须稳定可识别：

#### 组网参数（对应「组网参数对比」卡）

| 字段 | 说明 | 示例 |
|------|------|------|
| `device_type` / `DEVICE` | 设备类型 | `A3` |
| `dp` / `DP` | 数据并行度 | `8` |
| `tp` / `TP` | 张量并行度 | `16` |
| `pp` / `PP` | 流水线并行度 | `8` |
| `ep` / `EP` / `expert-parallel-size` | 专家并行度（MoE 模型） | `8` |

> `total_nodes` = `dp × tp × pp`，由 TrainMeshAgent 推导，无需脚本暴露。

#### 模型结构参数（对应「模型结构参数对比」卡）

| 字段 | 说明 | 示例 |
|------|------|------|
| `num_layers` / `NUM_LAYERS` | Transformer 层数 L | `64` |
| `d_model` / `D_MODEL` | 隐藏维度 H | `4096` |
| `num_heads` / `NUM_HEADS` | 注意力头数 A | `32` |
| `d_ffn` / `D_FFN` | FFN 维度 | `14336` |
| `vocab_size` / `VOCAB_SIZE` | 词表大小 | `18277` |
| `num_experts` / `NUM_EXPERTS` / `num-experts` | MoE：每层专家数 E | `256` |
| `moe_router_topk` / `MOE_ROUTER_TOPK` / `moe-router-topk` | MoE：Top-K 激活专家数 | `8` |
| `moe_layer_freq` / `MOE_LAYER_FREQ` / `moe-layer-freq` | MoE：层分布模式（支持 `moe_layer_parser` 的 4 种格式：整数全 MoE、周期整数、`-1` 哨兵、Python 列表表达式） | `[0]*3+[1]*58` |
| `num_moe_layers` / `NUM_MOE_LAYERS` / `num-moe-layers` | MoE：MoE 层数（与 `moe_layer_freq` 二选一暴露即可） | `58` |
| `moe_ffn_hidden_size` / `MOE_FFN_HIDDEN_SIZE` / `moe-ffn-hidden-size` | MoE：专家 FFN 维度 F_expert | `2048` |
| `has_shared_expert` / `HAS_SHARED_EXPERT` / `shared-expert` | MoE：是否含共享专家 | `true` |
| `expert_tensor_parallel_size` / `EXPERT_TENSOR_PARALLEL_SIZE` / `expert-tensor-parallel-size` | MoE：专家内部 TP | `1` |

> `d_head` = `d_model / num_heads`、`total_params` 由 TrainMeshAgent 推导（MoE 模型为含专家权重的估算值）。
>
> MoE 模型（`model_type="sparse"`）的脚本**必须**暴露上述 MoE 键（至少 `num_experts`、`moe_router_topk`、`moe_layer_freq` 或 `num_moe_layers`、`moe_ffn_hidden_size`、`ep`）；dense 模型可省略。脚本未暴露的字段前端显示 `—`，不阻塞下载。

#### 训练运行时参数（对应「模型训练参数对比」卡，当前全为 mock）

| 字段 | 说明 | 示例 |
|------|------|------|
| `global_batch_size` / `GLOBAL_BATCH_SIZE` | 全局批次大小 | `2048` |
| `micro_batch_size` / `MICRO_BATCH_SIZE` | 微批次大小 | `1` |
| `seq_length` / `SEQ_LENGTH` | 序列长度 | `4096` |
| `learning_rate` / `LEARNING_RATE` | 学习率 | `1.5e-4` |
| `optimizer` / `OPTIMIZER` | 优化器 | `AdamW` |
| `grad_accum_steps` / `GRAD_ACCUM_STEPS` | 梯度累积步数 | `1` |

> 若脚本未暴露某字段，TrainMeshAgent 在对应卡片显示 `—`，不阻塞下载。

### 调用时序

```
execute_task(topology, simulation_params) → task_id
                                          ↓
                  MCP Server 生成 pretrain.sh（与任务绑定）
                                          ↓
get_training_script(task_id) → script_content + 解析三组参数
                                          ↓
       TrainMeshAgent REST 转发 → 前端文件下载 + 填充三张对比卡
```

`execute_task` 返回 `task_id` 后即可调用，无需等待仿真 `completed`。

---

## 12. 任务状态机

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

## 13. 当前所有 Mock 点及接口归属分析

### 13.1 Mock 点全览

| 编号 | 文件 | Mock 内容 | 对应 REST 接口 |
|------|------|-----------|---------------|
| M1 | `app/routes/session.py:_generate_mock_operators` | 算子级 Trace（算子名、类型、耗时、FLOPs、通信量等） | `GET /api/session/<id>/simulation/<side>/<rank>/detail` |
| M2 | `app/routes/session.py:_generate_mock_hbm_detail` | HBM 分项（权重/梯度/优化器/激活值，单位 GB） | `GET /api/session/<id>/simulation/<side>/<rank>/hbm-detail` |
| M3 | `app/routes/session.py:_generate_mock_comm_detail` | TP/PP/DP 通信详情（次数/参与卡数/单次量/总量） | `GET /api/session/<id>/simulation/<side>/<rank>/tp-comm-detail` |
| M4 | 同上 | PP 通信详情 | `GET /api/session/<id>/simulation/<side>/<rank>/pp-comm-detail` |
| M5 | 同上 | DP 通信详情 | `GET /api/session/<id>/simulation/<side>/<rank>/dp-comm-detail` |
| M6 | `app/routes/session.py:638` | `task_id = "mock_task_id"`（无真实任务时占位）| 兜底值，不需单独接口 |
| M7 | `app/skills/training-mesh-profiler-skill/__init__.py` + `app/routes/session.py:_run_simulation_for_topology` | 无 task_id 时使用本地估算公式代替仿真结果（两处独立副本） | 估算模式，不需接口 |
| M8 | `static/index.html:_renderResultPanel` Card3「模型训练参数」 | `global_batch_size / micro_batch_size / seq_length / learning_rate / optimizer / grad_accum_steps` 全部前端写死（`index.html:6697`） | `GET /api/session/<id>/training-script/<side>`（解析自 MCP 脚本） |
| M9 | `static/index.html:_outputTrainingScript` | 训练脚本前端 mock（torchrun 模板 + alert 弹窗，无真实下载，与 Ascend/Mindspeed 语义不符） | `GET /api/session/<id>/training-script/<side>` |

### 13.2 接口类型归属

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

#### 类型二：纯 MCP Tool（调用外部仿真系统，9 个）

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
| `get_training_script` | §11 | REST 路由 → `mcp_client.get_training_script()` |

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
| `GET /api/session/<id>/training-script/<side>` | 训练脚本文本 + 组网/模型/训练参数（解析自脚本） | `task_id`（按 side 取 original/equivalent） |

> 以下三个 tool 已正式纳入规格，完整定义见 §8 `get_device_detail`、§9 `get_hbm_detail`、§10 `get_comm_detail`。若仿真系统不方便新增独立 Tool，也可将 §8~§10 的数据扩展到 `card_detail` 的每个 card 对象中，TrainMeshAgent 侧按需取用。

---

## 14. 错误处理与兼容性

- 参数错误：返回可识别错误信息（字段缺失/类型错误）
- `task_id` 不存在：明确返回 `task_id not found`
- 未知 tool：返回 `unknown tool`
- 系统异常：统一错误码 + 日志记录
- tool 名称必须稳定，不得随意改名
- `result` 字段必须始终存在
- 新增字段应向后兼容（优先可选字段）

---

## 15. 性能与可靠性建议

- `report_status`、`sync_logs` 响应建议 < 1s
- `get_device_detail`（§8）、`get_hbm_detail`（§9）、`get_comm_detail`（§10）、`get_training_script`（§11）等详情接口响应建议 < 3s
- 支持同时查询多个 `task_id`
- 服务重启后建议能恢复最近任务状态

---

## 16. 联调验收清单

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

- [ ] `get_device_detail`（§8）一次性模式返回全量 `operators[]` + `timeline{}`
- [ ] `get_device_detail`（§8）增量模式 `offset`/`next_offset`/`is_complete` 轮询流程正常
- [ ] `get_hbm_detail`（§9）返回 HBM 四项分解
- [ ] `get_comm_detail`（§10）支持 `comm_type: tp/pp/dp` 三种查询

**脚本下载（M8/M9 解 mock）**

- [ ] `get_training_script`（§11）返回 `script_content`，TrainMeshAgent 可作文件下载
- [ ] 从脚本解析出的组网/模型/训练参数可填充三张对比卡，值与 `execute_task` 入参一致

**MoE 模型适配（§3.1 / §7 / §8 / §11）**

- [ ] `execute_task` 传入 MoE 字段（`model_type="sparse"` + `ep`/`num_experts`/`moe_router_topk`/`num_moe_layers`/`moe_ffn_hidden_size`）成功返回 `task_id`，仿真按 MoE 语义执行
- [ ] `execute_task` 对 sparse 模型缺失 `ep` 等条件必填字段时返回可读参数错误，不 500、不按稠密模型静默执行
- [ ] `card_detail` 对 MoE 任务返回 `ep_comm_gb_per_step`（缺失时调用方按 0 兜底，不报错）
- [ ] `get_device_detail`（§8）对 MoE 任务的算子含 `moe_router` / `moe_dispatch` / `moe_expert_ffn` / `moe_combine` 等命名（或自有命名，能被前端展示）
- [ ] `get_training_script` 对 MoE 任务返回的脚本含 `ep` / `num_experts` / `moe_router_topk` / `moe_layer_freq`（或 `num_moe_layers`）/ `moe_ffn_hidden_size` 等键，可解析填充对比卡
- [ ] dense 模型兼容：不传任何 MoE 字段时行为与旧版一致

**容错**

- [ ] 非法 `task_id` 返回稳定错误，不 500
- [ ] 非法参数返回可读错误，不 500
- [ ] 并发多个 `task_id` 查询不互相干扰

---

## 17. 非目标（当前阶段不要求）

- 不强制 MCP Server 主动推送（调用方为轮询模型）
- 不强制限定 `get_result` 完整 schema（由仿真侧自行扩展）
- 不要求复杂权限系统（可先内网白名单）
- `simulation_params` 字段内容不作限制（当前传空对象即可）
