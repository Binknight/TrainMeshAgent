---
name: training-mesh-gen-skill
description: 根据设备类型(A2/A3/A5)和并行参数(DP/TP/PP)生成结构化JSON组网拓扑图。当用户需要生成组网、创建网络拓扑、构建训练集群拓扑时触发。
---
# Training Mesh Gen Skill

生成 AI 训练组网的拓扑结构。

## 输入参数

- **name**: 组网名称 (如 "原始组网" / "等效组网")
- **device_type**: 设备类型 (A2, A3, A5)
- **dp**: 数据并行度 (Data Parallel)
- **tp**: 张量并行度 (Tensor Parallel)
- **pp**: 流水线并行度 (Pipeline Parallel)

## 输出

结构化的 JSON 组网对象，包含：
- 节点列表 (每节点的 dp/tp/pp rank, 邻居关系)
- 通信域分组 (DP/TP/PP 维度的通信组)
- 总节点数 = dp × tp × pp

## 护栏

- 输入护栏: 校验设备类型、DP/TP/PP 范围和合法性
- 输出护栏: 校验 JSON 结构完整性、节点数与 dp×tp×pp 一致性
