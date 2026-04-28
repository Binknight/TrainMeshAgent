---
name: training-mesh-profiler-skill
description: 通过MCP获取仿真结果，计算单卡计算强度(TFLOPS)、内存占用(GB)、通信流量(GB/s)等性能指标。当需要分析组网性能、对比仿真结果、验证等效性时触发。
---
# Training Mesh Profiler Skill

分析 AI 训练组网仿真结果的性能指标。

## 输入参数

- **topology_name**: 组网名称
- **device_type**: 设备类型 (A2, A3, A5)
- **total_nodes**: 总节点数
- **dp**: 数据并行度
- **tp**: 张量并行度
- **pp**: 流水线并行度
- **task_id**: (可选) MCP 仿真任务 ID，提供时从仿真系统拉取真实数据

## 输出

- 逐卡指标: 计算强度 (TFLOPS)、内存占用 (GB)、通信流量 (GB/s)
- 聚合指标: 总计算强度、总内存占用、总通信流量

## 工作模式

1. **真实模式**: 提供 task_id 时，通过 MCP Client 从仿真系统获取真实结果
2. **估算模式**: 无 task_id 时，基于设备类型的经验模型估算
