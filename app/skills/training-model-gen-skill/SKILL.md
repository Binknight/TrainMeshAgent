---
name: training-model-gen-skill
description: "根据模型配置参数(num_layers, 可选d_model/num_heads/d_ffn)生成结构化Transformer等效训练模型JSON。传入is_equivalent=true+pp时按公式缩放层数: pp<=3保持不变, pp>3时actual_layers=num_layers/pp*3。当用户需要生成训练模型、构建模型架构、定义Transformer结构时触发。"
---
# Training Model Gen Skill

生成等效训练模型的结构化参数描述。

## 输入参数

- **num_layers**: (必填) Transformer 层数 (L)。等效模型时传入原始模型层数
- **d_model**: (可选) 隐藏维度, 默认 4096
- **num_heads**: (可选) 注意力头数, 默认 32
- **d_ffn**: (可选) FFN 隐藏层维度, 默认 11008
- **vocab_size**: (可选) 词表大小, 默认 32000
- **activation**: (可选) 激活函数, 默认 GELU
- **pp**: (可选) 原始组网流水线并行度, 用于等效缩放。pp≤3 时等效层数保持原始层数不变
- **is_equivalent**: (可选) 是否为等效模型, 默认 false
- **model_type**: (可选) 模型类型 `dense`(默认) / `sparse`(MoE)
- **num_experts**: (可选) MoE: 每层专家数 E
- **moe_router_topk**: (可选) MoE: 每 token 激活的 Top-K 专家数
- **num_moe_layers**: (可选) MoE: MoE 层数 (其余层为 Dense FFN)
- **moe_ffn_hidden_size**: (可选) MoE: 单个专家的 FFN 隐藏维度
- **moe_layer_freq**: (可选) MoE: 层分布模式, 如 `[0]*3+[1]*58`、`([0,1]*24)`; 缺省时由 MCP 按 num_moe_layers 推导
- **shared_expert_intermediate_size**: (可选) MoE: 共享专家 FFN 隐藏维度 (has_shared_expert=true 时建议提供)
- **has_shared_expert**: (可选) MoE: 是否有共享专家, 默认 false
- **expert_tensor_parallel_size**: (可选) MoE: 专家内部张量并行度, 默认 1

> MoE 参数仅在 `model_type=sparse` 时写入 config；dense 模型一律置空，行为与旧版一致。

## 等效模型缩放

等效模型层数计算规则:
- pp ≤ 3: 等效模型层数 = 原始模型层数 (保持不变)
- pp > 3: 等效模型层数 = 原始模型层数 / pp × 3

MoE (model_type=sparse) 的补充规则:
- 层数被缩减时 `num_moe_layers` 同步缩减, 保持「非 MoE 层数不变」: `L_moe_eq = L_eq - (L - L_moe)`
- 此时原 L 上的 `moe_layer_freq` 表达式不再适用于新层数, 置空后由 MCP 侧按新的 `num_moe_layers` 推导
  (前 L-L_moe 层 Dense、其余 MoE)

示例:
- pp=1,2,3: is_equivalent=true, num_layers=16, pp=2 → 输出 16 层 (pp≤3, 保持不变)
- pp>3: is_equivalent=true, num_layers=16, pp=4 → 输出 16/4×3 = 12 层
- 原始模型: is_equivalent=false, num_layers=16 → 输出 16 层
- num_layers 必须可被 pp 整除

## 输出

结构化的 JSON 模型对象，包含：
- **config**: 模型超参数配置 (4 核心字段)
- **computed**: 自动推导字段 (d_head = d_model / num_heads, 估算参数量)
- **layers**: 逐层结构描述 (Input Embedding → Transformer Blocks → Output)
  - 当 L > 8 时, 中间层使用 ellipsis 缩写
- **output_layer**: 输出层描述

## 护栏

- 输入护栏: 校验 num_layers 为正整数, d_model 可被 num_heads 整除; 等效时校验 num_layers 可被 pp 整除
- 输出护栏: 校验 JSON 结构完整性、层数与 num_layers 一致
