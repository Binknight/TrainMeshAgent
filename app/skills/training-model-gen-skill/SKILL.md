---
name: training-model-gen-skill
description: 根据模型配置参数(num_layers, 可选d_model/num_heads/d_ffn)生成结构化Transformer等效训练模型JSON。传入is_equivalent=true+pp时按公式缩放层数：actual_layers=num_layers/pp*3。当用户需要生成训练模型、构建模型架构、定义Transformer结构时触发。
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
- **pp**: (可选) 原始组网流水线并行度, 用于等效缩放
- **is_equivalent**: (可选) 是否为等效模型, 默认 false

## 等效模型缩放

等效模型层数 = 原始模型层数 / 原始组网PP × 3

- 原始模型: is_equivalent=false, num_layers=16 → 输出 16 层
- 等效模型: is_equivalent=true, num_layers=16, pp=4 → 输出 16/4×3 = 12 层
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
