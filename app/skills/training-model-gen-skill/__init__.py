"""
training-model-gen-skill: Generate structured Transformer model JSON from config params.
"""
from app.skills.base import BaseSkill, SkillContext, SkillResult
from app.models.schemas import (
    GuardrailResult,
    InputEmbeddingLayer,
    ModelSubLayer,
    OutputLayer,
    TrainingModel,
    TrainingModelComputed,
    TrainingModelConfig,
    TrainingModelLayer,
)


def _estimate_total_params(
    num_layers: int,
    d_model: int,
    d_ffn: int,
    vocab_size: int,
) -> str:
    attn_params = 4 * d_model * d_model
    ffn_params = 2 * d_model * d_ffn
    embedding_params = vocab_size * d_model
    output_params = vocab_size * d_model
    total = num_layers * (attn_params + ffn_params) + embedding_params + output_params
    billions = total / 1e9
    return f"~{billions:.1f}"


def _build_layers(num_layers: int, num_heads: int, d_head: int, d_ffn: int, activation: str) -> list[TrainingModelLayer]:
    layers: list[TrainingModelLayer] = []

    layers.append(TrainingModelLayer(
        type="input_embedding",
        desc="vocab_size × d_model + Positional Encoding",
    ))

    if num_layers <= 8:
        for i in range(num_layers):
            layers.append(TrainingModelLayer(
                type="transformer_block",
                id=i,
                sub_layers=[
                    ModelSubLayer(
                        type="multi_head_attention",
                        proj=["Q", "K", "V"],
                        heads=num_heads,
                        d_head=d_head,
                    ),
                    ModelSubLayer(type="layer_norm"),
                    ModelSubLayer(
                        type="feed_forward_network",
                        activation=activation,
                        d_ffn=d_ffn,
                    ),
                ],
                skip_connection=True,
            ))
    else:
        layers.append(TrainingModelLayer(
            type="transformer_block",
            id=0,
            sub_layers=[
                ModelSubLayer(
                    type="multi_head_attention",
                    proj=["Q", "K", "V"],
                    heads=num_heads,
                    d_head=d_head,
                ),
                ModelSubLayer(type="layer_norm"),
                ModelSubLayer(
                    type="feed_forward_network",
                    activation=activation,
                    d_ffn=d_ffn,
                ),
            ],
            skip_connection=True,
        ))
        layers.append(TrainingModelLayer(
            type="ellipsis",
            desc=f"{num_layers - 2} layers identical to block 0",
        ))
        layers.append(TrainingModelLayer(
            type="transformer_block",
            id=num_layers - 1,
            sub_layers=[
                ModelSubLayer(
                    type="multi_head_attention",
                    proj=["Q", "K", "V"],
                    heads=num_heads,
                    d_head=d_head,
                ),
                ModelSubLayer(type="layer_norm"),
                ModelSubLayer(
                    type="feed_forward_network",
                    activation=activation,
                    d_ffn=d_ffn,
                ),
            ],
            skip_connection=True,
        ))

    return layers


class TrainingModelGenSkill(BaseSkill):
    name = "training-model-gen-skill"
    description = (
        "根据模型配置参数(num_layers)生成结构化Transformer等效训练模型JSON，支持自定义d_model/num_heads/d_ffn等参数。"
        "传入 is_equivalent=true + pp 时自动按公式缩放等效模型层数: actual_layers = num_layers / pp * 3。"
        "当用户需要生成训练模型、构建模型架构、定义Transformer结构时触发。"
    )

    @property
    def tool_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "num_layers": {
                            "type": "integer",
                            "description": "Transformer 层数 (L), 必填。等效模型时为原始模型的层数",
                        },
                        "d_model": {
                            "type": "integer",
                            "description": "隐藏维度, 默认 4096",
                        },
                        "num_heads": {
                            "type": "integer",
                            "description": "注意力头数, 默认 32",
                        },
                        "d_ffn": {
                            "type": "integer",
                            "description": "FFN 隐藏层维度, 默认 11008",
                        },
                        "vocab_size": {
                            "type": "integer",
                            "description": "词表大小, 默认 32000",
                        },
                        "activation": {
                            "type": "string",
                            "description": "激活函数, 默认 GELU",
                        },
                        "pp": {
                            "type": "integer",
                            "description": "原始组网流水线并行度, 用于等效模型层数缩放: actual = num_layers / pp * 3",
                        },
                        "is_equivalent": {
                            "type": "boolean",
                            "description": "是否为等效模型, 默认 false",
                        },
                    },
                    "required": ["num_layers"],
                },
            },
        }

    def input_guard(self, arguments: dict) -> GuardrailResult:
        errors = []
        num_layers = arguments.get("num_layers", 0)
        if not isinstance(num_layers, int) or num_layers <= 0:
            errors.append(f"num_layers 必须为正整数, 收到: {num_layers}")
        d_model = arguments.get("d_model")
        num_heads = arguments.get("num_heads")
        if d_model is not None and num_heads is not None:
            if d_model > 0 and num_heads > 0 and d_model % num_heads != 0:
                errors.append(f"d_model ({d_model}) 必须可被 num_heads ({num_heads}) 整除")
        is_equivalent = arguments.get("is_equivalent", False)
        if is_equivalent:
            pp = arguments.get("pp", 1)
            if pp <= 0:
                errors.append(f"等效模型时 pp 必须 > 0, 收到: {pp}")
            elif num_layers > 0 and num_layers % pp != 0:
                errors.append(f"等效缩放: num_layers ({num_layers}) 必须可被 pp ({pp}) 整除")
        return GuardrailResult(passed=len(errors) == 0, errors=errors)

    def output_guard(self, result) -> GuardrailResult:
        if not isinstance(result, TrainingModel):
            return GuardrailResult(passed=False, errors=["Output is not a TrainingModel"])
        errors = []
        if result.config.num_layers <= 0:
            errors.append("num_layers must be > 0")
        if result.computed.d_head <= 0:
            errors.append("d_head must be > 0")
        expected_layers = result.config.num_layers + 1
        if result.config.num_layers > 8:
            actual_layers = sum(1 for l in result.layers if l.type == "transformer_block") + sum(
                1 for l in result.layers if l.type == "ellipsis"
            ) + result.config.num_layers
            pass
        else:
            block_count = sum(1 for l in result.layers if l.type == "transformer_block")
            if block_count != result.config.num_layers:
                errors.append(
                    f"transformer_block count ({block_count}) != num_layers ({result.config.num_layers})"
                )
        if result.layers and result.layers[0].type != "input_embedding":
            errors.append("First layer must be input_embedding")
        return GuardrailResult(passed=len(errors) == 0, errors=errors)

    def execute(self, arguments: dict, context: SkillContext) -> SkillResult:
        num_layers_input = int(arguments["num_layers"])
        is_equivalent = bool(arguments.get("is_equivalent", False))
        pp = int(arguments.get("pp", 1))

        if is_equivalent and pp > 1:
            num_layers = (num_layers_input // pp) * 3
        else:
            num_layers = num_layers_input

        d_model = int(arguments.get("d_model", 4096))
        num_heads = int(arguments.get("num_heads", 32))
        d_ffn = int(arguments.get("d_ffn", 11008))
        vocab_size = int(arguments.get("vocab_size", 32000))
        activation = arguments.get("activation", "GELU")

        d_head = d_model // num_heads
        total_params = _estimate_total_params(num_layers, d_model, d_ffn, vocab_size)

        config = TrainingModelConfig(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_ffn=d_ffn,
        )

        computed = TrainingModelComputed(
            d_head=d_head,
            total_params_billions=total_params,
        )

        layers = _build_layers(num_layers, num_heads, d_head, d_ffn, activation)

        model = TrainingModel(
            type="transformer_model",
            config=config,
            computed=computed,
            layers=layers,
            output_layer=OutputLayer(),
        )

        return SkillResult(success=True, data=model)
