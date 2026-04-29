from __future__ import annotations
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


class DeviceType(str, Enum):
    A2 = "A2"
    A3 = "A3"
    A5 = "A5"


class TopologyParams(BaseModel):
    """Input parameters for mesh topology generation."""
    device_type: DeviceType = Field(description="设备类型: A2 / A3 / A5")
    dp: int = Field(ge=1, le=1024, description="数据并行度 (Data Parallel)")
    tp: int = Field(ge=1, le=32, description="张量并行度 (Tensor Parallel)")
    pp: int = Field(ge=1, le=128, description="流水线并行度 (Pipeline Parallel)")


class MeshNode(BaseModel):
    """A single node in the mesh topology graph."""
    id: str
    device_type: DeviceType
    dp_rank: int
    tp_rank: int
    pp_rank: int
    global_rank: int
    neighbors: list[str] = Field(default_factory=list)


class MeshTopology(BaseModel):
    """Structured JSON mesh topology output."""
    name: str = Field(description="组网名称, e.g. '原始组网' or '等效组网'")
    device_type: DeviceType
    total_nodes: int
    dp_size: int
    tp_size: int
    pp_size: int
    nodes: list[MeshNode]
    communication_groups: dict[str, list[list[int]]] = Field(
        default_factory=dict,
        description="通信域分组: key为 dp/tp/pp, value为节点rank列表"
    )


class GuardrailResult(BaseModel):
    """Result of a guardrail check."""
    passed: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class CardMetrics(BaseModel):
    """Per-card simulation metrics."""
    card_id: str
    global_rank: int
    flops_per_card: float = Field(description="单卡FLOPs")
    hbm_gb: float = Field(description="HBM占用 (GB)")
    tp_comm_gb_per_micro: float = Field(description="TP通信量 (GB/micro)")
    pp_comm_mb_per_micro: float = Field(description="PP通信量 (MB/micro)")
    dp_comm_gb_per_step: float = Field(description="DP通信量 (GB/step)")


class SimulationResult(BaseModel):
    """Complete simulation result for a topology."""
    topology_name: str
    device_type: DeviceType
    total_nodes: int
    cards: list[CardMetrics]
    aggregate_flops: float
    aggregate_hbm_gb: float
    aggregate_tp_comm_gb_per_micro: float
    aggregate_pp_comm_mb_per_micro: float
    aggregate_dp_comm_gb_per_step: float


class ComparisonReport(BaseModel):
    """Comparison report between original and equivalent topology."""
    original: SimulationResult
    equivalent: SimulationResult
    flops_diff_pct: float
    hbm_diff_pct: float
    tp_comm_diff_pct: float
    pp_comm_diff_pct: float
    dp_comm_diff_pct: float
    is_equivalent: bool
    error_tolerance_pct: float = 5.0
    details: dict[str, Any] = Field(default_factory=dict)


class AgentEvent(BaseModel):
    """SSE event emitted during agent processing."""
    event_type: str  # thinking | tool_call | guard_check | mesh_json | message | error | done
    data: dict[str, Any] = Field(default_factory=dict)
    message: str = ""


class ModelSubLayer(BaseModel):
    """A sub-layer within a transformer block."""
    type: str = Field(description="Sub-layer type: multi_head_attention, layer_norm, feed_forward_network")
    proj: list[str] | None = Field(default=None, description="Projection matrices, e.g. ['Q','K','V']")
    heads: int | None = Field(default=None, description="Number of attention heads")
    d_head: int | None = Field(default=None, description="Dimension per head")
    activation: str | None = Field(default=None, description="Activation function, e.g. GELU")
    d_ffn: int | None = Field(default=None, description="FFN hidden dimension")


class TransformerBlock(BaseModel):
    """A single transformer block (layer)."""
    type: str = Field(default="transformer_block")
    id: int = Field(description="Layer index (0-based)")
    sub_layers: list[ModelSubLayer] = Field(default_factory=list)
    skip_connection: bool = True


class InputEmbeddingLayer(BaseModel):
    """Input embedding layer description."""
    type: str = Field(default="input_embedding")
    desc: str = Field(default="vocab_size × d_model + Positional Encoding")


class OutputLayer(BaseModel):
    """Output layer description."""
    type: str = Field(default="output")
    desc: str = Field(default="LayerNorm → Linear (d_model → vocab_size)")


class TrainingModelConfig(BaseModel):
    """Training model hyper-parameters (core config only)."""
    num_layers: int = Field(ge=1, description="Number of transformer layers")
    d_model: int = Field(ge=1, description="Model hidden dimension")
    num_heads: int = Field(ge=1, description="Number of attention heads")
    d_ffn: int = Field(ge=1, description="FFN hidden dimension")


class TrainingModelComputed(BaseModel):
    """Derived/computed fields from model config."""
    d_head: int = Field(description="d_model / num_heads")
    total_params_billions: str = Field(description="Estimated total params, e.g. '~7.0'")


class TrainingModelLayer(BaseModel):
    """A layer entry in the model layers list (union type)."""
    type: str
    desc: str | None = None
    id: int | None = None
    sub_layers: list[ModelSubLayer] | None = None
    skip_connection: bool | None = None


class TrainingModel(BaseModel):
    """Structured training model definition output."""
    type: str = Field(default="transformer_model")
    config: TrainingModelConfig
    computed: TrainingModelComputed
    layers: list[TrainingModelLayer]
    output_layer: OutputLayer = Field(default_factory=OutputLayer)

    def model_dump(self, **kwargs) -> dict[str, Any]:
        kwargs.setdefault("exclude_none", True)
        return super().model_dump(**kwargs)


class SessionState(BaseModel):
    """Persistent session state."""
    session_id: str
    original_params: TopologyParams | None = None
    equivalent_params: TopologyParams | None = None
    original_topology: MeshTopology | None = None
    equivalent_topology: MeshTopology | None = None
    original_simulation: SimulationResult | None = None
    equivalent_simulation: SimulationResult | None = None
    comparison_report: ComparisonReport | None = None
    original_training_model: TrainingModel | None = None
    equivalent_training_model: TrainingModel | None = None
    original_task_id: str | None = None
    equivalent_task_id: str | None = None
    step: str = "idle"  # idle | params_collected | topology_generated | simulating | completed
    history: list[dict[str, Any]] = Field(default_factory=list)
