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
    seq_len: int | None = Field(default=None, description="序列长度 S")
    batch_size: int | None = Field(default=None, description="批次大小 B")
    model_name: str | None = Field(default=None, description="模型名称")


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
    hbm_gb: float = Field(description="HBM总占用 (GB)")
    hbm_model_gb: float | None = Field(default=None, description="模型显存占用 (GB) = 权重+梯度+优化器，不含激活值")
    tp_comm_gb_per_micro: float = Field(description="TP通信量 (GB/micro)")
    pp_comm_mb_per_micro: float = Field(description="PP通信量 (MB/micro)")
    dp_comm_gb_per_step: float = Field(description="DP通信量 (GB/step)")


class SimulationResult(BaseModel):
    """Complete simulation result for a topology."""
    topology_name: str
    device_type: DeviceType
    total_nodes: int
    cards: list[CardMetrics]


class OperatorTrace(BaseModel):
    """Single operator execution trace record — aligned with simulation CSV output + MCP computed fields."""
    # ── CSV passthrough ──
    comm_type: str = Field(description="算子/通信类型, e.g. 'computation', 'all_reduce', 'send', 'recv'")
    comm_group: str | None = Field(default=None, description="通信组名, e.g. 'tp_group', 'dp_group'")
    comm_group_size: int | None = Field(default=None, description="通信组参与卡数")
    msg_size: float | None = Field(default=None, description="通信消息大小 (bytes)")
    stage: str = Field(description="执行阶段, e.g. 'forward/layer0', 'backward/layer14', 'optimizer', 'init'")
    dst: str | None = Field(default=None, description="目标 rank 或组")
    src: str | None = Field(default=None, description="源 rank 或组")
    additional: str | None = Field(default=None, description="附加说明, e.g. 'matmul', 'seed-sync'")
    nonblock: int = Field(default=0, description="是否非阻塞: 1=非阻塞, 0=阻塞")
    wait_n: int | None = Field(default=None, description="等待数量")
    elapsed_time: float = Field(default=0, description="仿真系统原始耗时 (微秒), CSV列 _elapsed_time")
    start_time: float = Field(description="算子开始时间 (微秒)")
    end_time: float = Field(description="算子结束时间 (微秒)")
    single_flops: float | None = Field(default=None, description="单算子 FLOPs; 通信类为 null")

    # ── MCP computed ──
    index: int = Field(default=0, description="算子序号, 从0递增, 用于增量offset对齐")
    operator_name: str = Field(default="", description="算子可读名称, 由MCP根据comm_type+stage推导")
    data_shape: str | None = Field(default=None, description="数据形状描述, e.g. '[32,4096,4096] → [32,4096,12288]'")
    data_type: str | None = Field(default=None, description="数据类型, e.g. 'bf16', 'fp32'")
    algo_name: str | None = Field(default=None, description="算法名, e.g. 'linear', 'Ring'")
    duration: float = Field(default=0, description="= end_time - start_time (微秒), 方便前端使用")


class TimelineSummary(BaseModel):
    """Operator timeline summary statistics."""
    total_time_ms: float
    compute_time_ms: float
    comm_time_ms: float
    compute_pct: float
    comm_pct: float
    total_flops: float
    total_comm_gb: float


class DeviceSimulationDetail(BaseModel):
    """Per-device simulation detail with operator timeline and tracing."""
    card_id: str
    global_rank: int
    task_id: str
    topology_name: str
    device_type: str
    dp_rank: int = 0
    tp_rank: int = 0
    pp_rank: int = 0
    operators: list[OperatorTrace] = Field(default_factory=list)
    timeline: TimelineSummary | None = None


class HbmDetail(BaseModel):
    """HBM usage breakdown by component."""
    global_rank: int
    weights_gb: float = Field(description="权重占用 (GB)")
    gradients_gb: float = Field(description="梯度占用 (GB)")
    optimizer_gb: float = Field(description="优化器状态占用 (GB)")
    activations_gb: float = Field(description="激活值占用 (GB)")
    model_hbm_gb: float = Field(description="模型显存占用 (GB) = 权重+梯度+优化器")
    total_hbm_gb: float = Field(description="HBM总占用 (GB)")


class CommDetail(BaseModel):
    """Communication detail for TP/PP/DP."""
    global_rank: int
    comm_type: str = Field(description="通信类型: tp | pp | dp")
    comm_count: int = Field(description="每 step 通信次数")
    comm_cards: int = Field(description="参与通信的卡数")
    comm_size_per_time_gb: float = Field(description="单次通信量 (GB)")
    total_comm_gb: float = Field(description="总通信量 (GB)")


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
    event_type: str  # thinking | tool_call | guard_check | mesh_json | model_json | equiv_formula_line | sim_data | workflow_state | message | error | done
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
    vocab_size: int = Field(default=32000, description="Vocabulary size")


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
    model_name: str | None = Field(default=None, description="User-facing model name, e.g. Llama-3.1-8B")
    config: TrainingModelConfig
    computed: TrainingModelComputed
    layers: list[TrainingModelLayer]
    output_layer: OutputLayer = Field(default_factory=OutputLayer)

    def model_dump(self, **kwargs) -> dict[str, Any]:
        kwargs.setdefault("exclude_none", True)
        return super().model_dump(**kwargs)


class SimulationParams(BaseModel):
    """Simulation parameters passed to the MCP simulation system."""
    epoch_num: int = 1
    model_name: str = ""
    device_type: str = "ASCEND_910B"
    vocab_size: str = "18277"
    frame: str = "Mindspeed"
    rank: int = 0
    rank_range: int = 1023
    comp_filepath: str = "/opt/traffic_modeling/aicm/default.txt"
    no_time_accumulation: bool = False
    level0_config: dict[str, Any] | None = None
    level1_config: dict[str, Any] | None = None
    visual_json_output: bool = True
    comm_group_output: bool = True
    debug_time: bool = False


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
    simulation_params: SimulationParams | None = None
    step: str = "idle"  # idle | params_collected | equiv_calculating | equiv_generated | simulating | completed
    history: list[dict[str, Any]] = Field(default_factory=list)
    # Step1 form metadata captured from frontend
    original_model_name: str | None = None
    original_seq_len: int | None = None
    original_batch_size: int | None = None
    original_dff: int | None = None
    original_vocab_size: int | None = None
    equivalent_seq_len: int | None = None
    equivalent_batch_size: int | None = None
    equivalent_dff: int | None = None
    original_micro_batch: int | None = None
    equivalent_micro_batch: int | None = None
