"""规格文档 §3~§10 对应的入参/出参模型。"""

from enum import Enum
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator


class TaskStatus(str, Enum):
    SUBMITTED = "submitted"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ERROR = "error"


# ---------------------------------------------------------------------------
# §3 execute_task
# ---------------------------------------------------------------------------


class SimulationTaskInput(BaseModel):
    """组网 + 模型 + 运行时参数（§3.1）。"""

    name: str = Field(..., description="组网名称，如 原始组网 / 等效组网")
    device_type: Literal["A2", "A3", "A5"] = Field(..., description="设备类型")
    dp_size: int = Field(..., ge=1, description="数据并行度 DP")
    tp_size: int = Field(..., ge=1, description="张量并行度 TP")
    pp_size: int = Field(..., ge=1, description="流水线并行度 PP")
    total_nodes: int = Field(..., ge=1, description="总卡数")
    num_layers: int = Field(..., ge=1, description="Transformer 层数 L")
    hidden_dim: int = Field(..., ge=1, description="隐藏层维度 H")
    num_heads: int = Field(..., ge=1, description="注意力头数 A")
    d_ffn: int = Field(14336, ge=1, description="FFN 隐藏层维度")
    seq_len: int = Field(2048, ge=1, description="序列长度 S")
    batch_size: int = Field(
        32,
        ge=1,
        description="等效前（参考组网）的 global-batch-size，随 DP 缩放写入脚本",
    )
    micro_batch_size: Optional[int] = Field(
        None,
        ge=1,
        description="micro-batch-size，默认 4",
    )
    reference_dp_size: Optional[int] = Field(
        None,
        ge=1,
        description="参考组网 DP；未传时由 batch_size/micro_batch_size 推断",
    )
    model_name: Optional[str] = Field(None, description="模型名称")
    # --- MoE 模型配置（§3.1，仅 model_type="sparse" 时生效，dense 可省略）---
    ep: Optional[int] = Field(None, ge=1, description="专家并行度 EP（仅 sparse 时必填）")
    model_type: Literal["dense", "sparse"] = Field(
        "dense", description='模型类型："dense"（默认）或 "sparse"（MoE）'
    )
    num_experts: Optional[int] = Field(None, ge=1, description="MoE：每层专家数 E")
    moe_router_topk: Optional[int] = Field(None, ge=1, description="MoE：每个 token 激活的专家数 Top-K")
    num_moe_layers: Optional[int] = Field(None, ge=1, description="MoE：MoE 层数 L_moe")
    moe_ffn_hidden_size: Optional[int] = Field(None, ge=1, description="MoE：单个专家 FFN 中间维度 F_expert")
    moe_layer_freq: Optional[Union[int, str]] = Field(
        None,
        description=(
            'MoE：层分布模式（整数周期 / "-1" 哨兵 / 列表表达式如 "[0]*3+[1]*58"）；'
            "缺省时由 num_moe_layers 推导（前 L-L_moe 层 Dense、其余 MoE）"
        ),
    )
    has_shared_expert: bool = Field(False, description="MoE：是否含共享专家")
    shared_expert_intermediate_size: Optional[int] = Field(
        None, ge=1, description="MoE：共享专家 FFN 中间维度（has_shared_expert=true 时建议提供）"
    )
    expert_tensor_parallel_size: Optional[int] = Field(
        None, ge=1, description="MoE：专家内部张量并行度 TP_e，默认 1"
    )
    # 额外拓扑字段，调用方可能携带，服务端可忽略
    nodes: Optional[list[dict[str, Any]]] = None
    communication_groups: Optional[Union[list[Any], dict[str, Any]]] = None

    model_config = {"extra": "allow"}

    @model_validator(mode="before")
    @classmethod
    def normalize_mesh_topology(cls, data: Any) -> Any:
        """兼容 equivalent-modeling-service MeshTopology.model_dump() 入参。"""
        if not isinstance(data, dict):
            return data
        normalized = dict(data)
        if normalized.get("seq_len") is None:
            seq = normalized.get("seq_length")
            if seq is not None:
                normalized["seq_len"] = seq
        if normalized.get("batch_size") is None:
            batch = normalized.get("global_batch_size")
            if batch is not None:
                normalized["batch_size"] = batch
        if normalized.get("micro_batch_size") is None:
            mbs = normalized.get("micro_batch") or normalized.get("micro-batch-size")
            if mbs is not None:
                normalized["micro_batch_size"] = mbs
        if normalized.get("reference_dp_size") is None:
            ref_dp = normalized.get("reference_dp") or normalized.get("source_dp_size")
            if ref_dp is not None:
                normalized["reference_dp_size"] = ref_dp
        cg = normalized.get("communication_groups")
        if isinstance(cg, dict) and not cg:
            normalized["communication_groups"] = None
        return normalized

    @model_validator(mode="after")
    def validate_moe_fields(self) -> "SimulationTaskInput":
        """§3.1：model_type="sparse" 时 MoE 条件必填字段 + 组合合法性校验。

        组合校验只挡「语义上不可能成立」的参数，避免脏组合被带到仿真侧才失败：
        MoE 层数不能超过总层数、Top-K 不能超过专家数、专家数必须能被 EP 整除
        （Megatron/MindSpeed 的硬约束）、EP 不能超过可用于专家的卡数 dp*tp。

        显式给出的 ``moe_layer_freq`` 表达式在此不做展开校验（解释权在仿真侧），
        由调用方保证其长度与 num_layers 一致。
        """
        if self.model_type != "sparse":
            return self
        missing = [
            name
            for name in (
                "ep",
                "num_experts",
                "moe_router_topk",
                "num_moe_layers",
                "moe_ffn_hidden_size",
            )
            if getattr(self, name) is None
        ]
        if missing:
            raise ValueError(
                'model_type="sparse" requires fields: ' + ", ".join(missing)
            )

        errors: list[str] = []
        if self.num_moe_layers > self.num_layers:
            errors.append(
                f"num_moe_layers ({self.num_moe_layers}) > "
                f"num_layers ({self.num_layers})"
            )
        if self.moe_router_topk > self.num_experts:
            errors.append(
                f"moe_router_topk ({self.moe_router_topk}) > "
                f"num_experts ({self.num_experts})"
            )
        if self.num_experts % self.ep != 0:
            errors.append(
                f"num_experts ({self.num_experts}) not divisible by ep ({self.ep})"
            )
        if self.ep > self.dp_size * self.tp_size:
            errors.append(
                f"ep ({self.ep}) > dp*tp ({self.dp_size * self.tp_size})"
            )
        if errors:
            raise ValueError("invalid MoE config: " + "; ".join(errors))
        return self


class SimulationRunnerParams(BaseModel):
    """仿真运行参数（§3.1 simulation_params）。"""

    script_path: Optional[str] = None
    epoch_num: Optional[int] = 1
    model_name: Optional[str] = ""
    device_type: Optional[str] = None
    vocab_size: Optional[str] = None
    frame: Optional[str] = None
    rank: Optional[int] = 0
    rank_range: Optional[int] = None
    comp_filepath: Optional[str] = None
    no_time_accumulation: Optional[bool] = False
    level0_config: Optional[Any] = None
    level1_config: Optional[Any] = None
    visual_json_output: Optional[bool] = True
    comm_group_output: Optional[bool] = True
    debug_time: Optional[bool] = False

    model_config = {"extra": "allow"}


class ExecuteTaskInput(BaseModel):
    topology: SimulationTaskInput
    simulation_params: SimulationRunnerParams = Field(default_factory=SimulationRunnerParams)


class ExecuteTaskOutput(BaseModel):
    task_id: str
    status: TaskStatus = TaskStatus.SUBMITTED


# ---------------------------------------------------------------------------
# §4 report_status
# ---------------------------------------------------------------------------


class ReportStatusInput(BaseModel):
    task_id: str


class ReportStatusOutput(BaseModel):
    status: TaskStatus
    progress: Optional[float] = Field(None, ge=0, le=100)
    message: Optional[str] = None


# ---------------------------------------------------------------------------
# §5 sync_logs
# ---------------------------------------------------------------------------


class SyncLogsInput(BaseModel):
    task_id: str
    offset: int = Field(0, ge=0)


class SyncLogsOutput(BaseModel):
    lines: list[str]
    next_offset: Optional[int] = None


# ---------------------------------------------------------------------------
# §6 get_result
# ---------------------------------------------------------------------------


class GetResultInput(BaseModel):
    task_id: str


class GetResultOutput(BaseModel):
    task_id: str
    status: TaskStatus = TaskStatus.COMPLETED
    summary: dict[str, Any] = Field(default_factory=dict)
    cards: list["CardMetrics"] = Field(default_factory=list)

    model_config = {"extra": "allow"}


# ---------------------------------------------------------------------------
# §7 card_detail
# ---------------------------------------------------------------------------


class FlopsDetail(BaseModel):
    """与 statistic_data Flops Detail 段对齐（同 get_hbm_detail / get_comm_detail 的二级明细）。"""

    total_flops: float = 0.0
    forward_flops: float = 0.0
    backward_b_flops: float = 0.0
    backward_w_flops: float = 0.0


class CardHbmDetail(BaseModel):
    """HBM 分项；effective_hbm_gb 与一级 hbm_gb 一致（total - activation - comm_buf）。"""

    weights_gb: float = 0.0
    gradients_gb: float = 0.0
    optimizer_gb: float = 0.0
    activations_gb: float = 0.0
    comm_buf_gb: float = 0.0
    total_hbm_gb: float = 0.0
    effective_hbm_gb: float = 0.0


class CommGroupDetail(BaseModel):
    """单通信组明细，字段与 CommDetailOutput 一致（不含 global_rank / comm_type）。"""

    comm_count: int = 0
    comm_cards: int = 0
    comm_size_per_time_gb: float = 0.0
    total_comm_gb: float = 0.0


class CardCommDetail(BaseModel):
    """TP / PP / DP / EP 通信二级明细（EP 仅 MoE 任务有量，dense 为 0）。"""

    tp: CommGroupDetail = Field(default_factory=CommGroupDetail)
    pp: CommGroupDetail = Field(default_factory=CommGroupDetail)
    dp: CommGroupDetail = Field(default_factory=CommGroupDetail)
    ep: CommGroupDetail = Field(default_factory=CommGroupDetail)


class CardMetrics(BaseModel):
    card_id: str
    global_rank: int
    flops_per_card: float = 0.0
    hbm_gb: float = 0.0
    tp_comm_gb_per_micro: float = 0.0
    pp_comm_mb_per_micro: float = 0.0
    dp_comm_gb_per_step: float = 0.0
    ep_comm_gb_per_step: float = 0.0
    flops_detail: FlopsDetail = Field(default_factory=FlopsDetail)
    hbm_detail: CardHbmDetail = Field(default_factory=CardHbmDetail)
    comm_detail: CardCommDetail = Field(default_factory=CardCommDetail)


class CardDetailInput(BaseModel):
    task_id: str
    card_ids: Optional[list[str]] = None


class CardDetailOutput(BaseModel):
    cards: list[CardMetrics]


# ---------------------------------------------------------------------------
# §8 get_device_detail
# ---------------------------------------------------------------------------


class OperatorTrace(BaseModel):
    index: int
    operator_name: Optional[str] = None
    comm_type: str
    comm_group: Optional[str] = None
    comm_group_size: Optional[int] = None
    msg_size: Optional[float] = None
    stage: str
    dst: Optional[str] = None
    src: Optional[str] = None
    additional: Optional[str] = None
    nonblock: int = 0
    wait_n: Optional[int] = None
    elapsed_time: Optional[float] = Field(None, serialization_alias="_elapsed_time")
    start_time: float
    end_time: float
    single_flops: Optional[float] = None
    data_shape: Optional[str] = None
    data_type: Optional[str] = None
    algo_name: Optional[str] = None
    duration: Optional[float] = None

    model_config = {"populate_by_name": True, "extra": "allow"}


class TimelineSummary(BaseModel):
    total_time_ms: float = 0.0
    compute_time_ms: float = 0.0
    comm_time_ms: float = 0.0
    compute_pct: float = 0.0
    comm_pct: float = 0.0
    total_flops: float = 0.0
    total_comm_gb: float = 0.0


class DeviceDetailInput(BaseModel):
    task_id: str
    global_rank: int = Field(..., ge=0)
    offset: int = Field(0, ge=0)


class DeviceDetailOutput(BaseModel):
    card_id: str
    global_rank: int
    task_id: str
    topology_name: str
    device_type: str
    dp_rank: Optional[int] = None
    tp_rank: Optional[int] = None
    pp_rank: Optional[int] = None
    operators: list[OperatorTrace] = Field(default_factory=list)
    next_offset: Optional[int] = None
    is_complete: bool = True
    timeline: Optional[TimelineSummary] = None


# ---------------------------------------------------------------------------
# §9 get_hbm_detail
# ---------------------------------------------------------------------------


class HbmDetailInput(BaseModel):
    task_id: str
    global_rank: int = Field(..., ge=0)


class HbmDetailOutput(BaseModel):
    global_rank: int
    weights_gb: float
    gradients_gb: float
    optimizer_gb: float
    activations_gb: float
    total_hbm_gb: float


# ---------------------------------------------------------------------------
# §10 get_comm_detail
# ---------------------------------------------------------------------------


class CommDetailInput(BaseModel):
    task_id: str
    global_rank: int = Field(..., ge=0)
    comm_type: Literal["tp", "pp", "dp", "ep"]


class CommDetailOutput(BaseModel):
    global_rank: int
    comm_type: Literal["tp", "pp", "dp", "ep"]
    comm_count: int
    comm_cards: int
    comm_size_per_time_gb: float
    total_comm_gb: float


# ---------------------------------------------------------------------------
# §11 get_training_script
# ---------------------------------------------------------------------------


class GetTrainingScriptInput(BaseModel):
    task_id: str


class GetTrainingScriptOutput(BaseModel):
    task_id: str
    topology_name: str
    script_path: Optional[str] = None
    script_filename: Optional[str] = None
    script_content: str
