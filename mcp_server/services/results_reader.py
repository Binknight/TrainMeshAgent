"""解析 workspace/{task_id}/results 下的仿真产物。"""

import csv
import re
from pathlib import Path
from typing import Any, Optional

from mcp_server.schemas.common import (
    CardCommDetail,
    CardHbmDetail,
    CardMetrics,
    CommDetailOutput,
    CommGroupDetail,
    DeviceDetailOutput,
    FlopsDetail,
    HbmDetailOutput,
    OperatorTrace,
    SimulationTaskInput,
    TaskStatus,
    TimelineSummary,
)
from mcp_server.services.task_store import TaskRecord

_KV_RE = re.compile(r"^\s*(.+?):\s*([\d.eE+-]+)\s*$")
_COMM_GROUP_RE = re.compile(r"^\s*(\w+):\s*count=(\d+),\s*bytes=([\d.eE+-]+)")

# 规范化后的 key -> build_card_metrics / get_hbm_detail 使用的字段名
_FLOPS_KEY_ALIASES: dict[str, str] = {
    "total_flops": "total_flops",
    "forward_flops": "forward_flops",
    "backward_b_flops": "backward_B_flops",
    "backward_w_flops": "backward_W_flops",
}
_HBM_KEY_ALIASES: dict[str, str] = {
    "grad": "gradient",
}


def _parse_scientific(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return 0.0


def _bytes_to_gb(n: float) -> float:
    return n / (1024**3)


def _effective_hbm_bytes(hbm: dict[str, float]) -> float:
    """card_detail 用 HBM：total 减去 activation 与 comm_buf。"""
    activation = hbm.get("activation", 0.0)
    comm_buf = hbm.get("comm_buf", 0.0)
    total = hbm.get("total", 0.0)
    if total:
        return max(0.0, total - activation - comm_buf)
    return sum(hbm.get(k, 0.0) for k in ("weight", "gradient", "optimizer"))


def _find_rank_csv(results_dir: Path, rank: int, suffix: str) -> Optional[Path]:
    mocked = results_dir / "mocked_workload"
    if not mocked.is_dir():
        return None
    matches = sorted(mocked.glob(f"*_rank{rank}_{suffix}.csv"))
    return matches[0] if matches else None


def list_available_ranks(results_dir: Path) -> list[int]:
    mocked = results_dir / "mocked_workload"
    if not mocked.is_dir():
        return []
    ranks: set[int] = set()
    for p in mocked.glob("*_rank*_time.csv"):
        m = re.search(r"_rank(\d+)_time\.csv$", p.name)
        if m:
            ranks.add(int(m.group(1)))
    if not ranks:
        for p in mocked.glob("*_rank*_workload.csv"):
            m = re.search(r"_rank(\d+)_workload\.csv$", p.name)
            if m:
                ranks.add(int(m.group(1)))
    return sorted(ranks)


def _normalize_field_key(raw: str) -> str:
    """将 'Total Flops' / 'total_flops' 等统一为小写下划线形式。"""
    return raw.strip().lower().replace(" ", "_")


def _parse_statistic_file(path: Path) -> dict[str, Any]:
    """解析 results/statistic_data/rank{N}.txt。

    兼容两种仿真输出格式：
    - 旧版（time_accumulate）：``total_flops:``、``Comm Detail``、``breakdown_by_group (count, bytes):``
    - 新版（workspace）：``Total Flops:``、``Communication Detail``、``breakdown_by_group(counts, bytes):``
    """
    out: dict[str, Any] = {
        "hbm": {},
        "flops": {},
        "comm_by_group": {},
    }
    if not path.is_file():
        return out

    section = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "HBM Detail" in line:
            section = "hbm"
            continue
        if "Flops Detail" in line:
            section = "flops"
            continue
        if "Comm Detail" in line or "Communication Detail" in line:
            section = "comm"
            continue
        if "breakdown_by_group" in line and "breakdown_by_group_and" not in line:
            if "_x_type" not in line and "_and_type" not in line:
                section = "comm_groups"
                continue

        gm = _COMM_GROUP_RE.match(line)
        if gm and section == "comm_groups":
            name, count, nbytes = gm.group(1), int(gm.group(2)), _parse_scientific(gm.group(3))
            out["comm_by_group"][name] = {"count": count, "bytes": nbytes}
            continue

        m = _KV_RE.match(line)
        if not m:
            continue

        norm_key = _normalize_field_key(m.group(1))
        val = _parse_scientific(m.group(2))

        if section == "hbm":
            out["hbm"][_HBM_KEY_ALIASES.get(norm_key, norm_key)] = val
        elif section == "flops":
            if norm_key in ("total_comm_count", "total_comm_bytes"):
                continue
            canonical = _FLOPS_KEY_ALIASES.get(norm_key, norm_key)
            out["flops"][canonical] = val

    return out


def _derive_parallel_ranks(global_rank: int, topology: SimulationTaskInput) -> tuple[int, int, int]:
    """把 global_rank 分解为 (dp_rank, tp_rank, pp_rank)。

    布局约定为 **TP-DP-PP**（规格 §8「Rank 布局约定」）：TP 为最低位（变化最快），
    其次 DP，PP 为最高位（最外层）::

        global_rank = pp_rank * (tp * dp) + dp_rank * tp + tp_rank

    旧布局 TP-PP-DP（``pp_rank = (rank // tp) % pp``）已废弃：它会让同一个
    global_rank 在两侧归属不同的流水段，PP 通信量按段分组比较时错配。

    注意：不可改为 import app.rank_layout —— 本服务部署在仿真系统侧，
    是独立交付物，不能反向依赖 TrainMeshAgent 的 app 包。此处为实现唯一副本。
    """
    tp, dp = max(1, topology.tp_size), max(1, topology.dp_size)
    tp_rank = global_rank % tp
    dp_rank = (global_rank // tp) % dp
    pp_rank = global_rank // (tp * dp)
    return dp_rank, tp_rank, pp_rank


def _operator_name(row: dict[str, str]) -> str:
    comm_type = row.get("comm_type") or ""
    stage = row.get("stage") or ""
    additional = row.get("additional") or ""
    if additional and additional not in ("None", "-1"):
        return f"{additional}_{stage}" if stage else additional
    if stage and comm_type:
        return f"{comm_type}_{stage}"
    return comm_type or "unknown"


def _row_to_operator(index: int, row: dict[str, str]) -> OperatorTrace:
    def _f(key: str) -> Optional[float]:
        v = row.get(key, "")
        if v in ("", "None", None):
            return None
        try:
            return float(v)
        except ValueError:
            return None

    def _i(key: str) -> Optional[int]:
        v = row.get(key, "")
        if v in ("", "None", None):
            return None
        try:
            return int(float(v))
        except ValueError:
            return None

    start = _f("start_time") or 0.0
    end = _f("end_time") or 0.0
    duration = end - start if end >= start else _f("_elapsed_time")

    return OperatorTrace(
        index=index,
        operator_name=_operator_name(row),
        comm_type=row.get("comm_type") or "",
        comm_group=row.get("comm_group") if row.get("comm_group") not in ("None", "") else None,
        comm_group_size=_i("comm_group_size"),
        msg_size=_f("msg_size"),
        stage=row.get("stage") or "",
        dst=row.get("dst") if row.get("dst") not in ("None", "-1", "") else None,
        src=row.get("src") if row.get("src") not in ("None", "-1", "") else None,
        additional=row.get("additional") if row.get("additional") not in ("None", "") else None,
        nonblock=int(_f("nonblock") or 0),
        wait_n=_i("wait_n"),
        elapsed_time=_f("_elapsed_time"),
        start_time=start,
        end_time=end,
        single_flops=_f("single_flops"),
        duration=duration,
        algo_name=row.get("additional") if row.get("comm_type") != "computation" else None,
    )


def load_operators_from_csv(
    csv_path: Path,
    offset: int = 0,
    limit: Optional[int] = None,
) -> list[OperatorTrace]:
    operators: list[OperatorTrace] = []
    with csv_path.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            if idx < offset:
                continue
            operators.append(_row_to_operator(idx, row))
            if limit is not None and len(operators) >= limit:
                break
    return operators


def count_operators_in_csv(csv_path: Path) -> int:
    if not csv_path.is_file():
        return 0
    with csv_path.open(newline="", encoding="utf-8", errors="replace") as f:
        return sum(1 for _ in csv.DictReader(f))


def build_timeline(operators: list[OperatorTrace]) -> TimelineSummary:
    total_time_us = 0.0
    compute_us = 0.0
    comm_us = 0.0
    total_flops = 0.0
    total_comm_bytes = 0.0

    for op in operators:
        dur = op.duration or 0.0
        total_time_us += dur
        if op.comm_type == "computation":
            compute_us += dur
            if op.single_flops:
                total_flops += op.single_flops
        else:
            comm_us += dur
            if op.msg_size:
                total_comm_bytes += op.msg_size

    total_ms = total_time_us / 1000.0
    compute_ms = compute_us / 1000.0
    comm_ms = comm_us / 1000.0
    denom = total_ms or 1.0
    return TimelineSummary(
        total_time_ms=total_ms,
        compute_time_ms=compute_ms,
        comm_time_ms=comm_ms,
        compute_pct=compute_ms / denom * 100.0,
        comm_pct=comm_ms / denom * 100.0,
        total_flops=total_flops,
        total_comm_gb=_bytes_to_gb(total_comm_bytes),
    )


def _build_flops_detail(flops: dict[str, float]) -> FlopsDetail:
    return FlopsDetail(
        total_flops=flops.get("total_flops", 0.0),
        forward_flops=flops.get("forward_flops", 0.0),
        backward_b_flops=flops.get("backward_B_flops", 0.0),
        backward_w_flops=flops.get("backward_W_flops", 0.0),
    )


def _build_card_hbm_detail(hbm: dict[str, float]) -> CardHbmDetail:
    weights = hbm.get("weight", 0.0)
    gradients = hbm.get("gradient", 0.0)
    optimizer = hbm.get("optimizer", 0.0)
    activations = hbm.get("activation", 0.0)
    comm_buf = hbm.get("comm_buf", 0.0)
    total = hbm.get("total", 0.0) or (weights + gradients + optimizer + activations)
    effective = _effective_hbm_bytes(hbm)
    return CardHbmDetail(
        weights_gb=_bytes_to_gb(weights),
        gradients_gb=_bytes_to_gb(gradients),
        optimizer_gb=_bytes_to_gb(optimizer),
        activations_gb=_bytes_to_gb(activations),
        comm_buf_gb=_bytes_to_gb(comm_buf),
        total_hbm_gb=_bytes_to_gb(total),
        effective_hbm_gb=_bytes_to_gb(effective),
    )


def _build_comm_group_detail(
    info: dict[str, Any],
    comm_cards: int,
) -> CommGroupDetail:
    count = int(info.get("count", 0))
    total_bytes = float(info.get("bytes", 0.0))
    per_time = total_bytes / count if count else 0.0
    return CommGroupDetail(
        comm_count=count,
        comm_cards=comm_cards,
        comm_size_per_time_gb=_bytes_to_gb(per_time),
        total_comm_gb=_bytes_to_gb(total_bytes),
    )


def _build_card_comm_detail(
    comm: dict[str, dict[str, Any]],
    topology: SimulationTaskInput,
) -> CardCommDetail:
    return CardCommDetail(
        tp=_build_comm_group_detail(
            comm.get("tp_group", {}),
            topology.tp_size,
        ),
        pp=_build_comm_group_detail(
            comm.get("pp_group", {}),
            topology.pp_size,
        ),
        dp=_build_comm_group_detail(
            comm.get("dp_group", {}),
            topology.dp_size,
        ),
    )


def build_card_metrics(
    rank: int,
    topology: SimulationTaskInput,
    results_dir: Path,
) -> CardMetrics:
    stats_path = results_dir / "statistic_data" / f"rank{rank}.txt"
    stats = _parse_statistic_file(stats_path)

    hbm = stats.get("hbm", {})
    flops = stats.get("flops", {})
    comm = stats.get("comm_by_group", {})

    effective_hbm = _effective_hbm_bytes(hbm)
    flops_detail = _build_flops_detail(flops)
    hbm_detail = _build_card_hbm_detail(hbm)
    comm_detail = _build_card_comm_detail(comm, topology)

    tp_bytes = comm.get("tp_group", {}).get("bytes", 0.0)
    pp_bytes = comm.get("pp_group", {}).get("bytes", 0.0)
    dp_bytes = comm.get("dp_group", {}).get("bytes", 0.0)
    ep_bytes = comm.get("ep_group", {}).get("bytes", 0.0)

    return CardMetrics(
        card_id=f"card_{rank}",
        global_rank=rank,
        flops_per_card=flops_detail.total_flops,
        hbm_gb=hbm_detail.effective_hbm_gb,
        tp_comm_gb_per_micro=comm_detail.tp.total_comm_gb,
        pp_comm_mb_per_micro=comm_detail.pp.total_comm_gb * 1024.0,
        dp_comm_gb_per_step=comm_detail.dp.total_comm_gb,
        ep_comm_gb_per_step=_bytes_to_gb(ep_bytes),
        flops_detail=flops_detail,
        hbm_detail=hbm_detail,
        comm_detail=comm_detail,
    )


def load_all_cards(
    record: TaskRecord,
    card_ids: Optional[list[str]] = None,
) -> list[CardMetrics]:
    results_dir = record.results_dir or Path()
    ranks = list_available_ranks(results_dir)
    if card_ids:
        wanted = set()
        for cid in card_ids:
            if cid.startswith("card_"):
                wanted.add(int(cid.replace("card_", "")))
            else:
                try:
                    wanted.add(int(cid))
                except ValueError:
                    pass
        ranks = [r for r in ranks if r in wanted]

    return [
        build_card_metrics(r, record.topology, results_dir)
        for r in ranks
    ]


def get_device_detail(
    record: TaskRecord,
    global_rank: int,
    offset: int,
    *,
    is_task_running: bool,
) -> DeviceDetailOutput:
    results_dir = record.results_dir or Path()
    time_csv = _find_rank_csv(results_dir, global_rank, "time")
    if time_csv is None:
        time_csv = _find_rank_csv(results_dir, global_rank, "workload")

    total = count_operators_in_csv(time_csv) if time_csv else 0
    operators = load_operators_from_csv(time_csv, offset=offset) if time_csv else []

    is_complete = not is_task_running and (offset + len(operators) >= total)
    if is_task_running and time_csv and time_csv.is_file():
        # 运行中：文件可能仍在增长
        is_complete = False

    next_offset = offset + len(operators) if operators else offset
    timeline = None
    if is_complete and time_csv:
        all_ops = load_operators_from_csv(time_csv, offset=0)
        timeline = build_timeline(all_ops)

    dp_rank, tp_rank, pp_rank = _derive_parallel_ranks(global_rank, record.topology)

    return DeviceDetailOutput(
        card_id=f"card_{global_rank}",
        global_rank=global_rank,
        task_id=record.task_id,
        topology_name=record.topology.name,
        device_type=record.topology.device_type,
        dp_rank=dp_rank,
        tp_rank=tp_rank,
        pp_rank=pp_rank,
        operators=operators,
        next_offset=next_offset if (operators or is_task_running) else None,
        is_complete=is_complete,
        timeline=timeline,
    )


def get_hbm_detail(record: TaskRecord, global_rank: int) -> HbmDetailOutput:
    results_dir = record.results_dir or Path()
    stats_path = results_dir / "statistic_data" / f"rank{global_rank}.txt"
    stats = _parse_statistic_file(stats_path)
    hbm = stats.get("hbm", {})

    weights = hbm.get("weight", 0.0)
    gradients = hbm.get("gradient", 0.0)
    optimizer = hbm.get("optimizer", 0.0)
    activations = hbm.get("activation", 0.0)
    total = hbm.get("total", 0.0) or (weights + gradients + optimizer + activations)

    return HbmDetailOutput(
        global_rank=global_rank,
        weights_gb=_bytes_to_gb(weights),
        gradients_gb=_bytes_to_gb(gradients),
        optimizer_gb=_bytes_to_gb(optimizer),
        activations_gb=_bytes_to_gb(activations),
        total_hbm_gb=_bytes_to_gb(total),
    )


_COMM_GROUP_MAP = {
    "tp": "tp_group",
    "pp": "pp_group",
    "dp": "dp_group",
}


def get_comm_detail(
    record: TaskRecord,
    global_rank: int,
    comm_type: str,
) -> CommDetailOutput:
    results_dir = record.results_dir or Path()
    stats_path = results_dir / "statistic_data" / f"rank{global_rank}.txt"
    stats = _parse_statistic_file(stats_path)
    group_name = _COMM_GROUP_MAP[comm_type]
    info = stats.get("comm_by_group", {}).get(group_name, {})
    count = int(info.get("count", 0))
    total_bytes = float(info.get("bytes", 0.0))

    # 参与卡数：从 topology 推断
    topo = record.topology
    comm_cards = {"tp": topo.tp_size, "pp": topo.pp_size, "dp": topo.dp_size}[comm_type]
    per_time = total_bytes / count if count else 0.0

    return CommDetailOutput(
        global_rank=global_rank,
        comm_type=comm_type,  # type: ignore[arg-type]
        comm_count=count,
        comm_cards=comm_cards,
        comm_size_per_time_gb=_bytes_to_gb(per_time),
        total_comm_gb=_bytes_to_gb(total_bytes),
    )


def has_results(results_dir: Path) -> bool:
    return bool(list_available_ranks(results_dir))
