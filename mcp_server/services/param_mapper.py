"""SimulationRunnerParams + topology → run.py CLI 参数。"""

import json
from pathlib import Path
from typing import Any, Optional

from mcp_server.schemas.common import SimulationRunnerParams, SimulationTaskInput
from mcp_server.services.device_mapping import resolve_ascend_device_type

_VALID_FRAMES = frozenset(
    {"Megatron", "DeepSpeed", "collective_test", "MindSpeed", "MindSpore"}
)
_FRAME_ALIASES = {
    "megatron": "Megatron",
    "deepspeed": "DeepSpeed",
    "collective_test": "collective_test",
    "mindspeed": "MindSpeed",
    "mindspore": "MindSpore",
}


def normalize_frame(frame: Optional[str]) -> str:
    """将调用方 frame 规范为 generate_megatron_workload 接受的枚举值。"""
    if not frame or not str(frame).strip():
        return "MindSpeed"
    raw = str(frame).strip()
    if raw in _VALID_FRAMES:
        return raw
    canonical = _FRAME_ALIASES.get(raw.lower())
    if canonical:
        return canonical
    return "MindSpeed"


def _resolve_path(base: Path, path_str: Optional[str]) -> Optional[Path]:
    if not path_str:
        return None
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (base / p).resolve()


def _write_json_config(data: Any, task_dir: Path, filename: str) -> Path:
    path = task_dir / filename
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def resolve_level_config(
    value: Any,
    task_dir: Path,
    filename: str,
    sim_home: Path,
) -> Optional[Path]:
    """level0_config / level1_config：路径字符串或 JSON 对象。"""
    if value is None:
        return None
    if isinstance(value, str) and value.strip():
        return _resolve_path(sim_home, value) or _resolve_path(task_dir, value)
    if isinstance(value, dict):
        return _write_json_config(value, task_dir, filename)
    return None


def build_run_py_command(
    *,
    sim_home: Path,
    task_dir: Path,
    script_path: Path,
    topology: SimulationTaskInput,
    params: SimulationRunnerParams,
) -> list[str]:
    """构造调用 aicm/run.py 的 argv（不含 python 可执行文件前缀）。"""
    run_py = sim_home / "run.py"
    if not run_py.is_file():
        raise FileNotFoundError(f"simulation run.py not found: {run_py}")

    time_args = sim_home / "examples" / "time_args.json"
    if not time_args.is_file():
        raise FileNotFoundError(f"time_args.json not found: {time_args}")

    model_name = (params.model_name or topology.model_name or "mcp_model").strip() or "mcp_model"
    device_type = resolve_ascend_device_type(
        topology.device_type,
        params.device_type,
    )

    vocab_raw = params.vocab_size
    try:
        vocab_size = int(vocab_raw) if vocab_raw not in (None, "") else 10277
    except (TypeError, ValueError):
        vocab_size = 10277

    level0 = resolve_level_config(
        params.level0_config, task_dir, "level0.json", sim_home
    )
    level1 = resolve_level_config(
        params.level1_config, task_dir, "level1.json", sim_home
    )

    cmd: list[str] = [
        str(run_py),
        "--script_path",
        str(script_path.resolve()),
        "--epoch_num",
        str(params.epoch_num or 1),
        "--model_name",
        model_name,
        "--device_type",
        device_type,
        "--vocab_size",
        str(vocab_size),
        "--frame",
        normalize_frame(params.frame),
        "--rank",
        str(params.rank if params.rank is not None else 0),
        "--rank_range",
        str(params.rank_range if params.rank_range is not None else -1),
        "--time_args",
        str(time_args.resolve()),
    ]

    if level0 is not None:
        cmd.extend(["--level0", str(level0)])
    if level1 is not None:
        cmd.extend(["--level1", str(level1)])

    if params.visual_json_output:
        cmd.append("--visual_json_output")
    if params.comm_group_output:
        cmd.append("--comm_group_output")
    if params.debug_time:
        cmd.append("--debug_time")
    if params.no_time_accumulation:
        cmd.append("--no_time_accumulation")

    return cmd
