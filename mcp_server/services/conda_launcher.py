"""通过 conda 环境启动仿真子进程。"""

import sys
from typing import List


def build_simulation_command(run_argv: List[str], *, conda_env: str) -> List[str]:
    """
    构造 Popen 使用的 argv。

    run_argv 来自 param_mapper.build_run_py_command（首项为 run.py 路径）。
    conda_env 为空时回退为当前解释器。
    """
    if not conda_env.strip():
        return [sys.executable, *run_argv]
    return [
        "conda",
        "run",
        "-n",
        conda_env.strip(),
        "--no-capture-output",
        "python",
        *run_argv,
    ]
