"""服务配置。"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# 默认路径：aicm_mcp_server 根目录下的 aicm 与 workspace
_SERVER_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AICM_MCP_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    host: str = "0.0.0.0"
    # 与 equivalent-modeling-service 侧 MCP_SERVER_URL 的默认值对齐（见 .env.example / app/config.py）
    port: int = 9000
    log_level: str = "info"
    task_poll_timeout_sec: int = 300

    # 仿真工具根目录（含 run.py、workload_generator/）
    sim_tool_home: Path = _SERVER_ROOT / "aicm"
    # 任务工作区：每个 task_id 一个子目录，其下 results/ 为仿真输出
    workspace_root: Path = _SERVER_ROOT / "workspace"
    # 为 True 时仅创建任务目录与脚本，不拉起子进程（测试用）
    dry_run: bool = False
    # 执行 run.py 时使用的 conda 环境名；默认置空 = 用当前 Python 解释器
    # （conda_launcher.build_simulation_command 对空值的处理），容器与本地都不再强依赖 conda
    conda_env: str = ""


settings = Settings()
