"""服务配置。"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# 默认路径：aicm_mcp_server 根目录下的 aicm 与 workspace（本地开发用）。
# 容器内由镜像 ENV / chart 覆盖：sim_tool_home 钉为 /home/aicm；数据挂载树
# （workspace/db）在 /home/data 下，与工具目录分置 —— 同名会被整目录挂载遮掉。
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

    # 调度：全局并发上限 + 排队/执行超时（0 表示关闭该项超时）。
    # 并发上限是「服务级」的，不按调用方区分 —— 仿真子进程吃满 CPU/内存，
    # 无限并发会让多个任务互相拖慢（见 mcp_server/services/task_scheduler.py）。
    # 默认 2 = 1 对仿真（原始 + 等效两个子进程），先保证单对任务跑稳；
    # 压测确认有余量后再调大（1 对 = 2 个进程）。
    max_concurrent_tasks: int = 2
    # 排队上限（背压）：队列长度达到该值且没有空闲执行槽位时，execute_task 直接
    # 返回 JSON-RPC -32004（message 含「队列已满」），不静默入队。0/负数表示不限。
    # 默认 20 ≈ 10 对仿真任务，足够吸收一次批量提交的毛刺，又不会让调用方无限堆积。
    max_queue_size: int = 20
    task_timeout_sec: int = 3600
    queue_timeout_sec: int = 900

    # 工作区磁盘清理（见 mcp_server/services/workspace_cleaner.py）。
    # 优先级（从高到低）：运行中/排队中任务永不删 → 磁盘应急水位 → min_keep_tasks
    # → workspace_retention_days（先删 results/，日志/脚本留到 log_retention_days）。
    # 默认：结果文件保留 7 天、日志等保留 14 天、最近 20 个任务必留、每天扫一次。
    workspace_cleanup_enabled: bool = True
    workspace_retention_days: float = 7
    log_retention_days: float = 14
    min_keep_tasks: int = 20
    # 使用率 >= disk_watermark_pct 进入应急模式（保留期/必留数降到 emergency_*）；
    # >= disk_critical_pct 进入临界模式（只保留最新 1 个任务，尽快止血）
    disk_watermark_pct: float = 80
    disk_critical_pct: float = 90
    emergency_retention_days: float = 2
    emergency_min_keep_tasks: int = 5
    cleanup_interval_hours: float = 24
    cleanup_initial_delay_sec: float = 10
    # True 时只统计不删除（上线前观察一周用）
    cleanup_dry_run: bool = False

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
