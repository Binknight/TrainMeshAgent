# syntax=docker/dockerfile:1
# ============================================================================
# TrainMeshAgent 业务镜像
#
# 依赖已全部预置在基础镜像 repository/python3.10-aicm-base:1.0 中，
# 本层只做「解压 + 启动」，构建期不联网、不装任何包。
#
# 包内容（TrainMeshAgent.tgz）解压后落在 /home/，即仓库根：
#   /home/app          Flask 应用 + Agent + skills
#   /home/static       前端静态资源（无 npm 构建，直接托管）
#   /home/mcp_server   MCP 仿真 Server
#   /home/aicm         仿真工具（run.py / examples/time_args.json / workload_generator/）
#   /home/docker       entrypoint.sh + wait_for_db.py
#
# 路径契约：mcp_server/config.py 以 parents[1] 推导默认值，而代码被解压到 /home，
# 故 sim_tool_home=/home/aicm、workspace_root=/home/workspace（与仓库根一致）。
# 下面用 ENV 显式钉住，不依赖隐式推导。
# ============================================================================
FROM repository/python3.10-aicm-base:1.0

WORKDIR /home/

COPY TrainMeshAgent*.tgz app.tgz

EXPOSE 5000 9000
ENV LANG C.UTF-8

# 解压应用包
RUN tar -xzf app.tgz && rm app.tgz

# 仿真任务工作区（每个 task_id 一个子目录，仿真产物写入其 results/）
RUN mkdir -p /home/workspace && chmod +x /home/docker/entrypoint.sh

# 设置权限
RUN chown 1000:1000 /home/ -R

# ---------------------------------------------------------------------------
# 运行期配置
# ---------------------------------------------------------------------------
# AICM_MCP_CONDA_ENV 必须显式置空：conda_launcher.build_simulation_command()
# 在 conda_env 为空串时回退为 [sys.executable, ...]，从而免装 conda。
# 端口两侧默认值已对齐为 9000（HEAD 5fa5f4b），此处显式写出作为单一来源。
# DATABASE_URL 指向外部 PostgreSQL，必须在运行时注入（无默认值可用）。
ENV PYTHONPATH=/home \
    FLASK_HOST=0.0.0.0 \
    FLASK_PORT=5000 \
    FLASK_DEBUG=false \
    FLASK_USE_RELOADER=false \
    AICM_MCP_HOST=0.0.0.0 \
    AICM_MCP_PORT=9000 \
    AICM_MCP_SIM_TOOL_HOME=/home/aicm \
    AICM_MCP_WORKSPACE_ROOT=/home/workspace \
    AICM_MCP_CONDA_ENV= \
    MCP_SERVER_URL=http://127.0.0.1:9000

# 健康检查：两个进程都要活着。用 python 而非 curl。
HEALTHCHECK --interval=30s --timeout=6s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request as u, sys; \
sys.exit(0 if all(u.urlopen(x, timeout=4).status < 400 for x in \
('http://127.0.0.1:5000/api/health', 'http://127.0.0.1:9000/health')) else 1)"

# 启动：Flask (SSE + WebSocket) 与 MCP 仿真 Server 同容器双进程
CMD umask 0027 && bash /home/docker/entrypoint.sh
