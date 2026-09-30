# syntax=docker/dockerfile:1
# ============================================================================
# equivalent-modeling-service 业务镜像
#
# 依赖已全部预置在基础镜像 repository/python3.10-aicm-base:1.2 中，
# 本层只做「解压 + 启动」，构建期不联网、不装任何包。
#
# 包内容（equivalent-modeling-service.tgz）解压后落在 /home/，即仓库根：
#   /home/app          Flask 应用 + Agent + skills
#   /home/static       前端静态资源（无 npm 构建，直接托管）
#   /home/mcp_server   MCP 仿真 Server
#   /home/aicm         仿真工具（run.py / examples/time_args.json / workload_generator/）
#   /home/docker       entrypoint.sh + check_workspace.py + check_db.py + checkpoint_db.py
#
# 路径契约：mcp_server/config.py 以 parents[1] 推导默认值（仓库根/aicm，本地开发用），
# 镜像 ENV 则显式钉住 AICM_MCP_SIM_TOOL_HOME=/home/aicm。
# 仿真工具保持 /home/aicm（镜像层）；数据挂载树**刻意移到 /home/data 下**：
#   /home/data/workspace   仿真任务产物（节点侧 /data/aicm/workspace）
#   /home/data/db          SQLite 数据库目录（节点侧 /data/aicm/db）
# 历史坑：数据挂载曾放在 /home/aicm 下，与工具目录同名 —— 整目录挂载 /home/aicm 时
# 会把镜像层里的工具代码遮掉（工具被迫落到宿主机）。工具目录与挂载路径分置后，
# /home/aicm 不再被任何挂载覆盖。
# 下面用 ENV 显式钉住，不依赖隐式推导。
#
# ── 数据库：默认 SQLite（本次改造）──
# 原先是「内嵌 PostgreSQL 14 服务端 + Unix socket」，现在是一**个文件**：
#   /home/data/db                  挂载点：部署侧必须把宿主机目录挂到这里
#   /home/data/db/equivalent_modeling_service.db    主库（WAL 模式下另有 -wal / -shm 伴生文件）
# 因此镜像里不再需要 PGDATA / PG_SOCKET_DIR / PG_BIN，也没有 initdb、版本守卫、
# 就绪探测与 socket 目录。数据库目录自检仍**不可豁免**（workspace 有 ALLOW_LOCAL
# 开关），因为数据库静默落在镜像层里意味着容器重建即丢全部会话历史。
# PG 逃生门保留：注入 DATABASE_URL=postgresql://... 即切回外部 PG（同一镜像，无需重建）。
# ============================================================================
FROM repository/python3.10-aicm-base:1.2

WORKDIR /home/

COPY equivalent-modeling-service*.tgz app.tgz

EXPOSE 5000 9000
ENV LANG C.UTF-8

# 解压应用包（aicm/ 工具目录保持 /home/aicm 不变；数据挂载点已移出到 /home/data，
# 因此工具目录不再与任何挂载路径同名，也不会被整目录挂载遮掉）。
RUN tar -xzf app.tgz && rm app.tgz

# 数据挂载点（workspace/db）**刻意不放在 /home/aicm 下**：/home/aicm 是仿真工具目录，
# 若数据挂载树与工具目录同名，整目录挂载会把镜像层里的工具代码遮掉（工具被迫落到
# 宿主机）。故数据统一放到 /home/data 下：
#   /home/data/workspace  仿真任务产物（节点侧 /data/aicm/workspace）
#   /home/data/db         SQLite 数据文件目录（节点侧 /data/aicm/db）
# 工作区（每个 task_id 一个子目录，仿真产物写入其 results/）与数据库目录都必须由
# 部署侧把宿主机目录挂到这里（k8s hostPath / docker -v），否则容器重建、镜像
# 升级会连产物与**全部会话历史**一起丢掉。
# 这里只预建挂载点并给出正确属主（容器以 uid 1000 运行），预建的另一个作用是
# 让 entrypoint 自检能明确报出「未挂载」，而不是静默写进镜像层。
#   - workspace 预置 0755：容器内 uid 1000 需在其下建 task 子目录；
#   - db        预置 0755：SQLite 需在该目录内创建主库与 -wal / -shm 伴生文件。
#     （改造前这里是 PGDATA 的 0700 —— 那是 PostgreSQL 对数据目录的特殊敏感点，
#      SQLite 无此要求，因此不再收窄权限。）
RUN mkdir -p /home/data/workspace \
    && chown 1000:1000 /home/data/workspace \
    && chmod +x /home/docker/entrypoint.sh \
    && mkdir -p /home/data/db \
    && chown 1000:1000 /home/data/db

# 设置权限
RUN chown 1000:1000 /home/ -R

# ---------------------------------------------------------------------------
# 运行期配置
# ---------------------------------------------------------------------------
# AICM_MCP_CONDA_ENV 必须显式置空：conda_launcher.build_simulation_command()
# 在 conda_env 为空串时回退为 [sys.executable, ...]，从而免装 conda。
# 端口两侧默认值已对齐为 9000（HEAD 5fa5f4b），此处显式写出作为单一来源。
# 数据库默认 SQLite：SQLITE_PATH 钉在挂载目录内；DATABASE_URL **留空**
# （留空即走 SQLite；集群侧注入 PG 串即切回外部 PostgreSQL 逃生门 ——
#  chart 侧是 values.yaml 的 config.databaseUrl，经 configMap.yaml 条件渲染下发）。
ENV PYTHONPATH=/home \
    FLASK_HOST=0.0.0.0 \
    FLASK_PORT=5000 \
    FLASK_DEBUG=false \
    FLASK_USE_RELOADER=false \
    AICM_MCP_HOST=0.0.0.0 \
    AICM_MCP_PORT=9000 \
    AICM_MCP_SIM_TOOL_HOME=/home/aicm \
    AICM_MCP_WORKSPACE_ROOT=/home/data/workspace \
    AICM_MCP_CONDA_ENV= \
    MCP_SERVER_URL=http://127.0.0.1:9000 \
    SQLITE_PATH=/home/data/db/equivalent_modeling_service.db \
    DATABASE_URL=

# 健康检查：两个进程都要活着。用 python 而非 curl。
# 改造前这里还有一段 pg_isready 探测 —— SQLite 无服务端，去掉；
# 数据库是否健康由 /api/health 覆盖（init_db 失败会让进程直接退出，容器即不健康）。
HEALTHCHECK --interval=30s --timeout=6s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request as u, sys; \
sys.exit(0 if all(u.urlopen(x, timeout=4).status < 400 for x in \
('http://127.0.0.1:5000/api/health', 'http://127.0.0.1:9000/health')) else 1)"

# 启动：MCP 仿真 Server + Flask（SSE + WebSocket）同容器多进程
CMD umask 0027 && bash /home/docker/entrypoint.sh
