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
#   /home/docker       entrypoint.sh + check_workspace.py + check_db.py + checkpoint_db.py
#
# 路径契约：mcp_server/config.py 以 parents[1] 推导默认值，而代码被解压到 /home，
# 故 sim_tool_home=/home/aicm。工作区与数据库则**都放在 /home/aicm 下**，二者
# 统一挂载（见下方 mkdir 与 entrypoint 自检）：
#   /home/aicm/workspace   仿真任务产物（节点侧 /data/aicm/workspace）
#   /home/aicm/db          SQLite 数据库目录（节点侧 /data/aicm/db）
# 容器内父目录 /home/aicm 不被整体挂载，只是两个子目录各自挂载，
# 因此仿真工具本体（/home/aicm/run.py 等）仍在镜像层里，不受挂载影响。
# 下面用 ENV 显式钉住，不依赖隐式推导。
#
# ── 数据库：默认 SQLite（本次改造）──
# 原先是「内嵌 PostgreSQL 14 服务端 + Unix socket」，现在是一**个文件**：
#   /home/aicm/db                  挂载点：部署侧必须把宿主机目录挂到这里
#   /home/aicm/db/train_mesh_agent.db    主库（WAL 模式下另有 -wal / -shm 伴生文件）
# 因此镜像里不再需要 PGDATA / PG_SOCKET_DIR / PG_BIN，也没有 initdb、版本守卫、
# 就绪探测与 socket 目录。数据库目录自检仍**不可豁免**（workspace 有 ALLOW_LOCAL
# 开关），因为数据库静默落在镜像层里意味着容器重建即丢全部会话历史。
# PG 逃生门保留：注入 DATABASE_URL=postgresql://... 即切回外部 PG（同一镜像，无需重建）。
# ============================================================================
FROM repository/python3.10-aicm-base:1.0

WORKDIR /home/

COPY TrainMeshAgent*.tgz app.tgz

EXPOSE 5000 9000
ENV LANG C.UTF-8

# 解压应用包
RUN tar -xzf app.tgz && rm app.tgz

# 仿真任务工作区与数据库目录，两者都在 /home/aicm 下：
# 工作区（每个 task_id 一个子目录，仿真产物写入其 results/）与数据库目录都必须由
# 部署侧把宿主机目录挂到这个路径（k8s hostPath / docker -v），否则容器重建、镜像
# 升级会连产物与**全部会话历史**一起丢掉。
# 这里只预建挂载点并给出正确属主（容器以 uid 1000 运行），预建的另一个作用是
# 让 entrypoint 自检能明确报出「未挂载」，而不是静默写进镜像层。
#
# 顺序有讲究：必须在 `tar -xzf` **之后**创建，否则 tar 展开 aicm/ 时会重建该目录。
#   - workspace 预置 0755：容器内 uid 1000 需在其下建 task 子目录；
#   - db        预置 0755：SQLite 需在该目录内创建主库与 -wal / -shm 伴生文件。
#     （改造前这里是 PGDATA 的 0700 —— 那是 PostgreSQL 对数据目录的特殊敏感点，
#      SQLite 无此要求，因此不再收窄权限。）
RUN mkdir -p /home/aicm/workspace \
    && chown 1000:1000 /home/aicm/workspace \
    && chmod +x /home/docker/entrypoint.sh \
    && mkdir -p /home/aicm/db \
    && chown 1000:1000 /home/aicm/db

# 设置权限
RUN chown 1000:1000 /home/ -R

# ---------------------------------------------------------------------------
# 运行期配置
# ---------------------------------------------------------------------------
# AICM_MCP_CONDA_ENV 必须显式置空：conda_launcher.build_simulation_command()
# 在 conda_env 为空串时回退为 [sys.executable, ...]，从而免装 conda。
# 端口两侧默认值已对齐为 9000（HEAD 5fa5f4b），此处显式写出作为单一来源。
# 数据库默认 SQLite：SQLITE_PATH 钉在挂载目录内；DATABASE_URL **留空**
# （留空即走 SQLite；集群 Secret 注入 PG 串即切回外部 PostgreSQL 逃生门）。
ENV PYTHONPATH=/home \
    FLASK_HOST=0.0.0.0 \
    FLASK_PORT=5000 \
    FLASK_DEBUG=false \
    FLASK_USE_RELOADER=false \
    AICM_MCP_HOST=0.0.0.0 \
    AICM_MCP_PORT=9000 \
    AICM_MCP_SIM_TOOL_HOME=/home/aicm \
    AICM_MCP_WORKSPACE_ROOT=/home/aicm/workspace \
    AICM_MCP_CONDA_ENV= \
    MCP_SERVER_URL=http://127.0.0.1:9000 \
    SQLITE_PATH=/home/aicm/db/train_mesh_agent.db \
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
