#!/usr/bin/env bash
# ============================================================================
# TrainMeshAgent 单容器 entrypoint
#   拉起 MCP 仿真 Server + Flask 应用，保证 SIGTERM 时连带回收仿真子进程。
#
#   容器内仓库根为 /home（业务镜像解压位置）。仿真工作区与数据库目录都挂在
#   /home/aicm 下：workspace=/home/aicm/workspace，db=/home/aicm/db。
#   这两个目录都必须由部署侧挂载宿主机目录，否则启动即失败。
#
#   ── 本次改造：内嵌 PostgreSQL 已替换为 SQLite ──────────────────────────────
#   原先这里还有六段：PG 版本守卫 → 空目录 initdb → 起 postgres →
#   pg_isready 就绪探测 → 建库 → 停机 pg_ctl stop。SQLite 是进程内库、无服务端、
#   无主版本概念，这六段全部不再需要；启动链因此显著变短、失败模式也少了几种。
#   （另有一处「PG 目录自检」是**改写**而非删除：语义从"PGDATA 在挂载点下且 0700"
#   变成"数据库目录在挂载点下且可写"，见下面第 2 步。）
#
#   保留的同源原则：
#     * DB 目录仍必须是挂载点（静默写进镜像层 = 容器重建即丢全部会话历史）
#     * PG 逃生门仍在：注入 DATABASE_URL 指向外部 PG 时，app 侧自动切换后端
#       （此时需要外部 PG 已就绪，但那是部署方的责任，本脚本不再代为等待）
# ============================================================================
set -euo pipefail

log() { printf '[entrypoint] %s\n' "$*"; }

# ---------- 仿真工作区自检：必须是挂载点且可写 ----------
# 放在其余步骤之前：配置错误应当立刻暴露，而不是白等一轮 DB 探测。
# 兜底原则：宁可启动失败，也不要静默把仿真产物写进镜像层（容器重建即丢）。
: "${AICM_MCP_WORKSPACE_ROOT:=/home/aicm/workspace}"
export AICM_MCP_WORKSPACE_ROOT
log "workspace=${AICM_MCP_WORKSPACE_ROOT}"
python /home/docker/check_workspace.py || exit 1

# ---------- 数据库目录自检：必须在挂载点下且可写 ----------
# 默认后端是 SQLite，数据文件 SQLITE_PATH 默认 /home/aicm/db/train_mesh_agent.db。
# 与 workspace 自检同一哲学，但这里的失败代价更高：静默落在镜像层里意味着
# 容器重建即丢全部会话历史。因此不做「豁免后继续」的降级，只认挂载点。
# PG 逃生门（DATABASE_URL 指向外部 PG）下这个目录不会被使用，check_db.py
# 会据此跳过挂载点校验，只提示。
: "${SQLITE_PATH:=/home/aicm/db/train_mesh_agent.db}"
export SQLITE_PATH
python /home/docker/check_db.py || exit 1

: "${AICM_MCP_HOST:=0.0.0.0}"
: "${AICM_MCP_PORT:=9000}"
: "${FLASK_PORT:=5000}"
export AICM_MCP_HOST AICM_MCP_PORT FLASK_PORT

# ---------- MCP 仿真 Server ----------
# setsid 放进独立进程组：它下面挂的是 run.py 仿真子进程，
# 退出时按进程组整体回收，避免产生孤儿进程。
setsid python -m mcp_server &
MCP_PID=$!
log "MCP server started (pid=$MCP_PID, ${AICM_MCP_HOST}:${AICM_MCP_PORT})"

# ---------- Flask 应用（SSE 对话流 + WebSocket 仿真状态） ----------
# 建表迁移由 app/main.py 的 create_app() 执行（app/db_migration.init_db()），
# SQLite 下 init_db 会先做一次可写性 preflight，因此不再需要单独的等库脚本。
python -m app.main &
APP_PID=$!
log "Flask app started (pid=$APP_PID, port=${FLASK_PORT})"

shutdown() {
    log "shutting down ..."
    # SQLite 无服务端可停；这里只做一次 WAL checkpoint（TRUNCATE），
    # 让 -wal 的内容并回主库、伴生文件清空，简化停机后的备份与体积判断。
    # 失败不阻断清理（最多留下 -wal，下次打开自动合并）。
    python /home/docker/checkpoint_db.py || log "警告：SQLite checkpoint 失败（不影响数据，下次启动会自动合并 WAL）"
    # 负 PID = 杀整个进程组，连带回收 run.py 仿真子进程
    kill -TERM "-${MCP_PID}" 2>/dev/null || true
    kill -TERM "${APP_PID}" 2>/dev/null || true
    wait "${MCP_PID}" 2>/dev/null || true
    wait "${APP_PID}" 2>/dev/null || true
    log "bye"
    exit 0
}
trap shutdown SIGTERM SIGINT

# ---------- 任一进程退出则整体退出（bash 4.3+） ----------
kill -0 "${MCP_PID}" 2>/dev/null && kill -0 "${APP_PID}" 2>/dev/null || { log "failed to start"; shutdown; }

set +e
wait -n "${APP_PID}" "${MCP_PID}"
CODE=$?
set -e

log "a service exited (code=${CODE}), stopping container"
shutdown
