#!/usr/bin/env bash
# ============================================================================
# TrainMeshAgent 单容器 entrypoint
#   拉起 MCP 仿真 Server + Flask 应用，并保证 SIGTERM 时连带回收仿真子进程。
#   容器内仓库根为 /home（业务镜像解压位置），仿真工作区在 /data/aicm/workspace
#   （必须由部署侧挂载宿主机目录，否则启动即失败）。
# ============================================================================
set -euo pipefail

log() { printf '[entrypoint] %s\n' "$*"; }

# ---------- 仿真工作区自检：必须是挂载点且可写 ----------
# 放在等数据库之前：配置错误应当立刻暴露，而不是白等一轮 DB 探测。
# 兜底原则：宁可启动失败，也不要静默把仿真产物写进镜像层（容器重建即丢）。
: "${AICM_MCP_WORKSPACE_ROOT:=/data/aicm/workspace}"
export AICM_MCP_WORKSPACE_ROOT
log "workspace=${AICM_MCP_WORKSPACE_ROOT}"
python /home/docker/check_workspace.py || exit 1

# ---------- 等待 PostgreSQL（app 启动即执行建表迁移） ----------
python /home/docker/wait_for_db.py || exit 1

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
python -m app.main &
APP_PID=$!
log "Flask app started (pid=$APP_PID, port=${FLASK_PORT})"

shutdown() {
    log "shutting down ..."
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
