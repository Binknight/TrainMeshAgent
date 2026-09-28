#!/usr/bin/env bash
# ============================================================================
# TrainMeshAgent 单容器 entrypoint
#   拉起 内嵌 PostgreSQL + MCP 仿真 Server + Flask 应用，并保证 SIGTERM 时
#   连带回收仿真子进程与数据库（-m fast，不做 crash recovery）。
#
#   容器内仓库根为 /home（业务镜像解压位置），仿真工作区在 /data/aicm/workspace，
#   数据库在 /home/aicm/db（PGDATA=/home/aicm/db/data）。这两个目录都必须由部署
#   侧挂载宿主机目录，否则启动即失败。
# ============================================================================
set -euo pipefail

log() { printf '[entrypoint] %s\n' "$*"; }

# ---------- 仿真工作区自检：必须是挂载点且可写 ----------
# 放在其余步骤之前：配置错误应当立刻暴露，而不是白等一轮 DB 探测。
# 兜底原则：宁可启动失败，也不要静默把仿真产物写进镜像层（容器重建即丢）。
: "${AICM_MCP_WORKSPACE_ROOT:=/data/aicm/workspace}"
export AICM_MCP_WORKSPACE_ROOT
log "workspace=${AICM_MCP_WORKSPACE_ROOT}"
python /home/docker/check_workspace.py || exit 1

# ---------- 内嵌数据库：路径与工具 ----------
# PG_SOCKET_DIR 必须与 DATABASE_URL 的 host= 查询参数一致；
# 两处任一改动都要同步（镜像 ENV 是单一来源，这里只做兜底默认值）。
: "${PGDATA:=/home/aicm/db/data}"
: "${PG_SOCKET_DIR:=/home/aicm/db/run}"
: "${PG_BIN:=/usr/lib/postgresql/14/bin}"
# PGDATA 的父目录：check_db.py 要求「它或它的祖先」是挂载点
DB_ROOT="$(dirname "${PGDATA}")"
export PGDATA PG_SOCKET_DIR

# ---------- 数据库目录自检：必须在挂载点下、可写、0700、属主一致 ----------
# 与 workspace 自检同一哲学，但这里的失败代价更高：静默落在镜像层里意味着
# 容器重建即丢全部会话历史。因此不做「豁免后继续」的降级，只认挂载点。
python /home/docker/check_db.py || exit 1

# socket 目录在挂载点里，镜像层的预建目录会被挂载遮蔽，这里补建一次。
mkdir -p "${PG_SOCKET_DIR}"

# ---------- 版本守卫：PGDATA 与镜像内 PG 主版本强绑定 ----------
# PostgreSQL 遇到不同主版本的 PGDATA 会直接拒绝启动（且报错信息不直观）。
# 这里提前比对，把「镜像升了 PG 但数据目录还是老版本」这种运维事故讲清楚。
if [ -s "${PGDATA}/PG_VERSION" ]; then
    DATA_VERSION="$(tr -d '[:space:]' < "${PGDATA}/PG_VERSION")"
    # 以镜像内实际二进制的版本为准；pg_config 意外缺失时退回本文件钉住的 14
    BIN_VERSION="$("${PG_BIN}/pg_config" --version 2>/dev/null \
        | sed -n 's/.*PostgreSQL \([0-9][0-9]*\).*/\1/p')"
    BIN_VERSION="${BIN_VERSION:-14}"
    if [ "${DATA_VERSION}" != "${BIN_VERSION}" ]; then
        log "FATAL: PGDATA 版本(${DATA_VERSION}) 与镜像内 PostgreSQL(${BIN_VERSION}) 不一致"
        log "       数据目录与 PG 主版本强绑定，不能直接跨版本启动。"
        log "       升级路径：旧容器内 pg_dumpall -> 换新镜像 -> 空数据目录 initdb -> psql 恢复。"
        log "       详见 docs/使用指南.md「数据库」章节。"
        exit 1
    fi
    log "数据库目录版本校验通过（PG ${DATA_VERSION}）"
fi

# ---------- 初始化 / 启动 PostgreSQL ----------
if [ ! -s "${PGDATA}/PG_VERSION" ]; then
    log "PGDATA 为空，执行 initdb（${PGDATA}）..."
    # --auth=trust：数据只在本容器内经 Unix socket 可达，不开 TCP，无需口令
    # --locale=C.UTF-8：与镜像 LANG 对齐，避免依赖宿主机 locale
    # 若 /home/aicm/db 被挂载而属主不是容器用户，initdb 会以权限错误失败，
    # 这里的提示指向部署侧（hostPath 下由 initContainer chown）。
    if ! "${PG_BIN}/initdb" -D "${PGDATA}" -U postgres \
            --auth=trust --encoding=UTF8 --locale=C.UTF-8; then
        log "FATAL: initdb 失败。检查 ${DB_ROOT} 是否由容器运行用户（uid=$(id -u)）可写；"
        log "       k8s hostPath 场景需由 initContainer 执行 chown。"
        exit 1
    fi
    log "initdb 完成"
else
    log "检测到已初始化的 PGDATA，复用现有数据目录"
fi

# 性能相关参数说明（都显式写出，不依赖默认值）：
#   listen_addresses=''        只走 Unix socket，不开 TCP
#   unix_socket_directories     PG 默认的 /var/run/postgresql 在 uid 1000 下建不出来
#   shared_buffers=64M          对齐容器默认 --shm-size=64MB，避免共享内存段扩容失败
#   max_connections=20          对齐 app/db.py 的 minconn=2/maxconn=10
#   logging_collector=off       日志直接进 stdout/stderr，由容器日志采集
#   -k 与 -c unix_socket_directories 重复指定是有意的：-k 在命令行层面兜底
setsid "${PG_BIN}/postgres" \
    -D "${PGDATA}" \
    -c listen_addresses='' \
    -c unix_socket_directories="${PG_SOCKET_DIR}" \
    -c shared_buffers=64MB \
    -c max_connections=20 \
    -c logging_collector=off &
PG_PID=$!
log "PostgreSQL 启动中 (pid=${PG_PID}, PGDATA=${PGDATA}, socket=${PG_SOCKET_DIR})"

# ---------- PG 就绪探测 ----------
# setsid 之后拿到的是进程组组长，直接探测 socket 比轮询进程状态可靠。
PG_RETRIES=60
for i in $(seq 1 "${PG_RETRIES}"); do
    if pg_isready -q -h "${PG_SOCKET_DIR}" -U postgres; then
        log "PostgreSQL 就绪（第 ${i} 次探测）"
        break
    fi
    if ! kill -0 "${PG_PID}" 2>/dev/null; then
        log "FATAL: PostgreSQL 进程已退出（pid=${PG_PID}）。"
        log "       常见原因：shared_memory 不足（--shm-size 太小）、数据目录权限错误、"
        log "       PGDATA 磁盘写满。上面 postgres 的输出即为崩溃原因。"
        exit 1
    fi
    if [ "${i}" -eq "${PG_RETRIES}" ]; then
        log "FATAL: 等待 PostgreSQL 就绪超时（${PG_RETRIES}s）"
        exit 1
    fi
    sleep 1
done

# ---------- 确保业务库存在（initdb 只建 postgres/template*）----------
# app/main.py 在 create_app() 阶段就会执行 db_migration.init_db()，若数据库不可用
# 会直接抛异常导致容器退出，因此这里在拉起业务进程前完成建库与最终校验。
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
    # -m fast：断开现有连接后立刻退出（rollback 未提交事务）。
    # 绝不用 -m immediate：那等于 kill -9，下次启动要走 crash recovery，
    # 极端情况下会丢最近若干已提交事务。
    # pg_ctl 会自己找监听 pid 并等它退出；失败也不阻断其余清理。
    "${PG_BIN}/pg_ctl" -D "${PGDATA}" -m fast stop -w -t 30 \
        || log "警告：pg_ctl stop 失败（PG 可能已退出），继续清理业务进程"
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
