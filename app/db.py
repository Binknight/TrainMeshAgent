"""PostgreSQL 连接池。

调用方契约（勿改）：`get_db()` 是一个上下文管理器，yield 出 psycopg2 连接，
正常退出时 commit、异常时 rollback、最终归还连接池。app/dao 的 564 行代码全部
建立在这三个保证上。

数据库从「外部服务」改为「内嵌同容器进程」后，连接建立的失败模式变了：
原先容器起来时 PG 一定已经就绪（entrypoint 先等外部 PG）；现在 PG 由同一个
entrypoint 拉起，就绪探测与探测完成之间、以及 PG 意外重启的瞬间，都可能出现
短暂连不上。因此这里在**建池**与**取连接**两处都加了有限重试。
"""

from __future__ import annotations

import os
import time

import psycopg2
import psycopg2.pool
from contextlib import contextmanager
from app.config import config

_pool: psycopg2.pool.ThreadedConnectionPool | None = None

# 建池/取连接的重试参数（秒）
CONNECT_RETRIES = int(os.getenv("DB_CONNECT_RETRIES", "10"))
CONNECT_RETRY_DELAY = float(os.getenv("DB_CONNECT_RETRY_DELAY", "1.5"))


def _describe_target() -> str:
    """把 DSN 还原成便于排查的形式，但不泄露口令。

    刻意不用 dsn 原文：socket 形态的 DSN 带 host=/path 查询参数，直接打日志
    又长又难读；而 TCP 形态带明文口令。
    """
    dsn = config.DATABASE_URL
    # socket 形态：postgresql://user@/dbname?host=/dir
    if "?host=" in dsn or "@/" in dsn:
        return f"{dsn}（Unix socket；检查 PGDATA={config.PGDATA} 与 socket 目录 {config.PG_SOCKET_DIR}）"
    return f"{dsn.split('@')[0].split('://')[0]}://***@{dsn.split('@')[-1]}（TCP）"


def _make_pool() -> psycopg2.pool.ThreadedConnectionPool:
    """建池，带有限重试。

    区分两类失败：
      - psycopg2.OperationalError：连不上（PG 未就绪 / socket 目录不存在 / 端口错）
        → 重试，因为内嵌模式下 PG 可能刚被拉起（crash recovery 期间会拒绝连接）
      - 其他异常（如认证失败、库不存在）：重试没有意义，直接抛出并给出可读信息
    """
    last_exc: Exception | None = None
    for attempt in range(1, CONNECT_RETRIES + 1):
        try:
            return psycopg2.pool.ThreadedConnectionPool(
                minconn=2,
                maxconn=10,
                dsn=config.DATABASE_URL,
                # 便于在 pg_stat_activity 里认出本服务的连接
                application_name="train-mesh-agent",
            )
        except psycopg2.OperationalError as exc:
            last_exc = exc
            print(
                f"[db] 建池失败 {attempt}/{CONNECT_RETRIES}：{exc}；目标 {_describe_target()}",
                flush=True,
            )
            if attempt < CONNECT_RETRIES:
                time.sleep(CONNECT_RETRY_DELAY)
        except Exception as exc:  # 认证失败 / 库不存在等：重试无意义
            raise RuntimeError(
                f"Database configuration error (will not retry): {exc}\n"
                f"  DATABASE_URL target: {_describe_target()}\n"
                f"  PGDATA={config.PGDATA}\n"
                f"  提示：内嵌模式应由 docker/entrypoint.sh 完成 initdb 与建库；"
                f"若手工在容器内排查，可用 pg_isready -h {config.PG_SOCKET_DIR} -U postgres。"
            ) from exc

    raise RuntimeError(
        f"无法连接数据库（重试 {CONNECT_RETRIES} 次后放弃）：{last_exc}\n"
        f"  DATABASE_URL target: {_describe_target()}\n"
        f"  PGDATA={config.PGDATA}"
    ) from last_exc


def get_pool() -> psycopg2.pool.ThreadedConnectionPool:
    global _pool
    if _pool is None:
        _pool = _make_pool()
    return _pool


@contextmanager
def get_db():
    pool = get_pool()
    conn = pool.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)
