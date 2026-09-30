"""数据库连接层：双后端（SQLite 默认 / PostgreSQL 逃生门）。

调用方契约（勿改）：`get_db()` 是一个上下文管理器，yield 出连接，
正常退出时 commit、异常时 rollback、最终归还连接池。`app/dao` 的全部代码
建立在这三个保证上，且路由层与 Agent 层通过 DAO 间接使用，因此这里的对外
行为是整条链路的公共前提。

后端侦测
--------
由 `DATABASE_URL` 前缀决定（见 `app/dbapi.detect_backend`）：
  - `postgresql://...` / `postgres://...` → PG 后端（逃生门）
  - 其余（空 / 未设置 / `sqlite://...`）→ SQLite 后端（默认）

刻意不引入 `DB_BACKEND` 这类独立开关：开关与 URL 一旦不一致就会出现
第四种状态，届时「为什么连不上」会成为排查陷阱。

两种后端的失败模式不同
----------------------
- PG：连不上可能是 PG 未就绪或意外重启 → **建池**与**取连接**两处都做有限重试。
- SQLite：没有「服务未就绪」这回事，失败通常是路径不存在、权限不对或磁盘满
  —— 重试无意义，直接抛出带路径的可读信息。唯一的可重试场景是
  `database is locked`，那由 PRAGMA `busy_timeout` 在引擎内部等待，不在这里重试。
"""

from __future__ import annotations

import os
import queue
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from app.config import config
from app.dbapi import (
    BACKEND_POSTGRES,
    BACKEND_SQLITE,
    DBConnection,
    configure_sqlite_connection,
    detect_backend,
    ensure_utf8_console,
)

# 所有后端都会经过本模块，因此在这里统一兜住非 UTF-8 控制台（中文日志的 print
# 在 Windows cp1252 下会抛 UnicodeEncodeError，且往往发生在建表成功之后）。
ensure_utf8_console()

# 建池/取连接的重试参数（秒）—— 仅 PG 后端使用
CONNECT_RETRIES = int(os.getenv("DB_CONNECT_RETRIES", "10"))
CONNECT_RETRY_DELAY = float(os.getenv("DB_CONNECT_RETRY_DELAY", "1.5"))

# 连接数上限：对齐原 PG 池的 minconn=2 / maxconn=10 上限。
# SQLite 下池本身就是「同一连接同一时刻只被一个线程借用」的串行化机制，
# 但写事务仍然串行（WAL 是库级单写者），池大小只影响并发读。
_POOL_MIN = 2
_POOL_MAX = 10

_backend: str | None = None
_pg_pool: Any = None

_sqlite_lock = threading.Lock()
_sqlite_all: set[sqlite3.Connection] = set()
_sqlite_idle: queue.LifoQueue | None = None


def backend() -> str:
    """当前后端（postgres / sqlite），由 DATABASE_URL 侦测。"""
    global _backend
    if _backend is None:
        _backend = detect_backend(config.DATABASE_URL)
    return _backend


def _describe_target() -> str:
    """把连接目标还原成便于排查的形式，但不泄露口令。"""
    if backend() == BACKEND_SQLITE:
        return f"{config.SQLITE_PATH}（SQLite 文件；检查父目录存在、可写、且在挂载点下）"
    dsn = config.DATABASE_URL
    # socket 形态：postgresql://user@/dbname?host=/dir
    if "?host=" in dsn or "@/" in dsn:
        return f"{dsn}（Unix socket；检查 PGDATA={config.PGDATA} 与 socket 目录 {config.PG_SOCKET_DIR}）"
    return f"{dsn.split('@')[0].split('://')[0]}://***@{dsn.split('@')[-1]}（TCP）"


# ────────────────────────── SQLite 后端 ──────────────────────────


def _open_sqlite_connection() -> sqlite3.Connection:
    path = config.SQLITE_PATH
    parent = Path(path).expanduser().resolve().parent
    if not parent.exists():
        # 不静默创建：父目录本该由镜像预建/部署侧挂载，缺失说明部署形态不对，
        # 这时"自动建目录"会让数据库悄悄落在镜像层里（与 workspace 同源的失效模式）。
        raise RuntimeError(
            f"SQLite 数据目录不存在：{parent}\n"
            f"  SQLITE_PATH={path}\n"
            f"  提示：容器内该目录应由镜像预建并由部署侧挂载（/home/data/db）。"
        )

    conn = sqlite3.connect(
        path,
        timeout=max(config.SQLITE_BUSY_TIMEOUT_MS / 1000.0, 0.1),
        check_same_thread=False,  # 连接由池保证同一时刻只被一个线程借用
        isolation_level="",       # 默认：隐式开事务，由 get_db() 显式 commit/rollback
    )
    conn.row_factory = None       # 保持 tuple 行，DAO 依赖位置下标
    configure_sqlite_connection(
        conn, config.SQLITE_BUSY_TIMEOUT_MS, config.SQLITE_SYNCHRONOUS
    )
    return conn


def _sqlite_pool_get() -> sqlite3.Connection:
    global _sqlite_idle
    with _sqlite_lock:
        if _sqlite_idle is None:
            _sqlite_idle = queue.LifoQueue(maxsize=_POOL_MAX)
            first = _open_sqlite_connection()   # 首次失败要显式抛出，不吞
            _sqlite_all.add(first)
            _sqlite_idle.put(first)
        try:
            return _sqlite_idle.get_nowait()
        except queue.Empty:
            pass
        if len(_sqlite_all) < _POOL_MAX:
            try:
                conn = _open_sqlite_connection()
            except Exception as exc:
                raise RuntimeError(
                    f"打开 SQLite 连接失败：{exc}\n  目标 {_describe_target()}"
                ) from exc
            _sqlite_all.add(conn)
            return conn
    # 池已满：退化为新建一个临时连接（不入选池集合），依赖 WAL + busy_timeout 自保。
    # 正常负载下不会走到这里（DAO 调用都是短事务）。
    return _open_sqlite_connection()


def _sqlite_pool_put(conn: sqlite3.Connection) -> None:
    with _sqlite_lock:
        if _sqlite_idle is None or conn not in _sqlite_all:
            try:
                conn.close()
            except Exception:
                pass
            return
        try:
            _sqlite_idle.put_nowait(conn)
        except queue.Full:
            _sqlite_all.discard(conn)
            try:
                conn.close()
            except Exception:
                pass


# ────────────────────────── PostgreSQL 后端 ──────────────────────────


def _make_pg_pool():
    """建池，带有限重试。

    区分两类失败：
      - OperationalError：连不上（PG 未就绪 / socket 目录不存在 / 端口错）→ 重试
      - 其他异常（如认证失败、库不存在）：重试没有意义，直接抛出并给出可读信息
    """
    import psycopg2
    import psycopg2.pool

    last_exc: Exception | None = None
    for attempt in range(1, CONNECT_RETRIES + 1):
        try:
            return psycopg2.pool.ThreadedConnectionPool(
                minconn=_POOL_MIN,
                maxconn=_POOL_MAX,
                dsn=config.DATABASE_URL,
                application_name="equivalent-modeling-service",
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
                f"  提示：若用外部 PG，请确认库已创建且账号有 DDL 权限；"
                f"若手工在容器内排查，可用 pg_isready -h {config.PG_SOCKET_DIR} -U postgres。"
            ) from exc

    raise RuntimeError(
        f"无法连接数据库（重试 {CONNECT_RETRIES} 次后放弃）：{last_exc}\n"
        f"  DATABASE_URL target: {_describe_target()}\n"
        f"  PGDATA={config.PGDATA}"
    ) from last_exc


def get_pool():
    """返回后端对应的连接池句柄（PG 返回 ThreadedConnectionPool；SQLite 返回内部池对象）。"""
    if backend() == BACKEND_POSTGRES:
        global _pg_pool
        if _pg_pool is None:
            _pg_pool = _make_pg_pool()
        return _pg_pool
    return _sqlite_idle


@contextmanager
def get_db():
    """连接上下文：正常退出 commit、异常 rollback、最终归还。**契约不可变**。"""
    conn = _acquire()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        _release(conn)


def _acquire() -> DBConnection:
    if backend() == BACKEND_POSTGRES:
        pool = get_pool()
        return DBConnection(pool.getconn(), BACKEND_POSTGRES)
    return DBConnection(_sqlite_pool_get(), BACKEND_SQLITE)


def _release(conn: DBConnection) -> None:
    if backend() == BACKEND_POSTGRES:
        get_pool().putconn(conn.raw)
        return
    try:
        # 归还前回滚未提交的隐式事务：否则下一个借用者会继承半截事务，
        # 极端情况下把上一个请求的写一起提交/回滚。
        conn.raw.rollback()
    except Exception:
        pass
    _sqlite_pool_put(conn.raw)


def reset_for_tests() -> None:
    """仅供测试/校验脚本使用：丢弃池与缓存的后端判定。"""
    global _backend, _pg_pool, _sqlite_idle
    with _sqlite_lock:
        if _sqlite_idle is not None:
            while True:
                try:
                    _sqlite_all.discard(_sqlite_idle.get_nowait())
                except queue.Empty:
                    break
        for conn in list(_sqlite_all):
            try:
                conn.close()
            except Exception:
                pass
        _sqlite_all.clear()
        _sqlite_idle = None
    if _pg_pool is not None:
        try:
            _pg_pool.closeall()
        except Exception:
            pass
    _pg_pool = None
    _backend = None
