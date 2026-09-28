"""等待内嵌 PostgreSQL 就绪，并确保目标库存在。

为什么需要这个脚本：
    app/main.py 在 create_app() 阶段就会执行 db_migration.init_db()，若数据库
    不可用会直接抛异常导致容器退出。改造后 PostgreSQL 由同一个 entrypoint 拉起，
    刚启动的实例可能还在 crash recovery（此时会明确拒绝新连接），因此这里先探测。

与改造前（等外部 PG）的区别：
    1. DATABASE_URL 为空不再静默 `return 0` —— 内嵌模式下 DSN 恒有值，为空一定是
       配置错误，静默跳过只会把问题推迟到 init_db 抛一个难懂的异常。
    2. 就绪判定叠加 pg_isready（PG 自带，比裸 TCP 探测更准确）；连接串优先用
       Unix socket，不做 TCP 端口探测。
    3. 顺带保证**目标库存在**：initdb 只建 postgres / template0 / template1，
       业务库 train_mesh_agent 需要显式 CREATE DATABASE。这一步放在这里而不是
       entrypoint 的 shell 里，是为了复用同一套连接与重试逻辑。

退出码：0 = 就绪；1 = 超时或配置错误（entrypoint 会据此终止容器）。
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

import psycopg2
from psycopg2 import sql

RETRIES = int(os.getenv("DB_WAIT_RETRIES", "60"))
DELAY = float(os.getenv("DB_WAIT_DELAY", "2"))

# 与 Dockerfile 中的镜像 ENV 对齐；两处任一改动都要同步
DEFAULT_DSN = "postgresql://postgres@/train_mesh_agent?host=/home/aicm/db/run"
DEFAULT_SOCKET_DIR = "/home/aicm/db/run"
ADMIN_DB = "postgres"


def log(message: str) -> None:
    print(f"[wait_for_db] {message}", flush=True)


def database_name(dsn: str) -> str:
    """从 DSN 取库名：postgresql://user@/dbname?host=/dir -> dbname"""
    tail = dsn.rsplit("/", 1)[-1]
    return tail.split("?")[0] or ADMIN_DB


def pg_isready(socket_dir: str) -> bool:
    """用 PG 自带的 pg_isready 做一次廉价探测。

    找不到 pg_isready 时返回 True（交给下面的真连接判定），避免因为 PATH 问题
    让容器永远起不来。
    """
    try:
        proc = subprocess.run(
            ["pg_isready", "-h", socket_dir, "-q"],
            capture_output=True,
            timeout=5,
        )
        return proc.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return True


def connect(dsn: str, dbname: str | None = None):
    """建立连接；指定 dbname 时只替换 DSN 的库名段（保留 host 等查询参数）。

    不用 psycopg2 的 dbname 关键字参数：socket 形态的 DSN 带 host=/path 查询
    参数，交由 psycopg2 统一解析更可靠。
    """
    if dbname is None:
        return psycopg2.connect(dsn, connect_timeout=3)

    prefix, _, tail = dsn.rpartition("/")
    query = ("?" + tail.split("?", 1)[1]) if "?" in tail else ""
    return psycopg2.connect(f"{prefix}/{dbname}{query}", connect_timeout=3)


def ensure_database(dsn: str, dbname: str) -> bool:
    """确保目标库存在。返回 True 表示已存在/已创建，False 表示建库失败。"""
    if dbname == ADMIN_DB:
        return True

    with connect(dsn, ADMIN_DB) as conn:
        conn.autocommit = True  # CREATE DATABASE 不能在事务块里执行
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (dbname,))
            if cur.fetchone():
                log(f"目标库 {dbname} 已存在")
                return True
            log(f"目标库 {dbname} 不存在，创建中 ...")
            cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(dbname)))

    # 建完立刻验证一次真连接（validate 一下，别只看 CREATE 的返回）
    with connect(dsn, dbname):
        pass
    log(f"目标库 {dbname} 创建完成并可连接")
    return True


def main() -> int:
    dsn = os.getenv("DATABASE_URL", "").strip() or DEFAULT_DSN
    if not os.getenv("DATABASE_URL", "").strip():
        log(f"DATABASE_URL 未设置，使用内嵌默认值：{DEFAULT_DSN}")
    socket_dir = os.getenv("PG_SOCKET_DIR", DEFAULT_SOCKET_DIR)
    dbname = database_name(dsn)

    last_exc: Exception | None = None
    for attempt in range(1, RETRIES + 1):
        try:
            if not pg_isready(socket_dir):
                raise RuntimeError(f"pg_isready 报告未就绪（socket 目录 {socket_dir}）")

            # 先连维护库：目标库可能还没建出来
            with connect(dsn, ADMIN_DB):
                pass

            if not ensure_database(dsn, dbname):
                return 1

            log(f"PostgreSQL 就绪（第 {attempt} 次尝试），目标库={dbname}")
            return 0
        except Exception as exc:  # noqa: BLE001 —— 就绪探测要吞掉所有异常继续重试
            last_exc = exc
            log(f"{attempt}/{RETRIES} 未就绪: {exc}")
            if attempt < RETRIES:
                time.sleep(DELAY)

    log(f"超时，PostgreSQL 仍不可用：{last_exc}")
    log(f"  排查建议：确认 PGDATA={os.getenv('PGDATA', '/home/aicm/db/data')} 已挂载且属主为容器运行用户；")
    log(f"           容器内可手工执行 pg_ctl -D $PGDATA status 查看日志。")
    return 1


if __name__ == "__main__":
    sys.exit(main())
