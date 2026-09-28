"""等待 PostgreSQL 就绪。

app/main.py 在 create_app() 阶段就会执行 db_migration.init_db()，若数据库不可用
会直接抛异常导致容器退出，因此 entrypoint 先做就绪探测。
"""

import os
import sys
import time

import psycopg2

RETRIES = int(os.getenv("DB_WAIT_RETRIES", "60"))
DELAY = float(os.getenv("DB_WAIT_DELAY", "2"))


def main() -> int:
    dsn = os.getenv("DATABASE_URL", "").strip()
    if not dsn:
        print("[wait_for_db] DATABASE_URL 未设置，跳过等待")
        return 0

    for attempt in range(1, RETRIES + 1):
        try:
            psycopg2.connect(dsn, connect_timeout=3).close()
            print(f"[wait_for_db] PostgreSQL 就绪（第 {attempt} 次尝试）")
            return 0
        except Exception as exc:  # noqa: BLE001
            print(f"[wait_for_db] {attempt}/{RETRIES} 未就绪: {exc}")
            time.sleep(DELAY)

    print("[wait_for_db] 超时，PostgreSQL 仍不可用")
    return 1


if __name__ == "__main__":
    sys.exit(main())
