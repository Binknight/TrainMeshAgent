"""SQL 方言适配层：让 `app/dao` 的 SQL 文案对 PostgreSQL 与 SQLite 都成立。

设计原则
--------
1. **DAO 里的 SQL 保持原样**（继续写 `%s` 占位符、`NOW()`、`EXCLUDED.x`）。
   26 个 DAO 函数如果为了换方言而全量改写参数风格，是纯噪音改动且极易改错。
   方言差异集中到本模块，由 `translate()` 在**语句下发前**做词法级转换。

2. **转换必须走词法扫描，不能裸 `str.replace`**。
   `%s` / `NOW()` 一旦出现在字符串字面量里（如 `WHERE name = 'NOW()'`），
   裸替换会把数据写坏。`translate()` 逐个字符扫描并跟踪引号状态，
   只在「引号之外」替换。这也是 `scripts/verify_static.py` 会断言的契约。

3. **只在 SQLite 后端转换**。PG 后端下 `translate()` 原样返回，
   因此「保留 `DATABASE_URL` 即回到原行为」这条逃生门是真的逐字不变。

不可由词法转换处理、必须写在 SQL 里的差异
------------------------------------------
`name_key = ANY(%s)` 这类 PG 专有数组操作符，SQLite 无对应语法。
DAO 侧统一改写为 `name_key IN (?, ?, ...)`（占位符仍由本模块按需转换），
候选集仍由 Python 侧提供，语义等价。`ILIKE` 统一写 `LIKE`：
PG 对 `text` 是大小写敏感、SQLite 对 ASCII 大小写不敏感，
但该查询另有 `name_key IN (...)` 兜底，两侧行为均可接受。
"""

from __future__ import annotations

import re
import sqlite3
from typing import Any

# 后端标识（全仓库统一用这两个字面量）
BACKEND_POSTGRES = "postgres"
BACKEND_SQLITE = "sqlite"

# 参数风格：SQLite 用 qmark
_QMARK = "?"

# ── 词法转换用的字面量正则 ──
#   $$...$$  Dollar-quoting（PG 特有）
#   '...'    SQLite/PG 单引号字符串（'' 为转义）
#   "..."    双引号标识符
#   `...`    SQLite 兼容的反引号标识符
#   [...]    SQLite 兼容的方括号标识符
_LITERAL_RE = re.compile(
    r"""
      (?P<dollar>\$(?P<dollar_tag>[A-Za-z_][A-Za-z0-9_]*)?\$)
    | (?P<single>'(?:[^']|'')*')
    | (?P<double>"(?:[^"]|"")*")
    | (?P<backtick>`[^`]*`)
    | (?P<bracket>\[[^\]]*\])
    """,
    re.VERBOSE,
)

# 占位符后允许出现的类型转换/转义：CAST(%s AS ...) / %%
# 只把紧跟在参数位置上的 `%s` 当作占位符。
_PLACEHOLDER_RE = re.compile(r"%s")
# NOW() 函数调用（大小写不敏感）
_NOW_RE = re.compile(r"\bNOW\s*\(\s*\)", re.IGNORECASE)
# upsert 伪表：PG 写 EXCLUDED，SQLite 关键字是 excluded
_EXCLUDED_RE = re.compile(r"\bEXCLUDED\b", re.IGNORECASE)


def _translate_segment(segment: str) -> str:
    """对「引号之外」的 SQL 片段做方言替换。"""
    segment = _PLACEHOLDER_RE.sub(_QMARK, segment)
    segment = _NOW_RE.sub("CURRENT_TIMESTAMP", segment)
    segment = _EXCLUDED_RE.sub("excluded", segment)
    return segment


def translate(sql: str, backend: str) -> str:
    """把 DAO 的 SQL 文案转换为目标方言。

    - backend == postgres：原样返回（逃生门逐字不变）
    - backend == sqlite：`%s`→`?`、`NOW()`→`CURRENT_TIMESTAMP`、`EXCLUDED`→`excluded`
      仅作用于引号之外的部分。

    遇到未闭合的 dollar-quote（例如游标声明）会抛 `TranslationError` —— 宁可显式失败，
    也不要静默产出半替换的 SQL。
    """
    if backend == BACKEND_POSTGRES:
        return sql

    out: list[str] = []
    pos = 0
    length = len(sql)
    while pos < length:
        match = _LITERAL_RE.search(sql, pos)
        if match is None:
            out.append(_translate_segment(sql[pos:]))
            break

        if match.start() > pos:
            out.append(_translate_segment(sql[pos:match.start()]))

        if match.lastgroup == "dollar":
            tag = match.group("dollar_tag")
            closer = f"${tag}$" if tag else "$$"
            end = sql.find(closer, match.end())
            if end == -1:
                raise TranslationError(
                    f"dollar-quoted 字面量未闭合，无法安全转换：...{sql[match.start():match.start()+40]}"
                )
            end += len(closer)
            # dollar-quote 内容是数据，不替换
            out.append(sql[match.start():end])
            pos = end
        else:
            out.append(match.group(0))
            pos = match.end()

    return "".join(out)


class TranslationError(RuntimeError):
    """SQL 无法被安全地转换到目标方言。"""


# ── 后端侦测 ──

_PG_SCHEMES = ("postgres://", "postgresql://")


def detect_backend(database_url: str | None) -> str:
    """按 DSN 前缀侦测后端，避免引入 `DB_BACKEND` 这类双开关。

    刻意不引入独立开关：开关与 URL 一旦不一致就会出现第四种状态，
    届时「为什么连不上」会成为排查陷阱。

    - 以 postgres:// / postgresql:// 开头 → PG 后端（逃生门）
    - 其余（空串 / 未设置 / sqlite://...）→ SQLite 后端（默认）
    """
    url = (database_url or "").strip()
    if url.lower().startswith(_PG_SCHEMES):
        return BACKEND_POSTGRES
    return BACKEND_SQLITE


# ── 类型/取值归一 ──
# SQLite 没有原生 BOOLEAN：写入 True/False 会存成 1/0，读回是 int。
# 对外 JSON 形态必须稳定（前端有 `=== true` 之类的判断），因此出口统一转换。


def to_bool(value: Any) -> bool | None:
    """把列值归一为 bool|None。

    SQLite 存储形态只有 0/1（无原生布尔类型），因此这里只需处理「None 透传 +
    其余走 truthiness」。PG 侧读回本来就是 bool，经此函数保持不变。
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    return bool(value)


# 需要归一为 bool 的列：这些列的读取结果会直接进入 `app/models/schemas.py` 的
# pydantic 模型（如 `is_equivalent: bool`、`has_shared_expert: bool`）或下发给前端
# 做 `=== true` 判断。SQLite 读回 1/0，pydantic 严格布尔校验会直接抛错，
# 因此出口归一 **不是**可选的美化，而是必需的契约。
BOOL_COLUMNS = frozenset({
    "is_simulated",
    "is_equivalent",
    "has_shared_expert",
    "no_time_accumulation",
    "visual_json_output",
    "comm_group_output",
    "debug_time",
})


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    """就地把 dict 里的布尔列归一为 bool（None 保持 None）。"""
    for key in BOOL_COLUMNS:
        if key in row:
            row[key] = to_bool(row[key])
    return row


# ── 启动期小工具 ──

_console_guard_done = False


def ensure_utf8_console() -> None:
    """把 stdout/stderr 切到 UTF-8（幂等，可重复调用）。

    为什么在数据库层做这件事：本模块被所有入口（`app.main`、`mcp_server`、
    `python -m app.db_migration`、`scripts/*`）间接依赖，而迁移与建表日志是中文。
    Windows 控制台默认 cp1252，一条中文 `print` 就会抛 `UnicodeEncodeError` ——
    更糟的是它发生在 `init_db()` 已建表之后的收尾日志里，表现为「表建好了但进程崩了」
    这种自相矛盾的现场。容器内 `LANG=C.UTF-8` 不受影响，这里是兜住本地开发。
    """
    global _console_guard_done
    if _console_guard_done:
        return
    _console_guard_done = True
    import sys

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001 非 tty / 被重定向 / 旧解释器
            pass


# ── Cursor / Connection 外观 ──


class DBCursor:
    """psycopg2 cursor 与 sqlite3 cursor 的共同外观。

    只实现本仓库实际用到的成员：execute / executemany / fetchone / fetchall /
    rowcount / description / close，以及上下文管理器协议。
    """

    __slots__ = ("_cur", "_backend")

    def __init__(self, cur: Any, backend: str) -> None:
        self._cur = cur
        self._backend = backend

    # -- 语句下发 --
    def execute(self, sql: str, params: Any = None) -> "DBCursor":
        sql = translate(sql, self._backend)
        if params is None:
            self._cur.execute(sql)
        elif isinstance(params, (list, tuple)):
            self._cur.execute(sql, tuple(params))
        else:
            self._cur.execute(sql, params)
        return self

    def executemany(self, sql: str, seq_params: Any) -> "DBCursor":
        sql = translate(sql, self._backend)
        self._cur.executemany(sql, seq_params)
        return self

    # -- 取值 --
    def fetchone(self) -> Any:
        return self._cur.fetchone()

    def fetchall(self) -> Any:
        return self._cur.fetchall()

    @property
    def rowcount(self) -> int:
        return self._cur.rowcount

    @property
    def description(self) -> Any:
        return self._cur.description

    @property
    def lastrowid(self) -> Any:
        return getattr(self._cur, "lastrowid", None)

    def close(self) -> None:
        self._cur.close()

    # -- 上下文协议 --
    def __enter__(self) -> "DBCursor":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False


class DBConnection:
    """对 sqlite3.Connection 的包装，补齐 psycopg2 的 `autocommit` 语义。

    sqlite3 用 `isolation_level` 表达隔离级别；`isolation_level = None` 即
    autocommit（每条语句独立事务），这正好对应 `db_migration.init_db()` 里
    对 `conn.autocommit = True` 的用法（DDL 逐条独立提交，一条失败不拖垮其余）。
    """

    __slots__ = ("_conn", "_backend")

    def __init__(self, conn: Any, backend: str) -> None:
        self._conn = conn
        self._backend = backend

    @property
    def raw(self) -> Any:
        """底层连接（PG 后端下就是 psycopg2 连接）。"""
        return self._conn

    def cursor(self) -> DBCursor:
        if self._backend == BACKEND_SQLITE:
            return DBCursor(self._conn.cursor(), self._backend)
        return DBCursor(self._conn.cursor(), self._backend)

    def commit(self) -> None:
        self._conn.commit()

    def rollback(self) -> None:
        self._conn.rollback()

    def close(self) -> None:
        self._conn.close()

    # -- autocommit --
    @property
    def autocommit(self) -> bool:
        if self._backend == BACKEND_SQLITE:
            return self._conn.isolation_level is None
        return bool(self._conn.autocommit)

    @autocommit.setter
    def autocommit(self, value: bool) -> None:
        if self._backend == BACKEND_SQLITE:
            # None = autocommit；"" = 默认（隐式开事务）
            self._conn.isolation_level = None if value else ""
        else:
            self._conn.autocommit = bool(value)


# ── SQLite 连接初始化 ──

#: SQLite 需要「每连接」设置的 PRAGMA（WAL 是持久属性，其余是连接属性）
def sqlite_pragmas(busy_timeout_ms: int, synchronous: str) -> list[str]:
    return [
        "PRAGMA journal_mode=WAL",       # 多读一写；跨进程可见
        "PRAGMA foreign_keys=ON",        # 默认关闭，不显式开启则 ON DELETE CASCADE 失效
        f"PRAGMA synchronous={synchronous}",
        f"PRAGMA busy_timeout={int(busy_timeout_ms)}",
    ]


def configure_sqlite_connection(conn: sqlite3.Connection, busy_timeout_ms: int, synchronous: str) -> None:
    """对每个新建的 SQLite 连接执行 PRAGMA。

    注意 `foreign_keys` 是每连接属性（不是持久属性）—— 池里每个连接都必须执行，
    漏掉任何一个，那条连接上的删除就不会级联。
    """
    for pragma in sqlite_pragmas(busy_timeout_ms, synchronous):
        conn.execute(pragma)
