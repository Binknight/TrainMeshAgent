# 计划：PostgreSQL → SQLite 迁移（双后端，SQLite 为默认）

> 分支：`feature/sqlite-migration`（自 `feature-dp` / `c31b3b7` 切出）
> 状态：**已实施完成**。第 1~10 节是开工前定下的设计与验收清单（含风险登记与决策记录，
> 保持原文以便对照判断"是否按计划做了"）；**第 11 节记录实施结果与偏离项** —— 请以它为准。
> 与 `docs/数据库内嵌化改造说明.md` 的关系：那篇记录了上次「保留 PostgreSQL」的结论；
> 本篇是对同一问题的**重新评估与推翻**，第 4 节逐条回应了上次提出的 6 个技术障碍。

---

## 1. 目标（Objective）

把持久化层从 PostgreSQL 迁移到 SQLite，**SQLite 成为默认后端**，同时**保留
`DATABASE_URL` 指向外部 PostgreSQL 的逃生门**（同一份代码/镜像切换，无需重建）。

预期收益：

1. 容器内不再需要 `postgres` server 进程、`initdb`、`pg_ctl`、Unix socket 与 `pg_isready` 探测；
   基础镜像可去掉 `postgresql-14`（约 -100~150MB）。
2. 启动链由「目录自检 → 版本守卫 → initdb → 起 server → 就绪探测 → 建库」缩为「目录可写 → 打开文件 → 建表」。
3. 消除两个强约束：**PGDATA 与 PG 主版本强绑定**、**PGDATA 不能被 NFS/跨节点复用**。
4. 数据库退化为单文件，备份 = 拷文件（配合 WAL checkpoint），排障成本显著下降。

明确**不在本次范围**内（见第 10 节）：数据搬运、`mcp_server/`（任务状态落盘在 workspace，不走本 DB）。

---

## 2. 现状评估（基于代码实测）

### 2.1 SQL 影响面很小，且边界干净

| 位置 | 内容 | 结论 |
|------|------|------|
| `app/dao/__init__.py` | 564 行、**26 个 `get_db()` 调用点**、全部裸 SQL | 唯一的需要改写的 SQL 集中地 |
| `app/db_migration.py` | `SCHEMA_SQL` 43 条（7 建表 + 6 建索引 + 30 加列）+ 硬校验 | 需要按方言生成 DDL |
| `app/routes/*`、`app/services/*`、`app/agent/*` | 搜索确认**无裸 SQL** | **零改动** |
| `app/db.py` | 连接池 + `get_db()` | 内部改写，**对外契约不变** |

即：像上次改造一样，`get_db()` 的 commit/rollback/归还契约保持不变，路由层与 Agent 层就不受影响。

### 2.2 并发形态（决定 SQLite 可行性）

- `app/routes/session.py:393`、`app/routes/simulation.py:81` 用 `ThreadPoolExecutor`
  并发拉多个卡的结果，会有多个线程同时经过 DAO。
- 写操作**短小**（单条 upsert / delete），无长事务；SSE（`flask-sock`）与轮询不持有连接。
- 结论：SQLite 在 **WAL + `busy_timeout` + 短事务**下可承受；但必须验证，见第 7 节 M2。

---

## 3. 上次结论的技术障碍逐条回应

`docs/数据库内嵌化改造说明.md` §2 列了 6 条「换 SQLite 代价被严重低估」的理由。
本次逐条给出解法，其中 3 条已无实质代价：

| # | 上次的障碍 | 本次解法 | 残余代价 |
|---|-----------|---------|---------|
| 1 | `UUID DEFAULT gen_random_uuid()` 主键 | 改为 **Python 侧 `uuid4()` 生成**并在 INSERT 时显式传值（7 张表） | 中：建表去默认值 + DAO 插入点多一个参数 |
| 2 | `JSONB` 列 | 退化为 `TEXT`；DAO 侧统一 `_json_out()` / `_json_in()` 处理 | 中：现有「是 str 才 parse」的分支要反过来，需一次对齐全 |
| 3 | `ALTER TABLE ... ADD COLUMN IF NOT EXISTS`（30 条） | 建表 DDL 里直接用**最终列集合**（新库无历史包袱）；加列逻辑由 `PRAGMA table_info()` 探测后动态 `ADD COLUMN` | 低 |
| 4 | `CREATE INDEX IF NOT EXISTS`（6 条） | **SQLite 支持** `CREATE INDEX IF NOT EXISTS` | **无** |
| 5 | `UUID REFERENCES ... ON DELETE CASCADE` | 显式 `PRAGMA foreign_keys=ON`（每个连接）+ FK 列类型改 `TEXT` | 低 |
| 6 | 行级 MVCC + 多连接写 | `journal_mode=WAL` + `busy_timeout=5000` + 短事务 + 连接池 | 中：WAL 是**库级单写者**，需实测并发 |

**结论修正**：上次判断「PG 特性依赖很深」的依据主要是**DDL 30 条加列**与 **JSONB/UUID 的 DAO 适配**，
这两项的解锁方式是「**改写一次 DDL 生成逻辑 + 一次对齐全 DAO 取值**」，属于**一次性成本**，
不是持续维护成本。**保留 PG 反倒是持续的部署复杂度**。因此本次改成双后端。

---

## 4. 技术设计

### 4.1 后端选择规则（自动侦测，零显式开关）

```
DATABASE_URL 以 postgres:// 或 postgresql:// 开头  →  PG 后端（逃生门，行为与现在完全一致）
其余（空 / sqlite://... / 未设置）                  →  SQLite 后端（默认）
```

这条规则保证两件事：

- 集群里已有的 `charts/templates/secret.yaml` 注入 `DATABASE_URL` 的部署**行为不变**；
- 不注入时自动落到 SQLite，无需新增必填环境变量，也无需 `DB_BACKEND` 这类双开关
  （双开关会产生「URL 与开关不一致」的第四种状态，故意不引入）。

新增/调整的配置项（`app/config.py`）：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `DATABASE_URL` | `""`（空） | 留空 = 用 SQLite。默认值**从 TCP PG 串改为空串**，是本次唯一的行为变更 |
| `SQLITE_PATH` | `/home/aicm/db/train_mesh_agent.db` | SQLite 数据文件；父目录即挂载点 `/home/aicm/db` |
| `SQLITE_BUSY_TIMEOUT_MS` | `5000` | `busy_timeout` |
| `PGDATA` / `PG_SOCKET_DIR` | 保留读取 | 仅 PG 后端与版本守卫使用；SQLite 路径忽略 |

### 4.2 SQL 方言抽象：把 `%s` 留在原地

不重写 26 处 SQL 的参数风格（那是纯噪音改动、且极易改错）。在 `app/dbapi.py` 提供薄适配层：

```python
class _Cursor:
    """psycopg2 cursor 与 sqlite3 cursor 的共同外观。"""
    def execute(self, sql, params=None):   # sqlite 后端下把 %s 逐字替换为 ?
    def executemany(self, sql, seq):       # 同上
    def fetchone/fetchall/description/rowcount/close
```

语义差异集中在一处处理：

| 差异 | 处理位置 |
|------|---------|
| 占位符 `%s` → `?` | `_Cursor.execute` 内的逐字替换（SQL 文案零改动） |
| `NOW()` → `CURRENT_TIMESTAMP` | 同上（逐字替换） |
| `EXCLUDED.x` → `excluded.x` | 同上（SQLite 关键字 `excluded` 可用，小写化即合法） |
| `ILIKE` | SQLite 支持 `LIKE` 默认对 ASCII 大小写不敏感；用 `LIKE` 替换 `ILIKE` |
| `name_key = ANY(%s)` | 改 `name_key IN (?, ?, ...)`，候选集仍来自 Python |
| 返回值类型 | 见 4.3 |

> 逐字替换只在 **SQL 字面量之外**安全（本仓库 SQL 无 `%` 出现在字面量里的情况，
> 但实现时仍要用「不处于引号内」的状态机，不能裸 `str.replace`）。

### 4.3 类型映射与返回值契约（最容易出错的地方）

**必须保持对外 JSON 形态不变**，否则前端静默错乱：

| 列 | PG 读回 | SQLite 读回 | 处理 |
|----|--------|------------|------|
| `created_at` / `updated_at` / `timestamp` | `datetime` → `.isoformat()` | `TEXT` 字符串 | 连接上注册 `detect_types=PARSE_DECLTYPES` + 为 `TIMESTAMP` 注册 converter，保证 DAO 里 `r[2].isoformat()` 不炸 |
| `is_simulated` / `is_equivalent` / `visual_json_output` / `has_shared_expert` 等 BOOLEAN | `bool` | `0/1` int | 在 DAO 出口统一 `bool()` 转换（`jsonify` 会把 1/0 当 int 下发，前端 `if (x)` 能跑但 `=== true` 会坏） |
| `cards` / `level0_config` / `level1_config` / `formula_lines` / `details` | `str`（psycopg2 默认不解析 JSONB） | `str` | 现有「`isinstance(x, str)` 才 parse」的分支**在 SQLite 下天然成立**，反而是 PG 若改用 `json.loads` 注册会更省事。此处反向对齐一次，两侧行为统一 |
| `id`(UUID) | `UUID` 对象 | `TEXT` | PG 后端下 psycopg2 返回 `UUID`，`jsonify` 会失败或输出非预期；**两侧都统一 `str()` 出口** |

### 4.4 连接池与并发

`app/db.py` 保留 `get_pool()` / `get_db()` 名称与语义，内部按后端分流：

- **SQLite 池**：`queue.Queue` 存 N 个 `sqlite3.Connection`（`check_same_thread=False`，
  同一连接同一时刻只被一个线程借用 → 池本身就是串行化机制），取连接时 `PRAGMA busy_timeout`。
- **建池时**（每连接）：`journal_mode=WAL`、`foreign_keys=ON`、`synchronous=NORMAL`、`busy_timeout`。
- `WAL` 是持久属性，`foreign_keys` 是**每连接**属性 —— 后者必须在 `_make_pool()` 里对每个连接执行，漏了就丢 `ON DELETE CASCADE`。
- `get_db()` 的 commit/rollback/归还语义**逐字不变**。

### 4.5 迁移脚本（`app/db_migration.py`）

- `SCHEMA_SQL` 拆为 **方言化生成**：新增 `_schema_statements(backend)` 返回语句列表；
  PG 分支保留现有 43 条（含 `gen_random_uuid()` / `JSONB` / 30 条加列）；
  SQLite 分支输出**最终形态**的 7 张 `CREATE TABLE` + 6 条 `CREATE INDEX IF NOT EXISTS`。
- SQLite 分支的幂等策略与 PG 分支不同：
  - 表/索引：`CREATE ... IF NOT EXISTS` 原生支持；
  - 加列：`PRAGMA table_info(<t>)` 读现有列 → 只为缺失列执行 `ALTER TABLE ADD COLUMN`；
  - `BENIGN_DDL_ERRORS` 的 PG 错误码（`42P07`/`42701`…）在 SQLite 下换成 `sqlite3.OperationalError`
    的文案匹配（`already exists` / `duplicate column name`），**「不静默吞异常」的原则保持不变**。
- 硬校验 `REQUIRED_TABLES` / `REQUIRED_COLUMNS` **不变**（这是防「历史列表标题退化成『新建任务』」的那道闸）；
  仅把取数来源从 `information_schema` 换成 SQLite 的 `sqlite_master` + `PRAGMA table_info`。
- **`gen_random_uuid` / `JSONB` 的全仓库不变量**（`scripts/verify_static.py` 断言）在 PG 分支上仍成立。

### 4.6 源码级单一事实来源（避免两套 DDL 漂移）

PG 分支的 `SCHEMA_SQL` 与 SQLite 分支的 DDL 是**两份手写 DDL**，天然会漂移。
因此新增 `scripts/verify_static.py` 的断言：**两侧产出的表名集合与列名集合必须逐一致**
（用两份 DDL 建到内存 SQLite / 解析 PG DDL 文本，比较 `table → columns` 映射）。
这是本次改造最重要的防回归闸门，见第 6 节 D1。

---

## 5. 阶段划分与文件级改动清单

> 本节保留开工时的计划原文。**各阶段均已完成**，逐条完成情况与偏离项见第 11 节。

每阶段**独立可验收**，不把全仓库改到一半再验证。

### Phase 0 — 抽象层（可独立验收：PG 行为零变化）

| 文件 | 改动 |
|------|------|
| `app/dbapi.py` | **新增**：占位符/`NOW()`/`EXCLUDED` 转换、`_Cursor` 外观、`_json_in/_json_out`、`bool` 归一 |
| `app/db.py` | 内部改走 `dbapi`；暴露 `backend()`；PG 分支行为逐字不变 |
| `scripts/verify_static.py` | 补「占位符转换不触碰字符串字面量」的单测 |

**验收**：existing PG 环境（本机 PG 18）跑通全部 DAO 路径；`python scripts/pg_real_check.py` 仍全绿。

### Phase 1 — DDL 双分支（验收：`sqlite_real_check.py` 正负向）

| 文件 | 改动 |
|------|------|
| `app/db_migration.py` | `_schema_statements(backend)`；SQLite 建表/索引/加列/校验分支 |
| `scripts/sqlite_real_check.py` | **新增**：对齐 `pg_real_check.py` 的 8 项（建表、幂等重放、硬校验正负向、UUID 主键、JSON 读写、`UNIQUE(session_id, role)`、FK cascade、WAL 生效） |
| `scripts/verify_static.py` | 增加 **PG/SQLite 两套 DDL 的表列一致性断言（D1）** |

**验收**：`sqlite_real_check.py` exit 0；`verify_static.py` exit 0。

### Phase 2 — 完整 DAO 路径（验收：端到端脚本 + 现有基线不退化）

| 文件 | 改动 |
|------|------|
| `app/dao/__init__.py` | 逐函数适配：UUID 显式生成、BOOLEAN 归一、`ILIKE`/`ANY`、`NOW()`、`RETURNING id` 的 SQLite 兼容写法（`cursor.lastrowid` 或先 `SELECT`） |
| `app/config.py` | 4.1 的配置项调整 |
| `scripts/sqlite_real_check.py` | 扩到**逐个 DAO 函数**的读写往返断言 |

**关注点（易踩坑）**：

- `save_simulation_result` 依赖 `RETURNING id`（PG 独有旧版语法；SQLite 3.35+ 支持 `RETURNING`，
  但**需要确认目标镜像的 Python 版本自带 sqlite 版本**，必要时降级为 `lastrowid`/再查一次）。
- `save_comparison_report` 用 `ON CONFLICT DO NOTHING` **无冲突目标**（PG 允许），
  而该表**没有** `UNIQUE(session_id)` 约束 → SQLite 侧必须给一个冲突目标，否则语义不同。
  → 决策：在 SQLite 建表中为 `comparison_reports` 补 `UNIQUE(session_id)`，并同步 PG 分支（D2）。
- `delete_session` 依赖 `ON DELETE CASCADE` → 验证 `PRAGMA foreign_keys=ON` 确实生效。

### Phase 3 — 容器与部署（验收：静态契约校验 + 手写渲染）

| 文件 | 改动 |
|------|------|
| `Dockerfile` | 预建 `/home/aicm/db`（去掉 `data`/`run` 子目录语义）；ENV 去 `PGDATA`/`PG_SOCKET_DIR`/`PG_BIN`，加 `SQLITE_PATH`；`DATABASE_URL` 默认值改空；HEALTHCHECK 去掉 `pg_isready`（改打 `/api/health`） |
| `docker/base/Dockerfile` | 去掉 `postgresql-14` 安装与版本自检；**保留 `libpq5`**（逃生门用 `psycopg2` 需要） |
| `docker/entrypoint.sh` | 删掉「版本守卫 / initdb / 起 postgres / pg_isready 就绪探测 / 建库 / 停机 pg_ctl stop」六段；保留 workspace 与 DB 目录自检 |
| `docker/check_db.py` | 语义改写：PGDATA 检查 → SQLite **父目录**是挂载点且可写（去掉 0700 / 属主与 PG 相关的部分） |
| `docker/wait_for_db.py` | SQLite 后端下退化为「文件可打开 + 能建表」，或直接删除并把这步并进 `db_migration` |
| `requirements.txt` | `psycopg2-binary` 标注为「仅 PG 逃生门需要，可选」 |
| `charts/values.yaml`、`charts/templates/deployment.yaml`、`configMap.yaml`、`secret.yaml` | `sim-db` 卷语义由 PGDATA 目录改为 SQLite 单文件所在目录；`replicas=1` 注释改为「SQLite 单写者」；`secrets.databaseUrl` 语义不变（逃生门） |
| `scripts/verify_consistency.py` | PGDATA/socket 三处对齐断言 → 改为 `SQLITE_PATH` / `db` 挂载点 / 镜像 ENV 三处对齐 |
| `scripts/render_chart.py` | 形态 A/B 断言保留，期望值同步 |

### Phase 4 — 文档与收尾

| 文件 | 改动 |
|------|------|
| `AGENTS.md` | §2 架构图、§3 常用命令、§4.4 容器约束、§5 陷阱表：PG 相关约束改为 SQLite 约束 |
| `README.md` | 架构图 `psycopg2 → PostgreSQL` 改为 `sqlite3 / psycopg2`；环境变量表 |
| `docs/使用指南.md` | 依赖表、`DATABASE_URL`/`SQLITE_PATH` 配置表、持久化契约、排障表、目录树 |
| `docs/数据库内嵌化改造说明.md` | 顶部加「已被 `docs/PG迁移SQLite改造计划.md` 取代」的指向说明，**不删除**（历史决策记录） |
| `.env.example` | 两个后端形态 |
| 全仓库 | 搜索悬空引用（上次删 `CLAUDE.md` 留下目录树悬空条目的教训） |

---

## 6. 验证策略

沿用仓库既有做法（独立脚本，非 pytest；本机无 docker/helm 时用静态校验替代），并明确每条断言属于哪类：

### 6.1 静态 / 跨文件契约

- **D1（新，最重要）**：PG 与 SQLite 两套 DDL 的 `表→列` 映射逐一致；`REQUIRED_TABLES`/`REQUIRED_COLUMNS` 均能在两套中解析出。
- **D2（新）**：`comparison_reports` 的冲突目标在两套 DDL 中都存在（PG 与 SQLite 都要有 `UNIQUE(session_id)`）。
- **D3（改）**：`verify_consistency.py` 的三处路径契约对齐（`SQLITE_PATH` / chart 挂载点 / config 默认值）。
- **D4（保留）**：`verify_static.py` 对 `init_db` 唯一性、`REQUIRED_*` 存在性、无破坏性 DDL 的断言。
- **D5（保留）**：换行符契约（`*.sh`/`Dockerfile` 必须 LF）。

### 6.2 真实引擎（不经 mock）

- `scripts/sqlite_real_check.py`（新）：建表 → 幂等重放 → 硬校验正负向 → 7 张表读写往返 →
  FK cascade 生效 → `UNIQUE` 冲突正确 → WAL 已生效（`PRAGMA journal_mode` 返回 `wal`）。
- `scripts/backend_detect_check.py`（新，实施时补）：`DATABASE_URL` 前缀判定的边界矩阵。
  起因是"写错前缀 → 静默回落 SQLite"这条规则没有异常可抓，只能靠穷举边界情形钉住
  （留空 / 大写 / 首尾空白 / 单斜杠 / `postgres@Data` 简写 / 别的协议 / 误写文件路径）。
- `scripts/pg_real_check.py`（保留）：证明逃生门未破。
  > 本机无 PG 实例时该脚本会 SKIP，需在验收环境补跑，不能以 SKIP 当作通过。

### 6.3 现有测试基线（**逐个运行**，非 pytest）

```bash
python tests/test_rank_layout.py / test_pp_compare_grouping.py / test_hbm_equiv.py
python tests/test_mesh_profiler.py / test_mcp_server_rank_layout.py / test_moe_mcp_contract.py
python tests/test_moe_agent_plumbing.py / test_check_workspace.py
# test_estimate_equivalence.py 是既有失败（HBM 16.9% / DP 缩放 52.46%），
# 与本次改造无关，不得为了让其变绿而改估算公式。
```

### 6.4 并发（M2，必须实测，不能推断）

脚本：N=8 线程 × 每线程 200 次「`create_session` + `update_session_step` + `save_message`」写入，
断言 0 次 `database is locked`、总数与预期一致。**这是 SQLite 方案唯一可能失效的地方**，
若不达标则回退备选（见第 9 节）。

> **实测结果（已完成）**：8 线程 × 200 轮，**0 错误、无 `database is locked`**，总数正确
> （8 sessions / 1600 messages）。**M2 通过**，无需启用第 9 节的备选方案。

---

## 7. 风险登记

| # | 风险 | 等级 | 缓解 | 实施结果 |
|---|------|------|------|---------|
| M1 | 两套 DDL 漂移（PG 分支忘记同步） | HIGH | D1/D2 断言进 CI（`verify_static.py`） | ✅ 断言已落地并全绿（108 列逐一致） |
| M2 | 并发写 `database is locked` | HIGH | WAL + `busy_timeout` + 短事务；**必须实测**（6.4） | ✅ 实测通过（8×200，0 错误） |
| M3 | 时间戳类型读回形态变化 → `.isoformat()` 崩 | HIGH | `detect_types` + converter；DAO 出口统一；往返断言覆盖 | ✅ 已消除，但**用的不是 converter**（改为 DAO 出口归一，见 11.2） |
| M4 | BOOLEAN 变 `0/1` → 前端 `=== true` 判断坏 | MEDIUM | DAO 出口统一 `bool()`；前端若有必要单独排查 | ✅ `normalize_row()` 覆盖（这层是**必须**的：`routes/session.py` 会把 DAO 输出喂给严格类型的 pydantic） |
| M5 | `RETURNING id` 依赖 sqlite 版本 | MEDIUM | 先探测 `sqlite3.sqlite_version`；不达标降级 `lastrowid` | ✅ 风险消失：最终不使用 `RETURNING`（见 11.2） |
| M6 | 现有外部 PG 部署不兼容（`DATABASE_URL` 默认值从 TCP 串变空） | MEDIUM | 自动侦测规则（4.1）保证「显式给了 PG 串」仍走 PG；仅「依赖旧默认值」的部署受影响，在升级说明中单列 | ⚠️ 行为如预期，但**镜像内 PG 服务端已删除**，因此"依赖旧默认值"的部署必须显式给串 |
| M7 | WAL 的 `-wal`/`-shm` 文件落在挂载目录内被误删/不一致 | LOW | 备份说明里写清要先 checkpoint；目录整体备份 | ✅ 停机自动 `wal_checkpoint(TRUNCATE)`（`docker/checkpoint_db.py`）+ 文档说明 |
| M8 | `synchronous=NORMAL` 极端掉电丢最后若干事务 | LOW | 明确记录为可接受折衷（会话历史非金融数据）；提供 `SQLITE_SYNCHRONOUS` 覆盖 | ✅ 已记录，`SQLITE_SYNCHRONOUS` 可覆盖 |

---

## 8. 决策记录（已定）

| 决策 | 取值 | 理由 |
|------|------|------|
| 后端形态 | **双后端，SQLite 默认** | 用户确认；保留逃生门，回退成本最低 |
| 存量数据搬运 | **不做** | 用户确认；`model_catalog` 由内置清单重新 seed（`db_migration.init_db` 已有幂等 seed） |
| 分支 | `feature/sqlite-migration`（自 `c31b3b7`） | 用户确认 |
| UUID 主键 | Python 侧生成，列类型 `TEXT` | 避免引入扩展，避免改 DAO 以外的地方 |
| 参数风格 | 保留 `%s`，在适配层转换 | 26 处 SQL 零改动，降低改错风险 |
| 显式开关 | **不引入** `DB_BACKEND` | 由 `DATABASE_URL` 前缀侦测，避免第四种不一致状态 |

## 9. 回退路径

1. **运行时回退（首选）**：注入 `DATABASE_URL=postgresql://...` 即回到 PG 后端，无需重建镜像。
   前提：Phase 3 保留镜像内 `postgresql-14` **或**（若已删除）改为「指向外部 PG」这一半仍可用。
   > **实施后的实际状态**：走的是后者 —— 镜像内 **已删除** `postgresql-14`，因此运行时回退
   > 只能指向**外部** PG。决策依据：保留内嵌 PG 服务端与"减掉 ~100~150MB 基础镜像"这一
   > 主要收益直接冲突，而逃生门的价值在于"不必重建镜像即可换库"，指向外部 PG 已满足。
2. **分支回退**：`git checkout feature-dp`，本分支所有改动都在独立提交里。
3. **SQLite 并发不达标时（M2）的备选**：不引入新引擎，退回「内嵌 PG」现状（本方案作废），
   或改「SQLite 单写线程 + 写队列」的串行化方案（仅当并发实测失败才评估）。

## 10. 非目标（明确不做）

- 不改 `mcp_server/`（仿真任务状态落盘在 workspace 的 `task_meta.json`，不走本 DB）。
- 不改 rank 布局、估算公式、等效性判断（与本次无关）。
- 不引入 ORM / 迁移框架（Alembic 等）—— 沿用现有「幂等 DDL 重放 + 收尾硬校验」模式。
- 不处理外部 PG 存量数据导入（用户确认不需要）。

---

## 11. 实施结果与偏离项

### 11.1 验收结论

| 项 | 结果 |
|----|------|
| Phase 0 抽象层（`app/dbapi.py`） | 完成。`translate()` 只改写字符串字面量之外的 `%s` / `NOW()` / `EXCLUDED` |
| Phase 1 双分支 DDL | 完成。`SCHEMA_SQL`（PG 43 条，原样保留） + `SCHEMA_SQLITE_SQL`（13 条：7 建表 + 6 建索引） |
| Phase 2 DAO 全路径 | 完成。`scripts/sqlite_real_check.py` 覆盖全 DAO 往返 |
| Phase 3 容器与部署 | 完成。PG 服务端、`wait_for_db.py` 已删除（非"退化保留"） |
| Phase 4 文档 | 完成 |
| **M2 并发实测** | **通过**：8 线程 × 200 轮（`create_session` + `update_session_step` + `save_message`）0 错误、无 `database is locked`，总数正确（8 sessions / 1600 messages） |
| D1 / D2 静态断言 | 通过。两套 DDL 的 7 张表、**108 个列名**逐一致 |
| 后端侦测边界矩阵 | 通过。`scripts/backend_detect_check.py`：10 条前缀判定全部符合预期 |
| 现有测试基线 | 8/8 原本通过的仍通过；`test_estimate_equivalence.py` 保持既有失败（未触碰估算公式） |
| `pg_real_check.py` | **未在真实 PG 上跑过**（本机无 PG，脚本 SKIP）。逃生门尚未被实机验证，见 11.4 |

### 11.2 与计划不一致的地方（实施时改的）

计划是开工前的判断，实施中有几处按实际代码形态调整，**以本节为准**：

| 计划 | 实际做法 | 原因 |
|------|---------|------|
| 4.3：`detect_types=PARSE_DECLTYPES` + 注册 TIMESTAMP converter，保证 `r[2].isoformat()` 不炸 | **不注册 converter**。改为 DAO 出口统一走 `normalize_row()` / `to_bool()`（`app/dbapi.py`） | converter 是进程级全局状态，会污染同进程其它 sqlite 连接，且 `datetime` 在 JSON 序列化后仍需二次处理。集中在 DAO 出口归一更可控，也让两套后端出口形态真正一致 |
| 4.1 配置：新增 `DB_BACKEND` 之外的开关倾向未定 | 确定为**不引入任何开关**，纯粹 `DATABASE_URL` 前缀侦测 | 见第 8 节决策 |
| 4.4：SQLite 池用 `queue.Queue` | 用 `queue.LifoQueue` + 一个 `set` 记录全部连接 | LIFO 让最近归还的连接优先被复用（缓存更热）；`set` 是为了在 `close_all()` 时能可靠关掉在借的连接 |
| 4.4：取连接时设 `PRAGMA busy_timeout` | 建池时**每连接**一次性设好所有 pragma（含 `busy_timeout`、`foreign_keys`） | 每次借出重设 pragma 有额外往返；pragma 是连接属性，建好即长期有效 |
| Phase 2：`save_simulation_result` 考虑 `cursor.lastrowid` 或 `RETURNING` | 改为**先 `SELECT` 已存在的 id，再 upsert** | 两者都要求"冲突时主键不变"这一性质，而 `lastrowid`/`RETURNING` 在 `ON CONFLICT DO UPDATE` 路径上都不保证返回**原有** id。先查才能让 `comparison_reports` 的外键始终指向正确行 |
| Phase 3：`RETURNING id` 依赖 sqlite 版本，先探测再降级 | 最终**完全不用 `RETURNING`**（因上一条） | 顺带消除了"目标镜像 sqlite 版本"这一变量 |
| Phase 3：`docker/wait_for_db.py`「退化保留 或 删除」 | **删除**，并把这步并进 `db_migration` | SQLite 没有"等 server 就绪"这件事；留着文件只会让启动链更难读 |
| Phase 3：`HEALTHCHECK` 改打 `/api/health` | 保留 HEALTHCHECK，去掉 `pg_isready` 探测 | —— |

### 11.3 实施中新发现、计划里没预见的坑

1. **`psycopg2` 缺失时 PG 分支的崩溃点很晚**：`import psycopg2` 在模块顶层，SQLite 路径已验证
   完全不依赖它（把 import 屏蔽后 DAO 仍全绿）。但反过来，PG 逃生门在**基础镜像里
   `autoremove` 之后**是否仍能 import，只能在构建期断言（已在 `docker/base/Dockerfile` 加了自检）。
2. **Windows 控制台 cp1252 会把"校验失败"伪装成编码崩溃**：`init_db()` 的中文日志在
   cp1252 控制台下抛 `UnicodeEncodeError`，而且发生在**建表成功之后** —— 极易被误判成迁移本身出错。
   已在 `app/dbapi.ensure_utf8_console()`（`app/db.py` import 时调用）与
   `scripts/_console.py` 两处兜底。容器内 `LANG=C.UTF-8` 本来不受影响。
3. **`sqlite3` 的 `Connection.isolation_level` 不能直接表达 psycopg2 的 `autocommit`**：
   `DBConnection` 需要显式映射（`autocommit=True` ↔ `isolation_level=None`），否则
   DAO 里那句 `conn.autocommit = True` 会静默失效，让本该自动提交的 DDL 卡在隐式事务里。
4. **`verify_static.py` 原来只解析 `SCHEMA_SQL` 一个常量**：新增 SQLite 分支后必须让它同时解析
   两套，并做 `table → columns` 比对。解析 `ALTER TABLE ... ADD COLUMN` 也是必需的 ——
   PG 分支是「最小建表 + 30 条加列」，只解析 `CREATE TABLE` 会**反向**漏掉后加列，
   把真实差异误报成一致。

### 11.4 未完成 / 需在验收环境补做

| 项 | 说明 |
|----|------|
| `scripts/pg_real_check.py` 真机运行 | 无本地 PG 时脚本 SKIP。**SKIP 不等于通过**，逃生门需在具备 PG 的环境补跑一次 |
| 镜像构建与容器启动 | 本机无 docker，Dockerfile / base 镜像 / entrypoint 的改动**只经过静态校验**，未实际构建运行 |
| helm 渲染 | 本机无 helm，`charts/` 改动只经过 `scripts/render_chart.py` 的近似渲染校验 |
| 存量数据搬运 | 按决策不做（见第 8 节） |
