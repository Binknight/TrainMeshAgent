# AGENTS.md — 项目说明（供 AI 编码助手使用）

面向在本仓库工作的 AI 助手。目标：**让你在动手前就掌握架构约束、硬性约定与已知陷阱**，
避免改错地方、踩到只有运行时才暴露的坑。

---

## 1. 项目是什么

**TrainMeshAgent**：AI 训练组网仿真测试 Agent。测试人员通过 Web 界面用自然语言描述组网
（设备类型 A2/A3/A5、DP/TP/PP 并行度），Agent 生成结构化组网图 → 下发仿真 → 对比原始组网与
等效组网的**计算强度 / 内存占用 / 通信流量**，验证"模型等效"是否成立。

需求原文见 `docs/需求规格.md`（按条目号被代码引用，勿删）。使用与 API 细节见 `docs/使用指南.md`。

---

## 2. 三个组件（理解这一点才能不迷路）

```
浏览器 ──> Flask app :5000 ──HTTP JSON-RPC──> MCP 仿真 Server :9000 ──拉起子进程──> aicm/run.py
                 │
                 └──psycopg2──> PostgreSQL :5432
```

| 组件 | 位置 | 职责 | 关键点 |
|------|------|------|--------|
| **Flask 应用** | `app/` | 前端 + Agent 编排 + REST/SSE/WebSocket | 入口 `app/main.py`；启动即执行建表迁移 |
| **MCP 仿真 Server** | `mcp_server/` | 暴露 9 个仿真工具，管理任务生命周期 | **独立部署单元**，有自己的 `requirements.txt`；`app/mcp/client.py` 只通过 HTTP 调它 |
| **仿真工具** | `aicm/`（需自行放入） | 真正的仿真程序 `run.py` | 不在版本库中；缺失时 MCP Server 仍能启动，但下发任务报 `sim_tool_home not found` |

**`app/` 与 `mcp_server/` 的依赖必须保持独立。** 前者是 Agent 侧，后者部署在仿真系统一侧，
两者只通过 HTTP 通信 —— 不要在 `app/` 里 import `mcp_server/`，反之亦然。

---

## 3. 常用命令

```bash
# 依赖
pip install -r requirements.txt              # Flask 侧
pip install -r mcp_server/requirements.txt   # MCP Server 侧

# 启动（三个服务，各开一个终端；必须在仓库根执行）
python -m app.main          # Flask       → http://localhost:5000
python -m mcp_server        # MCP Server  → http://localhost:9000
# PostgreSQL 需另行运行，且业务库 train_mesh_agent 须已存在（首次 createdb 一下）

# 校验
python scripts/verify_static.py        # db_migration.py 结构不变量 + 全仓库语法
python scripts/verify_consistency.py   # 镜像 ENV / config.py / charts 三处契约对齐
python scripts/render_chart.py         # Helm 模板渲染校验（本机无 helm 时的替代）
ADMIN_DSN=postgresql://postgres:<pw>@127.0.0.1:5432/postgres \
  python scripts/pg_real_check.py      # 连真实 PG 跑建表/幂等/硬校验正负向
```

部署与镜像相关细节见 `docs/数据库内嵌化改造说明.md`、`README.md`。

---

## 4. 硬性约定（改代码前务必确认）

### 4.1 `app/rank_layout.py` 是 rank 布局的单一事实来源

仿真系统（MCP Server）的维度顺序是 **TP → DP → PP**（变化最快 → 最慢）：

```
global_rank = pp_rank * (tp * dp) + dp_rank * tp + tp_rank
```

**所有消费方都必须用它**（`app/agent/orchestrator.py`、`app/routes/session.py`、
`mcp_server/` 与测试均已对齐）。历史上这里出过分组错配导致等效性误判，因此：

- 不要在别处重写 rank 换算公式
- 改动通信组划分后，务必跑 `python tests/test_rank_layout.py` 与
  `python tests/test_pp_compare_grouping.py` 验证契约未破

### 4.2 数据库迁移必须"要么全成，要么拒绝启动"

`app/db_migration.py` 的 `init_db()` 在应用启动时执行（`app/main.py:31`），
也可独立运行（`python -m app.db_migration`，文件末尾有 `__main__` 入口）：

- **不要恢复"静默吞异常"的写法**。当前实现把 DDL 错误分为「可忽略」（`BENIGN_DDL_ERRORS`，
  如建表重复）与「致命」，并在收尾用 `REQUIRED_TABLES` / `REQUIRED_COLUMNS` 做硬校验，
  缺失即抛 `RuntimeError`。
- **不要加 `DROP COLUMN IF EXISTS`**。启动路径上放破坏性 DDL，每次重启都会重放。
- schema 不完整会导致「历史列表标题显示『新建任务』」这类看似 UI 的问题，实际是缺表/缺列。

### 4.3 `app/db.py` 的连接池契约不可变

`get_db()` / `get_pool()`（`app/db.py`）的 **commit / rollback / 归还连接**语义是
`app/dao/` 等调用方的公共前提。可以改内部重试与错误提示，但不要改它们的对外行为。

调用点分布（共 26 处，改前请知悉影响面）：**23 处在 `app/dao/__init__.py`**，
其余在 `app/db.py` 自身与 `app/db_migration.py`。也就是说路由层是通过 DAO 间接使用的，
DAO 是真正的影响集中点。

### 4.4 容器相关约束

- **内嵌 PostgreSQL 14 由基础镜像提供**，`PGDATA=/home/aicm/db/data`，走 **Unix socket**
  （`host=/home/aicm/db/run`），不开 TCP。`docker/entrypoint.sh` 负责
  「目录自检 → 版本守卫 → 空目录 initdb → 启动 → 停机组」。
- **PGDATA 与 PG 主版本强绑定**。本机开发用 PG 18、容器用 PG 14 是正常组合，
  但**数据文件与 `pg_dump` 产物跨主版本不可直接复用**。
- **`replicas` 必须为 1**：PGDATA 是单写者。Pod 漂移到其他节点会看到"空"数据库
  （与 `/home/aicm/workspace` 同源的失效模式）。
- **禁止把 PGDATA 放 NFS**：PostgreSQL 依赖本地文件锁与 `fsync`。
- 存储方式支持双形态：镜像默认内嵌；集群 Secret 注入 `DATABASE_URL` 可切回外部 PG，
  **同一镜像无需重建**（回退逃生门，勿删）。

### 4.5 行末符与编码

`.gitattributes` 强制 `*.sh`、`Dockerfile`、`*.dockerignore`、`.gitattributes` 为 **LF**。
`build.sh` 是把工作区文件直接打进 tar 的，CRLF 会让容器内的 `trap` / 信号名带上 `\r` 而失败。
本仓库 `core.autocrlf=true`，因此**工作区看到 CRLF 是正常的，但要确认提交进仓库的是 LF**。

---

## 5. 已知陷阱（都是实际踩过的）

| 陷阱 | 说明 |
|------|------|
| **不要用 `python app/main.py`** | 实测报 `ModuleNotFoundError: No module named 'app'` —— Python 把**脚本所在目录**（`app/`）放进 `sys.path` 而非 CWD。必须用 `python -m app.main`（或 `PYTHONPATH=. python app/main.py`） |
| **`AICM_MCP_CONDA_ENV` 默认为空串** | 空值即回退到当前 Python 解释器（`mcp_server/services/conda_launcher.py` 对空值的处理），因此本机与容器都无须 conda。只有显式设了 `<环境名>` 而该环境不存在时，仿真子进程才会起不来 |
| **`aicm/` 缺失不阻止 MCP Server 启动** | 校验发生在 `simulation_runner.prepare_and_launch`，即真正下发任务时才报错。纯联调可用 `AICM_MCP_DRY_RUN=1`（只建任务不拉子进程） |
| **依赖名与 import 名不一致** | `python-dotenv`→`dotenv`、`pyyaml`→`yaml`、`psycopg2-binary`→`psycopg2`、`flask-cors`→`flask_cors`、`flask-sock`→`flask_sock`。做依赖审计时必须归一化，否则全部误报 |
| **`httpx` 是直接依赖** | `app/agent/orchestrator.py` 直接 import 并构造 OpenAI 客户端（承载 `OPENAI_SSL_VERIFY` / `EXTERNAL_PROXY`），已显式声明，不要当成 openai 的传递依赖而移除 |
| **不要引入 PGDG 第三方 apt 源** | 基础镜像设计前提是"内网源可用、外网不可用"，PG 只从 Ubuntu 官方源装 |

---

## 6. 测试现状（重要：基线并非全绿）

测试**不是 pytest/unittest**（无 `pytest.ini`/`conftest.py`/`pyproject.toml`，也没有
`def test_*`），而是自带 `__main__` 的独立脚本，**必须逐个运行**：

```bash
# 已验证通过（exit 0）
python tests/test_rank_layout.py              # PASS  rank 布局 TP-DP-PP 契约
python tests/test_pp_compare_grouping.py      # PASS  PP 段分组与仿真系统对齐
python tests/test_hbm_equiv.py                # PASS  HBM 等效性
python tests/test_mesh_profiler.py            # PASS  B 缩放
python tests/test_mcp_server_rank_layout.py   # PASS  MCP Server rank 分解契约
python tests/test_moe_mcp_contract.py          # PASS  MoE 脚本生成/参数校验/EP 数据返回契约
python tests/test_moe_agent_plumbing.py        # PASS  MoE 在 Agent 侧的透传与脚本解析契约
python tests/test_check_workspace.py          # PASS（Windows 上 SKIP mountinfo 检查）

# 当前基线为失败（exit 1）
python tests/test_estimate_equivalence.py

# 需先启动服务（真实 HTTP）
python tests/test_mcp_server_e2e.py           # 需 MCP Server :9000
```

**`tests/test_estimate_equivalence.py` 是既有失败，不是环境问题。** 退出码 1，失败项：

- `[FAIL] 每卡 HBM (±15%): 差异 16.9%`
- `[FAIL] DP 通信按 batch 缩放: 实际 52.46% vs 预期 25.00%`

这两项涉及估算公式本身。**若你的改动没有触及估算/等效逻辑，不要为本失败负责，也不要
为了让测试变绿而擅自改动公式** —— 先确认它是既有问题，再决定是否处理。

注意 `tests/test_check_workspace.py` 在非 Linux 平台会跳过 `/proc/self/mountinfo`
相关断言（该文件自身打印 `[SKIP]`），但整体仍 exit 0。

---

## 7. 改动前的检查清单

1. **明确要改哪个组件**（`app/` / `mcp_server/` / `docker/` / `charts/`），别跨组件耦合。
2. **搜索影响面**：`init_db()` 由 `app/main.py:31` 在启动时调用（另有
   `app/db_migration.py` 的 `__main__` 入口）；`get_db()`/`get_pool()` 的主要影响面是
   `app/dao/__init__.py`（26 处调用点中占 23 处）；rank 换算只应在 `app/rank_layout.py`。
3. **改完跑校验**：至少 `scripts/verify_static.py` + `scripts/verify_consistency.py`；
   涉及 schema 则加 `pg_real_check.py`；涉及 rank 则跑第 6 节相关测试。
4. **改 Dockerfile / 部署契约时同步更新**：`charts/values.yaml`、`.env.example`、
   `README.md`、`docs/使用指南.md`、`docs/数据库内嵌化改造说明.md` —— 这几处与代码存在
   跨文件契约，历史上多次因"只改一处"产生不一致。
5. **提交前检查是否有失效引用**：删除文件后，全仓库搜索该文件名（曾出现删了
   `CLAUDE.md` 但 `docs/使用指南.md` 的目录树仍列着它）。

---

## 8. 大文件与导航提示

| 文件 | 行数 | 说明 |
|------|------|------|
| `app/routes/session.py` | 1699 | **最大文件**：会话 + 拓扑 + 仿真 + 工作流 REST 端点全在这里 |
| `app/agent/orchestrator.py` | 935 | Agent 主编排循环（LLM + 技能 + 工具） |
| `app/models/model_catalog.py` | 681 | 模型参数解析（HF / MindSpeed / 内置） |
| `app/dao/__init__.py` | 565 | PostgreSQL DAO 全集 |
| `mcp_server/services/results_reader.py` | 507 | 仿真结果解析 |
| `app/skills/training-mesh-profiler-skill/` | 550 | 组网性能分析 skill（`__init__.py` + `moe_estimator.py`） |

改动这几个文件时尤其注意：它们体积大且被多处依赖，**小改动也可能产生大影响面**，
建议改前先搜引用方（如 §7 第 2 条）。

技能目录每个都带 `SKILL.md`，改了 skill 行为请同步更新对应的 `SKILL.md`。

---

## 9. 目录级说明

| 路径 | 说明 |
|------|------|
| `app/` | Flask 应用（见 §2） |
| `mcp_server/` | 仿真 MCP Server，独立部署单元 |
| `docker/` | `base/Dockerfile`（基础镜像）、`entrypoint.sh`（启动链）、`check_*.py`（启动自检） |
| `charts/` | Helm chart，含 `database` 卷与 socket 配置 |
| `scripts/` | 改造验证脚本（见 §3），可复跑 |
| `docs/` | 需求规格、使用指南、设计文档、改造说明 |
| `tests/` | 独立测试脚本（非 pytest，见 §6） |
| `static/` | 前端单页应用（`index.html` + 拓扑渲染 + 仿真数据流） |
| `mock/` | 仿真侧 mock 实现（不入镜像，见 `.dockerignore`） |
| `tmp/` | 设计稿与临时资源（**已在 `.gitignore` 中，勿依赖**） |
