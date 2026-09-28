# TrainMeshAgent

AI 训练组网仿真测试 Agent：以 Web 服务形式对接测试人员，用自然语言完成 AI 组网仿真与**模型等效性验证**。

- **需求来源**：`docs/需求规格.md`
- **使用指南（工作流 / API / 数据模型 / 排障）**：`docs/使用指南.md`
- **设计文档**：`docs/mcp-server-spec.md`、`docs/frontend-agent-interaction.md`

---

## 1. 本地启动总览

本地运行需要**三个服务**。它们相互独立，建议各开一个终端窗口：

| 服务 | 组成 | 默认地址 | 是否必需 |
|------|------|----------|----------|
| **PostgreSQL** | 数据库 | `127.0.0.1:5432` | 必需 —— Agent 启动即执行建表迁移 |
| **MCP 仿真 Server** | FastAPI + uvicorn | `http://localhost:9000` | 必需 —— 否则无法下发仿真任务 |
| **Flask 应用** | Web 前端 + Agent + SSE/WebSocket | `http://localhost:5000` | 必需 —— 入口，浏览器访问这个 |

启动顺序按上表从上到下：**数据库 → MCP Server → Flask**。

```
浏览器 ──> Flask :5000 ──HTTP/JSON-RPC──> MCP 仿真 Server :9000 ──拉起子进程──> aicm/run.py
                │                                    │
                │                                    └──写产物──> /home/aicm/workspace（挂载）
                └──psycopg2──> PostgreSQL :5432
                                   └──数据目录──> /home/aicm/db（挂载）

本地开发：PG 在本机 5432；容器内：PG 随容器内嵌，两个挂载点都在 /home/aicm 下
```

---

## 2. 前置准备（三个服务共用）

### 2.1 依赖

```bash
# 建议使用虚拟环境
python -m venv .venv
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\Activate.ps1       # Windows PowerShell

pip install -r requirements.txt              # Flask 侧依赖
pip install -r mcp_server/requirements.txt   # MCP Server 侧依赖（独立，仅服务端需要）
```

> 两份依赖是**独立**的：MCP Server 设计上部署在仿真系统一侧，Agent 只通过 HTTP 与它通信。

### 2.2 仿真工具 `aicm/`

仿真工具源码需放在仓库根的 **`aicm/`** 目录（含 `run.py`、`workload_generator/`）：

```
TrainMeshAgent/
├── aicm/          ← 仿真工具放这里
├── app/
├── mcp_server/
└── ...
```

MCP Server 按 `mcp_server/config.py` 的 `_SERVER_ROOT / "aicm"` 推导该路径（`_SERVER_ROOT` = 仓库根）。

**目录不存在时 MCP Server 本身仍能启动，但下发仿真任务会报 `sim_tool_home not found`** —— 因为该校验发生在真正拉起子进程时（`simulation_runner.prepare_and_launch`）。
若暂时只需联调前后端、不跑仿真，可以不开 `aicm/`，或用只建任务不拉子进程的 `AICM_MCP_DRY_RUN=1`（见 §3.2）。

### 2.3 环境变量

所有配置走环境变量，并支持仓库根下的 `.env` 文件（模板见 `.env.example`，由 `app/config.py` 加载）。

本地开发**通常只需配一个** `OPENAI_API_KEY`：

```bash
cp .env.example .env
# 编辑 .env，填入 OPENAI_API_KEY
```

其余关键变量都有可用默认值。本地会真正用到的：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `OPENAI_API_KEY` | *(空)* | **必填**，LLM 调用 |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | 可指向兼容接口（内网网关等） |
| `OPENAI_MODEL` | `gpt-4o` | 模型名 |
| `OPENAI_SSL_VERIFY` | `true` | 内网证书不全时设 `false` |
| `EXTERNAL_PROXY` | *(空)* | 出网代理，如 `http://proxy.company.com:8080` |
| `DATABASE_URL` | `postgresql://postgres:postgres@127.0.0.1:5432/train_mesh_agent` | 本机 PG 连接串 |
| `MCP_SERVER_URL` | `http://localhost:9000` | Flask 侧要连的 MCP 地址 |
| `FLASK_PORT` | `5000` | Flask 端口 |
| `FLASK_DEBUG` | `false` | 调试模式（本地可设 `true`） |
| `FLASK_USE_RELOADER` | `false` | 代码热重载 |

MCP Server 侧变量以 `AICM_MCP_` 为前缀（见 `mcp_server/config.py`），默认值已与 Flask 侧对齐：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `AICM_MCP_HOST` | `0.0.0.0` | 监听地址 |
| `AICM_MCP_PORT` | `9000` | 监听端口（须与 `MCP_SERVER_URL` 一致） |
| `AICM_MCP_SIM_TOOL_HOME` | `<仓库根>/aicm` | 仿真工具目录 |
| `AICM_MCP_WORKSPACE_ROOT` | `<仓库根>/workspace` | 仿真任务产物目录 |
| `AICM_MCP_CONDA_ENV` | `aicb` | 拉起 `run.py` 用的 conda 环境名 |
| `AICM_MCP_DRY_RUN` | `false` | `true` 时只建任务目录与脚本，不拉起子进程 |

> ⚠️ **`AICM_MCP_CONDA_ENV` 默认值是 `aicb`**，本机没有该 conda 环境时子进程会启动失败。
> 两种处理：`conda create -n aicb python=3.10`（并装上 `aicm` 的依赖），
> 或设 `AICM_MCP_CONDA_ENV=`（空串）让它**回退到当前 Python 解释器** ——
> 这是 `conda_launcher.build_simulation_command` 对空值的既有处理，容器镜像里走的就是后者。

---

## 3. 逐个启动

### 3.1 PostgreSQL

需要本地 PostgreSQL 已运行，且存在业务库 `train_mesh_agent`。

Agent 启动时会**自动建表**，但**不会自动建库**，所以首次需手工创建：

```bash
createdb -U postgres train_mesh_agent
# 或
psql -U postgres -c "CREATE DATABASE train_mesh_agent;"
```

验证：

```bash
psql -U postgres -d train_mesh_agent -c '\dt'
```

> **关于版本**：本机 PG 无需与容器内嵌的 PG 14 同版本。但两者**数据文件与 `pg_dump` 产物跨主版本不可直接复用**，
> 本机 PG 18 + 容器 PG 14 是正常组合，只是两份数据彼此独立。详见 `docs/使用指南.md` §3.2。

### 3.2 MCP 仿真 Server

```bash
# 必须在仓库根执行：mcp_server 需作为包被导入
python -m mcp_server
```

预期输出（uvicorn 启动日志）：

```
INFO:     Uvicorn running on http://0.0.0.0:9000 (Press CTRL+C to quit)
```

验证：

```bash
curl http://localhost:9000/health          # {"status":"ok"}
curl http://localhost:9000/tools           # 已注册工具及 JSON Schema（联调用）
```

自定义端口 / 免 conda / 只建任务不跑仿真：

```bash
AICM_MCP_PORT=9001 \
AICM_MCP_CONDA_ENV= \
AICM_MCP_DRY_RUN=1 \
python -m mcp_server
```

### 3.3 Flask 应用（Agent + 前端）

```bash
# 必须在仓库根执行
python -m app.main
```

> ⚠️ **不要用 `python app/main.py`。** 实测会报 `ModuleNotFoundError: No module named 'app'` ——
> Python 执行脚本时是把**脚本所在目录**（`app/`）放进 `sys.path`，而不是当前目录（仓库根），
> 因此 `from app.config import config` 找不到包。`-m` 形式则会把仓库根放入 `sys.path`，故可用。
> 确实想跑脚本形式的话，需显式指定：`PYTHONPATH=. python app/main.py`。

预期输出：

```
INFO [app.db_migration] All tables created successfully.
INFO [app.main] Starting TrainMesh Agent on 0.0.0.0:5000
```

启动后浏览器打开 **http://localhost:5000**。

验证：

```bash
curl http://localhost:5000/api/health    # {"status":"ok","service":"trainmesh-agent"}
curl http://localhost:5000/api           # 端点清单
```

> 这一步会**真实执行建表迁移**并写入 `model_catalog` 种子数据（38 个模型）。
> 数据库不可用时应用直接启动失败 —— 这是有意设计，避免带病运行。

---

## 4. 启动完成自检

三条都通过即三个服务就绪：

```bash
curl -s http://localhost:5000/api/health            # Flask
curl -s http://localhost:9000/health                # MCP
psql -U postgres -d train_mesh_agent -c '\dt'       # PostgreSQL（应列出 7 张表）
```

期望的 7 张表：`sessions`、`topology_params`、`simulation_params`、`simulation_results`、`comparison_reports`、`conversation_messages`、`model_catalog`。

MCP Server 注册 9 个工具：`execute_task`、`report_status`、`sync_logs`、`get_result`、`card_detail`、`get_device_detail`、`get_hbm_detail`、`get_comm_detail`、`get_training_script`。

---

## 5. 端口与停止

| 服务 | 端口 | 停止方式 |
|------|------|----------|
| Flask | 5000 | 前台 `Ctrl+C` |
| MCP Server | 9000 | 前台 `Ctrl+C` |
| PostgreSQL | 5432 | 系统服务管理（如 `systemctl stop postgresql`） |

端口占用排查：

```bash
# Linux / macOS
lsof -i :5000
lsof -i :9000

# Windows PowerShell
Get-NetTCPConnection -LocalPort 5000,9000 -State Listen
```

---

## 6. 常见问题

| 现象 | 原因 | 处理 |
|------|------|------|
| `ModuleNotFoundError: No module named 'app'` / `'mcp_server'` | 不在仓库根执行 | `cd` 到仓库根，并用 `python -m ...` 形式 |
| 启动即退出 / `[db] 建池失败 ...` | PostgreSQL 未启动或缺 `train_mesh_agent` 库 | 见 §3.1；`app/db.py` 会重试 10 次 × 1.5s 后放弃 |
| `sim_tool_home not found` | 仓库根缺 `aicm/` | 放入仿真工具（§2.2），或先设 `AICM_MCP_DRY_RUN=1` 联调 |
| 仿真子进程起不来、日志有 conda 报错 | `AICM_MCP_CONDA_ENV` 默认 `aicb`，本机无此环境 | 建该 conda 环境，或设 `AICM_MCP_CONDA_ENV=` 回退当前解释器（§2.3） |
| 前端对话无响应 / 报 LLM 错误 | 未配 `OPENAI_API_KEY`，或需走代理 | 配置 `.env` 的 `OPENAI_API_KEY` / `EXTERNAL_PROXY` |
| MCP 连不上 | 端口不一致 | 确认 `MCP_SERVER_URL` 与 `AICM_MCP_PORT` 相同（默认均 9000） |
| 历史列表标题显示「新建任务」 | schema 不完整（缺表/缺列） | 看启动日志的 `[migration]` 行；迁移失败会抛错而非静默继续 |
| 想重置数据库 | —— | `DROP DATABASE train_mesh_agent;` 后重建，重启 Flask 会自动建表并重新 seed 模型目录 |

更多排障（含容器内嵌数据库的 7 类故障）见 `docs/使用指南.md` §12。

---

## 7. 容器化部署（可选）

生产/联调环境走「数据库内嵌」形态：一个容器自洽启动，无需外部 PostgreSQL。

```bash
docker run -d --name train-mesh-agent \
  -v /data/aicm/workspace:/home/aicm/workspace \
  -v /data/aicm/db:/home/aicm/db \
  -p 5000:5000 <image>
```

容器内两个挂载点**统一放在 `/home/aicm` 下**（`workspace` 与 `db`），节点侧各自独立目录。
两个挂载**都是必需的**，缺失时容器明确报错退出。
构建流程、持久化契约、主版本升级路径与运维须知见 **`docs/数据库内嵌化改造说明.md`**。
