# equivalent-modeling-service

AI 训练组网仿真测试 Agent：以 Web 服务形式对接测试人员，用自然语言完成 AI 组网仿真与**模型等效性验证**。

- **需求来源**：`docs/需求规格.md`
- **使用指南（工作流 / API / 数据模型 / 排障）**：`docs/使用指南.md`
- **设计文档**：`docs/仿真系统MCP-Server需求规格.md`、`docs/前端与Agent交互逻辑分析.md`

---

## 1. 本地启动总览

本地运行需要**两个服务**，外加一个**内嵌数据库**。数据库无需单独启动，数据默认落在
本地 SQLite 单文件里（`SQLITE_PATH`）。

| 服务 | 组成 | 默认地址 | 是否必需 |
|------|------|----------|----------|
| **数据库** | 默认内嵌 SQLite（标准库 `sqlite3`） | 本地文件 `SQLITE_PATH` | 必需 —— Agent 启动即执行建表迁移，**无需单独启动** |
| **MCP 仿真 Server** | FastAPI + uvicorn | `http://localhost:9000` | 必需 —— 否则无法下发仿真任务 |
| **Flask 应用** | Web 前端 + Agent + SSE/WebSocket | `http://localhost:5000` | 必需 —— 入口，浏览器访问这个 |

启动顺序：**MCP Server → Flask**（数据库随 Flask 进程自动就绪）。

```
浏览器 ──> Flask :5000 ──HTTP/JSON-RPC──> MCP 仿真 Server :9000 ──拉起子进程──> aicm/run.py
                │                                    │
                │                                    └──写产物──> /home/data/workspace（挂载）
                └──sqlite3（默认）/ psycopg2（PG 逃生门）
                       └──数据目录──> /home/data/db（挂载）或本地 SQLITE_PATH

本地开发：数据库默认是仓库外的单文件，可用 SQLITE_PATH 指到任意可写位置；
容器内：数据库文件与 workspace 两个挂载点都在 /home/data 下（仿真工具在 /home/aicm）。
```

> **数据库是双后端的**：默认走 SQLite；把 `DATABASE_URL` 设成 `postgresql://...`
> 即切换到 PostgreSQL（逃生门，代码同一份）。切换机制与取舍见
> `docs/PostgreSQL迁移SQLite改造计划.md`。

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
equivalent-modeling-service/
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
| `DATABASE_URL` | *(空)* | **留空 = 用内置 SQLite**；填 `postgresql://...` 才切到外部 PostgreSQL |
| `SQLITE_PATH` | `/home/data/db/equivalent_modeling_service.db` | SQLite 数据文件路径（本地开发建议指到仓库外或 `.tmp/`） |
| `SQLITE_BUSY_TIMEOUT_MS` | `5000` | 写锁等待上限（毫秒）；WAL 下单写者，靠等待而非报错 |
| `SQLITE_SYNCHRONOUS` | `NORMAL` | `OFF` / `NORMAL` / `FULL` / `EXTRA`；WAL 下 `NORMAL` 只在 checkpoint 时 fsync |
| `MCP_SERVER_URL` | `http://localhost:9000` | Flask 侧要连的 MCP 地址 |
| `FLASK_PORT` | `5000` | Flask 端口 |
| `FLASK_DEBUG` | `false` | 调试模式（本地可设 `true`） |
| `FLASK_USE_RELOADER` | `false` | 代码热重载 |

> **`DATABASE_URL` 的判定只看前缀**：以 `postgres://` / `postgresql://` 开头才走 PostgreSQL，
> 其余（含留空）一律走 SQLite。**写错协议头不会报错**，只会静默连到 SQLite 而看起来像"数据丢了"，
> 这是最常见的配置困惑点。

MCP Server 侧变量以 `AICM_MCP_` 为前缀（见 `mcp_server/config.py`），默认值已与 Flask 侧对齐：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `AICM_MCP_HOST` | `0.0.0.0` | 监听地址 |
| `AICM_MCP_PORT` | `9000` | 监听端口（须与 `MCP_SERVER_URL` 一致） |
| `AICM_MCP_SIM_TOOL_HOME` | 本地 `<仓库根>/aicm`；容器内钉为 `/home/aicm` | 仿真工具目录（容器内数据挂载在 `/home/data`，与工具目录分置，避免整目录挂载遮掉工具） |
| `AICM_MCP_WORKSPACE_ROOT` | `<仓库根>/workspace` | 仿真任务产物目录 |
| `AICM_MCP_CONDA_ENV` | *(空)* | 拉起 `run.py` 用的 conda 环境名；默认为空 = 用当前 Python 解释器 |
| `AICM_MCP_DRY_RUN` | `false` | `true` 时只建任务目录与脚本，不拉起子进程 |

> **`AICM_MCP_CONDA_ENV` 默认为空串**：`conda_launcher.build_simulation_command` 在空值时
> **回退到当前 Python 解释器**，因此本机与容器都不需要 conda（容器镜像走的正是这条）。
> 只有确实要跑在某个 conda 环境里时才显式设置 `AICM_MCP_CONDA_ENV=<环境名>`
> （该环境需已装好 `aicm` 的依赖）。

---

## 3. 逐个启动

### 3.1 数据库（默认内嵌，无需启动）

**默认不需要做任何事。** Agent 启动时会自动建表、自动 seed 内置模型清单，数据落到
`SQLITE_PATH` 指定的单文件里。

```bash
# 本地开发建议把数据文件放到仓库外或 .tmp/，避免污染工作区
export SQLITE_PATH=/tmp/equivalent_modeling_service.db     # Windows: $env:SQLITE_PATH=".tmp\equivalent_modeling_service.db"
python -m app.db_migration                       # 可选：只跑迁移，不启动服务
```

验证（迁移结束会打印新建/复用的表清单）：

```
[migration] 首次建表完成，本次新建 7 张：sessions、topology_params、...
```

> **与 PG 时代的两点行为差异**：
> 1. **不再需要 `createdb`**。SQLite 按路径建库，库与表一起由迁移创建。
> 2. **数据文件是整体备份单位**。WAL 模式下同目录还有 `equivalent_modeling_service.db-wal` /
>    `-shm`，备份或搬迁要整目录拷，别只拷 `.db` 主文件。

#### 3.1.1 （可选）切到 PostgreSQL 逃生门

需要外部 PostgreSQL 时，只要设置 `DATABASE_URL`（**前缀必须是 `postgresql://` 或 `postgres://`**）：

```bash
export DATABASE_URL='postgresql://postgres:postgres@127.0.0.1:5432/equivalent_modeling_service'
python -m app.db_migration
```

此时业务库需已存在（Agent 只建表、不建库）:

```bash
createdb -U postgres equivalent_modeling_service
# 或
psql -U postgres -c "CREATE DATABASE equivalent_modeling_service;"
```

> **注意**：容器镜像里**已不装 PostgreSQL 服务端**，逃生门只能连**外部** PG。
> 本机 PG 与容器 PG 的数据文件、`pg_dump` 产物**跨主版本不可直接复用**。
> 详见 `docs/使用指南.md` §3.2 与 `docs/PostgreSQL迁移SQLite改造计划.md`。

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

自定义端口 / 只建任务不跑仿真（默认已免 conda，需要特定 conda 环境时才设 `AICM_MCP_CONDA_ENV=<环境名>`）：

```bash
AICM_MCP_PORT=9001 \
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
INFO [app.main] Starting equivalent-modeling-service on 0.0.0.0:5000
```

启动后浏览器打开 **http://localhost:5000**。

前端自带**暗色 / 浅色两套配色**，默认暗色：标题栏右上角的 ☀ / ☾ 按钮切换，选择记在
`localStorage["tms-theme"]`；想直接看另一套配色可在 URL 上带参数（不落盘）：
`http://localhost:5000/?theme=light` 或 `?theme=dark`。

配色的唯一事实来源是 `static/index.html` `<style>` 顶部的 token 层（`:root` = 暗色，
`:root[data-theme="light"]` = 浅色），组件样式一律引用 `var(--token)`，
详见 `docs/使用指南.md` §13「前端配色与主题」。

验证：

```bash
curl http://localhost:5000/api/health    # {"status":"ok","service":"equivalent-modeling-service"}
curl http://localhost:5000/api           # 端点清单
```

> 这一步会**真实执行建表迁移**并写入 `model_catalog` 种子数据（38 个模型）。
> 数据库不可用时应用直接启动失败 —— 这是有意设计，避免带病运行。

---

## 4. 启动完成自检

三条都通过即两个服务 + 数据库就绪：

```bash
curl -s http://localhost:5000/api/health            # Flask
curl -s http://localhost:9000/health                # MCP
python -c "import sqlite3,os;print(sorted(r[0] for r in sqlite3.connect(os.getenv('SQLITE_PATH','/home/data/db/equivalent_modeling_service.db')).execute(\"SELECT name FROM sqlite_master WHERE type='table'\")))"   # 数据库（应列出 7 张表）
```

（逃生门形态改用 `psql -U postgres -d equivalent_modeling_service -c '\dt'`。）

期望的 7 张表：`sessions`、`topology_params`、`simulation_params`、`simulation_results`、`comparison_reports`、`conversation_messages`、`model_catalog`。

MCP Server 注册 9 个工具：`execute_task`、`report_status`、`sync_logs`、`get_result`、`card_detail`、`get_device_detail`、`get_hbm_detail`、`get_comm_detail`、`get_training_script`。

---

## 5. 端口与停止

| 服务 | 端口 | 停止方式 |
|------|------|----------|
| Flask | 5000 | 前台 `Ctrl+C` |
| MCP Server | 9000 | 前台 `Ctrl+C` |
| PostgreSQL | 5432 | 仅逃生门使用；默认不涉及（数据库是内嵌 SQLite 文件） |

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
| 启动即退出 / `[db] 建池失败 ...` | 数据库不可用：`SQLITE_PATH` 目录不可写，或（逃生门）`DATABASE_URL` 指向的 PG 未启动 | 检查 `SQLITE_PATH` 父目录权限；PG 情形见 §3.1.1。`app/db.py` 会重试 10 次 × 1.5s 后放弃 |
| `sim_tool_home not found` | 仓库根缺 `aicm/` | 放入仿真工具（§2.2），或先设 `AICM_MCP_DRY_RUN=1` 联调 |
| 仿真子进程起不来、日志有 conda 报错 | 显式设了 `AICM_MCP_CONDA_ENV=<环境名>`，但本机无此环境 | 清空该变量（默认即空，回退当前解释器），或建好该 conda 环境并装上 `aicm` 依赖（§2.3） |
| 前端对话无响应 / 报 LLM 错误 | 未配 `OPENAI_API_KEY`，或需走代理 | 配置 `.env` 的 `OPENAI_API_KEY` / `EXTERNAL_PROXY` |
| MCP 连不上 | 端口不一致 | 确认 `MCP_SERVER_URL` 与 `AICM_MCP_PORT` 相同（默认均 9000） |
| 历史列表标题显示「新建任务」 | schema 不完整（缺表/缺列） | 看启动日志的 `[migration]` 行；迁移失败会抛错而非静默继续 |
| 明明设了 `DATABASE_URL` 却像在用 SQLite（"数据丢了"） | 前缀不是 `postgres://` / `postgresql://`（如写成 `postgresql:/`、`postgres@Data`） | 前缀不对就一律走 SQLite。启动日志的 `(backend=sqlite\|postgres)` 能直接确认真实后端 |
| 想重置数据库 | —— | 默认后端：停服后删除 `SQLITE_PATH` 指向的文件**及其 `-wal`/`-shm` 伴生文件**，重启 Flask 会自动建表并重新 seed 模型目录；PG 后端：`DROP DATABASE equivalent_modeling_service;` 后重建 |

更多排障（含容器内嵌数据库的 7 类故障）见 `docs/使用指南.md` §12。

---

## 7. 容器化部署（可选）

生产/联调环境走「数据库内嵌」形态：一个容器自洽启动，**无需任何外部数据库服务**
（默认后端是 SQLite 单文件，由镜像内的 Python 标准库驱动）。

```bash
docker run -d --name equivalent-modeling-service \
  -v /data/aicm/workspace:/home/data/workspace \
  -v /data/aicm/db:/home/data/db \
  -p 5000:5000 <image>
```

容器内两个挂载点**统一放在 `/home/data` 下**（`workspace` 与 `db`），节点侧各自独立目录。
两个挂载**都是必需的**，缺失时容器明确报错退出 —— 这是有意的：数据库文件一旦静默落进
镜像层，容器重建就会丢掉全部会话历史。
仿真工具本体（`run.py` 等）在镜像内 `/home/aicm`，**不**与数据挂载树同名 ——
数据挂载与工具目录分置后，整目录挂载不会再遮掉工具代码（工具始终留在镜像层）。

需要外部 PostgreSQL 时（逃生门），注入 `DATABASE_URL` 即可，**同一镜像无需重建**：

```bash
docker run -d --name equivalent-modeling-service \
  -v /data/aicm/workspace:/home/data/workspace \
  -v /data/aicm/db:/home/data/db \
  -e DATABASE_URL='postgresql://user:pw@pg.internal:5432/equivalent_modeling_service' \
  -p 5000:5000 <image>
```

> 镜像内**已不再安装 PostgreSQL 服务端**，因此逃生门只能连**外部** PG。

### 7.1 首次部署与重启的行为差异

**首次部署**：`docker/entrypoint.sh` 会自动完成全部初始化，无需人工介入：

```
工作区自检 → DB 目录自检 → 启动 MCP Server → 启动 Flask（建表迁移 + seed）
```

数据库文件不存在时由建表迁移直接创建 —— **既不需要 `initdb`，也不需要建库**，
这正是本次改造去掉的一整段启动链。

**重启 / 重新部署**：数据在挂载目录里，行为与首次部署完全一致（幂等）：

- 迁移同时承担「**校验** schema 完整性」的职责。只有每次启动都跑，才能保证「schema 不完整
  就拒绝启动」，而不是带着半可用的库对外服务。
- 迁移是**幂等的**：所有 DDL 都带 `IF NOT EXISTS`，重放会静默成功，
  所以重启不会报错、也不会重复建对象。
- 顺带这也是升级路径：镜像里新增了表/列，重启后自动补建。

日志可以直接区分这两种情况（无需猜数据卷是否为空）：

```
[migration] 首次建表完成，本次新建 7 张：sessions、topology_params、...
[migration] 复用已有数据库（后端 sqlite，7 张表均已存在），迁移以幂等方式重放
[migration] All tables created successfully. (backend=sqlite)
```

停机时 entrypoint 会对 SQLite 做一次 `wal_checkpoint(TRUNCATE)`，把 `-wal` 里的
已提交事务并回主文件 —— 这纯粹是为了让「拷贝数据目录」这类运维动作拿到完整快照，
失败也不会阻止容器退出。

> 若迁移失败（缺表/缺列），`init_db()` 会抛 `RuntimeError` 导致 Flask 进程退出；
> entrypoint 的 `wait -n` 随即回收整个容器 —— **不会**以半可用状态继续运行。

构建流程、持久化契约与运维须知见 **`docs/PostgreSQL迁移SQLite改造计划.md`**；
上一版「内嵌 PostgreSQL」形态的实现细节见 **`docs/数据库内嵌化改造说明.md`**（已被本次改造取代，仅作历史参考）。

### 7.2 反向代理 / 子路径前缀（如 nginx `/ftbot/equivalent/`）

本应用可以被 nginx（或任何反向代理）挂在**任意子路径**下，例如：

```nginx
location /ftbot/equivalent/ {
    proxy_pass http://7.185.127.30:38088/;   # 尾斜杠 = 剥掉前缀后再转发，必须保留
}
```

**前缀归属：浏览器一侧，不是后端。** `proxy_pass` 结尾的 `/` 会把 `/ftbot/equivalent/`
从前缀里剥掉，因此 Flask 收到的仍是 `/`、`/api/...`、`/static/...`、`/ws/...`，
后端**不需要**（也不应该）知道这个前缀。真正要知道前缀的是浏览器，所以前端把
所有地址都从**当前文档地址**推导（`static/index.html` 里的 `APP_BASE`，单一事实来源）：

- 6 处资源引用写成**相对路径** `static/xxx`（随文档地址解析，天然带前缀）；
- `API = APP_BASE + "/api"`（`topo-renderer.js` 的 6 处 fetch 都复用这个全局量）；
- WebSocket 用 `location.host + APP_BASE + "/ws/simulation/<id>"`。

于是**根路径部署与任意前缀部署共用同一份前端代码**，无需按环境改 HTML，
也不需要 `SCRIPT_NAME` / `X-Forwarded-Prefix` / `ProxyFix` 之类的后端配合。

> 历史故障：资源与接口曾写死为 `/static/...`、`/api`。挂在 `/ftbot/equivalent/` 下时
> 页面能返回 200，但浏览器会把资源请求打到 nginx **根**上（那里没有本应用的 location），
> 表现为「页面打开是白的、控制台一片 404」——静态资源、API、WebSocket 全挂。

四条容易漏掉的代理配置：

```nginx
map $http_upgrade $connection_upgrade { default upgrade; '' close; }

server {
    # ① 裸前缀（不带尾斜杠）匹配不到 location /ftbot/equivalent/ → 必须显式跳转
    location = /ftbot/equivalent { return 301 /ftbot/equivalent/; }

    location /ftbot/equivalent/ {
        proxy_pass http://7.185.127.30:38088/;
        proxy_http_version 1.1;

        # ② WebSocket（/ws/simulation/<id>）：缺这两个头握手直接失败
        proxy_set_header Upgrade    $http_upgrade;
        proxy_set_header Connection $connection_upgrade;

        # ③ SSE（/api/chat/stream、/api/session/<id>/workflow/step2/stream）：
        #    应用已回 X-Accel-Buffering: no，这里再显式关一层缓冲，
        #    否则「逐行推送」会被攒成一坨、流结束才一次性下发
        proxy_buffering off;
        proxy_cache off;

        # ④ 长任务（LLM 调用 / 仿真轮询）：默认 60s 读超时会被掐断
        proxy_read_timeout 3600s;

        proxy_set_header Host              $host;
        proxy_set_header X-Real-IP         $remote_addr;
        proxy_set_header X-Forwarded-For   $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

**端口映射别改容器内端口**：上面 `38088` 是宿主侧映射（`-p 38088:5000`）。
容器内保持 `FLASK_PORT=5000` —— `Dockerfile` 的 `HEALTHCHECK` 写死探测
`127.0.0.1:5000`，改成 38088 会让容器被判定为不健康。

自检（把 `<host>` 换成实际入口即可，应全部 2xx / 101）：

```bash
curl -sI http://<host>/ftbot/equivalent/                     # 页面 200
curl -sI http://<host>/ftbot/equivalent/static/d3.v7.min.js  # 静态资源 200
curl -s  http://<host>/ftbot/equivalent/api/health           # API 200 {"status":"ok",...}
```
