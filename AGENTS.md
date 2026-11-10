# AGENTS.md — 项目说明（供 AI 编码助手使用）

面向在本仓库工作的 AI 助手。目标：**让你在动手前就掌握架构约束、硬性约定与已知陷阱**，
避免改错地方、踩到只有运行时才暴露的坑。

---

## 1. 项目是什么

**equivalent-modeling-service**：AI 训练组网仿真测试 Agent。测试人员通过 Web 界面用自然语言描述组网
（设备类型 A2/A3/A5、DP/TP/PP 并行度），Agent 生成结构化组网图 → 下发仿真 → 对比原始组网与
等效组网的**计算强度 / 内存占用 / 通信流量**，验证"模型等效"是否成立。

需求原文见 `docs/需求规格.md`（顶部注记记录了实现偏离：未采用 openai agent sdk；其 §2.3 被
`app/skills/registry.py` 引用，勿删）。使用与 API 细节见 `docs/使用指南.md`。

---

## 2. 三个组件（理解这一点才能不迷路）

```
浏览器 ──> Flask app :5000 ──HTTP JSON-RPC──> MCP 仿真 Server :9000 ──拉起子进程──> aicm/run.py
                 │
                 └──sqlite3（默认）/ psycopg2（PG 逃生门）──> app/db.py
```

| 组件 | 位置 | 职责 | 关键点 |
|------|------|------|--------|
| **Flask 应用** | `app/` | 前端 + Agent 编排 + REST/SSE/WebSocket | 入口 `app/main.py`；启动即执行建表迁移 |
| **MCP 仿真 Server** | `mcp_server/` | 暴露 10 个仿真工具，管理任务生命周期（并发上限 + 排队 + 超时 + 取消）；另有只读产物文件 REST（`/tasks/<task_id>/files*`） | **独立部署单元**，有自己的 `requirements.txt`；`app/mcp/client.py` 只通过 HTTP 调它 |
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
# 数据库默认是 SQLite，无需任何外部服务；首次启动自动建表并 seed 模型清单。
# 只有要用外部 PostgreSQL 时才需要设 DATABASE_URL（见 §4.4 逃生门）。

# 校验
python scripts/verify_static.py        # db_migration.py 结构不变量（含两套 DDL 一致性 D1/D2）+ 全仓库语法
python scripts/verify_consistency.py   # 镜像 ENV / config.py / charts / entrypoint 路径契约对齐
python scripts/render_chart.py         # Helm 模板渲染校验（本机无 helm 时的替代）
python scripts/sqlite_real_check.py    # 真实 SQLite：建表/幂等/硬校验正负向/DAO 全路径/并发写
python scripts/backend_detect_check.py # DATABASE_URL 前缀判定矩阵（静默回落 SQLite 的边界）
ADMIN_DSN=postgresql://postgres:<pw>@127.0.0.1:5432/postgres \
  python scripts/pg_real_check.py      # PG 逃生门：连真实 PG 跑建表/幂等/硬校验正负向
```

数据库默认是 **SQLite**，不需要任何外部服务，因此 `python -m app.main` 直接可跑
（数据文件默认落在 `SQLITE_PATH`，本地开发可指到 `.tmp/` 下）。

数据库选型与迁移细节见 `docs/PostgreSQL迁移SQLite改造计划.md`；部署与镜像细节见 `README.md`。

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

调用点分布（共 24 处，改前请知悉影响面）：**23 处在 `app/dao/__init__.py`**，
其余在 `app/db_migration.py`。也就是说路由层是通过 DAO 间接使用的，
DAO 是真正的影响集中点。

### 4.4 数据库相关约定（双后端：SQLite 默认 / PostgreSQL 逃生门）

- **默认后端是 SQLite**：单文件 `SQLITE_PATH=/home/data/db/equivalent_modeling_service.db`，
  由标准库 `sqlite3` 驱动。基础镜像**不再安装 PostgreSQL 服务端**（`libpq5` 保留，
  供 PG 逃生门的 psycopg2 使用）。因此没有了 initdb / 版本守卫 / Unix socket /
  就绪探测这一整套启动链。
- **后端由 `DATABASE_URL` 的前缀自动侦测**（`app/dbapi.detect_backend`）：
  `postgres://` / `postgresql://` 开头 → PG；其余（含空串）→ SQLite。
  **不要引入 `DB_BACKEND` 这类独立开关** —— 开关与 URL 不一致会产生第四种状态。
- **Windows / 本机首启必须改 `SQLITE_PATH`**：默认值是**容器路径**
  `/home/data/db/equivalent_modeling_service.db`，在 Windows 上会被解析成 `\home\data\db`（即当前盘符
  根下的 `home\data\db`）而**不存在**，启动直接抛
  `RuntimeError: SQLite 数据目录不存在：\home\data\db`。`init_db()` 有意**不自动建父目录**
  （避免数据库悄悄落进可写镜像层），所以必须自己建好并指到可写位置，例如
  `SQLITE_PATH=./.tmp/equivalent_modeling_service.db`（`.tmp/` 已在 `.gitignore` 中，需先 `mkdir .tmp`）。
  正斜杠与反斜杠写法都能被 `python-dotenv` 正确读取。
- **`replicas` 必须为 1**：SQLite（WAL）是库级单写者。Pod 漂移到其他节点会看到"空"数据库
  （与 `/home/data/workspace` 同源的失效模式）。
- **禁止把数据库目录放 NFS**：SQLite 依赖本地文件锁（POSIX advisory lock）与 `fsync`，
  NFS 上的锁语义不可靠，会损坏数据。
- **WAL 伴生文件**：同目录还有 `-wal` / `-shm`。备份要整目录拷（或先走
  `docker/checkpoint_db.py` 的 TRUNCATE checkpoint，停机时 entrypoint 会自动做）。
- 逃生门：注入 `DATABASE_URL` 即切回**外部** PostgreSQL，**同一镜像无需重建**（勿删）。
  chart 侧对应 `values.yaml` 的 `config.databaseUrl`（条件渲染 → ConfigMap 下发，
  见 `charts/equivalent-modeling-service/templates/configMap.yaml`）。
  注意：镜像内已无 PG 服务端，逃生门只能连外部 PG。
- 两套 DDL 的一致性是**硬约束**（见 §4.2），由 `scripts/verify_static.py` 的
  D1/D2 断言把守。

### 4.5 行末符与编码

`.gitattributes` 强制 `*.sh`、`Dockerfile`、`*.dockerignore`、`.gitattributes` 为 **LF**。
`build.sh` 是把工作区文件直接打进 tar 的，CRLF 会让容器内的 `trap` / 信号名带上 `\r` 而失败。
本仓库 `core.autocrlf=true`，因此**工作区看到 CRLF 是正常的，但要确认提交进仓库的是 LF**。

### 4.6 前端颜色的单一事实来源是 `static/index.html` 的 token 层

前端有**暗色（默认）/ 浅色**两套配色，由 `<html data-theme="dark|light">` 切换（标题栏
右上角按钮，选择存 `localStorage["tms-theme"]`；`?theme=light|dark` 可临时覆盖且不落盘）。

- 颜色只允许写在一处：`static/index.html` `<style>` 顶部的 `:root { ... }`（暗色）与
  `:root[data-theme="light"] { ... }`（浅色，**只覆盖颜色**，不含组件规则）。
- **组件规则、SVG 表现属性一律用 `var(--token)`，禁止再出现颜色字面量**
  （`#39bae6` / `rgba(57,186,230,.08)` 等）。半透明写成 `rgba(var(--xxx-rgb), a)`，
  不要用 `color-mix()`（内网浏览器版本不可控）。
- **JS 侧同样适用**：`static/topo-renderer.js`、`static/moe-arch-renderer.js`、
  `static/moe-arch-renderer.css` 里的颜色也必须是变量。CSS 变量运行时解析，
  所以**切换主题不需要重绘 SVG**；需要按主题重算的非颜色逻辑请监听 `tms:themechange` 事件。
- 不要把 CSS 变量读成具体色值再缓存进 JS（`getComputedStyle` + 变量），那样切主题就不生效了。
- **不要用颜色字符串做逻辑判断**（历史写法 `color === "#3fb950"` 决定"是否仿真柱"）——
  改成显式参数或语义标记，否则换成变量后比较必然失效。
- 改了前端资源要同步 `static/index.html` 里的 `?v=` 版本号，否则浏览器继续用旧缓存。
- **禁止引用外链资源**（字体 / CSS / JS / 图片）。平台下发的 CSP 只放 `*.huawei.com`，
  外链样式表会被拦（控制台 `violates the following Content Security Policy directive`），
  内网本身也访问不到。依赖一律 vendored 进 `static/`（`d3.v7.min.js`、`marked.min.js` 已是），
  字体走 `local()` 的 `@font-face`（见 `--font-mono` 上方的说明）。

---

## 5. 已知陷阱（都是实际踩过的）

| 陷阱 | 说明 |
|------|------|
| **不要用 `python app/main.py`** | 实测报 `ModuleNotFoundError: No module named 'app'` —— Python 把**脚本所在目录**（`app/`）放进 `sys.path` 而非 CWD。必须用 `python -m app.main`（或 `PYTHONPATH=. python app/main.py`） |
| **`AICM_MCP_CONDA_ENV` 默认为空串** | 空值即回退到当前 Python 解释器（`mcp_server/services/conda_launcher.py` 对空值的处理），因此本机与容器都无须 conda。只有显式设了 `<环境名>` 而该环境不存在时，仿真子进程才会起不来 |
| **`aicm/` 缺失不阻止 MCP Server 启动** | 校验发生在 `simulation_runner.prepare_task`，即真正下发任务时才报错。纯联调可用 `AICM_MCP_DRY_RUN=1`（只建任务不拉子进程） |
| **依赖名与 import 名不一致** | `python-dotenv`→`dotenv`、`pyyaml`→`yaml`、`psycopg2-binary`→`psycopg2`、`flask-cors`→`flask_cors`、`flask-sock`→`flask_sock`。做依赖审计时必须归一化，否则全部误报 |
| **`httpx` 是直接依赖** | `app/agent/orchestrator.py` 直接 import 并构造 OpenAI 客户端（承载 `OPENAI_SSL_VERIFY` / `EXTERNAL_PROXY`），已显式声明，不要当成 openai 的传递依赖而移除 |
| **SQLite 读回 BOOLEAN 是 0/1，不是 bool** | `app/dao` 出口统一走 `normalize_row()`。漏了这一层，`is_equivalent` / `has_shared_expert` 等列进入 `app/models/schemas.py` 的 pydantic 模型时会因严格布尔校验直接抛错（不是美化问题） |
| **SQLite 主键必须应用侧生成** | 没有 `gen_random_uuid()`。`app/dao` 用 `uuid4()` 显式传 `id`；`save_simulation_result` 还必须**先查已存在的 id**再 upsert，否则冲突时主键漂移，`comparison_reports` 的外键会指错行 |
| **`ON CONFLICT` 在 SQLite 下必须能定位目标** | `comparison_reports` 两套 DDL 都建了 `session_id` 唯一约束（PG 侧是 `uq_comparison_reports_session`），否则「重复保存报告」在 SQLite 下会插入多行而非幂等。这条由 `verify_static.py` 的 D2 断言把守 |
| **不要引入 `DB_BACKEND` 这类独立开关** | 后端由 `DATABASE_URL` 前缀侦测。开关与 URL 不一致会产生第四种状态，届时「为什么连不上」会成为排查陷阱 |
| **非 postgres:// 前缀的 DSN 会被**静默**当成 SQLite** | 这是最容易误判的配置错误。DSN 写错协议头不会报错，只会连到 SQLite 而看起来「数据丢了」 |
| **chart 里不能有取值为 nil 的模板键** | Sprig 的 `quote(nil)` 返回**空串**（不是 `"None"`），于是 `KEY: {{ .Values.x \| quote }}` 渲染成 `KEY:` → YAML null → k8s 直接拒绝整次发布：`unknown object type "nil"`（2026-09-30 真实事故：平台未登记 `openaiApiKey` 占位符 → 空串 → nil，报在 `Secret.stringData.OPENAI_API_KEY` 上）。规则：**凡可能为空的键一律条件渲染**（`{{- if ... }}` 整行不渲染），不要渲染成空值。当前：`templates/secret.yaml` 已删除，LLM 四个键（`config.openaiBaseUrl` / `openaiModel` / `openaiSslVerify` / `openaiApiKey`）由**平台变量注入**（占位符即变量名，如 `@config.openaiModel@`），在 `configMap.yaml` 里全部条件渲染 —— 变量未登记时该键不下发、回退 `app/config.py` 默认值（API Key 缺失则首次调用报鉴权失败，不卡发布）。`scripts/render_chart.py` 的 `check_no_null_values` 与形态 C（变量全未登记）断言把守这条 |
| **`NO_PROXY` 含方括号 IPv6 会让 httpx 直接抛异常** | httpx 0.28 解析 `NO_PROXY=...,::1,[::1]` 这类值时，`URLPattern` 会崩在 `InvalidURL: Invalid port: ':1]'`。而 `app/agent/orchestrator.py` 用 `httpx.Client(verify=..., proxy=...)` 构造 OpenAI 客户端 —— 一旦运行环境设了这个值，**首次 LLM 调用即失败**（不是降级，是直接抛）。对策是清掉 `NO_PROXY` 里的 `[::1]` 写法（只留 `127.0.0.1` 与 `::1`），代码侧无法规避 |
| **前端写死根绝对路径会在子路径部署下全 404** | nginx `location /pfx/ { proxy_pass http://host:38088/; }` 的尾斜杠会把 `/pfx/` 剥掉再转发，所以**后端不需要知道前缀，浏览器必须知道**。`static/index.html` 里 6 处资源引用曾是 `src="/static/..."`、接口曾是 `const API = "/api"`、WS 曾是 `location.host + "/ws/simulation/"` —— 挂到 `/ftbot/equivalent/` 时页面返回 200 但资源/API/WS 全被浏览器打到 nginx **根**上（那里没有本应用 location）→ 白屏 + 控制台 404。现在统一由 `APP_BASE`（按 `location.pathname` 推导）作为**单一事实来源**：资源用相对路径 `static/xxx`，接口 `APP_BASE + "/api"`，WS `location.host + APP_BASE + "/ws/simulation/"`。**新增任何前端请求地址都必须走 `APP_BASE`/相对路径，禁止再写 `/api`、`/static/` 开头的绝对路径**（`topo-renderer.js` 的 6 处 fetch 复用 index.html 的全局 `API`，不要另建常量） |
| **前端颜色必须走 CSS 变量，不能写死** | 后端只有暗色时无所谓，加了浅色主题后，任何写死的 `#hex` / `rgba(...)` 都不跟随主题，页面上会出现"一块深色留在浅色界面里"。事实来源是 `static/index.html` 的 token 层（见 §4.6）。另注意：**不能用颜色字符串做逻辑判断**（如 `color === "#3fb950"` 判断是否仿真柱），改成变量后比较恒为假；SVG 的 `fill`/`stroke`/`stop-color`/`flood-color` 支持 `var()`，切换主题无需重绘 |
| **SSE 在 nginx 下要显式关缓冲** | `app/routes/session.py` 的两条 step2 流曾只给 `mimetype` 不带 `X-Accel-Buffering: no`（`app/routes/chat.py` 一直有）。缺这条时 nginx 会缓冲整个响应，逐行推送变成「长时间不动后一次性刷出」。现在两处共用 `session.py` 顶部的 `_SSE_HEADERS`（含 `Cache-Control` / `Connection` / `X-Accel-Buffering`），新增 SSE 端点请复用它 |
| **仿真完成状态不能只靠前端驱动** | `session.step` 曾只在 `run_simulation()` 里推进，而那次调用依赖「前端收到 WebSocket `complete` 后再补打一次 `run-simulation`」。页面刷新会掐断 WebSocket（MCP 侧任务仍在跑，任务记录在 MCP Server 内存里），于是会话永久停在 `simulating`、前端一直显示"仿真验证中"。现在 `GET /topology` 与 `GET /simulation` 会先调 `_fallback_reconcile_simulation()` 兜底判定，并且这两个端点必须回传 `original_task_id` / `equivalent_task_id`（前端刷新后靠它重建 WS 订阅）。**新增任何"读会话状态"的端点时，不要绕过这段兜底**，也不要让 `*_task_id` 只留在 DB 里 |
| **模型区几何量必须与组网布局模式解耦** | `canvasRebuild()` 里「是否渲染等效模型」看的是 `hasBothModels`（只有模型数据），而模型区 x/宽度曾是按**拓扑布局模式**分支赋值的 —— 只有三栏模式与「两侧组网都在」才给 `modelX0Eq` / `modelAreaWEq` 赋值。`restoreSession()` 先同步恢复两侧模型、再逐个 `await loadMeshData()`，中间态（两个模型 + 一个组网）于是拿到 `undefined`，`undefined + NaN` 一路算成 `NaN`，d3 报 `<text> attribute x: Expected length, "NaN"` 与 `translate(NaN,457) scale(N…`，并把整块等效模型图丢掉（界面表现只是"等效模型不见了"）。现在几何量由纯函数 `_modelLayoutGeom()`（`static/topo-renderer.js`，带 `@testable-model-layout` 标记）统一给出，不变量是 **`hasBothModels` 为真 ⇒ 四个几何量全是有限正数**；`_renderOneModel()` 另有非有限坐标守卫。改这段布局请跑 `python tests/test_topo_model_layout.py`。另注意 `restoreSession()` 必须清 `_formulaCardReady`（残留的 `true` 会提前走两段布局） |
| **画布 viewBox 高度必须按实际内容自愈，否则模型/公式卡被裁成「下方一大块空白」** | `canvasRebuild()` 的 `totalH = topoH + sectionGap + modelH` 只是**布局估算**：稠密模型 `modelH` 固定 520，而模型图真实高度 ≈ `44 + _TM_DESIGN.H(620)`；公式卡高度来自公式行数（如 55 行 ≈ 896），恢复会话时 `_replayFormulaLines()`、流式推送时 `_typewriterTick()` 都会在布局之后继续长高。只按估算值写 `viewBox` 会把内容从底部裁掉：表现是「画布下方一大块空白、模型/公式卡被一条横线截断，像被一个矩形盖住」（2026-11-09 用户实测：dense DeepSeek-R1-Distill-Qwen-7B + 2560 宽画布，viewBox 高 769、公式卡底 922）。修复是 `_syncCanvasHeightToContent()`（`static/topo-renderer.js`，带 `@testable-canvas-height` 标记）：在 `canvasRebuild` 末尾、逐行动画长高、公式行回放、折叠/展开切换后，按 `zoom-layer.getBBox()` 把 viewBox 补高到 `max(布局下限, 内容底 + 16)`（**只补不缩**，布局下限存 `data-base-h`）。**不要删这些调用，也不要改成无条件缩小**（内容少时画布会被压扁）；改这段跑 `python tests/test_topo_model_layout.py` |
| **前端外链资源在部署环境必被 CSP 拦掉** | 平台下发的 CSP 是 `style-src *.huawei.com 'unsafe-inline'`。`<head>` 里曾 `preconnect` + 引 `https://fonts.googleapis.com/css2?family=JetBrains+Mono…`，经域名 + nginx 访问时浏览器报 `Loading the stylesheet '…' violates the following Content Security Policy directive: "style-src *.huawei.com 'unsafe-inline'"` 并直接拦掉（该域名在内网本身也不可达），也就是说 JetBrains Mono **从来没有真正生效过**，只是白留一条控制台报错。现在外链已删除，等宽字体只保留 `local()` 源的 `@font-face`（零网络请求，本机装了就用），并且 `--font-mono` 把 Latin 等宽字体排在中文之前 —— 否则 webfont 缺席时数字/rank/公式会被"微软雅黑"渲染成比例字体。同源脚本/样式能加载，说明白名单覆盖本站域名；**新增字体/CSS/JS/图片前先确认平台 CSP，默认按"不许外链"处理，依赖一律 vendored 进 `static/`** |
| **「单卡极限等效」只是 UI 占位，未实现参数推导** | Step 1 表单的等效策略下拉里 `single_card_extreme`（单卡极限等效）没有实现，真实生效的只有 `min_equiv`（最小集群等效）。**不能让它静默回落到 min_equiv** —— 用户选了极限等效却拿到最小集群等效的结果（2026-11-09 用户实测报告）。入口校验有两处且必须保持一致：前端 `static/index.html` 的 `IMPLEMENTED_EQUIV_STRATEGIES`（`onEquivStrategyChange()` 即时红框 + 行内提示，`submitWorkflowParams()` 提交时拦截且不建会话/不推进节点）与后端 `app/routes/session.py` 的同名常量（`workflow_step1` 先校验、返回 400 `strategy_not_implemented`）。实现新策略时同步改这两处并跑 `python tests/test_equiv_strategy_guard.py` |
| **切换会话必须断开上一会话的仿真进度链路** | 仿真进行中时 `_simWs`（WebSocket）/`_simPollTimer`（HTTP 轮询）/`_simTaskBubbles` 都绑在旧会话上。历史上 `newSession()` / `restoreSession()` 只重置 `_simCompleted` 等标志、没关闭 socket，旧会话的迟到帧会走 `_simTaskEntry()` 在重建后的聊天区**重新创建旧任务的进度气泡**（2026-11-09 真实 bug：仿真中点「+ 新任务」，新任务交互区出现上一任务的进度条）。现在两个切换点都先调用 `_resetSimulationUiForSessionSwitch()`（关 socket + 停轮询 + 清气泡引用 + 重置 `_simCompleted/_simFailed/_simStarted` + 关闭上一会话的 Step2 公式回放流 `_closeAllStep2Streams()`）；WS `onmessage` 另有 `ownerSession` + `_isCurrent()` 陈旧帧守卫，HTTP 轮询带 `pollSessionId` 守卫。**新增会话切换入口（如按 URL 恢复、删除当前会话）时也必须调用它**，否则同样的串扰会再现 |
| **切换任务必须复位画布标签页** | `sessionStorage.activeCanvasTab` 是**全局**的（不按会话区分），`restoreSession()` 曾直接沿用它：从已完成任务的「等效结果判定」页切到未仿真任务时，result 分支因 `_comparisonReport` 为空而不切换，标签页 DOM/存储都不复位 → 任务已切换但界面停在空的结果页（2026-11-09 用户实测报告）。现在 `selectHistory()` 传 `{ resetTab: true }` 强制回「等效建模分析」；`restoreSession()` 还校验目标标签页对当前会话有效（result 需 `comparison_report`、simulation 需组网数据），无效即回落建模分析；标签页 DOM/存储统一由 `_setCanvasTabClasses()` 切换。**新增任务切换入口也要走 resetTab 或显式切回建模分析** |
| **同一会话的仿真提交必须持 session_lock 且幂等** | `SessionState` 是**可变共享对象**，HTTP 请求线程 / 仿真 WS 轮询线程 / SSE 线程都会改它。历史上 `run-simulation` 没有任何锁：两个并发请求（两个标签页各点一次「开始仿真」、或完成回调 + 刷新兜底同时到达）都读到 `task_id=None`，各自下发一遍 —— 实测 2 次调用产生 **4 个 MCP 任务**（2 个孤儿仿真进程），会话只记住最后 2 个。现在 `SessionManager` 提供 per-session `RLock`（`session_manager.session_lock()`），`run-simulation` 全程持锁、`save_session()` 同会话串行（`_persist_session` 是先删后插，必须避免交错）、`_fallback_reconcile_simulation` / `_mark_session_failed` 在锁内重读 `step`，`get_session()` 用双检锁避免并发 cache miss 各加载一份实例。重试语义：只有请求体 `force_retry=true` 才**取消旧任务 + 清空 `*_task_id`/仿真结果/报告后重新下发**（前端「重试仿真」与失败态再次点开始都已带该参数）；普通重复调用复用 task_id 补取结果，已完成会话直接返回缓存（`"cached": true`）。新增任何「读-改-写会话」的接口都必须走 `session_lock()`，改完跑 `python tests/test_session_concurrency.py` |
| **仿真并发上限/排队/超时/取消只能在 MCP Server 侧做** | Flask 侧只做会话级幂等（同会话不重复下发）；真正 `Popen` 拉起 `run.py` 的是 MCP Server，只有它知道有多少子进程在跑。现在 MCP 侧用 `max_concurrent_tasks`（默认 2 = 1 对仿真）做全局槽位，超出按 FIFO 排队（`status=queued` + `queue_position`），`task_timeout_sec` / `queue_timeout_sec` 兜住挂死与无限排队，`cancel_task` 幂等取消（排队出队、运行中 kill 进程组）；Flask 的 `DELETE /api/session/<id>` 与 force_retry 会调用它。前端依赖这些状态字符串（`queued`/`cancelled`）与 `queue_position` 字段。**性能红线**：`report_status` 不得读日志（曾每次整文件 `read_text` + O(n²) 去重，8k 行单次 162ms × 每秒每任务两次，会把 MCP 单事件循环打满，并行任务互相拖死），`sync_logs` 走 byte offset 增量；FastAPI 里的磁盘/DB 工具调用必须走 `run_in_threadpool`，否则阻塞单 worker 事件循环。改这些跑 `python tests/test_mcp_task_scheduler.py` 与 `python tests/test_mcp_server_e2e.py` |
| **workspace 磁盘清理的优先级不能改错** | `mcp_server/services/workspace_cleaner.py` 在后台删旧任务产物，避免结果文件把盘写满。优先级从高到低：**运行中/排队中任务永不删 → 磁盘水位应急（>=80% 保留期 2 天/必留 5；>=90% 只留最新 1 个）→ `min_keep_tasks`（默认 20；正常水位下与保留期冲突时**保留更多**，即它赢）→ `workspace_retention_days`（默认 7，超期先只删 `results/`）→ `log_retention_days`（默认 14，超期删整个任务目录）**。默认启动后等 10s 扫一次、之后每 24h 一次。常见改错：①把 `min_keep_tasks` 当成「最多保留」——它只保证最近 N 个不删，不是上限；②应急/临界模式还先保 `min_keep`，导致盘满时腾不出空间；③删除前没排除运行中/排队中、或没校验「workspace 直接子目录 + 含 `task_meta.json`」；④把两阶段合成一步，7~14 天的日志/脚本被提前删掉。清理只动磁盘、**不动数据库历史**；被清理的历史任务无法再看「单卡详情/训练脚本」（实时回读结果文件），主对比页与汇总指标仍可看。改这些跑 `python tests/test_workspace_cleanup.py` |
| **队列满必须用独立错误码 -32004（不是 -32003）** | `-32003` 已被 `TaskNotReadyError`（任务未完成/结果未就绪，get_result/card_detail 等抛出）占用；**队列满用 `-32004`**（`mcp_server/errors.py::QueueFullError`，message 含「队列已满」，data 带 queue_size/max_queue_size）。`app/mcp/client.py` 的 `_call_tool` 会保留 `error.code`，只有 `execute_task` 把 -32004 映射成 `MCPQueueFullError` —— 其余工具仍按普通失败处理，**不要全局按 code 判队列满**。Flask 把它转成 HTTP `429 simulation_queue_full` 且**不把会话标 failed**（背压不是失败）；前端提示稍后重试、完成态同步会自动重试。`AICM_MCP_MAX_QUEUE_SIZE` 默认 20，背压条件 = 排队数达上限且无空闲槽位。改错误码/阈值需三处同步（mcp_server/errors.py、app/mcp/client.py、前端）并跑 `python tests/test_session_concurrency.py`、`python tests/test_mcp_task_scheduler.py` |
| **会话写路径：长操作非阻塞互斥 + 短写 session_lock** | `SessionManager` 有两把不同用途的锁：`session_lock()`（RLock，短写串行）与 `try_begin_exclusive()/end_exclusive()`（长操作名额，非阻塞）。**Agent 对话 / Step1 / Step2 触发 / Step3 / run-simulation 必须占用长操作名额**：冲突返回 `409 task_busy`（流式接口以 SSE error 事件返回），前端对 409/429 只提示「稍后重试」、不标失败。**不要改成阻塞等待** —— 一个长对话会把另一个标签页卡到 LLM 超时。**Step2 已原子化（2026-11）**：`POST /workflow/step2` 在一次请求内算完并落库（公式行 + 等效组网/模型 + `step=equiv_generated`，毫秒级）；`GET /workflow/step2/stream` 只是「回放已落库公式行」的纯展示流，**不写库、不占长操作名额**，断线/切任务/刷新都不丢数据。前端 Step2 全链路必须用「发起时的 sessionId」快照（`submitStep2` 的 `sid` / `_step2Streams[sid]` / `_submitStep2GenerateMesh(sid)`），**不要再把完成动作绑到全局 `sessionId`** —— 历史上 A 的 step3 会打到 B 上，导致 A 永远停在 Step2、B 的公式行被写成 A+B（2026-11-09 实测），新增 Step2 相关异步回调时同样先取 sid 快照并在 UI 更新前校验 `sessionId === sid`。参数/公式行/删除等短写必须 `with session_manager.session_lock(sid)`；Agent 工具里改 session 的赋值块（orchestrator `_apply_*`）也要加锁。新增「读-改-写会话」的路由时按这个模式接入，跑 `python tests/test_session_write_locks.py` + `python tests/test_session_concurrency.py` |
| **MCP 启停必须走 lifespan；任务状态要可恢复** | `mcp_server/main.py` 用 FastAPI lifespan 管理后台线程：*startup* = `recover_interrupted_tasks()`（恢复扫描，必须在 scheduler.start 之前）→ `scheduler.start()` → `cleanup_worker.start()`；*shutdown* = `cleanup_worker.stop()` → `scheduler.shutdown(cancel_running=True)`（取消并 kill 运行中进程组、写 cancelled 终态、停 worker）。**不要在 import 时启动/停止后台线程**。任务状态迁移（prepare/queued/running/terminal）必须 best-effort 写 `workspace/<task_id>/task_state.json`（`mcp_server/services/task_state.py`，原子写、失败不影响主流程）；`task_loader` 恢复规则：terminal 原样；running/queued + 有结果 → completed，否则 failed 且消息含「服务重启，任务中断（进程已丢失）」；dry_run submitted 保持；无 state 文件回落 legacy。新增状态迁移点要同步写 state + 更新 loader 规则，跑 `python tests/test_restart_recovery.py` 与 `python tests/test_mcp_server_e2e.py` |
| **前端“只在某个初始化分支里加载数据”会造成刷新才恢复的间歇 bug** | Step1 的模型名称下拉（`#wf-orig-model-name`）历史上只在 `init()` 的「无会话」分支调用 `loadModelCatalogDropdown()`，而 `init()` 在「有可恢复会话」时会提前 `return`（跳过该行）→ 带会话刷新页面后下拉为空；点「+ 新任务」也不补加载，再刷新一次（`newSession()` 已清 localStorage）才恢复，表现为「有时候为空、刷新才好」。现在 `init()` 开头**无条件**发起加载，`newSession()` / `openWorkflowParams()` 兜底补加载；loader 幂等（`_modelCatalogLoaded/_modelCatalogLoading`）且失败重试 3 次（1.5s/3s 退避）。**新增下拉/目录类初始化不要只写在某个分支里**；改完必须用真实浏览器验证「写入 `localStorage.mesh_session` 后 reload、断言 option 数 > 1」这一场景（回归方式见提交说明/`.tmp/model_catalog_probe.mjs`） |
| **仿真产物文件只能经 MCP 校验后下载，Flask 不得直接读 workspace** | 下载入口在「等效仿真验证」tab 右上角（`static/index.html` 的 `#sim-download-btn`）。数据链路：前端 → Flask `GET /api/session/<id>/simulation-files*` → MCP Server 只读 REST `GET /tasks/<task_id>/files`、`/files/download?path=`、`/files/bundle?category=`（`mcp_server/services/task_files.py`）。**安全红线**：① `side` 只能 `original`/`equivalent`，task_id 只能来自会话的 `*_task_id`（否则 404），不能接受前端传 task_id；② `path` 由 MCP `_normalize_rel_path` + `resolve().relative_to(task_dir)` 校验，**拒绝 `..`、绝对路径、反斜杠穿越、符号链接逃逸**；③ 打包有 `files_max_list` / `files_bundle_max_bytes`（MCP）与 `SIM_FILES_MAX_BUNDLE_BYTES` / `SIM_FILES_MAX_FILES`（Flask 双侧合并）上限，超出 413；④ 任务目录被磁盘清理后返回 404（前端显示「已被清理/读取失败」），不要 fallback 到别的目录。改完跑 `python tests/test_simulation_files_api.py`（真实 MCP HTTP + Flask test client 端到端） |
| **Agent 自动填参后提交 Step1 会被会话锁拦下；对话结束的节点同步还会清空表单** | 右侧 Agent 对话 `POST /api/chat/stream` 在整个 LLM 回合（含多轮 tool call，实测 3~10s）持有该会话的长操作名额。Agent 调 `auto_fill_model_params` 自动填好 Step1 表单并打开参数层后，用户若立刻点「确认提交」，`POST /workflow/step1` 会被同一会话的互斥锁拦下返回 `409 task_busy`，旧前端把它当硬失败显示「Step 1 失败: 该任务正在执行其他操作…」（2026-11-10 用户实测）。更隐蔽的是：`handleDone` 里用 `workflow_state` 调 `syncWorkflowFromSession()` → `resetWorkflowState()` 会顺手 `closeWorkflowParams()+resetWorkflowFormFields()`，所以**即使用户等对话结束再提交，Agent 刚填好的参数也已经被清空/关掉了**。现在：① `submitWorkflowParams` 在 `isStreaming` 时把参数排队（`_pendingWorkflowSubmit`），`sendMessage` 的 finally 里 `_flushPendingWorkflowSubmit()` 在对话结束后自动补发（期间切任务则丢弃并提示）；② `submitStep1` 对 409 只做「稍后重试」软提示、不标红；③ Agent 回合结束的节点同步走 `syncWorkflowFromSession(data, {keepForm:true})`，保留表单/参数层（newSession/恢复会话仍正常重置）。**新增对话相关的前端提交逻辑时先查 `isStreaming`，不要直接 POST 会话写接口**。另外 chat 的 `done` 事件不带 `stage`，`handleDone()` 在没有待渲染数据时必须走快速路径直接返回：旧逻辑会白等最多 15 秒「Step2 子步骤完成」、把节点错标成已生成/仿真中，并因为 `renderThenSleep` 依赖 requestAnimationFrame 而在后台标签页长时间不返回，从而推迟排队的 Step1 提交（2026-11-10 实测 15.7s，页面被切到后台时更久；表现为「Agent 对话已完成，继续提交 Step1 参数…」之后画布迟迟不出图） |

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
python tests/test_simulation_reconcile.py     # PASS  刷新后仿真会话状态自愈（见 §5 陷阱表）
python tests/test_topo_model_layout.py        # PASS  模型区几何量不变量（无 NaN 坐标）+ 画布高度自愈（需 node，缺 node 则 SKIP）
python tests/test_equiv_strategy_guard.py     # PASS  未实现等效策略的入口拦截（单卡极限等效 → 400）
python tests/test_session_concurrency.py      # PASS  并发提交不重复下发 / force_retry / 删除取消 / get_session 单实例
python tests/test_mcp_task_scheduler.py       # PASS  MCP 并发上限 / FIFO 排队 / 排队与执行超时 / 取消 / 增量日志
python tests/test_workspace_cleanup.py        # PASS  workspace 清理优先级 / 两阶段 / 水位应急 / 临界 / dry_run / worker
python tests/test_session_write_locks.py      # PASS  会话长操作互斥（409）/ 短写并发 / Step2 原子化+回放隔离 / 删除保护
python tests/test_restart_recovery.py         # PASS  task_state 迁移 / loader 恢复规则 / 启动扫描幂等 / shutdown 取消 / lifespan 优雅停机
python tests/test_simulation_files_api.py     # PASS  仿真产物下载：MCP REST + Flask 代理 + ZIP 打包 + 路径穿越防护

# 当前基线为失败（exit 1）
python tests/test_estimate_equivalence.py

# 需先启动服务（真实 HTTP）
python tests/test_mcp_server_e2e.py           # 自带 dry-run 服务（8799 端口），需已装 fastapi
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
   `app/dao/__init__.py`（24 处调用点中占 23 处）；rank 换算只应在 `app/rank_layout.py`。
3. **改完跑校验**：至少 `scripts/verify_static.py` + `scripts/verify_consistency.py`；
   涉及 schema 则另跑 `scripts/sqlite_real_check.py`（默认后端）与 `scripts/pg_real_check.py`
   （逃生门）；涉及 rank 则跑第 6 节相关测试。
   **改了任一套 DDL（`SCHEMA_SQL` 或 `SCHEMA_SQLITE_SQL`）必须同步改另一套** ——
   `verify_static.py` 的 D1 断言会比对两套的「表→列」映射，漏改会直接让它失败。
4. **改 Dockerfile / 部署契约时同步更新**：`charts/values.yaml`、`.env.example`、
   `README.md`、`docs/使用指南.md`、`docs/PostgreSQL迁移SQLite改造计划.md` —— 这几处与代码存在
   跨文件契约，历史上多次因"只改一处"产生不一致。
5. **提交前检查是否有失效引用**：删除文件后，全仓库搜索该文件名（曾出现删了
   `CLAUDE.md` 但 `docs/使用指南.md` 的目录树仍列着它）。

---

## 8. 大文件与导航提示

| 文件 | 行数 | 说明 |
|------|------|------|
| `app/routes/session.py` | 2125 | **最大文件**：会话 + 拓扑 + 仿真 + 工作流 REST 端点全在这里（含 Step2 原子化：`_build_step2_formula_lines` / `_generate_equivalent` / 回放流） |
| `app/agent/orchestrator.py` | 935 | Agent 主编排循环（LLM + 技能 + 工具） |
| `app/models/model_catalog.py` | 681 | 模型参数解析（HF / MindSpeed / 内置） |
| `app/dao/__init__.py` | ~500 | DAO 全集（双后端 SQL；方言差异由 `app/dbapi.py` 在语句下发前转换） |
| `mcp_server/services/results_reader.py` | 507 | 仿真结果解析 |
| `app/skills/training-mesh-profiler-skill/` | 550 | 组网性能分析 skill（`__init__.py` + `moe_estimator.py`） |

改动这几个文件时尤其注意：它们体积大且被多处依赖，**小改动也可能产生大影响面**，
建议改前先搜引用方（如 §7 第 2 条）。

技能目录每个都带 `SKILL.md`，改了 skill 行为请同步更新对应的 `SKILL.md`。

---

## 9. 目录级说明

| 路径 | 说明 |
|------|------|
| `app/` | Flask 应用（见 §2）。`dbapi.py`=SQL 方言适配层，`db.py`=双后端连接池，`db_migration.py`=两套 DDL + 硬校验 |
| `mcp_server/` | 仿真 MCP Server，独立部署单元 |
| `docker/` | `base/Dockerfile`（基础镜像）、`entrypoint.sh`（启动链）、`check_*.py`（启动自检）、`checkpoint_db.py`（停机 checkpoint） |
| `charts/` | Helm chart，含 `database` 卷（SQLite 数据文件目录）与 `SQLITE_PATH` 配置 |
| `scripts/` | 改造验证脚本（见 §3），可复跑 |
| `docs/` | 需求规格、使用指南、设计文档、改造说明 |
| `tests/` | 独立测试脚本（非 pytest，见 §6） |
| `static/` | 前端单页应用（`index.html` + 拓扑渲染 + 仿真数据流） |
