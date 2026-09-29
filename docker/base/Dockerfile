# syntax=docker/dockerfile:1
# ============================================================================
# TrainMeshAgent 基础镜像 —— repository/python3.10-aicm-base:1.0
#
# 构建环境不支持外网，仅内网华为源可用：
#   apt -> http://mirrors.tools.huawei.com
#   pip -> http://mirrors.tools.huawei.com/pypi/simple/
# Ubuntu 22.04 系统自带 python3 = 3.10，与仿真工具验证环境 (Python 3.10.2)
# 天然吻合，因此无需从源码编译 Python。
#
# 本镜像承载「不常变」的部分，业务镜像构建期零网络：
#   - Python 3.10 运行时
#   - app 依赖            requirements.txt
#   - MCP Server 依赖     mcp_server/requirements.txt
#   - 仿真工具依赖        docker/requirements-sim.txt   (numpy / pandas)
#   - 仿真工具离线 wheel  docker/wheels/hccl_format-*.whl
#   - libpq5（仅作为 psycopg2 的运行时依赖，见下）
#
# 业务镜像见仓库根 Dockerfile。
# 构建命令见 docker/build_base_image.sh。
#
# ── 为什么数据库改成了 SQLite（本次改造），以及为什么不再装 PostgreSQL 服务端 ──
# 1) 服务原先依赖「外部 PostgreSQL + 运行期注入 DATABASE_URL」，后来改成「内嵌
#    PostgreSQL 14 服务端」，现在改为**默认 SQLite 单文件**：一个容器自洽启动，
#    且不再需要 initdb / 版本守卫 / Unix socket / 就绪探测这一整套启动链。
# 2) SQLite 由 Python 标准库 `sqlite3` 提供，**不需要任何 apt 包** —— 这是体积
#    收益的来源（原先 postgresql-14 + postgresql-common + ucf 约 100~150MB）。
# 3) **保留 libpq5**：`app/dao` 的 PG 逃生门（注入 DATABASE_URL 连外部 PG）依赖
#    可 import 的 psycopg2，而 psycopg2-binary 在运行期需要 libpq5。删掉它会让
#    逃生门在运行期 import 失败。
# 4) **删除 postgresql-14 带来的直接好处**：下面那套 ucf/ucfr 拦截层是为了绕开
#    `postgresql-common` 的 postinst 在 Docker<20.10 节点上必然失败而加的
#    （见 git 历史：ucf/ucfr 的 test -w 恒假 → postinst exit 1 → apt 返回 100）。
#    不再安装 postgresql-14 ⇒ 不再需要 ucf/ucfr divert ⇒ 构建链显著变短。
# ============================================================================
FROM kweecr05.his.huawei.com:80/public/ubuntu:x86-22.04

WORKDIR /home/

COPY requirements.txt            /tmp/build/requirements-app.txt
COPY mcp_server/requirements.txt /tmp/build/requirements-mcp.txt
COPY docker/requirements-sim.txt /tmp/build/requirements-sim.txt
COPY docker/wheels/              /tmp/wheels/

# 禁用 SSL 证书验证 - 设置全局不验证 SSL
ENV PYTHONHTTPSVERIFY=0
# 设置非交互式环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV DEBCONF_NONINTERACTIVE_SEEN=true
ENV LANG=C.UTF-8
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# SKIP_HCCL=1：尚未拿到 hccl_format-*.whl 时可先跳过，构建出基础镜像
ARG SKIP_HCCL=0

# 一次性完成所有系统配置、包安装和清理
RUN set -ex && \
    # 配置apt和dpkg不检查GPG签名和SSL证书
    echo 'APT::Get::AllowUnauthenticated "true";' > /etc/apt/apt.conf.d/99disable-security && \
    echo 'APT::Get::Assume-Yes "true";' >> /etc/apt/apt.conf.d/99disable-security && \
    echo 'Acquire::AllowInsecureRepositories "true";' >> /etc/apt/apt.conf.d/99disable-security && \
    echo 'Acquire::AllowDowngradeToInsecureRepositories "true";' >> /etc/apt/apt.conf.d/99disable-security && \
    echo 'Acquire::https::Verify-Peer "false";' >> /etc/apt/apt.conf.d/99disable-security && \
    echo 'Acquire::https::Verify-Host "false";' >> /etc/apt/apt.conf.d/99disable-security && \
    echo 'DPkg::Options "--force-confdef";' >> /etc/apt/apt.conf.d/99disable-security && \
    echo 'DPkg::Options "--force-confold";' >> /etc/apt/apt.conf.d/99disable-security && \
    echo 'force-confdef' >> /etc/dpkg/dpkg.cfg && \
    echo 'force-confold' >> /etc/dpkg/dpkg.cfg && \
    \
    # 配置华为的apt源
    sed -i "s@http://.*archive.ubuntu.com@http://mirrors.tools.huawei.com@g" /etc/apt/sources.list && \
    sed -i "s@http://.*security.ubuntu.com@http://mirrors.tools.huawei.com@g" /etc/apt/sources.list && \
    \
    # 更新包列表
    apt-get update --allow-unauthenticated --allow-insecure-repositories && \
    \
    # 修复dpkg状态并移除problematic packages
    dpkg --configure -a || true && \
    apt-get remove -y --allow-unauthenticated unattended-upgrades || true && \
    \
    # 安装所有需要的包
    # build-essential / python3-dev 只为 pip 可能的源码编译而装，装完即卸
    # libpq5：psycopg2 的运行时依赖，PG 逃生门需要（不再装 postgresql-14）
    apt-get install -y --allow-unauthenticated --no-install-recommends \
        python3 \
        python3-pip \
        python3-dev \
        python3-venv \
        python3-distutils \
        libpq5 \
        curl \
        wget \
        tree \
        vim \
        lsof \
        htop \
        unzip \
        tar \
        gzip \
        build-essential && \
    \
    # 创建Python软链接
    ln -sf /usr/bin/python3 /usr/local/bin/python && \
    ln -sf /usr/bin/pip3 /usr/local/bin/pip && \
    \
    # 配置pip源
    mkdir -p ~/.pip && \
    echo "[global]" > ~/.pip/pip.conf && \
    echo "index-url = http://mirrors.tools.huawei.com/pypi/simple/" >> ~/.pip/pip.conf && \
    echo "trusted-host = mirrors.tools.huawei.com" >> ~/.pip/pip.conf && \
    echo "disable-pip-version-check = true" >> ~/.pip/pip.conf && \
    \
    # 安装 Python 依赖：app + MCP Server + 仿真工具
    pip3 install --trusted-host mirrors.tools.huawei.com --no-cache-dir \
        -r /tmp/build/requirements-app.txt \
        -r /tmp/build/requirements-mcp.txt \
        -r /tmp/build/requirements-sim.txt && \
    \
    # 安装仿真工具随目录下发的离线 wheel（hccl_format-0.1-py3-none-any.whl）
    if ls /tmp/wheels/*.whl > /dev/null 2>&1; then \
        pip3 install --trusted-host mirrors.tools.huawei.com --no-cache-dir /tmp/wheels/*.whl; \
    elif [ "${SKIP_HCCL}" = "1" ]; then \
        echo "WARN: SKIP_HCCL=1，未安装 hccl-format"; \
    else \
        echo "ERROR: docker/wheels/ 下未找到 hccl_format-*.whl" >&2; \
        echo "       请把 hccl_format-0.1-py3-none-any.whl 放入 docker/wheels/ 后重建，" >&2; \
        echo "       或加 --build-arg SKIP_HCCL=1 先跳过。" >&2; \
        exit 1; \
    fi && \
    \
    # 依赖自检：任一 import 失败即构建失败，避免坏镜像流到业务层
    # 覆盖 app / mcp_server 主链路 + 仿真工具依赖闭包
    # (numpy, pandas, 及 pandas 传递出的 pytz / tzdata / python-dateutil -> six)
    python -c "import flask, flask_cors, flask_sock, openai, pydantic, psycopg2, requests, yaml, fastapi, uvicorn, pydantic_settings, numpy, pandas, pytz, tzdata, dateutil, six; print('[base] python deps OK')" && \
    \
    # SQLite 自检：标准库自带，无需 apt 包。断言 sqlite3 可用、且本机 sqlite 版本
    # 支持 UPSERT 的 `excluded` 伪表（3.24+）。
    # ⚠️ 一条 execute() 只能跑一条语句 —— 多条语句合在一个字符串里会抛
    # ProgrammingError: You can only execute one statement at a time（曾因此让构建失败）。
    # 另：脚本整体已在 shell 的 "..." 内，故用 chr(120) 代替字面量 'x'，少一层转义。
    # 输出刻意只用 ASCII：容器内 stdout 编码不保证是 UTF-8，中文 print 会出现乱码
    # （本机验证时已复现）。自检的读者是构建日志，用英文更稳。
    python -c "import sqlite3; v=sqlite3.sqlite_version_info; assert v >= (3, 24, 0), v; \
c=sqlite3.connect(':memory:'); c.execute('CREATE TABLE t(a)'); \
c.execute('INSERT INTO t VALUES (1) ON CONFLICT DO NOTHING'); \
c.execute('CREATE TABLE u(a TEXT PRIMARY KEY, b INT)'); \
c.execute('INSERT INTO u VALUES (?,1) ON CONFLICT (a) DO UPDATE SET b=excluded.b', (chr(120),)); \
c.execute('INSERT INTO u VALUES (?,2) ON CONFLICT (a) DO UPDATE SET b=excluded.b', (chr(120),)); \
row=c.execute('SELECT b FROM u').fetchone(); assert row[0]==2, row; \
c.close(); print('[base] sqlite3 OK', sqlite3.sqlite_version, '- upsert/excluded works')" && \
    # 这里同时断言 **镜像里不存在 PostgreSQL 服务端**：旧镜像残留会让「数据到底
    # 存在哪」变得难以判断（内嵌 PG 与 SQLite 两种数据目录混放）。
    test ! -e /usr/lib/postgresql/14/bin/postgres && \
    test -f /usr/lib/x86_64-linux-gnu/libpq.so.5 && \
    echo '[base] no embedded postgresql server; libpq5 kept for pg escape hatch' && \
    test ! -e /home/aicm/db && \
    echo '[base] no database data dir in image layer' && \
    \
    # 移除构建依赖以减少镜像大小
    apt-get remove -y --allow-unauthenticated \
        python3-dev \
        build-essential && \
    \
    # 彻底清理系统缓存和临时文件
    apt-get autoremove -y --allow-unauthenticated && \
    apt-get autoclean && \
    apt-get clean && \
    \
    # autoremove 之后的收尾断言：libpq5 与 psycopg2 必须仍然可用
    # （逃生门依赖它们；它们与本次移除的构建工具无依赖交集，但要实测）。
    test -f /usr/lib/x86_64-linux-gnu/libpq.so.5 && \
    python -c "import psycopg2; print('[base] psycopg2 still importable (pg escape hatch)')" && \
    \
    rm -rf /var/lib/apt/lists/* \
           /tmp/* \
           /var/tmp/* \
           /root/.cache \
           ~/.pip/cache \
           /var/cache/apt/* \
           /var/lib/dpkg/info/* \
           /etc/apt/sources.list.bak
