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
#   - PostgreSQL 14 服务端（内嵌数据库，见 docker/entrypoint.sh）
#
# 业务镜像见仓库根 Dockerfile。
# 构建命令见 docker/build_base_image.sh。
#
# ── 为什么数据库要装进镜像，且为什么是 PG 14 ──
# 1) 服务原先依赖「外部 PostgreSQL + 运行期注入 DATABASE_URL」，部署侧多一个必须
#    存在、必须可连的组件；镜像内嵌后，一个容器即可自洽启动。
# 2) 数据目录 PGDATA=/home/aicm/db/data，由部署侧挂载宿主机目录持久化
#    （节点侧 /data/aicm/db → 容器内 /home/aicm/db），与仿真工作区同一范式。
# 3) Ubuntu 22.04 官方源中的 PostgreSQL 只有 14 这一个主版本，因此这里钉
#    postgresql-14 而非浮动的 postgresql：PG 的 PGDATA 与主版本强绑定，
#    一旦容器升级主版本，已有数据目录会直接拒绝启动。升级必须走
#    dump → 换镜像 → initdb → restore 的人工流程（见 docs/使用指南.md）。
# 4) 不装 postgresql-contrib、不 CREATE EXTENSION uuid-ossp：建表用的
#    gen_random_uuid() 自 PG 13 起已内建（pgcrypto 之外的核心函数）。
# 5) libpq5 是 psycopg2-binary 的运行时依赖，已在下面安装；镜像内 PG 与
#    psycopg2 同源于 Ubuntu 22.04，因此不会出现客户端/服务端版本错配
#    （例如 libpq 17 对未启用加密的服务端报 "GSSAPI encryption required"）。
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
    # ── ucf 状态兜底（Docker < 20.10 的节点必需）────────────────────────────
    # 这类节点上容器内 test -w 对任何文件都返回假：默认 seccomp 白名单缺
    # faccessat2，而 jammy 的 glibc 用 faccessat2 实现 access()，且只在 ENOSYS
    # 时回退老 syscall。于是 postgresql-common 的 postinst
    # （setup_createclusterconf）里依次调用的
    #   ucf : [ -e $statedir/hashfile -a ! -w $statedir/hashfile ]  → 恒成立 exit 1
    #   ucfr: [ -e $statedir/registry -a ! -w $statedir/registry ]  → 恒成立 exit 1
    # 都会失败，导致 postgresql-common 未配置、postgresql-14 连带未配置、
    # apt 返回 100（与文件权限、基镜像内容无关，见 Ubuntu #1943049）。
    # 两个检查都是「文件存在且不可写」才报错，因此：先单独装 ucf 让它的 postinst
    # 把状态文件建出来，再删掉 —— 检查即跳过；ucf/ucfr 随后会自行重建
    # （ucf 的 replace_md5sum 与 ucfr 的 touch 分支都处理了文件缺失）。
    # 注意 ucf 是 postgresql-common 的 Depends，先装它不会影响后面的解析。
    apt-get install -y --allow-unauthenticated --no-install-recommends ucf && \
    rm -f /var/lib/ucf/hashfile /var/lib/ucf/registry && \
    \
    # 安装所有需要的包
    # build-essential / python3-dev 只为 pip 可能的源码编译而装，装完即卸
    apt-get install -y --allow-unauthenticated --no-install-recommends \
        software-properties-common \
        python3 \
        python3-pip \
        python3-dev \
        python3-venv \
        python3-distutils \
        libpq5 \
        postgresql-14 \
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
    # 内嵌数据库自检：二进制必须存在且版本恰好是 14。
    # 这里不 initdb —— 数据目录的属主与初始化的时机都属于运行期（容器以 uid 1000
    # 运行，见业务镜像 Dockerfile 与 charts/templates/deployment.yaml）。
    # 另外断言基础镜像里**不存在** PGDATA：一旦镜像层里带了初始化过的数据目录，
    # 它会被运行期挂载遮蔽而永远用不上，还会误导排查方向。
    # 用 test -f 而非 test -x：-x/-w 走 access()，在上面那类 Docker<20.10 的
    # 节点上恒为假，会在完全健康的镜像上误报失败；存在性检查走 stat，不受影响。
    test -f /usr/lib/postgresql/14/bin/initdb && \
    test -f /usr/lib/postgresql/14/bin/postgres && \
    # dpkg 层面必须真的 configured：postinst 失败时文件照样解包到位，
    # 只测二进制存在会放过「包未配置」的半成品（正是本次失败的形态）
    dpkg -s postgresql-common | grep -q '^Status: install ok installed' && \
    dpkg -s postgresql-14 | grep -q '^Status: install ok installed' && \
    pg_isready --version && \
    test "$(pg_isready --version | sed -n 's/.*PostgreSQL \([0-9]*\).*/\1/p')" = "14" && \
    test ! -e /home/aicm/db && \
    echo '[base] postgresql 14 OK' && \
    \
    # 移除构建依赖以减少镜像大小
    apt-get remove -y --allow-unauthenticated \
        python3-dev \
        build-essential \
        software-properties-common && \
    \
    # 彻底清理系统缓存和临时文件
    apt-get autoremove -y --allow-unauthenticated && \
    apt-get autoclean && \
    apt-get clean && \
    \
    # autoremove 之后的收尾断言：确认 postgresql-14 没有被当作「不再需要的依赖」
    # 连带卸掉（它的依赖链与本次移除的构建工具无交集，理论安全，但这里要实测）。
    test -f /usr/lib/postgresql/14/bin/initdb && \
    test -f /usr/lib/postgresql/14/bin/postgres && \
    dpkg -s postgresql-14 | grep -q '^Status: install ok installed' && \
    echo '[base] postgresql 14 survived autoremove' && \
    \
    rm -rf /var/lib/apt/lists/* \
           /tmp/* \
           /var/tmp/* \
           /root/.cache \
           ~/.pip/cache \
           /var/cache/apt/* \
           /var/lib/dpkg/info/* \
           /etc/apt/sources.list.bak
